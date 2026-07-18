# Plan — Candidate-batched GPU generative simulator for the Chi² loss

## Context

The joint MLE+Chi² loss keeps the MLE term on the GPU-vectorized population path
but loops the Chi² **generative simulation per candidate on CPU** each DE
generation ([`_jointVectorizedObjectiveWrapper`](rlmodel/model/fit.py)). That loop
is the joint mode's bottleneck and forces a CPU/GPU straddle. Goal: simulate **all
S DE candidates in one batched GPU pass** so the whole joint objective runs
GPU-vectorized, **preserving the exact generative Chi² statistic** (chosen over the
cheaper CPU-pool and the FP-reformulation alternatives).

Design mirrors the MLE population path: a sequential loop over trial position with
a flattened **(candidate · session)** batch axis. The CPU path
(`processMultipleSess`/`makeOneRun`/`chi2Loss`) stays **untouched** — it remains
the pure-`chisq` fit driver and the numerical reference for parity tests.

## What the Chi² path does today (grounding)

- [`makeOneRun`](rlmodel/model/logic.py#L467) → [`simulateDDMMultipleSess`](rlmodel/model/logic.py#L361)
  → [`processMultipleSess`](rlmodel/model/logic.py#L35) (sequential trial loop,
  **vectorized over sessions**, pandas-`.iloc`-centric) → [`simulateDDMTrial`](rlmodel/model/logic.py#L220)
  (Monte-Carlo: materialize `(num_sess, num_steps)` trajectory, `cumsum`, then
  `argmax` of the bound-crossing) → [`calcLoss`](rlmodel/model/logic.py#L443)/[`chi2Loss`](rlmodel/model/logic.py#L410)
  (per (correct×left) cell, histogram sim vs real RT into **data-derived fixed
  bins** + a no-choice null term).
- **Already reusable**: [`state_updates.py`](rlmodel/model/state_updates.py)
  (`compute_q_value`/`update_q_values`/`update_reward_rate`) is backend-agnostic
  (`xp=`), already used candidate-batched by the MLE path — the RL-update half is
  free. The MLE path's [`_compute_latent_population_equal_sessions`](rlmodel/model/mle.py)
  + `estimate_population_settings` + `resolve_array_backend` are the templates for
  the padded candidate×session loop, memory budgeting, and the xp/normal_cdf
  abstraction. `prepare_mle_data`/`PreparedMLEData` (with `session_slices`) is the
  prep template.
- **drift/bias/noise** ([drift.py](rlmodel/model/drift.py)/[bias.py](rlmodel/model/bias.py)/[noise.py](rlmodel/model/noise.py)):
  `np.`-based `(rows, steps)` math with **scalar** params; `noise.py` uses the
  numpy RNG; `drift.py` uses `scipy.ndimage.shift` on the rare `nondectime_Q=False`
  branch.

## Approach (phased)

### Phase 1 — prepared batch data (once per subject)
New `prepare_chisq_data(subject_df)` (next to `prepare_mle_data`): extract to
backend-agnostic arrays the equal-session-padded layout (reuse the MLE padding
helper), per-trial `DV`, and the **real** `(ChoiceCorrect, ChoiceLeft,
calcStimulusTime, valid)` plus **precomputed fixed Chi² bins** per (correct×left)
cell (and the `is_loss_no_dir` 2-cell variant) and the real per-bin/null counts.
Bins/real-counts are candidate-independent → computed once on host.

### Phase 2 — batched generative simulator (new `chisq_batch.py`, mirrors `mle_batch.py`)
`simulate_population(prepared, x_matrix, params_names, model_config, backend)`:
- Flatten to `rows = S · num_sess`; build per-candidate param vectors by repeating
  each candidate's scalars across its `num_sess` rows (`DRIFT_COEF`, `NOISE_SIGMA`,
  `BOUND`, bias params, `ALPHA`/`BETA`, `Q_VAL_*`).
- Sequential loop over trial position (≤ `max_trial`). Per step, on all `rows` at
  once via `backend.xp`: (a) starting point `z` from the bias fn (per-candidate
  params + current Q/R); (b) draw noise `(rows, num_steps)` with `cupy.random`;
  (c) `dx` via the drift fn (`cumsum`); (d) first-crossing `argmax` → `SimRT`,
  `SimChoiceLeft`; (e) `SimChoiceCorrect` from `SimChoiceLeft` vs `DV` sign;
  (f) `update_q_values`/`update_reward_rate` (xp, **simulated** choices) for the
  next step. Collect `SimRT`/`SimChoice*` as `(S, n_trials)`.
- **Memory**: the per-step `(rows, num_steps)` trajectory grows ×S; reuse the MLE
  `estimate_population_settings` budget and **tile candidates** when it exceeds the
  ceiling (process candidate chunks, concatenate results).

### Phase 3 — drift/bias/noise: xp + per-candidate params (single source of truth)
Generalize each fn to take `xp` and treat params as **array-or-scalar** with
explicit broadcasting (per-row `(rows,)` for terms multiplying `dvs`; `(rows,1)`
for terms multiplying `(rows, steps)`). Scalar inputs must reproduce the current
CPU output **bit-for-bit** (regression-pinned). `cupyx.scipy.ndimage.shift` for the
`nondectime_Q=False` branch. Noise RNG is `xp.random` (per-candidate streams) —
**accept CPU≠GPU numerics**.

### Phase 4 — batched Chi² (in `chisq_batch.py`)
`batched_chi2(prepared, sim_choice_correct, sim_choice_left, sim_rt, backend)`:
per (correct×left) cell, mask sim trials, `searchsorted` their `SimRT` into the
cell's **fixed** bins, scatter-add over the candidate axis → `(S, n_bins)`
`freq_sim`; `Σ (freq_sim − freq_real)² / freq_real` + null term → `(S,)`. Exactly
`chi2Loss`'s formula, vectorized over candidates.

### Phase 5 — wire into the joint objective (`fit.py`)
Add `mle_use_batched_chi2: bool = True` to `MLEModelConfig` (mirrors
`mle_use_batched_likelihood`). In `_jointVectorizedObjectiveWrapper`, when the
backend is cupy **and** the flag is set, replace the per-candidate `makeOneRun`
loop with one `simulate_population` + `batched_chi2` call → `(S,)` Chi²; otherwise
keep today's per-candidate CPU loop (reference/fallback, numpy backend). Build the
`prepare_chisq_data` fixture once in `_processSubject` (alongside
`prepared_subject`). MLE term is already GPU-batched ⇒ no CPU straddle remains.

## Files

| File | Change |
|---|---|
| new [model/chisq_batch.py](rlmodel/model/) | batched generative `simulate_population` + `batched_chi2` (mirrors `mle_batch.py`) |
| [model/logic.py](rlmodel/model/logic.py) | factor `prepare_chisq_data` + fixed-bin/real-count precompute out of `chi2Loss`/`processMultipleSess` for reuse; **CPU path otherwise untouched** |
| [model/drift.py](rlmodel/model/drift.py), [bias.py](rlmodel/model/bias.py), [noise.py](rlmodel/model/noise.py) | add `xp=`; per-candidate array-param broadcasting; `cupyx` shift; bit-exact scalar regression |
| [model/fit.py](rlmodel/model/fit.py) | batched-GPU branch in `_jointVectorizedObjectiveWrapper`; build chisq fixture in `_processSubject` |
| [model/mle.py](rlmodel/model/mle.py) | `mle_use_batched_chi2` on `MLEModelConfig` (+ validate) |
| new model/tests/test_chisq_batch.py | parity + statistical + histogram tests below |

## Reuse
- `state_updates.*` (xp, candidate-batched) — RL updates, as-is.
- `_compute_latent_population_equal_sessions`, `estimate_population_settings`,
  `resolve_array_backend`, `PreparedMLEData`/`prepare_mle_data` — templates &
  shared padding/budget/backends from the MLE path.
- Existing drift/bias/noise math (generalized, not rewritten); `chi2Loss` formula
  (vectorized, not changed).

## Verification
- **Deterministic (no RNG divergence)**: inject an identical `noise` array into the
  CPU drift fn and the batched fn → `dx`/crossings/`SimRT` match (float32 tol).
- **Param-broadcast parity**: batched fn with all candidates sharing one param set
  == the scalar CPU fn, per candidate.
- **RL propagation**: generative loop with injected fixed noise → Q/R trajectory
  matches the CPU `processMultipleSess`.
- **Histogram parity**: `batched_chi2` on given sim RTs == `chi2Loss` per candidate
  (exact).
- **Statistical**: over many seeds, GPU vs CPU sim RT distribution per cell agree
  in mean/quantiles within tolerance (RNG differs by design).
- **End-to-end**: `--mle-chi2-weight 1 --mle-backend GPU --dry-run` finite
  components; a short DE matches the CPU per-candidate loop in ballpark; measure
  speedup vs the loop.
- **Regression**: full `pytest rlmodel/model/tests/` (CPU chisq + joint tests
  unchanged; ignore the pre-existing `test_diffusion_equivalence.py` /
  `test_first_passage_mass_conservation.py` C-kernel segfaults and the 2
  pre-existing `_loss_title` failures).

## Risks / notes
- **Memory** is the main risk: per-step `(S·num_sess, num_steps)` trajectory — tile
  candidates via the MLE budget helper.
- **CPU≠GPU numerics** (RNG) ⇒ tests are statistical, not bit-exact; the CPU loop
  stays as the reference and the numpy-backend fallback.
- Keep the change behind `mle_use_batched_chi2` so the per-candidate loop remains a
  one-flag escape hatch.
