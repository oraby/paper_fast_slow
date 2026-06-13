# Scale-bound / Bound-RewardRate — plan + live TODO

Persistent in-repo record for the empirical-validation work behind
the proposed `--scale-bound` flag and `Bound-RewardRate*` drift family.
Phase 1 (this PR) is purely a verification notebook + a tiny backwards-
compat tweak to the MLE rowwise path. Phase 2 implementation is
deferred until Phase 1 establishes which rescaling is mathematically
sufficient.

## Context

The DDM rescaling identity says a first-passage process with varying
bound `b_t` is observationally equivalent to a process with bound
fixed at 1 IF mu, sigma, AND the starting point z are all rescaled
by `1/b_t`. Before writing production code that exploits this, we
want to empirically check which subset of rescalings is enough — and
whether the equivalence holds to within solver discretization noise.

If only one of mu / sigma needs rescaling to closely match the
varying-bound reference, Phase 2 is ~50 lines. If all three are
required, Phase 2 is ~150-200 lines. If none of them work, we'd need
solver work — but that's the expensive branch and we want to rule it
out (or in) before committing.

## Phase 2 — Branch P2b implementation (DONE)

Phase 1 established that full per-trial (μ, σ, z) rescaling matches
the varying-bound ground truth within solver discretization noise.
Phase 2 implemented branch P2b: new `Bound-RewardRate*` drift family,
`--scale-bound` flag, and all wiring through fit / GUI / notebook
surfaces.

**Test suite**: 143 passed, 1 skipped (up from 135 at the end of
Phase 1; +3 from `test_bound_rewardrate.py` covering the equivalence,
+5 from `test_scale_bound.py` covering the filename suffix grammar,
InitVal override, and absolute-bias semantics).

**Critical bug caught and fixed by the equivalence test**: the
batched evaluator was using path-A latents (mu, sigma, z) with a
scalar BOUND, producing wrong densities. Fixed by rescaling at the
`_evaluate_trial_likelihoods_batched` boundary — see
[mle.py](model/mle.py) `_evaluate_trial_likelihoods_batched`. The
rowwise path was already correct (per-trial bound passed through
from `_compute_latent_arrays`).

**Phase 2 files modified**:

| File | Change |
|---|---|
| `rlmodel/model/drift.py` | New `_boundGainRewardRate` / `_boundGainDecayingQ` (drift / r_t, noise / r_t) + 3 registry entries. |
| `rlmodel/model/mle.py` | `MLEModelConfig.uses_per_trial_bound` + `uses_scaled_bound` flags. Rowwise path emits `bound_per_trial` in latents + overrides sigma to constant. Batched evaluator rescales (mu, sigma, z) by 1/r_t = BOUND/bound_per_trial. Vectorized population path applies the same rescaling. |
| `rlmodel/model/state_updates.py` | `compute_starting_point_z(bound=…)` for absolute-units clip. |
| `rlmodel/model/initvals.py` | `_BOUND_WHEN_SCALED` / `_NOISE_WHEN_SCALED` private constants. |
| `rlmodel/model/fit.py` | `simulateDDM` accepts `scale_bound`, applies InitVal override; derives `uses_per_trial_bound` from `drift_fn_str`. `evolveFP` gains `_scaledB` suffix. |
| `rlmodel/model/logic.py` | `simulateDDMTrial(uses_scaled_bound=...)` skips the `* BOUND` multiplication when bias is in absolute units. |
| `rlmodel/model_runner.py` | `--scale-bound` CLI flag; plumbs through `runModel`. |
| `rlmodel/model/visualize.py` | "Scale Bound" checkbox; gates BOUND / NOISE_SIGMA sliders; extends `_preferred_modes_for` with `_scaledB` suffix; threads `scale_bound` into the GUI MLE evaluator. |
| `rlmodel/model/mle_notebooks/data.py` | `mle_uses_per_trial_bound` / `mle_uses_scaled_bound` columns surfaced in `flatten_mle_results`. |
| `rlmodel/model/mle_notebooks/histograms.py` | `bound·r_t` / `scale-B` in `DEFAULT_LATENT_COLUMNS`. |
| `rlmodel/model_interactive.ipynb` cell 10 | Parser strips `_scaledB` suffix; `fit_key` composes asym + scaledB. |
| `rlmodel/models_loop.sh` | New combo entries (commented) for `--scale-bound` + `Bound-RewardRate*`. |
| `rlmodel/model/tests/test_bound_rewardrate.py` | NEW. 3 tests: equivalence (rowwise A ≡ vectorized D), per-trial bound emitted in latents, symmetric path unaffected. |
| `rlmodel/model/tests/test_scale_bound.py` | NEW. 5 tests: suffix grammar, InitVal constants, InitVals override round-trip, absolute-bias semantics (scalar + array bound). |

## TODO

### 0. Plan + persistent record
- [x] Save this file under `rlmodel/`

### 1. Backwards-compat: rowwise per-trial bound
- [ ] [mle.py](model/mle.py): `_evaluate_trial_likelihoods_rowwise`
      reads per-trial `bound` from `latents["bound_per_trial"]` when
      present; falls back to scalar `_param(params, "BOUND", 1.0)`
      otherwise. Backwards-compat for every existing caller.

### 2. Verification notebook: `rlmodel/scale_bound_equivalence.ipynb`
- [ ] Cell 1 — package shim (mirrors mle_notebooks layout)
- [ ] Cell 2 — synthetic fixture builder (3 sessions × 50 trials,
      sinusoidal reward-rate trajectory in [0.4, 0.8])
- [ ] Cell 3 — `compute_likelihoods(parameterization, ...)` helper
      that dispatches the four paths:
      - A. Varying bound (rowwise; `bound[t] = BOUND_0 × r_t`)
      - B. Scale drift only (`mu[t] = mu_0 / r_t`)
      - C. Scale noise only (`sigma[t] = sigma_0 / r_t`)
      - D. Full rescaling (`mu/r_t, sigma/r_t, z/r_t`)
- [ ] Cell 4 — bulk-likelihood comparison: per-pair total
      |Δ neg_loglik|, per-trial max |Δ logL|, summary table
- [ ] Cell 5 — diff distribution plots: scatter vs. r_t,
      histogram of |Δ density| for each (B/C/D) vs A
- [ ] Cell 6 — interactive single-trial DDM viewer styled after
      `mle_population_explorer.ipynb`'s `ddm_viewer.py`; reuses
      `build_ddm_trial_buffer` and `plot_ddm_frame`
- [ ] Cell 7 — diff-overlay single-trial view (A vs B/C/D on the
      same density panels)
- [ ] Cell 8 — markdown conclusion template (to be filled in after
      running)

### 3. Run + capture results
- [ ] Execute the notebook end-to-end on the `wfield` env
- [ ] Fill in Cell 8 with the observed numbers
- [ ] Decide which Phase 2 branch (P2a / P2b / P2c — see below) applies

### 4. Smoke verification
- [ ] Existing test suite still green: `pytest rlmodel/model/tests/`
      → 135 passed, 1 skipped (the rowwise tweak is backwards-compat)

## Phase 2 (deferred; choice depends on Phase 1)

### Branch P2a — "scale-σ only" is good enough
If Cell 4 shows path C matches path A to within ~1e-6:
- `Bound-RewardRate*` drift family becomes pure re-parameterization
  of `NoiseGain-RewardRate*` (same Python function).
- `--scale-bound` flag implements "fit bound, fix noise" via
  `InitVals.override(...)`.
- No solver changes. ~50 lines total.

### Branch P2b — only "full rescaling" matches
If only path D matches:
- `Bound-RewardRate*` drift family rescales (mu, sigma, z) per trial
  in `_compute_latent_population_equal_sessions`. Solver unchanged.
- `--scale-bound` flag wires bias-in-absolute-units; rest is the
  same InitVal override.
- ~150-200 lines.

### Branch P2c — no re-parameterization is close enough
If even path D diverges from path A meaningfully:
- Investigate whether divergence is solver discretization or a math
  bug.
- Consider per-trial-bound solver work (the side-car solver from the
  earlier plan-mode discussion) — or fix the upstream assumption.

## Files modified (Phase 1)

| File | Change |
|---|---|
| `rlmodel/model/mle.py` | `_evaluate_trial_likelihoods_rowwise` reads optional `bound_per_trial` from latents; backwards-compat with the existing scalar-bound callers. |
| `rlmodel/scale_bound_equivalence.ipynb` (NEW) | Interactive 8-cell verification notebook. |
| `rlmodel/scale_bound_equivalence_plan.md` (NEW) | This file. |

## Reuse / patterns to mirror

- `mle.evaluate_neg_loglik` — single entry point per parameterization
- `MLEModelConfig(mle_use_batched_likelihood=False)` — toggles rowwise
  path on for paths A
- `ddm_viewer.build_ddm_trial_buffer` / `plot_ddm_frame` — reused for
  the single-trial DDM viewer; the notebook just imports them
- `prepare_mle_data` — turns DataFrame into PreparedMLEData consumed
  by both compute paths

## Phase 2.5 — InitVals consolidation + GUI Scale-How dropdown (DONE)

Phase 2 left two GUI rough edges: the `Scale Bound` checkbox driving
the scale-pair swap was a binary control for what is fundamentally a
2-way mode, and the InitVal ranges for the frozen counterparts lived
as module-level constants in `initvals.py` (invisible from the GUI).
Phase 2.5 cleans both up:

- **InitVals dataclass renames**:
  - `InitVals.BOUND` (old: `(1, 1, 1)`, frozen) → `InitVals._BOUND_FIXED`
  - module `_BOUND_WHEN_SCALED` (`(0.3, 5.0, 1.0)`, fittable) → `InitVals.BOUND`
  - module `_NOISE_WHEN_SCALED` (`(1.0, 1.0, 1.0)`, frozen) → `InitVals._NOISE_FIXED`
  - `InitVals.NOISE_SIGMA` unchanged
  - The two module-level constants are deleted; their bodies now live
    inside the dataclass with explicit underscored names that mean
    "this is the frozen counterpart of the paired axis".
- **`fit.simulateDDM`** reads `init_vals._BOUND_FIXED` /
  `init_vals._NOISE_FIXED` and applies one of the two as an
  `InitVals.override(...)` on the canonical `BOUND` / `NOISE_SIGMA`
  key, depending on `scale_bound`. The legacy default branch
  (`scale_bound=False`) is now an *active* override — necessary
  because the dataclass default for `BOUND` is now the fittable
  range, not frozen.
- **`visualize.py`**: dropped the `Scale Bound` checkbox; added a
  `Scale-How` dropdown (`Noise` / `Bound`) at the top of the first
  column, ahead of `Drift Fn`. The new `_BOUND_FIXED` / `_NOISE_FIXED`
  InitVals fields auto-appear as sliders via the existing
  `init_vals.items()` loop; placed alongside their fittable siblings
  (`NOISE_SIGMA` ↔ `_NOISE_FIXED`, `BOUND` ↔ `_BOUND_FIXED`) in the
  first column. The dropdown drives a 4-slider enable/disable gate
  (the inactive pair grays out) plus a kwarg-translation fix-up that
  maps the active-axis slider to the canonical `BOUND` /
  `NOISE_SIGMA` kwarg names `runAndPlot` consumes. The auto-apply
  path's change-detection tuple now includes the composed variant
  suffix (asym + scaled_b) so toggling Scale-How re-loads the
  matching `mle_scaledB` defaults.

**Test suite**: 144 passed, 1 skipped (was 143 at end of Phase 2; +1
from the new `test_init_vals_dict_exposes_all_four_scale_pair_fields`
asserting all four scale-pair fields surface in `InitVals.toDict()`).

**Phase 2.5 files modified**:

| File | Change |
|---|---|
| `rlmodel/model/initvals.py` | Rename `BOUND` → `_BOUND_FIXED`; add fittable `BOUND` (range from old `_BOUND_WHEN_SCALED`); add `_NOISE_FIXED` (from old `_NOISE_WHEN_SCALED`); drop module-level scale-pair constants; docstring; TODO marker near `MLE_TERMINAL_C`. |
| `rlmodel/model/fit.py` | `simulateDDM` uses `init_vals._BOUND_FIXED` / `init_vals._NOISE_FIXED`; override branches on `scale_bound` to preserve bit-exact legacy frozen-BOUND default. |
| `rlmodel/model_runner.py` | `--scale-bound` help text references `InitVals.BOUND` / `_NOISE_FIXED` instead of the deleted module-level constants. |
| `rlmodel/model/visualize.py` | Drop `Scale Bound` checkbox; add `Scale-How` dropdown to `drop_downs_labels`; layout pop swaps the checkbox for the dropdown at the top of `first_col`; new `_BOUND_FIXED` / `_NOISE_FIXED` sliders interleaved beside their fittable siblings; 4-slider enable/disable gating on `Scale-How.value`; kwarg-translation fix-up before `runAndPlot`; `_scaled_bound_suffix` + `_evaluate_mle_loss_for_gui` read the dropdown; `last_asym_suffix` renamed to `last_variant_suffix` (composed asym + scaled_b) so Scale-How toggles re-trigger the auto-apply. |
| `rlmodel/model/tests/test_scale_bound.py` | Drop deleted-constant imports; `test_init_val_fields_describe_scale_pair_contract` asserts on the four dataclass fields; `test_init_val_override_swaps_in_initvals_dict` uses `iv._BOUND_FIXED` / `iv._NOISE_FIXED`; new `test_init_vals_dict_exposes_all_four_scale_pair_fields`. |

## Follow-ups

- [ ] **Move `MLE_TERMINAL_C` into `InitVals`** (user-requested, deferred
      from Phase 2.5). Safe because the fit-param-set derivation in
      `fit.py:simulateDDM` keys off `_MAKEONERUN_FITTABLE_PARAMS` plus
      the bias/drift/noise kwarg lists rather than `InitVals.items()`,
      so non-fittable dataclass fields like `_BOUND_FIXED` already
      coexist with the optimizer without leaking into the fit vector.
      Touchpoints: drop the module-level `MLE_TERMINAL_C` constant from
      `initvals.py`; update every `from .initvals import MLE_TERMINAL_C`
      caller (`fit.py`, `model_runner.py`, `visualize.py`, several
      tests, `mle.py`) to read `InitVals().MLE_TERMINAL_C` instead;
      update the `createWidget` block that manually creates the
      `MLE_TERMINAL_C` slider (lines 110-116) so it falls out of the
      auto-loop like every other field.

## Checkpoint 1 (still parked)

> Whether to check for invalid trials at mle.py. Keep the explicit
> `xp.where(valid_t, ...)` mask in
> `_compute_latent_population_equal_sessions`. Not affected by this
> work.
