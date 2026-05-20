# Assessment: Is the Nashaat–Oraby MLE Fitting Spec Coherent?

## Context

The user wants a meta-evaluation of [rlmodel/nashaat_oraby_mle_fitting_spec.md](rlmodel/nashaat_oraby_mle_fitting_spec.md) — specifically whether it is coherent enough to be broken down into a concrete work plan, or whether further clarification is needed before an implementation agent could execute it.

I explored the existing codebase to ground the assessment in what actually exists in [rlmodel/model/](rlmodel/model/), in particular [logic.py](rlmodel/model/logic.py), [fit.py](rlmodel/model/fit.py), [drift.py](rlmodel/model/drift.py), [bias.py](rlmodel/model/bias.py), [noise.py](rlmodel/model/noise.py), [initvals.py](rlmodel/model/initvals.py), and the dataframe semantics consumed by the pipeline.

## Headline verdict

The spec is **mostly coherent and well-structured** — roughly 80% was implementation-ready as written. It correctly maps existing files, correctly identifies the noise-scaling bug (the current code in [noise.py:21](rlmodel/model/noise.py#L21) and [drift.py:24](rlmodel/model/drift.py#L24) does use `sigma * dt`, not `sigma * sqrt(dt)`), and correctly notes no first-passage / MLE code exists today.

It had 5 substantive ambiguities and 4 minor inconsistencies. **All 7 of the most consequential items have now been resolved by the user via the Q&A pass** — see the Spec Addendum section below. With the addendum applied to the spec, the work can be broken down into an implementation plan.

---

## Substantive ambiguities (need resolution before work-plan breakdown)

### 1. Variant mapping does not match the existing model family

The spec describes **four** variants: Classic / Q+DDM / R+DDM / Q+R+DDM, with Q-learning modifying the **starting point** and R-learning modifying the **noise gain**.

The actual code in [drift.py:174-181](rlmodel/model/drift.py#L174-L181) and [bias.py:52-60](rlmodel/model/bias.py#L52-L60) exposes a Cartesian product of *drift × bias × noise* functions. The Q effects in the existing code are mostly **"Decay Q"** drift variants (Q value decays across the trial inside the DDM accumulation), not pure Rescorla–Wagner Q-value-as-starting-point. Pure Q-as-starting-point exists only as the `Q-Val` / `Q-Val (Offset)` bias function.

The spec then says under "Out of scope": **"Implement Decaying-Q model variants"** — which would exclude most of the Q-bearing variants in the existing codebase from MLE coverage.

**Open question:** Does each of the 4 spec variants map to a specific existing (drift, bias, noise) combination, or is the MLE path defining a *new, simpler* model family that doesn't correspond to the existing one? If the latter, the relationship between MLE-Q and existing Decay-Q must be stated.

### 2. Default `--fit-mode` behavior is self-contradictory

Spec §"Backward compatibility":
> Default behavior must remain equivalent to the old user-facing workflow except for the requested correction to diffusion noise scaling. **That means existing calls without `--fit-mode` should fail with a clear message only that no mode is selected.**

These two sentences contradict each other. Either default = `chisq` (preserve old workflow) or default = error (force explicit choice). Pick one.

### 3. Choice-only MLE: in or out of scope?

- "Out of scope" section: "Implement choice-only MLE."
- "Choice-only likelihood" subsection (§lines 426–438): describes exactly how to implement it.
- CLI section: mentions `--mle-observation-model {choice, choice_rt}` and optional `mle-choice` mode.

Pick one — preferably out-of-scope-for-now, given the resolved decision to prioritize choice+RT.

### 4. Dataframe column-name mapping is implicit, not explicit

The spec uses generic names (`coherence_signed`, `sampling_time`, `observed_reward`, `observed_choice_left`) but the actual dataframe uses [util.py:41,47](rlmodel/model/util.py) and [logic.py:149](rlmodel/model/logic.py#L149) conventions:

| Spec name | Actual column |
|---|---|
| `coherence_signed` | `DV` |
| `sampling_time` (RT) | `calcStimulusTime` |
| `observed_reward` | `ChoiceCorrect` (1=rewarded, 0=not, NaN=no-choice) |
| `observed_choice_left` | `ChoiceLeft` |
| `is_valid` | `valid` |
| `is_padding` | implicit — duplicated rows with `valid=False` from [model_runner.py:142](rlmodel/model_runner.py#L142) |
| no-choice marker | `ChoiceLeft = NaN` / `ChoiceCorrect = NaN` |
| session id | `SessId` |

The spec hints at this in §"Resolved design decisions" point 3 ("if the existing dataframe has a clear no-choice marker, use it"), but never spells out the actual mapping. An implementation agent will either re-discover this or guess wrong. The spec should pin the mapping.

### 5. Starting point `z` when `include_Q=False` is unspecified

[`compute_starting_point_z(q_left, q_right, delta, offset, include_Q)`] signature: what value does it return when `include_Q=False`? Probably `offset` or 0, but the spec is silent. Also: is `z` in normalized [-1, 1] space, [-bound, bound] space, or [0, 1] DDM-literature space? The existing [_calcQVal in logic.py:14-22](rlmodel/model/logic.py#L14-L22) normalizes using `log(Q_L / Q_R) / _ciel_max` clipped to [-1, 1], so the convention is [-1, 1] — but the spec should say so explicitly, and the `mle_z` output column should declare its units.

---

## Minor inconsistencies (should fix but won't block work-plan breakdown)

### 6. `--noise-dt-scaling` flag — explicit decision deferred to coding agent

Spec §"Noise-scaling compatibility flag": *"The coding agent should not make this decision alone."* But then immediately delegates the call to the coding agent. Resolved design decision #7 says old results may be overwritten, which implies the legacy `dt` scaling is **not** needed — so the flag can be dropped. Just state this directly.

### 7. "Very poor log likelihood" for `DT_n ≤ 0` is undefined numerically

Spec §"Non-decision time": "If `DT_n ≤ 0`, return a very poor log likelihood for that trial." Define numerically — almost certainly `log(loglik_floor) = log(1e-300)` to stay consistent with the floor used elsewhere.

### 8. `mle-choice-rt` flag scheme is presented two ways

Spec §"Required fitting-mode flag" offers two schemes (single `--fit-mode {chisq, mle, mle-choice, mle-choice-rt}` vs. `--fit-mode` plus separate `--mle-observation-model`). Pick one. Given resolved decision #1 (choice+RT only), the simplest design is `--fit-mode {chisq, mle}` with no observation-model flag at all initially.

### 9. Performance budget is not stated

Spec acknowledges MLE may be slow but gives no acceptance threshold. Worth stating expected per-subject runtime upper bound (e.g., "must complete one subject fit in under 6× current chisq fit time" or similar), so the implementer knows when to stop optimizing.

---

## What is genuinely strong about the spec

- **Resolved design decisions section is excellent** — choice+RT is primary, fixed non-decision time, no-choice → survival mass at Tmax, fixed bound, R-noise stays as `RR * s`, old-result preservation not required. These eliminate the biggest design questions.
- **First-passage method is fully specified** — discrete probability-mass propagation with absorbing bounds, mass-conservation test, return-object dataclass.
- **Test list is concrete and actionable** — one test per claim, with named files.
- **Codegen separation is sensible** — single / single-logged / vectorized variants from one factory, with equivalence tests. The generated-file header convention is clear.
- **Teacher-forcing distinction is correctly emphasized** — MLE uses observed history; posterior simulation uses simulated history. This is the single most important conceptual point and the spec calls it out twice.
- **Output dataframe schema is specific** — concrete column names with `before`/`after` Q/R values and per-trial latents.
- **Refactoring guidance is appropriately cautious** — "branch at three points in logic.py first, only split into mle.py if it gets unreadable."

---

---

## Spec Addendum — Resolved decisions (from live Q&A)

The following items are now decided and should be appended to [nashaat_oraby_mle_fitting_spec.md](rlmodel/nashaat_oraby_mle_fitting_spec.md) as a "Spec Addendum" section. The existing spec body should be edited to remove contradictions with these decisions.

### A. Variant coverage

**Decision:** MLE eventually covers the full existing drift × bias × noise family from [drift.py:174-181](rlmodel/model/drift.py#L174-L181), [bias.py:52-60](rlmodel/model/bias.py#L52-L60), [noise.py:41-44](rlmodel/model/noise.py#L41-L44) — **including Decay-Q drift variants**. The "Out of scope: Implement Decaying-Q model variants" line in the existing spec is now superseded and must be removed.

**Implication for spec body:** "Scope > In scope" must list Decay-Q drift support. "Out of scope" section drops the Decay-Q line.

### B. Initial milestone

**Decision:** The initial milestone implements the full family in one go, including Decay-Q drift variants. No incremental staging.

**Implication for work plan:** the first PR must validate every existing chisq variant under MLE, not just the simple 4 mentioned in the spec.

### C. Default `--fit-mode`

**Decision:** Require `--fit-mode` to be passed explicitly. Existing scripts that omit it must fail with a clear error message ("--fit-mode is required; choose `chisq` or `mle`"). No silent default.

**Implication for spec body:** the contradiction in "Backward compatibility" section is resolved in favor of explicit-required behavior. Remove the "Default behavior must remain equivalent..." sentence.

### D. Choice-only MLE

**Decision:** Out of scope. Initial implementation supports choice+RT only. No `--mle-observation-model` flag. CLI is just `--fit-mode {chisq, mle}`.

**Implication for spec body:** delete the "Choice-only likelihood" subsection (§lines 426–438) and the `mle-choice` / `mle-choice-rt` optional values in the CLI section.

### E. Dataframe column mapping

**Decision:** Hard-code the column mapping in MLE code. No adapter layer.

Concrete mapping (must be documented in the spec):

| Spec name | Actual column | Notes |
|---|---|---|
| `coherence_signed` | `DV` | signed coherence; sign convention from existing code |
| `sampling_time` (RT) | `calcStimulusTime` | seconds |
| `observed_reward` | `ChoiceCorrect` | 1=rewarded, 0=not, NaN=no-choice |
| `observed_choice_left` | `ChoiceLeft` | 1=left, 0=right, NaN=no-choice |
| `valid_for_loss` | `valid` AND `ChoiceLeft.notna()` | excludes padding AND no-choice trials |
| `is_padding` | row duplicated by `_extendTrials` with `valid=False` | from [model_runner.py:142](rlmodel/model_runner.py#L142) |
| no-choice marker | `valid=True` AND `ChoiceLeft.isna()` | real trial, no bound hit |
| session id | `SessId` | from [util.py:47](rlmodel/model/util.py#L47) |

### F. Starting point `z` when `include_Q=False`

**Decision:** `z = 0` (centered, no bias). The `compute_starting_point_z(..., include_Q=False)` function returns 0 unconditionally. The `offset` parameter is only fit when `include_Q=True`.

**Implication:** existing chisq-path "Fixed (Dir)" / "Fixed (Corr/Incorr)" bias functions are *not* supported in MLE. Only `None_` and `Q-Val` / `Q-Val (Offset)` bias functions are supported by MLE in the initial milestone. The spec should declare this clearly.

### G. Decay-Q in first-passage solver (time-varying drift)

**Decision:** Expand the API to support time-varying drift.

**Concrete changes to the spec's first-passage interface:**

```python
# Old (spec as written):
def first_passage_density(z, mu: float, sigma, bound, dt, dx, tmax, backend, return_log) -> FirstPassageResult: ...

# New (after addendum):
def first_passage_density(z, mu, sigma, bound, dt, dx, tmax, backend, return_log) -> FirstPassageResult:
    """mu may be either a scalar (constant drift) or a callable mu(t) (time-varying drift)."""
```

Similarly `compute_trial_mu` returns `float | Callable[[float], float]` depending on whether the variant is Decay-Q. Implementer chooses the most natural representation (per-timestep array probably more practical than a callable).

**Required generated flavors becomes:**

1. `diffusion_single.py` — scalar OR time-varying drift, reference implementation
2. `diffusion_single_logged.py` — same, with state logging
3. `diffusion_vectorized_const_mu.py` — fast path for constant-drift variants (Classic, R-only, Q-without-decay)
4. `diffusion_vectorized_time_mu.py` — fast path for Decay-Q variants (uses shift-trick or FFT convolution)

The dispatcher in `first_passage.py` picks `const_mu` vs `time_mu` based on whether the variant's drift is time-varying. Equivalence tests cover both fast paths against the single reference.

**Performance expectation:** Decay-Q variants run ~3-10× slower than constant-drift variants in the optimized backend. Total fit-time across all 4 variants probably grows 2-4× vs. constant-drift-only.

---

## Recommendation

With the addendum applied to the spec, **the spec is ready to be broken down into a work plan**. The implementation proceeds in 6 sequenced milestones (see Implementation Work Plan below). The user chose "patch spec, then work plan" — so the spec-patch checklist below is executed first, then implementation begins.

---

## Spec-file patch checklist

The following edits to [rlmodel/nashaat_oraby_mle_fitting_spec.md](rlmodel/nashaat_oraby_mle_fitting_spec.md) apply the addendum to the spec body and resolve the contradictions identified in this assessment.

| # | Edit | Status |
|---|---|---|
| 1 | In-scope: add line about supporting full drift × bias × noise family including Decay-Q. Out-of-scope: remove "Changing the model family beyond Classic, Q, R, Q+R" and "Implement Decaying-Q model variants" lines. | **DONE** (applied in previous turn before plan mode re-entered) |
| 2 | "Required structure" file list: rename `diffusion_vectorized.py` to two files: `diffusion_vectorized_const_mu.py` and `diffusion_vectorized_time_mu.py`. | **DONE** |
| 3 | Insert new "Dataframe column mapping" subsection after "Required structure", containing the full mapping table from Spec Addendum decision E. | **DONE** |
| 4 | CLI section: delete the "Optional values, only if easy" block (`mle-choice` / `mle-choice-rt`) and the "If only one MLE mode..." paragraph. | **DONE** |
| 5 | "Backward compatibility" subsection: replace with a paragraph stating `--fit-mode` is required and calls without it fail explicitly. | **DONE** |
| 6 | "Noise-scaling compatibility flag" subsection: replace with a paragraph stating no flag exists, `sqrt_dt` only, metadata records this. | **DONE** |
| 7 | `compute_starting_point_z` signature: add docstring stating returns 0 when `include_Q=False`, declares z is in [-bound, bound] units. | **DONE** |
| 8 | `compute_trial_mu` signature: change return type to `float \| np.ndarray`, add docstring explaining scalar-vs-array semantics for Decay-Q. | **DONE** |
| 9 | Non-decision time subsection: replace "return a very poor log likelihood" with the concrete value `log(1e-300)`. | **DONE** |
| 10 | "No-choice trials" subsection: replace the 3-option open-decision text with the resolved policy (S(Tmax) survival mass). | **DONE** |
| 11 | "Choice-only likelihood" subsection: delete entirely. | **DONE** |
| 12 | `first_passage_density` API: change `mu: float` to `mu: float \| np.ndarray`, add docstring. | **DONE** |
| 13 | "Required generated flavors" subsection: split flavor #3 (`diffusion_vectorized.py`) into `diffusion_vectorized_const_mu.py` and `diffusion_vectorized_time_mu.py`, with rationale and performance expectation. | **DONE** |
| 14 | "Acceptance criteria": update criterion #3 to mention full family + bias-function restriction, update criterion #7 to mention const_mu and time_mu, update criterion #8 to mention time-varying drift. Remove criterion #11 ("Open design decisions need to cleared..."). | **DONE** |
| 15 | Append "Spec Addendum" section (verbatim copy from this plan file) at the very end of the spec. | **DONE** |
| 16 | "Required structure" file list: change `diffusion_codegen.py` (single file) to `diffusion_codegen/` (folder containing `__init__.py`, `templates.py`, `flavors.py`, `__main__.py`). | **DONE** then **SUPERSEDED by #19** (codegen pivot) |
| 17 | "## Generated diffusion functions > Generator location" subsection: replace "Create: `model/diffusion_codegen.py`" with the folder structure. | **DONE** then **SUPERSEDED by #19** |
| 18 | "## Generated diffusion functions > Generator documentation" subsection: docstring example now lives on `generate_all()` in `__init__.py`. | **DONE** then **SUPERSEDED by #19** |
| 19 | **Codegen pivot — replace all codegen references with hand-written design.** Multiple sub-edits to the spec, executed as one logical patch post-exit: <br>(a) "Required structure" file list: replace `diffusion_codegen/` folder with `diffusion/` folder containing `__init__.py`, `single.py`, `vectorized_const_mu.py`, `vectorized_time_mu.py`. Remove `generated/` folder entirely. Drop the 4 codegen test files; keep only `test_diffusion_equivalence.py` and `test_first_passage_mass_conservation.py`. <br>(b) "## Generated diffusion functions" section: rename to "## Diffusion solvers". Drop the "Generator location", "Generator documentation", and "Generation policy" subsections. Replace with a "Hand-written paired functions" rationale paragraph. <br>(c) "Required generated flavors" subsection: rename to "Three hand-written solvers". List the 3 files (`single`, `vectorized_const_mu`, `vectorized_time_mu`) with the new design. Drop `diffusion_single_logged` and `diffusion_vectorized_logged` (the reference always logs; no separate logged flavor). <br>(d) Remove the "AUTO-GENERATED FILE. DO NOT EDIT BY HAND." header convention. Replace with a one-line note that solver files are normal Python; the reference is intentionally slow and readable. <br>(e) Add a note under each function's signature that it accepts an `xp` keyword-only parameter defaulting to numpy, with `cupy` planned as a future GPU swap (not implemented in initial PR). <br>(f) Acceptance criterion #6: rewrite from "Generated diffusion functions exist as source files and include generated-file headers." to "Three diffusion solvers (`single`, `vectorized_const_mu`, `vectorized_time_mu`) exist as hand-written source files in `model/diffusion/`. No auto-gen headers, no regeneration command." <br>(g) Acceptance criterion #7: rewrite to drop "Generated" wording and the regeneration-determinism requirement. New text: "All three diffusion solvers produce numerically equivalent first-passage densities within `rtol=1e-6, atol=1e-9`." <br>(h) Note in the spec body that `diffusion_single` *always* returns logged memory steps — there is no separate `diffusion_single_logged.py`. | partial: file list & solver section DONE; criteria #6/#7 and "always logs" note still pending post-exit |

---

## Implementation Work Plan

After the spec is patched, implementation proceeds in 6 sequenced milestones. Each milestone produces one PR-sized commit with tests passing before moving to the next.

### Milestone 0 — Foundational fixes (smallest, lowest risk)

**Goal:** correct the noise-scaling bug and add the `--fit-mode` CLI flag. No new model logic.

**Files modified:**
- [rlmodel/model/noise.py:21](rlmodel/model/noise.py#L21) — change `* dt` to `* np.sqrt(dt)` in `_noiseNormal`. Update `_noiseDecayingQVal` similarly if it exists.
- [rlmodel/model/drift.py:24](rlmodel/model/drift.py#L24) — verify scaling is correct after the noise.py fix (the chain `noise *= noise_sigma` interacts with the `_noiseNormal` change).
- [rlmodel/model_runner.py:73-88](rlmodel/model_runner.py#L73-L88) — add required `--fit-mode {chisq,mle}` argument. Existing `argparse` setup. Fail with error if missing.
- [rlmodel/model/fit.py](rlmodel/model/fit.py) — accept `fit_mode` parameter in `simulateDDM()`. For now, only `chisq` is implemented; `mle` raises NotImplementedError.

**New files:**
- `code/rlmodel/model/tests/__init__.py`
- `code/rlmodel/model/tests/test_noise_scaling.py` — sample many one-step DDM updates, check empirical variance ≈ `sigma² · dt` (within MC tolerance), confirming `sigma · sqrt(dt) · ε` scaling.

**Verification:** existing chisq fit must still run end-to-end (with `--fit-mode chisq`) and produce a fit-result pickle with `noise_dt_scaling="sqrt_dt"` recorded in metadata. Numerical results will differ from pre-fix runs; that's expected per resolved decision #7.

### Milestone 1 — Shared state-update module

**Goal:** extract Q/R update logic into a shared module used by both chisq and (future) MLE paths.

**Files modified:**
- [rlmodel/model/logic.py](rlmodel/model/logic.py) — replace the inline `_calcQVal()` (lines 14-22), `_updateNextQL_Q()` (lines 25-35), `_updateNextRewardRate()` (lines 38-45) usages with calls to `state_updates.*`. Keep the inline definitions as thin wrappers for backward compatibility, or remove if call sites are easy to update.

**New files:**
- `code/rlmodel/model/state_updates.py` — implement the 6 functions from spec §"Required functions":
  - `initialize_latent_state(include_Q, include_RewardRate) -> LatentState` (dataclass with `q_left=0.5`, `q_right=0.5`, `reward_rate=0.5`)
  - `update_q_values(...)` — call shape matching existing `_updateNextQL_Q`
  - `update_reward_rate(...)` — call shape matching existing `_updateNextRewardRate`
  - `compute_starting_point_z(...)` — return 0 if `include_Q=False`, else clip `δ·log(QL/QR) + offset` to [-1,1]
  - `compute_trial_mu(coherence, drift_coef, time_grid=None)` — for constant drift returns `drift_coef * coherence`; for Decay-Q returns a per-timestep array (Milestone 1 may stub this and return scalar only, with array support added in Milestone 2)
  - `compute_trial_sigma(base_sigma, reward_rate, include_RewardRate)` — `reward_rate * base_sigma` if RR, else `base_sigma`
- `code/rlmodel/model/tests/test_state_updates.py` — Q update only changes chosen side; RR follows Rescorla–Wagner; initial values are 0.5; starting-point z returns 0 when include_Q=False.

**Verification:** existing chisq fit still produces identical (or near-identical, modulo MC noise) results to Milestone 0 output. Q/R updates pass unit tests.

### Milestone 2 — First-passage solver (reference + tests)

**Goal:** numerical first-passage density solver with both scalar and array `mu` support, single non-vectorized reference implementation only.

**New files:**
- `code/rlmodel/model/diffusion/__init__.py` — package init; re-exports the three solver functions.
- `code/rlmodel/model/diffusion/single.py` — reference implementation. Hand-written, always logs, handles both scalar and array `mu`, slow but easy to read. ~80 lines. Includes `xp` keyword-only parameter defaulting to numpy (future GPU swap point).
- `code/rlmodel/model/first_passage.py` — public API: `first_passage_density(z, mu, sigma, bound, dt, dx, tmax, *, backend="auto", xp=np)` and `FirstPassageResult` dataclass. For Milestone 2 it only dispatches to `diffusion_single`; Milestone 3 adds the two vectorized backends.
- `code/rlmodel/model/tests/test_first_passage_mass_conservation.py` — verify `Σf_upper·dt + Σf_lower·dt + survival ≈ 1` for several `(z, mu, sigma)` settings, including time-varying mu.

**Verification:** `diffusion_single` runs successfully on several test trials; mass conservation passes within `atol=1e-6`; logged fields (`x_grid`, `p_by_t`, `upper_mass_by_t`, `lower_mass_by_t`) are populated and shape-consistent with `times`.

### Milestone 3 — Vectorized diffusion backends

**Goal:** fast paths needed for fitting. Two new hand-written files, no codegen.

**Files modified:**
- `code/rlmodel/model/first_passage.py` — dispatcher: with `backend="auto"`, pick `const_mu` for scalar `mu`, `time_mu` for array `mu`, `reference` to force `single`.
- `code/rlmodel/model/diffusion/__init__.py` — re-export the new functions.

**New files (hand-written, normal Python — no auto-gen header):**
- `code/rlmodel/model/diffusion/vectorized_const_mu.py` — fast path for constant drift. Precomputes Gaussian transition kernel once and reuses it across all timesteps. ~80 lines.
- `code/rlmodel/model/diffusion/vectorized_time_mu.py` — fast path for time-varying drift (Decay-Q variants). Uses kernel-shift trick or FFT convolution to handle per-timestep kernel changes. ~80 lines.
- `code/rlmodel/model/tests/test_diffusion_equivalence.py` — numerical equivalence: for several `(z, mu, sigma)` settings, all three solvers must produce `f_upper`, `f_lower`, `survival` agreeing within `rtol=1e-6`, `atol=1e-9`. Covers both scalar and array `mu` inputs.

**Verification:** both vectorized backends agree with the reference within tolerance. Vectorized backends 10–100× faster than `single` on a representative trial. Mass conservation continues to pass for all three.

### Milestone 4 — MLE objective + dispatcher

**Goal:** end-to-end MLE fitting for all supported variants.

**Files modified:**
- [rlmodel/model/fit.py](rlmodel/model/fit.py) — dispatch on `fit_mode`. For `mle`, build the MLE objective and pass to `differential_evolution`. Output pickle gets MLE-specific fields per spec §"Saved fit-result structure".
- [rlmodel/model_runner.py](rlmodel/model_runner.py) — wire `--fit-mode mle` to the new path.

**New files:**
- `code/rlmodel/model/mle.py` — `neg_loglik(params, df, model_config)` implementing the teacher-forced trial loop from spec §"Teacher-forced trial loop". Hard-coded column mapping per addendum decision E (DV, calcStimulusTime, ChoiceCorrect, ChoiceLeft, valid, SessId). No-choice trials contribute `log S(Tmax)`. Loglik floor `log(1e-300)` for invalid decision times.
- `code/rlmodel/model/mle_likelihood.py` — `trial_choice_rt_loglik(...)` utility that calls `first_passage_density` and returns log of `f_{observed_bound}(RT - T0)` or `log S(Tmax)` for no-choice trials.
- `code/rlmodel/model/tests/test_mle_teacher_forcing.py` — tiny fake df with model-predicted choice ≠ observed choice; verify Q/R update uses *observed* choice.
- `code/rlmodel/model/tests/test_mle_smoke.py` — fit on a small synthetic dataset for each of the 4 named variants (Classic, Q-via-bias, R-via-noise, Q+R) plus one Decay-Q variant. Assert finite loss, expected output columns, AIC/BIC computed.

**Verification:** `python model_runner.py --drift Classic --bias None_ --fit-mode mle --dry-run` produces a valid MLE objective evaluation and a per-trial latent-variable dataframe with the columns in spec §"MLE trial-level dataframe".

### Milestone 5 — Visualization + posterior simulation

**Goal:** debug/diagnostic tooling and posterior predictive checks.

**New files:**
- `code/rlmodel/model/mle_visualize.py` — three functions: `visualize_first_passage_result`, `visualize_probability_flow` (uses logged backend), `debug_one_trial_mle_flow`.
- `code/rlmodel/model/posterior_simulate.py` — `simulate_from_fitted_params(df, fitted_params, model_config, n_repeats, seed, use_observed_history_for_inputs)`. Default: simulated-history propagation, calls existing chisq simulation path with MLE-fitted parameters.
- `code/rlmodel/run_posterior_predictive.py` (or similar) — CLI script: takes a saved MLE result pickle, runs posterior simulation, saves outputs in chisq-compatible format so existing non-MLE behavioral plots work.

**Verification:** load an MLE result from Milestone 4, run posterior simulation, confirm output dataframe has the same columns as a chisq output, and existing performance plots render.

---

## Diffusion solver architecture (hand-written, replaces codegen)

The diffusion solver is implemented as **three hand-written Python files** in `model/diffusion/`. The earlier codegen sketch has been retracted.

### Why hand-written over codegen

The original codegen rationale was to avoid `if logged:` branches inside hot loops by generating logged and unlogged variants as separate sources. Hand-written paired functions achieve the same outcome — the reference always logs and is intentionally slow; the fast paths never log. The choice happens at the call site (which function you call), not inside any loop. So the original concern is fully preserved without the templating machinery.

Benefits of hand-written over codegen:

- **Readability** — straight Python, no template indirection. New devs read and modify it without learning a codegen system.
- **Standard tooling works** — IDE jump-to-def, debugger breakpoints, linter, type checker all operate directly on the real code.
- **No regeneration step** — change a function, run tests. No `python -m diffusion_codegen` between edits, no auto-gen header bookkeeping, no risk of stale generated files in git.
- **Future GPU swap (`xp=numpy → xp=cupy`)** — one keyword arg on each function. Adding `xp` to a codegen template would make it another axis in an already 3-axis factor space.
- **Fewer tests** — only numerical equivalence between reference and fast paths matters. The 4-layer codegen test scheme (templates, flavors, compiles, deterministic) is unnecessary.

Cost: ~10-30 lines of duplicated diffusion math between the reference and each fast path. The math is short enough that this is cheaper than the codegen indirection.

### File layout

```text
code/rlmodel/model/diffusion/
    __init__.py
    single.py                  # reference, always logs, scalar or array mu, slow, easy to read (~80 lines)
    vectorized_const_mu.py     # fast, scalar mu only (~80 lines; precomputed kernel reused across steps)
    vectorized_time_mu.py      # fast, array mu only (~80 lines; kernel-shift or FFT convolution per step)
```

All three are normal Python — no auto-gen header, no regeneration command.

### Function signatures

```python
import numpy as np

def diffusion_single(
    z, mu, sigma, bound, dt, dx, tmax, *, xp=np,
) -> FirstPassageResult:
    """Reference implementation. Slow, always logs, scalar or array mu.

    `xp` is the array module — defaults to numpy. Pass `cupy` (or any
    numpy-API-compatible module) for GPU. Gaussian PDF is computed via
    `xp.exp(...)` directly (not scipy.stats) so the function works with
    both numpy and cupy arrays. Not exercised in initial PR — placeholder
    for the future GPU path.
    """

def diffusion_vectorized_const_mu(
    z, mu_scalar, sigma, bound, dt, dx, tmax, *, xp=np,
) -> FirstPassageResult:
    """Fast path for constant drift. `mu_scalar` must be a scalar."""

def diffusion_vectorized_time_mu(
    z, mu_array, sigma, bound, dt, dx, tmax, *, xp=np,
) -> FirstPassageResult:
    """Fast path for time-varying drift. `mu_array` must be a 1-D array of length tmax/dt."""
```

The reference returns the full `FirstPassageResult` dataclass with `x_grid`, `p_by_t`, `upper_mass_by_t`, `lower_mass_by_t` populated — used by visualization and the `debug_one_trial_mle_flow` utility. The fast paths leave those fields as `None`.

**No separate logged variant.** `diffusion_single` *always* returns the logged memory steps; there is no `diffusion_single_logged.py` twin. The memory cost (~one float64 array per step) is acceptable because the reference is never used for fitting — only for tests, visualization, and debugging. Per-call allocation is bounded by `tmax/dt × N_x_bins`, which for typical settings (T_max=3s, dt=1ms, 200 bins) is ~5 MB — negligible for the use cases.

### Dispatcher in `first_passage.py`

```python
def first_passage_density(z, mu, sigma, bound, dt, dx, tmax, *, backend="auto", xp=np):
    if backend == "reference":
        return diffusion_single(z, mu, sigma, bound, dt, dx, tmax, xp=xp)
    if backend == "auto":
        if np.isscalar(mu):
            return diffusion_vectorized_const_mu(z, mu, sigma, bound, dt, dx, tmax, xp=xp)
        return diffusion_vectorized_time_mu(z, mu, sigma, bound, dt, dx, tmax, xp=xp)
    raise ValueError(f"unknown backend: {backend}")
```

### Testing

Two test files only:

- `test_diffusion_equivalence.py` — for several `(z, mu, sigma)` settings, run all three solvers and check `f_upper`, `f_lower`, `survival` agree within `rtol=1e-6, atol=1e-9`. Constant-drift inputs use `single` and `const_mu`; time-varying-drift inputs use `single` and `time_mu`.
- `test_first_passage_mass_conservation.py` — verify `Σf_upper·dt + Σf_lower·dt + survival ≈ 1` for each backend, including time-varying mu.

The 4 codegen-specific test files (`test_codegen_templates`, `test_codegen_flavors`, `test_codegen_compiles`, `test_codegen_deterministic`) are dropped.

### Future GPU support (not in initial PR)

Each function accepts `xp` as a keyword-only parameter, defaulting to `numpy`. CuPy mirrors the numpy API for the operations we need (`exp`, `cumsum`, `clip`, array creation, basic arithmetic). To run on GPU later: `result = diffusion_vectorized_const_mu(z, mu, sigma, ..., xp=cupy)`. No code regeneration, no new file. Initial PR ships with `xp=np` only.

---

## Acceptance gates (top-level)

The implementation is complete when all spec acceptance criteria (as updated in patch #14) pass AND the 6 milestones above are merged. The most important end-to-end check:

```bash
# After Milestone 4:
python model_runner.py --drift "NoiseGain-RewardRate Decay Q" --bias "Q-Val" --fit-mode mle --dry-run
# After Milestone 5:
python run_posterior_predictive.py --fit-result /path/to/mle_result.pkl
# then run existing behavioral plots against the simulated output
```

## Verification strategy

- **Unit tests** (Milestones 0–3): run `pytest rlmodel/model/tests/`.
- **Integration test** (Milestone 4): run MLE in `--dry-run` mode on one subject for each variant; visually inspect the per-trial latent-variable dataframe.
- **Numerical sanity** (Milestone 4): for a constant-drift variant, MLE should recover known synthetic parameters within a reasonable tolerance (parameter recovery test on synthetic data).
- **End-to-end** (Milestone 5): full chain from raw df → MLE fit → posterior simulation → existing behavioral plots.

### Critical files an implementer will need to read first

- [rlmodel/model/logic.py](rlmodel/model/logic.py) — chi2 loss, `simulateDDMMultipleSess`, `_calcQVal`, `_updateNextQL_Q`, `_updateNextRewardRate`
- [rlmodel/model/fit.py](rlmodel/model/fit.py) — `simulateDDM`, differential-evolution wrapper, output pickle structure (lines 151–166)
- [rlmodel/model/drift.py](rlmodel/model/drift.py), [bias.py](rlmodel/model/bias.py), [noise.py](rlmodel/model/noise.py) — current function dictionaries
- [rlmodel/model/initvals.py](rlmodel/model/initvals.py) — parameter bounds (lines 29–42)
- [rlmodel/model_runner.py](rlmodel/model_runner.py) — CLI parser (lines 73–88) and `loadDF`/`_extendTrials` padding logic (line 142)
- [rlmodel/model/util.py](rlmodel/model/util.py) — `SessId` construction, `initDF` (around line 41–47)

### How to verify this assessment

Two checks would confirm or refute the assessment:
- Hand the spec to a fresh implementation agent and watch where it stops to ask clarifying questions — the questions should cluster around items #1–#5.
- Compare the spec's "Required structure" file list against what's actually buildable from the spec text alone, without reading the existing codebase. Items like the column-name mapping (#4) become impossible to write from the spec alone.
