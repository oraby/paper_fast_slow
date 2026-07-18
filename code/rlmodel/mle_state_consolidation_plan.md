# MLE state-update consolidation — plan + TODO

This file is the persistent in-repo record of the consolidation work
described in the project's plan-mode brief. See the approved plan
docstring at the head of this file for the rationale; this section is
the actionable checklist.

## Context (short form)

Three overlapping cleanups in one PR:

1. **State-update math owned by `state_updates.py`** — `mle.py`'s
   `_compute_latent_arrays` and `_compute_latent_population_equal_sessions`
   stop inlining the Q-update / reward-rate-update / Q-value
   normalization / starting-point math and call `state_updates`
   functions instead. `state_updates` gains an `xp=np` kwarg so the
   same code runs scalar/NumPy and on CuPy via the population path.
2. **Loud failure over silent fallback.** `MLEModelConfig` gains two
   flags — `uses_asymmetric_alpha`, `uses_asymmetric_beta` — that
   force strict access (KeyError on miss) for the corresponding
   params. The flags default to False, so old saved pickles still
   load as symmetric fits.
3. **Tidy `fit.py`.** Replace `inspect.signature(makeOneRun)`
   introspection with an explicit fittable-params tuple. Collapse the
   four `manually_passed_params` if-blocks into a declarative
   `_PARAM_FIT_GATES` dict. Switch the NaN sentinel in
   `_makeOneRunWrapper` to `None`. Pass the asym flags into
   `MLEModelConfig`.

## Investigation: keep `_compute_latent_arrays` as a thin wrapper

Eliminating the scalar path (forcing all callers through the
population path with n_candidates=1) costs ~50–80 lines at call
sites + extending the population path to also output per-trial Q
states for `_build_mle_df` + requiring equal-session-padded data
everywhere. Net: more code than saved. Decision: **keep the scalar
path as a ~25-line orchestration wrapper** that delegates Q-update /
reward-rate update / Q-value normalization to `state_updates`. Both
paths converge on the same math through `state_updates`, so the
drift risk is gone even though both functions remain.

## TODO

### state_updates.py
- [x] Add `xp=np` kwarg to `compute_q_value`, `update_q_values`,
      `update_reward_rate`, `compute_starting_point_z`,
      `compute_trial_mu`
- [x] Drop NaN-special-case fallback in asymmetric branch (None-only)

### mle.py
- [x] Add `uses_asymmetric_alpha` / `uses_asymmetric_beta` fields to
      `MLEModelConfig` (default `False`, preserves old pickles)
- [x] Rewrite `_compute_latent_arrays` as a thin wrapper calling
      `state_updates` (~25 lines)
- [x] Rewrite `_compute_latent_population_equal_sessions` to call
      `state_updates` with `xp=backend.xp` + broadcast inputs
- [x] Drop dead helpers: only `_row_record` was truly dead;
      `_compute_z` / `_compute_mu` / `_decaying_q_noise` are imported
      by `posterior_simulate.py`, `mle_visualize.py`, and
      `ddm_viewer.py` so they remain (documented as per-trial scalar
      helpers for external callers).
- [x] Replace silent `_param(name, default)` with strict access for
      flag-gated params; keep documented `_param_population` defaults
      only for genuinely-optional params (`BIAS_COEF`, `Q_VAL_OFFSET`,
      `Q_VAL_COEF`, `Q_VAL_DECAY_RATE`, `LAPSE_RATE`,
      `NON_DECISION_TIME`, `BOUND`)

### fit.py
- [x] Keep `inspect.signature(makeOneRun)` for structural kwarg
      discovery (load-bearing for fixed_params bookkeeping); document
      that fittability is the job of `_PARAM_FIT_GATES`, not discovery
- [x] Collapse the four `manually_passed_params` if-blocks into a
      declarative `_PARAM_FIT_GATES` table
- [x] Switch frozen-param sentinels from `np.nan` to `None` for
      `ALPHA_UNREWARDED` / `BETA_UNREWARDED` (ALPHA / BETA keep `NaN`
      — `makeOneRun` asserts on it when include_Q / include_RewardRate
      is True, so it's a deliberate "must be set" guard)
- [x] Pass `include_Q_asym` / `include_RewardRate_asym` into
      `MLEModelConfig(...)`
- [x] Add pre-flight assertions that flag-required params are in
      `fit_params_names`

### model_runner.py
- [x] Audit-only; CLI plumbing unchanged (asym flags derived inside
      `simulateDDM` from `bias_fn_str` / `drift_fn_str`)

### visualize.py
- [x] Pass asym flags from `biasFn_str` / `driftFn_str` into the
      `MLEModelConfig` built by `_evaluate_mle_loss_for_gui`

### Tests
- [x] `test_state_updates.py`: new `xp=np`-with-array tests pinning
      the vectorized (population-shape) semantics
- [x] `test_asymmetric_lr.py`: update fixtures with new
      `MLEModelConfig` fields; add `KeyError`-on-missing-param
      regression

### Notebooks (manual verification — please confirm on next run)
- [ ] `mle_debug.ipynb`: chisq vs MLE param comparison still renders
- [ ] `mle_population_explorer.ipynb`: all 80 pickles load (old
      configs default-False on the new flags); DDM viewer shows
      `-asym` fits side-by-side with canonical
- [ ] `model_interactive.ipynb`: "Run MLE" works for both canonical
      and `-asym` selections

## Files modified

| File | Change |
|---|---|
| `rlmodel/model/state_updates.py` | `xp` kwarg on `compute_q_value`, `update_q_values`, `update_reward_rate`, `compute_starting_point_z`, `compute_trial_sigma`. Drop NaN special-case in the asymmetric fallback. |
| `rlmodel/model/mle.py` | Add `uses_asymmetric_alpha` / `uses_asymmetric_beta` to `MLEModelConfig`. Rewrite `_compute_latent_arrays` as a thin wrapper that delegates to `state_updates`. Rewrite `_compute_latent_population_equal_sessions` to call `state_updates` with `xp=backend.xp`. Delete dead helpers. Convert silent `_param(name, default)` patterns to strict access on flag-gated params; keep documented `_param_population` defaults only for genuinely-optional MLE params. |
| `rlmodel/model/fit.py` | Replace `inspect.signature(makeOneRun)` introspection with explicit `_MAKEONERUN_FITTABLE_PARAMS` tuple. Collapse the four `manually_passed_params` if-blocks into a declarative `_PARAM_FIT_GATES` table. Switch `_makeOneRunWrapper` frozen-param sentinels from `np.nan` to `None`. Pass `include_Q_asym` / `include_RewardRate_asym` into `MLEModelConfig(...)`. Pre-flight assertions on flag↔fit-param consistency. |
| `rlmodel/model_runner.py` | Audit-only. |
| `rlmodel/model/visualize.py` | Compute asym flags from `biasFn_str` / `driftFn_str` in `_evaluate_mle_loss_for_gui` and pass into the `MLEModelConfig` it constructs. |
| `rlmodel/model/mle_batch.py` | Audit-only — no state-update duplication; preserve every batched optimization. |
| `rlmodel/model/tests/test_state_updates.py` | New `xp=np` array tests pinning the vectorized semantics. |
| `rlmodel/model/tests/test_asymmetric_lr.py` | Update fixtures with new `MLEModelConfig` fields. Add a `KeyError`-on-missing-param regression. |

## Verification

1. Unit tests: full `pytest rlmodel/model/tests/` still **127 passed,
   1 skipped**.
2. Chisqr↔MLE byte-equivalence test in `test_asymmetric_lr.py` still
   passes after the consolidation.
3. CLI smoke: `models_loop.sh` with `Classic|Q-Val-asym (Offset)|...`
   prints `ALPHA_UNREWARDED` in `Fit Params Names`,
   `uses_asymmetric_alpha=True` in the model_config dump, runs DE
   without KeyError.
4. Notebooks: see "Notebooks (manual verification)" above.
5. Failure mode: `MLEModelConfig(uses_asymmetric_alpha=True, ...)`
   with a params dict missing `ALPHA_UNREWARDED` raises
   `KeyError: 'ALPHA_UNREWARDED'`.
