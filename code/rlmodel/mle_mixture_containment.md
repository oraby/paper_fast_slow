# Plan — Contamination / Lapse Mixture for the MLE Loss

## Context

Introduce a new fittable parameter `λ ∈ [0, 1)` that mixes a uniform-RT,
uniform-choice lapse density into every trial's likelihood. With `λ = 0.02`,
per-trial likelihood is floored at `log(λ / (2·T_max)) ≈ −5.8` instead of −691,
and DE can find a meaningful optimum without being driven by anticipations /
fast guesses.

**User decision** (asked during planning): the lapse mixture applies to
*all* trials with the same formula — both choice and no-choice — i.e.
`L = (1−λ)·L_DDM + λ/(2·T_max)` uniformly. This is unit-mixed
(probability + density for no-choice) but gives the user a single knob.

## Formula

For any trial that today receives a non-floor likelihood (i.e. valid solver
input AND either a valid decision time OR a no-choice flag):

```
L_i = (1 − λ) · L_DDM_i  +  λ / (2 · T_max)
```

Where `L_DDM_i` is:
- choice trials: first-passage density at the observed (choice, RT)
- no-choice trials: `terminal_no_decision_mass` (already governed by
  `mle_terminal_c`)

For trials that today receive `LOGLIK_FLOOR` because the solver itself
rejected the inputs (NaN sigma, sigma ≤ 0, z out of bounds, NaN choice on
a non-no-choice trial, RT outside `(0, T_max]`), the floor remains —
the lapse mixture is not a back-door around the data-quality floor.

`λ` is fit jointly with the other DE parameters. Range / init from
`InitVals.LAPSE_RATE = InitVal(0.0, 0.1, 0.02)`. To disable lapse entirely
without code changes: `--init-val LAPSE_RATE=0,0,0`.

## Approach summary

1. Register `LAPSE_RATE` in `InitVals` — flows through the existing
   `toDict()` / `--init-val` machinery automatically.
2. Insert `LAPSE_RATE` into `fit_params_li` **only when `fit_mode == "mle"`**,
   and guard the chisq-side assertion in `fit.simulateDDM` so the MLE-only
   parameter doesn't trip the "every fit param must dispatch to a sub-fn"
   check.
3. Read the value out in the MLE population path
   (`_compute_latent_population_equal_sessions`) and the rowwise path
   (`_compute_latent_arrays`), the same way `non_decision_time` is read.
4. Thread `lapse_rate` (per-trial numpy/xp array, identical shape contract
   to `non_decision_time`) through `batched_choice_rt_loglik` →
   `_evaluate_batch` → the per-trial likelihood reduction. Apply the
   mixture there.
5. Update the rowwise path (`trial_choice_rt_loglik`) for completeness —
   apply the same mixture so batched and rowwise remain numerically
   consistent for any caller.
6. Tests: `λ=0` reproduces existing behavior; `λ=0.02` floors choice and
   no-choice likelihoods at `λ/(2·T_max)`; per-trial mixture invariant
   when DDM density is large; rowwise/batched equivalence at λ>0.

## Files to change

### `code/rlmodel/model/initvals.py`
Add one line in the `InitVals` dataclass field block:
```python
LAPSE_RATE : InitVal = InitVal(0.0, 0.1, 0.02)
```
No other changes. `toDict()`, `.items()`, and `--init-val` already handle
new fields uniformly.

### `code/rlmodel/model/fit.py` (`simulateDDM`)
- After `fit_params_li` is built from the union of
  `_makeOneRun_fit_params`, `driftFn_kwargs_li`, `noiseFn_kwargs_li`,
  `biasFn_kwargs_li` (around line ~395, before `list()` conversion): if
  `fit_mode == "mle"`, add `assertInBoundsAndDefaults("LAPSE_RATE")` to
  the set.
- Around lines 458–462, the assertion
  `assert not len(unused_fit_idxs), ...` will fail on the MLE path because
  no dispatch index set claims `LAPSE_RATE`. Add a `mle_only_param_names`
  set and exclude those indices before the assertion fires:
  ```python
  mle_only_param_names = {"LAPSE_RATE"} if fit_mode == "mle" else set()
  mle_only_idxs = {i for i, n in enumerate(fit_params_names)
                   if str(n).upper() in mle_only_param_names}
  unused_fit_idxs = list(
      set(range(len(fit_params_names))) - used_fit_idxs - mle_only_idxs)
  ```
- No `_makeOneRun` signature changes — keep `LAPSE_RATE` invisible to the
  chisq path entirely.

### `code/rlmodel/model/mle.py`
- `MLEModelConfig`: no new field needed (`λ` is a fit parameter, not a
  config knob — same as `ALPHA`/`BETA`).
- `_compute_latent_arrays` (rowwise per-subject): read
  `lapse_rate = _param(params, "LAPSE_RATE", 0.0)` and stash in the
  `latents` dict (or pass as scalar to the likelihood evaluator).
- `_compute_latent_population_equal_sessions` (population path): pull
  `lapse_rate = _param_population(theta, param_lookup, "LAPSE_RATE", xp, 0.0)`
  — gives a `(S,)` xp array. Don't flatten; the population path already
  flattens `non_decision_time` separately in `objective_from_population`.
- `objective_from_population`: mirror `non_decision_time`'s flattening
  pattern to produce `flat_lapse_rate` as `(S*N,)`:
  ```python
  flat_lapse = backend.xp.repeat(
      backend.xp.asarray(lapse_rate[valid_cand_idx], dtype=float),
      n_trials)
  ```
  Then forward `lapse_rate=flat_lapse` into `batched_choice_rt_loglik`.
- `_evaluate_trial_likelihoods_batched` (rowwise/single-subject MLE eval):
  build a constant `(n,)` `lapse_rate` array from the scalar param and
  forward.
- `_evaluate_trial_likelihoods_rowwise`: forward scalar `lapse_rate` into
  `trial_choice_rt_loglik`.
- Return the value alongside `mu`/`sigma`/`z` in the latents dict so
  `_build_mle_df` can optionally emit it as `mle_lapse_rate`
  (single-value-per-trial column for diagnostic display; trivial column
  add, mirrors how other constant-per-subject scalars are exposed).

### `code/rlmodel/model/mle_batch.py`
- `batched_choice_rt_loglik`: add `lapse_rate=0.0` keyword. Forward to
  `_evaluate_batch`. (Accept scalar OR `(n_trials,)` array — coerce to
  `(n_trials,)` like `non_decision_time` already does.)
- `_evaluate_batch`: accept `lapse_rate`, coerce to `(n_trials,)` numpy,
  push to `xp`, gather `lapse_rate_xp = xp.asarray(lapse_rate[valid_idx])`.
- Apply the mixture in the per-trial reduction (replace the existing
  no-choice / density branch):
  ```python
  density_at_t_xp = xp.where(is_left_xp, upper_at_t_xp, lower_at_t_xp)
  like_ddm = xp.where(no_choice_xp,
                      solver_result.terminal_no_decision_mass_xp,
                      density_at_t_xp)
  lapse_density = lapse_rate_xp / (2.0 * float(tmax))
  like_mixed = (1.0 - lapse_rate_xp) * like_ddm + lapse_density
  like_xp = xp.where(no_choice_xp | valid_decision_xp,
                     like_mixed, LOGLIK_FLOOR)
  like_xp = xp.where(xp.isfinite(like_xp) & (like_xp > 0.0),
                     like_xp, LOGLIK_FLOOR)
  ```

### `code/rlmodel/model/mle_likelihood.py`
- `trial_choice_rt_loglik`: add `lapse_rate=0.0` keyword. After computing
  `likelihood` (the existing branch on `no_choice` vs. choice density),
  apply `likelihood = (1 - lapse_rate) * likelihood + lapse_rate / (2*tmax)`
  before the `LOGLIK_FLOOR` clamp. Adds one line of code; preserves
  rowwise / batched parity. Removes the deferral note added with
  `terminal_c` — for lapse we keep them in sync.

### `code/rlmodel/model_runner.py`
No new CLI flag. `--init-val LAPSE_RATE=...` already controls range/init
via the existing override mechanism (e.g. `--init-val LAPSE_RATE=0,0,0`
disables lapse for an experiment).

### `code/rlmodel/model/tests/test_mle_lapse_rate.py` (new)
Five tests, mirroring the structure of `test_mle_terminal_c.py`:

1. `test_lapse_zero_reproduces_baseline` — with `LAPSE_RATE=0.0` in the
   params dict, the batched evaluator output equals the legacy output
   bit-for-bit (compare against a control config that doesn't pass
   `lapse_rate`).
2. `test_lapse_floors_choice_trials_at_density` — with `λ=0.02`,
   `T_max=3`, a choice trial whose DDM density is essentially zero
   (extreme z/μ) gets `loglik ≈ log(0.02 / (2*3)) ≈ −5.81`.
3. `test_lapse_floors_no_choice_trials_at_density` — same, for a
   no-choice trial whose `terminal_no_decision_mass = 0` (use
   `mle_terminal_c=0` to force this).
4. `test_lapse_invariant_when_ddm_density_dominates` — choice trial
   well inside the DDM bulk: `(1−λ)·f_DDM ≫ λ/(2·T_max)`, so
   `loglik(λ=0.02) ≈ loglik(λ=0)` within a tight tolerance.
5. `test_batched_matches_rowwise_with_lapse` — re-use the existing
   `evaluate_neg_loglik` × `mle_use_batched_likelihood` round-trip in
   `test_mle_batched.py`, but with a non-zero `LAPSE_RATE` in the params
   dict. Both backends must agree to `rtol=1e-8` (same standard as the
   existing equivalence tests).

### `code/rlmodel/model/mle_lapse_rate_plan.md` (new)
Short design doc next to `mle_terminal_c_plan.md`. Captures: the formula,
the deferred deferral-lift on the rowwise path, the rationale for picking
"same formula for both trial types" (user's decision), and the
`--init-val LAPSE_RATE=0,0,0` escape hatch.

## Reuse / patterns to mirror

- **Threading pattern**: `non_decision_time` is already a per-candidate
  scalar that flows through `objective_from_population` →
  `batched_choice_rt_loglik` as a `(n_trials,)` array — `lapse_rate`
  follows the same convention.
- **Parameter extraction**: `_param` (rowwise) and `_param_population`
  (population) in `code/rlmodel/model/mle.py`; both already support a
  default value, so `LAPSE_RATE=0.0` is the natural fallback for callers
  that don't set it.
- **Plan / addendum doc convention**: `code/rlmodel/model/mle_terminal_c_plan.md`
  is the template for the new lapse plan doc — same headings, same
  test-list table style.
- **Per-candidate broadcast in `objective_from_population`**: lines that
  build `flat_nondec` show exactly the broadcast pattern to copy.
- **Assertion guard in `simulateDDM`**: precedent for skipping ALPHA/BETA
  when not include_Q / include_RewardRate (via `manually_passed_params` +
  `extra_ignored_count`) — the LAPSE_RATE skip uses the same idea, just
  conditioned on `fit_mode` rather than include flags.

## Verification

1. **Unit tests**: full `pytest rlmodel/model/tests/` passes. The five new
   tests above + the existing 85 tests; expectation is that pre-existing
   tests are untouched because `LAPSE_RATE` defaults to `0.0` whenever it's
   not in the params dict.
2. **End-to-end MLE fit**: from the project root, run a dry run on the
   case-2 subject:
   ```
   python -m code.rlmodel.model_runner --drift Classic --bias "Q-Val (Offset)" \
       --fit-mode mle --mle-backend GPU --mle-device-id 2 --dry-run
   ```
   Confirm: `LAPSE_RATE` appears in the printed `Fit Params Names`, the
   DE result pickle has it in `params_names` / `OptimRes.x`, and the
   recorded `OptimRes.fun` (neg-loglik) is materially better than the
   pre-lapse baseline.
3. **Floor-count diagnostic**: in `mle_debug.ipynb`, re-run the same
   subject's MLE eval. The bottom histogram should show the −691 spike
   shrink to ~16 → essentially 0 (now bounded at ~−5.8), and
   `summarize_trial_losses` should show `trials at floor : 0`.
4. **Disable-lapse smoke test**: rerun with
   `--init-val LAPSE_RATE=0,0,0` and confirm the result is bitwise-equal
   (modulo DE seed) to a pre-lapse fit on the same code. This proves the
   escape hatch works.
