# MLE Lapse-Rate (Contamination) Mixture — Implementation Notes

## Goal

Add a contamination / lapse mixture to the MLE objective so that a small
number of outlier trials (anticipations, fast guesses, lapses-of-attention,
no-choice tails) can't dominate the loss by underflowing to
`LOGLIK_FLOOR = 1e-300` (log ≈ −691). With a new fittable parameter
`λ ∈ [0, 1)`, each trial's likelihood becomes::

    L_i = (1 − λ) · L_DDM_i  +  λ / (2 · T_max)

With `λ = 0.02` and `T_max = 3 s`, per-trial loglik is floored at
log(0.02 / (2·3)) ≈ −5.8 — small enough that DE optimizes the bulk of
trials, large enough that a meaningful λ value still emerges from the fit.

## Design decisions

**Same formula for choice AND no-choice trials** (user's choice during
planning). This is unit-mixed (density + probability for no-choice trials)
but gives a single knob that protects against every kind of outlier. The
plan considered three options (choice-only, uniform, no-mixture-for-
no-choice); the user picked uniform.

**MLE-only fit parameter.** `LAPSE_RATE` enters DE's fit vector only when
`fit_mode == "mle"`. The chisq path never sees it. This avoids wasting a
DE dimension on the chisq fit and keeps `_makeOneRun`'s signature
unchanged.

**No new CLI flag.** Range/init are controlled via the existing
`--init-val LAPSE_RATE=MIN,MAX[,DEFAULT]` override mechanism. To disable
lapse entirely for an experiment: `--init-val LAPSE_RATE=0,0,0`.

**Default range `InitVal(0.0, 0.1, 0.02)`.** 0.1 is the standard
psychophysics upper bound. 0.02 is the standard initial point — large
enough to be active, small enough to not distort fits when the data is
clean.

**Short-RT extension (added 2026-06).** Choice trials with `RT ≤ T0`
(i.e. `decision_time ≤ 0`) used to short-circuit to `LOGLIK_FLOOR`
before the mixture was applied — defeating its purpose for the very
trials it was designed to protect. After the extension, those trials
are routed through the mixture branch with `L_DDM = 0` (which is the
true DDM density there), so they end up at `log(λ/(2·T_max))` instead
of `log(LOGLIK_FLOOR)`. Mathematically clean: the DDM literally says
those observations are impossible, the lapse model literally says they
are uniformly likely in `[0, T_max]`, so the mixture is exactly
`λ/(2·T_max)`. **Side effect**: the fitted `NON_DECISION_TIME` will
typically be larger than under the pre-extension behavior, because the
optimizer is no longer forced to keep T0 small to avoid the
−691-per-trial penalty. That's the intended outcome — short-RT trials
get labelled as lapses, and the bulk of the data drives T0.

## File changes

| File | Change |
|---|---|
| `model/initvals.py` | One new field: `LAPSE_RATE : InitVal = InitVal(0.0, 0.1, 0.02)` |
| `model/fit.py` (`simulateDDM`) | Add `LAPSE_RATE` to `fit_params_li` when `fit_mode == "mle"`; exclude its index from the "every fit param dispatches to a sub-fn" assertion via a new `mle_only_param_names` set |
| `model/mle.py` (`_compute_latent_population_equal_sessions`) | Extract `lapse_rate` via `_param_population` → `(S,)` xp array → exposed in the returned latents dict |
| `model/mle.py` (`objective_from_population`) | Build `flat_lapse = repeat((S,) → (S·N,))` mirroring `flat_nondec`; pass to `batched_choice_rt_loglik` |
| `model/mle.py` (`_compute_latent_arrays`) | Extract `lapse_rate` via `_param` (scalar default 0.0) and add to latents dict |
| `model/mle.py` (`_evaluate_trial_likelihoods_batched`) | Pass `lapse_rate=float(latents["lapse_rate"])` to `batched_choice_rt_loglik` |
| `model/mle.py` (`_evaluate_trial_likelihoods_rowwise`) | Pass `lapse_rate=latents["lapse_rate"]` into each `trial_choice_rt_loglik` call |
| `model/mle.py` (`_build_mle_df`) | New constant-per-trial column `mle_lapse_rate` (lets the debug notebook display the fitted value alongside per-trial diagnostics) |
| `model/mle_batch.py` (`batched_choice_rt_loglik`) | New `lapse_rate=0.0` kw arg; forwarded to `_evaluate_batch` |
| `model/mle_batch.py` (`_evaluate_batch`) | Coerce `lapse_rate` (scalar or `(n_trials,)`) to a per-valid-trial xp array; apply `(1-λ)·L_DDM + λ/(2·T_max)` to `like_xp` before the floor mask |
| `model/mle_likelihood.py` (`trial_choice_rt_loglik`) | New `lapse_rate=0.0` kw arg; apply the mixture to `likelihood` before the floor clamp |
| `model/tests/test_mle_lapse_rate.py` | Five tests pinning the invariants below |

No changes needed in `model_runner.py` (`--init-val LAPSE_RATE=...` works
out of the box), `posterior_simulate.py` (DDM simulator deliberately
doesn't reproduce the lapse process), or `result_payload`
(params_names / OptimRes.x flow generically).

## Tests pin

1. `test_lapse_zero_reproduces_baseline` — `λ=0` gives identical loglik
   to passing `LAPSE_RATE` not at all.
2. `test_lapse_floors_choice_trials_at_lapse_density` — a tail-event
   choice trial gets loglik ≈ `log(λ/(2·T_max))` instead of LOGLIK_FLOOR.
3. `test_lapse_floors_no_choice_trials_at_lapse_density` — with
   `terminal_c=0`, no-choice trials also collapse to the lapse floor.
4. `test_lapse_invariant_when_ddm_density_dominates` — when
   `f_DDM ≫ λ/(2·T_max)`, per-trial Δloglik between λ=0 and λ=0.02
   is < 0.05.
5. `test_batched_matches_rowwise_with_lapse` — both paths agree to
   `rtol=1e-8` with `λ=0.05`.

## Verification (post-merge)

1. `pytest rlmodel/model/tests/` — full suite passes (90 existing + 5 new).
2. Dry-run a single MLE fit:
   ```
   python -m code.rlmodel.model_runner --drift Classic \
       --bias "Q-Val (Offset)" --fit-mode mle --mle-backend GPU \
       --mle-device-id 2 --dry-run
   ```
   Confirm `LAPSE_RATE` appears in the printed `Fit Params Names`.
3. Re-run the case-2 subject through `mle_debug.ipynb`. The −691 floor
   spike in `plot_trial_loss_histogram(mle_df)` should disappear and
   `summarize_trial_losses` should report `trials at floor : 0`.
4. Disable-lapse sanity: `--init-val LAPSE_RATE=0,0,0` reproduces a
   pre-lapse fit (modulo DE seed).
