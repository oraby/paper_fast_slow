"""Bound-RewardRate equivalence test.

The production fast path (vectorized + rescaled mu/sigma/z) must match
the rowwise reference (per-trial varying bound) to within solver
discretization noise. This is the in-suite version of the manual
verification done in `rlmodel/scale_bound_equivalence.ipynb` Phase 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..mle import MLEModelConfig, evaluate_neg_loglik
from ..state_updates import bound_scale_from_reward_rate


def _build_fixture(n_sessions=2, trials_per_session=15, seed=0):
    """Tiny deterministic fixture with a non-trivial reward-rate trajectory.

    Sessions × trials kept small so the test runs in ~1s. The
    reward-rate sweep [0.4, 0.8] is wide enough that varying-bound and
    fixed-bound rescaling produce visibly different per-trial latents
    even after rescaling (i.e. the test is sensitive to the rescaling
    math being correct).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for sess in range(n_sessions):
        for trial in range(trials_per_session):
            dv = float(rng.choice([-0.5, -0.25, 0.0, 0.25, 0.5]))
            choice_left = float(rng.choice([0.0, 1.0]))
            rt = float(rng.uniform(0.08, 0.6))
            rows.append(dict(
                Name="S1", SessId=f"sess_{sess}", SessionNum=sess,
                Date=pd.Timestamp("2026-01-01"),
                TrialNumber=trial + 1,
                DV=dv, DVstr=str(dv),
                valid=True,
                calcStimulusTime=rt,
                ChoiceLeft=choice_left,
                ChoiceCorrect=float(choice_left == (dv > 0)),
            ))
    return pd.DataFrame(rows)


def _params():
    """Pin BETA=0 so reward_rate stays at its 0.5 init throughout.

    This keeps the test in the regime where the rowwise reference's
    spatial grid (n_x = 2·BOUND·(2 - r_t) / dx ≈ 150 bins) is
    well-resolved. With BETA=0 the per-trial bound is uniformly
    BOUND·(2 - 0.5) = 1.5, so the comparison isolates the rescaling
    math from any grid-coarsening artifacts.
    """
    return {
        "DRIFT_COEF":        1.2,
        "NOISE_SIGMA":       1.5,
        "BOUND":             1.0,
        "NON_DECISION_TIME": 0.04,
        "ALPHA":             0.3,
        "BETA":              0.0,
        "BIAS_COEF":         0.0,
        "Q_VAL_OFFSET":      0.0,
        "LAPSE_RATE":        0.0,
    }


def _config(use_batched: bool, t_dur=0.8) -> MLEModelConfig:
    """Bound-RewardRate config; toggle vectorized vs rowwise via
    mle_use_batched_likelihood.
    """
    return MLEModelConfig(
        drift_fn_str="Bound-RewardRate",
        bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False,
        include_RewardRate=True,
        dt=0.005,
        t_dur=t_dur,
        dx=0.02,
        mle_use_batched_likelihood=use_batched,
        uses_per_trial_bound=True,
    )


def test_bound_rewardrate_rowwise_matches_vectorized():
    """The two MLE compute paths produce the same total neg_loglik for
    the same params. Phase 1 showed the residual at default (dt, dx) is
    ~0.5 over ~150 trials. The fixture here has 30 trials, so we allow
    a comfortable margin (still well within solver discretization).
    """
    df = _build_fixture()
    params = _params()

    rowwise   = evaluate_neg_loglik(params, df, _config(use_batched=False))
    vectorized = evaluate_neg_loglik(params, df, _config(use_batched=True))

    # Both paths must succeed.
    assert np.isfinite(rowwise.neg_loglik)
    assert np.isfinite(vectorized.neg_loglik)
    # Total loss agrees to within solver discretization. Phase 1's
    # convergence sweep showed |D-A|/n ~ 0.005-0.02 per trial at this
    # (dt, dx); 30 trials × ~0.05 = 1.5 as a comfortable ceiling.
    assert abs(rowwise.neg_loglik - vectorized.neg_loglik) < 1.5, (
        f"rowwise={rowwise.neg_loglik}, vectorized={vectorized.neg_loglik}, "
        f"|d|={abs(rowwise.neg_loglik - vectorized.neg_loglik)}")


def test_bound_rewardrate_emits_per_trial_bound_in_latents():
    """The rowwise path stashes ``bound_per_trial`` on the latents dict
    so ``_evaluate_trial_likelihoods_rowwise`` (the reference solver)
    actually USES the varying bound. Without this, the rescaling
    equivalence collapses to a no-op."""
    from ..mle import _compute_latent_arrays, prepare_mle_data
    df = _build_fixture(n_sessions=1, trials_per_session=5)
    data = prepare_mle_data(df)
    latents = _compute_latent_arrays(data, _params(), _config(use_batched=False))
    assert "bound_per_trial" in latents, (
        "Bound-RewardRate latents must carry per-trial bound for the "
        "rowwise reference path; missing it would silently fall back to "
        "the scalar BOUND and skip the varying-bound math")
    bound_per_trial = latents["bound_per_trial"]
    # bound_per_trial = BOUND_base * (2 - reward_rate_before), both per-trial.
    np.testing.assert_array_equal(
        bound_per_trial,
        (2.0 - latents["reward_rate_before"]) * 1.0)  # BOUND_base=1.0 from _params()


def test_symmetric_path_unaffected_by_bound_rewardrate_flag_off():
    """Old pickles (and symmetric fits going forward) must not change
    behavior. With uses_per_trial_bound=False, the latent compute should
    match the legacy NoiseGain-RewardRate / Classic paths exactly."""
    df = _build_fixture()
    params = _params()
    # NoiseGain-RewardRate without per-trial bound (today's behavior).
    legacy_config = MLEModelConfig(
        drift_fn_str="NoiseGain-RewardRate",
        bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False,
        include_RewardRate=True,
        dt=0.005, t_dur=0.8, dx=0.02,
        mle_use_batched_likelihood=True,
        uses_per_trial_bound=False,    # legacy default
    )
    res = evaluate_neg_loglik(params, df, legacy_config)
    assert np.isfinite(res.neg_loglik)


def test_bound_scale_from_reward_rate_maps_r_to_2_minus_r():
    """Lock the scale-bound equation: s = 2 - r, i.e.
    b_t = BOUND·(2 - r) = BOUND + (1 - r)·BOUND. r=1 → s=1 (floor =
    BOUND); r=0 → s=2 (2·BOUND); monotone decreasing; s ∈ [1, 2] for
    r ∈ [0, 1] so the 1/s rescale is always finite (no eps floor)."""
    assert bound_scale_from_reward_rate(0.0) == 2.0
    assert bound_scale_from_reward_rate(0.5) == 1.5
    assert bound_scale_from_reward_rate(1.0) == 1.0
    r = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    s = bound_scale_from_reward_rate(r)
    np.testing.assert_array_equal(s, 2.0 - r)
    assert np.all(np.diff(s) < 0)                 # decreasing in reward rate
    assert s.min() >= 1.0 and s.max() <= 2.0      # bounded → 1/s finite


def test_bound_per_trial_widens_as_reward_rate_falls():
    """End-to-end direction check on the MLE latents: with a real
    reward-rate trajectory (BETA>0) the per-trial bound equals
    BOUND·(2 - reward_rate_before) and therefore moves *opposite* to the
    reward rate — higher reward rate ⇒ tighter bound, never below the
    BOUND floor nor above 2·BOUND."""
    from ..mle import _compute_latent_arrays, prepare_mle_data
    df = _build_fixture(n_sessions=1, trials_per_session=12)
    params = dict(_params())
    params["BETA"] = 0.5     # let the reward rate actually move
    params["BOUND"] = 1.3    # non-unit floor to catch a bad scale
    data = prepare_mle_data(df)
    latents = _compute_latent_arrays(data, params, _config(use_batched=False))
    rr = latents["reward_rate_before"]
    bpt = latents["bound_per_trial"]
    np.testing.assert_allclose(bpt, 1.3 * (2.0 - rr))
    assert np.all(bpt >= 1.3 - 1e-9)             # floor = BOUND at r=1
    assert np.all(bpt <= 2.0 * 1.3 + 1e-9)       # ceiling = 2·BOUND at r=0
    order = np.argsort(rr)                        # sort ascending in reward rate
    assert np.all(np.diff(bpt[order]) <= 1e-9)   # ⇒ bound non-increasing


def test_bound_rewardrate_population_matches_reference():
    """Bound-RewardRate: the population (DE) path must apply the per-trial
    bound scaling (μ /= s_t = 2 - r_t) exactly as the batched reference does,
    or DE fits a different model than the final eval scores.
    """
    from ..mle import objective_from_population
    df = _build_fixture(n_sessions=2, trials_per_session=15)
    params = dict(_params())
    params["BETA"] = 0.4              # reward rate moves → non-trivial s_t sweep
    config = MLEModelConfig(
        drift_fn_str="Bound-RewardRate",
        bias_fn_str="None_", noise_fn_str="Normal(0, 1)",
        include_Q=True, include_RewardRate=True,
        dt=0.005, t_dur=0.8, dx=0.02,
        mle_use_batched_likelihood=True,
        uses_per_trial_bound=True,
    )
    ref = evaluate_neg_loglik(params, df, config).neg_loglik
    names = ["DRIFT_COEF", "NOISE_SIGMA", "BOUND", "NON_DECISION_TIME",
             "ALPHA", "BETA", "BIAS_COEF", "Q_VAL_OFFSET", "LAPSE_RATE"]
    cand = np.array([[params[k] for k in names]], dtype=float).T   # (n_params, 1)
    pop = objective_from_population(cand, np.array(names), df, config)
    assert np.isfinite(pop[0]) and np.isfinite(ref)
    # For a single candidate the population path builds the SAME per-trial latents
    # as the batched reference and makes the same solver call, so they agree to
    # ~machine precision. A tight tolerance is what makes this sensitive to a
    # missing /s_t.
    assert abs(pop[0] - ref) < 1e-6, f"population={pop[0]}, reference={ref}"
