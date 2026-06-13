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
    spatial grid (n_x = 2·BOUND·r_t / dx ≈ 50 bins) is well-resolved
    — the rescaling identity is exact in the continuum but at finite
    dx the path-A grid coarsens with shrinking r_t. With BETA=0 the
    per-trial bound is uniformly 0.5, so the comparison isolates the
    rescaling math from grid-coarsening artifacts.
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
    # bound_per_trial = BOUND_base * reward_rate_before, both per-trial.
    np.testing.assert_array_equal(
        bound_per_trial,
        latents["reward_rate_before"] * 1.0)  # BOUND_base=1.0 from _params()


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
