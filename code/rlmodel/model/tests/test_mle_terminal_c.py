"""Tests for the terminal-C residual-mass redistribution feature.

See ``rlmodel/model/mle_terminal_c_plan.md``. The batched MLE path partitions
the residual interior mass at ``t = t_dur`` into three buckets based on a
threshold ``C * BOUND``:

  x >  C * B   → terminal_upper_mass
  x < -C * B   → terminal_lower_mass
  |x| <= C * B → terminal_no_decision_mass    (used as no-choice likelihood)

These tests pin three properties:

  1. ``C = 0`` makes the no-decision bucket empty (bin centers fall on
     ``±dx/2, ±3dx/2, ...`` so none satisfy ``x == 0``), which floors
     no-choice trial likelihood to LOGLIK_FLOOR.
  2. ``C → 1`` reproduces the old survival-only behavior.
  3. Terminal masses partition the survival mass (conservation).
  4. Choice trials are unaffected by ``C``.
  5. Validation rejects out-of-range ``C``.
"""
import numpy as np
import pandas as pd
import pytest

from ..mle import MLEModelConfig, evaluate_neg_loglik, validate_mle_config
from ..mle_batch import BatchedDiffusionSolver
from ..mle_likelihood import LOGLIK_FLOOR


def _df_with_no_choice_trial():
    """One left, one right, one no-choice (ChoiceLeft=NaN, valid=True)."""
    rows = []
    for trial_num, choice_left, reward, rt, dv in [
        (1, 1.0, 1.0, 0.12, 0.7),
        (2, 0.0, 0.0, 0.14, -0.5),
        (3, np.nan, np.nan, np.nan, 0.2),
    ]:
        rows.append(dict(
            Name="S1",
            Date=pd.Timestamp("2026-01-01"),
            SessionNum=1,
            TrialNumber=trial_num,
            SessId="S1_2026-01-01_1",
            DV=dv,
            DVstr=str(dv),
            valid=True,
            calcStimulusTime=rt,
            ChoiceLeft=choice_left,
            ChoiceCorrect=reward,
        ))
    return pd.DataFrame(rows)


def _df_choice_only():
    """All trials are made + observed choices; no no-choice rows."""
    rows = []
    for trial_num, choice_left, reward, rt, dv in [
        (1, 1.0, 1.0, 0.12, 0.7),
        (2, 0.0, 0.0, 0.14, -0.5),
        (3, 1.0, 0.0, 0.18, 0.2),
    ]:
        rows.append(dict(
            Name="S1",
            Date=pd.Timestamp("2026-01-01"),
            SessionNum=1,
            TrialNumber=trial_num,
            SessId="S1_2026-01-01_1",
            DV=dv,
            DVstr=str(dv),
            valid=True,
            calcStimulusTime=rt,
            ChoiceLeft=choice_left,
            ChoiceCorrect=reward,
        ))
    return pd.DataFrame(rows)


def _config(terminal_c, dx=0.1):
    """Classic / None_ / Normal — no Q, no RR. Tiny grid for speed."""
    return MLEModelConfig(
        drift_fn_str="Classic",
        bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False,
        include_RewardRate=False,
        dt=0.01,
        t_dur=0.2,
        dx=dx,
        mle_terminal_c=terminal_c,
    )


def _params():
    return {
        "DRIFT_COEF": 1.0,
        "NOISE_SIGMA": 1.0,
        "BOUND": 1.0,
        "NON_DECISION_TIME": 0.02,
    }


# --- Validation ---------------------------------------------------------

def test_validate_mle_config_rejects_terminal_c_one_or_more():
    config = _config(terminal_c=1.0)
    with pytest.raises(ValueError, match="mle_terminal_c"):
        validate_mle_config(config)


def test_validate_mle_config_rejects_terminal_c_negative():
    config = _config(terminal_c=-0.01)
    with pytest.raises(ValueError, match="mle_terminal_c"):
        validate_mle_config(config)


def test_validate_mle_config_accepts_terminal_c_zero_and_close_to_one():
    validate_mle_config(_config(terminal_c=0.0))
    validate_mle_config(_config(terminal_c=0.999))


# --- C = 0: no-choice likelihood should be LOGLIK_FLOOR ----------------

def test_c_zero_collapses_no_choice_likelihood_to_floor():
    """With C=0, the no-decision band is `|x| <= 0` — empty on a
    bin-centered grid (centers at ±dx/2, ±3dx/2). The no-choice trial
    therefore gets terminal_no_decision_mass == 0, which the solver clamps
    to LOGLIK_FLOOR.
    """
    df = _df_with_no_choice_trial()
    result = evaluate_neg_loglik(
        _params(), df, _config(terminal_c=0.0), return_df=True)

    mle_df = result.mle_df
    no_choice_row = mle_df[mle_df["ChoiceLeft"].isna()]
    assert len(no_choice_row) == 1
    expected_floor_loglik = np.log(LOGLIK_FLOOR)
    np.testing.assert_allclose(
        float(no_choice_row["mle_loglik"].iloc[0]),
        expected_floor_loglik,
        rtol=0, atol=1e-12,
    )


# --- C close to 1: reproduces legacy survival behavior -----------------

def test_c_close_to_one_reproduces_full_survival():
    """When C is large enough that every bin center falls within `|x| <= C*B`,
    terminal_no_decision_mass equals the full survival mass, and no-choice
    log-likelihood matches log(survival_at_tmax) — the pre-feature behavior.
    """
    df = _df_with_no_choice_trial()
    # dx=0.1, bound=1 → bin centers at ±0.95, ±0.85, ..., ±0.05.
    # C=0.99 makes the threshold 0.99, so 0.95 < 0.99 — every bin counts.
    config = _config(terminal_c=0.99)
    result = evaluate_neg_loglik(_params(), df, config, return_df=True)
    mle_df = result.mle_df
    no_choice_row = mle_df[mle_df["ChoiceLeft"].isna()]
    survival = float(no_choice_row["mle_survival_at_tmax"].iloc[0])
    assert survival > 0, "test setup expects positive survival mass"
    np.testing.assert_allclose(
        float(no_choice_row["mle_loglik"].iloc[0]),
        np.log(survival),
        rtol=1e-12, atol=1e-12,
    )


# --- Mass conservation -------------------------------------------------

def test_terminal_masses_partition_survival():
    """terminal_upper + terminal_lower + terminal_no_decision == survival
    inside the solver, for any C in [0, 1)."""
    from scipy.special import ndtr

    solver = BatchedDiffusionSolver(
        xp=np,
        normal_cdf=lambda v: ndtr(np.asarray(v)),
    )
    bound = 1.0
    dt = 0.01
    dx = 0.05
    tmax = 0.3
    n_trials = 4
    z = np.array([0.0, 0.2, -0.1, 0.0])
    mu = np.array([0.5, -0.3, 0.0, 1.2])
    sigma = np.array([1.0, 1.0, 1.0, 1.0])
    valid = np.array([True, True, True, True])

    for c in [0.0, 0.25, 0.5, 0.99]:
        out = solver.solve(z, mu, sigma, valid, bound, dt, dx, tmax,
                           terminal_c=c)
        survival = np.asarray(out.survival_xp)
        upper = np.asarray(out.terminal_upper_mass_xp)
        lower = np.asarray(out.terminal_lower_mass_xp)
        no_dec = np.asarray(out.terminal_no_decision_mass_xp)
        np.testing.assert_allclose(
            upper + lower + no_dec, survival,
            rtol=1e-12, atol=1e-12,
            err_msg=f"mass partition failed at C={c}",
        )


# --- Choice trials are unaffected by C ---------------------------------

def test_choice_trial_loglik_invariant_under_c():
    """The choice-trial likelihood is a first-passage density at the
    observed decision time, not a function of residual mass at t_dur. It
    must not depend on terminal_c."""
    df = _df_choice_only()
    out_c0 = evaluate_neg_loglik(
        _params(), df, _config(terminal_c=0.0), return_df=True)
    out_c5 = evaluate_neg_loglik(
        _params(), df, _config(terminal_c=0.5), return_df=True)
    np.testing.assert_allclose(
        out_c0.mle_df["mle_loglik"].to_numpy(),
        out_c5.mle_df["mle_loglik"].to_numpy(),
        rtol=1e-12, atol=1e-12,
    )
    np.testing.assert_allclose(
        out_c0.neg_loglik, out_c5.neg_loglik, rtol=1e-12, atol=1e-12)


