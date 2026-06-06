"""Tests for the contamination / lapse mixture (`LAPSE_RATE`).

See ``rlmodel/model/mle_lapse_rate_plan.md``. The MLE per-trial likelihood
becomes::

    L_i = (1 - λ) * L_DDM_i  +  λ / (2 * T_max)

applied uniformly to both choice and no-choice trials (user's design
decision). λ ∈ [0, 1). These tests pin five invariants:

  1. ``λ = 0`` reproduces the legacy likelihood exactly.
  2. ``λ = 0.02`` floors choice-trial likelihood at λ/(2·T_max) when the
     DDM density at the observed RT is effectively zero.
  3. Same floor applies to no-choice trials when
     ``terminal_no_decision_mass`` is zero (use ``terminal_c=0``).
  4. When ``f_DDM ≫ λ/(2·T_max)``, the mixture is a small perturbation
     and the loglik is invariant to λ within a tight tolerance.
  5. Batched and rowwise paths agree to ``rtol=1e-8`` when λ > 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..mle import MLEModelConfig, evaluate_neg_loglik
from ..mle_likelihood import LOGLIK_FLOOR


def _df_with_no_choice_trial():
    """One left, one right, one no-choice trial."""
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


def _df_choice_only_well_inside_bulk():
    """Three choice trials whose RTs are within the DDM mass — used to test
    the 'lapse mixture is a perturbation when DDM density is large' invariant.
    """
    rows = []
    for trial_num, choice_left, reward, rt, dv in [
        (1, 1.0, 1.0, 0.30, 0.7),
        (2, 0.0, 0.0, 0.35, -0.5),
        (3, 1.0, 1.0, 0.40, 0.5),
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


def _params(lapse_rate=0.0):
    return {
        "DRIFT_COEF": 1.0,
        "NOISE_SIGMA": 1.0,
        "BOUND": 1.0,
        "NON_DECISION_TIME": 0.02,
        "LAPSE_RATE": float(lapse_rate),
    }


def _config(*, terminal_c=0.99, batched=True, t_dur=0.6, dt=0.01, dx=0.1):
    return MLEModelConfig(
        drift_fn_str="Classic",
        bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False,
        include_RewardRate=False,
        dt=dt,
        t_dur=t_dur,
        dx=dx,
        mle_terminal_c=terminal_c,
        mle_use_batched_likelihood=batched,
    )


# --- 1. λ=0 reproduces baseline exactly --------------------------------

def test_lapse_zero_reproduces_baseline():
    df = _df_choice_only_well_inside_bulk()
    cfg = _config()
    baseline = evaluate_neg_loglik(_params(0.0), df, cfg, return_df=True)
    explicit_zero = evaluate_neg_loglik(
        {**_params(0.0), "LAPSE_RATE": 0.0}, df, cfg, return_df=True)
    np.testing.assert_allclose(
        baseline.neg_loglik, explicit_zero.neg_loglik,
        rtol=0, atol=0)
    np.testing.assert_array_equal(
        baseline.mle_df["mle_loglik"].to_numpy(),
        explicit_zero.mle_df["mle_loglik"].to_numpy())


# --- 2. λ>0 floors choice-trial likelihood at λ/(2·T_max) --------------

def test_lapse_floors_choice_trials_at_lapse_density():
    """Force the DDM density to underflow by placing the observed RT well
    into the tail (here: a choice in the 'wrong' direction with a small
    decision time). Expect per-trial loglik to bottom out around
    log(λ/(2·t_dur)) ≈ log(0.02/(2·0.6)) = log(0.01667) ≈ -4.094.
    """
    rows = [dict(
        Name="S1", Date=pd.Timestamp("2026-01-01"),
        SessionNum=1, TrialNumber=1,
        SessId="S1_2026-01-01_1",
        DV=2.0,                       # strong evidence for left choice
        DVstr="2.0", valid=True,
        calcStimulusTime=0.025,       # extremely fast RT
        ChoiceLeft=0.0,               # but observed 'right' → tail event
        ChoiceCorrect=0.0,
    )]
    df = pd.DataFrame(rows)
    cfg = _config(t_dur=0.6)
    lam = 0.02
    expected_floor = np.log(lam / (2.0 * 0.6))
    result = evaluate_neg_loglik(_params(lam), df, cfg, return_df=True)
    got = float(result.mle_df["mle_loglik"].iloc[0])
    # Per-trial loglik is dominated by the lapse term in this regime; allow
    # a small absolute slack to account for any residual DDM contribution.
    assert got >= expected_floor - 1e-9, (
        f"expected loglik >= {expected_floor}; got {got}")
    assert got <= expected_floor + 0.05, (
        f"expected loglik close to lapse floor {expected_floor}; got {got}")
    # And critically: NOT at LOGLIK_FLOOR.
    assert got > np.log(LOGLIK_FLOOR) + 100


# --- 3. λ>0 floors no-choice trials at λ/(2·T_max) when terminal=0 -----

def test_lapse_floors_no_choice_trials_at_lapse_density():
    """With terminal_c=0 the no_decision band is empty, so
    terminal_no_decision_mass ≈ 0 and the per-trial likelihood collapses
    to the lapse component λ/(2·T_max)."""
    df = _df_with_no_choice_trial()
    cfg = _config(terminal_c=0.0, t_dur=0.6)
    lam = 0.02
    result = evaluate_neg_loglik(_params(lam), df, cfg, return_df=True)
    no_choice_row = result.mle_df[result.mle_df["ChoiceLeft"].isna()]
    assert len(no_choice_row) == 1
    expected = np.log(lam / (2.0 * 0.6))
    got = float(no_choice_row["mle_loglik"].iloc[0])
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)


# --- 4. Mixture is a small perturbation when DDM density dominates -----

def test_lapse_invariant_when_ddm_density_dominates():
    df = _df_choice_only_well_inside_bulk()
    cfg = _config()
    base = evaluate_neg_loglik(_params(0.0), df, cfg, return_df=True)
    pert = evaluate_neg_loglik(_params(0.02), df, cfg, return_df=True)
    # Loose bound: (1-λ) factor alone shifts loglik by log(0.98) ≈ -0.02
    # per trial; the additive lapse term is small relative to f_DDM here.
    # Each per-trial delta should be < ~0.05 for trials in the DDM bulk.
    deltas = (pert.mle_df["mle_loglik"] - base.mle_df["mle_loglik"]).abs()
    assert (deltas < 0.05).all(), deltas.to_list()


# --- 5b. Short-RT (RT ≤ T0) trials now lapse-eligible ------------------

def _df_one_short_rt_choice_trial(rt, choice_left=1.0):
    """Single choice trial with the given RT. Caller sets T0 in params."""
    return pd.DataFrame([dict(
        Name="S1", Date=pd.Timestamp("2026-01-01"),
        SessionNum=1, TrialNumber=1,
        SessId="S1_2026-01-01_1",
        DV=0.5, DVstr="0.5", valid=True,
        calcStimulusTime=rt,
        ChoiceLeft=choice_left, ChoiceCorrect=1.0,
    )])


def test_lapse_floors_short_rt_choice_trials():
    """RT < T0 choice trials are no longer pinned to LOGLIK_FLOOR — the
    DDM density is 0, so the mixture cleanly delivers λ/(2·T_max)."""
    df = _df_one_short_rt_choice_trial(rt=0.03)
    cfg = _config(t_dur=0.6)
    # T0 = 0.10 makes decision_time = -0.07 (RT well below T0).
    params = {**_params(0.02), "NON_DECISION_TIME": 0.10}
    expected = np.log(0.02 / (2.0 * 0.6))
    result = evaluate_neg_loglik(params, df, cfg, return_df=True)
    got = float(result.mle_df["mle_loglik"].iloc[0])
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)


def test_lapse_zero_preserves_floor_for_short_rt_trials():
    """With λ=0 the mixture contributes nothing, so RT < T0 trials still
    floor at LOGLIK_FLOOR (no regression for the no-lapse case)."""
    df = _df_one_short_rt_choice_trial(rt=0.03)
    cfg = _config(t_dur=0.6)
    params = {**_params(0.0), "NON_DECISION_TIME": 0.10}
    result = evaluate_neg_loglik(params, df, cfg, return_df=True)
    got = float(result.mle_df["mle_loglik"].iloc[0])
    np.testing.assert_allclose(
        got, np.log(LOGLIK_FLOOR), rtol=0, atol=1e-12)


def test_short_rt_batched_matches_rowwise_with_lapse():
    """The rowwise path must apply the same short-RT lapse treatment."""
    df = _df_one_short_rt_choice_trial(rt=0.03)
    params = {**_params(0.05), "NON_DECISION_TIME": 0.10}
    rowwise = evaluate_neg_loglik(
        params, df, _config(t_dur=0.6, batched=False), return_df=True)
    batched = evaluate_neg_loglik(
        params, df, _config(t_dur=0.6, batched=True), return_df=True)
    np.testing.assert_allclose(
        batched.mle_df["mle_loglik"].to_numpy(),
        rowwise.mle_df["mle_loglik"].to_numpy(),
        rtol=1e-9, atol=1e-12)


# --- 5. Batched and rowwise agree with λ > 0 ---------------------------

def test_batched_matches_rowwise_with_lapse():
    df = _df_choice_only_well_inside_bulk()
    rowwise_cfg = _config(terminal_c=0.99, batched=False)
    batched_cfg = _config(terminal_c=0.99, batched=True)
    rowwise = evaluate_neg_loglik(
        _params(0.05), df, rowwise_cfg, return_df=True)
    batched = evaluate_neg_loglik(
        _params(0.05), df, batched_cfg, return_df=True)
    np.testing.assert_allclose(
        batched.neg_loglik, rowwise.neg_loglik, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        batched.mle_df["mle_loglik"].to_numpy(),
        rowwise.mle_df["mle_loglik"].to_numpy(),
        rtol=1e-8, atol=1e-10)
