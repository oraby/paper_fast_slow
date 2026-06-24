"""Tests for the weighted choice-vs-RT MLE loss
(``--mle-choice-weight`` / ``--mle-rt-weight`` / ``--mle-choice-norm``).

The per-trial joint loglik is split into a choice component and an
RT-given-choice component and recombined as
``w_choice·log P(c) + w_rt·log p(rt|c)`` (chat Answer 29). Coverage:

- the explicit two-piece decomposition math in ``_apply_choice_rt_weights``
  (marginal vs conditional norm, lapse-exact);
- the ``marginal`` + ``(1, 1)`` bit-exact legacy fast path;
- rowwise/batched parity and composition with ``--mle-conditions``;
- the back-compat default split (dataclass ``marginal``; CLI/simulateDDM
  ``conditional``);
- filename invariance and validation.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from ..fit import evolveFP, simulateDDM
from ..mle import (MLEModelConfig, LOGLIK_FLOOR, evaluate_neg_loglik,
                   validate_mle_config, _apply_choice_rt_weights)
from ...model_runner import runModel


def _choice_rt_df():
    """8 valid trials, both choice directions, a couple no-choice
    (invalid) padding rows. Enough survival mass for the marginal and
    conditional norms to differ."""
    rows = []
    spec = [
        # trial, choice_left, reward, rt,   dv,   valid
        (1, 1.0, 1.0, 0.18,  0.7,  True),
        (2, 1.0, 1.0, 0.22,  0.5,  True),
        (3, 0.0, 0.0, 0.26, -0.6,  True),
        (4, 0.0, 1.0, 0.30, -0.4,  True),
        (5, 1.0, 0.0, 0.20,  0.3,  True),
        (6, 0.0, 0.0, 0.24, -0.5,  True),
        (7, 1.0, 1.0, 0.28,  0.6,  True),
        (8, 0.0, 1.0, 0.32, -0.7,  True),
        (9, np.nan, np.nan, np.nan, 0.0, False),  # no-choice / padding
    ]
    for trial_num, choice_left, reward, rt, dv, valid in spec:
        rows.append(dict(
            Name="S1", Date=pd.Timestamp("2026-01-01"),
            SessionNum=1, TrialNumber=trial_num,
            SessId="S1_2026-01-01_1",
            DV=dv, DVstr=str(dv), valid=valid,
            calcStimulusTime=rt, ChoiceLeft=choice_left,
            ChoiceCorrect=reward,
        ))
    return pd.DataFrame(rows)


_PARAMS = {
    "DRIFT_COEF": 1.0, "NOISE_SIGMA": 1.5, "BOUND": 1.0,
    "NON_DECISION_TIME": 0.04,
    "BIAS_COEF": 0.0, "Q_VAL_OFFSET": 0.0,
    "LAPSE_RATE": 0.0,
}


def _config(**overrides):
    base = dict(
        drift_fn_str="Classic", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False, include_RewardRate=False,
        dt=0.005, t_dur=0.5, dx=0.02,
    )
    base.update(overrides)
    return MLEModelConfig(**base)


# ---------------------------------------------------------------------------
# Helper math — the explicit two-piece decomposition
# ---------------------------------------------------------------------------

def test_helper_fast_path_returns_input_identity_marginal_unit_weights():
    """marginal + (1, 1) must return the SAME array object — guarantees a
    no-flag / explicit-(1,1)-marginal run is bit-exact with the legacy
    joint loss (no FP drift from a redundant recombination)."""
    loglik = np.array([-1.0, -2.0, -0.5])
    upper = np.array([0.4, 0.2, 0.3])
    lower = np.array([0.3, 0.5, 0.4])
    choice_left = np.array([1.0, 0.0, 1.0])
    no_choice = np.array([False, False, False])
    out = _apply_choice_rt_weights(
        loglik, upper, lower, choice_left, no_choice,
        0.0, 1.0, 1.0, "marginal")
    assert out is loglik


def test_helper_explicit_decomposition_marginal_lambda0():
    """Assert against the two-piece form (not the collapsed one):
    ``w_choice·log P_mix(c) + w_rt·(loglik − log P_mix(c))`` with the
    choice probability computed independently."""
    loglik = np.array([-1.0, -2.0])
    upper = np.array([0.4, 0.2])
    lower = np.array([0.3, 0.5])
    choice_left = np.array([1.0, 0.0])      # trial0→upper, trial1→lower
    no_choice = np.array([False, False])
    w_choice, w_rt = 2.0, 0.5

    out = _apply_choice_rt_weights(
        loglik, upper, lower, choice_left, no_choice,
        0.0, w_choice, w_rt, "marginal")

    p_ddm = np.array([0.4, 0.5])            # chosen side's hit prob
    log_choice = np.log(p_ddm)              # λ=0 → P_mix == P_DDM
    log_rt = loglik - log_choice
    expected = w_choice * log_choice + w_rt * log_rt
    np.testing.assert_allclose(out, expected)


def test_helper_conditional_subtracts_log_normalizer():
    """conditional == marginal − w_choice·log(P_mix(L)+P_mix(R)); at
    (1, 1), λ=0 it equals ``loglik − log(P_L+P_R)``."""
    loglik = np.array([-1.0, -2.0])
    upper = np.array([0.4, 0.2])
    lower = np.array([0.3, 0.5])
    choice_left = np.array([1.0, 0.0])
    no_choice = np.array([False, False])
    w_choice, w_rt = 2.0, 0.5

    marg = _apply_choice_rt_weights(
        loglik, upper, lower, choice_left, no_choice,
        0.0, w_choice, w_rt, "marginal")
    cond = _apply_choice_rt_weights(
        loglik, upper, lower, choice_left, no_choice,
        0.0, w_choice, w_rt, "conditional")

    log_norm = np.log(upper + lower)        # λ=0 → P_mix_total == P_L+P_R
    np.testing.assert_allclose(cond - marg, -w_choice * log_norm)

    # (1, 1) special case: conditional joint == loglik − log(P_L+P_R).
    cond11 = _apply_choice_rt_weights(
        loglik, upper, lower, choice_left, no_choice,
        0.0, 1.0, 1.0, "conditional")
    np.testing.assert_allclose(cond11, loglik - np.log(upper + lower))


def test_helper_lapse_exact_choice_masses():
    """λ>0 must use the lapse-consistent masses
    ``P_mix(c)=(1−λ)P_DDM(c)+λ/2`` and
    ``P_mix_total=(1−λ)(P_L+P_R)+λ`` in BOTH components."""
    loglik = np.array([-1.0, -2.0])
    upper = np.array([0.4, 0.2])
    lower = np.array([0.3, 0.5])
    choice_left = np.array([1.0, 0.0])
    no_choice = np.array([False, False])
    lam = 0.2
    w_choice, w_rt = 1.5, 1.0

    out = _apply_choice_rt_weights(
        loglik, upper, lower, choice_left, no_choice,
        lam, w_choice, w_rt, "conditional")

    p_ddm = np.array([0.4, 0.5])
    p_mix = (1 - lam) * p_ddm + lam / 2.0
    p_mix_total = (1 - lam) * (upper + lower) + lam
    log_choice = np.log(p_mix)
    log_rt = loglik - log_choice
    expected = (w_choice * log_choice + w_rt * log_rt
                - w_choice * np.log(p_mix_total))
    np.testing.assert_allclose(out, expected)


def test_helper_no_choice_trials_get_choice_only_weight():
    """No-decision trials (no RT to condition on) get ``w_choice·loglik``
    — no RT term, independent of norm."""
    loglik = np.array([-1.0, -3.0])
    upper = np.array([0.4, np.nan])         # no-choice trial has no hit prob
    lower = np.array([0.3, np.nan])
    choice_left = np.array([1.0, np.nan])
    no_choice = np.array([False, True])
    w_choice, w_rt = 2.0, 0.5

    for norm in ("marginal", "conditional"):
        out = _apply_choice_rt_weights(
            loglik, upper, lower, choice_left, no_choice,
            0.0, w_choice, w_rt, norm)
        assert out[1] == pytest.approx(w_choice * loglik[1])
        assert np.isfinite(out[0])


def test_helper_clips_zero_choice_prob_to_floor():
    """A degenerate P_mix(c)=0 (λ=0, no hit mass) must not produce
    ``-inf``/``nan`` — it's clipped to ``log(FLOOR)``."""
    loglik = np.array([np.log(LOGLIK_FLOOR)])
    upper = np.array([0.0])
    lower = np.array([0.0])
    choice_left = np.array([1.0])
    no_choice = np.array([False])
    out = _apply_choice_rt_weights(
        loglik, upper, lower, choice_left, no_choice,
        0.0, 2.0, 1.0, "marginal")
    assert np.isfinite(out[0])


# ---------------------------------------------------------------------------
# Integration through evaluate_neg_loglik
# ---------------------------------------------------------------------------

def test_marginal_unit_weights_bit_exact_with_default_config():
    """The dataclass default (marginal, 1, 1) reproduces today's loss;
    spelling the defaults out explicitly must give the IDENTICAL number."""
    df = _choice_rt_df()
    default = evaluate_neg_loglik(_PARAMS, df, _config()).neg_loglik
    explicit = evaluate_neg_loglik(
        _PARAMS, df,
        _config(mle_choice_norm="marginal",
                mle_choice_weight=1.0, mle_rt_weight=1.0)).neg_loglik
    assert default == explicit
    assert np.isfinite(default)


def test_conditional_default_differs_from_marginal_when_survival_present():
    """conditional ≠ marginal at (1, 1) whenever there's survival mass —
    the whole point of the norm switch (chat Answer 30's leakage)."""
    df = _choice_rt_df()
    marg = evaluate_neg_loglik(
        _PARAMS, df, _config(mle_choice_norm="marginal")).neg_loglik
    cond = evaluate_neg_loglik(
        _PARAMS, df, _config(mle_choice_norm="conditional")).neg_loglik
    assert np.isfinite(marg) and np.isfinite(cond)
    assert abs(marg - cond) > 1e-6


def test_choice_weight_changes_loss():
    """Bumping the choice weight must move the loss (otherwise the flag
    is inert)."""
    df = _choice_rt_df()
    base = evaluate_neg_loglik(
        _PARAMS, df, _config(mle_choice_norm="marginal")).neg_loglik
    up = evaluate_neg_loglik(
        _PARAMS, df,
        _config(mle_choice_norm="marginal", mle_choice_weight=5.0)).neg_loglik
    assert abs(base - up) > 1e-6


@pytest.mark.parametrize("norm", ["marginal", "conditional"])
def test_rowwise_matches_batched_under_weighted_loss(norm):
    """Both compute paths apply the same split; toggling the evaluator
    must not change the weighted neg_loglik beyond solver discretization."""
    df = _choice_rt_df()
    common = dict(mle_choice_norm=norm, mle_choice_weight=3.0,
                  mle_rt_weight=1.0)
    batched = evaluate_neg_loglik(
        _PARAMS, df, _config(mle_use_batched_likelihood=True, **common)
    ).neg_loglik
    rowwise = evaluate_neg_loglik(
        _PARAMS, df, _config(mle_use_batched_likelihood=False, **common)
    ).neg_loglik
    assert np.isfinite(batched) and np.isfinite(rowwise)
    assert abs(batched - rowwise) < 1.0


def test_composes_with_mle_conditions():
    """Choice/RT weighting and condition balancing stack: the result is
    finite and differs from the conditions-only loss."""
    df = _choice_rt_df()
    conditions_only = evaluate_neg_loglik(
        _PARAMS, df,
        _config(mle_condition_columns=("ChoiceCorrect", "ChoiceLeft"))
    ).neg_loglik
    both = evaluate_neg_loglik(
        _PARAMS, df,
        _config(mle_condition_columns=("ChoiceCorrect", "ChoiceLeft"),
                mle_choice_norm="marginal", mle_choice_weight=4.0)
    ).neg_loglik
    assert np.isfinite(conditions_only) and np.isfinite(both)
    assert abs(conditions_only - both) > 1e-6


def test_mle_df_exposes_choice_rt_components():
    """``return_df=True`` surfaces the per-trial split components for
    auditing."""
    df = _choice_rt_df()
    res = evaluate_neg_loglik(
        _PARAMS, df, _config(mle_choice_norm="conditional"), return_df=True)
    for col in ("mle_log_choice", "mle_rt_loglik",
                "mle_log_choice_normalizer"):
        assert col in res.mle_df.columns


# ---------------------------------------------------------------------------
# Back-compat default split + validation + filename invariance
# ---------------------------------------------------------------------------

def test_dataclass_default_is_marginal_but_callboundary_is_conditional():
    """Critical back-compat contract: the MLEModelConfig dataclass
    default stays ``marginal`` (so old pickles + direct-construct tests
    keep the legacy objective), while the user-facing call boundary
    (runModel / simulateDDM) defaults to ``conditional``."""
    cfg = MLEModelConfig(
        drift_fn_str="Classic", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)", include_Q=False,
        include_RewardRate=False, dt=0.005, t_dur=0.5)
    assert cfg.mle_choice_norm == "marginal"
    assert cfg.mle_choice_weight == 1.0
    assert cfg.mle_rt_weight == 1.0

    for fn in (runModel, simulateDDM):
        params = inspect.signature(fn).parameters
        assert params["mle_choice_norm"].default == "conditional"
        assert params["mle_choice_weight"].default == 1.0
        assert params["mle_rt_weight"].default == 1.0


def test_validate_rejects_bad_weights_and_norm():
    bad_norm = _config(mle_choice_norm="nope")
    with pytest.raises(ValueError, match="mle_choice_norm"):
        validate_mle_config(bad_norm)
    neg_w = _config(mle_choice_weight=-1.0)
    with pytest.raises(ValueError, match="mle_choice_weight"):
        validate_mle_config(neg_w)
    neg_rt = _config(mle_rt_weight=-0.5)
    with pytest.raises(ValueError, match="mle_rt_weight"):
        validate_mle_config(neg_rt)


def test_evolveFP_filename_unchanged_by_choice_rt_weights():
    """The weights/norm are NOT in the filename (A/B by overwriting), so
    ``evolveFP`` takes no such argument and its output is unchanged."""
    base = dict(
        drift_fn_str="Classic", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)", t_dur=3, dt=0.005,
        is_loss_no_dir=False, fit_mode="mle")
    fp = str(evolveFP(**base))
    assert fp.endswith("_3s_dt0.005.pkl")
    # evolveFP's signature must not have grown a weight/norm arg.
    params = inspect.signature(evolveFP).parameters
    assert "mle_choice_weight" not in params
    assert "mle_rt_weight" not in params
    assert "mle_choice_norm" not in params
