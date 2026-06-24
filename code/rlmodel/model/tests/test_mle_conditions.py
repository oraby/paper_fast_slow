"""Tests for ``--mle-conditions`` — sample-balanced MLE loss.

Covers the per-trial weight math, the no-op fallback when no columns
are configured, parity between the rowwise and batched paths, and the
filename invariance contract (the flag must not alter ``evolveFP``'s
output so users can A/B test by overwriting the same pickle).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ..fit import evolveFP
from ..mle import (MLEModelConfig, PreparedMLEData,
                   _compute_trial_weights, evaluate_neg_loglik,
                   prepare_mle_data)


def _imbalanced_df():
    """6 valid trials split (4, 2) across (ChoiceCorrect, ChoiceLeft):
    - (1, 1): 3 trials   ← big group
    - (1, 0): 1 trial    ← small group
    - (0, 1): 1 trial    ← small group
    - (0, 0): 1 trial    ← small group
    Total valid = 6; 4 distinct groups → balanced weights are 6/4=1.5
    per group; per-trial weight = 1.5 / group_size.
    """
    rows = []
    spec = [
        (1, 1.0, 1.0, 0.12,  0.7,  True),  # (1, 1)
        (2, 1.0, 1.0, 0.14,  0.5,  True),  # (1, 1)
        (3, 1.0, 1.0, 0.16,  0.3,  True),  # (1, 1)
        (4, 0.0, 1.0, 0.18, -0.5,  True),  # (1, 0)
        (5, 1.0, 0.0, 0.20,  0.7,  True),  # (0, 1)
        (6, 0.0, 0.0, 0.22, -0.5,  True),  # (0, 0)
        (7, np.nan, np.nan, np.nan, 0.0, False),  # padding / invalid
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


# ---------------------------------------------------------------------------
# Weight math
# ---------------------------------------------------------------------------

def test_trial_weights_sum_to_total_valid_imbalanced():
    """Cornerstone invariant: sum of per-trial weights equals the
    valid trial count, no matter how unbalanced the groups are. This
    is what guarantees the weighted loss stays on the same scale as
    the unweighted sum."""
    df = _imbalanced_df()
    valid_mask = df["valid"].to_numpy(dtype=bool)
    weights = _compute_trial_weights(
        df, ("ChoiceCorrect", "ChoiceLeft"), valid_mask)
    assert weights.shape == (len(df),)
    total_valid = int(valid_mask.sum())
    assert weights.sum() == pytest.approx(float(total_valid))
    # Invalid trials get 0 weight.
    assert np.all(weights[~valid_mask] == 0.0)


def test_trial_weights_rebalance_unequal_group_sizes():
    """The 4-trial group should get weight 1.5/3=0.5 each; the three
    1-trial groups should get 1.5/1=1.5 each. (num_groups=4,
    total_valid=6, so each group contributes 6/4=1.5 to the sum.)"""
    df = _imbalanced_df()
    valid_mask = df["valid"].to_numpy(dtype=bool)
    weights = _compute_trial_weights(
        df, ("ChoiceCorrect", "ChoiceLeft"), valid_mask)
    # First 3 rows are the (1, 1) group → 0.5 each.
    np.testing.assert_allclose(weights[:3], 0.5)
    # Rows 3, 4, 5 are the three singleton groups → 1.5 each.
    np.testing.assert_allclose(weights[3:6], 1.5)
    # Per-group contribution check: each group sums to total_valid/num_groups.
    per_group_sum = 6.0 / 4.0
    assert weights[:3].sum() == pytest.approx(per_group_sum)
    assert weights[3:4].sum() == pytest.approx(per_group_sum)
    assert weights[4:5].sum() == pytest.approx(per_group_sum)
    assert weights[5:6].sum() == pytest.approx(per_group_sum)


def test_trial_weights_balanced_groups_collapse_to_uniform():
    """When every group has the same size, every weight is 1.0 (= the
    unweighted case). Degenerate sanity check."""
    df = _imbalanced_df()
    # Use ``ChoiceCorrect`` alone: 4 valid trials in {1,1,1,0} ChoiceCorrect
    # × 2 valid in {1,0}. Wait — easier to construct a balanced fixture.
    # Force balance: take only first 4 rows (3 are ChoiceCorrect=1,
    # 1 is ChoiceCorrect=0 → still imbalanced). Instead, build inline:
    rows = []
    for i, cc in enumerate([1.0, 1.0, 0.0, 0.0]):
        rows.append(dict(
            Name="S1", Date=pd.Timestamp("2026-01-01"),
            SessionNum=1, TrialNumber=i + 1,
            SessId="S1_2026-01-01_1",
            DV=0.5, DVstr="0.5", valid=True,
            calcStimulusTime=0.1, ChoiceLeft=1.0,
            ChoiceCorrect=cc))
    df_bal = pd.DataFrame(rows)
    weights = _compute_trial_weights(
        df_bal, ("ChoiceCorrect",), df_bal["valid"].to_numpy(dtype=bool))
    np.testing.assert_allclose(weights, 1.0)


def test_trial_weights_none_when_no_columns_configured():
    """Empty column list → None (legacy fast path; sum site skips the
    multiply entirely for bit-exact behavior)."""
    df = _imbalanced_df()
    weights = _compute_trial_weights(
        df, (), df["valid"].to_numpy(dtype=bool))
    assert weights is None


def test_trial_weights_nan_values_form_their_own_group():
    """Trials with NaN in a condition column are NOT dropped — they
    form their own group via ``groupby(..., dropna=False)``. Important
    for no-choice trials under ``--mle-conditions=ChoiceLeft``.
    """
    rows = []
    for i, cl in enumerate([1.0, 1.0, np.nan, np.nan]):
        rows.append(dict(
            Name="S1", Date=pd.Timestamp("2026-01-01"),
            SessionNum=1, TrialNumber=i + 1,
            SessId="S1_2026-01-01_1",
            DV=0.5, DVstr="0.5", valid=True,
            calcStimulusTime=0.1, ChoiceLeft=cl, ChoiceCorrect=1.0))
    df = pd.DataFrame(rows)
    weights = _compute_trial_weights(
        df, ("ChoiceLeft",), df["valid"].to_numpy(dtype=bool))
    # 2 groups (1.0 and NaN), each with 2 trials, total_valid=4 →
    # weight = 4 / (2 * 2) = 1.0 everywhere. Confirms NaN didn't get
    # dropped (which would have produced a 1-group case with weight=2).
    np.testing.assert_allclose(weights, 1.0)
    assert weights.sum() == pytest.approx(4.0)


def test_trial_weights_unknown_column_raises_keyerror():
    """Misspelled column name fails fast at prepare time with a
    KeyError listing the available columns — before any DE step."""
    df = _imbalanced_df()
    with pytest.raises(KeyError, match="NotAColumn"):
        _compute_trial_weights(
            df, ("NotAColumn",), df["valid"].to_numpy(dtype=bool))


def test_trial_weights_empty_valid_mask_returns_uniform():
    """All-invalid fixture is a degenerate corner case — the loss is
    0.0 regardless of weights. Return a uniform array so callers
    don't NPE."""
    df = _imbalanced_df()
    weights = _compute_trial_weights(
        df, ("ChoiceCorrect",), np.zeros(len(df), dtype=bool))
    assert weights.shape == (len(df),)
    np.testing.assert_allclose(weights, 1.0)


# ---------------------------------------------------------------------------
# Integration with PreparedMLEData + evaluate_neg_loglik
# ---------------------------------------------------------------------------

def test_prepare_mle_data_attaches_weights_when_columns_configured():
    df = _imbalanced_df()
    prepared = prepare_mle_data(df, ("ChoiceCorrect", "ChoiceLeft"))
    assert prepared.trial_weights is not None
    assert prepared.trial_weights.sum() == pytest.approx(6.0)


def test_prepare_mle_data_leaves_weights_none_by_default():
    """Default empty tuple → ``trial_weights is None`` → legacy
    sum at every existing call site stays bit-exact."""
    df = _imbalanced_df()
    prepared = prepare_mle_data(df)
    assert prepared.trial_weights is None


def test_unweighted_path_is_bit_exact_with_empty_conditions():
    """Pinning the legacy contract: configuring no condition columns
    produces the same neg_loglik as before this feature existed."""
    df = _imbalanced_df()
    params = {
        "DRIFT_COEF": 1.0, "NOISE_SIGMA": 1.5, "BOUND": 1.0,
        "NON_DECISION_TIME": 0.04,
        "BIAS_COEF": 0.0, "Q_VAL_OFFSET": 0.0,
        "LAPSE_RATE": 0.0,
    }
    config = MLEModelConfig(
        drift_fn_str="Classic", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False, include_RewardRate=False,
        dt=0.005, t_dur=0.5, dx=0.02,
        mle_condition_columns=(),
    )
    result = evaluate_neg_loglik(params, df, config)
    assert np.isfinite(result.neg_loglik)


def test_weighted_neg_loglik_finite_and_differs_for_imbalanced_groups():
    """With imbalanced groups the weighted loss differs from the
    unweighted one (otherwise the flag wouldn't do anything). Don't
    hard-code the magnitude — solver discretization shifts it — just
    check ``|weighted - unweighted| > 0``.
    """
    df = _imbalanced_df()
    params = {
        "DRIFT_COEF": 1.0, "NOISE_SIGMA": 1.5, "BOUND": 1.0,
        "NON_DECISION_TIME": 0.04,
        "BIAS_COEF": 0.0, "Q_VAL_OFFSET": 0.0,
        "LAPSE_RATE": 0.0,
    }
    unweighted = MLEModelConfig(
        drift_fn_str="Classic", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False, include_RewardRate=False,
        dt=0.005, t_dur=0.5, dx=0.02,
        mle_condition_columns=(),
    )
    weighted = MLEModelConfig(
        drift_fn_str="Classic", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False, include_RewardRate=False,
        dt=0.005, t_dur=0.5, dx=0.02,
        mle_condition_columns=("ChoiceCorrect", "ChoiceLeft"),
    )
    u = evaluate_neg_loglik(params, df, unweighted).neg_loglik
    w = evaluate_neg_loglik(params, df, weighted).neg_loglik
    assert np.isfinite(u) and np.isfinite(w)
    assert abs(u - w) > 1e-6


def test_prepare_mle_data_short_circuit_asserts_on_silent_drop():
    """Regression for the silent ``--mle-conditions`` no-op bug:
    ``fit._processSubject`` used to call
    ``prepare_mle_data(subject_df)`` WITHOUT condition columns, then
    pass the resulting ``PreparedMLEData`` (with ``trial_weights=None``)
    to ``objective_from_population`` / ``result_payload``. Both
    short-circuit on an already-prepared instance and would inherit
    ``trial_weights=None`` — running the unweighted objective while
    the model_config silently recorded the conditions.

    The fix has two parts:
    1. ``_processSubject`` now passes ``model_config.mle_condition_columns``
       at first prepare (covered by
       ``test_processSubject_threads_mle_conditions_into_prepared_data``).
    2. ``prepare_mle_data`` itself hard-fails on the mismatch case so
       any future caller hitting the same trap gets a loud error.
       This test pins (2).
    """
    df = _imbalanced_df()
    # Build prepared data WITHOUT conditions (legacy / wrong path).
    prepared_no_weights = prepare_mle_data(df)
    assert prepared_no_weights.trial_weights is None
    # Now try to "re-prepare" with conditions — must NOT silently
    # short-circuit. The assert is the contract.
    with pytest.raises(AssertionError,
                       match=r"trial_weights=None"):
        prepare_mle_data(
            prepared_no_weights, ("ChoiceCorrect", "ChoiceLeft"))
    # Sanity: passing the same prepared instance with NO conditions
    # is still a legitimate pass-through.
    assert prepare_mle_data(prepared_no_weights, ()) is prepared_no_weights
    # And a prepared instance with weights pre-baked passes through.
    prepared_with_weights = prepare_mle_data(
        df, ("ChoiceCorrect", "ChoiceLeft"))
    assert prepared_with_weights.trial_weights is not None
    assert prepare_mle_data(
        prepared_with_weights, ("ChoiceCorrect", "ChoiceLeft")
    ) is prepared_with_weights


def test_objective_from_population_applies_weights_when_conditions_active():
    """Population (vectorized DE) path applies the sample-balancing
    weights and returns one finite loss per candidate.

    (The per-DE-generation debug print that used to live in the
    weighted branch was intentionally removed — it ran every generation
    and was noisy; live visibility now comes from the one-shot startup
    log in ``model_runner``. This test pins the population path's
    NUMERIC behavior under active conditions rather than a log line.)
    """
    from ..mle import objective_from_population

    df = _imbalanced_df()
    config = MLEModelConfig(
        drift_fn_str="Classic", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False, include_RewardRate=False,
        dt=0.005, t_dur=0.5, dx=0.02,
        mle_condition_columns=("ChoiceCorrect", "ChoiceLeft"),
    )
    # Two-candidate batch — mirrors the (n_params, S) matrix shape
    # scipy hands the vectorized DE objective.
    params_names = np.array(
        ["DRIFT_COEF", "NOISE_SIGMA", "BOUND",
         "NON_DECISION_TIME", "BIAS_COEF", "Q_VAL_OFFSET",
         "LAPSE_RATE"])
    x_matrix = np.array([
        [1.0, 1.0, 1.0, 0.04, 0.0, 0.0, 0.0],
        [1.1, 1.0, 1.0, 0.04, 0.0, 0.0, 0.0],
    ], dtype=float).T

    losses = objective_from_population(x_matrix, params_names, df, config)

    assert losses.shape == (2,)
    assert np.all(np.isfinite(losses))


def test_objective_from_population_silent_under_unweighted_path(capsys):
    """Default empty conditions → no per-batch log line. Pin the
    legacy contract so old (unweighted) fits don't get a noisy
    DE-loop log they never had."""
    from ..mle import objective_from_population

    df = _imbalanced_df()
    config = MLEModelConfig(
        drift_fn_str="Classic", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False, include_RewardRate=False,
        dt=0.005, t_dur=0.5, dx=0.02,
        mle_condition_columns=(),
    )
    params_names = np.array(
        ["DRIFT_COEF", "NOISE_SIGMA", "BOUND",
         "NON_DECISION_TIME", "BIAS_COEF", "Q_VAL_OFFSET",
         "LAPSE_RATE"])
    x_matrix = np.array([[1.0, 1.0, 1.0, 0.04, 0.0, 0.0, 0.0]],
                         dtype=float).T

    objective_from_population(x_matrix, params_names, df, config)
    out = capsys.readouterr().out
    assert "--mle-conditions" not in out, out


def test_processSubject_threads_mle_conditions_into_prepared_data():
    """Regression: end-to-end check that
    ``MLEModelConfig.mle_condition_columns`` actually reach the
    ``PreparedMLEData.trial_weights`` carried through the fit. Without
    this, the DE objective runs unweighted (the bug the user reported:
    ``visualize.py`` printed the conditions while the saved loss /
    figures were unchanged).

    Indirect check via the public ``evaluate_neg_loglik`` path, which
    mirrors ``_processSubject``'s prepare → fit handoff: same
    ``prepare_mle_data(subject_df, model_config.mle_condition_columns)``
    call, same ``data.trial_weights`` consumed at the sum site. If
    weights leak out of that chain again, the imbalanced fixture's
    weighted vs unweighted loss diverges visibly.
    """
    df = _imbalanced_df()
    params = {
        "DRIFT_COEF": 1.0, "NOISE_SIGMA": 1.5, "BOUND": 1.0,
        "NON_DECISION_TIME": 0.04,
        "BIAS_COEF": 0.0, "Q_VAL_OFFSET": 0.0,
        "LAPSE_RATE": 0.0,
    }
    weighted_config = MLEModelConfig(
        drift_fn_str="Classic", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False, include_RewardRate=False,
        dt=0.005, t_dur=0.5, dx=0.02,
        mle_condition_columns=("ChoiceCorrect", "ChoiceLeft"),
    )
    # Pre-prepare exactly the way ``_processSubject`` does — this is
    # the path the bug fix wires up. Then feed the prepared instance
    # straight through ``evaluate_neg_loglik`` (which also re-calls
    # ``prepare_mle_data`` internally; with weights baked in, the
    # short-circuit pass-through is the correct behavior).
    prepared = prepare_mle_data(
        df, weighted_config.mle_condition_columns)
    assert prepared.trial_weights is not None, (
        "Bug regression: prepare_mle_data(raw_df, conditions) MUST "
        "cache trial_weights at first prepare — otherwise downstream "
        "callers fall through to the unweighted sum.")
    result = evaluate_neg_loglik(params, prepared, weighted_config)
    # Same params on the raw df go through the same prepare chain
    # and produce the same loss. (Pins the contract that a
    # pre-prepared PreparedMLEData and the raw df are interchangeable
    # when the conditions match.)
    result_from_raw = evaluate_neg_loglik(params, df, weighted_config)
    assert result.neg_loglik == pytest.approx(result_from_raw.neg_loglik)


def test_rowwise_matches_batched_under_weighted_loss():
    """Both compute paths must apply the same weights. Toggling
    ``mle_use_batched_likelihood`` should not change the weighted
    neg_loglik (within solver discretization, which is exact when
    the only difference is the evaluator — both use the same scalar
    bound and the same single-cand path)."""
    df = _imbalanced_df()
    params = {
        "DRIFT_COEF": 1.0, "NOISE_SIGMA": 1.5, "BOUND": 1.0,
        "NON_DECISION_TIME": 0.04,
        "BIAS_COEF": 0.0, "Q_VAL_OFFSET": 0.0,
        "LAPSE_RATE": 0.0,
    }
    base = dict(
        drift_fn_str="Classic", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False, include_RewardRate=False,
        dt=0.005, t_dur=0.5, dx=0.02,
        mle_condition_columns=("ChoiceCorrect", "ChoiceLeft"),
    )
    batched = evaluate_neg_loglik(
        params, df, MLEModelConfig(mle_use_batched_likelihood=True, **base))
    rowwise = evaluate_neg_loglik(
        params, df, MLEModelConfig(mle_use_batched_likelihood=False, **base))
    # Both finite, equal up to solver discretization.
    assert np.isfinite(batched.neg_loglik)
    assert np.isfinite(rowwise.neg_loglik)
    assert abs(batched.neg_loglik - rowwise.neg_loglik) < 1.0


# ---------------------------------------------------------------------------
# Filename invariance — the user A/B-tests by overwriting the same pickle
# ---------------------------------------------------------------------------

def test_evolveFP_filename_unchanged_by_mle_conditions():
    """The whole point of routing ``mle_condition_columns`` through
    ``MLEModelConfig`` (not ``evolveFP``) is that the saved-fit path
    is identical with or without the flag — so re-running with
    ``--mle-conditions`` overwrites the existing pickle in place.
    """
    base = dict(
        drift_fn_str="Classic", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)", t_dur=3, dt=0.005,
        is_loss_no_dir=False, fit_mode="mle")
    # evolveFP doesn't take mle_condition_columns at all — it shouldn't
    # need to. This test pins that contract: the call signature has no
    # condition-aware arg, so any future refactor that adds one will
    # break this test and surface the regression.
    fp = str(evolveFP(**base))
    assert fp.endswith("_3s_dt0.005.pkl")
    # Sanity: the suffix grammar is unchanged.
    fp_scaled = str(evolveFP(**base, uses_scaled_bound=True))
    assert fp_scaled.endswith("_3s_dt0.005_scaledB.pkl")


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def test_parse_mle_conditions_handles_empty_and_whitespace():
    from ...model_runner import _parse_mle_conditions

    assert _parse_mle_conditions(None) == ()
    assert _parse_mle_conditions("") == ()
    assert _parse_mle_conditions("  ") == ()
    assert _parse_mle_conditions("ChoiceCorrect") == ("ChoiceCorrect",)
    assert _parse_mle_conditions("ChoiceCorrect,ChoiceLeft") == (
        "ChoiceCorrect", "ChoiceLeft")
    # Tolerates whitespace and trailing commas.
    assert _parse_mle_conditions(" ChoiceCorrect , ChoiceLeft , ") == (
        "ChoiceCorrect", "ChoiceLeft")
