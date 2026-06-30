"""Tests for the per-candidate DE loss trace.

Every ``differential_evolution`` candidate's ``(param values, loss)`` is
captured (in-process for the vectorized MLE/joint path; through a
``Manager().list()`` for the pooled chisq path) and assembled into a
``candidate_losses_df`` stored on the result payload: ``loss`` + one column per
fit param, plus (MLE) the weights/conditions the fit ran with.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import fit
from ..mle import MLEModelConfig


def _config():
    return MLEModelConfig(
        drift_fn_str="Classic", bias_fn_str="None_", noise_fn_str="Normal(0, 1)",
        include_Q=False, include_RewardRate=False, dt=0.01, t_dur=0.5,
        mle_choice_weight=3.0, mle_rt_weight=1.0, mle_choice_norm="conditional",
        mle_mle_weight=1.0, mle_chi2_weight=0.5,
        mle_condition_columns=("ChoiceCorrect", "ChoiceLeft"))


def test_trace_population_appends_per_candidate():
    trace = []
    # 3 params x 4 candidates; losses one per candidate.
    x_matrix = np.arange(12, dtype=float).reshape(3, 4)
    losses = np.array([10.0, 11.0, 12.0, 13.0])
    fit._trace_population(trace, x_matrix, losses)
    assert len(trace) == 4
    # Each row is (param-list aligned to the column, loss).
    first_params, first_loss = trace[0]
    assert first_params == [0.0, 4.0, 8.0]   # column 0 of x_matrix
    assert first_loss == 10.0
    assert [loss for _x, loss in trace] == [10.0, 11.0, 12.0, 13.0]


def test_trace_population_none_is_noop():
    # Must tolerate tracing disabled (None) without error.
    fit._trace_population(None, np.zeros((2, 2)), np.zeros(2))


def test_build_candidate_loss_df_mle_columns():
    trace = [([0.5, 1.5], 7.0), ([0.6, 1.4], 6.5)]
    df = fit._build_candidate_loss_df(trace, ["DRIFT_COEF", "BOUND"], _config())
    assert list(df["loss"]) == [7.0, 6.5]
    assert list(df["DRIFT_COEF"]) == [0.5, 0.6]
    assert list(df["BOUND"]) == [1.5, 1.4]
    # MLE metadata columns (constant per fit, self-describing).
    assert (df["mle_choice_weight"] == 3.0).all()
    assert (df["mle_rt_weight"] == 1.0).all()
    assert (df["mle_mle_weight"] == 1.0).all()
    assert (df["mle_chi2_weight"] == 0.5).all()
    assert (df["mle_choice_norm"] == "conditional").all()
    # Conditions stored as a string object so the frame round-trips to pickle.
    assert df["mle_condition_columns"].iloc[0] == "('ChoiceCorrect', 'ChoiceLeft')"
    assert df["mle_condition_columns"].dtype == object


def test_build_candidate_loss_df_chisq_has_no_metadata():
    trace = [([0.5, 1.5], 7.0)]
    df = fit._build_candidate_loss_df(trace, ["DRIFT_COEF", "BOUND"],
                                      model_config=None)
    assert list(df.columns) == ["loss", "DRIFT_COEF", "BOUND"]


def test_build_candidate_loss_df_empty_is_none():
    assert fit._build_candidate_loss_df([], ["A"], None) is None
    assert fit._build_candidate_loss_df(None, ["A"], None) is None
