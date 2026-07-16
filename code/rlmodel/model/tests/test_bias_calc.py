"""Regression tests for ``behavior/bias.py``'s calcBias.

Two bugs motivate these:

1. ``groupby("Name").apply(..., include_groups=False)`` dropped the "Name"
   column the inner groupby needs, raising ``KeyError: 'Name'`` for every
   caller (commit aadbdab, "avoid pandas Future deprecation warnings").
2. The first fix (a plain ``pd.concat``) restored correctness but changed the
   *shape*: ``apply`` collapses same-indexed per-group Series into a DataFrame,
   and ``_plotMotorBias`` depends on that — for one subject it does
   ``calcBias(df, col, ["SessId"]).unstack()`` and scatters against
   ``arange(len(...))``. ``pd.concat`` returned a Series, so ``.unstack()`` gave
   a 1×N frame (len 1) instead of a length-N Series → "x and y must be the same
   size".

So these pin the exact shapes the callers consume, matching the original
apply-based behaviour.
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ....behavior import bias as B
from ....behavior.bias import calcBias
from ..plotter import _plotMotorBias


def _df(names=("S1", "S2"), n_sess=3):
    rows = []
    for name in names:
        for sess in range(n_sess):
            for i in range(4):
                rows.append({"Name": name, "SessId": f"{name}_d{sess}_1",
                             "ChoiceLeft": float(i % 2),
                             "SimChoiceLeft": float(i % 2),
                             "DV": 0.5 if i % 2 else -0.5,
                             "Date": f"d{sess}", "SessionNum": 1,
                             "quantile_idx": 1})
    return pd.DataFrame(rows)


def _original_apply(df, choice_left_col, groupby_cols):
    """The pre-aadbdab implementation, as the golden shape reference."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return df.groupby("Name").apply(
            B.calcSubjQuantileBias, choice_left_col=choice_left_col,
            groupby_cols=groupby_cols)


@pytest.mark.parametrize("groupby_cols,names", [
    (None, ("S1", "S2")),          # plotBias: default groupby, many subjects
    (["Name"], ("S1", "S2")),      # _plotMotorBias multi-subject branch
    (["Name"], ("S1",)),           # subject_metrics: one subject
    (["SessId"], ("S1",)),         # _plotMotorBias single-subject branch
])
def test_calc_bias_matches_original_apply_shape(groupby_cols, names):
    df = _df(names=names)
    kw = {} if groupby_cols is None else {"groupby_cols": groupby_cols}
    got = calcBias(df, "ChoiceLeft", **kw)
    want = _original_apply(df, "ChoiceLeft",
                           groupby_cols or B._GROUPBY_COLS)
    assert type(got) is type(want)
    assert got.shape == want.shape
    assert got.equals(want)


def test_motor_bias_single_subject_unstack_matches_session_count():
    """The exact path that crashed: one subject, ['SessId'], then .unstack()
    must yield one value per session so scatter(x, arange(len)) aligns."""
    df = _df(names=("S1",), n_sess=4)
    unstacked = calcBias(df, "ChoiceLeft", groupby_cols=["SessId"]).unstack()
    assert isinstance(unstacked, pd.Series)
    assert len(unstacked) == 4               # one per session


def test_motor_bias_multi_subject_length_is_subject_count():
    df = _df(names=("S1", "S2", "S3"))
    res = calcBias(df, "ChoiceLeft", groupby_cols=["Name"])
    assert len(res) == 3


def test_subject_metrics_single_subject_unstack_iloc0_is_scalar():
    """subject_metrics does calcBias(...).unstack().iloc[0] and expects a
    per-subject scalar bias."""
    res = calcBias(_df(names=("S1",)), "ChoiceLeft", groupby_cols=["Name"])
    value = res.unstack().iloc[0]
    assert isinstance(value, float)


def test_calc_bias_accepts_a_sim_choice_column():
    res = calcBias(_df(), "SimChoiceLeft", groupby_cols=["Name"])
    assert len(res) == 2


def test_calc_bias_value_is_choice_minus_rewarded_rate():
    # All-left choices where only half the trials reward left -> +50% bias.
    df = _df(names=("S1",))
    df["ChoiceLeft"] = 1.0
    res = calcBias(df, "ChoiceLeft", groupby_cols=["Name"])
    assert res.unstack().iloc[0] == pytest.approx(50.0)


# --------------------------------------------------------------------------
# The concrete caller that surfaced the shape bug (ValueError: x and y must be
# the same size), for both its single- and multi-subject branches.
# --------------------------------------------------------------------------
def _motor_bias_df(names=("S1",), n_sess=4, n_trials=200):
    rng = np.random.default_rng(0)
    rows = []
    for name in names:
        for s in range(n_sess):
            for _ in range(n_trials):
                dv = rng.choice([-0.9, -0.5, -0.2, 0.2, 0.5, 0.9])
                rows.append({
                    "Name": name, "SessId": f"{name}_d{s}_1",
                    "DVstr": {0.9: "Easy", 0.5: "Med", 0.2: "Hard"}[abs(dv)],
                    "DV": dv, "ChoiceLeft": float(rng.random() < 0.5),
                    "SimChoiceLeft": float(rng.random() < 0.5),
                    "calcStimulusTime": float(rng.gamma(2, 0.15) + 0.1),
                    "SimRT": float(rng.gamma(2, 0.15) + 0.1)})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("names", [("S1",), ("S1", "S2", "S3")])
def test_plot_motor_bias_runs_for_single_and_multi_subject(names):
    _plotMotorBias(_motor_bias_df(names=names), plt.subplots()[1])
    plt.close("all")
