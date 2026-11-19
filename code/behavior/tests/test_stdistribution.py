'''Tests for the sampling-time distribution of the human contexts (Figure S2B).

Pinned here: speed-context trials over 3 s are dropped and accuracy trials are
not; small subject x context groups are dropped; the z-score is of
``log(t - min + 1)`` over one pool of every subject and both contexts, so the
difference between contexts and between subjects both stay; one mean per
subject and context; and the panel draws one curve and one dashed mean per
context.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from .. import stdistribution as sd
from ..stdistribution import (CONTEXT_COL, SUBJECT_KEY, collectContexts,
                              dropLongSpeedTrials, plotSTDistribution,
                              subjectContextMeans, zScorePooledLog)

ACCURACY, SPEED = sd.ACCURACY_CONTEXT, sd.SPEED_CONTEXT
SESSION_TYPES = [("Accuracy", ACCURACY), ("Speed", SPEED)]
COLORS = {"Accuracy": "C0", "Speed": "C1"}
N = 200


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _humans(specs, seed=0):
    '''specs: {name: {session_type: (mean_seconds, sd)}}.'''
    rng = np.random.default_rng(seed)
    rows = []
    for name, contexts in specs.items():
        for session_type, (mean, sd_) in contexts.items():
            rows.append(pd.DataFrame({
                SUBJECT_KEY: name, CONTEXT_COL: session_type,
                "calcStimulusTime": np.clip(rng.normal(mean, sd_, N), 0.05,
                                            2.9)}))
    return pd.concat(rows, ignore_index=True)


def _slowAccuracyCohort(seed=0):
    '''Every subject samples ~1 s longer under the accuracy instruction.'''
    return _humans({f"S{i}": {ACCURACY: (1.6 + i * 0.2, 0.2),
                              SPEED: (0.6 + i * 0.2, 0.2)}
                    for i in range(6)}, seed=seed)


def _frame(rows):
    return pd.DataFrame(rows, columns=[SUBJECT_KEY, CONTEXT_COL,
                                       "calcStimulusTime"])


# --------------------------------------------------------------------------
# Trial selection
# --------------------------------------------------------------------------

def test_only_speed_trials_over_the_limit_are_dropped():
    df = _frame([("S1", SPEED, 2.9), ("S1", SPEED, 3.0), ("S1", SPEED, 3.1),
                 ("S1", ACCURACY, 3.1), ("S1", ACCURACY, 4.9)])
    kept = dropLongSpeedTrials(df)
    assert sorted(kept.calcStimulusTime[kept[CONTEXT_COL] == SPEED]) == [2.9, 3.0]
    assert len(kept[kept[CONTEXT_COL] == ACCURACY]) == 2


def test_the_speed_limit_is_three_seconds():
    assert sd.SPEED_MAX_SECONDS == 3


def test_groups_report_trial_counts_after_the_speed_limit():
    df = _slowAccuracyCohort()
    df.loc[df.index[df[CONTEXT_COL] == SPEED][:5], "calcStimulusTime"] = 4.0
    df.loc[df.index[df[CONTEXT_COL] == ACCURACY][:5], "calcStimulusTime"] = 4.0
    accuracy, speed = collectContexts(df, colors=COLORS,
                                      session_types=SESSION_TYPES)
    assert accuracy["n_trials"] == 6 * N
    assert speed["n_trials"] == 6 * N - 5


def test_small_subject_context_groups_are_dropped():
    df = _slowAccuracyCohort()
    small = df.index[(df[SUBJECT_KEY] == "S0") & (df[CONTEXT_COL] == SPEED)]
    df = df.drop(small[30:])                       # S0 keeps 30 speed trials
    accuracy, speed = collectContexts(df, colors=COLORS,
                                      session_types=SESSION_TYPES)
    assert "S0" not in speed["values"].index
    assert "S0" in accuracy["values"].index


# --------------------------------------------------------------------------
# The normalisation
# --------------------------------------------------------------------------

def test_the_z_score_is_of_the_shifted_log_over_one_pool():
    df = _frame([("S1", ACCURACY, 2.0), ("S1", SPEED, 0.5),
                 ("S2", ACCURACY, 1.0), ("S2", SPEED, 0.0)])
    t = df.calcStimulusTime
    logged = np.log(t - t.min() + 1)
    expected = (logged - logged.mean()) / logged.std(ddof=0)
    np.testing.assert_allclose(zScorePooledLog(df).calcStimulusTime, expected)


def test_between_context_difference_survives():
    accuracy, speed = (g["values"].mean() for g in collectContexts(
        _slowAccuracyCohort(), colors=COLORS, session_types=SESSION_TYPES))
    assert accuracy > 0 > speed


def test_between_subject_differences_survive():
    '''One pool for everyone: a slower subject stays to the right.'''
    accuracy, speed = collectContexts(_slowAccuracyCohort(), colors=COLORS,
                                      session_types=SESSION_TYPES)
    for group in (accuracy, speed):
        values = group["values"]
        assert values["S5"] > values["S0"]
        assert values.std() > 0.3


def test_one_mean_per_subject_and_context():
    df = _humans({"S1": {ACCURACY: (2.0, .3), SPEED: (1.0, .3)},
                  "S2": {ACCURACY: (2.0, .3), SPEED: (1.0, .3)}})
    assert len(subjectContextMeans(df)) == 4
    accuracy, speed = collectContexts(df, colors=COLORS,
                                      session_types=SESSION_TYPES)
    assert list(accuracy["values"].index) == ["S1", "S2"]


def test_context_order_follows_the_request():
    groups = collectContexts(_slowAccuracyCohort(), colors=COLORS,
                             session_types=[("Speed", SPEED),
                                            ("Accuracy", ACCURACY)])
    assert [g["label"] for g in groups] == ["Speed", "Accuracy"]


# --------------------------------------------------------------------------
# The panel
# --------------------------------------------------------------------------

def _figure(**kwargs):
    return plotSTDistribution(_slowAccuracyCohort(), colors=COLORS,
                              session_types=SESSION_TYPES, verbose=False,
                              **kwargs)


def test_panel_draws_a_curve_and_a_dashed_mean_per_context():
    _, ax = plt.subplots()
    groups = _figure(ax=ax)
    dashed = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
    solid = [ln for ln in ax.lines if ln.get_linestyle() == "-"]
    assert len(solid) == len(dashed) == 2
    for line, group in zip(dashed, groups):
        assert line.get_xdata()[0] == pytest.approx(group["values"].mean())
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels[0] == f"Accuracy\n(n=6 Subjects, {6 * N:,} Trials)"
    assert ax.get_xlabel() == "Sampling time (z-scored)"
    assert ax.get_xlim() == sd.X_LIM


def test_panel_saves_under_the_prefix(tmp_path):
    _figure(save_prefix=tmp_path, save_figs=True)
    assert (tmp_path / f"{sd.DEF_FIG_NAME}.svg").exists()


def test_panel_saves_nothing_when_off(tmp_path):
    _figure(save_prefix=tmp_path, save_figs=False)
    assert not list(tmp_path.iterdir())


def test_panel_refuses_to_save_without_a_prefix():
    with pytest.raises(ValueError, match="save_prefix"):
        _figure(save_figs=True)
