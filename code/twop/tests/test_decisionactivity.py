'''Tests for the decision-aligned population activity (Figure S11B, left).

Pinned here: "fast" / "slow" are the shortest / longest duration bins, time
runs from movement onset at the data's rate and is cut to the window, a curve
is the mean over sessions of the choice-balanced neuron mean (+/- SEM across
sessions), points only one session has are dropped, and the figure draws the
four curves on one axes with the y axis on the right.
'''
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from .. import decisionactivity as da
from ...common.definitions import BrainRegion
from ..plot.activitysum import _plotDf, groupCurves, pruneSingleSession

RATE = 10.0
LEN = 8
ONSET = 6                         # movement onset sample
NAMES = ["-0.6s Movement to Lateral Port", da.MOVEMENT_EPOCH]
RANGES = [(0, ONSET - 1), (ONSET, LEN - 1)]
FAST, MID, SLOW = (pd.Interval(0.4, 0.7), pd.Interval(0.7, 1.0),
                   pd.Interval(1.0, 1.3))
M2, ALM = int(BrainRegion.M2_Bi), int(BrainRegion.ALM_Bi)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def trial(br, sess, dur_bin, choice_left, values, trial_num=0):
    '''One trial; ``values`` maps neuron id -> trace (or a constant).'''
    traces = {tid: (np.full(LEN, float(v)) if np.isscalar(v)
                    else np.asarray(v, float)) for tid, v in values.items()}
    return dict(BrainRegion=br, Layer="L23", ShortName=sess,
                TrialNumber=trial_num, ChoiceLeft=choice_left, dur_bin=dur_bin,
                acq_sampling_rate=RATE, epochs_names=NAMES,
                epochs_ranges=RANGES, trace_start_idx=0,
                trace_end_idx=LEN - 1, traces_sets={"neuronal": traces})


def frame():
    rows = []
    for br, offset in [(M2, 0.0), (ALM, 10.0)]:
        for b_idx, b in enumerate([FAST, MID, SLOW]):
            # s1: 3 left trials at 1, one right trial at 3 -> balanced mean 2
            for t in range(3):
                rows.append(trial(br, "s1", b, True,
                                  {1: offset + b_idx + 1}, t))
            rows.append(trial(br, "s1", b, False, {1: offset + b_idx + 3}, 3))
            # s2: two neurons, 4 and 6, both sides -> 5
            rows.append(trial(br, "s2", b, True,
                              {1: offset + b_idx + 4, 2: offset + b_idx + 6}, 0))
            rows.append(trial(br, "s2", b, False,
                              {1: offset + b_idx + 4, 2: offset + b_idx + 6}, 1))
    return pd.DataFrame(rows)


def test_movement_onset_is_read_from_the_epochs():
    assert da.movementOnsetIdx(frame()) == ONSET


def test_misaligned_movement_onsets_raise():
    df = frame()
    df.at[0, "epochs_ranges"] = [(0, 4), (5, 7)]
    with pytest.raises(ValueError, match="not aligned"):
        da.movementOnsetIdx(df)


def test_fast_and_slow_are_the_shortest_and_longest_bins():
    assert da.fastSlowBins(frame()) == (FAST, SLOW)


def test_curves_keep_only_fast_and_slow_in_the_window():
    curves = da.decisionCurves(frame(), window_sec=(-0.4, 0.1))
    assert set(curves.bin) == {FAST, SLOW}
    assert set(curves.Region) == {"MFC", "LFC"}
    for _, df in curves.groupby(["Region", "Strategy"]):
        np.testing.assert_allclose(df.time, [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1])
    fast = curves[(curves.Region == "MFC") & (curves.Strategy == "Fast")]
    assert (fast.bin_rank == 1).all() and (fast.n_bins == 3).all()
    slow = curves[(curves.Region == "MFC") & (curves.Strategy == "Slow")]
    assert (slow.bin_rank == 3).all()


def test_a_curve_is_the_session_mean_of_choice_balanced_neuron_means():
    curves = da.decisionCurves(frame())
    mfc_fast = curves[(curves.Region == "MFC") & (curves.Strategy == "Fast")]
    # s1: (1 + 3)/2 = 2, not the trial-pooled 1.5; s2: (4 + 6)/2 = 5
    np.testing.assert_allclose(mfc_fast["mean"], 3.5)
    np.testing.assert_allclose(mfc_fast["sem"], 1.5)      # std([2, 5], ddof=1)/sqrt(2)
    assert (mfc_fast.n_sessions == 2).all() and (mfc_fast.n_neurons == 3).all()
    lfc_slow = curves[(curves.Region == "LFC") & (curves.Strategy == "Slow")]
    np.testing.assert_allclose(lfc_slow["mean"], 15.5)  # (12 + 15)/2 + 2 bins up


def test_points_only_one_session_has_are_dropped():
    df = frame()
    head_nan = np.r_[np.full(3, np.nan), np.full(LEN - 3, 4.0)]
    sel = (df.ShortName == "s2") & (df.dur_bin == FAST) & (df.BrainRegion == M2)
    for idx in df.index[sel]:
        df.at[idx, "traces_sets"] = {"neuronal": {1: head_nan, 2: head_nan}}
    curves = da.decisionCurves(df, window_sec=(-0.6, 0.1))
    mfc_fast = curves[(curves.Region == "MFC") & (curves.Strategy == "Fast")]
    assert np.isnan(mfc_fast["mean"].iloc[:3]).all()
    np.testing.assert_allclose(mfc_fast["mean"].iloc[3:], 3.0)   # (2 + 4)/2


def test_more_than_one_layer_raises():
    df = frame()
    df.loc[df.index[:2], "Layer"] = "L5"
    with pytest.raises(ValueError, match="one layer"):
        da.decisionCurves(df)


def test_prune_blanks_where_the_sem_is_nan():
    out = pruneSingleSession([1.0, 2.0, 3.0], np.array([0.1, np.nan, 0.2]))
    assert out[0] == 1.0 and np.isnan(out[1]) and out[2] == 3.0


def test_plot_df_draws_the_group_curves():
    df = frame()
    df = df[df.BrainRegion == M2]
    _, ax = plt.subplots()
    _plotDf(df, "dur_bin", "before", [], "Blues", ax)
    curves = groupCurves(df, "dur_bin", "before")
    assert len(ax.lines) == len(curves) == 3
    for line, curve in zip(ax.lines, curves):
        np.testing.assert_allclose(line.get_ydata(), curve["mean"])
        assert line.get_label().startswith(f"Group={curve['group']} (2 sessions")


def test_figure_draws_four_curves_and_saves(tmp_path):
    fig = da.plotDecisionActivity(da.decisionCurves(frame()), save_figs=True,
                                  fig_save_prefix=str(tmp_path))
    ax = fig.axes[0]
    labels = [l.get_label() for l in ax.lines if not l.get_label().startswith("_")]
    assert labels == ["MFC (Fast)", "MFC (Slow)", "LFC (Fast)", "LFC (Slow)"]
    assert ax.get_xlim() == da.WINDOW_SEC
    assert ax.yaxis.get_ticks_position() == "right"
    assert ax.get_xlabel() == "Decision time (s)"
    assert os.path.exists(tmp_path / da.SAVE_DIR / da.SAVE_NAME)


def test_figure_saves_nothing_when_off(tmp_path):
    da.plotDecisionActivity(da.decisionCurves(frame()), save_figs=False,
                            fig_save_prefix=str(tmp_path))
    assert not os.listdir(tmp_path)
