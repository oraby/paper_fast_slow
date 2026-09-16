'''Tests for tuning balance across an outcome streak (Figure 6C).

Reproduced while extracting: the same balance table as the notebook cell, and
the same example neurons selected from it.

Pinned here: streaks are clipped rather than dropped, the no-streak bin is
removed, the activity window is the bin *after* sampling onset, and the example
panel picks neurons by a monotonic-rise rule rather than by hand.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ...common.definitions import BrainRegion
from ..balancechange import (MAX_EXAMPLES, _calcTraceBalance, balanceChange,
                             plotBalanceChangeSingleNeuron, risesWithTheStreak,
                             samplingWindow)

MFC = int(BrainRegion.M2_Bi)
RAW_LEN = 12


def trials(trace_id, rows):
    """rows: (streak, ChoiceLeft, level) -- level fills the whole Raw trace."""
    frames = []
    for streak, choice_left, level in rows:
        raw = pd.DataFrame([[float(level)] * RAW_LEN],
                           columns=pd.MultiIndex.from_product(
                               [["Raw"], range(RAW_LEN)]))
        raw[("long_trace_id", "")] = trace_id
        raw[("PrevOutcomeCount", "")] = streak
        raw[("ChoiceLeft", "")] = choice_left
        raw[("epochs_ranges", "")] = [[(0, 1), (2, 8), (9, 11)]]
        frames.append(raw)
    return pd.concat(frames, ignore_index=True)


def stats(trace_id, is_left_tuned=True, pval=0.01, data_col="ChoiceLeft"):
    return pd.DataFrame([{"long_trace_id": trace_id, "data_col": data_col,
                          "pval": pval, "prior_data_col": np.nan,
                          "DVstr": np.nan, "IsROCLeftTuend": is_left_tuned,
                          "BrainRegion": MFC}])


def active(*trace_ids):
    return pd.DataFrame({"trace_id": list(trace_ids)})


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_the_window_is_the_bin_after_sampling_onset():
    # epochs: before (0,1), sampling (2,8), after (9,11); 4 bins gives edges
    # [0, 2, 5.5, 9, 12], and the window is the second to the third, rounded.
    assert samplingWindow([(0, 1), (2, 8), (9, 11)], bins_count=4) == (2, 6)


def test_streaks_are_clipped_not_dropped():
    df = trials("n1", [(5, True, 10), (-9, True, 2), (1, True, 6)])
    out = balanceChange(stats("n1"), df, active("n1"), bins_count=4, pval=0.05,
                        max_count=3)
    assert sorted(out.PrevOutcomeCount) == [-3, 1, 3]


def test_the_no_streak_bin_is_removed():
    df = trials("n1", [(0, True, 10), (1, True, 6)])
    out = balanceChange(stats("n1"), df, active("n1"), bins_count=4, pval=0.05)
    assert list(out.PrevOutcomeCount) == [1]


def test_preferred_follows_the_roc_direction():
    df = trials("n1", [(1, True, 10), (1, False, 2)])
    left = balanceChange(stats("n1", is_left_tuned=True), df, active("n1"),
                         bins_count=4, pval=0.05).iloc[0]
    right = balanceChange(stats("n1", is_left_tuned=False), df, active("n1"),
                          bins_count=4, pval=0.05).iloc[0]
    assert (left.MeanPref, left.MeanNnoPref) == (10.0, 2.0)
    assert (right.MeanPref, right.MeanNnoPref) == (2.0, 10.0)


def test_untuned_neurons_are_excluded():
    df = trials("n1", [(1, True, 10)])
    with pytest.raises(ValueError):       # nothing left to concatenate
        balanceChange(stats("n1", pval=0.9), df, active("n1"), bins_count=4,
                      pval=0.05)


def test_neurons_outside_the_active_set_are_excluded():
    df = trials("n1", [(1, True, 10)])
    with pytest.raises(ValueError):
        balanceChange(stats("n1"), df, active("other"), bins_count=4, pval=0.05)


def test_a_neuron_tuned_both_ways_is_refused():
    df = trials("n1", [(1, True, 10), (1, False, 4)])
    both = pd.concat([stats("n1", is_left_tuned=True),
                      stats("n1", is_left_tuned=False)])
    with pytest.raises(AssertionError, match="tuned both ways"):
        balanceChange(both, df, active("n1"), bins_count=4, pval=0.05)


def test_trials_are_averaged_before_the_mean_is_taken():
    df = trials("n1", [(1, True, 4), (1, True, 8), (1, False, 1)])
    out = balanceChange(stats("n1"), df, active("n1"), bins_count=4,
                        pval=0.05).iloc[0]
    assert out.MeanPref == pytest.approx(6.0)


def test_the_sem_is_across_time_points_not_trials():
    """Two trials at 4 and 8 give a flat mean trace, so the SEM is 0, not 2."""
    df = trials("n1", [(1, True, 4), (1, True, 8), (1, False, 1)])
    out = balanceChange(stats("n1"), df, active("n1"), bins_count=4,
                        pval=0.05).iloc[0]
    assert out.MeanPrefSEM == pytest.approx(0.0)


def test_only_the_window_is_averaged():
    """Values outside bins[1]:bins[2] must not reach the mean."""
    raw = pd.DataFrame([[0.0] * RAW_LEN],
                       columns=pd.MultiIndex.from_product([["Raw"], range(RAW_LEN)]))
    raw.loc[0, ("Raw", 2)] = 6.0        # inside
    raw.loc[0, ("Raw", 3)] = 6.0        # inside
    raw.loc[0, ("Raw", 11)] = 999.0     # outside
    raw[("long_trace_id", "")] = "n1"
    raw[("PrevOutcomeCount", "")] = 1
    raw[("ChoiceLeft", "")] = True
    raw[("epochs_ranges", "")] = [[(0, 1), (2, 8), (9, 11)]]
    out = balanceChange(stats("n1"), raw, active("n1"), bins_count=4, pval=0.05)
    assert out.iloc[0].MeanPref == pytest.approx(12.0 / 5)   # idx 2..6


def test_rise_rule_accepts_only_monotonic_growth():
    rising = pd.DataFrame({"MeanPref": [1.0, 2.0, 3.0]})
    flat = pd.DataFrame({"MeanPref": [1.0, 1.0, 3.0]})
    assert risesWithTheStreak(rising)
    assert not risesWithTheStreak(flat)


def balance_rows(trace_id, means):
    return pd.DataFrame({"trace_id": trace_id, "BrainRegion": MFC,
                         "PrevOutcomeCount": range(len(means)),
                         "MeanPref": means, "MeanNnoPref": 1.0,
                         "MeanPrefSEM": 0.1, "MeanNnoPrefSEM": 0.1,
                         "IsLeftTuned": True})


def test_only_rising_neurons_are_drawn():
    df = pd.concat([balance_rows("flat", [1.0, 1.0, 1.0]),
                    balance_rows("rising", [1.0, 2.0, 3.0])])
    plotBalanceChangeSingleNeuron(df, save_figs=False)
    titles = [ax.get_title() for ax in plt.gcf().axes] + \
             [f.axes[0].get_title() for f in map(plt.figure, plt.get_fignums())]
    assert any("rising" in t for t in titles)
    assert not any("flat" in t for t in titles)


def test_the_filter_can_be_switched_off():
    df = balance_rows("flat", [1.0, 1.0, 1.0])
    plotBalanceChangeSingleNeuron(df, save_figs=False, find_pattern=False)
    assert any("flat" in f.axes[0].get_title() for f in
               map(plt.figure, plt.get_fignums()))


def test_it_stops_after_two_examples_per_region():
    df = pd.concat([balance_rows(f"n{i}", [1.0, 2.0, 3.0]) for i in range(5)])
    plotBalanceChangeSingleNeuron(df, save_figs=False)
    assert len(plt.get_fignums()) == MAX_EXAMPLES


def test_saved_under_the_published_name(tmp_path):
    (tmp_path / "RT_Stats").mkdir()
    plotBalanceChangeSingleNeuron(balance_rows("n1", [1.0, 2.0, 3.0]),
                                  save_figs=True, fig_save_prefix=str(tmp_path))
    assert (tmp_path / "RT_Stats" / "PrevChoiceCorrect" /
            "BalanceChangeCurChoiceEx_n1.svg").exists()
