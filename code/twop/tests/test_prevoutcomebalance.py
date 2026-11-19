'''Tests for the previous-outcome balance section (``prevoutcomebalance``).

What these pin: the per-run peak statistics, the "All" vs tuned-variable row
layout, that runs longer than ``max_outcome`` are clipped, which neurons count
as rising, and the per-session percentages the section prints.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ...common.definitions import BrainRegion
from .. import prevoutcomebalance as pb

MFC = int(BrainRegion.M2_Bi)
RAW_LEN = 6


@pytest.fixture(autouse=True)
def closeFigures():
    yield
    plt.close("all")


def trials(trace_id, rows, sess="s1"):
    '''rows: (streak, ChoiceLeft, level) -- level fills the whole Raw trace.'''
    frames = []
    for streak, choice_left, level in rows:
        raw = pd.DataFrame([[float(level)] * RAW_LEN],
                           columns=pd.MultiIndex.from_product([["Raw"], range(RAW_LEN)]))
        raw[("long_trace_id", "")] = trace_id
        raw[("ShortName", "")] = sess
        raw[("BrainRegion", "")] = MFC
        raw[("PrevOutcomeCount", "")] = streak
        raw[("ChoiceLeft", "")] = choice_left
        raw[("epochs_ranges", "")] = [[(0, 1), (2, 3), (4, 5)]]
        raw[("epochs_names", "")] = [["Pre", "Sampling", "Movement"]]
        frames.append(raw)
    return pd.concat(frames, ignore_index=True)


def stats(trace_id, pval=0.01, data_col="ChoiceLeft", is_left_tuned=True):
    return pd.DataFrame([{"long_trace_id": trace_id, "data_col": data_col,
                          "pval": pval, "prior_data_col": np.nan,
                          "DVstr": np.nan, "IsROCLeftTuend": is_left_tuned,
                          "BrainRegion": MFC, "ShortName": "s1"}])


def test_process_trace_peak_stats_per_run():
    df = trials("t", [(1, True, 2), (1, True, 4), (2, True, 6)])
    res = pb.processTrace(df)
    assert res["PrevOutcomeCount"] == [1, 2]
    assert res["AllMaxMean"] == [3.0, 6.0]
    assert res["AllNumTrials"] == [2, 1]
    assert np.isnan(res["PrefMaxMean"]).all()


def test_process_trace_splits_preferred_trials():
    df = trials("t", [(1, True, 5), (1, False, 1)])
    res = pb.processTrace(df, "ChoiceLeft", data_col_pref=True)
    assert res["PrefMaxMean"] == [5.0]
    assert res["NonPrefMaxMean"] == [1.0]
    assert np.isnan(res["AllMaxMean"]).all()


def test_balance_df_layout_and_clipping():
    df = trials("t", [(-5, True, 1), (1, True, 2), (7, False, 3)])
    out = pb.makeBalanceDf(stats("t"), df, pval=0.05, max_outcome=3)
    all_rows = out[out.data_col.isnull()]
    tuned_rows = out[out.data_col == "ChoiceLeft"]
    assert sorted(all_rows.PrevOutcomeCount) == [-3, 1, 3]
    assert sorted(tuned_rows.PrevOutcomeCount) == [-3, 1, 3]
    assert all_rows.AllMaxMean.notnull().all()
    assert tuned_rows.AllMaxMean.isnull().all()


def test_balance_df_skips_untuned_neurons():
    df = trials("t", [(1, True, 1), (2, True, 2)])
    out = pb.makeBalanceDf(stats("t", pval=0.5), df, pval=0.05)
    assert out.data_col.isnull().all()


def test_rising_neurons_counted_per_session():
    rising = trials("up", [(-1, True, 1), (1, True, 2), (2, True, 3)])
    flat = trials("flat", [(-1, True, 2), (1, True, 2), (2, True, 2)])
    df = pd.concat([rising, flat], ignore_index=True)
    balance = pb.makeBalanceDf(stats("up", pval=0.5), df, pval=0.05)
    out = pb.incrementingStats(balance, {"s1": 2}, plot=False, save_figs=False)
    glob = out[(out.BrainRegion == "Global") & (out.data_col == "Global")].iloc[0]
    assert (glob.inc, glob.total, glob.prcnt) == (1, 2, 50.0)
    per_br = out[(out.BrainRegion == "M2") & (out.data_col == "All")].iloc[0]
    assert per_br.inc == 1


def test_rising_neuron_figure_is_saved_under_reward_rate(tmp_path):
    # Two trials per run, so every run's mean trace has an SEM band
    df = trials("up", [(-1, True, 1), (-1, True, 1.5), (1, True, 2),
                       (1, True, 2.5), (2, True, 3), (2, True, 3.5)])
    balance = pb.makeBalanceDf(stats("up", pval=0.5), df, pval=0.05)
    pb.incrementingStats(balance, {"s1": 1}, plot=False, save_figs=True,
                         fig_save_prefix=str(tmp_path))
    saved = list((tmp_path / "RewardRate" / "IncNeuron").iterdir())
    assert [p.name for p in saved] == ["BalanceChangeCurChoiceEx_up_All.svg"]


def test_print_stats(capsys):
    stats_df = pd.DataFrame({"data_col": ["Global", "Global"],
                             "BrainRegion": ["M2", "M2"],
                             "ShortName": ["a", "b"], "total": [10, 10],
                             "inc": [1, 3], "prcnt": [10.0, 30.0]})
    pb.printStats(stats_df)
    assert "Global: 4/20 (20.00%)" in capsys.readouterr().out
