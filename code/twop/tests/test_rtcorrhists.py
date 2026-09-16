'''Tests for the rt/activity correlation distributions (Figures S10B, S10C).

Reproduced while extracting: the same panels and the same summary table as the
notebook cells produced.

Two things are pinned because they are easy to read the other way round:

- neurons are split into **early** (peaking within ``time_offset`` of the
  trial's start) and **remaining** — and "remaining" absorbs the late group,
  which is why there is no separate "late" row;
- each row's ``mean`` is the percentage of the **region's** neurons past the
  correlation threshold, not the session's, so the per-session rows of a region
  add up to that region's percentage.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ...common.definitions import BrainRegion
from ..rtcorrhists import plotCorrDistributions, plotCorrJointHist

MFC = int(BrainRegion.M2_Bi)
LFC = int(BrainRegion.ALM_Bi)
KEY = "zscore_filter_"
FIRING = KEY + "firing_pos_pearson_corr"
AUC = KEY + "amplitude_firing_pearson_corr"


def neurons(rows, session="s1", br=MFC, shuffle_num=None):
    """rows: (start_mean, end_mean, firing r, AUC r) per neuron."""
    out = pd.DataFrame([
        {"trace_id": i, "ShortName": session, "BrainRegion": br,
         KEY + "start_max_firing_pos_mean": start,
         KEY + "start_max_firing_pos_std": 0.0,
         KEY + "end_max_firing_pos_mean": end,
         KEY + "end_max_firing_pos_std": 0.0,
         FIRING: firing, AUC: auc}
        for i, (start, end, firing, auc) in enumerate(rows)])
    if shuffle_num is not None:
        out["shuffle_num"] = shuffle_num
    return out


def summarise(df, shuffle=None, time_offset=0.3, corr_thresh=0.3):
    return plotCorrDistributions(df, filter_key=KEY, time_offset=time_offset,
                                 incld_std=False, save_figs=False,
                                 corr_thresh=corr_thresh, res_shuffle_df=shuffle)


EARLY = (0.1, 5.0)          # peaks just after the trial starts
LATE = (5.0, 0.1)           # peaks just before it ends


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_the_table_has_a_row_per_session_group_and_metric():
    out = summarise(neurons([EARLY + (0.5, 0.2), LATE + (-0.4, 0.1)]))
    assert set(out.columns) == {"BrainRegion", "ShortName", "when", "metric",
                                "mean", "std", "is_shuffle"}
    assert set(out.metric) == {FIRING, AUC}


def test_neurons_are_split_into_early_and_remaining():
    out = summarise(neurons([EARLY + (0.5, 0.2), LATE + (0.5, 0.2)]))
    assert set(out.when) == {"early", "remaining"}


def test_a_late_peak_lands_in_remaining_not_a_group_of_its_own():
    out = summarise(neurons([LATE + (0.5, 0.2)]))
    assert set(out.when) == {"remaining"}


def test_the_offset_decides_which_group_a_neuron_falls_in():
    df = neurons([(0.5, 5.0, 0.5, 0.2)])
    assert set(summarise(df, time_offset=0.3).when) == {"remaining"}
    assert set(summarise(df, time_offset=1.0).when) == {"early"}


def test_the_percentage_is_out_of_the_regions_neurons():
    """Two sessions, one correlated neuron each: 25% of the region, twice."""
    df = pd.concat([neurons([EARLY + (0.9, 0.0), EARLY + (0.0, 0.0)], session="s1"),
                    neurons([EARLY + (0.9, 0.0), EARLY + (0.0, 0.0)], session="s2")])
    firing = summarise(df).query("metric == @FIRING and when == 'early'")
    assert sorted(firing["mean"]) == [25.0, 25.0]      # not 50% each


def test_only_correlations_past_the_threshold_count():
    df = neurons([EARLY + (0.9, 0.0), EARLY + (0.1, 0.0)])
    firing = summarise(df).query("metric == @FIRING")
    assert list(firing["mean"]) == [50.0]
    assert list(summarise(df, corr_thresh=0.95).query("metric == @FIRING")["mean"]) == [0.0]


def test_shuffled_rows_are_averaged_over_the_shuffles():
    df = neurons([EARLY + (0.9, 0.0), EARLY + (0.9, 0.0)])
    # Two shuffles: one with both neurons correlated, one with neither.
    shuffled = pd.concat([neurons([EARLY + (0.9, 0.0), EARLY + (0.9, 0.0)], shuffle_num=0),
                          neurons([EARLY + (0.0, 0.0), EARLY + (0.0, 0.0)], shuffle_num=1)])
    out = summarise(df, shuffle=shuffled).query("metric == @FIRING")
    real = out[~out.is_shuffle]["mean"].iloc[0]
    fake = out[out.is_shuffle]["mean"].iloc[0]
    assert real == 100.0
    assert fake == 50.0                      # mean of 100% and 0%
    assert out[out.is_shuffle]["std"].iloc[0] > 0


def test_without_a_shuffle_frame_every_row_is_real():
    out = summarise(neurons([EARLY + (0.5, 0.2)]))
    assert set(out.is_shuffle) == {False}
    assert set(out["std"]) == {0}


def test_each_region_is_summarised_separately():
    df = pd.concat([neurons([EARLY + (0.9, 0.0)], br=MFC),
                    neurons([EARLY + (0.0, 0.0)], session="s2", br=LFC)])
    out = summarise(df).query("metric == @FIRING")
    assert out[out.BrainRegion == MFC]["mean"].iloc[0] == 100.0
    assert out[out.BrainRegion == LFC]["mean"].iloc[0] == 0.0


def test_the_distribution_panel_is_saved_under_its_published_name(tmp_path):
    plotCorrDistributions(neurons([EARLY + (0.5, 0.2)]), filter_key=KEY,
                          time_offset=0.3, incld_std=False, save_figs=True,
                          corr_thresh=0.3, fig_save_prefix=str(tmp_path))
    assert (tmp_path / f"rt_corr_{KEY}offset_0.3.pdf").exists()


def test_the_joint_panel_is_saved_under_its_published_name(tmp_path):
    df = neurons([EARLY + (0.5, 0.2), LATE + (-0.4, 0.1)])
    plotCorrJointHist(df, filter_key=KEY, time_offset=0.3, incld_std=False,
                      save_figs=True, fig_save_prefix=str(tmp_path))
    assert (tmp_path / f"rt_corr_{KEY}offset_0.3.svg").exists()


@pytest.mark.parametrize("plotter, kwargs", [
    (plotCorrDistributions, dict(corr_thresh=0.3)),
    (plotCorrJointHist, {}),
])
def test_saving_needs_a_prefix(plotter, kwargs):
    with pytest.raises(AssertionError, match="fig_save_prefix"):
        plotter(neurons([EARLY + (0.5, 0.2)]), filter_key=KEY, time_offset=0.3,
                incld_std=False, save_figs=True, **kwargs)
