'''Tests for the rigid / stretching proportions (Figure 4K right).

Reproduced while extracting: the same wedges as the notebook cells drew.

The grouping is what these pin. A neuron is counted by *which* correlation
passes the threshold — peak timing, AUC, both, or neither — and "Rigid" is the
complement of the **union** of the two, so the four wedges partition the
neurons exactly once. Subtracting only "Both" (as an earlier version did) left
every single-metric neuron in Rigid and pushed the wedges past 100%.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ...common.definitions import BrainRegion
from ..rtcorrpies import _processSet, plotCorrThreshPieChart

MFC = int(BrainRegion.M2_Bi)
LFC = int(BrainRegion.ALM_Bi)
KEY = "zscore_filter_"


def neurons(rows, session="s1", br=MFC):
    """rows: (firing-position r, AUC r) per neuron."""
    return pd.DataFrame([
        {"long_trace_id": f"{session}_n{i}", "trace_id": i, "ShortName": session,
         "BrainRegion": br, KEY + "firing_pos_pearson_corr": firing,
         KEY + "amplitude_firing_pearson_corr": auc}
        for i, (firing, auc) in enumerate(rows)])


def wedges():
    """The pie's values, in the order the code passes them."""
    ax = plt.gcf().axes[0]
    return [w.theta2 - w.theta1 for w in ax.patches]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def drawOne(rows, corr_thresh=0.3):
    _processSet("MFC", neurons(rows), KEY, corr_thresh, save_figs=False,
                fig_save_prefix=None)
    return [round(w / 3.6, 1) for w in wedges()]      # degrees -> percent


def test_the_four_groups_partition_the_neurons():
    # one firing-only, one AUC-only, one both, one neither
    percents = drawOne([(0.9, 0.0), (0.0, 0.9), (0.9, 0.9), (0.0, 0.0)])
    assert percents == [25.0, 25.0, 25.0, 25.0]
    assert sum(percents) == 100.0


def test_rigid_excludes_neurons_correlated_on_either_metric():
    """Rigid is the complement of the union, not of the intersection."""
    percents = drawOne([(0.9, 0.0), (0.0, 0.9), (0.0, 0.0)])
    firing_only, both, auc_only, rigid = percents
    assert (firing_only, both, auc_only) == (pytest.approx(33.3, abs=0.1), 0.0,
                                             pytest.approx(33.3, abs=0.1))
    assert rigid == pytest.approx(33.3, abs=0.1)      # not 100%


def test_the_threshold_is_on_the_absolute_correlation():
    """A strong negative correlation counts as stretching too."""
    percents = drawOne([(-0.9, 0.0), (0.0, 0.0)])
    assert percents[0] == 50.0        # firing-position group
    assert percents[3] == 50.0        # rigid


def test_the_threshold_is_exclusive():
    percents = drawOne([(0.3, 0.0), (0.31, 0.0)], corr_thresh=0.3)
    assert percents[0] == 50.0        # only the 0.31 neuron passes
    assert percents[3] == 50.0


def test_a_higher_threshold_moves_neurons_into_rigid():
    rows = [(0.5, 0.0), (0.0, 0.0)]
    assert drawOne(rows, corr_thresh=0.3)[3] == 50.0
    assert drawOne(rows, corr_thresh=0.8)[3] == 100.0


def test_one_pie_per_region_plus_a_pooled_one():
    df = pd.concat([neurons([(0.9, 0.0), (0.0, 0.0)], br=MFC),
                    neurons([(0.0, 0.9), (0.0, 0.0)], session="s2", br=LFC)])
    plotCorrThreshPieChart(df, KEY, 0.3, save_figs=False, fig_save_prefix=None)
    titles = [f.axes[0].get_title() for f in map(plt.figure, plt.get_fignums())]
    assert any("Both_MFC_LFC" in t for t in titles)
    assert any(t.startswith("MFC") for t in titles)
    assert any(t.startswith("LFC") for t in titles)


def test_saved_under_the_published_name(tmp_path):
    df = neurons([(0.9, 0.0), (0.0, 0.0)])
    plotCorrThreshPieChart(df, KEY, 0.3, save_figs=True,
                           fig_save_prefix=str(tmp_path))
    assert (tmp_path / "Both_MFC_LFC_correlation_pie_chart.pdf").exists()
    assert (tmp_path / "MFC_correlation_pie_chart.pdf").exists()


def test_saving_needs_a_prefix():
    with pytest.raises(AssertionError, match="fig_save_prefix"):
        plotCorrThreshPieChart(neurons([(0.9, 0.0)]), KEY, 0.3, save_figs=True,
                               fig_save_prefix=None)
