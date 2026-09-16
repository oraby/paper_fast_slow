'''Tests for the firing-reliability CDF (Figure S9B).

Reproduced while extracting: the pooled panel and all 23 per-session panels
are identical to the ones the notebook cell drew.

Pinned here: each strategy's percentage is out of that strategy's own trials,
a neuron absent from a strategy is missing rather than zero, and the synthetic
100% point that lifts the curve to the top of the axis.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ...common.definitions import BrainRegion
from ..activetrialscdf import (ALL, FAST, SLOW, activeTrialPercentages,
                               cdfCurve, plotActiveDistCDF)

MFC = int(BrainRegion.M2_Bi)
LFC = int(BrainRegion.ALM_Bi)


def neuron(trace_id, active, counts=(10, 10, 10), session="s1", br=MFC):
    """active: the strategy of each trial the neuron fired on."""
    return {"trace_id": trace_id, "active_quantile_idx": list(active),
            "quantile_idxs_count": {1: counts[0], 2: counts[1], 3: counts[2]},
            "ShortName": session, "BrainRegion": br}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_percentages_are_out_of_that_strategys_trials():
    df = pd.DataFrame([neuron("n1", [FAST] * 5 + [SLOW] * 1, counts=(10, 10, 20))])
    out = activeTrialPercentages(df).set_index("quantile_idx").active_prcnt
    assert out[FAST] == 50.0      # 5 of the 10 fast trials
    assert out[SLOW] == 5.0       # 1 of the 20 slow trials


def test_the_all_row_divides_by_every_trial():
    df = pd.DataFrame([neuron("n1", [FAST] * 5 + [SLOW], counts=(10, 10, 20))])
    out = activeTrialPercentages(df).set_index("quantile_idx").active_prcnt
    assert out[ALL] == pytest.approx(100 * 6 / 40)


def test_a_strategy_with_no_active_trials_is_absent_not_zero():
    df = pd.DataFrame([neuron("n1", [FAST] * 3)])
    out = activeTrialPercentages(df)
    assert set(out.quantile_idx) == {ALL, FAST}
    assert SLOW not in set(out.quantile_idx)


def test_one_row_per_neuron_and_strategy():
    df = pd.DataFrame([neuron("n1", [FAST, SLOW]), neuron("n2", [SLOW])])
    out = activeTrialPercentages(df)
    assert len(out) == 3 + 2      # n1: All, fast, slow; n2: All, slow
    assert list(out.trace_id) == ["n1", "n1", "n1", "n2", "n2"]


def test_the_curve_runs_from_most_to_least_reliable():
    df = pd.DataFrame([neuron("n1", [FAST] * 2), neuron("n2", [FAST] * 8),
                       neuron("n3", [FAST] * 5)])
    curve = cdfCurve(activeTrialPercentages(df), FAST, num_total_neurons=3)
    assert list(curve)[:4] == [100.0, 80.0, 50.0, 20.0]


def test_a_synthetic_full_reliability_point_is_prepended():
    """Without it the curve would start at the best real neuron, not 100%."""
    df = pd.DataFrame([neuron("n1", [FAST] * 2)])
    curve = cdfCurve(activeTrialPercentages(df), FAST, num_total_neurons=1)
    assert list(curve) == [100.0, 20.0]


def test_the_index_is_the_neuron_rank_as_a_percentage():
    df = pd.DataFrame([neuron("n1", [FAST] * 2), neuron("n2", [FAST] * 8)])
    curve = cdfCurve(activeTrialPercentages(df), FAST, num_total_neurons=2)
    assert list(curve.index) == [0.0, 50.0, 100.0]


def test_the_pooled_panel_draws_fast_and_slow_for_each_region():
    df = pd.DataFrame([neuron("n1", [FAST, SLOW], br=MFC),
                       neuron("n2", [FAST, SLOW], br=LFC)])
    plotActiveDistCDF(df, save_figs=False, by_sess=False)
    labels = [line.get_label() for line in plt.gca().lines]
    assert labels == ["ALM Impuslive", "ALM Deliberate",
                      "M2 Impuslive", "M2 Deliberate"]


def test_the_per_session_panels_add_the_all_curve():
    df = pd.DataFrame([neuron("n1", [FAST, SLOW], session="s1")])
    plotActiveDistCDF(df, save_figs=False, by_sess=True)
    labels = [line.get_label() for line in plt.gcf().axes[0].lines]
    assert labels[0].endswith("All")
    assert len(labels) == 3


def test_axes_are_reversed_and_symlog():
    df = pd.DataFrame([neuron("n1", [FAST, SLOW])])
    plotActiveDistCDF(df, save_figs=False, by_sess=False)
    ax = plt.gca()
    assert ax.get_xlim() == (100, 0)
    assert ax.get_ylim() == (0, 100)
    assert ax.get_xscale() == "symlog"


def test_saving_needs_a_prefix():
    df = pd.DataFrame([neuron("n1", [FAST, SLOW])])
    with pytest.raises(AssertionError, match="fig_save_prefix"):
        plotActiveDistCDF(df, save_figs=True, by_sess=False)


def test_saving_creates_the_folder_it_needs(tmp_path):
    """results/2P/active_cdf/ is not in the repo; the first save makes it."""
    df = pd.DataFrame([neuron("n1", [FAST, SLOW])])
    plotActiveDistCDF(df, save_figs=True, by_sess=False,
                      fig_save_prefix=str(tmp_path))
    assert (tmp_path / "active_cdf" / "_all.svg").exists()


def test_saved_per_session_under_its_label(tmp_path):
    df = pd.DataFrame([neuron("n1", [FAST, SLOW], session="sess_a")])
    plotActiveDistCDF(df, save_figs=True, by_sess=True,
                      fig_save_prefix=str(tmp_path))
    assert (tmp_path / "active_cdf" / "M2 - sess_a.svg").exists()
