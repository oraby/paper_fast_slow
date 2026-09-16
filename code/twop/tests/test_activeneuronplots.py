'''Tests for the active-count panels (Figures S11A middle/right, S11B).

Reproduced while extracting: the three panels are identical to the ones the
notebook cells drew.

Pinned here: percentages are taken against each session's own neuron count,
very short trials are dropped before anything is drawn, and each panel writes
the file name the paper's figure was built from.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ...common.definitions import BrainRegion
from ..activeneuronplots import (_commonPreprocess, plotEarlyActiveCount,
                                 plotEarlyActivePerf, plotLastActiveCount)

MFC = int(BrainRegion.M2_Bi)
LFC = int(BrainRegion.ALM_Bi)


def counts(n=40, session="s1", br=MFC, total=10, active=2, dur=1.0,
           start_or_end="start", correct=1.0, dvstr="Hard"):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "BrainRegion": br, "ShortName": session,
        "trial_number": np.arange(n),
        "ChoiceCorrect": correct, "DVstr": dvstr,
        "trial_dur_sec": dur + rng.normal(0, 0.05, n),
        "total_count": total, "trial_active_count": active,
        f"trial_{start_or_end}_max_active_count": active,
        f"trial_{start_or_end}_extend_active_count": active})


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_percentages_are_out_of_the_sessions_neurons():
    df = counts(n=4, total=10, active=2)
    out = _commonPreprocess(df, "start")
    assert (out.active_start_prcnt == 20.0).all()


def test_each_session_uses_its_own_denominator():
    df = pd.concat([counts(n=4, session="s1", total=10, active=2),
                    counts(n=4, session="s2", total=100, active=2)])
    out = _commonPreprocess(df, "start").set_index("ShortName")
    assert out.loc["s1"].active_start_prcnt.iloc[0] == 20.0
    assert out.loc["s2"].active_start_prcnt.iloc[0] == 2.0


def test_very_short_trials_are_dropped():
    """Below 0.3 s there is barely a sampling period to speak of."""
    df = pd.concat([counts(n=3, dur=1.0), counts(n=3, dur=0.1)])
    assert len(_commonPreprocess(df, "start")) == 3


def test_the_early_panel_draws_both_regions():
    df = pd.concat([counts(session="mfc", br=MFC), counts(session="lfc", br=LFC)])
    fig = plotEarlyActiveCount(df, save_figs=False, start_sampling_cutoff=0.3)
    assert fig.axes, "nothing was drawn"
    assert "early 0.3s" in fig.axes[0].get_ylabel()


def test_the_late_panel_labels_its_window_as_a_percentage():
    df = pd.concat([counts(session="mfc", br=MFC, start_or_end="end"),
                    counts(session="lfc", br=LFC, start_or_end="end")])
    fig = plotLastActiveCount(df, save_figs=False, look_back_dur=0.3)
    assert "last 0.3%" in fig.axes[0].get_ylabel()


def test_the_early_panel_is_saved_under_its_published_name(tmp_path):
    df = pd.concat([counts(session="mfc", br=MFC), counts(session="lfc", br=LFC)])
    plotEarlyActiveCount(df, save_figs=True, start_sampling_cutoff=0.3,
                         fig_save_prefix=str(tmp_path))
    assert (tmp_path / "rt_vs_early_active_neurons_corr.svg").exists()


def test_the_late_panel_is_saved_under_its_published_name(tmp_path):
    df = pd.concat([counts(session="mfc", br=MFC, start_or_end="end"),
                    counts(session="lfc", br=LFC, start_or_end="end")])
    plotLastActiveCount(df, save_figs=True, look_back_dur=0.3,
                        fig_save_prefix=str(tmp_path))
    assert (tmp_path / "rt_vs_last_active_neurons_corr.svg").exists()


def perfCounts(session, br):
    """The performance panel keeps only easy trials, and bins above 70%."""
    mostly_correct = counts(n=40, session=session, br=br, dvstr="Easy")
    mostly_correct.loc[mostly_correct.index[:6], "ChoiceCorrect"] = 0.0
    return mostly_correct


def test_the_performance_panel_is_saved_under_its_published_name(tmp_path):
    df = pd.concat([perfCounts("mfc1", MFC), perfCounts("mfc2", MFC),
                    perfCounts("lfc1", LFC), perfCounts("lfc2", LFC)])
    plotEarlyActivePerf(df, save_figs=True, start_sampling_cutoff=0.3,
                        fig_save_prefix=str(tmp_path))
    assert (tmp_path / "rt_vs_early_active_neurons_perf_rel_corr.svg").exists()


@pytest.mark.parametrize("plotter, kwargs", [
    (plotEarlyActiveCount, dict(start_sampling_cutoff=0.3)),
    (plotLastActiveCount, dict(look_back_dur=0.3)),
])
def test_saving_needs_a_prefix(plotter, kwargs):
    df = pd.concat([counts(session="mfc", br=MFC, start_or_end="start"),
                    counts(session="lfc", br=LFC, start_or_end="start"),
                    counts(session="m2", br=MFC, start_or_end="end"),
                    counts(session="alm", br=LFC, start_or_end="end")])
    with pytest.raises(AssertionError, match="fig_save_prefix"):
        plotter(df, save_figs=True, **kwargs)
