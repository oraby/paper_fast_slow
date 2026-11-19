'''Tests for the single-neuron correlation panels (Figure 4K left).

The point of these tests is the pair of loops. ``loopNeuronsPlot`` returns a
**list of (ids, results) pairs**, which is what ``makeDF`` and the shuffled
control expect; ``loopNeuronsSummary`` returns a **DataFrame**.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ...common.definitions import BrainRegion
from ..rtcorrpanels import (ACQ_RATE, loopNeuronsPlot, loopNeuronsSummary,
                            summariseNeuronCorr)

MFC = int(BrainRegion.M2_Bi)
LFC = int(BrainRegion.ALM_Bi)
KEY = "smoothed2_"


def neuronTrials(n_trials=12, trace_id=7, session="s1", br=MFC, peak_moves=True,
                 n=40):
    """One row per trial; the peak tracks the trial duration when asked."""
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n_trials):
        dur = 0.5 + i * 0.1
        peak_sec = dur * 0.5 if peak_moves else 0.3
        trace = np.zeros(n)
        trace[int(round(peak_sec * ACQ_RATE))] = 10.0
        rows.append({
            "BrainRegion": br, "ShortName": session, "TrialNumber": i,
            "trial_dur_sec": dur, "trace_id": trace_id, "ChoiceCorrect": 1.0,
            "DVstr": "Hard", "is_active": True,
            KEY + "max_pos_sec": peak_sec, KEY + "max_amplitude": 10.0,
            KEY + "auc": 10.0 + rng.normal(0, 0.1), KEY + "trace": trace})
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_the_plot_loop_returns_pairs_of_ids_and_results():
    out = loopNeuronsPlot(neuronTrials(), key_prefix=KEY, plot=False,
                          save_figs=False)
    assert isinstance(out, list)
    ids, results = out[0]
    assert ids == {"BrainRegion": MFC, "ShortName": "s1", "trace_id": 7}
    assert isinstance(results, dict)


def test_the_summary_loop_returns_a_frame_instead():
    """The two return different shapes."""
    out = loopNeuronsSummary(neuronTrials(), key_prefix=KEY)
    assert isinstance(out, pd.DataFrame)
    assert {"BrainRegion", "ShortName", "trace_id"} <= set(out.columns)


def test_one_entry_per_neuron():
    df = pd.concat([neuronTrials(trace_id=7), neuronTrials(trace_id=9)])
    assert len(loopNeuronsPlot(df, key_prefix=KEY, plot=False,
                               save_figs=False)) == 2


def test_neurons_from_both_regions_are_walked():
    df = pd.concat([neuronTrials(br=MFC, session="mfc"),
                    neuronTrials(br=LFC, session="lfc")])
    out = loopNeuronsPlot(df, key_prefix=KEY, plot=False, save_figs=False)
    assert {ids["BrainRegion"] for ids, _ in out} == {MFC, LFC}


def test_a_peak_that_tracks_the_trial_correlates_positively():
    out = loopNeuronsPlot(neuronTrials(peak_moves=True), key_prefix=KEY,
                          plot=False, save_figs=False)
    _, results = out[0]
    corr = [v for k, v in _flatten(results).items()
            if "firing_pos" in k and "pearson_corr" in k]
    assert corr and all(c > 0.9 for c in corr)


def test_a_fixed_peak_does_not_correlate():
    out = loopNeuronsPlot(neuronTrials(peak_moves=False), key_prefix=KEY,
                          plot=False, save_figs=False)
    _, results = out[0]
    corr = [v for k, v in _flatten(results).items()
            if "firing_pos" in k and "pearson_corr" in k and not np.isnan(v)]
    assert all(abs(c) < 0.5 for c in corr)


def _flatten(obj, prefix=""):
    """The results are nested dicts; makeDF flattens them the same way."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}_"))
    else:
        out[prefix[:-1]] = obj
    return out


def test_shuffling_refuses_to_draw():
    """A shuffled pass is a control, not a panel; drawing it would be wrong."""
    with pytest.raises(AssertionError):
        loopNeuronsPlot(neuronTrials(), key_prefix=KEY, plot=True,
                        save_figs=False, shuffle_corr=True)


def test_saving_needs_a_prefix():
    with pytest.raises(AssertionError, match="fig_save_prefix"):
        loopNeuronsPlot(neuronTrials(), key_prefix=KEY, plot=False,
                        save_figs=True)


def test_the_summary_has_a_row_per_trial():
    out = loopNeuronsSummary(neuronTrials(n_trials=12), key_prefix=KEY)
    assert len(out) == 12


def test_the_summary_ranks_amplitudes_within_the_neuron():
    out = summariseNeuronCorr(neuronTrials(n_trials=4), key_prefix=KEY)
    assert f"{KEY}max_amplitude_rank" in out.columns


# --- Figure 4K (left): one neuron's sampling-aligned panel on its own -------

from .. import rtcorrpanels as rp


def binnedTrials(session="s1", trace_id=7, br=MFC, n_per_dur=6,
                 durs=(0.5, 0.9, 1.3)):
    """Trials whose trace length follows their duration, so the panel bins
    them: ``n_per_dur`` trials at each duration."""
    rows = []
    rng = np.random.default_rng(1)
    for dur in durs:
        n = int(round((dur + rp.TIME_BEFORE_SAMPLING + rp.TIME_AFTER_MOVEMENT)
                      * ACQ_RATE))
        for _ in range(n_per_dur):
            rows.append({"BrainRegion": br, "ShortName": session,
                         "TrialNumber": len(rows), "trial_dur_sec": dur,
                         "trace_id": trace_id, "is_active": True,
                         KEY + "max_pos_sec": 0.2, KEY + "auc": 1.0,
                         KEY + "trace": rng.normal(size=n)})
    return pd.DataFrame(rows)


def _solidLines(ax):
    return [(ln.get_xdata(), ln.get_ydata()) for ln in ax.get_lines()
            if ln.get_linestyle() == "-"]


def test_fast_trials_are_red_and_slow_ones_yellow():
    colors = rp.fastSlowColors(pd.Series([0.4, 1.0, 3.0]))
    assert len(colors) == 3
    assert colors[0][:3] == pytest.approx((1.0, 0.0, 0.0))
    assert colors[-1][:3] == pytest.approx((1.0, 1.0, 0.0))


def test_the_example_is_the_same_panel_the_diagnostic_figure_draws():
    df = binnedTrials()
    fig = rp.plotSamplingAlignedExample(df, "s1", 7, KEY, show=False)
    _, axs = plt.subplots(1, 4)
    rp._plotFilteredComb(axs, df, KEY, df.trial_dur_sec.max(),
                         rp.fastSlowColors(df.trial_dur_sec), "No Filter",
                         shuffle_corr=False)
    mine, theirs = _solidLines(fig.axes[0]), _solidLines(axs[1])
    assert len(mine) == len(theirs) == 3     # one mean trace per duration bin
    for (x1, y1), (x2, y2) in zip(mine, theirs):
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)


def test_the_example_has_a_solid_x_axis_and_no_dashed_zero_line():
    fig = rp.plotSamplingAlignedExample(binnedTrials(), "s1", 7, KEY,
                                        show=False)
    ax, = fig.axes
    assert ax.spines["bottom"].get_visible()
    assert ax.spines["bottom"].get_position() == ("data", 0)
    dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]
    # Only the vertical sampling-onset line stays dashed.
    assert len(dashed) == 1 and np.allclose(dashed[0].get_xdata(), 0)
    assert ax.get_title() == "s1 — neuron 7"


def test_the_example_saves_under_its_region(tmp_path):
    df = pd.concat([binnedTrials(br=LFC, session="sL", trace_id=3),
                    binnedTrials()])
    rp.plotSamplingAlignedExample(df, "sL", 3, KEY, save_figs=True,
                                  fig_save_prefix=tmp_path, show=False)
    expected = (tmp_path / "SingleTraces" / "rt_corr_examples" / "LFC" /
                "sL_3_sampling_aligned.svg")
    assert expected.is_file()
    assert rp.samplingAlignedSavePath(tmp_path, LFC, "sL", 3) == expected


def test_the_example_writes_nothing_without_save_figs(tmp_path):
    rp.plotSamplingAlignedExample(binnedTrials(), "s1", 7, KEY,
                                  fig_save_prefix=tmp_path, show=False)
    assert not any(tmp_path.iterdir())


def test_a_missing_example_neuron_raises():
    with pytest.raises(KeyError):
        rp.plotSamplingAlignedExample(binnedTrials(), "s1", 8, KEY, show=False)


def test_the_published_examples():
    assert rp.EXAMPLES_4K_LEFT == (("Rigid", "GP4_28_S1_L41_D250_M2L", 237),
                                   ("Stretching", "GP4_85_s10_L70_D250_ALM", 25))
