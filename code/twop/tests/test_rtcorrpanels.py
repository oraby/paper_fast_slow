'''Tests for the single-neuron correlation panels (Figures 4K left, S10A).

Reproduced while extracting: the same correlation table as the notebook cell,
and the same panels.

The point of these tests is the pair of loops the notebook gave the same name.
``loopNeuronsPlot`` returns a **list of (ids, results) pairs**, which is what
``makeDF`` and the shuffled control expect; ``loopNeuronsSummary`` returns a
**DataFrame**. The second used to overwrite the first, so anything later
reaching for the list quietly received a frame instead.
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
    """The two are different shapes; the notebook gave them one name."""
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
