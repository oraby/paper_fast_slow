'''Tests for the single-neuron fast/slow trace figures (Figure 4F).

What these pin: the active share in each panel title, the file a neuron is
saved to (the figure cell and the loop must agree), that a figure cell can ask
for just the published neuron, and that a neuron missing from the table raises
rather than leaving the figure silently undrawn.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ...common.definitions import BrainRegion
from .. import fastslowneurontraces as fs

MFC = int(BrainRegion.M2_Bi)
LFC = int(BrainRegion.ALM_Bi)


@pytest.fixture(autouse=True)
def closeFigures():
    yield
    plt.close("all")


def neurons(num_active, num_inactive, n_per_region=5):
    rows = []
    for br, sess in ((MFC, "sM"), (LFC, "sL")):
        for i in range(n_per_region):
            rows.append(dict(trace_id=f"{sess}_{i}", ShortName=sess,
                             BrainRegion=br,
                             whole_trace=[np.arange(33.)] * num_active,
                             invalid_whole_trace=[np.zeros(33)] * num_inactive))
    return pd.DataFrame(rows)


def inf():
    return pd.DataFrame([dict(ShortName=s, acq_sampling_rate=30.,
                              epochs_ranges=[(0, 2), (3, 29), (30, 32)],
                              epochs_names=["Before", "Sampling", "Movement"])
                         for s in ("sM", "sL")])


def test_the_titles_give_the_active_share():
    fig = fs.plotFastSlowNeuron(neurons(3, 1).iloc[0], neurons(1, 3).iloc[0],
                                inf(), "MFC", save_figs=False, show=False)
    assert [ax.get_title() for ax in fig.axes] == \
        ["Fast Trials - 75% Active", "Slow Trials - 25% Active"]
    assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()


def test_one_neuron_saves_under_its_region(tmp_path):
    drawn = fs.loopFastSlowNeurons(neurons(3, 1), neurons(1, 3), inf(),
                                   save_figs=True, fig_save_prefix=tmp_path,
                                   trace_ids=["sL_2"])
    assert drawn == ["sL_2"]
    assert [p.relative_to(tmp_path).as_posix()
            for p in tmp_path.rglob("*.svg")] == ["FastSlowTraces/LFC/sL_2.svg"]


def test_a_loop_that_does_not_save_draws_three_per_region():
    drawn = fs.loopFastSlowNeurons(neurons(3, 1), neurons(1, 3), inf(),
                                   save_figs=False, show=False)
    # Regions come in code order, LFC (ALM) before MFC (M2).
    assert drawn == ["sL_0", "sL_1", "sL_2", "sM_0", "sM_1", "sM_2"]


def test_a_saving_loop_draws_every_neuron(tmp_path):
    drawn = fs.loopFastSlowNeurons(neurons(3, 1), neurons(1, 3), inf(),
                                   save_figs=True, fig_save_prefix=tmp_path)
    assert len(drawn) == 10
    assert len(list(tmp_path.rglob("*.svg"))) == 10


def test_a_missing_neuron_raises():
    with pytest.raises(KeyError, match="sM_9"):
        fs.loopFastSlowNeurons(neurons(3, 1), neurons(1, 3), inf(),
                               save_figs=False, trace_ids=["sM_9"])


def test_region_names_follow_the_paper():
    assert fs.regionName(MFC) == "MFC"
    assert fs.regionName(LFC) == "LFC"
