'''Tests for the choice-selective share per time bin (Figure S12B).

Pinned here: a bin's value is the share of *that bin's* neurons that are
choice-selective, an empty bin is NaN rather than shifting the bins after it,
and the figure saves only when asked.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ...common.definitions import BrainRegion
from ..sgfproportion import (NUM_BINS, SAVE_DIR, SAVE_NAME,
                             plotActiveDistSgfProportion, sgfProportionPerBin)

MFC = int(BrainRegion.M2_Bi)
LFC = int(BrainRegion.ALM_Bi)
#: before, sampling, after: binEdges puts 8 equal bins across 10..90.
EPOCH_RANGES = [(0, 10), (10, 90), (90, 100)]


def neurons(peaks, br=MFC, prefix="n"):
    """One neuron per peak; every trial of a neuron peaks at the same sample."""
    return pd.DataFrame({"trace_id": [f"{prefix}{i}" for i in range(len(peaks))],
                         "BrainRegion": br,
                         "max_idxs": [[p, p, p] for p in peaks]})


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_share_is_out_of_the_bins_own_neurons():
    # Bin 0 (0..10): 4 neurons, 1 selective. Bin 1 (10..20): 2 neurons, both.
    df = neurons([1, 2, 3, 4, 12, 13])
    proportion, bins = sgfProportionPerBin(df, {"n0", "n4", "n5"}, EPOCH_RANGES)
    assert len(proportion) == len(bins) - 1 == NUM_BINS
    assert proportion[0] == pytest.approx(25.0)
    assert proportion[1] == pytest.approx(100.0)


def test_bin_with_no_selective_neuron_is_zero():
    df = neurons([1, 12])
    proportion, _ = sgfProportionPerBin(df, {"n1"}, EPOCH_RANGES)
    assert proportion[0] == 0.0
    assert proportion[1] == 100.0


def test_empty_bin_is_nan_and_later_bins_keep_their_place():
    # Nothing in bin 1; the neuron at 25 belongs to bin 2 and must stay there.
    df = neurons([1, 25])
    proportion, bins = sgfProportionPerBin(df, {"n1"}, EPOCH_RANGES)
    assert bins[2] <= 25 < bins[3]
    assert proportion[0] == 0.0
    assert np.isnan(proportion[1])
    assert proportion[2] == 100.0


def test_ids_outside_the_frame_are_ignored():
    df = neurons([1, 2])
    proportion, _ = sgfProportionPerBin(df, {"n0", "other_session_7"}, EPOCH_RANGES)
    assert proportion[0] == pytest.approx(50.0)


def _epochs_row():
    return pd.Series({"epochs_ranges": EPOCH_RANGES,
                      "epochs_names": ["Before", "Sampling", "Movement"]})


def test_plot_draws_one_curve_per_region():
    df = pd.concat([neurons([1, 12]), neurons([2, 50], br=LFC, prefix="l")])
    fig = plotActiveDistSgfProportion(df, {"n0", "l1"}, _epochs_row(),
                                      save_figs=False)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ["LFC Sgf./All %", "MFC Sgf./All %"]


def test_saves_only_when_asked(tmp_path):
    df = neurons([1, 12])
    plotActiveDistSgfProportion(df, {"n0"}, _epochs_row(), save_figs=False,
                                fig_save_prefix=str(tmp_path))
    assert not (tmp_path / SAVE_DIR).exists()
    plotActiveDistSgfProportion(df, {"n0"}, _epochs_row(), save_figs=True,
                                fig_save_prefix=str(tmp_path))
    assert (tmp_path / SAVE_DIR / SAVE_NAME).is_file()
