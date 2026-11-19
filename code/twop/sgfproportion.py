'''Share of choice-selective neurons along normalised sampling time (Figure S12B).

Backend for the "Share of choice-selective neurons along sampling — Figure S12B"
section of ``2p_2_sequences.ipynb``.

Every neuron is placed in a time bin by its median peak-firing sample
(:func:`twop.movementneurons.maxFiringHist`, 10 bins). Per region, each bin's
value is the percentage of that bin's neurons that are choice-selective
(``sgf_neurons_ids``, from :func:`twop.movementneurons.getSgfNeurons`). A bin
no neuron of the region falls in is NaN, and the stairs leave a gap there.
'''
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common.clr import BrainRegion as BRClr
from ..common.definitions import BrainRegion
from .fastslowneurontraces import regionName
from .movementneurons import maxFiringHist
from .tracereliability import setupEpochAxes

NUM_BINS = 10
SAVE_DIR = "sgf_proportion"
SAVE_NAME = "active_proportion_sgf.svg"
#: Narrow and tall, like the published panel.
FIG_SIZE = (5, 6)


def sgfProportionPerBin(df, sgf_neurons_ids, epoch_ranges, num_bins=NUM_BINS):
    '''``(proportion, bins)``: % of each bin's neurons that are in ``sgf_neurons_ids``.

    ``df`` is one region's ``max_firing_*_df``; ``proportion`` has one value per
    bin (``len(bins) - 1``), earliest first.
    '''
    hist_all, bins = maxFiringHist(df, num_bins=num_bins, epoch_ranges=epoch_ranges)
    is_sgf = df.trace_id.isin(sgf_neurons_ids).to_numpy()
    # bin_val is pd.cut's categorical: counting by it keeps the empty bins.
    all_counts = hist_all.bin_val.value_counts(sort=False)
    sgf_counts = hist_all.bin_val[is_sgf].value_counts(sort=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        proportion = 100 * sgf_counts.to_numpy() / all_counts.to_numpy()
    proportion = np.where(all_counts.to_numpy() == 0, np.nan, proportion)
    return proportion, bins


def plotActiveDistSgfProportion(df, sgf_neurons_ids, epochs_row, save_figs,
                                fig_save_prefix=None):
    '''Figure S12B: one stairs curve per region on a shared axes.

    ``epochs_row`` is a row of the sampling-normalised frame; its
    ``epochs_ranges`` set the bins and its epoch names label the x axis.
    '''
    fig, ax = plt.subplots(1, figsize=FIG_SIZE)
    for br, br_df in df.groupby("BrainRegion"):
        br = BrainRegion(br)
        br_str = regionName(br)
        proportion, bins = sgfProportionPerBin(br_df, sgf_neurons_ids,
                                               epochs_row.epochs_ranges)
        ax.stairs(proportion, bins, baseline=None, color=BRClr[br],
                  label=f"{br_str} Sgf./All %")
        print(br_str, "All neurons:", len(br_df),
              "- Sgf:", br_df.trace_id.isin(sgf_neurons_ids).sum())

    ax.set_title("Proportion of sgf. neurons/all neurons per bin")
    setupEpochAxes([ax], epochs_row)
    ax.set_ylabel("Proportion %")
    ax.set_xlabel("Normalized Time (s)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    if save_figs:
        assert fig_save_prefix is not None, "Pass fig_save_prefix to save"
        out_dir = os.path.join(fig_save_prefix, SAVE_DIR)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, SAVE_NAME))
    plt.show()
    return fig
