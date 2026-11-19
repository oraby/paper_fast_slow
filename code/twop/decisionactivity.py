'''Population activity before the decision, fast vs slow trials (Figure S11B, left).

Backend for the "Figure S11B (left)" section of ``2p_1_population.ipynb``.

The input is the notebook's ``step3_movement_df``: every neuron's z-scored
real-time trace (``zscoredsampling.cutSamplingZScored``) aligned to movement
onset, the end of sampling (``alignsampling.getSamplingVario``), its trials
grouped by sampling duration into ``dur_bin`` bins. "Fast" is the shortest
bin, "slow" the longest. The traces are those of all the neurons.

A curve is :func:`twop.plot.activitysum.groupCurves`: per session, the mean
over neurons of each neuron's trial-mean (left- and right-choice trials
weighted equally), then the mean +/- SEM across sessions. Points only one
session has are dropped.
'''
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fastslowneurontraces import regionName
from .plot.activitysum import groupCurves, pruneSingleSession

MOVEMENT_EPOCH = "Movement to Lateral Port"
#: Time window drawn, in seconds from movement onset.
WINDOW_SEC = (-0.4, 0.1)
#: Regions in drawing order, with their colour maps.
REGION_CMAPS = [("MFC", "Blues"), ("LFC", "Reds")]
SAVE_DIR = "activity_sum"
SAVE_NAME = "s11b_decision_aligned_activity.svg"


def movementOnsetIdx(df):
    '''The sample index of movement onset, the same in every row.'''
    onsets = {r.epochs_ranges[list(r.epochs_names).index(MOVEMENT_EPOCH)][0]
              - r.trace_start_idx for r in df.itertuples(index=False)}
    if len(onsets) != 1:
        raise ValueError(f"Movement onset is not aligned across rows: {onsets}")
    return onsets.pop()


def fastSlowBins(df, bin_col="dur_bin"):
    '''``(fast, slow)``: the shortest and the longest of ``df``'s bins.'''
    bins = sorted(df[bin_col].unique())
    return bins[0], bins[-1]


def decisionCurves(movement_df, window_sec=WINDOW_SEC, bin_col="dur_bin"):
    '''Per region, the fast and slow curves within ``window_sec``.

    One row per region, strategy and time point: ``Region`` ("MFC"/"LFC"),
    ``Strategy`` ("Fast"/"Slow"), ``bin``, ``bin_rank`` (1 = shortest of the
    ``n_bins`` bins), ``time`` (s from movement onset), ``mean``, ``sem``,
    ``n_sessions`` and ``n_neurons``.
    '''
    rate = float(movement_df.acq_sampling_rate.median())
    bins = sorted(movement_df[bin_col].unique())
    fast, slow = fastSlowBins(movement_df, bin_col)
    rows = []
    for br, br_df in movement_df.groupby("BrainRegion"):
        if br_df.Layer.nunique() != 1:
            raise ValueError(f"Expected one layer per region, got "
                             f"{sorted(br_df.Layer.unique())}")
        br_df = br_df[br_df[bin_col].isin([fast, slow])]
        onset = movementOnsetIdx(br_df)
        for curve in groupCurves(br_df, bin_col, append_at="before"):
            mean = pruneSingleSession(curve["mean"], curve["sem"])
            time = (np.arange(len(mean)) - onset) / rate
            keep = (time >= window_sec[0] - 1e-9) & (time <= window_sec[1])
            rows.append(pd.DataFrame({
                "Region": regionName(br),
                "Strategy": "Fast" if curve["group"] == fast else "Slow",
                "bin": curve["group"],
                "bin_rank": bins.index(curve["group"]) + 1,
                "n_bins": len(bins),
                "time": time[keep], "mean": mean[keep],
                "sem": np.asarray(curve["sem"], dtype=float)[keep],
                "n_sessions": len(curve["sessions"]),
                "n_neurons": int(curve["num_traces"])}))
    return pd.concat(rows, ignore_index=True)


def plotDecisionActivity(curves_df, save_figs, fig_save_prefix=None,
                         window_sec=WINDOW_SEC):
    '''MFC and LFC, fast and slow, on one axes; the y axis on the right.

    Colours as the 2P duration-bin figures: a region's colour map, palest for
    the shortest bin. Saves
    ``{fig_save_prefix}/activity_sum/s11b_decision_aligned_activity.svg``.
    Returns the figure.
    '''
    fig, ax = plt.subplots(figsize=(4, 4))
    for region, cmap_name in REGION_CMAPS:
        cm = plt.get_cmap(cmap_name)
        region_df = curves_df[curves_df.Region == region]
        for strategy in ["Fast", "Slow"]:
            df = region_df[region_df.Strategy == strategy]
            if not len(df):
                continue
            c = cm(df.bin_rank.iloc[0] / df.n_bins.iloc[0])
            ax.plot(df.time, df["mean"], color=c,
                    label=f"{region} ({strategy})")
            ax.fill_between(df.time, df["mean"] - df["sem"], df["mean"] + df["sem"],
                            color=c, alpha=0.2, lw=0)
    ax.axvline(0, ls="--", color="gray")
    ax.set_xlim(window_sec)
    ax.set_xlabel("Decision time (s)")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Mean activity (z-scored)")
    for side in ["top", "left"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize="small", ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    if save_figs:
        assert fig_save_prefix is not None
        save_dir = os.path.join(fig_save_prefix, SAVE_DIR)
        os.makedirs(save_dir, exist_ok=True)
        fp = os.path.join(save_dir, SAVE_NAME)
        print("Saving to:", fp)
        fig.savefig(fp)
    plt.show()
    return fig
