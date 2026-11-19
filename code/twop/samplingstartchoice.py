'''Share of choice-selective neurons at sampling start, fast vs slow (Figure S12C).

Backend for the "Figure S12C" section of ``2p_3_choice.ipynb``.

The paper's choice test, the one behind ``data_runs_Unnormalized`` (each
trial's peak activity, ``DimReduc.max``, compared between left and right
choices with ``StatsTest.mannwhitneyu``), run separately on fast, medium and
slow trials and only on the first 0.1 s of sampling. The input is the paper's
z-scored real-time traces (``dataload.loadSamplingUnnormalized``), cut with
:func:`twop.epochcrop.epochStart`.

A bar is the mean over sessions of the percentage of a session's neurons that
are choice-selective (``pval <= PVAL``), +/- SEM across sessions.
'''
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common.clr import BrainRegion as BRClr
from ..common.clr import Strategy as StrategyClr
from ..common.definitions import BrainRegion
from .dimreduc import DimReduc
from .epochcrop import epochStart
from .fastslowstats import splitDatTestQuantiles
from .genrundata import runPreprocessor, runSplitFn
from .statstest import StatsTest

SAMPLING_EPOCH = "Sampling"
WINDOW_SEC = 0.1
#: (quantile_idx, label) of the strategies drawn.
STRATEGIES = [(1, "Fast"), (3, "Slow")]
#: Regions in drawing order, with the paper's names.
REGIONS = [(BrainRegion.M2_Bi, "MFC"), (BrainRegion.ALM_Bi, "LFC")]
SAVE_DIR = "sampling_start_choice"
SAVE_NAME = "s12c_choice_share.svg"


def windowSamples(sampling_df, window_sec=WINDOW_SEC):
    '''Samples in ``window_sec`` at the data's (median) acquisition rate.'''
    rate = float(sampling_df.acq_sampling_rate.median())
    return max(1, int(round(window_sec*rate)))


def samplingStartChoiceTests(sampling_df, window_sec=WINDOW_SEC):
    '''Per neuron and quantile, the choice test on sampling's first ``window_sec``.

    ``sampling_df`` is ``dataload.loadSamplingUnnormalized()``. Returns the
    test frame (one row per neuron, test and ``quantile_idx``) as
    ``genrundata.runSplitFn`` gives it. Takes a few minutes.
    '''
    df = epochStart(sampling_df, SAMPLING_EPOCH,
                    windowSamples(sampling_df, window_sec))
    df_reduc = runPreprocessor(df, dimReducFn=DimReduc.max)
    return runSplitFn(splitDatTestQuantiles, df_reduc,
                      StatsTest.mannwhitneyu)


def choiceShareBySession(tests_df, pval, data_col="ChoiceLeft"):
    '''Per region, quantile and session: neurons, choice-selective ones, %.'''
    df = tests_df[(tests_df.data_col == data_col)
                  & tests_df.prior_data_col.isnull()
                  & tests_df.DVstr.isnull()]
    df = df.assign(is_sgf=df.pval <= pval)
    res = df.groupby(["BrainRegion", "quantile_idx", "ShortName"]).is_sgf \
            .agg(n_neurons="size", n_sgf="sum").reset_index()
    res["prcnt"] = 100*res.n_sgf/res.n_neurons
    return res


def summarise(share_df):
    '''Per region and quantile: mean and SEM of the session %, and totals.'''
    return share_df.groupby(["BrainRegion", "quantile_idx"]).agg(
        mean=("prcnt", "mean"), sem=("prcnt", "sem"),
        n_sessions=("ShortName", "nunique"), n_neurons=("n_neurons", "sum"),
        n_sgf=("n_sgf", "sum")).reset_index()


def plotChoiceShare(share_df, save_figs, fig_save_prefix=None):
    '''Bars: fast vs slow, one per region, mean +/- SEM across sessions.

    Saves ``{fig_save_prefix}/sampling_start_choice/s12c_choice_share.svg``.
    Returns :func:`summarise`'s table.
    '''
    summary = summarise(share_df)
    fig, ax = plt.subplots(figsize=(4, 5))
    width = 0.8
    for s_idx, (quantile_idx, _) in enumerate(STRATEGIES):
        for r_idx, (br, br_name) in enumerate(REGIONS):
            row = summary[(summary.BrainRegion == br)
                          & (summary.quantile_idx == quantile_idx)]
            if not len(row):
                continue
            row = row.iloc[0]
            label = (f"{br_name} ({int(row.n_sessions)} sessions)"
                     if s_idx == 0 else None)
            ax.bar(s_idx*2.5 + r_idx*width, row["mean"], width=width,
                   yerr=row["sem"], capsize=4, color=BRClr[br],
                   edgecolor="k", label=label)
    ax.set_xticks([s_idx*2.5 + width/2 for s_idx in range(len(STRATEGIES))])
    ax.set_xticklabels([name for _, name in STRATEGIES])
    for tick, (_, name) in zip(ax.get_xticklabels(), STRATEGIES):
        tick.set_color(getattr(StrategyClr, name))
        tick.set_fontweight("bold")
    ax.set_ylabel("% of total neurons")
    ax.legend(frameon=False, fontsize="small")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    if save_figs:
        save_dir = os.path.join(fig_save_prefix, SAVE_DIR)
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, SAVE_NAME))
    plt.show()
    return summary
