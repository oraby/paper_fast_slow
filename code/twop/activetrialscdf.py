'''Trial-by-trial firing reliability, as a CDF (Figure S9B).

Backend for the "CDF for active trials per quantiles" section of
``2pAnalysis.ipynb``.

Each neuron gets a percentage per strategy: on how many of that strategy's
trials was it active. Sorting those percentages descending and plotting them
against the neuron's rank gives one curve per region per strategy — fast
(dotted) and slow (dashed), MFC and LFC — so a curve that stays high means
many neurons fire reliably.

**The denominator is the strategy's own trial count.** ``quantile_idxs_count``
carries the trials per strategy for that neuron's session; the "All" curve
divides by their sum instead. A neuron that was never active in a strategy
contributes no row for it, so it is missing from that curve rather than
sitting at zero.

**A synthetic 100% point is appended** before sorting, so the curve reaches the
top of the axis instead of stopping at the most reliable real neuron. It is one
extra row whose columns are all set to ``100``; only ``active_prcnt`` is read.

**The axis labels read the other way round from the data.** ``x`` is the
neuron's rank as a percentage of the group's neurons and ``y`` is the
percentage of trials it was active, while the labels say "Trials % where
neurons are active" on x and "% Neurons" on y. Preserved as it was — the
published panel carries these labels — see ``docs/repo-audit.md``.

Extracted from the notebook unchanged except that ``fig_save_prefix`` is a
parameter, and the per-strategy percentages are computed by a function of their
own so they can be checked without drawing.
'''
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

from ..common.clr import BrainRegion as BRClr
from ..common.definitions import BrainRegion

ALL = "All"
#: Sampling-time tertiles, by the names the legend uses.
FAST, SLOW = 1, 3
#: (quantile, linestyle, label), in drawing order.
CURVES = [(FAST, ":", "Impuslive"), (SLOW, "--", "Deliberate")]


def activeTrialPercentages(df):
    '''One row per (neuron, strategy): percent of that strategy's trials active.

    ``quantile_idx`` is the strategy, or ``"All"`` for the pooled row.
    '''
    rows = []
    for _, row in df.iterrows():
        active = pd.DataFrame({"active_quantile_idx": row.active_quantile_idx})
        total_counts = row.quantile_idxs_count
        rows.append({"trace_id": row.trace_id, "quantile_idx": ALL,
                     "active_prcnt": 100 * len(active) / sum(total_counts.values())})
        for quantile_idx, quantile_df in active.groupby("active_quantile_idx"):
            rows.append({"trace_id": row.trace_id, "quantile_idx": quantile_idx,
                         "active_prcnt": 100 * len(quantile_df)
                                         / total_counts[quantile_idx]})
    return pd.DataFrame(rows, columns=["trace_id", "quantile_idx", "active_prcnt"])


def cdfCurve(active_quantile_df, quantile_idx, num_total_neurons):
    '''The plotted series: percentages sorted descending, indexed by neuron rank.'''
    q_df = active_quantile_df[active_quantile_df.quantile_idx == quantile_idx].copy()
    # Reach the top of the axis rather than stopping at the best real neuron.
    q_df.loc[q_df.index.max() + 1] = 100
    active_prcnt = q_df.active_prcnt.sort_values(ascending=False).reset_index(drop=True)
    active_prcnt.index = 100 * active_prcnt.index / num_total_neurons
    return active_prcnt


def _processSubjectActiveDistCDF(df, clr, label, ax, plot_all):
    active_quantile_df = activeTrialPercentages(df)
    curves = ([(ALL, "solid", ALL)] if plot_all else []) + CURVES
    for quantile_idx, ls, dscrp in curves:
        active_prcnt = cdfCurve(active_quantile_df, quantile_idx, len(df))
        ax.plot(active_prcnt.index, active_prcnt, ls=ls, color=clr,
                label=f"{label} {dscrp}")

    ax.set_title("Neurons Trials Firing Reliabilizty CDF")
    [ax.spines[_dir].set_visible(False) for _dir in ["left", "right", "top"]]
    ax.set_xlabel("Trials % where neurons are active")
    ax.set_ylabel("% Neurons")
    ax.set_ylim(0, 100)
    ax.set_xlim(100, 0)
    ax.set_xscale('symlog')
    ax.legend(loc="upper left")


def plotActiveDistCDF(df, save_figs, by_sess, fig_save_prefix=None):
    '''Figure S9B: one axes for both regions, or one per session.'''
    if not by_sess:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for br, br_df in df.groupby("BrainRegion"):
        br = BrainRegion(br)
        br_str = str(br).split("_")[0]
        br_clr = BRClr[BrainRegion(br)]

        if not by_sess:
            _processSubjectActiveDistCDF(br_df, clr=br_clr, label=br_str, ax=ax,
                                         plot_all=False)
        else:
            for sess, sess_df in br_df.groupby("ShortName"):
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                label = f"{br_str} - {sess}"
                _processSubjectActiveDistCDF(sess_df, clr=br_clr, label=label,
                                             ax=ax, plot_all=True)
                ax.set_title(ax.get_title() + f" - {label}")
                if save_figs:
                    plt.savefig(_savePath(fig_save_prefix, f"{label}.svg"))
                plt.show()

    if not by_sess:
        if save_figs:
            plt.savefig(_savePath(fig_save_prefix, "_all.svg"))
        plt.show()


def _savePath(fig_save_prefix, name):
    """``active_cdf/`` is not in the repo, so the first save has to create it."""
    assert fig_save_prefix is not None, "Pass fig_save_prefix to save"
    out_dir = os.path.join(fig_save_prefix, "active_cdf")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, name)
