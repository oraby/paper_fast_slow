'''Rigid / stretching neuron proportions (Figure 4K right, and its bars).

Backend for the "Pie-Chart for rigid/streteching neurons" section of
``plottraces3.ipynb``.

A neuron counts as **stretching** when its peak time correlates with sampling
duration past ``corr_thresh`` (|r| > 0.3 in the paper), and as **rigid** when
it does not; the AUC correlation is treated the same way, so a neuron can be
in one group, both, or neither.

- ``plotCorrThreshPieChart`` (Figure 4K right) -- one pie per region plus a
  pooled one, over the four groups, saved per region.
- ``plotCorrThreshRealVsShuffled`` -- the same proportions as bars, real
  against the shuffled control, with a pie beside them. The notebook called
  this one ``plotCorrThreshPieChart`` too, redefining the name in the next
  cell; the two take different arguments and draw different panels, so they
  are named apart here.

The region bars at the same threshold live in
:mod:`..twop.plot.corrthreshregions` (Figure S10D), which was extracted
earlier.

Extracted from the notebook unchanged, except that ``CORR_THRESH``,
``SAVE_FIGS`` and ``fig_save_prefix`` are parameters.
'''
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..common.clr import BrainRegion as BRClr
from ..common.definitions import BrainRegion

def _processSet(br_str, br_df, filter_key, corr_thresh, save_figs, fig_save_prefix):
    firing_pos_corr_key = filter_key + "firing_pos_pearson_corr"
    auc_key_corr_key = filter_key + "amplitude_firing_pearson_corr"

    fir_pos_traces_ids = set(br_df[br_df[firing_pos_corr_key].abs() >
                             corr_thresh].long_trace_id)
    auc_traces_ids = set(br_df[br_df[auc_key_corr_key].abs() >
                        corr_thresh].long_trace_id)
    both_traces_ids = fir_pos_traces_ids.intersection(auc_traces_ids)
    fir_pos_only_traces_ids = fir_pos_traces_ids.difference(both_traces_ids)
    auc_only_traces_ids = auc_traces_ids.difference(both_traces_ids)

    # "Rigid" = correlated with neither metric, so subtract the *union* of the
    # two sets (not just Both, which left every single-metric neuron in Rigid
    # and made the wedges sum past 100%).
    non_traces_ids = set(br_df.long_trace_id).difference(
                         fir_pos_traces_ids.union(auc_traces_ids))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([len(fir_pos_only_traces_ids), len(both_traces_ids),
            len(auc_only_traces_ids), len(non_traces_ids)],
            labels=["Firing Pos", "Both", "AUC", "Rigid"],
            colors=["C1", "olive", "C5", "gray"],
            startangle=90, counterclock=False,
            hatch=["", "xx", "", ""],
            pctdistance=1.37,
            autopct="%1.1f%%")
    ax.set_title(f"{br_str} Correlation with Sampling Time\n"
                    f"Correlation Threshold = {corr_thresh}")
    if save_figs:
        assert fig_save_prefix is not None, "Pass fig_save_prefix to save"
        fig.savefig(f"{fig_save_prefix}/{br_str}_correlation_pie_chart.pdf",
                    bbox_inches='tight')
    plt.show()

def plotCorrThreshPieChart(df, filter_key, corr_thresh, save_figs, fig_save_prefix):
    df = df.copy()
    df["long_trace_id"] = df.ShortName + "_" + df.trace_id.astype(str)

    _processSet("Both_MFC_LFC", df, filter_key, corr_thresh, save_figs,
                fig_save_prefix)
    for br, br_df in df.groupby("BrainRegion"):
        br_ = BrainRegion(br)
        br_str = str(BrainRegion(br_)).split("_")[0]
        br_str = "MFC" if br_str == "M2" else ("LFC" if br_str == "ALM" else br_str)
        _processSet(br_str, br_df, filter_key, corr_thresh, save_figs, fig_save_prefix)


def plotCorrThreshRealVsShuffled(df, save_figs, corr_thresh,
                                 fig_save_prefix=None):
    # display(df)
    has_early_seq = any("early" == df["when"])
    print("Has early seq:", has_early_seq)
    for key, key_df in df.groupby("metric"):
        for when, when_df in key_df.groupby("when"):
            if "firing_pos" in key and "early" in when:
                continue # Doesn't make sense to plot firing pos for early seq, since early doesn't move
            fig, (top_row, bottom_row) = plt.subplots(2, 2, figsize=(10, 6))
            if has_early_seq:
                when_str = "for Early Seq" if "early" in when else "for Remaining Seq"
            else:
                when_str = ""
            if "amplitude" in key:
                key_str = "Amplitude"
                fig.suptitle(f"{key_str} correlation with Sampling Time {when_str}")
            elif "firing_pos" in key:
                key_str = "Firing Position"
                fig.suptitle(f"{key_str} correlation with Sampling Time {when_str}")
            top_row = iter(reversed(top_row))
            bottom_row = iter(reversed(bottom_row))
            for br, br_df in when_df.groupby("BrainRegion"):
                ax_pie = next(top_row)
                ax_bar = next(bottom_row)
                for sess, sess_df in br_df.groupby("ShortName"):
                    real_row = sess_df[~sess_df.is_shuffle]
                    shuffle_row = sess_df[sess_df.is_shuffle]
                    assert len(real_row) == 1
                    assert len(shuffle_row) == 1
                    ax_bar.plot([0.05, .95],
                                [real_row["mean"].values[0], shuffle_row["mean"].values[0]],
                                marker="o", color="gray", zorder=10, alpha=.5)

                for is_shuffle, is_shuffle_df in br_df.groupby("is_shuffle"):
                    mean_mean = is_shuffle_df["mean"].mean()
                    mean_sem = is_shuffle_df["mean"].sem()
                    ax_bar.bar(int(is_shuffle), mean_mean, yerr=mean_sem,
                               label=f"{br} {is_shuffle}", color=BRClr[br],
                               alpha=.5 if is_shuffle else 1)
                    if is_shuffle:
                        continue
                    # print("BR:", br, "Key:", key, "When:", when)
                    # display(is_shuffle_df)
                    # print("Mean mean:", mean_mean, "Mean sem:", mean_sem)
                    label = f"{mean_mean:.2f}% ±{mean_sem:.2f}%"
                    ax_pie.pie([100-mean_mean, mean_mean], labels=["", label],
                           startangle=90,
                           colors=["gray", BRClr[br]])

                ax_bar.spines[["top", "right"]].set_visible(False)
                ax_bar.set_xticks([0, 1])
                ax_bar.set_xticklabels(["Real", "Shuffled"])
                ax_bar.set_ylabel(f"Above {corr_thresh} Correlation (%)")
            if save_figs:
                assert fig_save_prefix is not None, "Pass fig_save_prefix to save"
                save_fp = f"{fig_save_prefix}/rt_corr_{key_str}_{when_str}_above_{corr_thresh}.svg"
                print("Save fp=", save_fp)
                fig.savefig(save_fp, bbox_inches="tight")
            plt.show()
