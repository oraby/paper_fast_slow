
from ...common.clr import BrainRegion as BRClr
from ...common.definitions import BrainRegion
from .plotutil import applyFunOnEpochPart, getEpochTotalCountDf
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

def plotEarlyLateSmplingDist(df, data_col, pval, active_traces_df,
                             fig_save_prefix, neurouns_total_count_df=None):
    epoch = df.epoch.unique()
    assert len(epoch) == 1
    epoch = epoch[0]
    epoch_total_counts_df = None if neurouns_total_count_df is None else \
                            getEpochTotalCountDf(df, epoch,
                                                 neurouns_total_count_df)
    epoch_active_traces_df = active_traces_df[active_traces_df.epoch == epoch]
    total_parts = epoch_active_traces_df.total_parts.unique()
    assert len(total_parts) == 1, f"Found mix of total_parts: {total_parts}"
    total_parts = total_parts[0]
    assert total_parts > 1, "Expected multiple parts"
    brain_regions = df.BrainRegion.unique()
    layers = df.Layer.unique()
    assert len(layers) == 1, "Not implemented for multiple layers"

    epochs_labels = []
    res_dict = {br:[] for br in brain_regions}
    USE_ACTIVE_TRACES = False

    def applyPart(part_df, epoch, pval, fig_save_prefix,
                  active_traces_df,    total_neurons_count_df):
        part_active_traces_df = active_traces_df
        part_epoch_total_counts_df = total_neurons_count_df
        del active_traces_df # Avoid name confusion
        del total_neurons_count_df # Avoid name confusion
        nonlocal res_dict, brain_regions, epochs_labels
        part_df = part_df[part_df.data_col == data_col]
        part_df = part_df[part_df.prior_data_col.isna()]
        assert len(part_df) == len(part_df.trace_id.unique())
        for br in brain_regions:
            ys = []
            br_df = part_df[part_df.BrainRegion == br]
            assert len(br_df) == len(br_df.trace_id.unique())
            for sess, sess_df in br_df.groupby("ShortName"):
                sess_active_traces = sess_df[
                          sess_df.trace_id.isin(part_active_traces_df.trace_id)]
                if not len(sess_active_traces):
                    print("No active traces for sess:", sess, "in epoch:",
                          epoch, "Sess traces:", len(sess_df))
                    if USE_ACTIVE_TRACES: # Avoid division by zero
                        continue
                sess_all_traces = sess_df.trace_id
                assert len(sess_all_traces) == len(sess_all_traces.unique())
                if part_epoch_total_counts_df is not None:
                    sess_total_neurons = part_epoch_total_counts_df[
                                   part_epoch_total_counts_df.ShortName == sess]
                    sess_total_neurons = \
                                        sess_total_neurons.total_neurons.iloc[0]
                    assert sess_total_neurons == len(sess_all_traces), (
                         f"{sess_total_neurons = } != {len(sess_all_traces) =}")
                sess_sgf_traces = sess_df[sess_df.pval <= pval].trace_id
                sgf_active_traces = sess_sgf_traces[
                              sess_sgf_traces.isin(sess_active_traces.trace_id)]
                if USE_ACTIVE_TRACES:
                    prcnt = 100*len(sgf_active_traces)/len(sess_active_traces)
                else:
                    prcnt = 100*len(sgf_active_traces)/len(sess_all_traces)
                ys.append(prcnt)
            res_dict[br].append(ys)
        epochs_labels.append(epoch)

    applyFunOnEpochPart(applyPart, epoch_df=df, epoch=epoch,
                        epoch_active_traces_df=epoch_active_traces_df,
                        pval=pval, fig_save_prefix=fig_save_prefix,
                        epoch_total_counts_df=epoch_total_counts_df)

    fig, ax = plt.subplots(figsize=(15, 10))
    offset = -0.01
    for br in brain_regions:
        br_res = res_dict[br]
        br_str = str(BrainRegion(br)).split("_")[0]
        br_clr = BRClr[BrainRegion(br)]
        phase_mean = [np.nanmean(ys) for ys in br_res]
        print("br:", br_str, "phase_mean:", phase_mean, "for data_col:",
              data_col)
        phase_sem = [sem(ys) for ys in br_res]
        x = np.arange(len(phase_mean))
        ls = "--" if "Prev" in data_col else "-"
        ax.errorbar(x=x + offset, y=phase_mean, yerr=phase_sem, label=br_str,
                    color=br_clr, ls=ls)
        offset += 0.01
    ax.set_xticks(x)
    ax.set_xticklabels(epochs_labels)
    # ax.set_xlim(-0.5)
    ax.set_ylabel("% Selective Neurons")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    # ax.legend(loc="upper center")
    if "ChoiceLeft" in data_col:
        data_col = data_col.replace("ChoiceLeft", "Trial Choice Direction")
    if "Prev" in data_col:
        data_col = data_col.replace("Prev", "Previous")
    else:
        data_col = "Current " + data_col

    ax.set_title(f"Change in selectivity for {data_col} between\n"
                  "Early and Late Sampling")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if fig_save_prefix is not None:
        fig.savefig(f"{fig_save_prefix}/early_late_sampling_{data_col}.jpeg",
                    dpi=300, bbox_inches="tight")
    plt.show()
