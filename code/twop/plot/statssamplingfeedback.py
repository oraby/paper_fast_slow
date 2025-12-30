from ...common.clr import BrainRegion as BRClr
from ...common.definitions import BrainRegion
try:
    from IPython.display import display
except ModuleNotFoundError:
    pass
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

def plotSamplingFeedbackDist(df, data_col_li, pval, active_traces_df,
                             fig_save_prefix, neurouns_total_count_df=None,
                             limit_to_brain_regions=None):
    # Combine active traces parts for early and late sampling
    sampling_active = active_traces_df[active_traces_df.epoch == "Sampling"]
    sampling_active = sampling_active.drop_duplicates(subset=["trace_id"])
    feedback_active = active_traces_df[active_traces_df.epoch == "Feedback"]

    df = df[df.data_col.isin(data_col_li)]
    df = df[df.prior_data_col.isna()]
    if limit_to_brain_regions is not None:
        df = df[df.BrainRegion == limit_to_brain_regions]
    else:
        assert False, "Not implemented for multiple brain regions"

    USE_ACTIVE_TRACES = False
    epochs = "Sampling", "Feedback"
    # res_data_cols = {col:[] for col in data_col_li}
    res_data_dict = {"data_col":[], "epoch":[], "ShortName":[], "prcnt":[]}
    for epoch in epochs:
        part_active_traces_df = sampling_active if epoch == "Sampling" else \
                                feedback_active
        epoch_df = df[df.epoch == epoch]
        # Sub-optimum but easier to argue about
        for col in data_col_li:
            ys = []
            for sess, sess_df in epoch_df.groupby("ShortName"):
                sess_df = sess_df[sess_df.data_col == col]
                sess_active_traces = sess_df[
                          sess_df.trace_id.isin(part_active_traces_df.trace_id)]
                if not len(sess_active_traces):
                    print("No active traces for sess:", sess, "in epoch:",
                          epoch, "Sess traces:", len(sess_df))
                    if USE_ACTIVE_TRACES: # Avoid division by zero
                        continue
                sess_all_traces = sess_df.trace_id
                assert len(sess_all_traces) == len(sess_all_traces.unique())
                if neurouns_total_count_df is not None:
                    sess_total_neurons = neurouns_total_count_df[
                                      neurouns_total_count_df.ShortName == sess]
                    sess_total_neurons = \
                                        sess_total_neurons.total_neurons.iloc[0]
                    assert sess_total_neurons == len(sess_all_traces), (
                         f"{sess_total_neurons = } != {len(sess_all_traces) =}")
                sess_sgf_traces = sess_df[sess_df.pval <= pval].trace_id
                sgf_active_traces = sess_sgf_traces[
                              sess_sgf_traces.isin(sess_active_traces.trace_id)]
                if USE_ACTIVE_TRACES:
                    # prcnt = 100*len(sgf_active_traces)/len(sess_active_traces)
                    prcnt = len(sgf_active_traces)
                else:
                    # prcnt = 100*len(sgf_active_traces)/len(sess_all_traces)
                    prcnt = len(sgf_active_traces)
                # ys.append(prcnt)
                res_data_dict["data_col"].append(col)
                res_data_dict["epoch"].append(epoch)
                res_data_dict["ShortName"].append(sess)
                res_data_dict["prcnt"].append(prcnt)
            # res_data_cols[col].append(ys)
    # print("res_data_cols:", res_data_cols)
    res_df = pd.DataFrame(res_data_dict)
    # display(res_df)
    new_df_li = []
    for sess, sess_df in res_df.groupby("ShortName"):
        max_prcnt = sess_df.prcnt.max()
        sess_df["prcnt_normed"] = sess_df["prcnt"]/max_prcnt
        new_df_li.append(sess_df)
    res_df = pd.concat(new_df_li)
    display(res_df[(res_df.data_col == "ChoiceLeft") &
                   (res_df.epoch == "Feedback")])
    br_str = str(BrainRegion(limit_to_brain_regions)).split("_")[0]
    br_clr = BRClr[BrainRegion(limit_to_brain_regions)]

    xs = np.arange(len(epochs))
    def geColdata(col, single_sess=False):
        col_df = res_df[res_df.data_col == col]
        if single_sess:
            col_df = col_df[col_df.ShortName ==
                                             col_df.ShortName.sample(1).iloc[0]]
            prcnt_col = "prcnt"
        else:
            prcnt_col = "prcnt_normed"
        col_mean = [col_df[col_df.epoch == epoch][prcnt_col].mean()
                    for epoch in epochs]
        col_sem = [col_df[col_df.epoch == epoch][prcnt_col].sem()
                   for epoch in epochs]
        ls = "--" if "Prev" in col else "-"
        if "ChoiceLeft" in col:
            col = col.replace("ChoiceLeft", "Trial Choice Direction")
        if "Prev" in col: col = col.replace("Prev", "Previous ")
        else:             col = "Current " + col
        return col_mean, col_sem, ls, col

    def setAxis(ax, single_sess=False):
        ax.legend(loc='lower center')
        ax.set_xticks(xs)
        ax.set_xticklabels(epochs)
        norm_str = "Normalized " if not single_sess else ""
        ax.set_title(f"{norm_str}tuning across epochs in {br_str}")
        ax.set_ylabel(f"{norm_str}% of signficant neurons")
        if single_sess:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig, ax = plt.subplots(figsize=(15,10))
    offset = -0.01
    for col in data_col_li:
        col_mean, col_sem, ls, col = geColdata(col, single_sess=True)
        ax.errorbar(xs + offset, col_mean, yerr=col_sem, color=br_clr, ls=ls,
                    label=col)
        offset += 0.01
    setAxis(ax, single_sess=True)
    plt.show()

    fig, ax = plt.subplots(figsize=(15,10))
    offset = -0.01
    for col    in data_col_li:
        col_mean, col_sem, ls, col = geColdata(col)
        ax.errorbar(xs + offset, col_mean, yerr=col_sem, color=br_clr, ls=ls,
                    label=col)
        offset += 0.01
    setAxis(ax)
    if fig_save_prefix is not None:
        fig.savefig(f"{fig_save_prefix}/sampling_feedback_tuning_{br_str}.jpeg",
                    dpi=300, bbox_inches="tight")
    plt.show()
