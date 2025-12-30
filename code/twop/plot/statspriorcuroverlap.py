from ...common.clr import BrainRegion as BRClr
from ...common.definitions import BrainRegion
from .plotutil import applyFunOnEpoch
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

def plotPriorsCurOVeral(*applyFunOnEpochArgs, **applyFunOnEpochKwargs):
    applyFunOnEpoch(_plotPriorsCurBetweenRegions, *applyFunOnEpochArgs,
                    **applyFunOnEpochKwargs)

def _plotPriorsCurBetweenRegions(df, epoch, pval, fig_save_prefix,
                                 active_traces_df=None,
                                 total_neurons_count_df=None):
    df = df[df.prior_data_col.isna()]
    org_df = df.copy()
    if active_traces_df is not None:
        df = df[df.trace_id.isin(active_traces_df.trace_id)]
    df = df[df.pval <= pval]
    data_cols_vals = df.data_col.unique()
    prev_conds = [col for col in data_cols_vals if col.startswith("Prev")]
    cur_conds = [col for col in data_cols_vals if col not in prev_conds]

    def dropDuplicates(df):
        return df.drop_duplicates(subset=["trace_id", "ShortName"])

    labels = []
    clrs = []
    ys_means = []
    uniq_brain_regions = df.BrainRegion.unique()
    for br in uniq_brain_regions:
        layer_br_df = df[df.BrainRegion == br]
        assert layer_br_df.Layer.nunique() == 1
        ys = []
        for sess, sess_df in layer_br_df.groupby("ShortName"):
            prior_df = sess_df[sess_df.data_col.isin(prev_conds)]
            prior_df = dropDuplicates(prior_df)
            cur_df = sess_df[sess_df.data_col.isin(cur_conds)]
            cur_df = dropDuplicates(cur_df)
            # print("Len prior_df:", len(prior_df), "Len cur_df:", len(cur_df),
            #             "sess:", sess)
            USE_ACTIVE_TRACES = True
            if USE_ACTIVE_TRACES:
                all_sess_traces = org_df[
                                     org_df.ShortName == sess].trace_id.unique()
                cur_active_traces = active_traces_df[
                                active_traces_df.trace_id.isin(all_sess_traces)]
                # display(active_traces_df)
                assert len(cur_active_traces)
                all_uniq = cur_active_traces.trace_id
                assert len(all_uniq) == all_uniq.nunique()
            else:
                all_uniq = set(prior_df.trace_id.unique()) | \
                           set(cur_df.trace_id.unique())
            overlap = set(prior_df.trace_id.unique()) & \
                      set(cur_df.trace_id.unique())
            # print("Overlap:", len(overlap), "All:", len(all_uniq),
            #       "sess:", sess)
            overlap_prcnt = 100*len(overlap)/len(all_uniq)
            ys.append(overlap_prcnt)
        ys_means.append(ys)
        br_clr = BRClr[BrainRegion(br)]
        clrs.append(br_clr)
        br_str = str(BrainRegion(br)).split("_")[0]
        labels.append(br_str)

    data_means = [np.nanmean(ys) for ys in ys_means]
    data_sem = [sem(ys) for ys in ys_means]
    # print("data_means:", data_means)
    fig, ax = plt.subplots(figsize=(15, 10))
    VERT = True
    POS_SCALING = 1.2
    pos = np.arange(len(labels))*POS_SCALING
    if VERT:
        ax.bar(pos, data_means, yerr=data_sem, color=clrs)
        ax.set_xlim(-1.5, len(labels)+5)
    else:
        ax.barh(pos, data_means, xerr=data_sem, color=clrs)
        ax.set_ylim(-1.5, len(labels)+5)

    ax.set_title(f"Overlap between priors and current for {epoch}")
    msg = "Overlap %"
    ax.set_ylabel(msg) if VERT else ax.set_xlabel(msg)
    ax.set_xticks(np.arange(len(labels))*POS_SCALING)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    if fig_save_prefix is not None:
        out_of = "epoch_active" if USE_ACTIVE_TRACES else "sigf_traces"
        save_path = Path(f"{fig_save_prefix}/"
                         f"prior_cur_overlap_prcnt_of_{out_of}_"
                         f"{epoch.replace('/', '_')}.jpeg")
        print(f"Saving to {save_path}")
        save_path.parent.mkdir(exist_ok=True, parents=False)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
