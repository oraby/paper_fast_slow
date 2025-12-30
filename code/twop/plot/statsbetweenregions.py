from .plotutil import applyFunOnEpoch
from ...common.clr import BrainRegion as BRClr
from ...common.definitions import BrainRegion
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from pathlib import Path

def plotOverallTUningAcrossRegions(*applyFunOnEpochArgs,
                                   **applyFunOnEpochKwargs):
    applyFunOnEpoch(_plotChartDistAcrossRegions, *applyFunOnEpochArgs,
                    **applyFunOnEpochKwargs)

def _plotChartDistAcrossRegions(df, epoch, pval, fig_save_prefix,
                                active_traces_df=None,
                                total_neurons_count_df=None):
    df = df.copy().reset_index(drop=True)
    if active_traces_df is not None:
        df = df[df.trace_id.isin(active_traces_df.trace_id)]
    def getSgfDF(df):
        sgf_df = df[df.pval <= pval]
        sgf_no_prior_df = sgf_df[sgf_df.prior_data_col.isna()]
        sgf_prior_df = sgf_df[~sgf_df.prior_data_col.isna()]
        non_sgf_no_prior_df = df[(df.pval > pval) & df.prior_data_col.isna()]
        assert len(sgf_no_prior_df) + len(sgf_prior_df) == len(sgf_df)
        return sgf_no_prior_df, non_sgf_no_prior_df
    sgf_no_prior_df, non_sgf_no_prior_df = getSgfDF(df)
    print("non_sgf_no_prior_df:", len(non_sgf_no_prior_df))
    data_cols_vals = sgf_no_prior_df.data_col.unique()
    # Plot by either Prev or Current
    prev_conds = [col for col in data_cols_vals if col.startswith("Prev")]
    cur_conds = [col for col in data_cols_vals if col not in prev_conds]
    for condition, cond_li in [("Prior", prev_conds), ("Current", cur_conds)]:
        print(f"Plotting {condition}={cond_li} conditions")
        condition = f"{condition} Trial"
        condition_df = sgf_no_prior_df[sgf_no_prior_df.data_col.isin(cond_li)]
        cond_non_sgf = non_sgf_no_prior_df[
                                     non_sgf_no_prior_df.data_col.isin(cond_li)]
        _plotConditionDistAcrossRegions(df=condition_df,
                                        condition_str=condition,
                                        epoch=epoch,
                                        fig_save_prefix=fig_save_prefix,
                                        non_sgf_no_prior_df=cond_non_sgf,
                                        total_neurons_count_df=
                                                         total_neurons_count_df)
    # Now plot by each condition
    renaming_dict = {"ChoiceCorrect": "Trial Outcome",
                     "ChoiceLeft": "Animal Direction",
                     "IsEasy": "Trial Difficulty"}
    for condition in data_cols_vals:
        condition_df = sgf_no_prior_df[sgf_no_prior_df.data_col == condition]
        cond_non_sgf = non_sgf_no_prior_df[
                                      non_sgf_no_prior_df.data_col == condition]
        renamed = renaming_dict[condition.rsplit("Prev")[-1]]
        prefix = "Previous" if "Prev" in condition else "Current"
        print("condition:", condition, "Is: ", prefix, renamed)
        condition_str = f"{prefix} {renamed}"
        _plotConditionDistAcrossRegions(df=condition_df,
                                        condition_str=condition_str,
                                        epoch=epoch,
                                        fig_save_prefix=fig_save_prefix,
                                        non_sgf_no_prior_df=cond_non_sgf,
                                        total_neurons_count_df=
                                                        total_neurons_count_df)


def _plotConditionDistAcrossRegions(df, condition_str, epoch,
                                    non_sgf_no_prior_df, fig_save_prefix,
                                    total_neurons_count_df=None):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title(f"Neurons selectivity for {condition_str} during {epoch}")
    x = 0
    def countTotalNeurons(df):
        uniq_sesss = df.ShortName.unique()
        uniq_sgf_count = len(
                           df.drop_duplicates(subset=["ShortName", "trace_id"]))
        if total_neurons_count_df is not None:
            sess_total_neurons = total_neurons_count_df[
                              total_neurons_count_df.ShortName.isin(uniq_sesss)]
            assert len(sess_total_neurons) == len(uniq_sesss), (
                        f"{sess_total_neurons.ShortName = } != {uniq_sesss = }")
            uniq_total_count = sess_total_neurons.total_neurons.sum()
        else:
            # Read from the non-sgf df and add to sgf
            _df = non_sgf_no_prior_df # Create a shorter name
            _df = _df[(_df.BrainRegion == br) & (_df.Layer == layer) &
                                (_df.ShortName.isin(uniq_sesss))]
            if sorted(df.data_col.unique()) != sorted(_df.data_col.unique()):
                print(f"{sorted(df.data_col.unique())} != "
                      f"{sorted(_df.data_col.unique())} "
                      f"for epoch: {epoch} in session(s): {uniq_sesss}")
            # We might get duplicated traces if it's across many conditions
            _df = _df.drop_duplicates(subset=["ShortName", "trace_id"])
            len_non_sgf = len(_df)
            uniq_total_count = len_non_sgf + len(df)
        return uniq_sgf_count, uniq_total_count
    BOXPLOT = True
    if BOXPLOT:
        xs_li = []
        ys_li = []
        clrs_li = []
    for (br, layer), br_layer_df in df.groupby(["BrainRegion", "Layer"]):
        br_layer_df_count = br_layer_df.trace_id.nunique()
        br_clr = BRClr[BrainRegion(br)]
        if BOXPLOT:
            prcnts_li = []
            for sess, sess_df in br_layer_df.groupby("ShortName"):
                uniq_sgf_count, uniq_total_count = countTotalNeurons(sess_df)
                prcnts_li.append(100*uniq_sgf_count/uniq_total_count)
            xs_li.append(x)
            ys_li.append(prcnts_li)
            clrs_li.append(br_clr)
            # print("Prcnts li", prcnts_li)
            y = np.mean(prcnts_li)
        else:
            uniq_sgf_count, uniq_total_count = countTotalNeurons(br_layer_df)
            # We might get duplicated traces if it's across many conditions
            print("Counts:", uniq_sgf_count, "/", uniq_total_count)
            assert uniq_sgf_count == br_layer_df_count
            y = 100*uniq_sgf_count/uniq_total_count
            ax.bar(x, y, color=br_clr, width=0.5)
        layer_str = "L2/3" if layer == "L23" else layer
        br_str = str(BrainRegion(br)).split("_")[0]
        # For now, don't print the layer
        text = f"{br_str}\n{y:.3g}%" # {layer_str}"
        ax.text(x, -0.03, text, ha="center", va="top", fontsize="large",
                transform=ax.get_xaxis_transform())
        x += 0.35
    if BOXPLOT:
        w = ax.boxplot(ys_li, positions=xs_li, widths=0.3,
                       patch_artist=True, #, showfliers=False,
                       medianprops=dict(color="black", linewidth=1.5,
                       linestyle="--"))
        # Fill box plot with the relevant colors
        for i, (box, clr) in enumerate(zip(w["boxes"], clrs_li)):
            box.set_facecolor(clr)
            # Set edge width
            box.set_linewidth(1.5)
            if "Prev" in condition_str:
                box.set_hatch("//")
                box.set_edgecolor((0, 0, 0, 0.8))
    ax.set_xlim(-0.5, x+2)
    if False and "Difficulty" in condition_str:
        max_y = 30 if "Current" in condition_str else 60
        if total_neurons_count_df is not None:
            max_y = int(max_y/2)
    else:
        max_y = 100 if total_neurons_count_df is None else 80
    ax.set_ylim(0, max_y)
    ax.set_ylabel("% sgf. neurons", fontsize="large")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_xticks([])
    ax.spines[['top', 'right']].set_visible(False)
    if fig_save_prefix is not None:
        out_of = "epoch_active" if total_neurons_count_df is None else \
                 "total_neurons"
        save_path = Path(f"{fig_save_prefix}/selectivity_{out_of}_"
                         f"{epoch.replace('/', '_')}_"
                         f"{condition_str.replace(' ', '_')}.jpeg")
        print(f"Saving to {save_path}")
        save_path.parent.mkdir(exist_ok=True, parents=False)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
