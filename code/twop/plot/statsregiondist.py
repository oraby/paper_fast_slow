from ...common.clr import BrainRegion as BRClr
from ...common.definitions import BrainRegion
try:
    from IPython.display import display
except ModuleNotFoundError:
    pass
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from pathlib import Path

def plotOverallTuning(df, pval, fig_save_prefix, left_tuning_col,
                      active_traces_df=None, neurouns_total_count_df=None):
    for epoch, epoch_df in    df.groupby(df.epoch):
        for br, br_df in epoch_df.groupby(epoch_df.BrainRegion):
            for layer, layer_df in br_df.groupby(br_df.Layer):
                _plotOVerallTuingEpochBRLayer(epoch_br_layer_df=layer_df,
                                              epoch=epoch, br=br, layer=layer,
                                              pval=pval,
                                              left_tuning_col=left_tuning_col,
                                              fig_save_prefix=fig_save_prefix,
                                              active_traces_df=active_traces_df,
                                              neurouns_total_count_df=
                                                        neurouns_total_count_df)

def _plotOVerallTuingEpochBRLayer(epoch_br_layer_df, epoch, br, layer, pval,
                                  left_tuning_col, fig_save_prefix,
                                  active_traces_df=None,
                                  neurouns_total_count_df=None):
    if neurouns_total_count_df is not None:
        cur_total_counts_df = neurouns_total_count_df[
                                   (neurouns_total_count_df.BrainRegion == br) &
                                   (neurouns_total_count_df.epoch == epoch) &
                                   (neurouns_total_count_df.Layer == layer)]
        assert epoch_br_layer_df.ShortName.nunique() == len(
                                                         cur_total_counts_df), (
            display(sorted(epoch_br_layer_df.ShortName.unique())) or
            display(sorted(cur_total_counts_df.ShortName.unique()))
        )
        br_epoch_total_neurons = cur_total_counts_df.total_neurons.sum()
    else:
        br_epoch_total_neurons = None

    if active_traces_df is not None:
        # TODO: Add layer also to active_traces_df
        br_epoch_active_traces_df = active_traces_df[
                                              (active_traces_df.epoch == epoch)]
        assert len(br_epoch_active_traces_df)
        for part, part_traces_df in br_epoch_active_traces_df.groupby("part"):
            assert len(br_epoch_active_traces_df)
            total_parts = part_traces_df.total_parts.iloc[0]
            epoch_name = epoch
            if total_parts > 1:
                epoch_name += f" {part}/{total_parts}"
            _pieChartOverallTuning(epoch_br_layer_df, epoch=epoch_name, br=br,
                                   layer=layer, pval=pval,
                                   left_tuning_col=left_tuning_col,
                                   fig_save_prefix=fig_save_prefix,
                                   active_traces_df=part_traces_df,
                                   total_neurons_count=br_epoch_total_neurons)
    else:
        _pieChartOverallTuning(epoch_br_layer_df, epoch=epoch, br=br,
                               layer=layer, pval=pval,
                               left_tuning_col=left_tuning_col,
                               fig_save_prefix=fig_save_prefix,
                               active_traces_df=None,
                               total_neurons_count=br_epoch_total_neurons)

def _pieChartOverallTuning(df, epoch, br, layer, pval,# epochs_traces,
                           left_tuning_col, fig_save_prefix,
                           active_traces_df=None, total_neurons_count=None):
    df = df.reset_index(drop=True)
    if active_traces_df is not None:
        df = df[df.trace_id.isin(active_traces_df.trace_id)]
    num_traces = df.trace_id.nunique()
    sgf_df = df[df.pval <= pval]
    # motor_neurons_idxs = sgf_df.trace_id.isin(epochs_traces["motor"])
    # # display(df[motor_neurons_idxs)])
    # sgf_df = sgf_df[~motor_neurons_idxs]
    # Both below have duplicates for under different {prior}_data_columns
    sgf_no_prior_df = sgf_df[sgf_df.prior_data_col.isna()]
    sgf_prior_df = sgf_df[~sgf_df.prior_data_col.isna()]
    assert len(sgf_no_prior_df) + len(sgf_prior_df) == len(sgf_df)
    prev_cols = sgf_df[sgf_df.data_col.str.startswith("Prev")].data_col.unique()
    cur_cols = sgf_df[~sgf_df.data_col.str.startswith("Prev")].data_col.unique()

    non_sgf_no_prior_df = df[(df.pval > pval) & df.prior_data_col.isna()]
    # data_col_sizes = df.groupby(df.trace_id).data_col.count()
    # assert data_col_sizes.nunique() == 1, (
    #                                  f"Found {data_col_sizes.value_counts()}")
    br_str = str(BrainRegion(br)).split("_")[0]
    layer_str = "L2/3" if layer == "L23" else layer
    print("prev_cols:", prev_cols)

    cur_sgf_no_prior_df = sgf_no_prior_df[
                                        sgf_no_prior_df.data_col.isin(cur_cols)]
    prev_sgf_no_prior_df = sgf_no_prior_df[
                                       sgf_no_prior_df.data_col.isin(prev_cols)]
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"No. sgf. traces {br_str} {layer_str} Tuning during {epoch}")

    conditions = prev_sgf_no_prior_df.data_cols.unique()
    for data_col_val, in conditions:
        cur_df = cur_sgf_no_prior_df[cur_sgf_no_prior_df.data_col ==
                                                                   data_col_val]
        prev_df = prev_sgf_no_prior_df[prev_sgf_no_prior_df.data_col ==
                                                                   data_col_val]
        non_sgf_data_col = non_sgf_no_prior_df[non_sgf_no_prior_df.data_col ==
                                                                  data_col_name]
        len_sgf_df = len(data_col_df)
        len_non_sgf_df = len(non_sgf_data_col)
        if total_neurons_count is not None:
            cur_total_neurons = total_neurons_count
        else:
            cur_total_neurons = len_non_sgf_df + len_sgf_df
        bottom = 0
        if sub_df.groupby("data_col").ngroups == 2:
            idx += 1
        for clr, tuning_df in (('b', data_col_df[data_col_df[left_tuning_col]]),
                               ('orange',
                                   data_col_df[~data_col_df[left_tuning_col]])):
            xs.append(idx)
            cur_y = 100*len(tuning_df)/cur_total_neurons
            ys.append(cur_y)
            bottoms.append(bottom)
            labels_li.append(None)
            colors.append(clr)
            bottom += cur_y
        # xs.append(idx)
        # ys.append(100*len_sgf_df/cur_total_neurons)
        # bottoms.append(bottom)
        # labels_li.append(None)
        labels_li[-1] = (f"{data_col_name}\n"
                         f"sgf. = {len_sgf_df}/{cur_total_neurons}\n"
                         f"ns. = {len_non_sgf_df}/{cur_total_neurons}")
    for ax_row, sub_df, desc in zip(axs,
                                    [cur_sgf_no_prior_df, prev_sgf_no_prior_df],
                                    ["Current", "Previous"]):
        # len_ref = len(ref_df)
        ax = ax_row[0]
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f"Num. (repeated) nuerons = {len(sub_df)} across alll "
                     f"{desc} conditions",  fontsize=10)
        xs = []
        ys = []
        colors = []
        bottoms = []
        labels_li = []

        # Create a total unique
        if len(sub_df):
            idx += 1
            sgf_traces_set = set(sub_df.trace_id)
            initial_non_sgf_neurons = set(non_sgf_no_prior_df.trace_id)
            non_sgf_neurons_set = initial_non_sgf_neurons - sgf_traces_set
            if total_neurons_count is None:
                all_neurons_set = sgf_traces_set | non_sgf_neurons_set
                all_neurons_count = len(all_neurons_set)
            else:
                all_neurons_count = total_neurons_count
            xs.append(idx)
            # TODO: Assert the length of the all neurons set matches what we expect
            # in another way
            ys.append(100*len(sgf_traces_set)/all_neurons_count)
            bottoms.append(0)
            colors.append('k')
            labels_li.append(f"Total unique sgf.\n"
                 f"active sgf. = {len(sgf_traces_set)}/{all_neurons_count}\n"
                 f"active ns. = {len(non_sgf_neurons_set)}/{all_neurons_count}")
        # ax.pie(grp_by.size().values, labels=grp_by.groups.keys())
        ax.bar(xs, ys, tick_label=labels_li, bottom=bottoms, color=colors)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_ylabel("% sgf. neurons", fontsize=9)
        max_y = 100 if total_neurons_count is None else 50 # None
        ax.set_ylim(0, max_y)
        ax.set_xlim(-0.5)


        # Draw venn diagram
        def vennCountFormatter(count):
            # print("Count:", count)
            prcnt = 100*count/all_neurons_count
            return "" if prcnt == 0 else (
                   f"{prcnt:.1g}%" if prcnt < 1 else
                   f"{int(prcnt)}%")
        ax = ax_row[1]
        circle = plt.Circle((0, -0.1), 0.7, color='gray', alpha=0.2,
                            clip_on=False)
        ax.add_patch(circle)
        active_neurons_set = sgf_traces_set | non_sgf_neurons_set
        active_neurons_count = len(active_neurons_set )
        ax.text(0.2, -0.75, vennCountFormatter(active_neurons_count),
                ha='right',  va='center')
        ax.text(0.25, -0.8, f"Epoch Active Nuerons", ha='left', va='center',
                fontsize='large')
        sets = []
        labels_li = []
        for idx, (data_col_name, data_col_df) in enumerate(
                                                    sub_df.groupby("data_col")):
            sgf_traces = set(data_col_df.trace_id)
            sets.append(sgf_traces)
            labels_li.append(data_col_name)
        if len(sets) == 2:
            venn2(sets, labels_li, ax=ax,
                  subset_label_formatter=vennCountFormatter)
        elif len(sets) == 3:
            venn3(sets, labels_li, ax=ax,
                  subset_label_formatter=vennCountFormatter)
    epoch_save = epoch.replace("/", "_")
    if fig_save_prefix is not None:
        totals_used = "total_" if total_neurons_count is not None else ""
        save_path = Path(f"{fig_save_prefix}/distribution/{totals_used}"
                         f"{epoch_save}_{br_str}_{layer}_sigf_no_prior.jpeg")
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    return
