from ...pipeline import pipeline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from itertools import groupby

def plotTraceTrialsHeatmap(df, sess_name, trace_id, conditions=[],
                           sortHeatmapFn=None, set_name="neuronal",
                           fig_save_prefix=None, save_figs=False):
    # plotCellHeatmap(df_all_by_trial_normalized, "GP4_85_s2_L100_D250_mm2", 6,
    #                                 ["PrevChoiceLeft", "ChoiceLeft"])
    df = df.copy()
    vals_li = []
    index_levels_names = []
    index_vals = []
    # root_node = hierarchy.ClusterNode("root", count=len(df))
    _extractData(df, conditions, trace_id, set_name, output_vals_li=vals_li,
                 output_index_level_name=index_levels_names,
                 output_index_vals=index_vals)
    print("index_vals:", len(index_vals), len(index_vals[0]))
    print("index_levels_names:", index_levels_names, len(index_levels_names))
    new_df = pd.DataFrame(vals_li, index=pd.MultiIndex.from_tuples(index_vals,
                          names=index_levels_names))
    new_df.sort_values(by=index_levels_names, inplace=True)
    if sortHeatmapFn is not None:
        new_df = sortHeatmapFn(new_df)
        vals_li = new_df.values
    # display(new_df)
    fig, ax = plt.subplots(figsize=(20,10))
    ax = sns.heatmap(vals_li, cmap="inferno", #colorMapParula(),
                     xticklabels=False, yticklabels=False, ax=ax,
                     cbar_kws=dict(location="left", #use_gridspec=False,
                     pad=0.02))
    _addManualLabels(new_df, ax)
    ax.set_title(f"Session: {sess_name} - Trace: {trace_id}")
    row = df.iloc[0]
    for epoch, epoch_range in zip(row.epochs_names, row.epochs_ranges):
        ax.axvline(epoch_range[0], color="gray", ls="--")
        ax.text(epoch_range[0], 0, epoch, rotation=45, color="black",
                transform=ax.get_xaxis_transform(), va="top", ha="right")
    if save_figs:
        conds_str = "_".join(conditions)
        filename = f"{trace_id}_{conds_str}_{sess_name}"
        print("Saving:", filename)
        fig.savefig(f"{fig_save_prefix}/{filename}.jpeg", dpi=300)
        plt.close()
    else:
        plt.show()

def _extractData(df, remaining_cond, trace_id, set_name, output_vals_li,
                                 output_index_level_name, output_index_vals,
                                 index_vals_prefix=[]):
    # cond_dict = ()
    if (not len(output_index_level_name) or remaining_cond[0] !=
                                            output_index_level_name[-1]):
        output_index_level_name.append(remaining_cond[0])
    for cond, cond_df in df.groupby(remaining_cond[0]):
        # Probably no need to copy, we can just append to the list
        cur_index_vals_prefix = index_vals_prefix.copy() + [cond]
        if len(remaining_cond) > 1:
            _extractData(cond_df, remaining_cond[1:], trace_id, set_name,
                         output_vals_li, output_index_level_name,
                         output_index_vals,
                         cur_index_vals_prefix)
            # cond_dict[(remaining_cond[0], cond)] = res
        else:
            time_ax = pipeline.TIME_AX_LOOKUP[set_name]
            # res_li = []
            for row_idx, row in cond_df.iterrows():
                traces_dict = pipeline.getRowTracesSets(row)[set_name]
                trace_data = traces_dict[trace_id].take(
                                                np.arange(row.trace_start_idx,
                                                          row.trace_end_idx+1),
                                                axis=time_ax)
                # res_li.append(trace_data)
                output_vals_li.append(trace_data)
                output_index_vals.append(cur_index_vals_prefix)
            # cond_dict[(remaining_cond[0], cond)] = res_li
    # return cond_dict

def _addManualLabels(df, ax):
    # https://stackoverflow.com/a/58917086/11996983
    def label_len(my_index, level):
        labels = my_index.get_level_values(level)
        label_len = [(k, sum(1 for i in g)) for k,g in groupby(labels)]
        # print("label_len:", label_len)
        return label_len

    def add_line(ax, xpos_s, xpos_e, ypos, add_center_line):
        if add_center_line: # xpos_s < 1:
            line = plt.Line2D([0, 1], [ypos, ypos], color='white', alpha=0.8,
                               ls="--", transform=ax.transAxes)
            ax.add_line(line)
        else:
            line = plt.Line2D([xpos_s, xpos_e], [ypos, ypos], color='black',
                               transform=ax.transAxes)
            line.set_clip_on(False)
            ax.add_line(line)
    xpos = 1.02
    scale = 1./df.index.size
    xpos_line_end = .12
    print("Scale:", scale)
    for level in reversed(range(df.index.nlevels)):
        pos = df.index.size
        labels_li = label_len(df.index,level)
        # first_label = labels_li[0][0]
        for label, rpos in labels_li:
            add_center_line = level == df.index.nlevels - 1
            xpos_s = 1 if level == 0 else xpos
            add_line(ax, xpos_s, xpos+xpos_line_end, pos*scale,
                     add_center_line=add_center_line)
            pos -= rpos
            lypos = (pos + .5 * rpos)*scale
            if np.around(lypos, decimals=2) == 0.76:
                lypos = 0.788
            elif np.around(lypos, decimals=2) == 0.26:
                lypos = 0.25
            print("xpos:", xpos, "lypos:", lypos,
                  "Rounded:", np.around(lypos, decimals=2), "label:", label)
            ax.text(xpos+.05, lypos, label, ha='center', va='center',
                    transform=ax.transAxes)
        add_center_line = level == df.index.nlevels - 1
        xpos_s = 1 if level == 0 else xpos
        add_line(ax, xpos_s, xpos+xpos_line_end, pos*scale,
                 add_center_line=add_center_line)
        xpos += .12
