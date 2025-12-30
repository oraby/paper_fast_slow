from ...common.clr import BrainRegion as BRClr
from ...common.definitions import BrainRegion
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
try:
    from IPython.display import display
except ModuleNotFoundError:
    pass
try:
    from matplotlib_venn import venn3
except ModuleNotFoundError:
    import sys
    print("matplotlib_venn is not installed", file=sys.stderr)
import numpy as np
import pandas as pd
from pathlib import Path

def loopPlotResults(df, pval, left_tuning_col, fig_save_prefix,
                    active_traces_df):
    # First without priors
    df = df.copy()
    # df_without_priors = df[df.prior_data_val.isna()]
    # print(f"Full df: {len(df):,} - "
    #       f"without priors: {len(df_without_priors):,}")
    print("active_traces_df.epochs:", active_traces_df.epoch.unique())
    print("Epoch df groups:", df.epoch.unique())
    assert [any(epoch == active_traces_df.epoch) for epoch in df.epoch.unique()]
    for epoch, epoch_df in df.groupby(df.epoch):
        # if epoch != "Sampling":
        #     continue
        epoch_active_traces = active_traces_df[active_traces_df.epoch == epoch]
        assert len(epoch_active_traces), (
                                       f"Didn't find active traces for {epoch}")
        for part, part_traces_df in epoch_active_traces.groupby("part"):
            total_parts = part_traces_df.total_parts.iloc[0]
            epoch_name = epoch
            if total_parts > 1:
                epoch_name += f" {part}/{total_parts}"
            _loopPlotEpochResults(epoch_name, epoch_df, pval, left_tuning_col,
                                  fig_save_prefix,
                                  active_traces_df=part_traces_df)

def _loopPlotEpochResults(epoch, epoch_df, pval, left_tuning_col,
                          fig_save_prefix, active_traces_df):
    for col_name, epoch_col_df in epoch_df.groupby(epoch_df.data_col):
        epoch_col_df_no_priors = epoch_col_df[
                                           epoch_col_df.prior_data_col.isnull()]
        epoch_col_df_with_priors = epoch_col_df[
                                          epoch_col_df.prior_data_col.notnull()]
        epoch_col_df_as_prior = epoch_df[epoch_df.prior_data_col == col_name]
        _loopPlotEpochColResults(epoch=epoch, col=col_name,
                                 epoch_col_df_no_priors=epoch_col_df_no_priors,
                                 epoch_col_df_with_priors=
                                                       epoch_col_df_with_priors,
                                 epoch_col_df_as_prior=epoch_col_df_as_prior,
                                 pval=pval, left_tuning_col=left_tuning_col,
                                 fig_save_prefix=fig_save_prefix,
                                 active_traces_df=active_traces_df)


def _loopPlotEpochColResults(epoch, col, epoch_col_df_no_priors,
                              epoch_col_df_with_priors, epoch_col_df_as_prior,
                              left_tuning_col, pval, fig_save_prefix,
                              active_traces_df):

    save_col_name = col.replace(" ", "_")
    epoch_save = epoch.replace("/", "_")
    # print(epoch, "Print col name:", print_col_name)
    if len(epoch_col_df_no_priors):
        # display(without_priors_df)
        df_plot = _countSigfTraces(res_df=epoch_col_df_no_priors,
                                   left_tuning_col=left_tuning_col,
                                   active_traces_df=active_traces_df, pval=pval)
        _plotSimpleBars(df_plot, title=f"{col} {epoch} DI",
                        save_name=f"{save_col_name}_{epoch_save}",
                        fig_save_prefix=fig_save_prefix)
        # Pie-chart without priors
        _plotDistribitionPieChart(no_priors_df=epoch_col_df_no_priors,
                                  has_prior_df=None,
                                  active_traces_df=active_traces_df,
                                  pval=pval,
                                  print_col_name=col,
                                  title=f"{col} during {epoch}",
                                  save_name=f"{save_col_name}_{epoch_save}",
                                  fig_save_prefix=fig_save_prefix)
    for prior_col_name, with_priors_df in epoch_col_df_with_priors.groupby(
                                                              "prior_data_col"):
        _plot3dDistributionGivenPrior(priors_df=with_priors_df,
                                      non_prior_df=epoch_col_df_no_priors,
                                      active_traces_df=active_traces_df,
                                      epoch=epoch, pval=pval,
                                      fig_save_prefix=fig_save_prefix)
    # Now with priors
    prior_col_name = col
    prior_save_col_name = save_col_name
    ref_df = epoch_col_df_no_priors
    if not len(ref_df):
        print(f"Didn't find ref df for for prior {prior_col_name} in {epoch}")
        print("Hoping to find it in another epoch")
        # ref_df = epoch_df[(epoch_df.data_col == col_name) &
        #                                        epoch_df.prior_data_val.isna()]
        # assert len(ref_df), "Didn't find ref df for prior"
        assert False, "Didn't find ref df for prior"
    for data_col, data_col_with_cur_col_as_prior_df in \
                  epoch_col_df_as_prior.groupby(epoch_col_df_as_prior.data_col):
        save_col_name = data_col.replace(" ", "_")
        for prior_val, prior_val_df in \
                    data_col_with_cur_col_as_prior_df.groupby("prior_data_val"):

            title = (f"{data_col} given {prior_col_name}={prior_val} "
                     f"during {epoch}")
            save_name = (f"{save_col_name}_given_{prior_save_col_name}_"
                         f"{prior_val}_{epoch_save}")
            print(f"Plotting {title} DI")
            df_plot = _countSigfTraces(res_df=prior_val_df, ref_count_df=ref_df,
                                       left_tuning_col=left_tuning_col,
                                       active_traces_df=active_traces_df,
                                       pval=pval)
            _plotSimpleBars(df_plot, title=f"{col} {epoch} DI",
                            save_name=save_name,
                            fig_save_prefix=fig_save_prefix)

def _countSigfTraces(res_df, pval, left_tuning_col, active_traces_df=None,
                     ref_count_df=None, totals_before_filtering=False):
    # res_dict[br][layer][sess_name] = pvals_statistics_li
    def assertLenNotChanged(df):
        df_len = len(df)
        def displayDup():
            ex_df = df[df.long_trace_id ==
                       df.long_trace_id[df.long_trace_id.duplicated()].iloc[0]]
            display(ex_df)
            return "Dataframe has duplicates"
        # assert df_len == len(df.drop_duplicates(subset=["long_trace_id"])), \
        #                                                           displayDup()
        # return df_len
        return len(df.drop_duplicates(subset=["long_trace_id"]))

    # Follow the 2 comments instructions below if you want to count the totals
    # before filtering
    if totals_before_filtering:
        res_df_totals = res_df.copy()
    if active_traces_df is not None:
        res_df = res_df[
                      res_df.long_trace_id.isin(active_traces_df.long_trace_id)]
        print("Filtered")
        # Commennt out if you want to count the totals before filtering
        if ref_count_df is not None:
            ref_count_df = ref_count_df[ref_count_df.long_trace_id.isin(
                                                active_traces_df.long_trace_id)]
    if not totals_before_filtering:
        res_df_totals = res_df.copy()
    # display(res_df)
    xs = []
    ys = []
    sgf_count_li = []
    total_count_li = []
    labels = []
    clrs = []
    brs_li = []
    x_pos = 0
    RIGT_SIDE_X = 4
    if left_tuning_col is not None:
        groups = res_df.groupby(left_tuning_col)
    else:
        groups = [(None, res_df)]
    for is_left_tuned_or_None, tuning_df in groups:
        if is_left_tuned_or_None is None:
            tuning_label = \
                        tuning_df.data_val_left if is_left_tuned_or_None else \
                        tuning_df.data_val_right
            assert tuning_label.nunique() == 1, tuning_label.unique()
            tuning_label = f" {tuning_label.iloc[0]}"
        else:
            tuning_label = ""
        for br, br_df in tuning_df.groupby("BrainRegion"):
            prcnts_li, cur_sgf_count_li, cur_total_count_li = [], [], []
            for layer, layer_df in br_df.groupby(br_df.Layer):
                for sess_name, sess_df in layer_df.groupby(layer_df.ShortName):
                    if ref_count_df is not None:
                        ref_count_sess_df = ref_count_df[
                                            ref_count_df.ShortName == sess_name]
                    else:
                        ref_count_sess_df = res_df_totals[
                                           res_df_totals.ShortName == sess_name]
                    # TODO: If you have priors, you can have traces that are
                    # significant under both labels.
                    len_all_traces = assertLenNotChanged(ref_count_sess_df)
                    assertLenNotChanged(sess_df)
                    sig_df = sess_df[sess_df.pval <= pval]
                    # print("Sess len:", len(sess_df),
                    #       "Other len:", len_all_traces)
                    sig_prcnt = 100*len(sig_df)/len_all_traces
                    prcnts_li.append(sig_prcnt)
                    cur_sgf_count_li.append(len(sig_df))
                    cur_total_count_li.append(len_all_traces)
            if len(prcnts_li) == 0:
                continue
            xs.append(x_pos)
            ys.append(prcnts_li)
            sgf_count_li.append(cur_sgf_count_li)
            total_count_li.append(cur_total_count_li)
            clrs.append(BRClr[BrainRegion(br)])
            br_str = str(BrainRegion(br)).split('_')[0]
            labels.append(f"{br_str}{tuning_label}")
            brs_li.append(br_str)
            x_pos += 1
        x_pos += RIGT_SIDE_X

    df_plot = pd.DataFrame({"x": xs, "y_li": ys, "clr": clrs, "label": labels,
                            "sgf_count_li": sgf_count_li,
                            "total_count_li": total_count_li,
                            "BrainRegion": brs_li})
    return df_plot


def _plotSimpleBars(df_plot, title, save_name, fig_save_prefix):
    fig, ax = plt.subplots()
    text_transform = transforms.blended_transform_factory(ax.transData,
                                                          ax.transAxes)
    for _, row in df_plot.iterrows():
        x, y_li, clr, label = row.x, row.y_li, row.clr, row.label
        box = ax.boxplot(y_li, positions=[x], widths=0.5, patch_artist=True)
        for el in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(box[el], color=clr)
        ax.text(x, 0, label, ha="right", va="top", rotation=45,
                transform=text_transform)
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{title} - Signficant Neurons", fontsize=10)
    ax.set_ylabel("Neurons Percentage")
    if fig_save_prefix is not None:
        save_path = Path(f"{fig_save_prefix}/{save_name}.jpeg")
        print(f"Saving to {save_path}")
        # save_path.parent.mkdir(exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()

def _plotSignficantSlicesTogether(ax, ax_pie_rets):
    # Plot signficant pies slices together, thanks to:
    # https://stackoverflow.com/a/70578760/11996983
    wedges, texts, percs = ax_pie_rets
    groups = [[1, 2]] # Assuming that second and third are the significant ones
    radfraction = 0.1
    for group in groups:
        ang = np.deg2rad(
                       (wedges[group[-1]].theta2 + wedges[group[0]].theta1) / 2)
        for j in group:
            center = radfraction * wedges[j].r * np.array([np.cos(ang),
                                                           np.sin(ang)])
            wedges[j].set_center(center)
            texts[j].set_position(np.array(texts[j].get_position()) + center)
            percs[j].set_position(np.array(percs[j].get_position()) + center)
    ax.autoscale(True)

def _plotDistribitionPieChart(no_priors_df, pval, print_col_name,
                              title, save_name, fig_save_prefix,
                              has_prior_df=None, active_traces_df=None):
    assert no_priors_df.data_col.nunique() == 1
    assert no_priors_df.prior_data_val.isna().all()
    if has_prior_df is not None:
        assert has_prior_df.data_col.nunique() == 1, ("Found "
            f"{has_prior_df.data_col.unique()} for "
            f"{no_priors_df.data_col.iloc[0]}")
        assert has_prior_df.data_col.iloc[0] == no_priors_df.data_col.iloc[0]
        has_prior_df = has_prior_df.sort_values(
                                      by=["ShortName", "long_trace_id", "pval"])
        uniq_has_prior_df = has_prior_df.drop_duplicates(
                            subset=["ShortName", "long_trace_id"], keep="first")
    else:
        uniq_has_prior_df = None

    if active_traces_df is not None:
        no_priors_df = no_priors_df[no_priors_df.long_trace_id.isin(
                                                active_traces_df.long_trace_id)]
        if uniq_has_prior_df is not None:
            uniq_has_prior_df = uniq_has_prior_df[
                uniq_has_prior_df.long_trace_id.isin(
                                                active_traces_df.long_trace_id)]
    print_col_name = print_col_name.lower()
    for br, br_df in no_priors_df.groupby(no_priors_df.BrainRegion):
        for layer, layer_df in br_df.groupby(br_df.Layer):
            num_all_traces = len(layer_df)
            num_sgf_all = np.sum(layer_df.pval <= pval)
            if uniq_has_prior_df is not None:
                uniq_has_prior_layer_df = has_prior_df[
                                              (has_prior_df.BrainRegion == br) &
                                              (has_prior_df.Layer == layer)]
                num_sgf_multiplex = np.sum(uniq_has_prior_layer_df.pval <= pval)
                count_li = [num_sgf_multiplex]
                # Bad hack
                posterior = "current" if "prev" in print_col_name else "previous"
                labels_li = [f"sgf. {posterior} | sgf. {print_col_name}"]
                clrs_li = ["orange"]
            else:
                num_sgf_multiplex = 0
                count_li = []
                labels_li = []
                clrs_li = []
            br_clr = BRClr[BrainRegion(br)]
            # Draw a pie chart for signficant neurons in
            print("xs?:", [num_all_traces-num_sgf_all,
                           num_sgf_all-num_sgf_multiplex] + count_li,)
            fig, ax = plt.subplots()
            ax.set_axis_off()
            rets = ax.pie([num_all_traces-num_sgf_all,
                           num_sgf_all-num_sgf_multiplex] + count_li,
                           #explode=[0, 0, 0.1],
                           labels=["ns.", f"sgf. {print_col_name}"] + labels_li,
                           colors=["grey", br_clr] + clrs_li,
                           shadow=True,
                           autopct='%1.1f%%', startangle=70)
            if len(count_li):
                _plotSignficantSlicesTogether(ax, rets)
            layer_str = "L2/3" if layer == "L23" else layer
            br_str = str(BrainRegion(br)).split("_")[0]
            ax.set_title(f"Neurons selectivity for {br_str} {layer_str} {title} ",
                         fontsize=9)
            if fig_save_prefix is not None:
                save_path = Path(f"{fig_save_prefix}/pie_charts/"
                                 f"{save_name}_{br_str}_{layer}.jpeg")
                print(f"Saving to {save_path}")
                save_path.parent.mkdir(exist_ok=True)
                fig.savefig(save_path)
            plt.show()

def _plot3dDistributionGivenPrior(priors_df, non_prior_df, active_traces_df,
                                  epoch, pval, fig_save_prefix):
    for br, br_priors_df in priors_df.groupby(priors_df.BrainRegion):
        br_non_prior_df = non_prior_df[non_prior_df.BrainRegion == br]
        _plot3dDistributionGivenPriorByBrainRegion(
                                              priors_df=br_priors_df,
                                              non_prior_df=br_non_prior_df,
                                              epoch=epoch, pval=pval, br=br,
                                              fig_save_prefix=fig_save_prefix,
                                              active_traces_df=active_traces_df)

def _plot3dDistributionGivenPriorByBrainRegion(priors_df, non_prior_df,
                                               epoch, pval, br, fig_save_prefix,
                                               active_traces_df=None):
    assert non_prior_df.prior_data_val.isna().all()
    assert non_prior_df.data_col.nunique() == 1
    assert priors_df.prior_data_col.nunique() == 1
    assert priors_df.data_col.nunique() == 1
    data_col = non_prior_df.data_col.iloc[0]
    assert data_col == priors_df.data_col.iloc[0], (
                            f"Unmatching {data_col=} != "
                            f"priors_df.data_col={priors_df.data_col.iloc[0]}")
    prior_data_col = priors_df.prior_data_col.iloc[0]

    possible_prior_vals = priors_df.prior_data_val.unique()
    assert len(possible_prior_vals) == 2
    x_conds = priors_df[priors_df.prior_data_val == possible_prior_vals[0]]
    y_conds = priors_df[priors_df.prior_data_val == possible_prior_vals[1]]
    z_conds = non_prior_df[
                        non_prior_df.long_trace_id.isin(y_conds.long_trace_id)]
    z_conds["pval_z"] = z_conds.pval
    assert len(x_conds) == len(y_conds)
    assert len(z_conds) == len(x_conds)
    merge_on = ["long_trace_id", "ShortName", "Layer", "BrainRegion"]
    df_comb = x_conds.merge(y_conds, on=merge_on, suffixes=("_x", "_y"))
    df_comb = df_comb.merge(z_conds, on=merge_on, suffixes=("", "_z"))
    assert len(df_comb) == len(x_conds)
    # Remove non signficant neurons
    if active_traces_df is not None:
        df_comb = df_comb[df_comb.long_trace_id.isin(
                                                active_traces_df.long_trace_id)]
        print("Len after filtering:", len(df_comb))
    total_neurons = len(df_comb)
    # Change to "all neurons" always if count total before filtering
    total_descrp = "active neurons " if active_traces_df is not None else \
                                 "all neurons "
    class Clr:
        BothPriorsAndNoPrior = "red"
        WithBothPriorsOnly = "green"
        WithOnePriorOnly = "blue"
        WithOnePriorAndNoPrior = "yellow"
        OnlyNoPrior = "black"
        NonSignificant = "grey"
    colors = []
    clrs_counts = {Clr.BothPriorsAndNoPrior: 0, Clr.WithOnePriorAndNoPrior: 0,
                   Clr.WithBothPriorsOnly: 0, Clr.WithOnePriorOnly: 0,
                   Clr.OnlyNoPrior: 0, Clr.NonSignificant: 0}
    for row in df_comb.itertuples():
        if row.pval_x <= pval and row.pval_y <= pval and row.pval_z <= pval:
            clr = Clr.BothPriorsAndNoPrior
        elif row.pval_x <= pval and row.pval_y <= pval:
            clr = Clr.WithBothPriorsOnly
        elif row.pval_x <= pval or row.pval_y <= pval:
            if row.pval_z <= pval:
                clr = Clr.WithOnePriorAndNoPrior
            else:
                clr = Clr.WithOnePriorOnly
        elif row.pval_z <= pval:
            clr = Clr.OnlyNoPrior
        else:
            clr = Clr.NonSignificant
        if clr != Clr.NonSignificant:
            colors.append(clr)
        clrs_counts[clr] += 1
    # Create a 3d plot
    br_str = str(BrainRegion(br)).split("_")[0]
    import matplotlib.gridspec as gridspec
    fig = plt.figure()
    fig.set_size_inches(20, 10)
    fig.subplots_adjust(top=1)
    fig.suptitle(f"{data_col} distribution on given {prior_data_col} "
                 f"during {epoch} in {br_str}", fontsize=16)
    # gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[3, 2, 2])
    # ax = fig.add_subplot(gs[0], projection='3d')
    ax = fig.add_subplot(121, projection='3d')
    df_comb = df_comb[(df_comb.pval_x <= pval) | (df_comb.pval_y <= pval) |
                      (df_comb.pval_z <= pval)]
    ax.scatter(df_comb.pval_x, df_comb.pval_y, df_comb.pval_z, s=1, c=colors)
    ax.set_title("P-val distribution", y=.95)
    common_msg = f"{data_col}\nwhen {prior_data_col}="
    labels_li = [f"{common_msg}{df_comb.prior_data_val_x.iloc[0]}",
                 f"{common_msg}{df_comb.prior_data_val_y.iloc[0]}",
                 f"{data_col} without no priors"]
    ax.set_xlabel(f"P-val {labels_li[0]}")
    ax.set_ylabel(f"P-val {labels_li[1]}")
    ax.set_zlabel(f"P-val {labels_li[2]}")
    # ax.set_ylim(1, 0)
    # ax.set_zlim(1, 0)
    ax.view_init(elev=10., azim=70, roll=0)
    ax.dist = 8

    # ax = fig.add_subplot(gs[1])
    ax = fig.add_subplot(122)
    sets = [set(df_comb[df_comb.pval_x <= pval].long_trace_id),
                    set(df_comb[df_comb.pval_y <= pval].long_trace_id),
                    set(df_comb[df_comb.pval_z <= pval].long_trace_id)]
    v = venn3(sets, labels_li, ax=ax)
    def setPatchColor(set_id, clr):
        p = v.get_patch_by_id(set_id)
        if p:
            p.set_color(clr)
    setPatchColor('100', Clr.WithOnePriorOnly)
    setPatchColor('010', Clr.WithOnePriorOnly)
    setPatchColor('101', Clr.WithOnePriorAndNoPrior)
    setPatchColor('011', Clr.WithOnePriorAndNoPrior)
    setPatchColor('001', Clr.OnlyNoPrior)
    setPatchColor('110', Clr.WithBothPriorsOnly)
    setPatchColor('111', Clr.BothPriorsAndNoPrior)
    for text in v.set_labels:
        text.set_fontsize(9)
    ax.set_title("Significant neurons distribution", y=.95)
    # Draw figure legend
    from matplotlib.lines import Line2D
    def addPatch(text, clr):
        count = clrs_counts[clr]
        return Line2D([], [], marker="o", markerfacecolor=clr,
                      markeredgecolor=clr, linestyle='None',
                      label=(f"{text} ({count:,} - "
                             f"{100*count/total_neurons:.3g}%)"))
    legend_elements = [
        addPatch("sgf. with & without priors", Clr.BothPriorsAndNoPrior),
        addPatch("Sgf. only without priors", Clr.OnlyNoPrior),
        addPatch("Sgf. only on a single prior and withour priors",
                                                    Clr.WithOnePriorAndNoPrior),
        addPatch("Sgf. only on a single prior but not without priors",
                                                          Clr.WithOnePriorOnly),
        addPatch("Sgf. only on both priors", Clr.WithBothPriorsOnly),
        addPatch(f"Non-significant {total_descrp}(not plotted)",
                                                            Clr.NonSignificant),
    ]
    fig.legend(handles=legend_elements, loc='upper center', fontsize=9,
               frameon=False, ncol=2, bbox_to_anchor=(0.4, 0.95))
    # plt.tight_layout()
    epoch_save = epoch.replace("/", "_")
    save_path = Path(f"{fig_save_prefix}/priors_distribution/"
                     f"{data_col}_given_{prior_data_col}_{br_str}_"
                     f"{epoch_save}.jpeg")
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path)
    plt.show()
