from . import splitconditions as sc
from .testscomb import prepareDf
from .plot import rocplot
from .runstattest import loopConditions
from ..common.definitions import BrainRegion
from ..pipeline import pipeline
try:
    from IPython.display import display
except ModuleNotFoundError:
    pass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from typing import Literal, Union

def preprocessDF(df, dim_reduc_wrapper, filter_df_and_layer : bool):
    assert len(dim_reduc_wrapper._sess_trials_traces_res) == 0, (
                                                           "Create new instace")
    len_all = len(df)
    df_valid = df[df.ChoiceCorrect.notnull()]
    len_valid = len(df_valid)
    # Next line doesn't make a difference here
    df_valid = df_valid[df_valid.Difficulty3.notnull()]
    len_3d = len(df_valid)
    if filter_df_and_layer:
        df_valid = df_valid[
             df_valid.BrainRegion.isin([BrainRegion.M2_Bi, BrainRegion.ALM_Bi])]
        df_valid = df_valid[df_valid.Layer == "L23"]
    del df # Don't use again by mistake
    print(f"Num all trials: {len_all} - Num completed trials: {len_valid} - "
                f"Num of completed @3 difficulties: {len_3d}")
    # df_bins = df_valid.dur_bin.unique()
    # display(df_bins)
    # display(df_valid.ShortName.unique())
    # splits = df_valid.groupby("dur_bin")
    # splits = splitStimulusTimeByQuantile(df_valid, use_assignDVStr=True)
    dim_redic_name = dim_reduc_wrapper.dimReducName()
    print("Running dim reduc:", dim_redic_name)
    chain = pipeline.Chain(
                pipeline.BySession(),
                    pipeline.LoopTraces(dim_reduc_wrapper),
                pipeline.RecombineResults(),
    )
    # df_valid = df_valid[
    #                  df_valid.ShortName.isin(df_valid.ShortName.unique()[:3])]
    df_valid = chain.run(df_valid)
    print("Assigning traces to df")
    new_rows = []
    # TODO: Store the stats parameters for each neurons somewhere else
    for sess, sess_df in df_valid.groupby("ShortName"):
        sess_trial_traces = dim_reduc_wrapper._sess_trials_traces_res[sess]
        for row_idx, row in sess_df.iterrows():
            traces_sets = pipeline.getRowTracesSets(row)
            traces_neuronal = traces_sets["neuronal"]
            dim_reduc_neuronal = sess_trial_traces[row.TrialNumber]
            assert traces_neuronal.keys() == dim_reduc_neuronal.keys()
            row = row.copy()
            row["neuronal_reduc"] = dim_reduc_neuronal
            new_rows.append(row)
    df_valid = pd.DataFrame(new_rows)
    df_valid["DimReduc"] = dim_redic_name
    return df_valid


def _processDf(splitFn, priorSplitFn, df, epoch, df_res_li, processors_li,
               quantile_idx=None, dv_str=None, statsTestFn=None):
    res = loopConditions(df=df.copy(), epoch=epoch, splitFns=[splitFn],
                         priorSplitFn=priorSplitFn, statsTestFn=statsTestFn)
    del df # Avoid using by mistake
    df_res, df_processor = res
    if quantile_idx is not None:
        df_res["quantile_idx"] = quantile_idx
    if dv_str is not None:
        df_res["DVstr"] = dv_str
    df_res_li.append(df_res)
    processors_li.append(df_processor)

def _runProcessDFGroup(df, df_res_li, processors_li, epoch, quantile_idx=None,
                       statsTestFn=None):
    kargs = dict(epoch=epoch, df_res_li=df_res_li,
                 processors_li=processors_li, quantile_idx=quantile_idx,
                 statsTestFn=statsTestFn)
    [_processDf(sc.conditionsLeftRight, None, df=dv_df, dv_str=dv, **kargs)
     for dv, dv_df in df.groupby("DVstr")]
    kargs["df"] = df
    kargs["dv_str"] = np.nan; print("Change back to None")
    _processDf(sc.conditionsLeftRight, None, **kargs)
    _processDf(sc.conditionsEasyDifficult, None, **kargs)
    _processDf(sc.conditionsEasyDifficult, sc.conditionsLeftRight, **kargs)
    # priors
    _processDf(sc.conditionsPrevLeftRight, None, **kargs)
    _processDf(sc.conditionsPrevEasyDifficult, None, **kargs)
    _processDf(sc.conditionsPrevCorrectIncorrect, None, **kargs)

def splitDatTestAll(df, statsTestFn):
    assert "neuronal_reduc" in df.columns, "Run preprocessDF() first"
    uniq_epoch = df.epoch.unique()
    assert len(uniq_epoch) == 1
    epoch = uniq_epoch[0]
    print("Processing for:", epoch)
    df_res_li, processors_li = [], []
    _runProcessDFGroup(df, df_res_li, processors_li, epoch=epoch,
                       quantile_idx=None, statsTestFn=statsTestFn)
    return pd.concat(df_res_li), processors_li

def splitDatTestQuantiles(df, statsTestFn):
    df_res_li, processors_li = [], []
    for quantile_idx, quantile_df in df.groupby("quantile_idx"):
        # display(quantile_df)
        quantile_df = quantile_df.copy()
        # quantile_df["epoch"] = "Sampling"
        print("quantile_idx:", quantile_idx, "Mean reaction time:",
              quantile_df.calcStimulusTime.mean())
        # _runProcessDFGroup(quantile_df, df_res_li, processors_li, epoch=epoch,
        #                    quantile_idx=quantile_idx, statsTestFn=statsTestFn)
        stats_quant_df, p_li = splitDatTestAll(df=quantile_df,
                                               statsTestFn=statsTestFn)
        stats_quant_df["quantile_idx"] = quantile_idx
        df_res_li.append(stats_quant_df)
        processors_li += p_li
    return pd.concat(df_res_li), processors_li

def loopShortLongQUantiles(df, epoch, active_traces_df, pval,
                           left_tuning_col : Union[None, str],
                           active_neurons_per_quantile=True,
                           total_before_filtering=False):
    df = df.copy()
    # df = prepareDf(df, src_df=src_sess_layer_br_df, rename_cols=False)
    df["epoch"] = epoch
    if "DVstr" not in df.columns:
        df["DVstr"] = None
    df_plot_li = []
    X_STEP = 4
    # display(df[["BrainRegion", "Layer", "data_col", "prior_data_col",
    #             "DVstr"]].value_counts(dropna=False))
    for quantile_idx, quantile_df in df.groupby("quantile_idx"):
        if active_traces_df is not None:
            print("Active neurons is not None")
            if active_neurons_per_quantile:
                cur_active_traces_df = active_traces_df[
                                  active_traces_df.quantile_idx == quantile_idx]
            else:
                assert ("quantile_idx" not in active_traces_df.columns or
                        active_traces_df.quantile_idx.nunique() == 1)
                cur_active_traces_df = active_traces_df
        else:
            cur_active_traces_df = None
        quantile_idx = int(quantile_idx)
        # print("Quantile idx:", quantile_idx)
        quantile_df = quantile_df[quantile_df.DVstr.isnull()]
        plot_df = _loopQuantileCols(quantile_idx, quantile_df, pval,
                                    left_tuning_col, cur_active_traces_df,
                                  total_before_filtering=total_before_filtering)
        df_plot_li.append(plot_df)
    df_plot = pd.concat(df_plot_li)
    # rocplot._plotSimpleBars(df_plot,
    #                        title="Direction Tuning by Reaction-Time Quantile",
    #                        save_name="quantile_reaction_time",
    #                        fig_save_prefix=None)
    return df_plot


def loopShortLongQUantilesDict(df_dict, *args, **kargs):
    df_li = []
    for df_name, df in df_dict.items():
        df_res = loopShortLongQUantiles(df, *args, **kargs)
        df_res["group"] = df_name
        df_li.append(df_res)
    return pd.concat(df_li)

def _loopQuantileCols(quantile_idx, quantile_df, pval, left_tuning_col,
                      active_traces_df, total_before_filtering=False):
    plot_col_dfs = []
    for data_col, data_col_df in quantile_df.groupby("data_col"):
        # print("data_col:", data_col)
        data_col_df = quantile_df[quantile_df.data_col == data_col]
        # if data_col != "ChoiceLeft":
        #     continue
        # display(data_col_df)ff
        # display(data_col_df[["BrainRegion", "Layer", "data_col",
        #                      "prior_data_col", "DVstr"]
        #                    ].value_counts(dropna=False))
        for prior_col, prior_or_no_prior_df in data_col_df.groupby(
                                                "prior_data_col", dropna=False):
            # print("data_col:", data_col, "- prior_col:", prior_col)
            if not pd.isnull(prior_col):
                ref_df = quantile_df[(quantile_df.data_col == prior_col) &
                                      quantile_df.prior_data_col.isnull()]
                ref_df = ref_df[ref_df.DVstr.isnull()]
                # display(ref_df)
            else:
                ref_df = None
            plot_df = rocplot._countSigfTraces(prior_or_no_prior_df,
                                               pval=pval,
                                               left_tuning_col=left_tuning_col,
                                              active_traces_df=active_traces_df,
                                               ref_count_df=ref_df,
                                 totals_before_filtering=total_before_filtering)
            plot_df["quantile_idx"] = quantile_idx
            # plot_df["x"] = plot_df["x"] + X_STEP * quantile_idx # quantile_idx
            plot_df["x"] = quantile_idx
            plot_df["label"] = plot_df.label + f"{data_col} (Q={quantile_idx})"
            plot_df["data_col"] = data_col
            plot_df["prior_data_col"] = prior_col
            plot_col_dfs.append(plot_df)
    return pd.concat(plot_col_dfs)


def plotQuantiles(dfs_dfs_labels_li, min_num_traces_per_sess,
                  method : Literal["sum", "mean"], threshold=None,
                  save_fig=False, fig_save_prefix=None):

    if save_fig:
        assert fig_save_prefix is not None, "must specify fig_save_prefix"
    num_cols = len(dfs_dfs_labels_li)
    ex_df = dfs_dfs_labels_li[0][0]
    num_rows = len(ex_df[["data_col", "prior_data_col"]].value_counts(
                                                                  dropna=False))
    fig, axs = plt.subplots(num_rows, num_cols,
                            figsize=(num_cols*10, num_rows*8))
    if axs.ndim == 1:
        axs = np.array([axs])
    for idx, (df_plot, df_label) in enumerate(dfs_dfs_labels_li):
        count = 0
        if "group" not in df_plot.columns:
            df_plot = df_plot.copy()
            df_plot["group"] = 0
        for (data_col, prior_data_col), plot_df in df_plot.groupby(["data_col",
                                                              "prior_data_col"],
                                                              dropna=False):
            # fig, ax = plt.subplots(figsize=(20, 10))
            ax = axs[count, idx]
            offset = 0
            ls = "solid"
            for br, br_df in plot_df.groupby("BrainRegion"):
                main_df = br_df[br_df.group == 0]
                rand_dfs = br_df[br_df.group != 0]
                for ls, dfs_grps in [("solid", main_df), ("dashed", rand_dfs)]:
                    if not len(dfs_grps):
                        assert dfs_grps is rand_dfs, "Empty main data?"
                        continue
                    _plotQuantileGroup(dfs_grps,
                                min_num_traces_per_sess=min_num_traces_per_sess,
                                offset=offset, method=method,
                                threshold=threshold, br=br, ls=ls, ax=ax)
                offset += 0.01
                # offset += 0.4
            ax.set_xticks(sorted(plot_df.x.unique()))
            ax.legend(fontsize="small")
            title = f"{df_label} - {data_col} - Prior={prior_data_col}"
            if min_num_traces_per_sess:
                title += f" - Min Traces/Sess: {min_num_traces_per_sess}"
            else:
                title += "(no. min num traces/sess filtering applied)"
            ax.set_title(title)
            count += 1
    if save_fig:
        if threshold is not None:
            threshold_str = f"_threshold_{threshold}"
        else:
            threshold_str = "_without_threshold_filtering"
        save_fp = (f"{fig_save_prefix}/RT_Stats/"
                   f"sgf_quantiles_reaction_time{threshold_str}_"
                   f"min_traces_{min_num_traces_per_sess}.pdf")
        print("Saving to:", save_fp)
        fig.savefig(save_fp)
    plt.show()

def _plotQuantileGroup(grps_df, min_num_traces_per_sess, offset, method,
                        threshold, br, ls, ax):
    (xs, ys, ys_sem, ys_err, ys_err_sem, clrs, mean_sgf_count, mean_total_count,
     org_num_sess, num_sess_used) = _plotQuantileGetAvg(grps_df,
                                                        min_num_traces_per_sess,
                                                        method,
                                                        offset)
    # print("mean_sgf_count len:", len(mean_sgf_count), "shape:",
    #       np.array(sgf_count_li).shape)
    mean_sgf_total_str = [#f"Q{q_idx}={100*mean_sgf/mean_total:.2g}%"
                          f"Q{q_idx}={mean_sgf_q_count}/{mean_total_q_count}"
                          for q_idx, mean_sgf_q_count, mean_total_q_count in
                          zip(xs, mean_sgf_count, mean_total_count)]
    mean_sgf_total_str = ", ".join(mean_sgf_total_str)
    method_str = "All area" if method == "sum" else "Mean Sess."
    if threshold is not None:
        label = \
            f"{br} ({method_str} Active sgf/Total Active: {mean_sgf_total_str})"
    else:
        total = "Total" if method == "sum" else "Sess. Total"
        label = f"{br} ({method_str} sgf/{total}: {mean_sgf_total_str})"
    label += f" #Ses={num_sess_used}/{org_num_sess} Used"
    # print("Xs:", xs)
    err = ys_sem #if ys_sem[0] != 0 else ys_err
    print("Single sessoins y sem is not plotted")
    # print("Xs:", xs.shape)
    # print("Ys:", ys.shape)
    # print("Yerr:", err.shape)
    ax.errorbar(xs, ys, yerr=err, color=clrs.iloc[0], ls=ls, label=label)

def _plotQuantileGetAvg(grps_df, min_num_traces_per_sess, method, offset):
    xs_final, clrs_final = None, None
    y_total, y_err_total, mean_sgf_count_total, mean_total_count_total = \
                                                                  [], [], [], []
    final_org_num_sess, final_num_sess_used = None, None
    for grp_name, grp_df in grps_df.groupby("group"):
        xs, y_lis, labels, clrs = \
                      grp_df.quantile_idx, grp_df.y_li, grp_df.label, grp_df.clr
        sgf_count_li = grp_df.sgf_count_li
        total_count_li = grp_df.total_count_li
        # Convert to numpy array so we can filter too few trials out of them
        # print("br:", br, "before ys_lis:", y_lis.shape)
        # print("1. y_lis:", y_lis)
        y_lis, sgf_count_li, total_count_li = [
                                np.array([v for v in s.values])
                                for s in  [y_lis, sgf_count_li, total_count_li]]
        # print("br:", br, "ys_lis:", y_lis.shape)
        assert y_lis.shape == sgf_count_li.shape
        assert y_lis.shape == total_count_li.shape
        assert (abs(y_lis - 100*(sgf_count_li/total_count_li)) < 0.0001).all()
        too_few_traces = total_count_li < min_num_traces_per_sess
        del y_lis # No need for it here
        first_dim, org_num_sess = total_count_li.shape
        assert first_dim == 3, "Are you not breaking by 3 quantiles?"
        total_count_li = total_count_li[~too_few_traces].reshape(first_dim, -1)
        sgf_count_li = sgf_count_li[~too_few_traces].reshape(first_dim, -1)
        # y_lis = y_lis[~too_few_traces].reshape(first_dim, -1)
        num_sess_used = total_count_li.shape[1]
        if final_num_sess_used is not None:
            assert final_num_sess_used == num_sess_used, (
                                             f"Shouldn't change between groups")
            assert final_org_num_sess == org_num_sess
        else:
            final_org_num_sess = org_num_sess
            final_num_sess_used = num_sess_used
        if method == "sum":
            sgf_count_li = sgf_count_li.sum(axis=1)
            total_count_li = total_count_li.sum(axis=1)

        mean_sgf_count = [np.mean(counts) for counts in sgf_count_li]
        mean_total_count = [np.mean(counts) for counts in total_count_li]
        mean_sgf_count_total.append(mean_sgf_count)
        mean_total_count_total.append(mean_total_count)
        y = 100.0 * sgf_count_li / total_count_li
        # print("Y:", y)
        # y_err = [stats.sem(y_li) for y_li in y_lis]
        y_err = stats.sem(y, axis=1) if method == "mean" else [0]*3
        if method == "mean":
            y = y.mean(axis=-1)
        xs += offset
        xs = xs.values
        if xs_final is None:
            xs_final = xs
            clrs_final = clrs
        else:
            assert (xs_final == xs).all()
        y_total.append(y)
        y_err_total.append(y_err)

    assert len(y_total), "Empty df passed?"
    ys_final = np.mean(y_total, axis=0)
    ys_err_final = np.mean(y_err_total, axis=0)
    final_mean_sgf_count = np.mean(mean_sgf_count_total, axis=0)
    final_mean_total_count = np.mean(mean_total_count_total, axis=0)
    if len(y_total) > 1:
        ys_sem_final = stats.sem(y_total, axis=0)#np.std(y_total, axis=0)
        ys_err_sem_final = stats.sem(y_err_total, axis=0)
    else:
        ys_sem_final = np.array([0]*ys_final.shape[0])
        ys_err_sem_final = np.array([0]*ys_final.shape[0])
        mean_sgf_count_total = [mean_sgf_count_total]
        mean_total_count_total = [mean_total_count_total]
    final_mean_sgf_count = np.mean(mean_sgf_count_total, axis=0).squeeze()
    final_mean_total_count = np.mean(mean_total_count_total, axis=0).squeeze()
    return (xs_final, ys_final, ys_sem_final, ys_err_final, ys_err_sem_final,
            clrs_final, final_mean_sgf_count, final_mean_total_count,
            final_org_num_sess, final_num_sess_used)


def plotQuantilesDVStr(df, pval, left_tuning_col):
    # return
    # display(df)
    df_plot_li = []
    for quantile_idx, quantile_df in df.groupby("quantile_idx"):
        for dv_str, difficulty_df in quantile_df.groupby("DVstr"):
        # try:
            quantile_idx = int(quantile_idx)
            # print("Quantile idx:", quantile_idx)
            test_df = difficulty_df[difficulty_df.data_col == "ChoiceLeft"]
            # Next line shouldn't make a difference here
            test_df = test_df[test_df.prior_data_col.isnull()]
            ref_df = None#epoch_col_df_no_priors
            # rocplot.loopPlotResults(test_df, pval=PVAL,
            #                         left_tuning_col=left_tuning_col,
            #                         active_traces_df=active_traces_df,
            #                         fig_save_prefix=None)
            plot_df = rocplot._countSigfTraces(test_df, pval=pval,
                                               left_tuning_col=left_tuning_col,
                                            # active_traces_df=active_traces_df,
                                               active_traces_df=None,
                                               ref_count_df=ref_df)
            plot_df["quantile_idx"] = quantile_idx
            plot_df["x"] = quantile_idx#plot_df["x"] + X_STEP * quantile_idx
            plot_df["label"] =    plot_df.label + f"{dv_str} (Q={quantile_idx})"
            plot_df["DVstr"] = dv_str
            df_plot_li.append(plot_df)
        # except Exception as e:
        #     pass
    if len(df_plot_li):
        df_plot = pd.concat(df_plot_li)
        _plotRes(df_plot)

def _plotRes(df_plot):
    fig, ax = plt.subplots(figsize=(20, 10))
    offset = 0
    for dv_str, dv_df in df_plot.groupby("DVstr"):
        if dv_str == "Easy":
            ls = "solid"
        elif dv_str == "Hard":
            ls = "dashed"
        else:
            ls = "dotted"
        for br, br_df in dv_df.groupby("BrainRegion"):
            xs, y_lis, clrs = br_df.x, br_df.y_li, br_df.clr
            xs += offset
            y_means = [np.array(y_li).mean() for y_li in y_lis]
            y_errs = [stats.sem(y_li) for y_li in y_lis]
            ax.errorbar(xs, y_means, yerr=y_errs, color=clrs.iloc[0],
                        ls=ls, label=f"{br} {dv_str}")
            offset += 0.01
    ax.legend()
    plt.show()


def _plotSide(choice_dir, all_df, preferred_df, non_preferred_df, non_sgf_df,
              trace_count_all, trace_count_preferred,
              trace_count_non_preferred, trace_count_non_sgf,
              ax, method :Literal["mean", "sum"], sep_side, STRIDE,
              already_printed_count, plot_all=True, plot_non_sgf=True,
              group_neurons_trials=False):
    if group_neurons_trials:
        all_df = all_df.groupby("long_trace_id")
        preferred_df = preferred_df.groupby("long_trace_id")
        non_preferred_df = non_preferred_df.groupby("long_trace_id")
        non_sgf_df = non_sgf_df.groupby("long_trace_id")

    mean_all = getattr(all_df.MeanActivity, method)()
    mean_pref = getattr(preferred_df.MeanActivity, method)()
    mean_non_pref =    getattr(non_preferred_df.MeanActivity, method)()
    mean_non_sgf = getattr(non_sgf_df.MeanActivity, method)()
    # Calculate SEM
    if group_neurons_trials:
        # sem_all: mean of sum or mean of mean
        mean_all,      sem_all = mean_all.mean(), mean_all.sem()
        mean_pref,     sem_pref = mean_pref.mean(), mean_pref.sem()
        mean_non_pref, sem_non_pref = mean_non_pref.mean(), mean_non_pref.sem()
        mean_non_sgf,  sem_non_sgf= mean_non_sgf.mean(), mean_non_sgf.sem()
    else:
        if method == "mean":
            sem_all =      stats.sem(all_df.MeanActivity)
            sem_pref =     stats.sem(preferred_df.MeanActivity)
            sem_non_pref = stats.sem(non_preferred_df.MeanActivity)
            sem_non_sgf =  stats.sem(non_sgf_df.MeanActivity)
        else:
            sem_all, sem_pref, sem_non_pref, sem_non_sgf = \
                                                          None, None, None, None
    # print("Choice dir:", choice_dir, "quantile_idx:", quantile_idx,
    #       "br:", br, "- len:", len(choice_dir_df))
    # print("preferred_dir_df:", len(preferred_dir_df), "mean:", mean_pref)
    # print("non_preferred_dir_df:", len(non_preferred_dir_df),
    #       "mean:", mean_non_pref)
    # print("non_sgf_dir_df:", len(non_sgf_dir_df), "mean:", mean_non_sgf)
    # print()
    print_labels_once = already_printed_count == 0 #and choice_dir
    dir_str = (" Left" if choice_dir else " Right") if sep_side else ""
    hatch = None if not sep_side or choice_dir == 1 else "/"
    width = 0.5 if sep_side else 1
    x = (already_printed_count+1)*STRIDE + (choice_dir*width if sep_side else 0)
    if plot_all:
        ax.bar(x - 2, mean_all, yerr=sem_all,           color="w",
               label=(f"All{dir_str} (#N={trace_count_all:,})"
                      if print_labels_once else None),
               edgecolor="black", hatch=hatch, width=width)
    ax.bar(x - 1,     mean_pref, yerr=sem_pref,         color="green",
           label=(f"Preferred{dir_str} (#N={trace_count_preferred:,})"
                  if print_labels_once else None),
           edgecolor="black", hatch=hatch, width=width)
    ax.bar(x,         mean_non_pref, yerr=sem_non_pref, color="orange",
           label=(f"Non-Preferred{dir_str} (#N={trace_count_non_preferred:,})"
                  if print_labels_once else None),
           edgecolor="black", hatch=hatch, width=width)
    if plot_non_sgf:
        ax.bar(x + 1, mean_non_sgf, yerr=sem_non_sgf,   color="gray",
               label=(f"Non-Sgf{dir_str} (#N={trace_count_non_sgf:,})"
                      if print_labels_once else None),
               edgecolor="black", hatch=hatch, width=width)

def _runSgfTest(df1, df2, method):
    df1 = df1.groupby("long_trace_id")
    df2 = df2.groupby("long_trace_id")
    df1_dict = {"long_trace_id":[], "metric1":[]}
    df2_dict = {"long_trace_id":[], "metric2":[]}
    for trace_id, grp_by in df1:
        metric = getattr(grp_by.MeanActivity, method)()
        df1_dict["long_trace_id"].append(trace_id)
        df1_dict["metric1"].append(metric)
    for trace_id, grp_by in df2:
        metric = getattr(grp_by.MeanActivity, method)()
        df2_dict["long_trace_id"].append(trace_id)
        df2_dict["metric2"].append(metric)
    df1 = pd.DataFrame(df1_dict)
    df2 = pd.DataFrame(df2_dict)
    assert len(df1) == len(df2)
    df_merged = df1.merge(df2, on="long_trace_id")
    assert len(df_merged) == len(df1)
    test_res = stats.ttest_rel(df_merged.metric1, df_merged.metric2)
    print("test_res:", test_res.pvalue)
    return test_res.pvalue


def _processSamplingSlowVsFast(df, df_dscrp_str, method :Literal["mean", "sum"],
                               sep_side, label, save_fig, fig_save_prefix=None,
                               skip_qidx2=False, plot_all=True,
                               plot_non_sgf=True, group_neurons_trials=False):
    already_printed_count = 0
    x_ticks = []
    x_ticks_labels = []
    STRIDE = 3
    if plot_all:
        STRIDE += 1
    if plot_non_sgf:
        STRIDE += 1
    fig, ax = plt.subplots(figsize=(4*STRIDE, 10))
    for quantile_idx, quantile_df in df.groupby("quantile_idx"):
        if skip_qidx2 and quantile_idx == 2:
            continue
        if not sep_side:
            all_li, preferred_li, non_preferred_li, non_sgf_li = [], [], [], []
            trace_count_all_li, trace_count_preferred_li = [], []
            trace_count_non_preferred_li, trace_count_non_sgf_li = [], []
        for choice_dir, choice_dir_df in quantile_df.groupby(
                                                        quantile_df.ChoiceLeft):
            # trials_sgf_df = choice_dir_df[choice_dir_df.long_trace_id.isin(
            #                                         sgf_choice.long_trace_id)]
            # TODO: Split by ShortName
            non_choice_dir = (choice_dir + 1) % 2
            preferred_dir_df =  choice_dir_df[
                               choice_dir_df.is_left_choice_tuned == choice_dir]
            non_preferred_dir_df = choice_dir_df[
                           choice_dir_df.is_left_choice_tuned == non_choice_dir]
            non_sgf_dir_df = choice_dir_df[
                                    choice_dir_df.is_left_choice_tuned.isnull()]
            trace_count_all = choice_dir_df.long_trace_id.nunique()
            trace_count_preferred = preferred_dir_df.long_trace_id.nunique()
            trace_count_non_preferred = \
                                    non_preferred_dir_df.long_trace_id.nunique()
            trace_count_non_sgf = non_sgf_dir_df.long_trace_id.nunique()
            if sep_side:
                _plotSide(choice_dir, choice_dir_df,
                          preferred_dir_df, non_preferred_dir_df,
                          non_sgf_dir_df, trace_count_all,
                          trace_count_preferred,  trace_count_non_preferred,
                          trace_count_non_sgf, ax=ax, method=method,
                          sep_side=sep_side,
                          already_printed_count=already_printed_count,
                          STRIDE=STRIDE, plot_all=plot_all,
                          plot_non_sgf=plot_non_sgf,
                          group_neurons_trials=group_neurons_trials)
            else:
                all_li.append(choice_dir_df)
                preferred_li.append(preferred_dir_df)
                non_preferred_li.append(non_preferred_dir_df)
                non_sgf_li.append(non_sgf_dir_df)
                # Track counts as well
                trace_count_all_li.append(trace_count_all)
                trace_count_preferred_li.append(trace_count_preferred)
                trace_count_non_preferred_li.append(trace_count_non_preferred)
                trace_count_non_sgf_li.append(trace_count_non_sgf)
        if not sep_side:
            # Converts counts to np arrays and take the mean
            trace_count_all = np.array(trace_count_all_li).mean()
            trace_count_preferred = np.array(trace_count_preferred_li).mean()
            trace_count_non_preferred = np.array(
                                            trace_count_non_preferred_li).mean()
            trace_count_non_sgf = np.array(trace_count_non_sgf_li).mean()
            all_df = pd.concat(all_li)
            preferred_df = pd.concat(preferred_li)
            non_preferred_df = pd.concat(non_preferred_li)
            non_sgf_df = pd.concat(non_sgf_li)
            if group_neurons_trials:
                # display(preferred_dir_df)
                print("trace_count_preferred:", trace_count_preferred,
                      "- preferred len:", len(preferred_dir_df))
                grp_pval = _runSgfTest(preferred_df, non_preferred_df,
                                       method=method)
            _plotSide(None, all_df, preferred_df, non_preferred_df, non_sgf_df,
                      trace_count_all, trace_count_preferred,
                      trace_count_non_preferred, trace_count_non_sgf,
                      ax=ax, method=method,    sep_side=sep_side,
                      already_printed_count=already_printed_count,
                      STRIDE=STRIDE, plot_all=plot_all,
                      plot_non_sgf=plot_non_sgf,
                      group_neurons_trials=group_neurons_trials)
            already_printed_count += 1
            x_ticks.append(already_printed_count*STRIDE)
            x_ticks_labels.append(f"Q={quantile_idx}")
        # print()
    ax.legend()
    method_str = method[0].upper() + method[1:]
    ax.set_title(f"{df_dscrp_str} - {method_str} of Mean Activity by Quantile "
                 f"and Choice Preference {label}")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_labels)
    ax.set_xlabel("Quantile")
    ax.set_ylabel(f"{method_str} of Mean Z-Scored Activity")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    if save_fig:
        sep_side_str = "sep_side" if sep_side else "comb_side"
        trials_group_str = "trials_grouped" if group_neurons_trials else \
                           "without_trials_grouping"
        fig.savefig(f"{fig_save_prefix}/RT_Stats/bias_{label}_{method_str}_"
                    f"of_mean_activity_quantiles_{df_dscrp_str}_"
                    f"{sep_side_str}_{trials_group_str}.pdf")
    plt.show()

def _setSrcDfTuning(src_activity_df, stats_df, left_tune_col, pval):
    old_len = len(src_activity_df)
    if "quantile_idx" in stats_df and stats_df.quantile_idx.nunique() > 1:
        df_li = []
        src_activity_df["is_left_choice_tuned"] = -1
        for q_idx, stats_q_df in stats_df.groupby("quantile_idx"):
            left_sgf_choice_df, right_sgf_choice_df = _getSgfLeftRight(
                                                    stats_df=stats_q_df,
                                                    left_tune_col=left_tune_col,
                                                    pval=pval)
            cur_src_activity_df = src_activity_df[
                                          src_activity_df.quantile_idx == q_idx]
            cur_src_activity_df = _setSrcSubDfTuning(cur_src_activity_df,
                                                     left_sgf_choice_df,
                                                     right_sgf_choice_df)
            df_li.append(cur_src_activity_df)
        src_activity_df = pd.concat(df_li)
    else:
        left_sgf_choice_df, right_sgf_choice_df = _getSgfLeftRight(
                                                    stats_df=stats_df,
                                                    left_tune_col=left_tune_col,
                                                    pval=pval)
        src_activity_df = _setSrcSubDfTuning(src_activity_df,
                                             left_sgf_choice_df,
                                             right_sgf_choice_df)
    assert len(src_activity_df) == old_len
    assert (src_activity_df.is_left_choice_tuned == -1).sum() == 0
    display(src_activity_df.is_left_choice_tuned.value_counts(dropna=False))
    return src_activity_df

def _setSrcSubDfTuning(src_activity_df, left_sgf_choice_df,
                       right_sgf_choice_df):
    # Convert the other df to have the same left vs right sgf.
    src_activity_df = src_activity_df.copy()
    src_activity_df["is_left_choice_tuned"] = np.nan
    src_activity_df.loc[src_activity_df.long_trace_id.isin(
                                              left_sgf_choice_df.long_trace_id),
                        "is_left_choice_tuned"] = 1
    src_activity_df.loc[src_activity_df.long_trace_id.isin(
                                             right_sgf_choice_df.long_trace_id),
                        "is_left_choice_tuned"] = 0
    return src_activity_df

def _getSgfLeftRight(stats_df, left_tune_col, pval):
    non_sgf_choice_df = stats_df[stats_df.pval > pval]
    sgf_choice_df = stats_df[stats_df.pval <= pval]
    # display(sgf_choice)
    left_sgf_choice_df    = sgf_choice_df[sgf_choice_df[left_tune_col] == True]
    right_sgf_choice_df = sgf_choice_df[sgf_choice_df[left_tune_col] == False]
    # Assert first no repeated neurons
    non_sgf_choice_df = non_sgf_choice_df.sort_values("long_trace_id")
    assert non_sgf_choice_df.long_trace_id.nunique() == len(non_sgf_choice_df),(
        display(non_sgf_choice_df[non_sgf_choice_df.duplicated("long_trace_id",
                                                               keep=False)]))
    assert right_sgf_choice_df.long_trace_id.nunique() == len(
                                                            right_sgf_choice_df)
    assert left_sgf_choice_df.long_trace_id.nunique() == len(left_sgf_choice_df)
    # Assert no overlap between the groups
    assert left_sgf_choice_df.long_trace_id.isin(
                                   non_sgf_choice_df.long_trace_id).sum() == 0,(
                                                    display(left_sgf_choice_df))
    assert right_sgf_choice_df.long_trace_id.isin(
                                     non_sgf_choice_df.long_trace_id).sum() == 0
    assert left_sgf_choice_df.long_trace_id.isin(
                                   right_sgf_choice_df.long_trace_id).sum() == 0
    return left_sgf_choice_df, right_sgf_choice_df

def samplingSlowVsFast(stats_df, src_activitiy_df, left_tune_col,
                       method : Literal["mean", "sum"], pval, sep_side,
                       sep_brain_region : bool, label, save_fig,
                       fig_save_prefix=None, skip_qidx2=False, plot_all=True,
                       plot_non_sgf=True, group_neurons_trials=False):
    '''
    from . import
    src_activitiy_df = expanAlignDF(df_rt_sampling_fixed.copy())
    '''
    assert method in ("mean", "sum")
    if save_fig:
        assert fig_save_prefix is not None

    src_activitiy_df = src_activitiy_df.copy()
    src_activitiy_df = src_activitiy_df[src_activitiy_df.Layer == "L23"]
    src_activitiy_df = src_activitiy_df[src_activitiy_df.BrainRegion.isin(
                                       [BrainRegion.M2_Bi, BrainRegion.ALM_Bi])]

    src_activitiy_df = _setSrcDfTuning(src_activitiy_df, stats_df,
                                       left_tune_col, pval)

    if sep_brain_region:
        for br, br_df in src_activitiy_df.groupby("BrainRegion"):
            br_str = str(BrainRegion(br)).split("_")[0]
            _processSamplingSlowVsFast(br_df, df_dscrp_str=br_str,
                                       method=method, sep_side=sep_side,
                                       label=label,
                                      group_neurons_trials=group_neurons_trials,
                                       skip_qidx2=skip_qidx2,
                                       plot_all=plot_all,
                                       plot_non_sgf=plot_non_sgf,
                                       save_fig=save_fig,
                                       fig_save_prefix=fig_save_prefix)
    else:
        df_dscrp_str = "Brain-Regions Combined"
        _processSamplingSlowVsFast(src_activitiy_df, df_dscrp_str=df_dscrp_str,
                                   method=method, sep_side=sep_side,
                                   label=label,
                                   group_neurons_trials=group_neurons_trials,
                                   skip_qidx2=skip_qidx2,
                                   plot_all=plot_all, plot_non_sgf=plot_non_sgf,
                                   save_fig=save_fig,
                                   fig_save_prefix=fig_save_prefix)


def loopEveryWay(shortlong_df, shortlong_quantile_df, pval, active_traces_df,
                 active_thresh_str, active_quantiles_traces_df,
                 active_quantiles_thresh_str):
    '''Everyway we have is:
    Which neurons to choose:
    1. Neurons that are signficant in all the sessions trials
    2. Neurons that are signficant only in their own quantile
    3. Neurons that are common to overall and to their own quantile
    4. Neurons that are signficant in all quantiles?
    5. Neurons that are common to overall and all quantiles?

    Which subneurons to choose:
    a. for whatever we are measuring
    b. only choice neurons

    Which subsubneurons to choose:
    i. for only active neurons above threshold
    ii. all neurons

    For priors:
    - Sgf. under prior condition regardless of sgf the prior itself
    - Sgf. under prior condition regardless but split by prior condition
    - Sgf. same but also sgf. for prior AND matching prior tuning

    - Things we can measure:
        i. Percentage of neurons
        ii. coding efficiency?

    Filters:
    - Min traces per sess?
    - Active neurons threshold'''
    results_dict = {}
    for (data_col, prior_data_col), grp_Df in \
             shortlong_df.groupby(["data_col", "prior_data_col"], dropna=False):
        print(f"Processing {data_col}/Prior={prior_data_col}")
        count = 0
        for descrp_sgf, (sgf_df, sgf_choice_df, sgf_prior_df, is_per_quantile) in _splitBySigfNeurons(shortlong_df, shortlong_quantile_df, data_col, prior_data_col, pval).items():
            for data_prior_descrp, data_prior_df in _splitPriorNeuronsperDF(sgf_df, sgf_prior_df, data_col, prior_data_col, pval).items():
                for dscrp_only_choice, any_or_only_choice_df in _splitByByAnyOrOnlyChoice(data_prior_df, sgf_choice_df, data_col, prior_data_col).items():
                    for active_or_not_dscrp, (active_or_not_df, activity_df_or_none) in _splotByAnyOrOnlyActive(any_or_only_choice_df, is_per_quantile, active_traces_df, active_thresh_str, active_quantiles_traces_df, active_quantiles_thresh_str).items():
                        all_dscrp = data_col, str(prior_data_col), descrp_sgf, data_prior_descrp, dscrp_only_choice, active_or_not_dscrp
                        # print(all_dscrp, "- Len:", len(active_or_not_df))
                        results_dict[all_dscrp] = active_or_not_df, activity_df_or_none
                        count += 1
        # print("Count:", count)
        # print()
    return results_dict

def _commonSgfFilter(df, data_col, prior_data_col, pval, prior_data_val=None):
    df = df[df.DVstr.isnull()]
    df = df[df.pval <= pval]
    df = df[df.data_col == data_col]
    if isinstance(prior_data_col, float) and np.isnan(prior_data_col):
        df = df[df.prior_data_col.isnull()]
    else:
        df = df[df.prior_data_col == prior_data_col]
        if prior_data_val is not None:
            df = df[df.prior_data_val == prior_data_val]
    return df

def _splitByByAnyOrOnlyChoice(df, choice_df, data_col, prior_data_col):
    if data_col != "ChoiceLeft" and (
        pd.isnull(prior_data_col) or (isinstance(prior_data_col, str) and
                                      prior_data_col != "ChoiceLeft")):
        choice_df_traces_ids = choice_df.long_trace_id.unique()
        splits = {"(regardless of whether sgf. under choice)":df,
                  "for only neurons sgf. under choice":df[
                                   df.long_trace_id.isin(choice_df_traces_ids)],
        }
    else: # This will be split later by the priors split function
        splits = {"":df,
        }
    return splits

def _splitBySigfNeurons(shortlong_df, shortlong_quantile_df, data_col,
                        prior_data_col, pval):
    shortlong_sgf_df = _commonSgfFilter(shortlong_df, data_col,
                                        prior_data_col,
                                        pval=pval)
    shortlong_quantile_sgf_df = _commonSgfFilter(shortlong_quantile_df, data_col,
                                                 prior_data_col,
                                                 pval=pval)

    shortlong_sgf_choice_df = _commonSgfFilter(shortlong_df, "ChoiceLeft",
                                               prior_data_col=np.nan,
                                               pval=pval)
    shortlong_sgf_quantiled_choice_df = _commonSgfFilter(shortlong_quantile_df,
                                                         "ChoiceLeft",
                                                         prior_data_col=np.nan,
                                                         pval=pval)
    def getQuantiledInOverall(quantlid_df, overall_df):
        return quantlid_df[quantlid_df.long_trace_id.isin(
                                             overall_df.long_trace_id.unique())]
    shortlong_quantiled_in_overall_sgf_df = getQuantiledInOverall(
                                    shortlong_quantile_sgf_df, shortlong_sgf_df)
    shortlong_sgf_quantiled_in_overall_choice_df = getQuantiledInOverall(
                     shortlong_sgf_quantiled_choice_df, shortlong_sgf_choice_df)
    if isinstance(prior_data_col, str):
        shortlong_sgf_prior_df = _commonSgfFilter(shortlong_df, prior_data_col,
                                                  prior_data_col=np.nan,
                                                  pval=pval)
        shortlong_quantiled_sgf_prior_df = _commonSgfFilter(
                                                        shortlong_quantile_df,
                                                        prior_data_col,
                                                        prior_data_col=np.nan,
                                                        pval=pval)
        shortlong_quantiled_in_overall_sgf_prior_df = getQuantiledInOverall(
                       shortlong_quantiled_sgf_prior_df, shortlong_sgf_prior_df)
    else:
        shortlong_sgf_prior_df = None
        shortlong_quantiled_sgf_prior_df = None
        shortlong_quantiled_in_overall_sgf_prior_df = None

    IS_PER_QUANTILE = True
    NOT_PER_QUANTILED = not IS_PER_QUANTILE
    splits = {
        "Sgf. in all trials":(shortlong_sgf_df, shortlong_sgf_choice_df,
                              shortlong_sgf_prior_df, NOT_PER_QUANTILED),
        "Sgf. in each quantile":(shortlong_quantile_sgf_df,
                                 shortlong_sgf_quantiled_choice_df,
                                 shortlong_quantiled_sgf_prior_df,
                                 IS_PER_QUANTILE),
        "Sgf. in each quantile AND in all trials":(
                                 shortlong_quantiled_in_overall_sgf_df,
                                 shortlong_sgf_quantiled_in_overall_choice_df,
                                 shortlong_quantiled_in_overall_sgf_prior_df,
                                 IS_PER_QUANTILE),
    }
    return splits

def _splotByAnyOrOnlyActive(df, is_per_quantile, active_traces_df,
                            activity_thresh_str, active_quantiles_traces_df,
                            active_quantiles_thresh_str):
    # active_uniq = active_traces_df.long_trace_id.unique()
    # df_active = df[df.long_trace_id.isin(active_uniq)]
    splits = {
        "regardless of neuron Peak activity position": (df, None),
        f"for only active (session overall) neurons: {activity_thresh_str}":
                                                         (df, active_traces_df),
    }
    if is_per_quantile:
        key = \
         f"for only active (in quantile) neurons: {active_quantiles_thresh_str}"
        splits.update({
            key:(df, active_quantiles_traces_df)
        })
    return splits


def _splitPriorNeuronsperDF(sgf_df, sgf_prior_df, data_col, prior_data_col,
                            pval):
    if isinstance(prior_data_col, float) and np.isnan(prior_data_col):
        return {f"sgf. for {data_col}":sgf_df}

    sgf_priors_traces_ids = sgf_prior_df.long_trace_id.unique()
    priors_data_vals = sgf_df[(sgf_df.data_col == data_col) &
                              (sgf_df.prior_data_col == prior_data_col)]\
                                                                 .prior_data_val
    priors_data_vals = priors_data_vals.unique()
    assert len(priors_data_vals) == 2, (
            f"Didn't implement for no. of prior values={len(priors_data_vals)}")
    prior_val_left, prior_val_right = priors_data_vals
    sgf_data = _commonSgfFilter(sgf_df, data_col=data_col,
                                prior_data_col=prior_data_col,
                                pval=pval)
    sgf_data_left = _commonSgfFilter(sgf_df, data_col=data_col,
                                     prior_data_col=prior_data_col,
                                     prior_data_val=prior_val_left,
                                     pval=pval)
    sgf_data_right = _commonSgfFilter(sgf_df, data_col=data_col,
                                      prior_data_col=prior_data_col,
                                      prior_data_val=prior_val_right,
                                      pval=pval)
    sgf_data_and_prior = sgf_data[
                             sgf_data.long_trace_id.isin(sgf_priors_traces_ids)]
    sgf_data_left_and_prior = sgf_data_left[
                        sgf_data_left.long_trace_id.isin(sgf_priors_traces_ids)]
    sgf_data_right_and_prior = sgf_data_right[
                       sgf_data_right.long_trace_id.isin(sgf_priors_traces_ids)]
    splits = {
        f"sgf. for {data_col}/{prior_data_col}=ANY regardless if {prior_data_col}=Any is sgf.":sgf_data,
        # f"sgf. for {data_col}/{prior_data_col}={prior_val_left} regardless if {prior_data_col}={prior_val_left} is sgf.":sgf_data_left,
        # f"sgf. for {data_col}/{prior_data_col}={prior_val_right} regardless if {prior_data_col}={prior_val_right} is sgf.":sgf_data_right,
        f"sgf. for {data_col}/{prior_data_col}=ANY *AND* {prior_data_col}=Any is sgf.":sgf_data_and_prior,
        # f"sgf. for {data_col}/{prior_data_col}={prior_val_left} *AND* {prior_data_col}={prior_val_left} is sgf.":sgf_data_left_and_prior,
        # f"sgf. for {data_col}/{prior_data_col}={prior_val_right} *AND* {prior_data_col}={prior_val_right} is sgf.":sgf_data_right_and_prior,
    }
    return splits

def _getSgf(sgf_traces_df, expanded_left_df, expanded_right_df,
            active_neurons_ids_set):
    n_sgf = len(sgf_traces_df)
    n_actve_sgf = np.sum([(trace_id in active_neurons_ids_set)
                          for trace_id in sgf_traces_df.long_trace_id])
    sgf_expanded_left = expanded_left_df[expanded_left_df.long_trace_id.isin(
                                                   sgf_traces_df.long_trace_id)]
    sgf_expanded_right = expanded_right_df[expanded_right_df.long_trace_id.isin(
                                                   sgf_traces_df.long_trace_id)]
    return n_sgf, n_actve_sgf, sgf_expanded_left, sgf_expanded_right

def _getTracesMeanActivity(expanded_df, quantile_idx):
        cur_expd = expanded_df[expanded_df.quantile_idx == quantile_idx]
        n_uniq_traces = cur_expd.long_trace_id.nunique()
        rows_li = []
        for trace_id, trace_df in cur_expd.groupby("long_trace_id"):
            raw_traces_2d = trace_df.Raw
            trace_mean = raw_traces_2d.mean(axis=0)
            ex_row = trace_df.iloc[0].copy()
            ex_row = ex_row[["long_trace_id"]]
            ex_row["Raw"] = trace_mean
            rows_li.append(ex_row)
        res_df = pd.DataFrame(rows_li)
        res_df.columns = res_df.columns.to_flat_index()
        # display(res_df)
        assert len(res_df) == n_uniq_traces, f"{n_uniq_traces} != {len(res_df)}"
        return res_df

def plotQuantilesBrainRegionAUC(stats_df, expanded_left_df,
                                expanded_right_df, pval,
                                plot_special, active_df=None,
                                big_diff_neurons=None):
    # df_li = []
    # merged_expanded = expanded_left.merge(expanded_right,
    #                                      on=["long_trace_id", "quantile_idx"],
    #                                      suffixes=["_left", "_right"])
    # display(merged_expanded)
    res_dict = {"BrainRegion":[], "ShortName":[], "quantile_idx":[],
                "long_trace_id":[], "mean_statistic_auc":[], "is_active":[],
                "n_all":[], "n_active":[], "n_sgf":[], "n_actve_sgf":[]
    }
    # When we finish this block, if stats_df cotnains quantile_idx column, then
    # active_df will be a dict mapping quantile idx to set of active neurons
    active_has_quantiles = False
    stats_has_quantiles = "quantile_idx" in stats_df.columns
    if active_df is not None:
        if "long_trace_id" not in active_df:
            print("Fix-Me")
            active_df = active_df.copy()
            active_df["long_trace_id"] = active_df.trace_id
        if "quantile_idx" in active_df.columns and \
             len(active_df.quantile_idx.unique()) == 1 and \
             active_df.quantile_idx.iloc[0] == "All":
            active_df = active_df.drop(columns=["quantile_idx"])
        if "quantile_idx" in active_df.columns:
            assert "quantile_idx" in stats_df.columns
            active_quantile_idxs = active_df["quantile_idx"].unique()
            assert len(active_quantile_idxs) > 1, active_quantile_idxs
            active_neurons_ids = {quantile_idx:set(
                active_df[active_df.quantile_idx == quantile_idx].long_trace_id)
                for quantile_idx in active_quantile_idxs}
            active_has_quantiles = True
        else:
            active_neurons_ids = set(active_df.long_trace_id)
    else:
        # Add everything
        active_neurons_ids = set(expanded_right_df.long_trace_id)
    if stats_has_quantiles:
        if not active_has_quantiles:
            active_quantile_idxs = stats_df.quantile_idx.unique()
            # Repeat the values for each quantile
            active_neurons_q_ids = {quantile_idx:active_neurons_ids
                                    for quantile_idx in active_quantile_idxs}
            active_neurons_ids = active_neurons_q_ids
    else:
        assert not active_has_quantiles, ("Don't know how to handle active "
                                          "with quanitles to stats without "
                                          "quantiles")
    quantiles_idxs = expanded_left_df.quantile_idx.unique()
    getSgf_args = dict(expanded_left_df=expanded_left_df,
                                         expanded_right_df=expanded_right_df)
    for br, br_df in stats_df.groupby("BrainRegion"):
        for short_name, sess_df in br_df.groupby("ShortName"):
            expanded_sess = expanded_left_df[expanded_left_df.ShortName ==
                                             short_name]
            sess_all_traces = set(expanded_sess.long_trace_id)
            n_all = len(sess_all_traces)
            sgf_sess_df = sess_df[sess_df.pval <= pval]
            if not stats_has_quantiles:
                # Again this assert should have already ran, repeat here for
                # readability
                assert not active_has_quantiles
                cur_active_neuron_ids = active_neurons_ids
                n_active = np.sum([(trace_id in cur_active_neuron_ids)
                                               for trace_id in sess_all_traces])
                n_sgf, n_actve_sgf, sgf_expanded_left, sgf_expanded_right = \
                        _getSgf(sgf_sess_df,
                                active_neurons_ids_set=cur_active_neuron_ids,
                                **getSgf_args)
            for quantile_idx in quantiles_idxs:
                if stats_has_quantiles:
                    sgf_sess_quantile_df = sgf_sess_df[
                                       sgf_sess_df.quantile_idx == quantile_idx]
                    # Next line corner case that unfortunately happens
                    if quantile_idx in active_neurons_ids:
                        cur_active_neuron_ids = active_neurons_ids[quantile_idx]
                    else:
                        cur_active_neuron_ids = set()
                    n_active = np.sum([(trace_id in cur_active_neuron_ids)
                                        for trace_id in sess_all_traces])
                    n_sgf, n_actve_sgf, sgf_expanded_left, sgf_expanded_right =\
                        _getSgf(sgf_sess_quantile_df,
                                active_neurons_ids_set=cur_active_neuron_ids,
                                **getSgf_args)
                mean_left_df = _getTracesMeanActivity(sgf_expanded_left,
                                                      quantile_idx)
                mean_right_df = _getTracesMeanActivity(sgf_expanded_right,
                                                       quantile_idx)
                if not len(mean_left_df):
                    # print("*****Handle no sgf. traces in sessions")
                    res_dict["long_trace_id"].append(-1)
                    res_dict["mean_statistic_auc"].append(np.nan)
                    res_dict["quantile_idx"].append(quantile_idx)
                    res_dict["ShortName"].append(short_name)
                    res_dict["BrainRegion"].append(br)
                    res_dict["n_all"].append(n_all)
                    res_dict["n_sgf"].append(0)
                    res_dict["n_active"].append(n_active)
                    res_dict["n_actve_sgf"].append(0)
                    res_dict["is_active"].append(False)
                    continue
                merged_means_df = mean_left_df.merge(mean_right_df,
                                                   on=[("long_trace_id",'')],
                                                   suffixes=["_left", "_right"])
                assert len(mean_left_df) == len(mean_right_df)
                assert len(mean_left_df) == len(merged_means_df)
                # display(merged_means_df)
                left_raw = merged_means_df["('Raw', '')_left"].values
                right_raw = merged_means_df["('Raw', '')_right"].values
                mean_diff =    left_raw - right_raw
                mean_diff = np.array([row for row in mean_diff])

                long_traces_ids = merged_means_df[("long_trace_id", '')]
                mean_diff_sums = abs(mean_diff).sum(axis=1)
                assert len(mean_diff_sums) == len(merged_means_df)
                for trace_id, trace_mean_diff_sum in zip(long_traces_ids,
                                                         mean_diff_sums):
                    # res_dict["mean_statistic_auc"].append(rows_sum_mean)
                    if plot_special and trace_id in big_diff_neurons:
                        trace_left_rows = expanded_left_df[
                               (expanded_left_df.long_trace_id == trace_id) &
                               (expanded_left_df.quantile_idx == quantile_idx)]
                        trace_right_rows = expanded_right_df[
                               (expanded_right_df.long_trace_id == trace_id) &
                               (expanded_right_df.quantile_idx == quantile_idx)]
                        _plotSpecialNeurons(trace_id, quantile_idx,
                                            trace_left_rows, trace_right_rows)
                    res_dict["long_trace_id"].append(trace_id)
                    res_dict["mean_statistic_auc"].append(trace_mean_diff_sum)
                    res_dict["quantile_idx"].append(quantile_idx)
                    res_dict["ShortName"].append(short_name)
                    res_dict["BrainRegion"].append(br)
                    res_dict["n_all"].append(n_all)
                    res_dict["n_sgf"].append(n_sgf)
                    res_dict["n_active"].append(n_active)
                    res_dict["n_actve_sgf"].append(n_actve_sgf)
                    res_dict["is_active"].append(
                                              trace_id in cur_active_neuron_ids)
                        # df_li.append(res_df)
    # return pd.concat(df_li)
    return pd.DataFrame(res_dict)

def _getExpandedDf(stats_df, expanded_df):
    data_col = stats_df.data_col.unique()
    assert len(data_col) == 1, f"Expected only single data col, not: {data_col}"
    data_col = data_col[0]
    prior_data_col = stats_df.prior_data_col.unique()
    assert len(prior_data_col) <= 1, (
                    f"Expected zero or 1 prior data col, not: {prior_data_col}")
    if len(prior_data_col) and isinstance(prior_data_col, str) and \
         prior_data_col != "nan":
        prior_data_col = prior_data_col[0]
        has_prior = True
    else:
        has_prior = False

    if has_prior:
        prior_data_vals = stats_df.prior_data_vals.unique()
        if len(prior_data_vals) > 1:
            cur_expanded_df = expanded_df[expanded_df[prior_data_col].notnull()]
        else:
            assert len(prior_data_vals) == 1, f"len(0) for prior data vals?"
            cur_expanded_df = expanded_df[expanded_df[prior_data_col] ==
                                          prior_data_vals[0]]
    else:
        cur_expanded_df = expanded_df
    # cur_expanded_df = cur_expanded_df[
    #                cur_expanded_df.long_trace_id.isin(stats_df.long_trace_id)]
    left_val = stats_df.data_val_left.iloc[0]
    right_val = stats_df.data_val_right.iloc[0]
    expanded_left_df = cur_expanded_df[cur_expanded_df[data_col] == left_val]
    expanded_right_df = cur_expanded_df[cur_expanded_df[data_col] == right_val]
    return expanded_left_df, expanded_right_df

def plottingChangeinAUC(stats_df, expanded_df, pval, active_df=None,
                                                big_diff_neurons=None):
    df_li = []
    expanded_df = expanded_df.copy()
    display(expanded_df.PrevIsEasy.value_counts(dropna=False))
    # display(shortlong_quantiled_df)
    for (data_col, prior_data_col, prior_data_val), data_df in \
              stats_df.groupby(["data_col", "prior_data_col", "prior_data_val"],
                               dropna=False):
        data_df = data_df[data_df.DVstr.isnull()]
        expanded_left_df, expanded_right_df = _getExpandedDf(data_df,
                                                             expanded_df)

        print(data_col, prior_data_col, prior_data_val, "df len:", len(data_df))
        res_df = plotQuantilesBrainRegionAUC(data_df, expanded_left_df,
                                             expanded_right_df,
                                             plot_special=False,
                                             #data_col == "ChoiceLeft",
                                             pval=pval,
                                             active_df=active_df,
                                             big_diff_neurons=big_diff_neurons)
        res_df["data_col"] = data_col
        res_df["prior_data_col"] = prior_data_col
        res_df["prior_data_val"] = prior_data_val
        df_li.append(res_df)
    return pd.concat(df_li)


def _plotSpecialNeurons(trace_id, quantile_idx, left_rows_traces,
                        right_rows_traces):
    if trace_id != "GP4_23_s6_L50_D250_ALM_72":
        return
    left_mean = left_rows_traces.Raw.mean(axis=0)
    right_mean = right_rows_traces.Raw.mean(axis=0)
    left_sem = left_rows_traces.Raw.sem(axis=0)
    right_sem = right_rows_traces.Raw.sem(axis=0)
    mean_diff =    left_mean - right_mean
    mean_diff = mean_diff.abs()
    mean_diff_sum = mean_diff.sum()
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = np.arange(len(left_mean))
    ax.plot(xs, left_mean, label="Mean Left-Trials", color="orange")
    ax.plot(xs, right_mean, label="Mean Right-Trials", color="violet")
    # ax.fill_between(xs, left_mean - left_sem, left_mean + left_sem,
    #                                 color="orange", alpha=0.3)
    # ax.fill_between(xs, right_mean - right_sem, right_mean + right_sem,
    #                                 color="violet", alpha=0.3)
    ax.fill_between(xs, left_mean, right_mean, facecolor="none", hatch="//",
                    edgecolor="gray", linewidth=0.0)
    print("Trace id:", trace_id)
    ax.set_title(f"{trace_id} - " + r'$\sum_ |L - R|=$' +
                    f"{mean_diff_sum:.2f}")
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.set_xticks([1])
    ax.set_xticklabels(["Sampling Start"], rotation=45, ha="right")
    ax.axvline(1, ls="dashed", color="gray", alpha=0.3, zorder=-500)
    ax.set_ylabel("Neural Response (Z-Scored)")
    ax.legend()
    # plt.savefig(f"{fig_save_prefix}/RT_Stats/AUC_diff/{trace_id}_"
    #             f"q={quantile_idx}.pdf", dpi=300, bbox_inches='tight')
    plt.show()
