
import os
from ...pipeline import pipeline
from ...common.definitions import BrainRegion
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats
from typing import Literal

def _setupAxes(ax, df, y_ticks_at_right, y_lim):
    ex_row = df.iloc[0]
    xs = [rng[0] for rng in ex_row.epochs_ranges]
    xs_labels = ex_row.epochs_names
    if " Sampling" in xs_labels[0]:
        xs = xs[1:]
        xs_labels = xs_labels[1:]
    ax.set_xticks(xs)
    ax.set_xticklabels(xs_labels)
    [ax.axvline(x, ls="--", c="gray") for x in xs]
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines['left'].set_position(('data', -0.5))

    if y_lim is not None:
        ax.set_ylim(y_lim)
    if y_ticks_at_right:
        ax.yaxis.tick_right()
    ax.set_xlim(0, df.trace_len.max())

def _makeAxes(y_ticks_at_right, df, y_lim, plot_title=False):
    fig, ax = plt.subplots()
    fig.set_size_inches((15, 10))
    _setupAxes(ax, df, y_ticks_at_right=y_ticks_at_right, y_lim=y_lim)
    if plot_title:
        q_idx = df.quantile_idx.iloc[0]
        br = str(BrainRegion(df.BrainRegion.iloc[0])).split('_')[0]
        ax.set_title(f"Activity sum {br} - Q={q_idx}")
    return ax


def _calcDurBinSum(df, append_at, set_name="neuronal", only_traces_ids=[],
                   avg_neurons=False, is_left=None, longest_trace_len=None):
    assert append_at in ["before", "after"]
    check_traces = len(only_traces_ids) != 0
    if not len(df):
        return None, set()
    ex_row = df.iloc[0]
    traces_sets = pipeline.getRowTracesSets(ex_row)
    ex_trace = next(iter(traces_sets[set_name].values()))
    sum_arr = np.zeros(ex_trace.shape)

    if longest_trace_len is None:
        longest_trace_len = (df.trace_end_idx - df.trace_start_idx).max() + 1
    nans = np.full(longest_trace_len, np.nan)
    res_dict = {
        "ShortName":[],
        "TrialNumber":[],
        "trace_id":[],
        "trace":[],
    }

    for _, row in df.iterrows():
        trace_dict = pipeline.getRowTracesSets(row)[set_name]
        s, e = row.trace_start_idx, row.trace_end_idx + 1
        len_append = longest_trace_len - (e-s)
        assert len_append >= 0
        for trace_id, trace in trace_dict.items():
            if check_traces:
                pref_direction = only_traces_ids.get(
                                            f"{trace_id}_{row.ShortName}", None)
                if pref_direction is not None:
                    if is_left is not None:
                        should_use = pref_direction == is_left
                    else:
                        should_use = True
                else:
                    should_use = False

            else:
                should_use = True
            if should_use:
                res_dict["ShortName"].append(row.ShortName)
                res_dict["TrialNumber"].append(row.TrialNumber)
                res_dict["trace_id"].append(trace_id)
                trace = trace[s:e]
                if len_append:
                    if append_at == "after":
                        trace = np.append(trace, nans[:len_append])
                    else:
                        trace = np.append(nans[:len_append], trace)
                    assert len(trace) == longest_trace_len
                res_dict["trace"].append(trace)

    res_df = pd.DataFrame(res_dict)
    if len(res_df) == 0:
        return None, set()
    res_df["long_trace_id"] = res_df.ShortName + "_" + res_df.trace_id.astype(
                                                                            str)
    res_df = res_df.groupby("long_trace_id").trace.agg([np.nanmean,
                                                        ]).reset_index()
    used_traces_ids = set(res_df.long_trace_id.unique())
    assert len(used_traces_ids) == len(res_df)
    num_traces = len(res_df)
    sum_arr = np.sum(res_df.nanmean.values, axis=0)
    sum_arr /= num_traces
    return sum_arr, used_traces_ids

def _nonNanApply(arr, applyFn):
    all_nan_axis = np.all(np.isnan(arr), axis=0)
    mean = np.full(arr.shape[1], np.nan)
    mean[~all_nan_axis] = applyFn(arr[:,~all_nan_axis], axis=0)
    return mean

def _calcSidePref(grp_df, append_at, only_traces_ids, longest_trace_len=None):
    sums_li = []
    used_traces_ids = set()
    if longest_trace_len is None:
        longest_trace_len = (
                        grp_df.trace_end_idx - grp_df.trace_start_idx).max() + 1
    for is_left in [True, False]:
        choice_df = grp_df[grp_df.ChoiceLeft == is_left]
        _sum, _used_traces_ids = _calcDurBinSum(choice_df, append_at,
                                                only_traces_ids=only_traces_ids,
                                                is_left=is_left,
                                           longest_trace_len=longest_trace_len)
        if _sum is None and len(_used_traces_ids) == 0:
            continue
        sums_li.append(_sum)
        used_traces_ids.update(_used_traces_ids)
    if len(sums_li) == 0:
        return None, 0
    sums_arr = np.array(sums_li)
    _sum_mean = _nonNanApply(sums_arr, np.nanmean)
    num_traces = len(used_traces_ids)
    return _sum_mean, num_traces

def groupCurves(df, groupby_cols, append_at, only_traces_ids=[],
                split_by_sess=True):
    '''Per group of ``groupby_cols``: the mean activity trace and its SEM.

    Each session's trace is the mean over its neurons of each neuron's
    trial-mean, averaged over left- and right-choice trials
    (:func:`_calcSidePref`); the group's trace is the mean of the session
    traces, the SEM taken across sessions. Traces are NaN-padded to the
    group's longest trial on the ``append_at`` side. Returns one dict per group,
    in group order: ``group``, ``mean``, ``sem``, ``num_traces`` (neurons) and
    ``sessions`` (the set of sessions used).

    With ``split_by_sess=False`` all sessions are pooled and ``sem`` is 0.
    '''
    curves = []
    for grp_name, grp_df in df.groupby(groupby_cols):
        sess_names = set()
        if split_by_sess:
            longest_trace_len = (
                        grp_df.trace_end_idx - grp_df.trace_start_idx).max() + 1
            sess_sums = []
            num_traces = 0
            for sess, sess_df in grp_df.groupby("ShortName"):
                sess_sum, sess_num_traces = _calcSidePref(sess_df, append_at,
                                                only_traces_ids=only_traces_ids,
                                            longest_trace_len=longest_trace_len)
                if sess_sum is None and sess_num_traces == 0:
                    print("Skipping:", sess)
                    continue
                sess_sums.append(sess_sum)
                num_traces += sess_num_traces
                sess_names.add(sess)
            sess_sums = np.array(sess_sums)
            all_nan_axis = np.all(np.isnan(sess_sums), axis=0)
            _sum = np.full(sess_sums.shape[1], np.nan)
            _sem = np.full(sess_sums.shape[1], np.nan)
            _sum[~all_nan_axis] = np.nanmean(sess_sums[:,~all_nan_axis], axis=0)
            _sem[~all_nan_axis] = stats.sem(sess_sums[:,~all_nan_axis], axis=0)
        else:
            _sum, num_traces = _calcSidePref(grp_df, append_at,
                                            only_traces_ids=only_traces_ids)
            _sem = 0
        curves.append(dict(group=grp_name, mean=_sum, sem=_sem,
                           num_traces=num_traces, sessions=sess_names))
    return curves


def pruneSingleSession(mean, sem):
    '''``mean`` with NaN wherever ``sem`` is NaN (a point only one session has).'''
    mean = np.array(mean, dtype=float)
    mean[np.isnan(sem)] = np.nan
    return mean


def _plotDf(df, groupby_cols, append_at, only_traces_ids, clr_map_name, ax,
            split_by_sess=True):
    curves = groupCurves(df, groupby_cols, append_at, only_traces_ids,
                         split_by_sess=split_by_sess)
    sess_names = set().union(*[c["sessions"] for c in curves])

    cm = plt.get_cmap(clr_map_name)
    NPOINTS = len(curves)
    clrs = [cm(1.*i/NPOINTS) for i in range(1, NPOINTS+1)]
    count = 0
    for curve in curves:
        _sum, _sem = curve["mean"], curve["sem"]
        grp_name, num_traces = curve["group"], curve["num_traces"]
        if int(num_traces) == num_traces:
            num_traces = int(num_traces)
        sess_str = f"{len(sess_names)} sessions - " if split_by_sess else ""
        keyword = "Quantile" if groupby_cols == "quantile_idx" else "Group"
        label = f"{keyword}={grp_name} ({sess_str}{num_traces} neuron)"
        c = clrs[count]
        if any(np.isnan(_sem)):
            # Prune the parts with a single point, i.e, SEM == 0
            print("Pruning")
            _sum = pruneSingleSession(_sum, _sem)
        ax.plot(_sum, c=c, label=label)
        ax.fill_between(np.arange(len(_sum)), _sum - _sem, _sum + _sem, color=c,
                        alpha=0.2)
        count += 1
    ax.legend()


def plotActivitySum(df, y_lim, append_at : Literal["before", "after"],
                    groupby_cols, only_traces_ids=[], save_figs=False,
                    fig_save_prefix=None, save_label=None):
    assert append_at in ["before", "after"]
    df = df.copy()
    y_ticks_at_right = append_at == "before"

    for br, br_df in df.groupby("BrainRegion"):
        ax = _makeAxes(y_ticks_at_right=y_ticks_at_right, df=df,
                        plot_title=False, y_lim=y_lim)
        br = BrainRegion(br)
        for layer, layer_df in br_df.groupby("Layer"):
            clr_map_name = "Blues" if "M2" in str(br) else "Reds"
            _plotDf(df=layer_df, groupby_cols=groupby_cols, append_at=append_at,
                    only_traces_ids=only_traces_ids, clr_map_name=clr_map_name,
                    ax=ax)
        br_str = str(br).split("_")[0]
        if save_figs:
            assert fig_save_prefix is not None
            assert save_label is not None
            fp = f"{fig_save_prefix}/activity_sum/{save_label}_{br_str}.svg"
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            print("Saving to:", fp)
            plt.savefig(fp)
        plt.show()
