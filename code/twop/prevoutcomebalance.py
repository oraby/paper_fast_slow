'''Neurons whose response grows with the run of previous outcomes.

Backend of the "Reward Rate" section of ``2p_3_choice``. No panel of it is in
the paper; it reports, per region, the share of neurons whose peak response
rises steadily with ``PrevOutcomeCount`` (the signed length of the current run
of correct (+) or incorrect (-) trials, clipped to +/- ``max_outcome``).

- ``makeBalanceDf`` -- per neuron and outcome-run length: the mean / median /
  SEM of the trial peak, and the mean trace. Every neuron is entered once over
  all its trials ("All" values); a neuron significantly tuned to a variable
  (``pval``) is entered again under that variable, split into its preferred
  and non-preferred trials ("Pref" / "NonPref" values).
- ``incrementingStats`` -- which neurons rise monotonically, as a per-session
  table of counts and percentages; it can also draw (and save) each such
  neuron (``plotTraceBalance``), to
  ``{fig_save_prefix}/RewardRate/IncNeuron/BalanceChangeCurChoiceEx_*.svg``.
- ``printStats`` -- the region / variable summary of that table.
'''
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common.definitions import BrainRegion

#: Outcome runs longer than this are counted as this long.
MAX_OUTCOME = 3


def processTrace(trace_df, data_col=None, data_col_pref=None,
                 shuffle=False, track_raw=True):
    '''One neuron's peak statistics per ``PrevOutcomeCount``.'''
    if data_col is not None:
        assert data_col_pref is not None
    trace_res_dict = {"PrevOutcomeCount": []}

    def addOrUpdateField(prefix, key, val):
        trace_res_dict.setdefault(f"{prefix}{key}", []).append(val)

    def addOutcomeGroup(outcome_group_df, prefix, populate_nans=False):
        if not populate_nans:
            outcome_max = outcome_group_df.Raw.max(axis=1)
            max_mean = outcome_max.mean()
            max_median = outcome_max.median()
            max_sem = outcome_max.sem()
            if track_raw:
                raw_mean = outcome_group_df.Raw.mean(axis=0)
                raw_sem = outcome_group_df.Raw.sem(axis=0)
            num_trials = len(outcome_group_df)
            assert len(outcome_max) == num_trials
        else:
            max_mean = max_median = max_sem = np.nan
            raw_mean = raw_sem = np.nan
            num_trials = np.nan
        addOrUpdateField(prefix, "MaxMean", max_mean)
        addOrUpdateField(prefix, "MaxMedian", max_median)
        addOrUpdateField(prefix, "MaxSEM", max_sem)
        if track_raw:
            addOrUpdateField(prefix, "RawMean", raw_mean)
            addOrUpdateField(prefix, "RawSEM", raw_sem)
        addOrUpdateField(prefix, "NumTrials", num_trials)

    if shuffle:
        trace_df = trace_df.copy()
        trace_df["PrevOutcomeCount"] = trace_df.PrevOutcomeCount.sample(frac=1).values

    for outcome, outcome_df in trace_df.groupby("PrevOutcomeCount"):
        trace_res_dict["PrevOutcomeCount"].append(outcome)
        if data_col is not None:
            pref_df = outcome_df[outcome_df[data_col] == data_col_pref]
            non_pref_df = outcome_df[outcome_df[data_col] != data_col_pref]
            addOutcomeGroup(outcome_df, "All", populate_nans=True)
            addOutcomeGroup(pref_df, "Pref")
            addOutcomeGroup(non_pref_df, "NonPref")
        else:
            addOutcomeGroup(outcome_df, "All")
            addOutcomeGroup(outcome_df, "Pref", populate_nans=True)
            addOutcomeGroup(outcome_df, "NonPref", populate_nans=True)
    return trace_res_dict


def makeBalanceDf(short_long_df, expanded_df, pval, max_outcome=MAX_OUTCOME,
                  shuffle=False, only_all=False):
    '''Row layout: a trace appears either under a data col, with pref/non-pref
    values and no "all" values, or under "All", with only "all" values.'''
    track_raw = not shuffle
    short_long_df = short_long_df[short_long_df.prior_data_col.isnull()]
    short_long_df = short_long_df[short_long_df.DVstr.isnull()]
    short_long_df = short_long_df[short_long_df.pval <= pval]

    expanded_df = expanded_df.copy()
    expanded_df.loc[expanded_df.PrevOutcomeCount < -max_outcome, "PrevOutcomeCount"] = -max_outcome
    expanded_df.loc[expanded_df.PrevOutcomeCount > max_outcome, "PrevOutcomeCount"] = max_outcome

    res_dict = {"BrainRegion": [], "ShortName": [], "long_trace_id": [],
                "data_col": [], "PrevOutcomeCount": [], "is_left_tuned": [],
                "epochs_ranges": [], "epochs_names": []}

    def invokeProcessTrace(br, sess, trace_id, expanded_trace_df, data_col,
                           is_left_tuned):
        trace_dict = processTrace(expanded_trace_df, data_col,
                                  data_col_pref=is_left_tuned,
                                  shuffle=shuffle, track_raw=track_raw)
        for key, val in trace_dict.items():
            res_dict.setdefault(key, [])
            res_dict[key] += val
        len_res = len(trace_dict["PrevOutcomeCount"])
        res_dict["BrainRegion"] += [br] * len_res
        res_dict["ShortName"] += [sess] * len_res
        res_dict["long_trace_id"] += [trace_id] * len_res
        res_dict["data_col"] += [data_col] * len_res
        res_dict["is_left_tuned"] += [is_left_tuned] * len_res
        if len(expanded_trace_df):
            epochs_ranges = expanded_trace_df.epochs_ranges.iloc[0]
            epochs_names = expanded_trace_df.epochs_names.iloc[0]
        else:
            epochs_ranges, epochs_names = np.nan, np.nan
        res_dict["epochs_ranges"] += [epochs_ranges] * len_res
        res_dict["epochs_names"] += [epochs_names] * len_res

    if not shuffle:
        print("All traces")
    for br, expanded_br_df in expanded_df.groupby("BrainRegion"):
        for sess, sess_df in expanded_br_df.groupby("ShortName"):
            for trace_id, trace_df in sess_df.groupby("long_trace_id"):
                invokeProcessTrace(br, sess=sess, trace_id=trace_id,
                                   expanded_trace_df=trace_df,
                                   data_col=None, is_left_tuned=None)

    if not only_all:
        for data_col, sgf_data_col_df in short_long_df.groupby("data_col"):
            if not shuffle:
                print("Data col:", data_col)
            for br, br_df in sgf_data_col_df.groupby("BrainRegion"):
                if not shuffle:
                    print("   BrainRegion:", br)
                for sess, sess_df in br_df.groupby("ShortName"):
                    for _idx, trace_row in sess_df.iterrows():
                        trace_id = trace_row.long_trace_id
                        expanded_trace_df = expanded_df[expanded_df.long_trace_id == trace_id]
                        invokeProcessTrace(br, sess=sess, trace_id=trace_id,
                                           expanded_trace_df=expanded_trace_df,
                                           data_col=data_col,
                                           is_left_tuned=trace_row.IsROCLeftTuend)
    return pd.DataFrame(res_dict)


_PREF_COLORS = {"ChoiceLeft": ("green", "orange"),
                "PrevChoiceLeft": ("yellow", "purple"),
                "PrevChoiceCorrect": ("magenta", "cyan")}


def plotTraceBalance(br_str, trace_id, trace_df, data_col, plot, save_figs,
                     fig_save_prefix=None, max_outcome=MAX_OUTCOME):
    '''One neuron: peak per outcome run (bars) and the mean trace of each run.'''
    plot_all_cols = data_col == "All"
    color_all = "gray"
    color_pref, color_non_pref = _PREF_COLORS.get(data_col, ("gray", "darkgray"))

    fig, all_axs = plt.subplots(1, 1 + max_outcome * 2, figsize=(25, 3))
    ax_bar, *ax_raws = all_axs
    prev_outcome_counts = np.concatenate([np.arange(-max_outcome, 0),
                                          np.arange(1, max_outcome + 1)])
    assert len(prev_outcome_counts) == len(ax_raws)

    if plot_all_cols:
        ax_bar.bar(trace_df.PrevOutcomeCount, trace_df.AllMaxMean,
                   yerr=trace_df.AllMaxSEM, width=0.8, color=color_all, label="All")
    else:
        ax_bar.bar(trace_df.PrevOutcomeCount - 0.2, trace_df.PrefMaxMean,
                   yerr=trace_df.PrefMaxSEM, width=0.4, color=color_pref, label="Pref")
        ax_bar.bar(trace_df.PrevOutcomeCount + 0.2, trace_df.NonPrefMaxMean,
                   yerr=trace_df.NonPrefMaxSEM, width=0.4, color=color_non_pref,
                   label="Non-Pref")
    ax_bar.set_xlabel("PrevOutcomeCount")
    ax_bar.set_ylabel("Max Activity")
    ax_bar.legend()
    ax_bar.axhline(0, ls="--", c="gray")
    ax_bar.set_xticks(prev_outcome_counts)

    len_raw_col = "AllRawMean" if plot_all_cols else "PrefRawMean"
    x = np.arange(len(trace_df[len_raw_col].iloc[0]))

    def plotRaws(prev_outcome_count, ax_raw, used_df, color, prefix):
        used_df = used_df[used_df.PrevOutcomeCount == prev_outcome_count]
        assert len(used_df) <= 1, (f"Expected 1 row for PrevOutcomeCount="
                                   f"{prev_outcome_count}, got {len(used_df)}")
        if len(used_df) == 0:
            return None
        row = used_df.iloc[0]
        num_trials = row[f"{prefix}NumTrials"]
        label = f"{prefix} - PrevCount={row.PrevOutcomeCount} - #Trials={num_trials}"
        ax_raw.plot(x, row[f"{prefix}RawMean"], color=color, label=label)
        y_upper = row[f"{prefix}RawMean"] + row[f"{prefix}RawSEM"]
        y_lower = row[f"{prefix}RawMean"] - row[f"{prefix}RawSEM"]
        ax_raw.fill_between(x, y_lower, y_upper, alpha=0.1, color=color)
        return y_lower.min(), y_upper.max()

    epochs_ranges = trace_df.epochs_ranges[trace_df.epochs_ranges.notnull()]
    epochs_ranges = set(tuple(_rng) for _rng in epochs_ranges)
    assert len(epochs_ranges) == 1, "Expected all epochs_ranges to be the same"
    epochs_ranges = epochs_ranges.pop()
    epochs_names = trace_df.epochs_names[trace_df.epochs_names.notnull()].iloc[0]
    x_ticks, x_labels = [], []
    for epoch_name, (_start, end) in zip(epochs_names[1:], epochs_ranges[:-1]):
        x_ticks.append(end)
        x_labels.append(epoch_name)

    y_min, y_max = np.inf, -np.inf
    for prev_outcome_count, ax_raw in zip(prev_outcome_counts, ax_raws):
        if plot_all_cols:
            all_df = trace_df[trace_df.AllMaxMean.notnull()]
            lims = [plotRaws(prev_outcome_count, ax_raw, all_df, color_all, "All")]
        else:
            pref_df = trace_df[trace_df.PrefMaxMean.notnull()]
            non_pref_df = trace_df[trace_df.NonPrefMaxMean.notnull()]
            lims = [plotRaws(prev_outcome_count, ax_raw, pref_df, color_pref, "Pref"),
                    plotRaws(prev_outcome_count, ax_raw, non_pref_df,
                             color_non_pref, "NonPref")]
        for lim in lims:
            if lim is not None:
                y_min = min(y_min, lim[0])
                y_max = max(y_max, lim[1])
        for x_tick in x_ticks:
            ax_raw.axvline(x_tick, ls="--", color="gray", alpha=0.7)
        ax_raw.set_xticks(x_ticks)
        ax_raw.set_xticklabels(x_labels, rotation=45, ha="right")
        ax_raw.set_xlabel("Normalized Time")
        ax_raw.set_ylabel("Mean Activity")
        title_str = "Correct" if prev_outcome_count > 0 else "Incorrect"
        ax_raw.set_title(f"{abs(prev_outcome_count)} {title_str} Trials")
        ax_raw.legend(fontsize="xx-small")
    for ax_raw in ax_raws:
        ax_raw.set_ylim(y_min * 1.1, y_max * 1.1)
    fig.suptitle(f"Balance Change - {br_str} - {trace_id} - {data_col}", y=1.05)
    for ax in all_axs:
        ax.spines[["right", "top", "left", "bottom"]].set_visible(False)
    if save_figs:
        assert fig_save_prefix is not None, "Pass fig_save_prefix to save"
        save_path = pathlib.Path(f"{fig_save_prefix}/RewardRate/IncNeuron/"
                                 f"BalanceChangeCurChoiceEx_{trace_id}_{data_col}.svg")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    if plot:
        plt.show()
    else:
        plt.close()


def _brStr(br):
    return str(BrainRegion(br)).split("_")[0]


def incrementingStats(balance_df, sess_totals, plot, save_figs,
                      fig_save_prefix=None, max_outcome=MAX_OUTCOME):
    '''Per-session counts of neurons whose peak rises with every outcome run.

    ``sess_totals`` maps each session to its number of neurons (the
    denominator). Every rising neuron is drawn when ``plot`` or ``save_figs``.
    '''
    balance_df = balance_df[balance_df.PrevOutcomeCount != 0]
    inc = {"BrainRegion": [], "ShortName": [], "trace_id": [],
           "sess_total_neurons": [], "data_col": []}

    def add(br_str, sess, trace_id, data_col, total):
        inc["BrainRegion"].append(br_str)
        inc["ShortName"].append(sess)
        inc["trace_id"].append(trace_id)
        inc["data_col"].append(data_col)
        inc["sess_total_neurons"].append(total)

    for data_col, data_df in balance_df.groupby("data_col", dropna=False):
        if not isinstance(data_col, str) and np.isnan(data_col):
            data_col = "All"
        for sess, sess_total_neurons in sess_totals.items():
            sess_df = data_df[data_df.ShortName == sess]
            if len(sess_df) == 0:
                # Still count the session, with no rising neuron
                sess_all_df = balance_df[balance_df.ShortName == sess]
                add(_brStr(sess_all_df.BrainRegion.iloc[0]), sess, np.nan,
                    data_col, sess_total_neurons)
                continue
            br_str = _brStr(sess_df.BrainRegion.iloc[0])
            for trace_id, trace_df in sess_df.groupby("long_trace_id"):
                trace_df = trace_df.sort_values("PrevOutcomeCount")
                pref_max = (trace_df.AllMaxMean if data_col == "All"
                            else trace_df.PrefMaxMean)
                # Only increases are counted
                if not pref_max.diff().dropna().gt(0).all():
                    continue
                add(br_str, sess, trace_id, data_col, sess_total_neurons)
                if plot or save_figs:
                    plotTraceBalance(br_str, trace_id, trace_df, data_col,
                                     plot, save_figs,
                                     fig_save_prefix=fig_save_prefix,
                                     max_outcome=max_outcome)
    return _getStats(pd.DataFrame(inc), sess_totals)


def _getStats(inc_df, sess_totals):
    stats_dict = {"data_col": [], "BrainRegion": [], "ShortName": [],
                  "total": [], "inc": [], "prcnt": []}

    def add(br, sess, data_col, total, inc):
        stats_dict["BrainRegion"].append(br)
        stats_dict["ShortName"].append(sess)
        stats_dict["data_col"].append(data_col)
        stats_dict["total"].append(total)
        stats_dict["inc"].append(inc)
        stats_dict["prcnt"].append((inc / total) * 100)

    sess_to_br = {sess: sess_df.BrainRegion.iloc[0]
                  for sess, sess_df in inc_df.groupby("ShortName")}

    def sessInc(df, sess):
        sess_df = df[df.ShortName == sess]
        return sess_df.trace_id.nunique() if len(sess_df) else 0

    for sess, br in sess_to_br.items():
        sess_inc = sessInc(inc_df, sess)
        add("Global", sess, "Global", sess_totals[sess], sess_inc)
        add(br, sess, "Global", sess_totals[sess], sess_inc)
    for data_col, data_col_df in inc_df.groupby("data_col"):
        for sess, br in sess_to_br.items():
            sess_inc = sessInc(data_col_df, sess)
            add("Global", sess, data_col, sess_totals[sess], sess_inc)
            add(br, sess, data_col, sess_totals[sess], sess_inc)
    return pd.DataFrame(stats_dict)


def printStats(stats_df, save_figs=False, fig_save_prefix=None):
    '''Per region and variable: rising / total neurons, and mean +/- SEM %.

    The "Global" row (every variable, plus the untuned neurons) is also drawn
    as a pie, saved to ``{fig_save_prefix}/RewardRate_Pie_*.svg`` when
    ``save_figs``.
    '''
    for br, br_df in stats_df[stats_df.BrainRegion != "Global"].groupby("BrainRegion"):
        print(br)
        for data_col, data_col_df in br_df.groupby("data_col"):
            header_str = data_col_df.data_col.iloc[0]
            inc_pooled = data_col_df.inc.sum()
            total_pooled = data_col_df.total.sum()
            prcnt_mean = data_col_df.prcnt.mean()
            prcnt_sem = data_col_df.prcnt.sem()
            print(f"    {header_str}: {inc_pooled}/{total_pooled} "
                  f"({prcnt_mean:.2f}%) ± {prcnt_sem:.2f}")
            if data_col != "Global":
                continue
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.pie([prcnt_mean, 100 - prcnt_mean],
                   labels=[f"{prcnt_mean:.2f}% ± {prcnt_sem:.2f}", None],
                   colors=["#64C7C2", "gray"], startangle=90,
                   counterclock=False, explode=[0.1, 0], labeldistance=.8,
                   textprops={"ha": "left"})
            ax.axis("equal")
            title_col = "Across all data cols + non-sgf"
            ax.set_title(f"{br} - Reward Rate Neurons - {title_col}")
            ax.spines[["right", "top", "left", "bottom"]].set_visible(False)
            if save_figs:
                assert fig_save_prefix is not None, "Pass fig_save_prefix to save"
                plt.savefig(f"{fig_save_prefix}/RewardRate_Pie_{br} "
                            f"data_col={title_col} .svg")
            plt.show()
