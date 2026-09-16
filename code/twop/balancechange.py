'''Tuning balance across an outcome streak (Figure 6C).

Backend for the "Balance Change - New Way" section of ``2pAnalysis.ipynb``.

For each direction-tuned, active neuron this measures how strongly it separates
its preferred from its non-preferred choice, **as a function of how long the
current run of correct or incorrect trials is**. The published panel is one
example neuron whose separation grows with the streak.

- ``PrevOutcomeCount`` is a signed streak length: negative after a run of
  errors, positive after a run of correct trials. It is **clipped** to
  ``+/- max_count`` (3 in the paper), so longer runs fold into the extremes
  rather than forming thin bins of their own, and **0 is dropped** — there is
  no streak to speak of.
- The activity per trial is averaged over one window: the bin that starts just
  after sampling onset, ``bins[1]:bins[2]`` of ``bins_count`` bins spanning the
  epoch. The first bin is the animal still moving to the poke, so it is skipped.
- ``MeanPref``/``MeanNnoPref`` average the trials **first**, giving one mean
  trace over that window, and then average it. **The SEM is therefore across
  the window's time points, not across trials** — it says how flat the mean
  response is within the window, not how much it varied trial to trial. Easy
  to misread as a trial-count error bar. (Note also the misspelled
  ``MeanNnoPref``, kept as it is used elsewhere in the notebook.)
- A neuron must be tuned (``pval <= pval_threshold``, not split by a prior or
  by difficulty) **and** in the active set. Left- and right-tuned neurons are
  processed separately, and an assertion guarantees no neuron appears in both.

**The example is chosen by a filter, not by hand**:
``plotBalanceChangeSingleNeuron`` draws only neurons whose ``MeanPref``
increases monotonically across the whole streak axis, and stops after two. That
is the "FIND_PATTERN" switch in the notebook, kept here because the published
panel is one of the neurons it selects.

Extracted from the notebook unchanged except that ``PVAL`` and
``fig_save_prefix`` are parameters, a hardcoded debug dump of one trace id is
gone, and the ``display`` calls are plain prints so the module works outside
IPython.
'''
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common.definitions import BrainRegion

#: Streak lengths beyond this fold into the extreme bins.
DEFAULT_MAX_COUNT = 3
#: Stop after this many example neurons per region.
MAX_EXAMPLES = 2


def _calcTraceBalance(trace_id, brain_region, left_tuned_df, right_tuned_Df,
                      start_idx, end_idx, is_left_tuned):
    '''One row per streak length for a single neuron.'''
    left_tuned_df = left_tuned_df[left_tuned_df.long_trace_id == trace_id]
    right_tuned_Df = right_tuned_Df[right_tuned_Df.long_trace_id == trace_id]

    preferred_df = left_tuned_df if is_left_tuned else right_tuned_Df
    non_preferred_df = right_tuned_Df if is_left_tuned else left_tuned_df
    res_dict = {"trace_id": [], "BrainRegion": [], "PrevOutcomeCount": [],
                "MeanPref": [], "MeanNnoPref": [], "MeanPrefSEM": [],
                "MeanNnoPrefSEM": [], "IsLeftTuned": []}
    outcome_li = sorted(list(preferred_df.PrevOutcomeCount.unique()))
    if 0 in outcome_li:
        outcome_li.remove(0)        # no streak
    for outcome in outcome_li:
        outcome_preferred_df = preferred_df[preferred_df.PrevOutcomeCount == outcome]
        outcome_non_preferred_df = non_preferred_df[
            non_preferred_df.PrevOutcomeCount == outcome]
        mean_pref = outcome_preferred_df.Raw.loc[:, start_idx:end_idx].mean()
        mean_non_pref = outcome_non_preferred_df.Raw.loc[:, start_idx:end_idx].mean()
        res_dict["trace_id"].append(trace_id)
        res_dict["BrainRegion"].append(brain_region)
        res_dict["PrevOutcomeCount"].append(outcome)
        res_dict["MeanPref"].append(mean_pref.mean())
        res_dict["MeanNnoPref"].append(mean_non_pref.mean())
        res_dict["MeanPrefSEM"].append(mean_pref.sem())
        res_dict["MeanNnoPrefSEM"].append(mean_non_pref.sem())
        res_dict["IsLeftTuned"].append(is_left_tuned)
    return pd.DataFrame(res_dict)


def samplingWindow(epochs_ranges, bins_count):
    '''The bin just after sampling onset, as ``(start_idx, end_idx)``.'''
    before, mid, after = epochs_ranges
    bins = [before[0]] + list(np.linspace(mid[0], after[0], bins_count - 1,
                                          endpoint=True)) + [after[1] + 1]
    print(len(bins), bins)
    bins = np.array(bins)
    second, third = int(round(bins[1])), int(round(bins[2]))
    print("second:", second, "third:", third)
    return second, third


def balanceChange(shortlong_df, normalized_expanded_df, active_neurons_df,
                  bins_count, pval, data_col="ChoiceLeft",
                  max_count=DEFAULT_MAX_COUNT):
    '''Per neuron and streak length: preferred vs non-preferred mean activity.'''
    sgf_prev = shortlong_df.query(f"data_col == '{data_col}' and pval <= {pval}"
                                  f" and prior_data_col.isnull() and DVstr.isnull()")
    sgf_left_tuned = sgf_prev[sgf_prev.IsROCLeftTuend == True]
    sgf_right_tuned = sgf_prev[sgf_prev.IsROCLeftTuend == False]
    sub_expanded_df = normalized_expanded_df[
        normalized_expanded_df.long_trace_id.isin(sgf_prev.long_trace_id)]
    del sgf_prev        # Don't reuse by mistake

    sgf_left_tuned = sgf_left_tuned[
        sgf_left_tuned.long_trace_id.isin(active_neurons_df.trace_id)]
    sgf_right_tuned = sgf_right_tuned[
        sgf_right_tuned.long_trace_id.isin(active_neurons_df.trace_id)]

    sub_expanded_df = sub_expanded_df.copy()
    sub_expanded_df.loc[sub_expanded_df.PrevOutcomeCount < -max_count,
                        "PrevOutcomeCount"] = -max_count
    sub_expanded_df.loc[sub_expanded_df.PrevOutcomeCount > max_count,
                        "PrevOutcomeCount"] = max_count
    print(sub_expanded_df.PrevOutcomeCount.value_counts())
    left_tuned_trials = sub_expanded_df[sub_expanded_df[data_col] == True]
    right_tuned_trials = sub_expanded_df[sub_expanded_df[data_col] == False]

    start_idx, end_idx = samplingWindow(normalized_expanded_df.epochs_ranges.iloc[0],
                                        bins_count)
    df_li = []
    traces_processed = set()
    for tuned_df, is_left_tuned in [(sgf_left_tuned, True), (sgf_right_tuned, False)]:
        for (trace_id, brain_region), _ in tuned_df.groupby(["long_trace_id",
                                                             "BrainRegion"]):
            assert trace_id not in traces_processed, \
                f"{trace_id} is tuned both ways"
            traces_processed.add(trace_id)
            df_li.append(_calcTraceBalance(trace_id, brain_region,
                                           left_tuned_trials, right_tuned_trials,
                                           start_idx, end_idx,
                                           is_left_tuned=is_left_tuned))
    return pd.concat(df_li)


def risesWithTheStreak(trace_df):
    '''True when the preferred-side mean increases at every step.'''
    return trace_df.MeanPref.diff().dropna().gt(0).all()


def plotBalanceChangeSingleNeuron(df, save_figs, fig_save_prefix=None,
                                  find_pattern=True):
    '''Figure 6C: the example neurons whose separation grows with the streak.'''
    print(df.PrevOutcomeCount.value_counts())
    for br, br_df in df.groupby("BrainRegion"):
        count = 0
        for trace_id, trace_df in br_df.groupby("trace_id"):
            trace_df = trace_df.sort_values("PrevOutcomeCount")
            if find_pattern and not risesWithTheStreak(trace_df):
                print(f"Trace {trace_id} is not increasing")
                continue

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.bar(trace_df.PrevOutcomeCount - 0.2, trace_df.MeanPref,
                   yerr=trace_df.MeanPrefSEM, width=0.4, color="g", label="Pref")
            ax.bar(trace_df.PrevOutcomeCount + 0.2, trace_df.MeanNnoPref,
                   yerr=trace_df.MeanNnoPrefSEM, width=0.4, color="orange",
                   label="Non-Pref")
            count += 1

            br_str = str(BrainRegion(trace_df.BrainRegion.iloc[0])).split("_")[0]
            ax.set_xlabel("PrevOutcomeCount")
            ax.set_ylabel("Mean Activity")
            ax.set_title(f"Balance Change - Example neuron: {br_str} - {trace_id}")
            ax.legend()
            ax.axhline(0, ls="--", c="gray")
            ax.spines[["right", "top", "left", "bottom"]].set_visible(False)
            if save_figs:
                assert fig_save_prefix is not None, "Pass fig_save_prefix to save"
                path = pathlib.Path(f"{fig_save_prefix}/RT_Stats/PrevChoiceCorrect/"
                                    f"BalanceChangeCurChoiceEx_{trace_id}.svg")
                path.parent.parent.mkdir(exist_ok=True)
                path.parent.mkdir(exist_ok=True)
                plt.savefig(path)
            plt.show()
            if count > MAX_EXAMPLES - 1:
                break
