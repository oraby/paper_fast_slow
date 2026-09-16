'''Previous-outcome modulation of tuned neurons (Figure S14A).

Backend for the "Sgf. between Prev. Correct vs Incorrect" section of
``2pAnalysis.ipynb``.

For every neuron already found tuned to a variable, this asks whether the
*size* of that tuning depends on how the previous trial ended. Per neuron:

1. reduce each trial to one number — by default the peak of the neuron's trace
   over that trial's window (``np.nanmax``);
2. split the trials by the previous outcome, and within each half take
   **preferred minus anti-preferred** mean, where preferred is the side the ROC
   said the neuron favours (``IsROCLeftTuend``);
3. the modulation is the difference of those two differences.

``PrevCorrect`` and ``PrevIncorrect`` in the result are the two halves;
``PrevDiff`` is the modulation; ``PermutePVal`` is its permutation p-value.

**The permutation is not seeded** — see ``rng`` below — and it shuffles the two
halves *separately*: preferred values are reshuffled between prev-correct and
prev-incorrect, and anti-preferred values likewise, so the null keeps each
neuron's preferred/anti-preferred split intact and only breaks its link to the
previous outcome. The p-value is the usual ``(hits + 1) / (N + 1)``.

**Two variables are skipped**, along with anything split by a prior:
``IsEasy``, ``PrevIsEasy`` and ``PrevChoiceCorrect`` — the last because
splitting previous outcome by previous outcome is vacuous.

The pies show the **mean across sessions** of each session's percentage of
modulated neurons, with the SEM across sessions; the caption's neuron counts,
though, are pooled over all neurons and not per session.

Extracted from the notebook unchanged except that ``PVAL`` and
``fig_save_prefix`` are parameters, and the shuffling takes an optional ``rng``
(defaulting to ``numpy.random``, i.e. the unseeded global stream the notebook
used) so a caller can make it reproducible.
'''
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..common.definitions import BrainRegion

#: Variables this analysis does not ask about.
SKIP_DATA_COLS = ["IsEasy", "PrevIsEasy", "PrevChoiceCorrect"]
#: Legend names for the variables it does ask about.
DATA_COL_NAMES = {"ChoiceLeft": "Cur. Direction",
                  "PrevChoiceLeft": "Prev. Direction",
                  "PrevChoiceCorrect": "Prev. Outcome"}
PIE_COLORS = {"Cur. Direction": "g", "Prev. Direction": "gold",
              "PrevChoiceCorrect": "turquoise"}
DEFAULT_PIE_COLOR = "purple"


def _diff(pref_reducs, anti_pref_reducs):
    return pref_reducs.mean() - anti_pref_reducs.mean()


def _prevDiff(prev_corr_pref_anti_pref, prev_incorr_pref_anti_pref):
    return _diff(*prev_corr_pref_anti_pref) - _diff(*prev_incorr_pref_anti_pref)


def _restrictToNeuron(df, trace_id, data_col, reducFn=np.nanmax,
                      prior_data_col=None):
    '''One row per trial: the reduced activity of this neuron in that trial.'''
    common_cols = [data_col, "PrevChoiceCorrect", "trace_reduc"]
    extra_cols = ["traces_sets", "trace_start_idx", "trace_end_idx"]
    if prior_data_col is not None:
        df = df[common_cols + extra_cols + [prior_data_col]]
    else:
        df = df[common_cols + extra_cols]
    assert "trace_reduc" in df.columns, "Please pre-allocate col for efficiency"

    df_rows = []
    for _, row in df.iterrows():
        dst_trace = row.traces_sets["neuronal"][trace_id]
        dst_trace = dst_trace[row.trace_start_idx:row.trace_end_idx + 1]
        row["trace_reduc"] = reducFn(dst_trace)
        df_rows.append(row)
    return pd.DataFrame(df_rows)[common_cols]


def _reducPrefAntiPref(df, data_col, data_val_left, data_val_right, is_left_tuned):
    '''The neuron's values on its preferred side, then the other side.'''
    cond1 = data_val_left if is_left_tuned else data_val_right
    cond2 = data_val_right if is_left_tuned else data_val_left
    return (df[df[data_col] == cond1].trace_reduc.values,
            df[df[data_col] == cond2].trace_reduc.values)


def permutePVal(prev_corr_pref_anti_pref, prev_incorr_pref_anti_pref, diff_res,
                shuffle_iterations, rng=np.random):
    '''Two-sided p-value for the modulation, by reshuffling the outcome labels.

    Preferred and anti-preferred values are shuffled in separate pools, so the
    null keeps the tuning and breaks only its dependence on the last outcome.
    '''
    prev_corr_pref, prev_corr_anti_pref = prev_corr_pref_anti_pref
    prev_incorr_pref, prev_incorr_anti_pref = prev_incorr_pref_anti_pref
    agg_pref = np.concatenate([prev_corr_pref, prev_incorr_pref])
    shuffle_pref_mask = np.zeros_like(agg_pref)
    shuffle_pref_mask[:len(prev_corr_pref)] = 1
    agg_anti_pref = np.concatenate([prev_corr_anti_pref, prev_incorr_anti_pref])
    shuffle_anti_pref_mask = np.zeros_like(agg_anti_pref)
    shuffle_anti_pref_mask[:len(prev_corr_anti_pref)] = 1

    shuffle_diff_li = []
    for _ in range(shuffle_iterations):
        rng.shuffle(shuffle_pref_mask)
        rng.shuffle(shuffle_anti_pref_mask)
        pref_mask = shuffle_pref_mask.astype(bool)
        anti_mask = shuffle_anti_pref_mask.astype(bool)
        shuffle_diff_li.append(_prevDiff(
            (agg_pref[pref_mask], agg_pref[~pref_mask]),
            (agg_anti_pref[anti_mask], agg_anti_pref[~anti_mask])))

    shuffle_diff_li = np.array(shuffle_diff_li)
    N = len(shuffle_diff_li)
    return (np.sum(np.abs(shuffle_diff_li) >= abs(diff_res)) + 1) / (N + 1)


def _processTrace(row, sess_trials_df, diffs_dict, shuffle_iterations, data_col,
                  data_val_left, data_val_right, prior_data_col, sess_name,
                  br_str, prior_data_col_val=None, rng=np.random):
    restricted_df = _restrictToNeuron(sess_trials_df, row.trace_id,
                                      data_col=data_col,
                                      prior_data_col=prior_data_col)
    if prior_data_col_val is not None:
        restricted_df = restricted_df[restricted_df[prior_data_col] == prior_data_col_val]
    else:
        assert prior_data_col is None
    diff_kwargs = dict(data_col=data_col, data_val_left=data_val_left,
                       data_val_right=data_val_right,
                       is_left_tuned=row.IsROCLeftTuend)
    prev_corr = _reducPrefAntiPref(restricted_df[restricted_df.PrevChoiceCorrect == 1],
                                   **diff_kwargs)
    prev_incorr = _reducPrefAntiPref(restricted_df[restricted_df.PrevChoiceCorrect == 0],
                                     **diff_kwargs)
    diff_res = _prevDiff(prev_corr, prev_incorr)
    diffs_dict["BrainRegion"].append(br_str)
    diffs_dict["ShortName"].append(sess_name)
    diffs_dict["trace_id"].append(row.trace_id)
    diffs_dict["PrevCorrect"].append(_diff(*prev_corr))
    diffs_dict["PrevIncorrect"].append(_diff(*prev_incorr))
    diffs_dict["PrevDiff"].append(diff_res)
    if shuffle_iterations:
        diffs_dict["PermutePVal"].append(
            permutePVal(prev_corr, prev_incorr, diff_res, shuffle_iterations, rng))
    diffs_dict["DataCol"].append(DATA_COL_NAMES.get(data_col, data_col))
    diffs_dict["PriorDataCol"].append("" if prior_data_col is None
                                      else str(prior_data_col))


def loopSgfPrevOutcome(stats_df, trials_df, shuffle_iterations, pval,
                       rng=np.random):
    '''One row per tuned neuron per variable, with its modulation and p-value.'''
    stats_df = stats_df[stats_df.DVstr.isnull()]
    stats_df = stats_df[stats_df.pval <= pval]
    trials_df = trials_df.copy()
    trials_df["trace_reduc"] = np.nan
    diffs = {"BrainRegion": [], "ShortName": [], "trace_id": [],
             "PrevCorrect": [], "PrevIncorrect": [], "PrevDiff": [],
             "DataCol": [], "PriorDataCol": []}
    if shuffle_iterations > 0:
        diffs["PermutePVal"] = []

    for (data_col, prior_data_col), grp_df in stats_df.groupby(
            ["data_col", "prior_data_col"], dropna=False):
        if not isinstance(prior_data_col, str):
            assert np.isnan(prior_data_col)
            prior_data_col = None
        if prior_data_col is not None or data_col in SKIP_DATA_COLS:
            continue

        print("For:", data_col, prior_data_col)
        data_val_left = grp_df.data_val_left.unique()[0]
        data_val_right = grp_df.data_val_right.unique()[0]
        for br, br_df in grp_df.groupby("BrainRegion"):
            br_str = str(BrainRegion(br)).split("_")[0]
            br_str = "MFC" if br_str == "M2" else "LFC"
            for sess, sess_df in tqdm(br_df.groupby("ShortName"),
                                      total=len(br_df.ShortName.unique()),
                                      desc=br_str):
                sess_trials_df = trials_df[trials_df.ShortName == sess]
                for _, row in tqdm(sess_df.iterrows(), total=len(sess_df),
                                   leave=False, desc=f"{br_str} - {sess}"):
                    _processTrace(row=row, sess_trials_df=sess_trials_df,
                                  diffs_dict=diffs,
                                  shuffle_iterations=shuffle_iterations,
                                  data_col=data_col,
                                  data_val_left=data_val_left,
                                  data_val_right=data_val_right,
                                  prior_data_col=prior_data_col,
                                  sess_name=sess, br_str=br_str,
                                  prior_data_col_val=None, rng=rng)
    return pd.DataFrame(diffs)


def modulatedPercentage(df, pval):
    '''Percent of this frame's neurons whose modulation reached significance.'''
    return 100 * (df.PermutePVal < pval).sum() / len(df)


def _plotPrevModulatedPie(df, pval, dscrp, data_col, save_prefix=None,
                          save_figs=False):
    color = PIE_COLORS.get(data_col, DEFAULT_PIE_COLOR)
    # Mean across sessions, not over pooled neurons.
    sgf_neurons_prcnt = df.groupby("ShortName").apply(modulatedPercentage,
                                                      pval=pval)
    sgf_mean, sgf_sem = sgf_neurons_prcnt.mean(), sgf_neurons_prcnt.sem()
    num_neurons = len(df)
    num_sgf_neurons = (df.PermutePVal < pval).sum()
    print(f"\t{dscrp}: {sgf_mean:.2f}% ±{sgf_sem:.2f}%")
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.pie([100 - sgf_mean, sgf_mean],
           labels=["", f"{sgf_mean:.2f}% ±{sgf_sem:.2f}%\n"
                       f"{num_sgf_neurons:,} / {num_neurons:,} neurons"],
           startangle=90, explode=[0, 0.1], labeldistance=0.5,
           textprops={"va": "bottom", "ha": "center"},
           colors=["gray", color])
    ax.set_title(f"{dscrp}")
    if save_figs:
        _savePanel(fig, save_prefix, f"pie_{dscrp}")
    plt.show()
    return sgf_mean, sgf_sem


def _savePanel(fig, save_prefix, name):
    assert save_prefix is not None, "Pass save_prefix to save"
    parent_dir = pathlib.Path(f"{save_prefix}/sgf_tests/prev_outcome_modulated")
    parent_dir.parent.mkdir(exist_ok=True)
    parent_dir.mkdir(exist_ok=True)
    fig.savefig(parent_dir / f"{name}.svg", bbox_inches='tight')


def plotDiff(df, pval, save_prefix=None, save_figs=False):
    '''Figure S14A: per variable, a pooled pie, then bars and a pie per region.'''
    for data_col, data_col_df in df.groupby("DataCol"):
        for prior_data_col, prior_data_col_df in data_col_df.groupby("PriorDataCol"):
            dscrp = data_col + (" - " + prior_data_col if prior_data_col else "")
            _plotPrevModulatedPie(prior_data_col_df, pval=pval, data_col=data_col,
                                  dscrp=f"{dscrp} - MFC & LFC",
                                  save_prefix=save_prefix, save_figs=save_figs)
            for br_str, br_df in prior_data_col_df.groupby("BrainRegion"):
                fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                ax.bar(["Prev. Incorrect", "Prev. Correct"],
                       [br_df.PrevIncorrect.mean(), br_df.PrevCorrect.mean()],
                       yerr=[br_df.PrevIncorrect.sem(), br_df.PrevCorrect.sem()],
                       color=["red", "green"])
                ax.set_title(f"{dscrp} modulation by prev. outcome | {br_str}")
                ax.set_ylabel("Diff mean neuron Pref - Anti-Preferred")
                ax.spines[["top", "right"]].set_visible(False)
                if save_figs:
                    _savePanel(fig, save_prefix, f"mean_diff_{dscrp}_{br_str}")
                plt.show()
                _plotPrevModulatedPie(br_df, pval=pval, data_col=data_col,
                                      dscrp=f"{dscrp} - {br_str}",
                                      save_prefix=save_prefix, save_figs=save_figs)
