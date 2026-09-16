'''Choice decoder over the population (Figure S12G).

Backend for the "Decoders" section of ``plottraces3.ipynb``.

``runTest`` fits one decoder per session on that session's simultaneously
recorded neurons and asks how well the animal's choice can be read out, at the
start and at the end of sampling. ``runCombinations`` repeats that across the
splits the panel shows -- by strategy (fast/slow), and optionally by difficulty.

**Accuracy is measured over many random splits**, ``num_runs`` of them
(50 in the notebook), each holding out a share of the trials; the panel plots
the spread across those runs, not a single fit. Sessions are never pooled: a
decoder never sees two sessions' neurons, because they were not recorded
together.

The heavy lifting -- the classifier itself and the label construction -- is in
:mod:`.classifyplayground` and :mod:`.classifiercreatelabels`, which were
already modules.

**The published table is cached.** ``data/2p/svm_df.pkl`` holds the result, and
the notebook loads it rather than re-running: a full run is minutes of fitting,
and re-running would change the numbers, since the splits are random and
unseeded.

Extracted from the notebook unchanged, except that ``num_runs`` is a parameter
rather than a notebook global.
'''
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import classifiercreatelabels
from .classifyplayground import _runClassifier, test

#: Random train/test splits per decoder, as the paper ran it.
DEFAULT_NUM_RUNS = 50

# The first few sessions print their feature shape, once per session, as a
# sanity check on how many neurons went into the decoder.
print_count = 0
printed_last_sess = False

def _processDF(br, sess, sess_df, only_traces_ids=[],
               num_runs=DEFAULT_NUM_RUNS):
    global print_count, printed_last_sess
    features = []
    labels = []
    sess_df = sess_df[sess_df.ChoiceLeft.notnull()]
    for _, row in sess_df.iterrows():
        labels.append(row.ChoiceLeft)
        traces_dict = row.traces_sets["neuronal"]
        s, e = row.trace_start_idx, row.trace_end_idx + 1
        cut_traces =  [trace_data[s:e].max() # .mean()
                       for trace_id, trace_data in traces_dict.items()
                       if not len(only_traces_ids) or
                          f"{trace_id}_{sess}" in only_traces_ids]
        if not len(cut_traces):
            continue
        whole_traces = np.array(cut_traces)
        # whole_traces = np.concatenate(cut_traces)
        if print_count < 3 and printed_last_sess != sess:
            print(f"{whole_traces.shape = }")
            print_count += 1
            printed_last_sess = sess
        features.append(whole_traces)
    if not len(features):
        return None
    features = np.array(features)
    labels = np.array(labels)
    accuracy_li, recall_li, precision_li, predictions_li_li, labels_li_li = \
        _runClassifier(features,  labels, labels_weights=None,
                       num_runs=num_runs, return_all=True)
    res_df = dict(accuracy=accuracy_li, recall=recall_li,
                  precision=precision_li,
                  predictions_li=predictions_li_li, labels_li=labels_li_li,
                  iteration_idx=list(range(1, num_runs+1))
                )
    res_df = pd.DataFrame(res_df)
    res_df["BrainRegion"] = br
    res_df["ShortName"] = sess
    return res_df

def _runSession(br, sess, sess_df, only_traces_ids=[], groupby_cols=None,
                num_runs=DEFAULT_NUM_RUNS):
    if groupby_cols is None:
        return _processDF(br, sess, sess_df, only_traces_ids, num_runs)
    else:
        res_df_li = []
        for group, group_df in sess_df.groupby(groupby_cols):
            res_df = _processDF(br, sess, group_df, only_traces_ids, num_runs)
            if res_df is None:
                continue
            if not isinstance(group, tuple):
                group = (group,)
                groupby_cols_iter = (groupby_cols,)
            else:
                groupby_cols_iter = groupby_cols
            for col, val in zip(groupby_cols_iter, group):
                res_df[col] = val
            res_df_li.append(res_df)
        if not len(res_df_li):
            return None
        return pd.concat(res_df_li)

def runTest(df, split_by_quantile, split_by_difficulty, restrict_df=None,
            num_runs=DEFAULT_NUM_RUNS):
    df = df.copy()
    with open("../data/2p/sgf_choice.pkl", "rb") as f:
        sgf_traces_df = pickle.load(f)
        # sgf_traces_df = set(sgf_traces_df.long_trace_id)
        # sgf_traces_df = {f"{_id.rsplit('_', 1)[1]}_{_id.rsplit('_', 1)[0]}"
        # for _id in sgf_traces_df}
        if restrict_df is not None:
            sgf_traces_df = sgf_traces_df[sgf_traces_df.long_trace_id.isin(
                                        restrict_df.trace_id)]
        sgf_traces_df["long_trace_id"] = sgf_traces_df.long_trace_id.apply(
            lambda _id: f"{_id.rsplit('_', 1)[1]}_{_id.rsplit('_', 1)[0]}")
        # display(sgf_traces_df)
        sgf_traces_set = {row.long_trace_id:row.IsROCLeftTuend
                          for _, row in sgf_traces_df.iterrows()}
    # sgf_traces_set = []
    # df = df.sort_values(by="BrainRegion")
    dfs_descrp = [("Choice", df, "ChoiceLeft"),
    ]

    # print("col_label:", col_label)
    imp_cols = ["BrainRegion", "ShortName"
                # "ChoiceCorrect"
                ]
    groupby_cols = []
    if split_by_quantile:
        groupby_cols.append("quantile_idx")
        imp_cols.append("quantile_idx")
    if split_by_difficulty:
        groupby_cols.append("DVstr")
        imp_cols.append("DVstr")
    groupby_cols = None if not len(groupby_cols) else (groupby_cols[0]
                   if len(groupby_cols) == 1 else groupby_cols)
    res_df_li = []
    for descrp, df, col_label in tqdm(dfs_descrp):
        for br, br_df in tqdm(df.groupby(df.BrainRegion)):
            for sess, sess_df in tqdm(br_df.groupby("ShortName")):
                res_df = _runSession(br, sess, sess_df, sgf_traces_set,
                                      groupby_cols=groupby_cols,
                                      num_runs=num_runs,
                                      #  groupby_cols=["ChoiceCorrect"],
                                     )
                if res_df is None:
                    print("Skipping:", sess)
                    continue
                print("Sess accurancy:", res_df.accuracy.mean())
                res_df_li.append(res_df)

    all_res_df = pd.concat(res_df_li)
    # Bring br and sess as first columns
    all_res_df = all_res_df[imp_cols + [
                      col for col in all_res_df.columns if col not in imp_cols]]
    return all_res_df

def runCombinations(sampling_df, movement_df, num_runs=DEFAULT_NUM_RUNS):
    df_li = []
    for split_by_quantile in [False,
                              True
                              ]:
        for split_by_difficulty in [False,
                                    #True
                                    ]:
            kargs = dict(split_by_quantile=split_by_quantile, split_by_difficulty=split_by_difficulty)
            print(f"Running Sampling with {split_by_quantile = } and {split_by_difficulty = }")
            svm_sampling_df = runTest(sampling_df, restrict_df=None, num_runs=num_runs, #active_neurons_sampling_df)
                                      **kargs)
            svm_sampling_df["epoch"] = "Sampling"
            print(f"Running movement with {split_by_quantile = } and {split_by_difficulty = }")
            svm_movement_df = runTest(movement_df, restrict_df=None, num_runs=num_runs, #active_neurons_movement_df)
                                      **kargs)
            svm_movement_df["epoch"] = "Movement"
            df_li.append(svm_sampling_df)
            df_li.append(svm_movement_df)
    return pd.concat(df_li)
