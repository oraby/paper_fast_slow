'''Which task variable each neuron is tuned to (Figure 6B, feeds 6E).

Backend for the "Extract sampling and feedback tuned neurons" section of
``2pAnalysis.ipynb``.

The ROC stage upstream tests every neuron against every task variable and
leaves one row per (neuron, variable) with a p-value and the mean activity on
each side of the split. This reduces that to **one row per neuron, one column
per variable**: the preferred-side value where the neuron is tuned
(``pval <= pval_threshold``), NaN where it is not. Downstream, "is this neuron
tuned to X" is then just ``notnull()`` on column X.

Three details worth keeping:

- **Only the plain comparisons are used.** Rows carrying a ``prior_data_col``
  (a variable split by a second, previous-trial variable) or a ``DVstr``
  (split by difficulty) are dropped first, so each variable is judged once.
- **The tuned value is the ROC-preferred side**, ``data_val_left`` when
  ``IsROCLeftTuend`` else ``data_val_right`` — not the larger mean. The
  notebook had tried ``IsLeftTunedMean`` and left it commented out.
- **A variable that no neuron in the frame was ever tested on is dropped**,
  because its column would be shorter than the others. The counts before and
  after are printed for exactly that reason.

``min_num_tuning`` and ``must_tuning_col_li`` keep only neurons tuned to at
least N variables, or to specific ones; the paper's call takes every neuron.

Extracted from the notebook unchanged except that ``PVAL`` is now the
``pval_threshold`` parameter rather than a notebook global.
'''
from __future__ import annotations

import numpy as np
import pandas as pd

#: The notebook's PVAL; every published call uses it.
DEFAULT_PVAL = 0.05
#: Columns that identify the neuron rather than a task variable.
ID_COLS = ["trace_id", "ShortName", "BrainRegion"]


def extractTracesPreferences(df, pval_threshold=DEFAULT_PVAL,
                             must_tuning_col_li=[], min_num_tuning=0,
                             verbose=True):
    '''One row per neuron; one column per task variable it was tested on.'''
    df = df.copy()
    df = df[df.prior_data_col.isnull()]     # not split by a previous-trial variable
    df = df[df.DVstr.isnull()]              # not split by difficulty
    traces_tuning = {"trace_id": [], "ShortName": [], "BrainRegion": []}
    traces_tuning.update({col: [] for col in df.data_col.unique()})

    for (sess, trace_id), trace_epoch_df in df.groupby(["ShortName", "trace_id"]):
        tuning = {}
        non_nan_cols = []
        for row in trace_epoch_df.itertuples():
            if row.pval <= pval_threshold:
                tuning[row.data_col] = (row.data_val_left if row.IsROCLeftTuend
                                        else row.data_val_right)
                non_nan_cols.append(row.data_col)
            else:
                tuning[row.data_col] = np.nan
        if len(non_nan_cols) < min_num_tuning:
            continue
        if len(must_tuning_col_li):
            if any((col not in non_nan_cols for col in must_tuning_col_li)):
                continue
        traces_tuning["BrainRegion"].append(row.BrainRegion)
        traces_tuning["ShortName"].append(sess)
        traces_tuning["trace_id"].append(f"{sess}_{trace_id}")
        for k, v in tuning.items():
            traces_tuning[k].append(v)

    if verbose:
        print("Before:")
        [print(col, len(li), end=", ") for col, li in traces_tuning.items()]
        print()
    # A variable no kept neuron was tested on has an empty column, which would
    # not line up with the rest.
    traces_tuning = {k: li for k, li in traces_tuning.items() if len(li)}
    if verbose:
        print("After:")
        [print(col, len(li), end=", ") for col, li in traces_tuning.items()]
        print()
    return pd.DataFrame(traces_tuning)
