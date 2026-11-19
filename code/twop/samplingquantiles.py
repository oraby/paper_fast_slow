'''Label each trial of a 2P frame with its sampling-time tercile.

Several 2P notebooks split trials into fast (``quantile_idx == 1``), typical
(2) and slow (3) by sampling time, per session, through
``behavior.util.splitdata.splitStimulusTimeByQuantile``. Two layouts of frame
need the label:

- ``assignQuantileIdx`` -- one row per trial (the sampling-aligned frames read
  from ``data/2p``). The rows keep their index and order.
- ``assignQuantileIdxByTrial`` -- several rows per trial (one per epoch). The
  tercile is decided on one row per trial, then copied to every row of that
  trial; rows with no sampling time are dropped, and the result is sorted by
  trial and epoch.
'''
from __future__ import annotations

import pandas as pd

from ..behavior.util.splitdata import splitStimulusTimeByQuantile

TRIAL_KEYS = ["Name", "Date", "SessionNum", "TrialNumber"]


def assignQuantileIdx(df: pd.DataFrame) -> pd.DataFrame:
    '''``df`` with a ``quantile_idx`` column, one row per trial, in index order.'''
    assert df.index.nunique() == len(df)
    df_li = []
    for quantile_idx, quantile_df in splitStimulusTimeByQuantile(df):
        quantile_df = quantile_df.copy()
        quantile_df["quantile_idx"] = quantile_idx
        df_li.append(quantile_df)
    return pd.concat(df_li).sort_index()


def assignQuantileIdxByTrial(df: pd.DataFrame) -> pd.DataFrame:
    '''``df`` with a ``quantile_idx`` column copied to every row of a trial.'''
    assert df.index.nunique() == len(df)
    df = df[df.calcStimulusTime.notnull()]
    org_df = df.copy()
    # Decide the tercile on one row per trial
    df = df.drop_duplicates(subset=TRIAL_KEYS).reset_index(drop=True)
    df_li = []
    for quantile_idx, quantile_df in splitStimulusTimeByQuantile(df):
        quantile_df = quantile_df.copy()
        quantile_df["quantile_idx"] = quantile_idx
        df_li.append(quantile_df)
    df = pd.concat(df_li)
    # ...and copy it back to every row of that trial
    org_df["quantile_idx"] = -1
    org_df_li = []
    for _, row in df.iterrows():
        mask = ((org_df["Name"] == row["Name"]) &
                (org_df["Date"] == row["Date"]) &
                (org_df["SessionNum"] == row["SessionNum"]) &
                (org_df["TrialNumber"] == row["TrialNumber"]))
        temp_df = org_df[mask].copy()
        temp_df["quantile_idx"] = row["quantile_idx"]
        org_df_li.append(temp_df)
    reassigned_df = pd.concat(org_df_li)
    reassigned_df = reassigned_df.sort_values(TRIAL_KEYS + ["trace_start_idx",
                                                            "state_id"])
    return reassigned_df.reset_index(drop=True)
