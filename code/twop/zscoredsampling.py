'''The sampling epoch cut from the dF/F traces and z-scored per session.

The first two steps of the "Z-score before cutting" section of
``2p_1_population`` (Figures 4E, S8, S11A), also the input of the choice
decoders in ``2p_3_choice`` (Figure S12G) when they are re-run rather than
read from ``data/2p/svm_df.pkl``.

- ``dropDecisionTimeTrials`` -- keeps the trials that reached the lateral port
  without a "Decision Time" epoch and have a sampling time.
- ``cutSamplingZScored`` -- per session, the sampling epoch plus
  ``time_before`` s before it and ``time_after`` s after movement onset,
  z-scored after cutting, in real time (not stretched).
'''
from __future__ import annotations

import pandas as pd

from ..common.imaging import getSamplingEpoch
from ..pipeline import tracesnormalize

TRIAL_KEYS = ["Name", "Date", "SessionNum", "TrialNumber"]


def dropDecisionTimeTrials(df):
    '''Trials that moved to the lateral port without a "Decision Time" epoch.'''
    print("Removing decision time trials and nan-time trials...")
    len_before = len(df)
    df = df.groupby(TRIAL_KEYS).filter(
        lambda trial_df: not any(trial_df.epoch == "Decision Time") and any(
            trial_df.epoch == "Movement to Lateral Port"))
    df = df[df.calcStimulusTime.notnull()]
    len_now = len(df)
    print(f"Decision time filter: {len_now:,}/{len_before:,} "
          f"({len_before-len_now:,} rows (not trials) removed)")
    print("Done...")
    return df.reset_index(drop=True)


def cutSamplingZScored(df, time_before, time_after):
    '''The sampling epoch with its margins, z-scored per session after cutting.'''
    sess_df_li = []
    for _sess, sess_df in df.groupby("ShortName"):
        sess_df = getSamplingEpoch(
            sess_df.copy(), epoch="Sampling",
            normalization=tracesnormalize.NormalizeZScore(set_name="neuronal"),
            time_epoch_before=time_before, time_epoch_after=time_after,
            normalize_after_cutting=True, normalize_epoch_time=False,
            concatenate_final_epochs=False)
        sess_df_li.append(sess_df)
    return pd.concat(sess_df_li).reset_index(drop=True)
