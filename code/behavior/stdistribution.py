'''Sampling-time distribution of the two human contexts (Figure S2B).

Backend for the "Plot user data — Figure S2B" section of ``behavior.ipynb``.

One curve per context, accuracy and speed: the density of the subjects' mean
z-scored sampling times, with a dashed line at each context's mean.

The normalisation, in order:

1. Long sampling-duration trials are excluded.
2. Subject x context groups with ``MIN_TRIALS`` trials or fewer are dropped.
3. The sampling times of all subjects and both contexts, as one pool, are
   log-transformed as ``log(t - min + 1)`` and z-scored
   (:func:`zScorePooledLog`).
4. Each subject's mean is taken per context (:func:`subjectContextMeans`).

Because both contexts share one z-score, the difference between them stays in
the panel.
'''
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .stdispersion import DEF_ST_COL, MIN_TRIALS, filterSmallGroups

#: Subject identifier.
SUBJECT_KEY = "Name"
#: Column separating the two human instructions.
CONTEXT_COL = "session_type"
ACCURACY_CONTEXT = "ReactionTime"
SPEED_CONTEXT = "Competition"
#: Longest speed-context trial kept, in seconds.
SPEED_MAX_SECONDS = 3
#: Seaborn KDE bandwidth multiplier, and the edges of the density axis.
BW_ADJUST = 2
X_LIM = (-2, 3)
DEF_FIG_NAME = "humans_sampling_time_dist"


def dropLongSpeedTrials(df: pd.DataFrame, st_col: str=DEF_ST_COL,
                        max_seconds: float=SPEED_MAX_SECONDS) -> pd.DataFrame:
    '''``df`` without the speed-context trials longer than ``max_seconds``.'''
    too_long = (df[CONTEXT_COL] == SPEED_CONTEXT) & (df[st_col] > max_seconds)
    return df[~too_long]


def zScorePooledLog(df: pd.DataFrame, st_col: str=DEF_ST_COL) -> pd.DataFrame:
    '''``st_col`` replaced by the z-score of ``log(t - min + 1)`` over all rows.

    One pool: every subject and every context together. Uses the population
    sigma (``ddof=0``).
    '''
    df = df[df[st_col].notnull()].copy()
    st = df[st_col]
    logged = np.log(st - st.min() + 1)
    df[st_col] = (logged - logged.mean()) / logged.std(ddof=0)
    return df


def subjectContextMeans(df: pd.DataFrame, st_col: str=DEF_ST_COL) -> pd.Series:
    '''Mean of ``st_col`` per subject and context.'''
    return df.groupby([SUBJECT_KEY, CONTEXT_COL], observed=True)[st_col].mean()


def collectContexts(df_users: pd.DataFrame, *, colors: Mapping[str, str],
                    session_types: Sequence, st_col: str=DEF_ST_COL,
                    min_trials: int=MIN_TRIALS,
                    speed_max_seconds: float=SPEED_MAX_SECONDS) -> list:
    '''One payload per context: subject means, colour, label, trial count.

    ``session_types`` is a sequence of ``(label, session_type)`` pairs naming
    the contexts to show, in plotting order. The normalisation is the one in
    the module docstring.
    '''
    df = dropLongSpeedTrials(df_users, st_col, speed_max_seconds)
    df = filterSmallGroups(df, [SUBJECT_KEY, CONTEXT_COL], st_col=st_col,
                           min_trials=min_trials)
    df = zScorePooledLog(df, st_col)
    groups = []
    for label, session_type in session_types:
        context_df = df[df[CONTEXT_COL] == session_type]
        means = subjectContextMeans(context_df, st_col).droplevel(CONTEXT_COL)
        groups.append(dict(label=label, color=colors[label], values=means,
                           n_trials=len(context_df)))
    return groups


def plotSTDistribution(df_users: pd.DataFrame, *,
                       colors: Mapping[str, str],
                       session_types: Sequence,
                       st_col: str=DEF_ST_COL,
                       min_trials: int=MIN_TRIALS,
                       speed_max_seconds: float=SPEED_MAX_SECONDS,
                       bw_adjust: float=BW_ADJUST, x_lim: Sequence=X_LIM,
                       ax=None,
                       save_prefix: Optional[Union[str, Path]]=None,
                       save_figs: bool=False,
                       fig_name: str=DEF_FIG_NAME,
                       verbose: bool=True) -> list:
    '''Figure S2B: density of the subject-mean sampling time per context.

    Saves ``{save_prefix}/{fig_name}.svg`` when ``save_figs``. Returns the
    :func:`collectContexts` payloads.
    '''
    if save_figs and save_prefix is None:
        raise ValueError("save_figs=True needs a save_prefix")
    groups = collectContexts(df_users, colors=colors,
                             session_types=session_types, st_col=st_col,
                             min_trials=min_trials,
                             speed_max_seconds=speed_max_seconds)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()
    ax.spines[["top", "right"]].set_visible(False)

    for group in groups:
        values, color = group["values"], group["color"]
        label = (f"{group['label']}\n(n={len(values):,} Subjects, "
                 f"{group['n_trials']:,} Trials)")
        if verbose:
            print(f"{group['label']} - Num of subjects: {len(values):,} - "
                  f"Num of trials: {group['n_trials']:,} - "
                  f"Mean: {values.mean():+.3f}")
        sns.kdeplot(values, ax=ax, color=color, label=label,
                    bw_adjust=bw_adjust)
        ax.axvline(values.mean(), color=color, lw=2, ls="--")

    ax.set_xlim(x_lim)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Density")
    ax.set_xlabel("Sampling time (z-scored)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), fontsize="x-small",
              frameon=False)

    if save_figs:
        save_fp = Path(save_prefix) / f"{fig_name}.svg"
        save_fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fp, bbox_inches="tight")
    return groups
