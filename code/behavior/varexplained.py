'''Per-predictor variance explained in sampling time (Figure 2A).

Backend for the ``# Behavior Parameters Variance Explained`` section of
``behavior.ipynb``.

For each animal, sampling time is log-transformed and fitted with OLS against
four predictors chosen to be as close to non-overlapping as possible. The
measure of fit is the model log-likelihood. Fitting a null (intercept-only)
model and the full model gives the **log-likelihood improvement** the
predictors buy; refitting with one predictor left out and comparing that
improvement to the full one gives each predictor's share:

    share(p) = 100 * (1 - improvement(full without p) / improvement(full))

Leave-one-out attributes shared variance to *neither* of two collinear
predictors, so the shares only add up to ~100% when the design matrix is close
to orthogonal. That is why the predictors are the ones they are:

* ``DVabs`` -- stimulus difficulty, ``1 - |DV|``, increasing from easy to hard.
  The *unsigned* value is used because the sign of DV correlates with the
  choice-side term.
* ``PrevOutcomeCount`` -- a single signed outcome-streak count (+k after k
  rewarded trials, -k after k unrewarded ones) rather than separate
  previous-outcome and reward-rate regressors: its sign already carries the
  previous outcome and its magnitude the run length.
* ``ChoiceLeft`` -- choice side, the motor-bias term.
* ``Stay`` -- one stay/switch indicator rather than separate win-stay and
  lose-stay terms, for the same reason.

``totalVarianceExplained`` and the ``cond_no`` column returned by
``loopSubjects`` are the two reported checks: in the published fit the four
shares sum to 101.9% and the per-animal condition numbers span 14.3-22.2.

**Read the sum with care.** Log-likelihood improvement is logarithmic in the
residual sum of squares, so leave-one-out shares are only near-additive while
the predictors explain little of the variance. Two *perfectly orthogonal*
regressors sum to ~100% at R^2 = 0.03 but to ~160% at R^2 = 0.96. Here the
per-animal R^2 is 0.006-0.081, which is exactly the regime where the sum lands
near 100% whether or not the design is orthogonal -- so a ~100% total is weak
evidence of orthogonality. A *low* sum does indicate collinearity (duplicated
predictors drive both shares to ~0), and ``cond_no`` is the direct check.
See ``tests/test_varexplained.py`` for both regimes.

Note that ``prepareSubject`` also computes ``RewardRate``, ``PrevOutcome``,
``WinStay`` and ``LoseStay``. They are not in ``PREDICTORS`` -- they are the
collinear alternatives the list above rejects -- but they are available for
callers that want to refit with a different set.

**A trial is dropped if any of** ``REQUIRED_COLS`` **is null**, which includes
``RewardRate5`` even though the reward rate is not itself a default predictor.
That is deliberate: it is what the published figure did, and changing it would
change the trial count.
'''
from __future__ import annotations

from pathlib import Path
from typing import Mapping, NamedTuple, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from .rewardrate import calcAvgRewardRate

#: Predictors of the published full model, in plotting order.
PREDICTORS = ("DVabs", "PrevOutcomeCount", "ChoiceLeft", "Stay")

#: An animal needs *more* than this many trials to be fitted at all. Counted
#: before the null-predictor rows are dropped, so it is a floor on session
#: coverage rather than on the fitted sample. Nine of the twenty animals pass.
MIN_TRIALS = 2_500

#: Columns ``prepareSubject`` reads. A trial missing any of them is dropped.
REQUIRED_COLS = ("Name", "Date", "SessionNum", "DV", "ChoiceLeft",
                 "calcStimulusTime", "PrevChoiceCorrect", "Stay",
                 "RewardRate5", "PrevOutcomeCount")

#: Bar colours. Covers the non-default predictors too, so a caller refitting
#: with a different set still gets a colour.
PREDICTOR_COLORS = {
    "DVabs": "gray",
    "RewardRate": "b",
    "PrevOutcomeCount": "orange",
    "PrevOutcome": "g",
    "ChoiceLeft": "k",
    "Stay": "purple",
    "WinStay": "r",
    # Was "rose" inline, which is not a matplotlib colour name and would have
    # raised had LoseStay ever been fitted. "mistyrose" is the nearest real one.
    "LoseStay": "mistyrose",
}

DEF_ST_COL = "calcStimulusTime"
DEF_FIG_NAME = "model_OLS_var_explained_w_filter.svg"
#: Where the NaN markers sit on the y-axis, in variance-explained percent.
NAN_MARKER_Y = -5


class SubjectFit(NamedTuple):
    '''One animal's leave-one-out fit.

    ``var_explained`` maps predictor -> percent of the full model's
    log-likelihood improvement lost when that predictor is dropped.
    ``cond_no`` is the full model's design-matrix condition number, the
    collinearity check quoted in the Methods.
    '''
    var_explained: dict
    cond_no: float
    n_trials: int
    null_llf: float
    full_llf: float


def prepareSubject(df: pd.DataFrame, st_col: str=DEF_ST_COL) -> pd.DataFrame:
    '''Drop incomplete trials and derive the regressors for one animal.

    Returns a frame carrying ``RT`` (**not** yet logged; ``varianceExplained``
    does that) plus every column named in :data:`PREDICTOR_COLORS` that can be
    derived from the input.
    '''
    df = df[list(REQUIRED_COLS)]
    for col in df.columns:
        df = df[df[col].notnull()]

    df = df.copy()
    df["uniq_sess"] = (df.Name + "_" + df.Date.astype(str) + "_"
                       + df.SessionNum.astype(str))
    df["RT"] = df[st_col]
    df["DVabs"] = 1 - df.DV.abs()
    df["RewardRate"] = df.RewardRate5
    df["PrevOutcome"] = df.PrevChoiceCorrect.astype(int)
    prev_win = df.PrevChoiceCorrect
    prev_lose = (1 - prev_win).abs()
    stay = df.Stay
    df["WinStay"] = (prev_win * stay).astype(int)
    df["LoseStay"] = (prev_lose * stay).astype(int)
    df["Stay"] = df.Stay.astype(int)
    df["ChoiceLeft"] = df.ChoiceLeft.astype(int)
    return df.drop(columns=[st_col, "DV", "PrevChoiceCorrect", "RewardRate5"])


def varianceExplained(subject_df: pd.DataFrame,
                      predictors: Sequence[str]=PREDICTORS,
                      rt_col: str="RT") -> SubjectFit:
    '''Leave-one-out share of the log-likelihood improvement, per predictor.

    ``subject_df`` is the output of :func:`prepareSubject`; ``rt_col`` is
    log-transformed here, so pass raw seconds.
    '''
    predictors = list(predictors)
    df = subject_df.copy()
    df[rt_col] = np.log(df[rt_col])

    null_llf = smf.ols(f"{rt_col} ~ 1", data=df).fit().llf
    full_fit = smf.ols(f"{rt_col} ~ " + " + ".join(predictors), data=df).fit()
    # Improvement the full model buys over the intercept-only model. Higher
    # llf is better, so this is positive whenever the predictors help.
    full_improvement = full_fit.llf - null_llf

    var_explained = {}
    for col in predictors:
        cur_cols = [c for c in predictors if c != col]
        cur_df = df[[rt_col] + cur_cols]
        cur_fit = smf.ols(f"{rt_col} ~ " + " + ".join(cur_cols),
                          data=cur_df).fit()
        cur_improvement = cur_fit.llf - null_llf
        var_explained[col] = 100 * (1 - cur_improvement / full_improvement)

    return SubjectFit(var_explained=var_explained,
                      cond_no=float(full_fit.condition_number),
                      n_trials=len(df), null_llf=float(null_llf),
                      full_llf=float(full_fit.llf))


def loopSubjects(df: pd.DataFrame, predictors: Sequence[str]=PREDICTORS,
                 min_trials: int=MIN_TRIALS,
                 st_col: str=DEF_ST_COL) -> pd.DataFrame:
    '''Fit every eligible animal; one row per animal.

    Columns: ``Name``, one per predictor (percent), plus the ``cond_no`` and
    ``n_trials`` diagnostics.

    Sessions containing any null ``DV`` are dropped whole before the
    ``min_trials`` count is taken, so an animal is judged on the sessions that
    can actually be fitted.
    '''
    df = df.copy()
    df = df.sort_values(by=["Name", "Date", "SessionNum", "TrialNumber"])
    df = calcAvgRewardRate(df)
    df = df.groupby(["Name", "Date", "SessionNum"]).filter(
        lambda sess_df: sess_df.DV.isnull().sum() == 0)
    df = df.groupby("Name").filter(lambda grp: len(grp) > min_trials)

    rows = []
    for subject, subject_df in df.groupby("Name"):
        fit = varianceExplained(prepareSubject(subject_df, st_col=st_col),
                                predictors=predictors)
        rows.append({"Name": subject, **fit.var_explained,
                     "cond_no": fit.cond_no, "n_trials": fit.n_trials})
    return pd.DataFrame(rows)


def totalVarianceExplained(var_exp_df: pd.DataFrame,
                           predictors: Sequence[str]=PREDICTORS) -> float:
    '''Sum of the across-animal mean shares.

    ~100% is the evidence that the predictors are close to non-overlapping;
    the published fit gives 101.9%.
    '''
    means = var_exp_df.groupby("Name")[list(predictors)].mean()
    return float(sum(means[col].dropna().mean() for col in predictors))


def plotVarianceExplained(var_exp_df: pd.DataFrame,
                          predictors: Sequence[str]=PREDICTORS,
                          colors: Mapping[str, str]=PREDICTOR_COLORS,
                          ax: Optional[plt.Axes]=None,
                          save_prefix: Optional[Union[str, Path]]=None,
                          save_figs: bool=False,
                          fig_name: str=DEF_FIG_NAME):
    '''Bar chart of the mean +- SEM share per predictor (Figure 2A).

    Animals whose fit produced a NaN for a predictor are marked with a red x
    at ``NAN_MARKER_Y`` rather than being silently dropped from the bar.
    '''
    if save_figs and save_prefix is None:
        raise ValueError("save_figs=True needs a save_prefix")

    predictors = list(predictors)
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    xs = np.arange(len(predictors))
    grouped_by_name = var_exp_df.groupby("Name")

    for idx, col in zip(xs, predictors):
        vals = grouped_by_name[col].mean().values
        nan_idx = np.isnan(vals)
        vals = vals[~nan_idx]
        sem = stats.sem(vals)
        ax.bar(idx, vals.mean(), yerr=sem, color=colors[col],
               label=f"{col}\nn={len(vals):,}\n{vals.mean():.2f}% ±{sem:.2f}%")
        if nan_idx.any():
            offset_idx = np.linspace(-.2, .2, nan_idx.sum()) + idx
            ax.scatter(offset_idx, [NAN_MARKER_Y] * nan_idx.sum(), color='r',
                       marker='x')

    ax.axhline(0, color='gray', linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(predictors)
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("Variance Explained by each feature")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1), fontsize="small")
    ax.spines[["top", "right"]].set_visible(False)

    if save_figs:
        save_fp = Path(save_prefix) / fig_name
        save_fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fp, bbox_inches="tight")
    return fig
