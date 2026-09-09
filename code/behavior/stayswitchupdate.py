'''Win/lose update: how the previous outcome shifts staying versus switching.

Backend for the "Stay/Switch reaction time" and "Stay/Switch over time"
sections of ``behavior.ipynb`` (Figures 1I-right and S3G).

**Why the baseline subtraction.** A raw stay probability cannot be read
directly, because the sequence of rewarded sides has repetition structure of
its own: an animal that always chose the rewarded side would still look like it
was "staying" whenever the generator repeated a side. ``StayBaseline`` records,
per trial, whether repeating the previous choice *would have* matched the
rewarded side — the pattern an error-free agent would have produced given the
same history. The update is the excess over that baseline::

    update = 100 * (sum(Stay) - sum(StayBaseline)) / N

Positive means the previous choice was repeated more often than the sequence
alone explains; negative means switching was more common. It is computed
separately over Win trials (previous trial correct) and Lose trials (previous
trial incorrect).

**Figure 1I-right** takes each animal's Fast and Slow trials and reduces the
pair of updates to one number per strategy: the mean of the *absolute* Win and
Lose updates. The absolute value is deliberate — the question is how strongly
the previous outcome moved behaviour, not in which direction, so an animal
playing win-stay and one playing win-switch equally strongly should score the
same.

**Figure S3G** instead pools trials across animals, sorts them by z-scored
sampling time and walks a window along that axis, so the update can be read as
a function of how long the animal sampled.

Two animals are dropped from Figure S3G. The Methods define them by rule: an
abnormally large fraction of their trials falls below a sampling-time z-score
of -1, above the 90th percentile of the group. :func:`outlierSubjects` derives
that set rather than naming it, and on the published data it selects exactly
the two the notebook hard-coded (``RDK_WT6`` at 18.62%, ``RDK_WT1`` at 13.26%,
against a 90th percentile of 12.04%). The "group mean of 3.28%" the Methods
quote for comparison is the mean over the 18 *retained* animals; across all 20
it is 4.55%.

Note ``calcWinLoseUpdates`` is duplicated at ``rlmodel/model/plotter.py`` in a
reduced form that drops the trial counts. The arithmetic is identical.
'''
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional, Sequence, Union

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy import stats

from ..figcode.util import normalizeSTAcrossSubjects
from ..pipeline.utils import filterNanGaussianConserving

#: ``quantile_idx`` values for the within-difficulty sampling-time tertiles.
FAST, TYPICAL, SLOW = 1, 2, 3
#: Sampling-time z-score below which a trial counts toward the outlier
#: fraction, and the percentile of that fraction above which an animal is
#: dropped from Figure S3G.
OUTLIER_CUTOFF_Z = -1
OUTLIER_PERCENTILE = 90
#: Figure S3G walks this many trials per point, then smooths across points.
GROUP_EVERY = 800
SMOOTH_SIGMA = 1
#: Column ``normalizeSTAcrossSubjects`` writes the within-animal z-score into.
NORMED_ST_COL = "transformedCalcStimulusTime"
WIN_COLOR = "dodgerblue"
LOSE_COLOR = "m"


class UpdateComparison(NamedTuple):
    '''Paired Fast-vs-Slow comparison behind Figure 1I-right.'''
    per_subject: pd.DataFrame
    statistic: float
    pvalue: float
    fast_mean: float
    fast_sem: float
    slow_mean: float
    slow_sem: float


def calcWinLoseUpdates(df: pd.DataFrame, col_prefix: str="") -> tuple:
    '''Baseline-corrected update over Win trials and over Lose trials.

    Returns ``(win_update, n_win, lose_update, n_lose)``. Win and Lose are
    taken from ``PrevOutcomeCount``, whose sign carries the previous outcome.
    An empty set scores 0 rather than NaN, matching the published figures.
    '''
    def _calcUpdate(sub_df):
        if not len(sub_df):
            return 0
        return 100 * (sub_df[f"{col_prefix}Stay"].sum()
                      - sub_df[f"{col_prefix}StayBaseline"].sum()) / len(sub_df)

    count_col = f"{col_prefix}PrevOutcomeCount"
    prev_win_df = df[df[count_col] >= 1]
    prev_lose_df = df[df[count_col] <= -1]
    return (_calcUpdate(prev_win_df), len(prev_win_df),
            _calcUpdate(prev_lose_df), len(prev_lose_df))


def quantileUpdates(df: pd.DataFrame) -> pd.DataFrame:
    '''Per-animal Win/Lose updates within Fast and within Slow trials.

    Adds ``FastUpdate`` / ``SlowUpdate``: the mean of the two *absolute*
    updates, which is the quantity Figure 1I-right plots.
    '''
    rows = []
    for subject, subject_df in df.groupby("Name"):
        row = {"Name": subject}
        for label, idx in (("Fast", FAST), ("Slow", SLOW)):
            win, n_win, lose, n_lose = calcWinLoseUpdates(
                subject_df[subject_df.quantile_idx == idx])
            row |= {f"Win{label}Update": win, f"Win{label}Count": n_win,
                    f"Lose{label}Update": lose, f"Lose{label}Count": n_lose}
        for label in ("Fast", "Slow"):
            row[f"{label}Update"] = (abs(row[f"Win{label}Update"])
                                     + abs(row[f"Lose{label}Update"])) / 2
        rows.append(row)
    return pd.DataFrame(rows)


def compareQuantileUpdates(df: pd.DataFrame) -> UpdateComparison:
    '''Paired t-test of Slow against Fast update, one pair per animal.'''
    per_subject = quantileUpdates(df)
    test = stats.ttest_rel(per_subject.SlowUpdate, per_subject.FastUpdate)
    return UpdateComparison(
        per_subject=per_subject,
        statistic=float(test.statistic), pvalue=float(test.pvalue),
        fast_mean=float(per_subject.FastUpdate.mean()),
        fast_sem=float(per_subject.FastUpdate.sem()),
        slow_mean=float(per_subject.SlowUpdate.mean()),
        slow_sem=float(per_subject.SlowUpdate.sem()))


def significanceLabel(pvalue: float) -> str:
    '''"***" / "**" / "*" / "ns" at the usual thresholds.'''
    for threshold, label in ((0.001, "***"), (0.01, "**"), (0.05, "*")):
        if pvalue < threshold:
            return label
    return "ns"


def plotQuantileUpdate(df: pd.DataFrame, ax: Optional[plt.Axes]=None,
                       save_prefix: Optional[Union[str, Path]]=None,
                       save_figs: bool=False,
                       save_postfix: str="Mice") -> tuple:
    '''Figure 1I-right: Fast vs Slow update, paired within animal.

    Returns ``(fig, UpdateComparison)``.
    '''
    if save_figs and save_prefix is None:
        raise ValueError("save_figs=True needs a save_prefix")
    result = compareQuantileUpdates(df)
    per_subject = result.per_subject

    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(
        1, 1, figsize=(10, 8))
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar(1, result.fast_mean, yerr=result.fast_sem, color="r",
           label=f"Fast n={(df.quantile_idx == FAST).sum():,} Trials")
    ax.bar(2, result.slow_mean, yerr=result.slow_sem, color="yellow",
           label=f"Slow n={(df.quantile_idx == SLOW).sum():,} Trials")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Fast", "Slow"])
    for _, row in per_subject.iterrows():
        ax.plot([1.03, 1.97], [row.FastUpdate, row.SlowUpdate], c="gray",
                alpha=0.3, marker="o")

    max_y = max(per_subject.SlowUpdate.max(), per_subject.FastUpdate.max())
    ax.annotate(f"{significanceLabel(result.pvalue)} - p={result.pvalue:.3f}",
                xy=(1.5, max_y), ha="center", va="bottom", color="k")
    ax.set_ylabel("Update %")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_title("Win/Lose Update\n"
                 f"n={per_subject.Name.nunique()} subjects - "
                 rf"$\overline{{Fast}}={result.fast_mean:.2f}$ - "
                 rf"$\overline{{Slow}}={result.slow_mean:.2f}$")
    ax.set_xlim(left=0, right=3)
    ax.legend()

    if save_figs:
        _save(fig, save_prefix, "QuantileWinLoseUpdate",
              f"_total_{save_postfix}.svg" if save_postfix else "_total.svg")
    return fig, result


# --------------------------------------------------------------------------
# Figure S3G -- update as a function of sampling time
# --------------------------------------------------------------------------

def outlierFraction(df: pd.DataFrame,
                    cutoff_z: float=OUTLIER_CUTOFF_Z) -> pd.Series:
    '''Percentage of each animal's trials below ``cutoff_z`` in sampling time.'''
    def _fraction(subject_df):
        z = stats.zscore(subject_df.calcStimulusTime)
        return 100 * (z < cutoff_z).sum() / len(z)
    return df.groupby("Name").apply(_fraction, include_groups=False)


def outlierSubjects(df: pd.DataFrame, cutoff_z: float=OUTLIER_CUTOFF_Z,
                    percentile: float=OUTLIER_PERCENTILE) -> list:
    '''Animals whose outlier fraction exceeds the group ``percentile``.

    Derives the Methods' exclusion rule rather than naming the animals, so it
    stays correct if the dataset changes.
    '''
    fractions = outlierFraction(df, cutoff_z)
    threshold = np.percentile(fractions, percentile)
    return sorted(fractions[fractions > threshold].index)


def updateCurve(df: pd.DataFrame, st_col: str=NORMED_ST_COL,
                group_every: int=GROUP_EVERY,
                sigma: float=SMOOTH_SIGMA) -> tuple:
    '''Update as a function of sampling time, plus the flat average.

    Trials are sorted by ``st_col`` and cut into consecutive blocks of
    ``group_every``; each block contributes one point, placed at its smallest
    sampling time. Returns ``(x, y, average_update)``.
    '''
    df = df[df.Stay.notnull()].sort_values(by=st_col).reset_index(drop=True)
    average = 100 * (df.Stay.sum() - df.StayBaseline.sum()) / len(df)

    grouped = df.groupby(df.index // group_every)
    x = grouped[st_col].min()
    y = grouped.apply(lambda grp: 100 * (grp.Stay.sum()
                                         - grp.StayBaseline.sum()) / len(grp),
                      include_groups=False)
    x, y = x[y.notnull()], y[y.notnull()]
    return x, filterNanGaussianConserving(y, sigma=sigma, axis=0), average


def plotUpdateOverSamplingTime(df: pd.DataFrame,
                               exclude: Optional[Sequence[str]]=None,
                               st_col: str=NORMED_ST_COL,
                               title: str="Mice Strategy over decision times",
                               ax: Optional[plt.Axes]=None,
                               save_prefix: Optional[Union[str, Path]]=None,
                               save_figs: bool=False):
    '''Figure S3G: Win and Lose update against z-scored sampling time.

    ``exclude`` defaults to :func:`outlierSubjects`. The Fast / Typical / Slow
    bands behind the curves mark each tertile's mean sampling time.
    '''
    if save_figs and save_prefix is None:
        raise ValueError("save_figs=True needs a save_prefix")
    df = normalizeSTAcrossSubjects(df.copy())
    if exclude is None:
        exclude = outlierSubjects(df)
    df = df[~df.Name.isin(exclude)]

    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(
        1, 1, figsize=(10, 6))
    min_x, max_x = np.inf, -np.inf
    for label, outcome, color in (("Win", True, WIN_COLOR),
                                  ("Lose", False, LOSE_COLOR)):
        x, y, average = updateCurve(df[df.PrevChoiceCorrect == outcome],
                                    st_col=st_col)
        ax.axhline(average, color=color, linestyle="--",
                   label=f"{label} All Average")
        ax.plot(x, y, color=color, linestyle="-",
                label=f"{label} All (Grouped every {GROUP_EVERY} Trials"
                      f"- Gaussian sigma={SMOOTH_SIGMA})")
        min_x, max_x = min(min_x, x.min()), max(max_x, x.max())

    ax.set_xlabel("Stimulus Time (Z-Score)")
    ax.set_ylabel("Update (%)\nSwitch" + " " * 50 + "Stay", va="bottom")
    blended = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.yaxis.set_label_coords(x=-0.08, y=0, transform=blended)
    ax.set_ylim(-8, None)
    ax.set_xlim(min_x, 2.9)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.set_title(title)

    _shadeStrategyBands(ax, df, st_col)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.legend(fontsize="small", loc="upper right")

    if save_figs:
        _save(fig, save_prefix, None, f"win_lose_stay_switch_{title}.svg")
    return fig


def _shadeStrategyBands(ax, df: pd.DataFrame, st_col: str) -> None:
    '''Fast / Typical / Slow background, split at the tertile means.'''
    y_lim, x_lim = ax.get_ylim(), ax.get_xlim()
    stats_by_tertile = {}
    for idx in (FAST, SLOW):
        col = df[df.quantile_idx == idx][st_col]
        stats_by_tertile[idx] = (col.mean(), col.sem())
    for mean, sem in stats_by_tertile.values():
        ax.fill_betweenx(y_lim, mean - sem, mean + sem, color="gray",
                         alpha=0.3)
    fast_mean, slow_mean = (stats_by_tertile[FAST][0],
                            stats_by_tertile[SLOW][0])
    for lo, hi, color, label in ((x_lim[0], fast_mean, "r", "Fast Trials"),
                                 (fast_mean, slow_mean, "orange",
                                  "Typical Trials"),
                                 (slow_mean, x_lim[1], "yellow",
                                  "Slow Trials")):
        ax.fill_betweenx(y_lim, lo, hi, color=color, alpha=0.3, zorder=-1,
                         label=label)


def plotOutlierFraction(df: pd.DataFrame, cutoff_z: float=OUTLIER_CUTOFF_Z,
                        percentile: float=OUTLIER_PERCENTILE,
                        ax: Optional[plt.Axes]=None,
                        save_prefix: Optional[Union[str, Path]]=None,
                        save_figs: bool=False):
    '''The exclusion rule behind Figure S3G, drawn per animal.

    Animals above the percentile line are the ones :func:`outlierSubjects`
    drops; they keep their default colour, the retained ones are grey.
    '''
    if save_figs and save_prefix is None:
        raise ValueError("save_figs=True needs a save_prefix")
    fractions = outlierFraction(df, cutoff_z)
    threshold = np.percentile(fractions, percentile)

    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(
        1, 1, figsize=(8, 6))
    for subject, subject_df in df.groupby("Name"):
        fraction = fractions[subject]
        ax.scatter(len(subject_df), fraction,
                   label=f"{subject}= {fraction:.2f}%",
                   color=None if fraction > threshold else "gray")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(threshold, color="red", linestyle="--",
               label=f"{percentile:g}th Percentile={threshold:.2f}%")
    ax.set_xlabel("Number of Trials")
    ax.set_ylabel(f"Percentage of Trials with RT Z-Score < {cutoff_z} (%)")
    ax.set_title("Outlier Percentage by Subject")
    ax.legend(loc="upper right", fontsize="x-small", bbox_to_anchor=(1.3, 1.0))

    if save_figs:
        for ext in ("svg", "png"):
            _save(fig, save_prefix, None,
                  f"outlier_percentage_by_subject.{ext}")
    return fig


def _save(fig, save_prefix, subdir: Optional[str], name: str) -> None:
    base = Path(save_prefix)
    save_fp = (base / subdir / name) if subdir else (base / name)
    save_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_fp, bbox_inches="tight")
