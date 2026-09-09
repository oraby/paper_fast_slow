'''Fast versus slow accuracy on easy trials (Figure S2M).

Backend for the "Humans fast/slow mean performance" section of
``behavior.ipynb``.

The comparison isolates the clearest case for the paper's claim that slow
trials are more accurate: on **easy** trials the evidence is unambiguous, so a
gap between fast and slow choices cannot be blamed on the stimulus. Each
subject contributes one fast and one slow accuracy, they are compared within
subject with a paired t-test, and the three contexts -- humans instructed for
accuracy, humans instructed for speed, and head-fixed mice -- are corrected
together with Holm-Bonferroni.

"Easy" is defined differently for the two species because their psychometric
functions sit at different coherences: humans are taken as ``|DV| >= 0.15``,
mice as the ``DVstr == "Easy"`` label already assigned upstream. Both are
preserved here as :data:`HUMAN_EASY_ABS_DV` and :data:`MICE_EASY_LABEL`.

Published values, all Holm-corrected: humans-accuracy P = 0.165 (n = 18),
humans-speed P = 0.006 (n = 18), mice P = 0.000 (n = 20).

Note the significance stars use a 0.025 threshold for a single star, not the
0.05 used elsewhere in the notebook. That is carried over unchanged --
:data:`STAR_THRESHOLDS` makes it explicit rather than hiding it in a
conditional -- because changing it would change the published annotation.
'''
from __future__ import annotations

from pathlib import Path
from typing import Mapping, NamedTuple, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ..common.clr import ExperimentContext

#: ``quantile_idx`` values for the fast and slow tertiles.
FAST, SLOW = 1, 3
#: Humans count a trial as easy at or above this absolute decision variable.
HUMAN_EASY_ABS_DV = 0.15
#: Mice use the difficulty label assigned upstream instead.
MICE_EASY_LABEL = "Easy"
#: Star thresholds. Note the single star is 0.025 here, not 0.05.
STAR_THRESHOLDS = ((0.001, "***"), (0.01, "**"), (0.025, "*"))
#: Gap between the three group's bar pairs along x.
GROUP_SPACING = 3


class GroupResult(NamedTuple):
    '''One context's paired fast-vs-slow comparison.'''
    label: str
    fast: np.ndarray
    slow: np.ndarray
    pvalue: float
    pvalue_corrected: float

    @property
    def n_subjects(self) -> int:
        return len(self.fast)


def easyHumanTrials(df: pd.DataFrame,
                    abs_dv: float=HUMAN_EASY_ABS_DV) -> pd.DataFrame:
    '''Human easy trials: coherence at or above ``abs_dv``.'''
    return df[df.DV.abs() >= abs_dv]


def easyMiceTrials(df: pd.DataFrame,
                   label: str=MICE_EASY_LABEL) -> pd.DataFrame:
    '''Mouse easy trials, by the upstream difficulty label.'''
    return df[df.DVstr == label]


def fastSlowAccuracy(df: pd.DataFrame) -> tuple:
    '''Mean percent-correct on fast and on slow trials of one subject.'''
    df = df[df.ChoiceCorrect.notnull()]
    return (df[df.quantile_idx == FAST].ChoiceCorrect.mean() * 100,
            df[df.quantile_idx == SLOW].ChoiceCorrect.mean() * 100)


def collectGroup(df: pd.DataFrame, subjects: Optional[Sequence]=None) -> tuple:
    '''Per-subject (fast, slow) accuracies for one context.

    ``subjects`` fixes the roster, which matters for the two human contexts:
    the published figure pairs each participant with itself across contexts,
    so both are iterated over the same subject list.
    '''
    if subjects is None:
        subjects = df.Name.unique()
    fast, slow = [], []
    for subject in subjects:
        subject_df = df[df.Name == subject]
        if not len(subject_df):
            raise ValueError(f"no trials for subject {subject!r}")
        fast_perf, slow_perf = fastSlowAccuracy(subject_df)
        fast.append(fast_perf)
        slow.append(slow_perf)
    return np.array(fast), np.array(slow)


def compareGroups(groups: Mapping[str, tuple],
                  method: str="holm") -> list:
    '''Paired t-test per context, then correct across contexts.

    ``groups`` maps a label to the ``(fast, slow)`` arrays from
    :func:`collectGroup`. Returns one :class:`GroupResult` per context, in the
    order given.
    '''
    labels = list(groups)
    pvalues = [stats.ttest_rel(fast, slow).pvalue
               for fast, slow in groups.values()]
    _, corrected, _, _ = multipletests(pvalues, method=method)
    return [GroupResult(label=label, fast=groups[label][0],
                        slow=groups[label][1], pvalue=float(raw),
                        pvalue_corrected=float(adj))
            for label, raw, adj in zip(labels, pvalues, corrected)]


def significanceLabel(pvalue: float,
                      thresholds: Sequence=STAR_THRESHOLDS) -> str:
    for threshold, star in thresholds:
        if pvalue <= threshold:
            return star
    return "ns"


def collectAllGroups(humans_accuracy_df: pd.DataFrame,
                     humans_speed_df: pd.DataFrame,
                     mice_df: pd.DataFrame) -> list:
    '''The three published contexts, easy trials only, already corrected.

    Human subjects are rostered from the speed frame so the two human contexts
    describe the same people.
    '''
    subjects = humans_speed_df.Name.unique()
    groups = {
        "Humans Accuracy": collectGroup(easyHumanTrials(humans_accuracy_df),
                                        subjects),
        "Humans Max Outcome": collectGroup(easyHumanTrials(humans_speed_df),
                                           subjects),
        "Mice": collectGroup(easyMiceTrials(mice_df)),
    }
    return compareGroups(groups)


def plotFastSlowAccuracy(results: Sequence[GroupResult],
                         colors: Optional[Sequence[str]]=None,
                         ax: Optional[plt.Axes]=None,
                         save_prefix: Optional[Union[str, Path]]=None,
                         save_figs: bool=False):
    '''Figure S2M: paired fast-vs-slow accuracy on easy trials.'''
    if save_figs and save_prefix is None:
        raise ValueError("save_figs=True needs a save_prefix")
    if colors is None:
        colors = (ExperimentContext.Accuracy, ExperimentContext.MaxOutcome,
                  ExperimentContext.Mice)

    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(
        1, 1, figsize=(10, 6))
    ticks, tick_labels = [], []
    for idx, (result, color) in enumerate(zip(results, colors)):
        x0 = idx * GROUP_SPACING
        x1 = x0 + 1
        for fast_perf, slow_perf in zip(result.fast, result.slow):
            ax.plot([x0 + .1, x1 - .1], [fast_perf, slow_perf], color="gray",
                    alpha=0.3, marker="o")
        ax.bar([x0, x1], [result.fast.mean(), result.slow.mean()],
               yerr=[stats.sem(result.fast), stats.sem(result.slow)],
               color=color, edgecolor=["red", "yellow"], alpha=1, zorder=-10,
               label=f"{result.label} (n={result.n_subjects})")
        ax.annotate(f"{significanceLabel(result.pvalue_corrected)} - "
                    f"pval={result.pvalue_corrected:.3f}",
                    xy=((x0 + x1) / 2,
                        max(result.fast.max(), result.slow.max()) + 1),
                    ha="center", va="bottom", fontsize="large", color="k")
        ticks += [x0, x0 + .5, x1]
        tick_labels += ["Fast", f"\n{result.label}", "Slow"]

    ax.legend(loc="upper right", fontsize="small", ncol=2)
    ax.set_title("Fast vs Slow Performance in Easy Trials")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.set_ylabel("Performance (%)")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_ylim(70, 104)

    if save_figs:
        save_fp = Path(save_prefix) / "slow_fast_perf.svg"
        save_fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fp, bbox_inches="tight")
    return fig
