'''Sampling-time distribution across behavioural contexts (Figure S2B).

Backend for the "Plot user data" section of ``behavior.ipynb``.

The panel asks whether the same person samples for different lengths of time
under an accuracy instruction than under a speed instruction, and where mice
sit relative to both. Each subject contributes one point per context — the
mean of its z-scored sampling times — so the comparison is between subjects'
*positions*, not between individual trials.

**The normalisation pools each subject's contexts.** A subject is z-scored
once, over all of its trials in both contexts together, and only then split by
context. That is what preserves the thing the figure exists to show: if each
context were z-scored separately, every context would centre on zero by
construction and the difference would vanish. It also removes between-subject
offsets, so a naturally slow participant does not drag its whole context
rightward. The Methods say the data are "normalized across both experiment
types", which is this.

**Mice are the exception, because they have only one context.** Applying the
same per-subject rule to them is degenerate: centring each animal on its own
mean makes every animal's mean z-score exactly 0, so the group collapses to a
spike with sd = 0 and contributes nothing to the panel or the test. They are
therefore z-scored across the animals pooled, which keeps between-animal
differences. ``animals_per_subject=True`` restores the symmetric-but-degenerate
behaviour. See :func:`_zScoreAnimals`.

**This replaces a cell that could not run.** The inline version called

```python
df.groupby(grpby_keys, as_index=False).filter(
    lambda df: len(df) > 50).calcStimulusTime.transform(zscore)
```

which raises ``ValueError: Function did not transform`` under pandas 2.3.3,
because ``zscore`` returns an ndarray that ``Series.transform`` will not
align. Underneath that, ``.filter()`` returns a DataFrame rather than a
groupby, so the z-score was applied to the whole pooled frame and
``grpby_keys`` only ever drove the ``> 50`` trial cut-off — the normalisation
was never per subject at all. See ``docs/manuscript-issues.md``.

The statistics — Shapiro-Wilk per group, then one-way ANOVA + Tukey if every
group is normal, else Kruskal-Wallis + Dunn with Holm correction — are shared
with :mod:`behavior.stdispersion`, which draws the companion dispersion panel
(Figure S2C) from the same trials under a different normalisation.
'''
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .stdispersion import (DEF_ST_COL, MIN_TRIALS, annotateSignificance,
                           compareGroups, zScoreWithinSubject)

#: Subject identifier. Z-scoring is done per subject, across every context.
SUBJECT_KEY = "Name"
#: Column separating the two human instructions.
CONTEXT_COL = "session_type"
#: Seaborn KDE bandwidth multiplier, and the right edge of the density axis.
BW_ADJUST = 2
X_RIGHT = 3
DEF_FIG_NAME = "humans_mice_reaction_time_dist"
#: Whether the single-context animal group is z-scored per animal too.
#: See :func:`_zScoreAnimals` -- True collapses the group to a spike.
ANIMALS_PER_SUBJECT = False


def subjectContextMeans(df: pd.DataFrame, context_col: Optional[str]=None,
                        st_col: str=DEF_ST_COL) -> pd.Series:
    '''Mean z-scored sampling time per subject (per context, if given).

    Expects an already-normalised frame — see :func:`collectContexts`, which
    normalises before splitting so the contexts stay comparable.
    '''
    keys = [SUBJECT_KEY] + ([context_col] if context_col else [])
    return df.groupby(keys, observed=True)[st_col].mean()


def collectContexts(df_users: pd.DataFrame,
                    animals_df: Optional[pd.DataFrame]=None, *,
                    colors: Mapping[str, str],
                    session_types: Sequence,
                    animals_label: str="Mice",
                    st_col: str=DEF_ST_COL,
                    min_trials: int=MIN_TRIALS,
                    animals_per_subject: bool=ANIMALS_PER_SUBJECT) -> list:
    '''One payload per context: subject means, colour, label, trial count.

    ``session_types`` is a sequence of ``(label, session_type)`` pairs naming
    the human contexts to show, in plotting order.

    Order of operations matters and is the point of this function: humans are
    z-scored **once across both contexts** and only then split, so the
    between-context difference survives.
    '''
    groups = []
    humans = zScoreWithinSubject(df_users, [SUBJECT_KEY], st_col=st_col,
                                 min_trials=min_trials)
    for label, session_type in session_types:
        context_df = humans[humans[CONTEXT_COL] == session_type]
        groups.append(dict(label=label, color=colors[label],
                           values=subjectContextMeans(context_df, CONTEXT_COL,
                                                      st_col),
                           n_trials=len(context_df)))
    if animals_df is not None:
        animals_df = animals_df[animals_df.ChoiceCorrect.notnull()]
        animals = _zScoreAnimals(animals_df, animals_per_subject, st_col,
                                 min_trials)
        groups.append(dict(label=animals_label, color=colors[animals_label],
                           values=subjectContextMeans(animals, None, st_col),
                           n_trials=len(animals)))
    return groups


def _zScoreAnimals(animals_df: pd.DataFrame, per_subject: bool, st_col: str,
                   min_trials: int) -> pd.DataFrame:
    '''Normalise the animal group, which has only one context.

    ``per_subject=True`` applies the same rule as the humans. It is degenerate
    here: with a single context, centring each animal on its own mean makes
    every animal's mean z-score exactly 0, so the group collapses to a spike
    and contributes no spread to the panel or the omnibus test.

    ``per_subject=False`` (the default) z-scores across the animals pooled, so
    between-animal differences survive and the group stays informative. The
    asymmetry is deliberate: removing a subject's offset is meaningful for the
    humans because each of them spans both contexts, and there is no such
    contrast to protect for the mice.
    '''
    if per_subject:
        return zScoreWithinSubject(animals_df, [SUBJECT_KEY], st_col=st_col,
                                   min_trials=min_trials)
    df = animals_df[animals_df[st_col].notnull()].copy()
    df[st_col] = (df[st_col] - df[st_col].mean()) / df[st_col].std(ddof=0)
    return df


def plotSTDistribution(df_users: pd.DataFrame,
                       animals_df: Optional[pd.DataFrame]=None, *,
                       colors: Mapping[str, str],
                       session_types: Sequence,
                       animals_label: str="Mice",
                       st_col: str=DEF_ST_COL,
                       min_trials: int=MIN_TRIALS,
                       animals_per_subject: bool=ANIMALS_PER_SUBJECT,
                       bw_adjust: float=BW_ADJUST, x_right: float=X_RIGHT,
                       axes: Optional[Sequence]=None,
                       save_prefix: Optional[Union[str, Path]]=None,
                       save_figs: bool=False,
                       fig_name: str=DEF_FIG_NAME,
                       verbose: bool=True) -> dict:
    '''Figure S2B: subject-mean sampling time by context.

    Left: density of the per-subject means, with each context's median marked.
    Right: the same values as violins, with the omnibus test and its post-hoc
    brackets. Returns the :func:`~behavior.stdispersion.compareGroups` result.
    '''
    if save_figs and save_prefix is None:
        raise ValueError("save_figs=True needs a save_prefix")
    groups = collectContexts(df_users, animals_df, colors=colors,
                             session_types=session_types,
                             animals_label=animals_label, st_col=st_col,
                             min_trials=min_trials,
                             animals_per_subject=animals_per_subject)

    if axes is not None:
        ax, ax_violin = axes
        fig = ax.get_figure()
    else:
        fig, (ax, ax_violin) = plt.subplots(1, 2, figsize=(12, 5))
    for axis in (ax, ax_violin):
        axis.spines[["top", "right", "left", "bottom"]].set_visible(False)

    tick_labels, samples = [], {}
    for position, group in enumerate(groups, 1):
        values, color = group["values"], group["color"]
        legend_label = (f"{group['label']}\n (n={len(values):,} Subjects,\n"
                        f"{group['n_trials']:,} Trials)")
        if verbose:
            print(f"{group['label']} - Num of subjects: {len(values):,} - "
                  f"Num of trials: {group['n_trials']:,}")
        sns.kdeplot(values, ax=ax, color=color, label=legend_label,
                    bw_adjust=bw_adjust)
        ax.axvline(values.median(), color=color, lw=2, ls="--")

        parts = ax_violin.violinplot(values, positions=[position], widths=0.5,
                                     showextrema=False, showmedians=True)
        for body in parts["bodies"]:
            body.set_facecolor(color)
        parts["cmedians"].set_color(color)
        ax_violin.scatter([position] * len(values), values, s=8, c=color,
                          alpha=0.6)
        tick_labels.append(legend_label.replace(" ", "\n", 1))
        samples[legend_label] = values.values

    ax.set_xlim(right=x_right)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Density")
    ax.set_xlabel("Z-Scored Sampling Time")
    ax.set_title("Sampling-Time distribution\n"
                 "(each subject z-scored across both contexts)")
    ax.legend(loc="upper right", fontsize="xx-small")

    stats_res = compareGroups(samples, verbose=verbose)
    all_values = np.concatenate([g["values"].values for g in groups])
    span = np.nanmax(all_values) - np.nanmin(all_values)
    y_step = span * .18
    max_y = annotateSignificance(ax_violin, stats_res["posthoc"],
                                 y_start=np.nanmax(all_values) + y_step * .4,
                                 y_step=y_step)

    ax_violin.axhline(0, color="gray", alpha=0.4, ls="--")
    ax_violin.set_xticks(range(1, len(groups) + 1))
    ax_violin.set_xticklabels(tick_labels, fontsize="small")
    ax_violin.set_xlabel("Session Type")
    ax_violin.set_ylabel("Z-Scored Sampling Time")
    ax_violin.set_ylim(np.nanmin(all_values) - y_step * .2, max_y + y_step * .2)
    ax_violin.set_title(
        f"Sampling-Time distribution violin-plot\n"
        f"Sgf. test: {stats_res['test_name']} - "
        f"p-val: {stats_res['pvalue']:.3g}\n"
        f"Post-hoc: {stats_res['post_hoc_str']}")

    if save_figs:
        save_fp = Path(save_prefix) / f"{fig_name}.svg"
        save_fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fp, bbox_inches="tight")
    return stats_res
