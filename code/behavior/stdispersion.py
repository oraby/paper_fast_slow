'''Within-subject z-scored sampling time and per-subject dispersion.

Companion to the pooled "Plot user data" figure in ``behavior.ipynb``. Two
differences from that version:

* Each subject is z-scored against **its own** trials, and for the human data
  each experiment type (``session_type``) of a subject is z-scored separately
  instead of the subject's sessions being pooled together. Mice are z-scored
  per animal rather than across all animals. The left panel therefore shows
  trial-level z-scores -- per-subject means are 0 by construction, so plotting
  subject means would collapse to a spike at 0.
* The right panel is a dispersion measure of the **raw** (seconds) sampling
  time of each subject x experiment type, one point per subject: either the
  standard deviation (``measure="std"``, the default) or the inter-quartile
  range (``measure="iqr"``, robust to the long right tail). The z-scored
  dispersion is degenerate by construction -- sigma is 1.0 for everyone -- so
  it has to be read off the untransformed column either way.

The significance test follows the notebook's convention: Shapiro-Wilk on each
group first, then one-way ANOVA + Tukey HSD if every group is normal, else
Kruskal-Wallis + Dunn (Holm-corrected).
'''
from __future__ import annotations

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from scipy import stats

DEF_ST_COL = "calcStimulusTime"
# Same cut-off as the pooled notebook version: a subject needs more than this
# many trials in a group before its z-score / sigma is meaningful.
MIN_TRIALS = 50
DEF_MEASURE = "std"
DEF_IQR_QUANTILES = (0.25, 0.75)
MEASURES = ("std", "iqr")


def dispersionLabel(measure: str=DEF_MEASURE,
                    iqr_quantiles=DEF_IQR_QUANTILES) -> str:
    '''Axis-label tail naming the dispersion measure ("Subject σ" / "IQR").'''
    if measure == "std":
        return "Subject σ"
    lo, hi = iqr_quantiles
    if tuple(iqr_quantiles) == DEF_IQR_QUANTILES:
        return "Subject IQR"
    return f"Subject IQR ({lo*100:g}-{hi*100:g}%)"


def filterSmallGroups(df: pd.DataFrame, grpby_keys, st_col: str=DEF_ST_COL,
                      min_trials: int=MIN_TRIALS) -> pd.DataFrame:
    '''Drop null-``st_col`` rows and groups with ``<= min_trials`` trials.'''
    df = df[df[st_col].notnull()]
    if not min_trials:
        return df
    sizes = df.groupby(list(grpby_keys), observed=True)[st_col].transform("size")
    return df[sizes > min_trials]


def zScoreWithinSubject(df: pd.DataFrame, grpby_keys, st_col: str=DEF_ST_COL,
                        min_trials: int=MIN_TRIALS) -> pd.DataFrame:
    '''Z-score ``st_col`` within each ``grpby_keys`` group independently.

    Groups with ``<= min_trials`` non-null trials are dropped, as are rows with
    a null ``st_col``. Uses the population sigma (``ddof=0``) to match
    ``scipy.stats.zscore``.
    '''
    grpby_keys = list(grpby_keys)
    df = filterSmallGroups(df, grpby_keys, st_col=st_col,
                           min_trials=min_trials).copy()
    grp = df.groupby(grpby_keys, observed=True)[st_col]
    df[st_col] = (df[st_col] - grp.transform("mean")) / grp.transform("std",
                                                                     ddof=0)
    return df


def subjectDispersion(df: pd.DataFrame, grpby_keys, st_col: str=DEF_ST_COL,
                      min_trials: int=MIN_TRIALS, measure: str=DEF_MEASURE,
                      ddof: int=1, iqr_quantiles=DEF_IQR_QUANTILES
                      ) -> pd.Series:
    '''Dispersion of the raw ``st_col`` per ``grpby_keys`` group.

    One value per subject: ``measure="std"`` gives the standard deviation,
    ``measure="iqr"`` the inter-quartile range (the ``iqr_quantiles`` spread,
    robust to the long right tail of the sampling-time distribution).

    Must be fed the *untransformed* column: within-subject z-scoring makes every
    subject's sigma exactly 1.0.
    '''
    assert measure in MEASURES, f"measure must be one of {MEASURES}"
    grpby_keys = list(grpby_keys)
    df = df[df[st_col].notnull()]
    grp = df.groupby(grpby_keys, observed=True)[st_col]
    counts = grp.size()
    if measure == "std":
        dispersion = grp.std()
        if ddof != 1:  # pandas' std is ddof=1; rescale rather than re-aggregate
            dispersion *= np.sqrt((counts - 1) / (counts - ddof))
    else:
        lo, hi = iqr_quantiles
        dispersion = grp.quantile(hi) - grp.quantile(lo)
    return dispersion[counts > min_trials].rename(st_col)


def compareGroups(samples: dict, alpha: float=0.05, verbose: bool=True) -> dict:
    '''Normality-gated omnibus test + post-hoc, as done in the notebook.

    ``samples`` maps a group label to its 1-D array of values (here: one sigma
    per subject). Returns the omnibus test name / statistic / p-value, the
    per-group Shapiro-Wilk results and the post-hoc p-value matrix (indexed
    1..n in the order of ``samples``).
    '''
    labels = list(samples)
    values = [np.asarray(samples[label], dtype=float) for label in labels]

    normality = {label: stats.shapiro(vals)
                 for label, vals in zip(labels, values)}
    is_normal = all(res.pvalue >= alpha for res in normality.values())
    if verbose:
        for label, res in normality.items():
            print(f"{label} - Normality test:", res)
        print("Data is normal" if is_normal else "Data is not normal")

    if is_normal:
        omnibus = stats.f_oneway
        # Tukey HSD already controls the family-wise error rate, so it takes no
        # further p-value adjustment.
        posthoc = sp.posthoc_tukey(values)
        post_hoc_str = "Tukey's HSD test"
    else:
        omnibus = stats.kruskal
        p_adjust = "holm"
        posthoc = sp.posthoc_dunn(values, p_adjust=p_adjust)
        post_hoc_str = f"Dunn's test with {p_adjust} correction"

    statistic, p_val = omnibus(*values)
    if verbose:
        print(omnibus.__name__, "statistics:", statistic, "p_val:", p_val)
        print("post_hoc_str:", post_hoc_str)
    return dict(labels=labels, test_name=omnibus.__name__, statistic=statistic,
                pvalue=p_val, posthoc=posthoc, post_hoc_str=post_hoc_str,
                normality=normality, is_normal=is_normal)


def _pValStar(p_val: float) -> str:
    return ("***" if p_val <= 0.001 else "**" if p_val <= 0.01 else
            "*" if p_val <= 0.05 else "ns")


def pValLabel(p_val: float) -> str:
    '''Bracket text: the star plus the p-value, always spelled out at 4dp.

    Non-significant pairs carry their number too -- "ns" alone hides how close
    (or how far) the comparison actually was.
    '''
    return f"{_pValStar(p_val)} p={p_val:.4f}"


def _annotateSignificance(ax, posthoc: pd.DataFrame, y_start: float,
                          y_step: float, fontsize="small"):
    max_y = y_start
    for i, (_, row) in enumerate(posthoc.iterrows()):
        j_start = i + 1  # skip the mirrored half of the matrix
        for j, p_val in enumerate(row[j_start:], j_start):
            i_, j_ = i + 1, j + 1
            ax.plot([i_, i_, j_, j_],
                    [max_y, max_y + y_step*.12, max_y + y_step*.12, max_y],
                    lw=.5, c='k')
            ax.text((i_ + j_)/2, max_y + y_step*.1, pValLabel(p_val),
                    ha='center', va='bottom', c='k', fontsize=fontsize)
            max_y += y_step
    return max_y


def collectGroups(df_users: pd.DataFrame,
                  animals_df: Optional[pd.DataFrame]=None, *,
                  colors: dict, session_types: Union[list, tuple],
                  animals_label: str="Mice",
                  human_keys=("Name", "session_type"),
                  animal_keys=("Name",),
                  df_query: Optional[str]=None,
                  st_col: str=DEF_ST_COL, min_trials: int=MIN_TRIALS,
                  measure: str=DEF_MEASURE,
                  iqr_quantiles=DEF_IQR_QUANTILES) -> list:
    '''Build the per-group (z-scored trials, per-subject dispersion) payloads.

    ``session_types`` is a list of ``(label, session_type)`` pairs selecting the
    human experiment types. Z-scores are computed on all of a subject's trials
    in the group; ``df_query`` (e.g. correct-choice-only) is applied afterwards,
    to both the plotted z-scores and the dispersions, matching the pooled
    version. ``measure`` picks the dispersion statistic ("std" or "iqr").
    '''
    def _collect(df, keys, label):
        # The min-trials cut-off is a subject-inclusion criterion, so it is
        # applied on the whole group before ``df_query`` narrows the trials.
        raw_df = filterSmallGroups(df, keys, st_col=st_col,
                                   min_trials=min_trials)
        zscored_df = zScoreWithinSubject(raw_df, keys, st_col=st_col,
                                         min_trials=0)
        if df_query is not None:
            raw_df = raw_df.query(df_query)
            zscored_df = zscored_df.query(df_query)
        dispersion = subjectDispersion(raw_df, keys, st_col=st_col,
                                       min_trials=0, measure=measure,
                                       iqr_quantiles=iqr_quantiles)
        return dict(label=label, color=colors[label], zscores=zscored_df[st_col],
                    dispersion=dispersion, n_trials=len(zscored_df))

    groups = []
    for label, session_type in session_types:
        groups.append(_collect(df_users[df_users.session_type == session_type],
                               human_keys, label))
    if animals_df is not None:
        animals_df = animals_df[animals_df.ChoiceCorrect.notnull()]
        groups.append(_collect(animals_df, animal_keys, animals_label))
    return groups


def plotSTDispersion(df_users: pd.DataFrame,
                     animals_df: Optional[pd.DataFrame]=None, *,
                     colors: dict, session_types: Union[list, tuple],
                     animals_label: str="Mice",
                     human_keys=("Name", "session_type"),
                     animal_keys=("Name",),
                     df_query: Optional[str]=None, query_labels: dict={},
                     st_col: str=DEF_ST_COL, min_trials: int=MIN_TRIALS,
                     measure: str=DEF_MEASURE,
                     iqr_quantiles=DEF_IQR_QUANTILES,
                     bw_adjust: float=2, x_right: float=3,
                     save_fp: str="", save_figs: bool=False,
                     fname: Optional[str]=None) -> dict:
    '''Left: within-subject z-scored sampling-time density. Right: dispersion.

    ``measure`` selects the right-panel dispersion statistic: "std" (subject
    sigma, the default) or "iqr" (subject inter-quartile range). Returns the
    ``compareGroups`` result for that comparison.
    '''
    if fname is None:
        fname = f"humans_mice_sampling_time_dispersion_{measure}"
    groups = collectGroups(df_users, animals_df, colors=colors,
                           session_types=session_types,
                           animals_label=animals_label, human_keys=human_keys,
                           animal_keys=animal_keys, df_query=df_query,
                           st_col=st_col, min_trials=min_trials,
                           measure=measure, iqr_quantiles=iqr_quantiles)

    fig, (ax, ax_disp) = plt.subplots(1, 2, figsize=(12, 5))
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax_disp.spines[["top", "right", "left", "bottom"]].set_visible(False)

    x_ticks_labels = []
    samples = {}
    for count_x, group in enumerate(groups, 1):
        label, color = group["label"], group["color"]
        zscores, dispersion = group["zscores"], group["dispersion"]
        n_str = (f" (n={len(dispersion):,} Subjects,\n"
                 f"{group['n_trials']:,} Trials)")
        print(f"{label} - Num of subjects: {len(dispersion):,} - "
              f"Num of trials: {group['n_trials']:,}")
        legend_label = f"{label}\n{n_str}"
        sns.kdeplot(zscores, ax=ax, color=color, label=legend_label,
                    bw_adjust=bw_adjust)
        ax.axvline(zscores.median(), color=color, lw=2, ls="--")

        _dict = ax_disp.violinplot(dispersion, positions=[count_x], widths=0.5,
                                   showextrema=False, showmedians=True)
        for pc in _dict['bodies']:
            pc.set_facecolor(color)
        _dict['cmedians'].set_color(color)
        ax_disp.scatter([count_x]*len(dispersion), dispersion, s=8, c=color,
                        alpha=0.6)
        x_ticks_labels.append(legend_label.replace(" ", "\n", 1))
        samples[legend_label] = dispersion.values

    extra_label = query_labels.get(df_query, "")
    if len(extra_label):
        extra_label = f" {extra_label}"
    ax.set_xlim(right=x_right)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Density")
    ax.set_xlabel("Within-Subject Z-Scored Sampling Time")
    ax.set_title("Sampling-Time distribution\n"
                 "(each subject/exp-type z-scored to its own)" + extra_label)
    ax.legend(loc='upper right', fontsize="xx-small")

    stats_res = compareGroups(samples)

    all_disp = np.concatenate([g["dispersion"].values for g in groups])
    y_step = (np.nanmax(all_disp) - np.nanmin(all_disp))*.18
    max_y = _annotateSignificance(ax_disp, stats_res["posthoc"],
                                  y_start=np.nanmax(all_disp) + y_step*.4,
                                  y_step=y_step)

    ax_disp.set_xticks(range(1, len(groups) + 1))
    ax_disp.set_xticklabels(x_ticks_labels, fontsize="small")
    ax_disp.set_xlabel("Session Type")
    extr_str = "" if df_query is None else (
               "Incorrect-Choice " if "ChoiceCorrect == False" in df_query else
               "Correct-Choice ")
    ax_disp.set_ylabel(f"{extr_str}Sampling Time Dispersion - "
                       f"{dispersionLabel(measure, iqr_quantiles)}")
    ax_disp.set_ylim(0, max_y + y_step*.2)
    ax_disp.set_title(
        f"Sampling-Time dispersion violin-plot{extra_label}\n"
        f"Sgf. test: {stats_res['test_name']} - "
        f"p-val: {stats_res['pvalue']:.3g}\n"
        f"Post-hoc: {stats_res['post_hoc_str']}")
    if save_figs:
        plt.savefig(f"{save_fp}/{fname}{extra_label}.svg", bbox_inches='tight')
    plt.show()
    return stats_res
