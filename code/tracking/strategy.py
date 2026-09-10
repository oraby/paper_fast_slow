'''Do the strategies differ in posture? (Figure S3M)

Backend for the last section of ``Tracking.ipynb``.

Figures S3J-L show the centroid rotation angle *distribution* per tertile.
This panel asks the sharper question, one animal at a time: within a single
mouse, does the spread of postures differ between its fast, typical and slow
trials? Each trial contributes its mean angle; each animal gets a **mode**
angle (5-degree bins) treated as that animal's neutral posture; and each trial
is scored by how far it sits from that neutral. A Kruskal-Wallis across the
three tertiles then runs **per animal**, Holm-corrected across animals.

The published answer is **0 of 4 mice significant** -- posture does not track
strategy, which is what makes the sampling-time tertiles readable as a
decision variable rather than a movement artefact.

Two properties of the procedure are worth stating because they are not
obvious from the plot:

- the omnibus test only runs for an animal when **all three** of its tertiles
  fail Shapiro-Wilk. ``assertAllNonNormal`` then checks that this held
  everywhere, because a normal group would mean the Kruskal-Wallis was the
  wrong test and that animal silently contributed no p-value;
- an animal needs at least :data:`MIN_TRIALS_PER_QUANTILE` trials in a tertile
  for that tertile to count, and all three tertiles present to be tested at
  all.

Extracted from the notebook unchanged except that ``save_prefix`` is a
parameter (it was read from the notebook's globals) and the scatter jitter can
be seeded. See ``docs/repo-audit.md``.
'''
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .centroids import (ANGLE_COL, QUANTILE_COLOR, QUANTILE_NAME,
                        trimTrialOutliers)

#: A tertile with fewer trials than this is ignored for that animal.
MIN_TRIALS_PER_QUANTILE = 10
#: Bin width used to find each animal's modal (neutral) posture, in degrees.
MODE_BIN_STEP = 5
#: Shapiro-Wilk threshold above which a group counts as normal.
NORMALITY_ALPHA = 0.05
DEF_FIG_NAME = "strategy_dist_sgf"

TRIAL_MEAN_KEYS = ["Name", "Date", "File", "SessionNum", "quantile_idx",
                   "TrialNumber"]


def trialMeanAngles(df: pd.DataFrame, outliers_ratio: float=0.,
                    angle_col: str=ANGLE_COL) -> pd.DataFrame:
    '''One row per trial, carrying that trial's mean rotation angle.

    ``outliers_ratio=0`` -- what the published panel uses -- skips trimming
    entirely rather than trimming nothing, which is the same by value but
    much faster on 76k frames.
    '''
    if outliers_ratio > 0:
        df = trimTrialOutliers(df, outliers_ratio, angle_col)
    return df.groupby(TRIAL_MEAN_KEYS)[angle_col].mean().reset_index()


def distanceFromMode(trials: pd.DataFrame, angle_col: str=ANGLE_COL,
                     bin_step: int=MODE_BIN_STEP) -> pd.DataFrame:
    '''Score each trial by |angle - the animal's modal angle|.

    The mode is taken per **animal**, not per session, so an animal that shifts
    posture between days is measured against one neutral throughout.
    '''
    trials = trials.copy()
    trials["binned_angle"] = (trials[angle_col] / bin_step).round() * bin_step
    per_animal = []
    for _name, animal_df in trials.groupby("Name"):
        animal_df = animal_df.copy()
        base_angle = animal_df["binned_angle"].mode().iloc[0]
        animal_df["base_angle"] = base_angle
        animal_df["abs_angle_from_base"] = (
            animal_df[angle_col] - base_angle).abs()
        per_animal.append(animal_df)
    return pd.concat(per_animal, ignore_index=True)


def animalQuantileGroups(trials: pd.DataFrame,
                         min_trials: int=MIN_TRIALS_PER_QUANTILE) -> dict:
    '''``{animal: {quantile_idx: distances}}``, keeping only complete animals.

    An animal missing any tertile -- or with too few trials in one -- is
    dropped, because the omnibus test compares all three.
    '''
    groups: dict = {}
    for quantile_idx, quantile_df in trials.groupby("quantile_idx"):
        for name, animal_df in quantile_df.groupby("Name"):
            if len(animal_df) < min_trials:
                continue
            groups.setdefault(name, {})[quantile_idx] = \
                animal_df["abs_angle_from_base"].values
    return {name: per_quantile for name, per_quantile in groups.items()
            if len(per_quantile) == len(QUANTILE_NAME)}


def compareStrategies(groups: dict, alpha: float=NORMALITY_ALPHA) -> dict:
    '''Per-animal Kruskal-Wallis, Holm-corrected across animals.

    Returns the raw and corrected p-values, which animals were rejected, and
    ``all_non_normal`` -- see :func:`assertAllNonNormal`.
    '''
    all_non_normal = True
    raw_pvalues = {}
    normality = {}
    for name, per_quantile in groups.items():
        animal_non_normal = True
        for quantile_idx, distances in per_quantile.items():
            _statistic, pvalue = stats.shapiro(distances)
            is_normal = pvalue > alpha
            normality[(name, quantile_idx)] = is_normal
            if is_normal:
                all_non_normal = False
                animal_non_normal = False
        if animal_non_normal:
            ordered = [per_quantile[idx] for idx in sorted(per_quantile)]
            _statistic, pvalue = stats.kruskal(*ordered)
            raw_pvalues[name] = pvalue

    names = list(raw_pvalues)
    if names:
        reject, corrected, _, _ = multipletests(list(raw_pvalues.values()),
                                                method="holm")
    else:
        reject, corrected = np.array([], dtype=bool), np.array([])
    return dict(names=names, raw=raw_pvalues,
                corrected=dict(zip(names, corrected)),
                rejected={name: bool(flag)
                          for name, flag in zip(names, reject)},
                n_rejected=int(reject.sum()), n_tested=len(names),
                all_non_normal=all_non_normal, normality=normality)


def assertAllNonNormal(result: dict) -> None:
    '''Fail loudly if any group was normal, as the notebook does.

    A normal group means that animal never reached the Kruskal-Wallis and so
    contributed no p-value at all -- the panel would understate how many
    animals were tested rather than showing a gap.
    '''
    assert result["all_non_normal"], (
        "Some distributions are normal, check Shapiro-Wilk results: "
        + ", ".join(f"{name}/{QUANTILE_NAME[idx]}"
                    for (name, idx), is_normal in result["normality"].items()
                    if is_normal))


def plotStrategyComparison(df: pd.DataFrame, outliers_ratio: float=0., *,
                           angle_col: str=ANGLE_COL,
                           min_trials: int=MIN_TRIALS_PER_QUANTILE,
                           seed: Optional[int]=None,
                           ax=None,
                           save_prefix: Optional[Union[str, Path]]=None,
                           save_figs: bool=False,
                           fig_name: str=DEF_FIG_NAME,
                           verbose: bool=True) -> dict:
    '''Figure S3M. Returns the :func:`compareStrategies` result.

    ``seed`` fixes the horizontal jitter of the scatter, which is cosmetic but
    otherwise makes the figure differ between runs.
    '''
    if save_figs and save_prefix is None:
        raise ValueError("save_figs=True needs a save_prefix")

    trials = distanceFromMode(trialMeanAngles(df, outliers_ratio, angle_col),
                              angle_col)
    groups = animalQuantileGroups(trials, min_trials)
    if verbose:
        print(f"Animals with all three tertiles: {len(groups)}")
    result = compareStrategies(groups)

    if ax is None:
        _fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
    fig = ax.get_figure()
    rng = np.random.default_rng(seed) if seed is not None else np.random

    x_ticks, x_labels, animal_x = [], [], {}
    for position, (name, per_quantile) in enumerate(groups.items()):
        for quantile_idx, distances in per_quantile.items():
            jitter = (rng.random(len(distances)) - 0.5) * 0.8 \
                if seed is not None else \
                (np.random.rand(len(distances)) - 0.5) * 0.8
            ax.scatter(jitter + position * 5 + quantile_idx - 1, distances,
                       color=QUANTILE_COLOR[quantile_idx], s=10, alpha=0.6)
        centre = position * 5 + 1
        animal_x[name] = centre
        x_ticks.append(centre)
        x_labels.append(f"Mouse#{position + 1}")

    for name in result["names"]:
        if verbose:
            print(f"Session {name} corrected p-value: "
                  f"{result['corrected'][name]:.3f}, "
                  f"reject null: {result['rejected'][name]}")
        if not result["rejected"][name]:
            continue
        top = max(distances.max() for distances in groups[name].values()) + 2
        centre = animal_x[name]
        ax.plot([centre - 1.5, centre - 1.5, centre + 1.5, centre + 1.5],
                [top, top + 0.5, top + 0.5, top], color="k")
        ax.text(centre, top + 2, "*", color="k", ha="center", va="bottom",
                fontsize=16)

    assertAllNonNormal(result)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_title("Strategy Distribution Comparison per Mouse\n"
                 f"Kruskal-Wallis test: {result['n_rejected']}/"
                 f"{result['n_tested']} mice with significant differences")
    ax.set_xlabel("Mice")
    ax.set_ylabel("Distance from mode Rotation Angle (deg)")
    ax.spines[["top", "right"]].set_visible(False)

    if save_figs:
        save_dir = Path(save_prefix)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{fig_name}.svg"
        if verbose:
            print(f"Saving figure to {save_path}")
        fig.savefig(save_path, dpi=300)
    return result
