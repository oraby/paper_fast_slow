'''Body-centroid rotation angle by strategy (Figures S3J-L).

Backend for the "Plot Centroids degree" section of ``Tracking.ipynb``.

Four mice were filmed over three days each while performing the task, and
SLEAP pose estimates were reduced to one **centroid rotation angle** per video
frame -- the angle between the head-fixation screw and the centroid of the
tracked limb points, measured from straight ahead. The question these panels
answer is whether the fast, typical and slow strategies differ in how the
animal *moves*, which would undercut a purely decision-theoretic reading of
the sampling-time tertiles.

Three panels come out of one function:

- **S3J** ``distance_*.svg`` -- how far the angle travels within a trial
  (``max - min``), stacked by tertile;
- **S3K** ``centroid_rotation_*.svg`` -- polar histogram of each trial's mean
  angle, 5-degree bins;
- **S3L** ``centroid_rotation_preferred_*.svg`` -- the same, after flipping
  each session that leans left so every animal's preferred side points the
  same way.

**Per-session normalisation matters.** Counts are divided by the session's own
trial count before averaging across sessions, so a long session does not
dominate the cohort panel; the error bars are then the SEM *across sessions*.
Note that the divisor is taken after outlier trimming and counts every trial
in the session, not only that tertile's.

Extracted from the notebook unchanged except that ``save_prefix`` is now a
parameter. It used to be read from the notebook's globals, so the function
raised ``NameError`` anywhere else -- see ``docs/repo-audit.md``.
'''
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem

#: Angle column produced by the interpolation step of the notebook.
ANGLE_COL = "track_centroid_rotation_angle"
#: One row per video frame; a trial is many rows.
TRIAL_KEYS = ["Name", "Date", "SessionNum", "TrialNumber"]
SESSION_KEYS = ["Name", "Date", "SessionNum"]

#: Sampling-time tertiles, as labelled everywhere else in the paper.
QUANTILE_NAME = {1: "Fast", 2: "Typical", 3: "Slow"}
QUANTILE_COLOR = {1: "red", 2: "yellow", 3: "orange"}

#: Polar axis limits and bin width, in degrees.
MIN_LIM, MAX_LIM = -70, 70
BIN_STEP = 5
#: Radial limit; the preferred-side panel needs more room after flipping.
POLAR_TOP = 0.125
POLAR_TOP_PREFERRED = 0.175

ALL_SUBJECTS = "All Tracked Subjects"


def trimTrialOutliers(df: pd.DataFrame, outliers_ratio: float,
                      angle_col: str=ANGLE_COL) -> pd.DataFrame:
    '''Drop the tail of each *trial*'s angles, both ends.

    Tracking glitches put occasional frames far outside the animal's actual
    range; trimming per trial removes them without touching the trial-to-trial
    spread the panels are about. ``outliers_ratio=0`` is a no-op by value but
    still rebuilds the frame.
    '''
    def _trim(trial_df):
        angles = trial_df[angle_col]
        low = angles.quantile(outliers_ratio)
        high = angles.quantile(1 - outliers_ratio)
        return trial_df[(angles >= low) & (angles <= high)]

    return df.groupby(TRIAL_KEYS, group_keys=False)[df.columns].apply(
        _trim).reset_index(drop=True)


def normaliseToPreferredSide(df: pd.DataFrame,
                             angle_col: str=ANGLE_COL) -> pd.DataFrame:
    '''Flip whole sessions so every animal's preferred side is positive.

    A session with more negative than positive frames has all its angles
    negated. Without this the cohort polar histogram averages a left-leaning
    animal against a right-leaning one and both disappear.
    '''
    def _flip(session_df):
        negative = (session_df[angle_col] < 0).sum()
        positive = (session_df[angle_col] > 0).sum()
        if negative > positive:
            session_df = session_df.copy()
            session_df[angle_col] = -session_df[angle_col]
        return session_df

    return df.groupby(SESSION_KEYS, group_keys=False)[df.columns].apply(
        _flip).reset_index(drop=True)


def normaliseToChoice(quantile_df: pd.DataFrame,
                      angle_col: str=ANGLE_COL) -> pd.DataFrame:
    '''Sign each angle by whether it agrees with the choice the animal made.

    Not used by any published panel -- all three notebook calls pass
    ``choice_normed=False`` -- but kept because it is the natural companion to
    :func:`normaliseToPreferredSide`.
    '''
    quantile_df = quantile_df.copy()
    agrees = (quantile_df[angle_col] > 0) == (quantile_df.ChoiceLeft == 1)
    magnitude = quantile_df[angle_col].abs()
    quantile_df.loc[agrees, angle_col] = magnitude[agrees]
    quantile_df.loc[~agrees, angle_col] = -magnitude[~agrees]
    return quantile_df


def _sessionTrialCount(df: pd.DataFrame, session_df: pd.DataFrame) -> int:
    '''Trials in the whole session, across every tertile.'''
    return df[df.File == session_df.File.iloc[0]].TrialNumber.nunique()


def distanceTravelledHistogram(quantile_df: pd.DataFrame, df: pd.DataFrame,
                               *, many_subjects: bool,
                               angle_col: str=ANGLE_COL) -> np.ndarray:
    '''Per-session histograms of within-trial angular travel (Figure S3J).

    Returns one row per session. ``many_subjects`` divides each row by the
    session's trial count, turning counts into fractions so sessions of
    different lengths can be averaged.
    '''
    bins = np.arange(0, MAX_LIM + BIN_STEP, BIN_STEP)
    rows = []
    for _session, session_df in quantile_df.groupby(SESSION_KEYS):
        per_trial = session_df.groupby("TrialNumber")[angle_col]
        travelled = abs(per_trial.max() - per_trial.min())
        heights, _edges = np.histogram(travelled, bins=bins)
        if many_subjects:
            heights = heights / _sessionTrialCount(df, session_df)
        rows.append(heights)
    return np.array(rows)


def meanAngleHistogram(quantile_df: pd.DataFrame, df: pd.DataFrame, *,
                       many_subjects: bool,
                       angle_col: str=ANGLE_COL) -> np.ndarray:
    '''Per-session histograms of each trial's mean angle (Figures S3K-L).'''
    bins = np.arange(MIN_LIM, MAX_LIM + BIN_STEP, BIN_STEP)
    rows = []
    for _session, session_df in quantile_df.groupby(SESSION_KEYS):
        trial_means = session_df.groupby("TrialNumber")[angle_col].mean()
        counts, _edges = np.histogram(trial_means, bins=bins)
        if many_subjects:
            counts = counts / _sessionTrialCount(df, session_df)
        rows.append(counts)
    return np.array(rows)


def sessionLabel(session_file: str) -> str:
    '''``day1/MLA-73/...mat`` -> ``day1_MLA-73``, as the filenames use.'''
    path = Path(session_file)
    return f"{path.parent.parent.name}_{path.parent.name}"


def plotCentroids(df: pd.DataFrame, outliers_ratio: float, *,
                  choice_normed: bool=False,
                  preferred_side_normed: bool=False,
                  create_session_plots: bool=False,
                  create_all_subjects_plot: bool=True,
                  angle_col: str=ANGLE_COL,
                  save_prefix: Optional[Union[str, Path]]=None,
                  save_figs: bool=False, verbose: bool=True) -> list:
    '''Figures S3J-L. Returns one result dict per panel drawn.'''
    if save_figs and save_prefix is None:
        raise ValueError("save_figs=True needs a save_prefix")

    results = []
    if create_all_subjects_plot:
        results.append(_drawPanel(
            ALL_SUBJECTS, df, outliers_ratio, many_subjects=True,
            choice_normed=choice_normed,
            preferred_side_normed=preferred_side_normed, angle_col=angle_col,
            save_prefix=save_prefix, save_figs=save_figs, verbose=verbose))

    if create_session_plots:
        for session_file, session_df in df.groupby("File"):
            results.append(_drawPanel(
                sessionLabel(session_file), session_df, outliers_ratio,
                many_subjects=False, choice_normed=choice_normed,
                preferred_side_normed=False, angle_col=angle_col,
                save_prefix=save_prefix, save_figs=save_figs,
                verbose=verbose))
    return results


def _drawPanel(label, df, outliers_ratio, *, many_subjects, choice_normed,
               preferred_side_normed, angle_col, save_prefix, save_figs,
               verbose) -> dict:
    if verbose:
        print(f"Plotting centroids for session {label}")
    df = trimTrialOutliers(df, outliers_ratio, angle_col)
    if many_subjects and preferred_side_normed:
        df = normaliseToPreferredSide(df, angle_col)

    # The travel panel is only drawn for the un-normalised view: flipping or
    # re-signing angles leaves within-trial travel unchanged, so it would be
    # the same figure three times.
    draw_distance = not (choice_normed or preferred_side_normed)
    fig_distance, ax_distance = (plt.subplots(figsize=(10, 4),
                                              layout="constrained")
                                 if draw_distance else (None, None))
    fig_polar, ax_polar = plt.subplots(figsize=(8, 8), layout="constrained",
                                       subplot_kw={"projection": "polar"})

    distance_bins = np.arange(0, MAX_LIM + BIN_STEP, BIN_STEP)
    theta = np.radians(np.arange(MIN_LIM, MAX_LIM, BIN_STEP) + BIN_STEP / 2)
    distance_base = np.zeros(len(distance_bins) - 1)
    polar_base = np.zeros(len(theta))
    per_quantile = {}

    for quantile_idx, quantile_df in df.groupby("quantile_idx"):
        color = QUANTILE_COLOR[quantile_idx]
        if choice_normed:
            quantile_df = normaliseToChoice(quantile_df, angle_col)

        if draw_distance:
            per_session = distanceTravelledHistogram(
                quantile_df, df, many_subjects=many_subjects,
                angle_col=angle_col)
            mean_heights = per_session.mean(axis=0)
            ax_distance.bar(distance_bins[:-1] + BIN_STEP / 2, mean_heights,
                            width=BIN_STEP, bottom=distance_base, color=color)
            if many_subjects:
                ax_distance.errorbar(distance_bins[:-1] + BIN_STEP / 2,
                                     mean_heights + distance_base,
                                     yerr=sem(per_session, axis=0),
                                     linestyle="none", color="black",
                                     zorder=10)
            distance_base = mean_heights + distance_base

        polar_sessions = meanAngleHistogram(quantile_df, df,
                                            many_subjects=many_subjects,
                                            angle_col=angle_col)
        polar_mean = polar_sessions.mean(axis=0)
        ax_polar.bar(theta, polar_mean, width=np.radians(BIN_STEP),
                     bottom=polar_base, align="center", color=color)
        if many_subjects:
            ax_polar.errorbar(theta, polar_mean + polar_base,
                              yerr=sem(polar_sessions, axis=0),
                              linestyle="none", color="black", zorder=10)
        polar_base = polar_mean + polar_base
        per_quantile[QUANTILE_NAME[quantile_idx]] = polar_mean

    fraction_str = " (Fraction)" if many_subjects else ""
    if draw_distance:
        ax_distance.set_xlim(0, MAX_LIM)
        ax_distance.set_xlabel("Centroid Rotation Angle Distance Traveled "
                               "(deg) within Trial")
        ax_distance.set_ylabel(f"Number of Trials{fraction_str}")
        ax_distance.spines[["top", "right", "left"]].set_visible(False)
        ax_distance.set_title("Distance Traveled per Trial Distribution\n"
                              f"Session: {label}")

    ax_polar.set_thetamin(MAX_LIM)
    ax_polar.set_thetamax(MIN_LIM)
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)                       # clockwise
    ax_polar.grid(True)
    normed_title = ((" (Normed to Choice Direction)" if choice_normed else "")
                    + (" (Normed to Preferred Side)" if preferred_side_normed
                       else ""))
    ax_polar.set_title(f"Centroid Rotation Angle Distribution{normed_title}\n"
                       f"Session: {label}")
    ax_polar.set_xlabel(f"Mean Trial Angle (deg) - Bin-Size={BIN_STEP} deg")
    if many_subjects:
        ax_polar.set_ylim(0, POLAR_TOP_PREFERRED if preferred_side_normed
                          else POLAR_TOP)
    ax_polar.set_ylabel(f"Trial Count (deg){fraction_str}")

    if save_figs:
        save_dir = Path(save_prefix)
        save_dir.mkdir(parents=True, exist_ok=True)
        if draw_distance:
            _save(fig_distance, save_dir / f"distance_{label}.svg", verbose)
        suffix = ("_choice_normed" if choice_normed else "") + \
                 ("_preferred" if preferred_side_normed else "")
        _save(fig_polar, save_dir / f"centroid_rotation{suffix}_{label}.svg",
              verbose)

    return dict(label=label, polar=per_quantile, ax_polar=ax_polar,
                ax_distance=ax_distance, fig_polar=fig_polar,
                fig_distance=fig_distance)


def _save(fig, path: Path, verbose: bool) -> None:
    if verbose:
        print(f"Saving figure to {path}")
    fig.savefig(path, dpi=300)
