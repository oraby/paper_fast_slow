'''Tests for the centroid rotation-angle panels (Figures S3J-L).

The published panels were reproduced exactly while extracting this code: bar
geometry identical to the notebook cell on all 76,311 tracked frames, and the
three saved SVGs text-identical to the committed ones.

What the tests guard is the part that is easy to get subtly wrong later --
the per-session normalisation. Counts are divided by the session's own trial
count *before* averaging across sessions, and that divisor counts every trial
in the session rather than only the tertile being drawn. Get either wrong and
the cohort panel silently becomes a long-session panel.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.container import ErrorbarContainer

from ..centroids import (ALL_SUBJECTS, ANGLE_COL, BIN_STEP, MAX_LIM,
                         distanceTravelledHistogram, meanAngleHistogram,
                         normaliseToChoice, normaliseToPreferredSide,
                         plotCentroids, sessionLabel, trimTrialOutliers)

FAST, TYPICAL, SLOW = 1, 2, 3


def trial(name, day, trial_number, angles, *, quantile=FAST, choice_left=1):
    angles = np.asarray(angles, dtype=float)
    return pd.DataFrame({
        "Name": name, "Date": day, "SessionNum": 1,
        "File": f"{day}/{name}/{name}_session.mat",
        "TrialNumber": trial_number, "quantile_idx": quantile,
        "ChoiceLeft": choice_left, ANGLE_COL: angles})


def session(name="MLA-73", day="day1", n_trials=9, spread=10., centre=0.,
            n_frames=12, seed=0):
    '''One session with trials spread evenly across the three tertiles.'''
    rng = np.random.default_rng(seed)
    return pd.concat(
        [trial(name, day, index,
               rng.normal(centre, spread, n_frames),
               quantile=(index % 3) + 1)
         for index in range(n_trials)], ignore_index=True)


def cohort(n_animals=4, n_days=3, **kwargs):
    return pd.concat([session(f"MLA-{73 + animal}", f"day{day + 1}",
                              seed=animal * 10 + day, **kwargs)
                      for animal in range(n_animals)
                      for day in range(n_days)], ignore_index=True)


# --------------------------------------------------------------------------
# Outlier trimming
# --------------------------------------------------------------------------

def test_trimming_is_per_trial_not_across_the_session():
    '''A trial sitting far from the others must survive intact.

    Trimming the pooled session would delete the whole of the second trial,
    which sits far from the first. Trimming per trial keeps it and removes
    only each trial's own extremes.
    '''
    df = pd.concat([trial("A", "day1", 0, [0., 1, 2, 3, 100]),
                    trial("A", "day1", 1, [50., 51, 52, 53, 54])],
                   ignore_index=True)
    trimmed = trimTrialOutliers(df, 0.1)
    assert set(trimmed.TrialNumber) == {0, 1}
    assert 100. not in trimmed[ANGLE_COL].values
    # The far-off trial keeps its middle, rather than being trimmed away.
    assert trimmed[trimmed.TrialNumber == 1][ANGLE_COL].tolist() == [51., 52., 53.]


def test_a_zero_ratio_keeps_every_frame():
    df = session()
    assert len(trimTrialOutliers(df, 0.)) == len(df)


def test_trimming_removes_both_tails():
    df = trial("A", "day1", 0, list(range(101)))
    trimmed = trimTrialOutliers(df, 0.1)
    assert trimmed[ANGLE_COL].min() == pytest.approx(10.)
    assert trimmed[ANGLE_COL].max() == pytest.approx(90.)


# --------------------------------------------------------------------------
# Preferred-side normalisation (Figure S3L)
# --------------------------------------------------------------------------

def test_a_left_leaning_session_is_flipped():
    df = trial("A", "day1", 0, [-30., -25, -20, 5])
    flipped = normaliseToPreferredSide(df)
    assert (flipped[ANGLE_COL] > 0).sum() == 3


def test_a_right_leaning_session_is_left_alone():
    df = trial("A", "day1", 0, [30., 25, 20, -5])
    pd.testing.assert_series_equal(
        normaliseToPreferredSide(df)[ANGLE_COL].reset_index(drop=True),
        df[ANGLE_COL].reset_index(drop=True))


def test_sessions_are_flipped_independently():
    '''Two animals leaning opposite ways must end up pointing the same way.'''
    df = pd.concat([trial("A", "day1", 0, [-30., -25, -20]),
                    trial("B", "day1", 0, [30., 25, 20])], ignore_index=True)
    flipped = normaliseToPreferredSide(df)
    assert (flipped[ANGLE_COL] > 0).all()


def test_flipping_preserves_the_frame_size_and_columns():
    df = cohort(n_animals=2, n_days=1)
    flipped = normaliseToPreferredSide(df)
    assert len(flipped) == len(df)
    assert set(flipped.columns) == set(df.columns)


# --------------------------------------------------------------------------
# Choice normalisation
# --------------------------------------------------------------------------

def test_angles_agreeing_with_the_choice_become_positive():
    df = pd.concat([trial("A", "day1", 0, [20.], choice_left=1),
                    trial("A", "day1", 1, [-20.], choice_left=0)],
                   ignore_index=True)
    assert (normaliseToChoice(df)[ANGLE_COL] > 0).all()


def test_angles_disagreeing_with_the_choice_become_negative():
    df = pd.concat([trial("A", "day1", 0, [20.], choice_left=0),
                    trial("A", "day1", 1, [-20.], choice_left=1)],
                   ignore_index=True)
    assert (normaliseToChoice(df)[ANGLE_COL] < 0).all()


# --------------------------------------------------------------------------
# The two histograms
# --------------------------------------------------------------------------

def test_travel_is_the_within_trial_range():
    df = pd.concat([trial("A", "day1", 0, [0., 10, 20]),      # travels 20
                    trial("A", "day1", 1, [5., 6, 7])],       # travels 2
                   ignore_index=True)
    heights = distanceTravelledHistogram(df, df, many_subjects=False)
    assert heights.shape == (1, MAX_LIM // BIN_STEP)
    assert heights.sum() == 2
    assert heights[0][0] == 1                       # 2 degrees -> bin [0, 5)
    assert heights[0][20 // BIN_STEP] == 1          # 20 degrees -> bin [20, 25)


def test_one_row_comes_back_per_session():
    df = cohort(n_animals=2, n_days=3)
    fast = df[df.quantile_idx == FAST]
    assert distanceTravelledHistogram(fast, df, many_subjects=True).shape[0] == 6
    assert meanAngleHistogram(fast, df, many_subjects=True).shape[0] == 6


def test_counts_become_fractions_of_the_whole_session():
    '''The divisor is every trial in the session, not the tertile's own.'''
    df = session(n_trials=9)                       # 3 trials per tertile of 9
    fast = df[df.quantile_idx == FAST]
    counts = distanceTravelledHistogram(fast, df, many_subjects=False)
    fractions = distanceTravelledHistogram(fast, df, many_subjects=True)
    assert counts.sum() == 3
    assert fractions.sum() == pytest.approx(3 / 9)


def test_a_long_session_does_not_outweigh_a_short_one():
    '''The reason for normalising before averaging across sessions.'''
    short = session("A", "day1", n_trials=3, seed=1)
    long_session = session("B", "day1", n_trials=30, seed=2)
    df = pd.concat([short, long_session], ignore_index=True)
    fast = df[df.quantile_idx == FAST]
    rows = distanceTravelledHistogram(fast, df, many_subjects=True)
    assert rows.shape[0] == 2
    assert rows[0].sum() == pytest.approx(rows[1].sum(), abs=.15)


def test_the_polar_histogram_bins_trial_means_over_the_full_sweep():
    df = trial("A", "day1", 0, [10., 12, 14])       # mean 12 -> the 10-15 bin
    counts = meanAngleHistogram(df, df, many_subjects=False)
    assert counts.sum() == 1
    edges = np.arange(-70, 70 + BIN_STEP, BIN_STEP)
    assert counts[0][np.searchsorted(edges, 12.) - 1] == 1


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

def test_a_session_is_labelled_by_day_and_animal():
    assert sessionLabel("day1/MLA-73/MLA-73_Mouse2AFC_Oct22_2025_Session1.mat") \
        == "day1_MLA-73"


# --------------------------------------------------------------------------
# The panels
# --------------------------------------------------------------------------

def test_the_cohort_panel_draws_both_figures():
    result, = plotCentroids(cohort(), outliers_ratio=0.1, verbose=False)
    assert result["label"] == ALL_SUBJECTS
    assert result["ax_distance"] is not None
    assert set(result["polar"]) == {"Fast", "Typical", "Slow"}
    plt.close("all")


def test_the_preferred_side_panel_drops_the_travel_figure():
    '''Flipping cannot change within-trial travel, so it would be a repeat.'''
    result, = plotCentroids(cohort(), outliers_ratio=0.1,
                            preferred_side_normed=True, verbose=False)
    assert result["ax_distance"] is None
    plt.close("all")


def test_the_tertiles_are_stacked_not_overlaid():
    result, = plotCentroids(cohort(), outliers_ratio=0.1, verbose=False)
    bottoms = {round(patch.get_y(), 6) for patch in result["ax_polar"].patches}
    assert bottoms != {0.0}
    plt.close("all")


def test_one_panel_is_drawn_per_session_when_asked():
    results = plotCentroids(cohort(n_animals=2, n_days=2), outliers_ratio=0.1,
                            create_all_subjects_plot=False,
                            create_session_plots=True, verbose=False)
    assert len(results) == 4
    assert sorted(r["label"] for r in results) == [
        "day1_MLA-73", "day1_MLA-74", "day2_MLA-73", "day2_MLA-74"]
    plt.close("all")


def test_only_the_cohort_panel_carries_error_bars():
    '''There is nothing to take a SEM across within one session.'''
    cohort_result, = plotCentroids(cohort(), outliers_ratio=0.1, verbose=False)
    session_result, = plotCentroids(session(), outliers_ratio=0.1,
                                    create_all_subjects_plot=False,
                                    create_session_plots=True, verbose=False)
    errorbars = lambda ax: [c for c in ax.containers
                            if isinstance(c, ErrorbarContainer)]
    assert errorbars(cohort_result["ax_polar"])
    assert not errorbars(session_result["ax_polar"])
    plt.close("all")


def test_the_panels_save_under_the_prefix(tmp_path):
    plotCentroids(cohort(), outliers_ratio=0.1, save_prefix=tmp_path,
                  save_figs=True, verbose=False)
    assert (tmp_path / f"distance_{ALL_SUBJECTS}.svg").exists()
    assert (tmp_path / f"centroid_rotation_{ALL_SUBJECTS}.svg").exists()
    plt.close("all")


def test_the_preferred_side_panel_gets_its_own_filename(tmp_path):
    plotCentroids(cohort(), outliers_ratio=0.1, preferred_side_normed=True,
                  save_prefix=tmp_path, save_figs=True, verbose=False)
    assert (tmp_path / f"centroid_rotation_preferred_{ALL_SUBJECTS}.svg"
            ).exists()
    plt.close("all")


def test_saving_without_a_prefix_is_refused():
    '''It used to read ``save_prefix`` from the notebook's globals.'''
    with pytest.raises(ValueError, match="save_prefix"):
        plotCentroids(cohort(), outliers_ratio=0.1, save_figs=True,
                      verbose=False)
    plt.close("all")


def test_the_panel_raises_no_pandas_deprecation_warning():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        plotCentroids(cohort(n_animals=2, n_days=1), outliers_ratio=0.1,
                      preferred_side_normed=True, verbose=False)
    plt.close("all")
