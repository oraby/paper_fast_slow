'''Tests for the win/lose update figures (Figures 1I-right and S3G).

The property everything rests on is the baseline subtraction: because the
rewarded side repeats with structure of its own, a raw stay probability is not
interpretable, and an agent that simply tracked the rewarded side must score
zero update. That is pinned first.

Figure 1I-right then reduces each animal to the mean of its *absolute* Win and
Lose updates, so an animal playing win-stay and one playing win-switch equally
strongly score the same; that is pinned too, because taking the absolute value
is easy to lose in a refactor and silently halves the group mean.

For Figure S3G the two excluded animals are derived from the Methods' rule
rather than named, so the tests check the rule, including the boundary (it is
strictly greater than the percentile).
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ..stayswitchupdate import (FAST, GROUP_EVERY, LOSE_COLOR, SLOW, WIN_COLOR,
                                calcWinLoseUpdates, compareQuantileUpdates,
                                outlierFraction, outlierSubjects,
                                plotOutlierFraction, plotQuantileUpdate,
                                plotUpdateOverSamplingTime, quantileUpdates,
                                significanceLabel, updateCurve)


def _trials(n, *, stay, baseline, prev_outcome_count, name="M1",
            quantile_idx=FAST):
    '''n trials with fixed Stay / StayBaseline / previous-outcome sign.'''
    return pd.DataFrame({
        "Name": name, "quantile_idx": quantile_idx,
        "Stay": np.full(n, float(stay)),
        "StayBaseline": np.full(n, float(baseline)),
        "PrevOutcomeCount": np.full(n, float(prev_outcome_count)),
    })


# --------------------------------------------------------------------------
# The baseline subtraction
# --------------------------------------------------------------------------

def test_tracking_the_rewarded_side_scores_zero_update():
    '''The whole point of StayBaseline: sequence structure is removed.'''
    df = pd.concat([_trials(50, stay=1, baseline=1, prev_outcome_count=1),
                    _trials(50, stay=0, baseline=0, prev_outcome_count=1)])
    win_update, n_win, _, _ = calcWinLoseUpdates(df)
    assert n_win == 100
    assert win_update == pytest.approx(0.0)


def test_always_staying_scores_the_excess_over_baseline():
    '''Staying on every trial when the baseline stays on 40% of them.'''
    df = pd.concat([_trials(40, stay=1, baseline=1, prev_outcome_count=1),
                    _trials(60, stay=1, baseline=0, prev_outcome_count=1)])
    win_update, _, _, _ = calcWinLoseUpdates(df)
    assert win_update == pytest.approx(60.0)


def test_switching_more_than_baseline_is_negative():
    df = _trials(50, stay=0, baseline=1, prev_outcome_count=1)
    win_update, _, _, _ = calcWinLoseUpdates(df)
    assert win_update == pytest.approx(-100.0)


def test_win_and_lose_are_split_by_the_sign_of_the_streak():
    df = pd.concat([_trials(30, stay=1, baseline=0, prev_outcome_count=2),
                    _trials(20, stay=0, baseline=1, prev_outcome_count=-3)])
    win_update, n_win, lose_update, n_lose = calcWinLoseUpdates(df)
    assert (n_win, n_lose) == (30, 20)
    assert win_update == pytest.approx(100.0)
    assert lose_update == pytest.approx(-100.0)


def test_streak_count_of_zero_belongs_to_neither_set():
    '''PrevOutcomeCount == 0 marks a trial with no usable previous outcome.'''
    df = _trials(25, stay=1, baseline=0, prev_outcome_count=0)
    _, n_win, _, n_lose = calcWinLoseUpdates(df)
    assert (n_win, n_lose) == (0, 0)


def test_empty_set_scores_zero_not_nan():
    '''Matches the published figures, which never show a gap.'''
    df = _trials(10, stay=1, baseline=0, prev_outcome_count=1)
    _, _, lose_update, n_lose = calcWinLoseUpdates(df)
    assert n_lose == 0
    assert lose_update == 0


def test_column_prefix_selects_a_parallel_set_of_columns():
    '''The model code reuses this on simulated columns.'''
    df = _trials(20, stay=1, baseline=0, prev_outcome_count=1)
    df = df.rename(columns={c: f"Sim{c}" for c in
                            ("Stay", "StayBaseline", "PrevOutcomeCount")})
    win_update, n_win, _, _ = calcWinLoseUpdates(df, col_prefix="Sim")
    assert (n_win, win_update) == (20, pytest.approx(100.0))


# --------------------------------------------------------------------------
# Figure 1I-right
# --------------------------------------------------------------------------

def _twoStrategyAnimal(name, fast_win, fast_lose, slow_win, slow_lose, n=40):
    '''An animal whose Fast and Slow updates are set by construction.

    ``stay=1, baseline=0`` gives +100; ``stay=0, baseline=1`` gives -100, so a
    requested update u is produced by splitting the trials in proportion.
    '''
    parts = []
    for idx, (win, lose) in ((FAST, (fast_win, fast_lose)),
                             (SLOW, (slow_win, slow_lose))):
        for count, update in ((1, win), (-1, lose)):
            n_stay = int(round(n * (update + 100) / 200))
            parts.append(_trials(n_stay, stay=1, baseline=0,
                                 prev_outcome_count=count, name=name,
                                 quantile_idx=idx))
            parts.append(_trials(n - n_stay, stay=0, baseline=1,
                                 prev_outcome_count=count, name=name,
                                 quantile_idx=idx))
    return pd.concat(parts, ignore_index=True)


def test_aggregate_update_is_the_mean_of_the_two_absolute_updates():
    df = _twoStrategyAnimal("M1", fast_win=50, fast_lose=-30,
                            slow_win=20, slow_lose=-10)
    row = quantileUpdates(df).iloc[0]
    assert row.FastUpdate == pytest.approx((50 + 30) / 2)
    assert row.SlowUpdate == pytest.approx((20 + 10) / 2)


def test_win_stay_and_win_switch_of_equal_strength_score_the_same():
    '''Why the absolute value is taken: strength, not direction.'''
    stayer = _twoStrategyAnimal("A", 60, -60, 20, -20)
    switcher = _twoStrategyAnimal("B", -60, 60, -20, 20)
    both = quantileUpdates(pd.concat([stayer, switcher], ignore_index=True))
    assert both.FastUpdate.nunique() == 1
    assert both.SlowUpdate.nunique() == 1


def test_comparison_is_paired_across_animals():
    df = pd.concat([_twoStrategyAnimal(f"M{i}", 60, -60, 20, -20)
                    for i in range(6)], ignore_index=True)
    result = compareQuantileUpdates(df)
    assert len(result.per_subject) == 6
    assert result.fast_mean == pytest.approx(60.0)
    assert result.slow_mean == pytest.approx(20.0)
    assert result.pvalue < 0.05


def test_significance_label_thresholds():
    assert significanceLabel(0.0001) == "***"
    assert significanceLabel(0.005) == "**"
    assert significanceLabel(0.04) == "*"
    assert significanceLabel(0.2) == "ns"


def test_quantile_panel_draws_a_line_per_animal():
    df = pd.concat([_twoStrategyAnimal(f"M{i}", 60, -60, 20, -20)
                    for i in range(5)], ignore_index=True)
    fig, result = plotQuantileUpdate(df)
    ax = fig.axes[0]
    paired = [ln for ln in ax.lines if len(ln.get_xdata()) == 2]
    assert len(paired) == 5
    assert result.per_subject.Name.nunique() == 5
    plt.close(fig)


def test_quantile_panel_saves_under_its_subdirectory(tmp_path):
    df = pd.concat([_twoStrategyAnimal(f"M{i}", 60, -60, 20, -20)
                    for i in range(4)], ignore_index=True)
    fig, _ = plotQuantileUpdate(df, save_prefix=tmp_path, save_figs=True,
                                save_postfix="Mice")
    assert (tmp_path / "QuantileWinLoseUpdate" / "_total_Mice.svg").exists()
    plt.close(fig)


def test_quantile_panel_refuses_to_save_without_a_prefix():
    df = _twoStrategyAnimal("M1", 60, -60, 20, -20)
    with pytest.raises(ValueError, match="save_prefix"):
        plotQuantileUpdate(df, save_figs=True)


# --------------------------------------------------------------------------
# Figure S3G -- the derived exclusion
# --------------------------------------------------------------------------

def _samplingTimes(name, values):
    n = len(values)
    return pd.DataFrame({
        "Name": name, "calcStimulusTime": np.asarray(values, dtype=float),
        "quantile_idx": np.resize([FAST, SLOW], n),
        "Stay": np.resize([1.0, 0.0], n),
        "StayBaseline": np.zeros(n),
        "PrevOutcomeCount": np.resize([1.0, -1.0], n),
        "PrevChoiceCorrect": np.resize([True, False], n),
    })


def test_outlier_fraction_counts_trials_below_the_z_cutoff():
    rng = np.random.default_rng(0)
    clean = _samplingTimes("Clean", rng.normal(1.0, 0.2, 500))
    # A long left tail pushes many trials below z = -1.
    skewed = _samplingTimes("Skewed",
                            np.concatenate([rng.normal(1.0, 0.05, 400),
                                            rng.normal(0.35, 0.02, 100)]))
    fractions = outlierFraction(pd.concat([clean, skewed], ignore_index=True))
    assert fractions["Skewed"] > fractions["Clean"]


def test_outlier_subjects_uses_a_strict_greater_than():
    '''An animal exactly at the percentile is kept, not dropped.'''
    rng = np.random.default_rng(1)
    df = pd.concat([_samplingTimes(f"M{i}", rng.normal(1.0, 0.2, 300))
                    for i in range(10)], ignore_index=True)
    fractions = outlierFraction(df)
    threshold = np.percentile(fractions, 90)
    dropped = outlierSubjects(df)
    assert all(fractions[name] > threshold for name in dropped)
    assert not any(fractions[name] <= threshold for name in dropped)


def test_outlier_subjects_returns_a_sorted_list():
    rng = np.random.default_rng(2)
    df = pd.concat([_samplingTimes(f"M{i}", rng.normal(1.0, 0.2, 300))
                    for i in range(12)], ignore_index=True)
    dropped = outlierSubjects(df)
    assert dropped == sorted(dropped)


# --------------------------------------------------------------------------
# Figure S3G -- the curve
# --------------------------------------------------------------------------

def _curveDf(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    st = rng.normal(0.0, 1.0, n)
    return pd.DataFrame({
        "Name": "M1", "quantile_idx": np.resize([FAST, SLOW], n),
        # plotUpdateOverSamplingTime z-scores within animal itself, so the
        # raw seconds column is the one that has to be present.
        "calcStimulusTime": 1.0 + 0.3 * st,
        "transformedCalcStimulusTime": st,
        "Stay": rng.integers(0, 2, n).astype(float),
        "StayBaseline": np.zeros(n),
        "PrevOutcomeCount": np.resize([1.0, -1.0], n),
        "PrevChoiceCorrect": np.resize([True, False], n),
    })


def test_curve_has_one_point_per_block_of_trials():
    df = _curveDf(n=4000)
    x, y, _ = updateCurve(df)
    assert len(x) == len(y) == int(np.ceil(4000 / GROUP_EVERY))


def test_curve_points_sit_at_the_start_of_their_block():
    '''x is the block's smallest sampling time, so the curve is increasing.'''
    x, _, _ = updateCurve(_curveDf())
    assert list(x) == sorted(x)


def test_curve_drops_trials_without_a_previous_choice():
    df = _curveDf(n=1600)
    df.loc[df.index[:400], "Stay"] = np.nan
    x, _, _ = updateCurve(df)
    assert len(x) == int(np.ceil(1200 / GROUP_EVERY))


def test_curve_average_matches_the_pooled_update():
    df = _curveDf(n=1600)
    _, _, average = updateCurve(df)
    kept = df[df.Stay.notnull()]
    expected = 100 * (kept.Stay.sum() - kept.StayBaseline.sum()) / len(kept)
    assert average == pytest.approx(expected)


def test_sampling_time_panel_uses_the_documented_colours():
    '''The manuscript legend says green/red; the figure is blue/magenta.'''
    df = _curveDf(n=4000)
    fig = plotUpdateOverSamplingTime(df, exclude=[])
    colors = {ln.get_color() for ln in fig.axes[0].lines}
    assert {WIN_COLOR, LOSE_COLOR} <= colors
    plt.close(fig)


def test_sampling_time_panel_saves_under_the_prefix(tmp_path):
    fig = plotUpdateOverSamplingTime(_curveDf(n=4000), exclude=[],
                                     title="T", save_prefix=tmp_path,
                                     save_figs=True)
    assert (tmp_path / "win_lose_stay_switch_T.svg").exists()
    plt.close(fig)


def test_outlier_panel_writes_both_formats(tmp_path):
    rng = np.random.default_rng(4)
    df = pd.concat([_samplingTimes(f"M{i}", rng.normal(1.0, 0.2, 300))
                    for i in range(10)], ignore_index=True)
    fig = plotOutlierFraction(df, save_prefix=tmp_path, save_figs=True)
    for ext in ("svg", "png"):
        assert (tmp_path / f"outlier_percentage_by_subject.{ext}").exists()
    plt.close(fig)


def test_panels_refuse_to_save_without_a_prefix():
    df = _curveDf(n=1600)
    with pytest.raises(ValueError, match="save_prefix"):
        plotUpdateOverSamplingTime(df, exclude=[], save_figs=True)
    with pytest.raises(ValueError, match="save_prefix"):
        plotOutlierFraction(_samplingTimes("M1", np.linspace(0.5, 2, 100)),
                            save_figs=True)
