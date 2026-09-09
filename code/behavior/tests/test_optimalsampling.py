'''Tests for the reward-optimal sampling-time figure (Figure 2B).

These pin the Methods equations (1)-(3) and the aggregation rules around them:
performance is pinned to chance at t=0 and saturates below 100%, expected trial
duration mixes the correct and incorrect overheads by the predicted accuracy,
and the optimum is interior -- sampling longer buys accuracy but costs trials.

The aggregations are nested rather than flat (per session, then per animal,
then across animals), which matters whenever sessions differ in length, so that
is pinned too.

``test_population_panel_uses_each_animals_own_sd`` is a regression test. The
inline version drew every animal's error bar with a stale loop variable, so all
17 bars in the published figure carry one animal's SD (0.621 s) instead of
their own (which range 0.347-0.924 s).
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ..optimalsampling import (MIN_TRIALS, OverallMetrics, avgExtraTime,
                               avgPerf, buildPerfCurve, collectMetrics,
                               defaultBins, fitPerfCurve,
                               optimalSamplingTime, perfModel,
                               plotAlignedPopulation, plotSubjectRewardCurve,
                               rewardCurve)


def _sessionRows(name, date, n, seed, st_mean=1.0, st_sd=0.4, perf=0.75,
                 iti=3.0):
    '''One session of trials with a controlled sampling-time distribution.'''
    rng = np.random.default_rng(seed)
    st = np.clip(rng.normal(st_mean, st_sd, n), 0.31, 4.9)
    correct = rng.random(n) < perf
    # Trial start times advance by the sampling time plus a fixed overhead, so
    # the derived TrialDuration - calcStimulusTime is exactly ``iti``.
    starts = np.cumsum(np.concatenate([[0.0], st[:-1] + iti]))
    return pd.DataFrame({
        "Name": name, "Date": date, "SessionNum": 1,
        "TrialNumber": np.arange(1, n + 1),
        "TrialStartSysTime": starts,
        "calcStimulusTime": st,
        "ChoiceCorrect": correct,
        "DVstr": rng.choice(["Easy", "Med", "Hard"], n),
    })


def _animalDf(name, seed, n_per_session=700, sessions=2, **kwargs):
    return pd.concat([_sessionRows(name, f"2024-01-{s + 1:02d}",
                                   n_per_session, seed + s, **kwargs)
                      for s in range(sessions)], ignore_index=True)


# --------------------------------------------------------------------------
# Methods eq. (1) -- the performance curve
# --------------------------------------------------------------------------

def test_perf_model_starts_at_chance_and_saturates_below_ceiling():
    assert perfModel(0.0, alpha=1.0, beta=2.0, lapse=5.0) == pytest.approx(50.0)
    far = perfModel(1e6, alpha=1.0, beta=2.0, lapse=5.0)
    assert far == pytest.approx(95.0, abs=1e-3)      # 100 - lapse


def test_perf_model_is_monotonically_increasing():
    t = np.linspace(0, 5, 200)
    p = perfModel(t, alpha=1.0, beta=2.0, lapse=0.0)
    assert np.all(np.diff(p) >= 0)


def test_perf_model_reaches_half_way_at_alpha():
    '''alpha is the half-saturation time, which is what makes it readable.'''
    p = perfModel(2.5, alpha=2.5, beta=3.0, lapse=10.0)
    assert p == pytest.approx(50.0 + (50.0 - 10.0) / 2)


def test_fit_recovers_known_parameters():
    t = np.linspace(0.3, 5, 60)
    truth = perfModel(t, alpha=1.2, beta=2.5, lapse=8.0)
    fitted = fitPerfCurve(t, truth, t)
    np.testing.assert_allclose(fitted, truth, atol=0.5)


def test_fit_falls_back_to_interpolation_with_too_few_points():
    '''Three parameters need at least three finite bins.'''
    bins = np.array([1.0, 2.0, 3.0])
    perf = np.array([60.0, np.nan, np.nan])
    out = fitPerfCurve(bins, perf, np.array([1.0, 1.5]))
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(60.0)


def test_fit_ignores_nan_bins():
    t = np.linspace(0.3, 5, 60)
    perf = perfModel(t, alpha=1.2, beta=2.5, lapse=8.0)
    holed = perf.copy()
    holed[::7] = np.nan
    np.testing.assert_allclose(fitPerfCurve(t, holed, t), perf, atol=1.0)


# --------------------------------------------------------------------------
# Methods eqs. (2)-(3) -- the reward-rate curve
# --------------------------------------------------------------------------

def test_reward_curve_matches_hand_computation():
    '''At 100% accuracy only the correct overhead applies.'''
    tph, reward = rewardCurve([1.0], [100.0], correct_extra=1.0,
                              incorrect_extra=2.0)
    assert tph[0] == pytest.approx(3600 / 2.0)       # E[T] = 1 + 1
    assert reward[0] == pytest.approx(1800.0)


def test_reward_curve_mixes_the_two_overheads_by_accuracy():
    tph, reward = rewardCurve([1.0], [50.0], correct_extra=1.0,
                              incorrect_extra=2.0)
    # E[T] = (1+1)*0.5 + (2+1)*0.5 = 2.5
    assert tph[0] == pytest.approx(3600 / 2.5)
    assert reward[0] == pytest.approx(3600 / 2.5 * 0.5)


def test_longer_sampling_is_not_always_better():
    '''The trade-off that makes an interior optimum exist.'''
    t = np.linspace(0.05, 5, 500)
    perf = perfModel(t, alpha=0.4, beta=3.0, lapse=0.0)
    _, reward = rewardCurve(t, perf, correct_extra=1.0, incorrect_extra=2.0)
    t_opt = optimalSamplingTime(t, reward)
    assert t.min() < t_opt < t.max()
    assert reward[-1] < reward[np.argmax(reward)]


def test_optimum_moves_later_when_evidence_accumulates_more_slowly():
    '''While the optimum stays interior, slower accumulation pushes it out.'''
    t = np.linspace(0.05, 5, 500)
    opt = lambda alpha: optimalSamplingTime(
        t, rewardCurve(t, perfModel(t, alpha=alpha, beta=3.0, lapse=0.0),
                       1.0, 2.0)[1])
    assert opt(0.8) > opt(0.3)


def test_optimum_moves_later_when_trials_are_expensive():
    '''Long overheads make each trial worth more accuracy.'''
    t = np.linspace(0.05, 5, 500)
    perf = perfModel(t, alpha=0.8, beta=3.0, lapse=0.0)
    cheap = optimalSamplingTime(t, rewardCurve(t, perf, 0.2, 0.4)[1])
    pricey = optimalSamplingTime(t, rewardCurve(t, perf, 6.0, 12.0)[1])
    assert pricey > cheap


# --------------------------------------------------------------------------
# Aggregation rules
# --------------------------------------------------------------------------

def test_extra_time_is_duration_minus_sampling_time():
    df = _animalDf("M1", seed=0, n_per_session=50, sessions=1, iti=2.5)
    df["TrialDuration"] = (df.TrialStartSysTime.shift(-1)
                           - df.TrialStartSysTime)
    assert avgExtraTime(df) == pytest.approx(2.5, abs=1e-6)


def test_aggregations_are_nested_not_flat():
    '''Sessions are averaged before animals, so a long session cannot dominate.

    A 900-trial session at 100% and a 100-trial session at 0% average to 50%,
    not to the trial-weighted 90%.
    '''
    big = _sessionRows("M1", "2024-01-01", 900, seed=1, perf=1.0)
    small = _sessionRows("M1", "2024-01-02", 100, seed=2, perf=0.0)
    small["SessionNum"] = 2
    assert avgPerf(pd.concat([big, small])) == pytest.approx(50.0, abs=1e-6)


def test_sparse_bins_are_dropped():
    df = _animalDf("M1", seed=3, n_per_session=400, sessions=1)
    loose = buildPerfCurve(df, min_bin_count=1)[1]
    strict = buildPerfCurve(df, min_bin_count=250)[1]
    assert np.isnan(strict).sum() > np.isnan(loose).sum()


def test_bin_centres_sit_between_edges():
    bins = defaultBins()
    centres = buildPerfCurve(_animalDf("M1", seed=4, n_per_session=200,
                                       sessions=1), bins)[2]
    assert len(centres) == len(bins) - 1
    np.testing.assert_allclose(centres, (bins[:-1] + bins[1:]) / 2)


def test_animals_below_the_trial_floor_are_excluded():
    df = pd.concat([_animalDf("Keep", seed=5, n_per_session=400, sessions=2),
                    _animalDf("Drop", seed=6, n_per_session=40, sessions=1)],
                   ignore_index=True)
    _, subj = collectMetrics(df, min_trials=100)
    assert set(subj) == {"Keep"}


def test_min_trials_default_matches_the_methods():
    assert MIN_TRIALS == 1_000


def test_animal_without_both_outcomes_is_skipped_not_raised():
    df = pd.concat([_animalDf("Mixed", seed=7, n_per_session=400, sessions=2),
                    _animalDf("AlwaysRight", seed=8, n_per_session=400,
                              sessions=2, perf=1.0)], ignore_index=True)
    _, subj = collectMetrics(df, min_trials=100)
    assert set(subj) == {"Mixed"}


def test_collect_metrics_reports_sd_not_sem():
    '''The paper's error bars are +- SD; SEM would shrink with trial count.'''
    df = _animalDf("M1", seed=9, n_per_session=600, sessions=2, st_sd=0.5)
    _, subj = collectMetrics(df, min_trials=100)
    expected = df.calcStimulusTime.std()
    assert subj["M1"]["observed_sampling_time_sd"] == pytest.approx(expected)


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------

def _twoAnimals():
    return pd.concat([
        _animalDf("Tight", seed=10, n_per_session=600, sessions=2, st_sd=0.10),
        _animalDf("Broad", seed=20, n_per_session=600, sessions=2, st_sd=0.80),
    ], ignore_index=True)


def _errorBarWidths(ax):
    '''Horizontal extent of each errorbar, read off its LineCollection.'''
    widths = []
    for coll in ax.collections:
        if not isinstance(coll, matplotlib.collections.LineCollection):
            continue
        for seg in coll.get_segments():
            if len(seg) == 2 and abs(seg[0][1] - seg[1][1]) < 1e-9:
                widths.append(float(abs(seg[1][0] - seg[0][0])))
    return widths


def test_population_panel_uses_each_animals_own_sd():
    '''Regression test for the stale-loop-variable bug.

    The inline version read the SD from a variable left over from the previous
    loop, so every animal was drawn with the last animal's SD.
    '''
    _, subj = collectMetrics(_twoAnimals(), min_trials=100)
    assert len(subj) == 2
    fig = plotAlignedPopulation(subj)
    widths = _errorBarWidths(fig.axes[0])
    assert len(widths) == 2
    assert widths[0] != pytest.approx(widths[1], rel=0.05)
    expected = sorted(2 * m["observed_sampling_time_sd"]
                      for m in subj.values())
    np.testing.assert_allclose(sorted(widths), expected, rtol=1e-6)
    plt.close(fig)


def test_population_panel_marks_one_row_per_animal():
    _, subj = collectMetrics(_twoAnimals(), min_trials=100)
    fig = plotAlignedPopulation(subj)
    assert len(fig.axes[0].get_yticks()) == len(subj)
    plt.close(fig)


def test_population_panel_rejects_an_empty_cohort():
    with pytest.raises(ValueError, match="no animal"):
        plotAlignedPopulation({})


def test_subject_panel_marks_the_optimum_and_the_observation():
    overall, subj = collectMetrics(_twoAnimals(), min_trials=100)
    fig = plotSubjectRewardCurve(overall, "Tight", subj)
    ax = fig.axes[0]
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert {"Reward-optimal", "Observed"} <= labels
    # The observed marker carries this animal's own SD.
    assert _errorBarWidths(ax) == pytest.approx(
        [2 * subj["Tight"]["observed_sampling_time_sd"]], rel=1e-6)
    plt.close(fig)


def test_panels_save_under_the_prefix(tmp_path):
    overall, subj = collectMetrics(_twoAnimals(), min_trials=100)
    plt.close(plotAlignedPopulation(subj, save_prefix=tmp_path,
                                    save_figs=True))
    plt.close(plotSubjectRewardCurve(overall, "Tight", subj,
                                     save_prefix=tmp_path, save_figs=True))
    saved = {p.name for p in (tmp_path / "optimal_sampling").iterdir()}
    assert saved == {"all_subj_aligned.svg", "subj__Tight.svg"}


def test_panels_refuse_to_save_without_a_prefix():
    overall, subj = collectMetrics(_twoAnimals(), min_trials=100)
    with pytest.raises(ValueError, match="save_prefix"):
        plotAlignedPopulation(subj, save_figs=True)
    with pytest.raises(ValueError, match="save_prefix"):
        plotSubjectRewardCurve(overall, "Tight", subj, save_figs=True)


def test_subject_filenames_escape_dots():
    '''Animal names like "vgat2.5" must not truncate the file extension.'''
    df = _animalDf("vgat2.5", seed=30, n_per_session=600, sessions=2)
    overall, subj = collectMetrics(df, min_trials=100)
    fig = plotSubjectRewardCurve(overall, "vgat2.5", subj)
    plt.close(fig)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        fig = plotSubjectRewardCurve(overall, "vgat2.5", subj,
                                     save_prefix=tmp, save_figs=True)
        plt.close(fig)
        from pathlib import Path
        assert (Path(tmp) / "optimal_sampling"
                / "subj__vgat2_5.svg").exists()
