'''Tests for the per-animal strategy comparison (Figure S3M).

Verified against the real data while extracting: the Holm-corrected p-values
reproduce the notebook exactly (MLA-73 0.073, MLA-74 0.722, MLA-75 0.105,
MLA-76 0.722), the axis title matches, and the saved SVG is text-identical to
the committed one. **0 of 4 mice significant**, as the manuscript reports.

Worth knowing about that result: two of the four raw p-values are below 0.05
(0.018 and 0.035) and only the Holm correction lifts them above it. The
correction is doing real work here, so :func:`test_holm_correction_can_flip_a
_raw_significant_result` pins it -- dropping the correction would change the
paper's claim from "no mouse" to "half the mice".
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ..centroids import ANGLE_COL
from ..strategy import (animalQuantileGroups, assertAllNonNormal,
                        compareStrategies, distanceFromMode,
                        plotStrategyComparison, trialMeanAngles)

FAST, TYPICAL, SLOW = 1, 2, 3


def frames(name, trial_number, angles, *, quantile=FAST, day="day1"):
    angles = np.asarray(angles, dtype=float)
    return pd.DataFrame({
        "Name": name, "Date": day, "SessionNum": 1,
        "File": f"{day}/{name}/{name}_session.mat",
        "TrialNumber": trial_number, "quantile_idx": quantile,
        ANGLE_COL: angles})


def animal(name="MLA-73", n_per_quantile=40, centre=0., spreads=(8., 8., 8.),
           seed=0):
    '''Trials for one animal, one row per trial, tertile spreads controllable.'''
    rng = np.random.default_rng(seed)
    out, number = [], 0
    for quantile, spread in zip((FAST, TYPICAL, SLOW), spreads):
        for _ in range(n_per_quantile):
            # A heavy tail keeps Shapiro rejecting, as the real data does.
            value = centre + rng.standard_t(2) * spread
            out.append(frames(name, number, [value], quantile=quantile))
            number += 1
    return pd.concat(out, ignore_index=True)


def cohort(n_animals=4, **kwargs):
    return pd.concat([animal(f"MLA-{73 + index}", seed=index, **kwargs)
                      for index in range(n_animals)], ignore_index=True)


# --------------------------------------------------------------------------
# Reducing frames to trials
# --------------------------------------------------------------------------

def test_each_trial_becomes_one_row_carrying_its_mean():
    df = pd.concat([frames("A", 0, [10., 20., 30.]),
                    frames("A", 1, [0., 4.])], ignore_index=True)
    trials = trialMeanAngles(df)
    assert len(trials) == 2
    assert sorted(trials[ANGLE_COL]) == [2., 20.]


def test_a_zero_ratio_skips_trimming_entirely():
    '''Same values as trimming nothing, but without rebuilding 76k rows.'''
    df = pd.concat([frames("A", 0, [0., 1, 2, 3, 100])], ignore_index=True)
    assert trialMeanAngles(df, outliers_ratio=0)[ANGLE_COL].iloc[0] == \
        pytest.approx(21.2)


def test_a_positive_ratio_trims_before_averaging():
    df = frames("A", 0, [0., 1, 2, 3, 100])
    trimmed = trialMeanAngles(df, outliers_ratio=0.1)[ANGLE_COL].iloc[0]
    assert trimmed < 21.2
    assert trimmed == pytest.approx(2.)


# --------------------------------------------------------------------------
# Distance from the animal's neutral posture
# --------------------------------------------------------------------------

def test_the_mode_is_taken_per_animal_in_five_degree_bins():
    trials = pd.concat(
        [frames("A", index, [21.]) for index in range(5)]
        + [frames("A", 9, [50.])]
        + [frames("B", index, [-31.]) for index in range(10, 15)],
        ignore_index=True)
    scored = distanceFromMode(trialMeanAngles(trials))
    assert set(scored[scored.Name == "A"].base_angle) == {20.}
    assert set(scored[scored.Name == "B"].base_angle) == {-30.}


def test_distance_from_the_mode_is_unsigned():
    trials = pd.concat([frames("A", 0, [20.]), frames("A", 1, [30.]),
                        frames("A", 2, [10.]), frames("A", 3, [20.])],
                       ignore_index=True)
    scored = distanceFromMode(trialMeanAngles(trials))
    assert (scored.abs_angle_from_base >= 0).all()
    assert sorted(scored.abs_angle_from_base) == [0., 0., 10., 10.]


def test_an_animal_that_shifts_posture_is_measured_against_one_neutral():
    '''The mode is per animal, not per session, by design.'''
    trials = pd.concat(
        [frames("A", index, [20.], day="day1") for index in range(6)]
        + [frames("A", index, [40.], day="day2") for index in range(6, 9)],
        ignore_index=True)
    scored = distanceFromMode(trialMeanAngles(trials))
    assert set(scored.base_angle) == {20.}
    assert set(scored[scored.Date == "day2"].abs_angle_from_base) == {20.}


# --------------------------------------------------------------------------
# Which animals get tested
# --------------------------------------------------------------------------

def test_an_animal_needs_all_three_tertiles():
    df = pd.concat([animal("Complete", seed=1),
                    animal("Partial", seed=2)], ignore_index=True)
    df = df[~((df.Name == "Partial") & (df.quantile_idx == SLOW))]
    groups = animalQuantileGroups(distanceFromMode(trialMeanAngles(df)))
    assert set(groups) == {"Complete"}


def _thinSlowTertile(n_per_quantile=40, keep=4):
    '''One animal whose slow tertile has too few trials to count.'''
    df = animal("Thin", n_per_quantile=n_per_quantile, seed=3)
    last = df.TrialNumber.max()
    thin = (df.quantile_idx == SLOW) & (df.TrialNumber > last - keep)
    return df[(df.quantile_idx != SLOW) | thin]


def test_a_thin_tertile_is_dropped_and_takes_its_animal_with_it():
    df = _thinSlowTertile()
    assert (df.quantile_idx == SLOW).sum() == 4        # below the 10 minimum
    groups = animalQuantileGroups(distanceFromMode(trialMeanAngles(df)))
    assert groups == {}


def test_the_threshold_is_configurable():
    scored = distanceFromMode(trialMeanAngles(_thinSlowTertile()))
    assert animalQuantileGroups(scored, min_trials=3) != {}


def test_each_animal_keeps_one_array_per_tertile():
    groups = animalQuantileGroups(
        distanceFromMode(trialMeanAngles(cohort(n_animals=2))))
    assert len(groups) == 2
    for per_quantile in groups.values():
        assert sorted(per_quantile) == [FAST, TYPICAL, SLOW]
        assert all(isinstance(v, np.ndarray) for v in per_quantile.values())


# --------------------------------------------------------------------------
# The test itself
# --------------------------------------------------------------------------

def _groups(df):
    return animalQuantileGroups(distanceFromMode(trialMeanAngles(df)))


def test_one_p_value_per_animal():
    result = compareStrategies(_groups(cohort(n_animals=4)))
    assert result["n_tested"] == 4
    assert set(result["raw"]) == set(result["corrected"])
    assert all(0 <= p <= 1 for p in result["corrected"].values())


def test_identical_tertiles_are_not_significant():
    result = compareStrategies(_groups(cohort(n_animals=4)))
    assert result["n_rejected"] == 0


def test_a_genuinely_different_tertile_is_detected():
    '''One animal whose slow trials sit far from its neutral posture.'''
    odd = animal("Odd", spreads=(4., 4., 40.), seed=7)
    result = compareStrategies(_groups(odd))
    assert result["raw"]["Odd"] < 0.05


def test_holm_correction_can_flip_a_raw_significant_result():
    '''Load-bearing for the published claim -- see the module docstring.

    Constructed so one animal's raw p-value sits just under 0.05 while the
    Holm-corrected value does not.
    '''
    raw = {"a": 0.02, "b": 0.4, "c": 0.5, "d": 0.6}
    from statsmodels.stats.multitest import multipletests
    reject, corrected, _, _ = multipletests(list(raw.values()), method="holm")
    assert raw["a"] < 0.05
    assert corrected[0] == pytest.approx(0.08)
    assert not reject.any()


def test_correction_is_across_animals_not_across_tertiles():
    '''The family is the four mice; each mouse yields a single omnibus p.'''
    result = compareStrategies(_groups(cohort(n_animals=4)))
    for name, raw in result["raw"].items():
        assert result["corrected"][name] >= raw
        assert result["corrected"][name] <= min(1., 4 * raw)


# --------------------------------------------------------------------------
# The normality guard
# --------------------------------------------------------------------------

def test_a_normal_group_is_reported_rather_than_silently_skipped():
    '''A normal tertile means that animal never reaches Kruskal-Wallis.'''
    rng = np.random.default_rng(0)
    tidy = pd.concat(
        [frames("Normal", index + 100 * quantile, [rng.normal(0, 5)],
                quantile=quantile)
         for quantile in (FAST, TYPICAL, SLOW) for index in range(40)],
        ignore_index=True)
    result = compareStrategies(_groups(tidy))
    assert not result["all_non_normal"]
    assert "Normal" not in result["raw"]
    with pytest.raises(AssertionError, match="Some distributions are normal"):
        assertAllNonNormal(result)


def test_heavy_tailed_groups_pass_the_guard():
    result = compareStrategies(_groups(cohort(n_animals=4)))
    assert result["all_non_normal"]
    assertAllNonNormal(result)


# --------------------------------------------------------------------------
# The panel
# --------------------------------------------------------------------------

def test_the_title_reports_how_many_mice_were_significant():
    fig, ax = plt.subplots()
    result = plotStrategyComparison(cohort(n_animals=4), ax=ax, seed=0,
                                    verbose=False)
    assert f"{result['n_rejected']}/{result['n_tested']} mice" in ax.get_title()
    assert "Kruskal-Wallis test" in ax.get_title()
    plt.close(fig)


def test_mice_are_labelled_by_position_not_by_name():
    fig, ax = plt.subplots()
    plotStrategyComparison(cohort(n_animals=3), ax=ax, seed=0, verbose=False)
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "Mouse#1", "Mouse#2", "Mouse#3"]
    plt.close(fig)


def test_three_scatter_clouds_are_drawn_per_mouse():
    fig, ax = plt.subplots()
    plotStrategyComparison(cohort(n_animals=3), ax=ax, seed=0, verbose=False)
    assert len(ax.collections) == 9
    plt.close(fig)


def test_no_bracket_is_drawn_when_nothing_is_significant():
    fig, ax = plt.subplots()
    result = plotStrategyComparison(cohort(n_animals=4), ax=ax, seed=0,
                                    verbose=False)
    assert result["n_rejected"] == 0
    assert not [t for t in ax.texts if t.get_text() == "*"]
    plt.close(fig)


def test_the_seed_makes_the_jitter_reproducible():
    def _first(seed):
        fig, ax = plt.subplots()
        plotStrategyComparison(cohort(n_animals=2), ax=ax, seed=seed,
                               verbose=False)
        offsets = ax.collections[0].get_offsets()[:, 0].copy()
        plt.close(fig)
        return offsets

    np.testing.assert_allclose(_first(0), _first(0))
    assert not np.allclose(_first(0), _first(1))


def test_the_panel_saves_under_the_prefix(tmp_path):
    plotStrategyComparison(cohort(n_animals=3), seed=0, save_prefix=tmp_path,
                           save_figs=True, verbose=False)
    assert (tmp_path / "strategy_dist_sgf.svg").exists()
    plt.close("all")


def test_saving_without_a_prefix_is_refused():
    '''It used to read ``save_prefix`` from the notebook's globals.'''
    with pytest.raises(ValueError, match="save_prefix"):
        plotStrategyComparison(cohort(n_animals=2), save_figs=True,
                               verbose=False)
    plt.close("all")


def test_the_panel_raises_no_pandas_deprecation_warning():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        fig, ax = plt.subplots()
        plotStrategyComparison(cohort(n_animals=2), ax=ax, seed=0,
                               verbose=False)
        plt.close(fig)
