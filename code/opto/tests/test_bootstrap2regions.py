'''Tests for the two-region hierarchical bootstrap (Figures 4C and S6G).

Where :mod:`opto.bootstrapping` pools every resampled trial and applies one
statistic, this module keeps the nesting in the *estimator* too: it computes
one effect per subject x region x phase block, then averages those. The two
give different answers on unbalanced data, so the tests below pin which one
this module implements.

Three separate schemes live in ``bootstrapSignTestApproach2`` and are easy to
confuse:

- **within-region** (MFC vs 0, LFC vs 0) resamples subjects, then sessions,
  then trials, and averages *subject* effects;
- **cross-region** (delta = MFC - LFC) resamples *sessions only* and compares
  20%-trimmed means of session effects;
- p-values come from the sign-change rule, Holm-corrected within phase for
  the within-region tests and across phases for the cross-region ones.

Two dead helpers are pinned deliberately. ``_holm_step_down`` is an exact
duplicate of the ``statsmodels`` call the module actually uses, and
``_bh_fdr`` is a *wrong* Benjamini-Hochberg (it takes a forward cumulative
minimum where the step-up procedure needs a backward one, making adjusted
p-values too small). Neither is called; both are recorded in
``docs/repo-audit.md`` for removal, and the tests state why.
'''
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats import multitest

from ..bootstrap2regions import (_aggregate_cross_region_effects,
                                 _aggregate_group_effects, _bh_fdr,
                                 _bootstrap_iteration_subject_entries,
                                 _effect_from_trials_block, _holm_step_down,
                                 _sign_change_p_two_sided,
                                 _subject_entries_once,
                                 bootstrapSignTestApproach2)

#: Effect in percentage points: positive means opto hurt performance.
def perfDrop(control_mean: float, opto_mean: float) -> float:
    return 100 * (control_mean - opto_mean)


#: 40 trials per arm keeps every rate used below a whole number of trials.
N_ARM = 40


def _correct(n_arm: int, rate: float, *, exact: bool) -> int:
    '''Successes for a rate, refusing to round when the test asserts a value.

    ``round(30 * .85)`` is 26, not 25.5, which quietly shifts the effect a
    fixture claims to produce. Tests that assert an exact effect pass
    ``exact=True`` so a future fixture tweak fails here rather than in an
    assertion three files away.
    '''
    scaled = n_arm * rate
    if exact:
        assert abs(scaled - round(scaled)) < 1e-9, (
            f"{rate} of {n_arm} trials is {scaled}, not a whole number")
    return round(scaled)


def session(name, region, is_early, sess_id, *, n_arm=N_ARM, p_control=.85,
            effect=.25, exact=True) -> pd.DataFrame:
    '''One session block with exact success counts in each arm.'''
    n_correct_control = _correct(n_arm, p_control, exact=exact)
    n_correct_opto = _correct(n_arm, p_control - effect, exact=exact)
    outcome = ([1.] * n_correct_control + [0.] * (n_arm - n_correct_control)
               + [1.] * n_correct_opto + [0.] * (n_arm - n_correct_opto))
    return pd.DataFrame({"Name": name, "OptoBrainRegion": region,
                         "IsEarly": is_early, "SessId": sess_id,
                         "OptoEnabled": [False] * n_arm + [True] * n_arm,
                         "ChoiceCorrect": outcome})


def cohort(n_subjects=4, n_sessions=2, mfc_effect=.25, lfc_effect=.10,
           jitter=0., subject_step=0., exact=True) -> pd.DataFrame:
    '''MFC hurt more than LFC, in both phases, for every animal.

    ``jitter`` spreads the effect across sessions, which the cross-region
    trimmed means need in order to have any spread at all; ``subject_step``
    spreads it across animals, which is what the subject-level resampling
    acts on. Both are centred, so the group mean stays at the nominal effect.
    '''
    frames = []
    for subject in range(n_subjects):
        offset = (subject - (n_subjects - 1) / 2) * subject_step
        for region, effect in (("MFC", mfc_effect), ("LFC", lfc_effect)):
            for is_early in (True, False):
                for index in range(n_sessions):
                    step = (index - (n_sessions - 1) / 2) * jitter
                    frames.append(session(
                        f"m{subject}", region, is_early,
                        f"m{subject}_{region}_{int(is_early)}_{index}",
                        effect=effect + step + offset, exact=exact))
    return pd.concat(frames, ignore_index=True)


def block(df, name="m0", region="MFC", is_early=True):
    return df[(df.Name == name) & (df.OptoBrainRegion == region)
              & (df.IsEarly == is_early)]


# --------------------------------------------------------------------------
# One subject x region x phase block
# --------------------------------------------------------------------------

def test_a_block_pools_its_sessions_before_taking_the_effect():
    '''85% control against 60% opto is a 25-point drop, however it is split.'''
    assert _effect_from_trials_block(block(cohort()), perfDrop) == pytest.approx(25.)


def test_a_block_missing_one_arm_gives_nan_rather_than_an_error():
    control_only = block(cohort())
    control_only = control_only[~control_only.OptoEnabled]
    assert np.isnan(_effect_from_trials_block(control_only, perfDrop))


def test_trial_resampling_is_stratified_so_an_arm_is_never_lost():
    '''Drawing within each label separately is what keeps this NaN-free.

    An unstratified draw over the whole session could, by chance, return no
    opto trials and silently drop the block from that iteration.
    '''
    rng = np.random.default_rng(0)
    draws = [_effect_from_trials_block(block(cohort()), perfDrop, rng=rng,
                                       resample_trials=True)
             for _ in range(200)]
    assert not np.isnan(draws).any()
    assert np.std(draws) > 0


def test_resampling_leaves_the_effect_centred_where_it_was():
    rng = np.random.default_rng(0)
    draws = [_effect_from_trials_block(block(cohort()), perfDrop, rng=rng,
                                       resample_trials=True)
             for _ in range(300)]
    assert np.mean(draws) == pytest.approx(25., abs=2.)


def test_resampling_requires_an_rng():
    with pytest.raises(AssertionError):
        _effect_from_trials_block(block(cohort()), perfDrop, resample_trials=True)


# --------------------------------------------------------------------------
# Subject entries
# --------------------------------------------------------------------------

def test_one_entry_per_subject_region_and_phase():
    entries = _subject_entries_once(cohort(n_subjects=4), perfDrop)
    assert len(entries) == 4 * 2 * 2
    assert list(entries.columns) == ["Name", "OptoBrainRegion", "Time", "effect"]
    assert set(entries.Time) == {"Early", "Late"}


def test_is_early_becomes_the_phase_label():
    entries = _subject_entries_once(cohort(n_subjects=1), perfDrop)
    early = entries[entries.Time == "Early"]
    assert len(early) == 2                       # one per region
    assert set(entries.Time) == {"Early", "Late"}


def test_blocks_that_evaluate_to_nan_are_dropped_not_carried():
    df = cohort(n_subjects=2)
    strip = ((df.Name == "m1") & (df.OptoBrainRegion == "LFC")
             & df.IsEarly & df.OptoEnabled)
    entries = _subject_entries_once(df[~strip], perfDrop)
    assert len(entries) == 2 * 2 * 2 - 1
    assert entries.effect.notna().all()


def test_any_region_that_is_not_mfc_is_relabelled_lfc():
    '''A hazard, pinned so it is visible rather than surprising.

    The mapping is ``'MFC' if region == 'MFC' else 'LFC'``, so a third region
    reaching this function is silently folded into the LFC group instead of
    raising. ``optoprocessor`` currently passes only two regions.
    '''
    df = pd.concat([cohort(n_subjects=1),
                    session("m9", "V1", True, "m9_V1", effect=.5)],
                   ignore_index=True)
    entries = _subject_entries_once(df, perfDrop)
    assert set(entries.OptoBrainRegion) == {"MFC", "LFC"}
    assert entries[entries.Name == "m9"].OptoBrainRegion.tolist() == ["LFC"]


def test_an_empty_frame_gives_the_declared_columns_back():
    empty = cohort().iloc[:0]
    assert _bootstrap_iteration_subject_entries(
        empty, perfDrop, np.random.default_rng(0)).empty


# --------------------------------------------------------------------------
# Aggregation across subjects
# --------------------------------------------------------------------------

def test_group_means_and_the_cross_region_delta():
    aggregate = _aggregate_group_effects(
        _subject_entries_once(cohort(mfc_effect=.25, lfc_effect=.10), perfDrop))
    assert aggregate["MFC_Early"] == pytest.approx(25.)
    assert aggregate["LFC_Early"] == pytest.approx(10.)
    assert aggregate["Delta_Early"] == pytest.approx(15.)
    assert aggregate["Delta_Late"] == pytest.approx(15.)


def test_a_missing_cell_leaves_nan_and_kills_only_its_own_delta():
    entries = _subject_entries_once(cohort(n_subjects=2), perfDrop)
    entries = entries[~((entries.OptoBrainRegion == "LFC")
                        & (entries.Time == "Early"))]
    aggregate = _aggregate_group_effects(entries)
    assert np.isnan(aggregate["LFC_Early"])
    assert np.isnan(aggregate["Delta_Early"])
    assert not np.isnan(aggregate["Delta_Late"])


def test_subjects_are_weighted_equally_not_by_trial_count():
    '''The estimator averages subject effects, so a big animal cannot dominate.

    This is the difference from ``bootstrapping.bootstrapPerf``, which pools
    every resampled trial first. Here one animal with ten times the trials
    still contributes exactly one entry.
    '''
    small = session("m0", "MFC", True, "m0_s", n_arm=10, p_control=.8, effect=.5)
    large = session("m1", "MFC", True, "m1_s", n_arm=100, p_control=.8, effect=.1)
    aggregate = _aggregate_group_effects(
        _subject_entries_once(pd.concat([small, large], ignore_index=True),
                              perfDrop))
    assert aggregate["MFC_Early"] == pytest.approx(30.)   # (50 + 10) / 2

    arms = pd.concat([small, large]).groupby("OptoEnabled").ChoiceCorrect.mean()
    assert perfDrop(arms[False], arms[True]) == pytest.approx(13.64, abs=.01)


def test_cross_region_delta_uses_trimmed_means_over_sessions():
    '''A separate scheme from the within-region one: sessions, not subjects.'''
    delta = _aggregate_cross_region_effects(
        cohort(mfc_effect=.25, lfc_effect=.10), perfDrop)
    assert delta["Delta_Early"] == pytest.approx(15.)
    assert delta["Delta_Late"] == pytest.approx(15.)


def test_cross_region_session_resampling_needs_an_rng():
    with pytest.raises(AssertionError):
        _aggregate_cross_region_effects(cohort(), perfDrop, resample=True)


def test_cross_region_resampling_varies_the_delta():
    df = cohort(n_sessions=4, jitter=.05)
    rng = np.random.default_rng(0)
    draws = [_aggregate_cross_region_effects(df, perfDrop, resample=True,
                                             rng=rng)["Delta_Early"]
             for _ in range(40)]
    assert np.nanstd(draws) > 0
    assert np.nanmean(draws) == pytest.approx(15., abs=3.)


# --------------------------------------------------------------------------
# Turning draws into a p-value
# --------------------------------------------------------------------------

def test_draws_that_never_change_sign_give_p_zero():
    assert _sign_change_p_two_sided(5., np.array([1., 2., 3.])) == 0.


def test_the_p_value_is_twice_the_flipped_fraction():
    assert _sign_change_p_two_sided(5., np.array([1., 1., 1., -1.])) == 0.5


def test_the_two_tailed_doubling_is_clamped_at_one():
    assert _sign_change_p_two_sided(5., np.array([1., -1., -1., -1.])) == 1.


def test_a_negative_observed_effect_counts_positive_draws_as_flips():
    assert _sign_change_p_two_sided(-5., np.array([1., 1., -1., -1.])) == 1.
    assert _sign_change_p_two_sided(-5., np.array([-1., -1., -1., -1.])) == 0.


def test_an_exactly_zero_effect_has_no_sign_to_change():
    assert _sign_change_p_two_sided(0., np.array([1., -1.])) == 1.


def test_nan_in_gives_nan_out_and_nan_draws_are_ignored():
    assert np.isnan(_sign_change_p_two_sided(np.nan, np.array([1., 2.])))
    assert np.isnan(_sign_change_p_two_sided(1., np.array([np.nan, np.nan])))
    assert _sign_change_p_two_sided(5., np.array([np.nan, 1., 1.])) == 0.


# --------------------------------------------------------------------------
# The two unused correction helpers (see docs/repo-audit.md)
# --------------------------------------------------------------------------

def test_holm_step_down_duplicates_the_statsmodels_call_that_is_used():
    '''Evidence for deleting it: same answers, and nothing calls it.'''
    pvals = pd.Series({"a": .01, "b": .04, "c": .03, "d": .2})
    expected = multitest.multipletests(pvals.values, method="holm")[1]
    np.testing.assert_allclose(_holm_step_down(pvals).values, expected)


def test_holm_step_down_preserves_nans_and_is_monotonic():
    adjusted = _holm_step_down(pd.Series({"a": .01, "b": np.nan, "c": .03}))
    assert np.isnan(adjusted["b"])
    assert adjusted["a"] <= adjusted["c"]
    assert _holm_step_down(pd.Series({"a": np.nan})).isna().all()


def test_bh_fdr_is_wrong_and_reports_p_values_that_are_too_small():
    '''Pinned as a defect, not as behaviour to rely on.

    Benjamini-Hochberg steps *up* from the largest p-value, so the running
    minimum must be taken in descending order. ``_bh_fdr`` uses ``cummin`` on
    the ascending sort, which drags every adjusted p down to the smallest one.
    Nothing calls it; it should be removed rather than fixed.
    '''
    pvals = pd.Series({"a": .01, "b": .04, "c": .03})
    correct = multitest.multipletests(pvals.values, method="fdr_bh")[1]
    np.testing.assert_allclose(correct, [.03, .04, .04])
    np.testing.assert_allclose(_bh_fdr(pvals).values, [.03, .03, .03])
    assert (_bh_fdr(pvals).values <= correct).all()


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def run():
    return bootstrapSignTestApproach2(cohort(n_sessions=3, jitter=.05),
                                      perfDrop, iterations=40, seed=1)


def test_every_test_reports_the_same_summary_fields(run):
    results, _ = run
    tests = {(phase, test) for phase, test, _ in results}
    assert tests == {(phase, test) for phase in ("Early", "Late")
                     for test in ("MFC", "LFC", "crossregion")}
    for phase, test in tests:
        for metric in ("observed", "sd", "ci_low", "ci_high", "p"):
            assert (phase, test, metric) in results


def test_observed_effects_recover_the_simulated_ones(run):
    results, _ = run
    assert results[("Early", "MFC", "observed")] == pytest.approx(25., abs=1.)
    assert results[("Early", "LFC", "observed")] == pytest.approx(10., abs=1.)
    assert results[("Late", "crossregion", "observed")] == pytest.approx(15., abs=2.)


def test_the_observed_entries_come_back_unresampled(run):
    _, entries = run
    assert len(entries) == 4 * 2 * 2
    assert entries.effect.notna().all()


def test_confidence_intervals_bracket_the_observed_value(run):
    results, _ = run
    for phase in ("Early", "Late"):
        for test in ("MFC", "LFC"):
            low = results[(phase, test, "ci_low")]
            high = results[(phase, test, "ci_high")]
            assert low <= results[(phase, test, "observed")] <= high


def test_holm_is_applied_within_phase_and_across_phases(run):
    '''Two families, matching the Methods: not one correction over all six.'''
    results, _ = run
    for phase in ("Early", "Late"):
        for test in ("MFC", "LFC"):
            raw = results[(phase, test, "p")]
            adjusted = results[(phase, test, "p_holm_within_phase")]
            assert adjusted >= raw
            assert adjusted <= min(1., 2 * raw)   # family of two
        assert (phase, "crossregion", "p_holm_across_phases") in results
    assert ("Early", "MFC", "p_holm_across_phases") not in results


def test_a_real_effect_survives_correction_and_a_null_one_does_not():
    '''The null cohort disagrees between animals, so the sign keeps flipping.'''
    strong, _ = bootstrapSignTestApproach2(
        cohort(mfc_effect=.30, lfc_effect=.30, jitter=.05),
        perfDrop, iterations=60, seed=3)
    null, _ = bootstrapSignTestApproach2(
        cohort(mfc_effect=.01, lfc_effect=.01, subject_step=.05, exact=False),
        perfDrop, iterations=60, seed=3)
    assert strong[("Early", "MFC", "p_holm_within_phase")] < 0.05
    assert null[("Early", "MFC", "p_holm_within_phase")] > 0.05


def test_the_seed_makes_the_run_reproducible():
    '''Unlike ``bootstrapping.bootstrapPerf``, this one takes its own seed.'''
    df = cohort(n_sessions=2, jitter=.05)
    first, _ = bootstrapSignTestApproach2(df, perfDrop, iterations=25, seed=11)
    repeat, _ = bootstrapSignTestApproach2(df, perfDrop, iterations=25, seed=11)
    other, _ = bootstrapSignTestApproach2(df, perfDrop, iterations=25, seed=12)
    assert first == repeat
    assert first != other


def test_the_run_raises_no_pandas_deprecation_warning():
    '''Pandas 3 readiness: ``.apply`` must not touch its own grouping column.

    The per-session callback reads ``OptoEnabled`` and ``ChoiceCorrect`` only,
    so excluding ``SessId`` from it leaves every number unchanged.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        bootstrapSignTestApproach2(cohort(n_subjects=2, jitter=.05), perfDrop,
                                   iterations=5, seed=1)


def test_trials_without_a_choice_are_dropped_before_anything_else():
    df = cohort(n_subjects=2)
    df.loc[df.index[:20], "ChoiceCorrect"] = np.nan
    results, entries = bootstrapSignTestApproach2(df, perfDrop, iterations=10,
                                                  seed=1)
    assert entries.effect.notna().all()
    assert not np.isnan(results[("Early", "MFC", "observed")])
