'''Tests for the hierarchical bootstrap behind Figure 3D.

``bootstrapPerf`` resamples subjects, then sessions within each drawn subject,
then trials within each drawn session, and hands the pooled control and opto
outcomes to a caller-supplied statistic. The p-value in Figure 3D is read off
the resulting distribution by the sign rule (Methods: "the proportion of
iterations in which the sign of the effect differed from the sign calculated
using all the data", two-tailed) -- so the tests here pin both the resampling
scheme and the distribution's behaviour under a real effect and under none.

The point of the nesting is that trials within an animal are not independent.
:func:`test_clustering_widens_the_distribution` and
:func:`test_more_trials_per_animal_do_not_shrink_the_clustered_spread` are the
regression tests for that: they fail if the nesting is ever flattened away,
which would silently shrink every confidence interval in the panel.

By default the draws come from the global ``numpy.random`` state, as they
always have; nothing seeds it before Figure 3D is computed. ``rng`` makes a
run reproducible without changing that default. Empty-arm handling is left to
the statistic, and that is pinned as current behaviour.
'''
from __future__ import annotations

import numpy as np
import pytest

from ..bootstrapping import bootstrapPerf

CONTROL, OPTO = 0, 1


def dataset(spec: dict) -> dict:
    '''Build the four parallel lookups from ``{subject: {session: trials}}``.

    ``trials`` is a list of ``(outcome, label)`` pairs, label 0 = control.
    Outcome values double as tracers: giving each subject or session its own
    value lets a resampled pool be attributed back to what produced it.
    '''
    subj_to_sess, outcomes, labels = {}, {}, {}
    for subject, sessions in spec.items():
        subj_to_sess[subject] = np.array(list(sessions))
        for sess_id, trials in sessions.items():
            outcomes[sess_id] = np.asarray([t[0] for t in trials], dtype=float)
            labels[sess_id] = np.asarray([t[1] for t in trials], dtype=int)
    return dict(subj_arr=np.array(list(spec)),
                subj_to_sess_id_arr_dict=subj_to_sess,
                sess_id_to_trial_outcome_arr_dict=outcomes,
                sess_id_to_trial_label_arr_dict=labels)


def block(n_control: int, p_control: float, n_opto: int, p_opto: float) -> list:
    '''A session with exact success counts, so no fixture noise leaks in.'''
    n_correct_control = round(n_control * p_control)
    n_correct_opto = round(n_opto * p_opto)
    outcomes = ([1.] * n_correct_control
                + [0.] * (n_control - n_correct_control)
                + [1.] * n_correct_opto + [0.] * (n_opto - n_correct_opto))
    return list(zip(outcomes, [CONTROL] * n_control + [OPTO] * n_opto))


def perfDrop(control, opto) -> float:
    '''Percentage-point drop under opto -- positive means opto hurt.'''
    return 100 * (np.mean(control) - np.mean(opto))


def recorder(seen: list, value: float=0.):
    '''A statistic that records what it was handed and returns a constant.'''
    def calcPerfFn(control, opto):
        seen.append((control, opto))
        return value
    return calcPerfFn


def cohort(n_subjects: int, p_control: float, p_opto: float, n_arm: int=30):
    return {f"S{i}": {f"S{i}s0": block(n_arm, p_control, n_arm, p_opto)}
            for i in range(n_subjects)}


# --------------------------------------------------------------------------
# What the caller gets back
# --------------------------------------------------------------------------

def test_one_value_per_iteration():
    np.random.seed(0)
    draws = bootstrapPerf(**dataset(cohort(4, .9, .6)), num_iterations=25,
                          calcPerfFn=perfDrop)
    assert draws.shape == (25,)


def test_a_tuple_valued_statistic_stacks_into_a_column_per_element():
    '''Figure 3D uses a scalar, but the array is built by ``np.array`` alone.'''
    np.random.seed(0)
    draws = bootstrapPerf(**dataset(cohort(4, .9, .6)), num_iterations=25,
                          calcPerfFn=lambda c, o: (np.mean(c), np.mean(o)))
    assert draws.shape == (25, 2)


def test_the_statistic_receives_the_two_arms_already_split():
    seen = []
    np.random.seed(0)
    bootstrapPerf(**dataset(cohort(3, 1., 0.)), num_iterations=5,
                  calcPerfFn=recorder(seen))
    assert len(seen) == 5
    for control, opto in seen:
        assert (control == 1.).all()   # p_control = 1 -> every control correct
        assert (opto == 0.).all()      # p_opto = 0    -> every opto wrong


def test_outcomes_stay_paired_with_their_own_label():
    '''Both arrays are indexed by the same draw, so pairing cannot drift.

    Trial ``i`` carries outcome ``i`` and label ``i % 2``, so every value
    landing in the control arm must be even and every opto value odd.
    '''
    spec = {"A": {"A0": [(float(i), i % 2) for i in range(40)]}}
    seen = []
    np.random.seed(0)
    bootstrapPerf(**dataset(spec), num_iterations=30, calcPerfFn=recorder(seen))
    for control, opto in seen:
        assert (control % 2 == 0).all()
        assert (opto % 2 == 1).all()


# --------------------------------------------------------------------------
# The three resampling levels
# --------------------------------------------------------------------------

def _pools(spec, iterations, seed=0):
    seen = []
    np.random.seed(seed)
    bootstrapPerf(**dataset(spec), num_iterations=iterations,
                  calcPerfFn=recorder(seen))
    return [np.concatenate(arms) for arms in seen]


TWO_SUBJECTS = {"A": {"A0": [(10., CONTROL), (10., OPTO)] * 10},
                "B": {"B0": [(20., CONTROL), (20., OPTO)] * 10}}


def test_subjects_are_drawn_with_replacement():
    '''Two subjects with disjoint tracer values, one session each.

    Each iteration draws 2 of 2 subjects with replacement, so a quarter of
    iterations should see one subject twice and the other not at all.
    '''
    pools = _pools(TWO_SUBJECTS, 200)
    only_a = [p for p in pools if set(np.unique(p)) == {10.}]
    only_b = [p for p in pools if set(np.unique(p)) == {20.}]
    both = [p for p in pools if set(np.unique(p)) == {10., 20.}]
    assert only_a and only_b and both
    assert len(only_a) + len(only_b) + len(both) == 200


def test_a_subject_drawn_twice_contributes_its_trials_twice():
    '''0, 1 or 2 copies of a subject -- never a partial one.'''
    pools = _pools(TWO_SUBJECTS, 200)
    assert {int((p == 10.).sum()) for p in pools} == {0, 20, 40}
    assert {len(p) for p in pools} == {40}    # pool size is always preserved


def test_sessions_are_drawn_within_the_subject_that_owns_them():
    '''One subject, two sessions -- so only the session level can vary.'''
    spec = {"A": {"A0": [(10., CONTROL), (10., OPTO)] * 10,
                  "A1": [(20., CONTROL), (20., OPTO)] * 10}}
    pools = _pools(spec, 200)
    assert any(set(np.unique(p)) == {10.} for p in pools)
    assert any(set(np.unique(p)) == {20.} for p in pools)
    assert all(set(np.unique(p)) <= {10., 20.} for p in pools)


def test_trials_are_drawn_within_their_session():
    '''40 distinct trials resampled to 40 draws must repeat something.'''
    spec = {"A": {"A0": [(float(i), i % 2) for i in range(40)]}}
    pools = _pools(spec, 50)
    assert all(len(p) == 40 for p in pools)
    assert all(len(np.unique(p)) < 40 for p in pools)


# --------------------------------------------------------------------------
# Why the nesting is there at all
# --------------------------------------------------------------------------

def _clusteredAndFlat(n_arm: int, opto_rates=(.1, .3, .5, .7, .9, 1.)):
    '''The same trials, once nested under six animals and once under one.'''
    clustered = {f"S{i}": {f"S{i}s0": block(n_arm, .9, n_arm, rate)}
                 for i, rate in enumerate(opto_rates)}
    pooled = [trial for sessions in clustered.values()
              for trials in sessions.values() for trial in trials]
    return clustered, {"one": {"one_s0": pooled}}


def _spread(spec, iterations=400, seed=0):
    np.random.seed(seed)
    return float(np.nanstd(bootstrapPerf(**dataset(spec),
                                         num_iterations=iterations,
                                         calcPerfFn=perfDrop)))


def test_clustering_widens_the_distribution():
    '''Identical trials; only the nesting differs.

    Flattening six animals into one erases the between-animal variance and the
    bootstrap distribution collapses. Every confidence interval and every
    sign-rule p-value in Figure 3D depends on that not happening.
    '''
    clustered, flat = _clusteredAndFlat(n_arm=50)
    assert _spread(clustered) > 2 * _spread(flat)


def test_more_trials_per_animal_do_not_shrink_the_clustered_spread():
    '''The sharper form of the same property.

    Quadrupling trials per animal halves the flat bootstrap's spread (it goes
    as 1/sqrt(n)) but barely moves the clustered one, because six animals is
    still six animals. A regression that quietly drops a level would show up
    here as the clustered spread starting to track the flat one.
    '''
    small_clustered, small_flat = _clusteredAndFlat(n_arm=25)
    large_clustered, large_flat = _clusteredAndFlat(n_arm=100)
    assert _spread(large_flat) < 0.65 * _spread(small_flat)
    assert _spread(large_clustered) > 0.8 * _spread(small_clustered)


# --------------------------------------------------------------------------
# The sign rule that turns the draws into a p-value
# --------------------------------------------------------------------------

def signRulePValue(draws: np.ndarray, observed: float) -> float:
    '''The rule ``optoprocessor`` applies to ``bootstrapPerf``'s output.'''
    draws = draws[~np.isnan(draws)]
    wrong_sign = (draws < 0) if observed > 0 else (draws > 0)
    return min(2 * wrong_sign.sum() / len(draws), 1.)


def test_a_consistent_effect_never_flips_sign():
    np.random.seed(1)
    draws = bootstrapPerf(**dataset(cohort(5, .95, .25)), num_iterations=300,
                          calcPerfFn=perfDrop)
    assert signRulePValue(draws, observed=70.) == 0.


def test_no_effect_flips_sign_about_half_the_time():
    np.random.seed(1)
    draws = bootstrapPerf(**dataset(cohort(5, .7, .7)), num_iterations=300,
                          calcPerfFn=perfDrop)
    assert 0.3 < (draws < 0).mean() < 0.7
    assert signRulePValue(draws, observed=0.1) > 0.5


def test_the_distribution_is_centred_on_the_observed_effect():
    np.random.seed(2)
    draws = bootstrapPerf(**dataset(cohort(6, .9, .6)), num_iterations=400,
                          calcPerfFn=perfDrop)
    assert np.nanmean(draws) == pytest.approx(30., abs=4.)


# --------------------------------------------------------------------------
# Current behaviour worth knowing about (see docs/repo-audit.md)
# --------------------------------------------------------------------------

def test_by_default_the_global_numpy_state_drives_the_draws():
    """Unchanged behaviour: seeding the global state reproduces a run."""
    spec = dataset(cohort(4, .9, .6))
    np.random.seed(7)
    first = bootstrapPerf(**spec, num_iterations=20, calcPerfFn=perfDrop)
    np.random.seed(7)
    repeat = bootstrapPerf(**spec, num_iterations=20, calcPerfFn=perfDrop)
    unseeded = bootstrapPerf(**spec, num_iterations=20, calcPerfFn=perfDrop)
    np.testing.assert_allclose(first, repeat)
    assert not np.allclose(first, unseeded)


def test_an_integer_rng_reproduces_a_run_regardless_of_global_state():
    spec = dataset(cohort(4, .9, .6))
    np.random.seed(1)
    first = bootstrapPerf(**spec, num_iterations=20, calcPerfFn=perfDrop, rng=3)
    np.random.seed(99)
    repeat = bootstrapPerf(**spec, num_iterations=20, calcPerfFn=perfDrop, rng=3)
    np.testing.assert_allclose(first, repeat)


def test_an_integer_rng_matches_seeding_the_global_state_with_it():
    """Same legacy stream, so a run done with ``np.random.seed(s)`` can be
    reproduced by passing ``rng=s`` instead."""
    spec = dataset(cohort(4, .9, .6))
    np.random.seed(11)
    global_seeded = bootstrapPerf(**spec, num_iterations=20, calcPerfFn=perfDrop)
    explicit = bootstrapPerf(**spec, num_iterations=20, calcPerfFn=perfDrop,
                             rng=11)
    np.testing.assert_allclose(global_seeded, explicit)


def test_an_explicit_rng_leaves_the_global_state_untouched():
    spec = dataset(cohort(4, .9, .6))
    np.random.seed(5)
    expected_next = np.random.random()
    np.random.seed(5)
    bootstrapPerf(**spec, num_iterations=5, calcPerfFn=perfDrop, rng=0)
    assert np.random.random() == expected_next


def test_a_generator_is_accepted_as_the_rng():
    spec = dataset(cohort(4, .9, .6))
    first = bootstrapPerf(**spec, num_iterations=20, calcPerfFn=perfDrop,
                          rng=np.random.default_rng(2))
    repeat = bootstrapPerf(**spec, num_iterations=20, calcPerfFn=perfDrop,
                           rng=np.random.default_rng(2))
    np.testing.assert_allclose(first, repeat)


def test_an_empty_arm_is_left_for_the_statistic_to_handle():
    '''With no opto trials anywhere the opto array is empty, not absent.

    ``bootstrapPerf`` does not guard this; ``optoprocessor`` drops the
    resulting NaNs afterwards. Pinned so the contract stays explicit.
    '''
    seen = []
    np.random.seed(4)
    bootstrapPerf(**dataset({"A": {"A0": [(1., CONTROL)] * 20}}),
                  num_iterations=3, calcPerfFn=recorder(seen))
    for control, opto in seen:
        assert control.shape == (20,)
        assert opto.shape == (0,)


def test_a_session_missing_one_arm_still_contributes_to_the_pool():
    '''Trials are pooled across sessions before the statistic is applied.

    A control-only session is therefore not dropped -- it adds to the control
    arm while other sessions supply the opto arm.
    '''
    spec = {"A": {"A0": [(1., CONTROL)] * 20,
                  "A1": [(1., CONTROL)] * 10 + [(0., OPTO)] * 10}}
    seen = []
    np.random.seed(5)
    bootstrapPerf(**dataset(spec), num_iterations=20, calcPerfFn=recorder(seen))
    assert any(len(opto) for _, opto in seen)
    assert all(len(control) for control, _ in seen)
