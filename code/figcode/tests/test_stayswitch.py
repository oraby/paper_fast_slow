'''Tests for the win/lose stay-switch update arithmetic (Figure S3H).

``_calcUpdate`` compares how often an animal repeated its previous choice
(``Stay``) against how often it would have by chance (``StayBaseline``), and
``_calcGroupUpdate`` wraps it for one bin of the figure -- either for a single
animal or, in the ``n = 9`` many-subjects mode, by averaging one value per
animal.

Two conventions are easy to misread and are pinned explicitly:

- the update is divided by the **number of trials**, so it is in percentage
  *points of trials*, not a percentage change relative to the baseline (the
  alternatives are still visible commented out beside it);
- in many-subjects mode ``nTrials`` holds the **mean trials per animal**, not
  the total, because every column is averaged the same way.

The related, tested-elsewhere implementation is
``behavior.stayswitchupdate.calcWinLoseUpdates``, which computes the same
quantity for Figures 1I-right and S3G and keeps its trial counts.
'''
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ..stayswitch import _calcGroupUpdate, _calcUpdate

WIN = "Win"


def trials(name: str, stay, baseline=0.5, correct=1.) -> pd.DataFrame:
    '''One animal's trials in a single bin of the figure.'''
    stay = list(stay)
    return pd.DataFrame({"Name": name, "Stay": np.asarray(stay, dtype=float),
                         "StayBaseline": [baseline] * len(stay),
                         "ChoiceCorrect": [correct] * len(stay)})


# --------------------------------------------------------------------------
# The update itself
# --------------------------------------------------------------------------

def test_the_update_is_stay_minus_baseline_over_trial_count():
    '''5 stays against a baseline of 4 over 8 trials is +12.5 points.'''
    result = _calcUpdate(trials("A", [1, 1, 1, 0, 0, 0, 1, 1]))
    assert result["StayUpdate"] == pytest.approx(12.5)
    assert result["StayMean"] == pytest.approx(62.5)
    assert result["RefStayMean"] == pytest.approx(50.)


def test_matching_the_baseline_gives_no_update():
    assert _calcUpdate(trials("A", [1, 1, 0, 0]))["StayUpdate"] == pytest.approx(0.)


def test_switching_more_than_chance_gives_a_negative_update():
    assert _calcUpdate(trials("A", [0, 0, 0, 1]))["StayUpdate"] < 0


def test_the_denominator_is_the_trial_count_not_the_baseline():
    '''Doubling the baseline halves nothing -- it shifts the update instead.

    Dividing by ``StayBaseline.sum()`` (one of the commented-out variants)
    would make these two cases equal; dividing by trial count does not.
    '''
    low = _calcUpdate(trials("A", [1, 1, 1, 1], baseline=.25))["StayUpdate"]
    high = _calcUpdate(trials("A", [1, 1, 1, 1], baseline=.5))["StayUpdate"]
    assert low == pytest.approx(75.)
    assert high == pytest.approx(50.)
    assert low - high == pytest.approx(25.)


def test_the_update_does_not_depend_on_how_many_trials_carry_it():
    '''Percentage points scale with rate, not with sample size.'''
    short = _calcUpdate(trials("A", [1, 1, 1, 0]))["StayUpdate"]
    long = _calcUpdate(trials("A", [1, 1, 1, 0] * 25))["StayUpdate"]
    assert short == pytest.approx(long)


def test_performance_and_trial_count_travel_with_the_update():
    result = _calcUpdate(trials("A", [1, 0, 1, 0], correct=1.))
    assert result["Performance"] == pytest.approx(100.)
    assert result["nTrials"] == 4
    assert result["Name"] == "A"


def test_more_than_one_animal_in_a_call_is_refused():
    '''``_calcUpdate`` is per animal; the averaging happens a level up.'''
    mixed = pd.concat([trials("A", [1, 1]), trials("B", [0, 0])])
    with pytest.raises(AssertionError):
        _calcUpdate(mixed)


# --------------------------------------------------------------------------
# One bin, one animal
# --------------------------------------------------------------------------

def test_a_single_animal_bin_reports_its_share_of_that_animal_s_trials():
    row = _calcGroupUpdate(trials("A", [1, 1, 1, 0, 0, 0, 1, 1]),
                           is_many_subjects=False, prev_trial_val=WIN,
                           num_trials=16)
    assert isinstance(row, pd.Series)
    assert row["StayUpdate"] == pytest.approx(12.5)
    assert row["nTrials"] == 8
    assert row["nTrialsPrnct"] == pytest.approx(50.)
    assert row["Description"] == "All"


def test_a_single_animal_bin_needs_the_total_trial_count():
    with pytest.raises(AssertionError, match="num_trials"):
        _calcGroupUpdate(trials("A", [1, 1]), is_many_subjects=False,
                         prev_trial_val=WIN)


def test_trials_with_no_stay_value_are_dropped_from_the_bin():
    df = trials("A", [1, 1, 1, 0])
    df.loc[3, "Stay"] = np.nan
    row = _calcGroupUpdate(df, is_many_subjects=False, prev_trial_val=WIN,
                           num_trials=8)
    assert row["nTrials"] == 3
    assert row["StayUpdate"] == pytest.approx(100 * (3 - 1.5) / 3)


# --------------------------------------------------------------------------
# One bin, many animals
# --------------------------------------------------------------------------

def _manyAnimals(**counts):
    return pd.concat([trials(name, stay) for name, stay in counts.items()],
                     ignore_index=True)


def test_each_animal_contributes_one_value_to_the_average():
    '''A stays every trial (+50), B stays once in four (-25): mean +12.5.'''
    row = _calcGroupUpdate(_manyAnimals(A=[1, 1, 1, 1], B=[1, 0, 0, 0]),
                           is_many_subjects=True, prev_trial_val=WIN,
                           num_trials_subject={"A": 8, "B": 8})
    assert row["StayUpdate"] == pytest.approx(12.5)
    assert row["StayUpdate_SEM"] == pytest.approx(37.5)


def test_an_animal_with_more_trials_does_not_count_for_more():
    '''The same two animals, one of them recorded four times as long.'''
    balanced = _calcGroupUpdate(_manyAnimals(A=[1, 1, 1, 1], B=[1, 0, 0, 0]),
                                is_many_subjects=True, prev_trial_val=WIN,
                                num_trials_subject={"A": 8, "B": 8})
    lopsided = _calcGroupUpdate(_manyAnimals(A=[1, 1, 1, 1] * 4,
                                             B=[1, 0, 0, 0]),
                                is_many_subjects=True, prev_trial_val=WIN,
                                num_trials_subject={"A": 32, "B": 8})
    assert lopsided["StayUpdate"] == pytest.approx(balanced["StayUpdate"])


def test_the_many_animal_trial_count_is_a_per_animal_mean():
    '''Named ``nTrials`` but averaged like every other column -- pinned.'''
    row = _calcGroupUpdate(_manyAnimals(A=[1, 1, 1, 1] * 4, B=[1, 0, 0, 0]),
                           is_many_subjects=True, prev_trial_val=WIN,
                           num_trials_subject={"A": 32, "B": 8})
    assert row["nTrials"] == pytest.approx(10.)      # (16 + 4) / 2, not 20
    assert row["nTrialsPrnct"] == pytest.approx(50.)


def test_a_single_animal_in_many_animal_mode_has_no_error_bar():
    row = _calcGroupUpdate(_manyAnimals(A=[1, 1, 1, 0]), is_many_subjects=True,
                           prev_trial_val=WIN, num_trials_subject={"A": 8})
    assert row["StayUpdate"] == pytest.approx(25.)
    assert np.isnan(row["StayUpdate_SEM"])


def test_many_animal_mode_needs_the_per_animal_totals():
    with pytest.raises(AssertionError, match="num_trials_subject"):
        _calcGroupUpdate(_manyAnimals(A=[1, 1]), is_many_subjects=True,
                         prev_trial_val=WIN)


def test_the_bin_takes_its_description_from_the_group_name():
    df = _manyAnimals(A=[1, 1, 1, 0], B=[1, 1, 0, 0])
    df.name = "Easy"
    assert _calcGroupUpdate(df, is_many_subjects=True, prev_trial_val=WIN,
                            num_trials_subject={"A": 8, "B": 8}
                            )["Description"] == "Easy"


# --------------------------------------------------------------------------
# Pandas 3 readiness
# --------------------------------------------------------------------------

def test_the_many_animal_path_raises_no_pandas_deprecation_warning():
    '''``_calcUpdate`` reads and returns ``Name``, so it cannot be excluded.

    The per-animal loop is written out instead of going through
    ``groupby(...).apply``, which keeps the grouping column available without
    tripping the pandas 3 change.
    '''
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        _calcGroupUpdate(_manyAnimals(A=[1, 1, 1, 0], B=[1, 0, 0, 0]),
                         is_many_subjects=True, prev_trial_val=WIN,
                         num_trials_subject={"A": 8, "B": 8})
