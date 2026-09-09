'''Tests for the per-animal sampling-time normalisation used across figcode.

``normalizeSTAcrossSubjects`` z-scores ``calcStimulusTime`` and every
``PrevCalcStimulusTime*`` column **within each animal**, writing the result to
``transformed<Column>`` and leaving the originals alone. Figures S3E and S3G
read the transformed columns; the whole point of doing it per animal is that
a naturally slow mouse must not read as "slow trial" everywhere.

That is the same normalisation question as :mod:`behavior.stdistribution`,
which pools each human across both contexts before splitting. The rule here is
simpler because a mouse has only one context -- see that module's docstring.

The z-score itself is ``scipy.stats.zscore(..., nan_policy="omit")``, so the
population (ddof=0) standard deviation is used and NaNs survive as NaN rather
than poisoning the animal's mean.
'''
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import zscore

from ..util import normalizeSTAcrossSubjects

ST = "calcStimulusTime"
PREV1, PREV2 = "PrevCalcStimulusTime1", "PrevCalcStimulusTime2"


def frame(per_subject: dict, extra: dict=None) -> pd.DataFrame:
    '''``{name: [sampling times]}`` with the previous-trial columns derived.'''
    rows = []
    for name, times in per_subject.items():
        times = np.asarray(times, dtype=float)
        rows.append(pd.DataFrame({"Name": name, ST: times,
                                  PREV1: times + 1, PREV2: times + 2,
                                  **(extra or {})}))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# What it produces
# --------------------------------------------------------------------------

def test_a_transformed_column_is_added_per_sampling_time_column():
    out = normalizeSTAcrossSubjects(frame({"A": [1., 2, 3, 4]}))
    added = [col for col in out.columns if col.startswith("transformed")]
    assert sorted(added) == ["transformedCalcStimulusTime",
                             "transformedPrevCalcStimulusTime1",
                             "transformedPrevCalcStimulusTime2"]


def test_the_original_columns_are_left_untouched():
    df = frame({"A": [1., 2, 3, 4]})
    before = df.copy()
    normalizeSTAcrossSubjects(df)
    pd.testing.assert_frame_equal(df, before)


def test_the_row_order_and_index_survive():
    df = frame({"A": [1., 2, 3], "B": [10., 20, 30]})
    out = normalizeSTAcrossSubjects(df)
    assert list(out.index) == list(df.index)
    assert out.Name.tolist() == df.Name.tolist()
    assert len(out) == len(df)


def test_columns_without_the_prefix_are_ignored():
    df = frame({"A": [1., 2, 3, 4]}, extra={"Untouched": 7.})
    out = normalizeSTAcrossSubjects(df)
    assert "transformedUntouched" not in out.columns
    assert (out.Untouched == 7.).all()


# --------------------------------------------------------------------------
# Per animal, not across the cohort
# --------------------------------------------------------------------------

def test_each_animal_is_centred_on_its_own_mean():
    out = normalizeSTAcrossSubjects(frame({"Fast": [1., 2, 3, 4],
                                           "Slow": [10., 20, 30, 40]}))
    for name in ("Fast", "Slow"):
        values = out[out.Name == name].transformedCalcStimulusTime
        assert values.mean() == pytest.approx(0., abs=1e-12)
        assert values.std(ddof=0) == pytest.approx(1.)


def test_a_uniformly_slow_animal_does_not_read_as_slow_trials():
    '''The reason the normalisation is per animal at all.

    Both animals have the same shape, ten-fold apart in absolute time. After
    normalising they must be indistinguishable.
    '''
    out = normalizeSTAcrossSubjects(frame({"Fast": [1., 2, 3, 4],
                                           "Slow": [10., 20, 30, 40]}))
    fast = out[out.Name == "Fast"].transformedCalcStimulusTime.values
    slow = out[out.Name == "Slow"].transformedCalcStimulusTime.values
    np.testing.assert_allclose(fast, slow)


def test_pooling_the_animals_instead_would_change_the_answer():
    '''Guards the grouping key: a cohort-wide z-score is not the same thing.'''
    df = frame({"Fast": [1., 2, 3, 4], "Slow": [10., 20, 30, 40]})
    per_animal = normalizeSTAcrossSubjects(df).transformedCalcStimulusTime
    pooled = zscore(df[ST].values)
    assert not np.allclose(per_animal.values, pooled)


def test_each_previous_trial_column_is_normalised_on_its_own():
    '''Not on the current-trial column: each gets its own mean and spread.'''
    out = normalizeSTAcrossSubjects(frame({"A": [1., 2, 3, 4]}))
    for col in ("transformedPrevCalcStimulusTime1",
                "transformedPrevCalcStimulusTime2"):
        assert out[col].mean() == pytest.approx(0., abs=1e-12)
        assert out[col].std(ddof=0) == pytest.approx(1.)


def test_a_single_animal_frame_still_gets_normalised():
    out = normalizeSTAcrossSubjects(frame({"A": [1., 2, 3, 4, 5]}))
    assert out.transformedCalcStimulusTime.std(ddof=0) == pytest.approx(1.)


# --------------------------------------------------------------------------
# Missing values
# --------------------------------------------------------------------------

def test_a_missing_time_stays_missing_and_does_not_poison_the_animal():
    df = frame({"A": [1., 2, 3, 4], "B": [1., 2, 3, 4]})
    df.loc[0, ST] = np.nan
    out = normalizeSTAcrossSubjects(df)
    assert np.isnan(out.loc[0, "transformedCalcStimulusTime"])
    rest = out.loc[1:3, "transformedCalcStimulusTime"]
    assert np.isfinite(rest).all()
    assert rest.mean() == pytest.approx(0., abs=1e-12)


def test_an_animal_with_one_trial_yields_no_usable_z_score():
    '''Zero spread -- pinned so the NaN is expected rather than alarming.'''
    out = normalizeSTAcrossSubjects(frame({"A": [1.], "B": [1., 2, 3]}))
    assert out[out.Name == "A"].transformedCalcStimulusTime.isna().all()
    assert out[out.Name == "B"].transformedCalcStimulusTime.notna().all()


# --------------------------------------------------------------------------
# Pandas 3 readiness
# --------------------------------------------------------------------------

def test_no_deprecation_warning_is_raised():
    '''``groupby(...).apply`` used to warn about operating on ``Name``.

    The callback needs ``Name`` neither to compute nor to return -- it only
    reads the sampling-time columns -- so excluding the grouping column keeps
    the numbers identical. This test is what makes that safe to rely on.
    '''
    df = frame({"A": [1., 2, 3, 4], "B": [5., 6, 7, 8]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        normalizeSTAcrossSubjects(df)


def test_the_grouping_column_comes_back_in_the_result():
    '''Excluding ``Name`` from the callback must not drop it from the output.'''
    out = normalizeSTAcrossSubjects(frame({"A": [1., 2, 3], "B": [4., 5, 6]}))
    assert "Name" in out.columns
    assert out.Name.tolist() == ["A"] * 3 + ["B"] * 3
