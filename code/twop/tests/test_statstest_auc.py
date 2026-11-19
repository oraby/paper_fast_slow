'''Tests for the ROC-AUC test behind the choice-selectivity runs.

``StatsTest.AUCPermutations`` feeds every ``data_runs_*.pkl`` rebuild
(``genrundata.genRunDataInst``), so it must run on the NumPy the lockfile pins.
'''
from __future__ import annotations

import numpy as np
import pytest

from ..statstest import StatsTest


def test_identical_samples_give_half():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    res = StatsTest.AUCPermutations(x, x.copy(), skip_permutations=True)
    assert res.resultDict()["statistic"] == pytest.approx(0.5)


def test_separated_samples_give_zero_or_one():
    low, high = np.array([1.0, 2.0, 3.0]), np.array([10.0, 11.0, 12.0])
    a = StatsTest.AUCPermutations(low, high, skip_permutations=True).resultDict()
    b = StatsTest.AUCPermutations(high, low, skip_permutations=True).resultDict()
    assert {round(a["statistic"], 6), round(b["statistic"], 6)} == {0.0, 1.0}
    assert np.isnan(a["pval"])


def test_permutation_path_returns_a_pvalue():
    rng = np.random.default_rng(0)
    low, high = rng.normal(0, 1, 30), rng.normal(3, 1, 30)
    res = StatsTest.AUCPermutations(
        low, high, permutation_test_kargs=dict(n_resamples=200, random_state=0)
    ).resultDict()
    assert 0 < res["pval"] < 0.05
    assert (res["num_samples_left"], res["num_samples_right"]) == (30, 30)
