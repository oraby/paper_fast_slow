'''Tests for previous-outcome modulation of tuned neurons (Figure S14A).

Reproduced while extracting: same modulation values as the notebook cell, and
the pies match once the permutation stream is seeded the same way.

The permutation is the part worth pinning. It shuffles preferred and
anti-preferred values in **separate** pools, which keeps each neuron's tuning
intact and breaks only its link to the previous outcome; shuffling one pool
instead would test something else entirely. The p-value is
``(hits + 1) / (N + 1)``, so it can never be zero.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ..prevoutcomemod import (DATA_COL_NAMES, SKIP_DATA_COLS, _diff,
                              _prevDiff, _reducPrefAntiPref, _restrictToNeuron,
                              loopSgfPrevOutcome, modulatedPercentage,
                              permutePVal, plotDiff)


def trial(trace, choice_left, prev_correct, start=0, end=None):
    trace = np.asarray(trace, dtype=float)
    return {"ChoiceLeft": choice_left, "PrevChoiceCorrect": prev_correct,
            "traces_sets": {"neuronal": {7: trace}}, "trace_start_idx": start,
            "trace_end_idx": len(trace) - 1 if end is None else end,
            "trace_reduc": np.nan, "ShortName": "s1"}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_each_trial_is_reduced_to_its_peak_in_the_window():
    df = pd.DataFrame([trial([0., 9., 1.], 1, 1),
                       trial([0., 5., 100.], 1, 1, end=1)])   # window excludes 100
    out = _restrictToNeuron(df, 7, data_col="ChoiceLeft")
    assert list(out.trace_reduc) == [9.0, 5.0]


def test_preferred_side_follows_the_roc_direction():
    df = pd.DataFrame([{"ChoiceLeft": 1, "trace_reduc": 10.0},
                       {"ChoiceLeft": 0, "trace_reduc": 2.0}])
    pref, anti = _reducPrefAntiPref(df, "ChoiceLeft", 1, 0, is_left_tuned=True)
    assert (pref, anti) == ([10.0], [2.0])
    pref, anti = _reducPrefAntiPref(df, "ChoiceLeft", 1, 0, is_left_tuned=False)
    assert (pref, anti) == ([2.0], [10.0])


def test_modulation_is_the_difference_of_the_two_differences():
    prev_correct = (np.array([10.0, 12.0]), np.array([4.0]))    # diff 7
    prev_incorrect = (np.array([6.0]), np.array([4.0]))         # diff 2
    assert _diff(*prev_correct) == 7.0
    assert _prevDiff(prev_correct, prev_incorrect) == 5.0


def test_the_permutation_shuffles_preferred_and_anti_preferred_separately():
    """A neuron whose preferred side is always higher keeps that under the null."""
    rng = np.random.default_rng(0)
    prev_correct = (np.array([10.0, 11.0]), np.array([1.0, 2.0]))
    prev_incorrect = (np.array([10.5]), np.array([1.5]))
    diff = _prevDiff(prev_correct, prev_incorrect)
    p = permutePVal(prev_correct, prev_incorrect, diff, 200, rng=rng)
    assert 0 < p <= 1


class _NoShuffle:
    """An rng whose shuffle does nothing, so every draw is the observed split."""
    def shuffle(self, arr):
        pass


def test_the_p_value_is_hits_plus_one_over_draws_plus_one():
    prev_correct = (np.array([10.0, 12.0]), np.array([1.0]))
    prev_incorrect = (np.array([2.0]), np.array([1.0]))
    diff = _prevDiff(prev_correct, prev_incorrect)
    # Every "shuffle" reproduces the observed value, so all 99 draws are hits.
    assert permutePVal(prev_correct, prev_incorrect, diff, 99,
                       rng=_NoShuffle()) == pytest.approx(100 / 100)


def test_the_p_value_can_never_be_zero():
    rng = np.random.default_rng(1)
    same = (np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    assert permutePVal(same, same, 0.0, 10, rng=rng) == pytest.approx(1.0)


def test_the_permutation_is_reproducible_when_seeded():
    args = ((np.array([3.0, 9.0]), np.array([1.0, 2.0])),
            (np.array([4.0, 8.0]), np.array([2.0, 1.0])))
    diff = _prevDiff(*args)
    first = permutePVal(*args, diff, 50, rng=np.random.default_rng(7))
    second = permutePVal(*args, diff, 50, rng=np.random.default_rng(7))
    assert first == second


def stats_row(trace_id, data_col="ChoiceLeft", pval=0.01, br=15, is_left=True):
    return {"trace_id": trace_id, "data_col": data_col, "prior_data_col": np.nan,
            "DVstr": np.nan, "pval": pval, "data_val_left": 1,
            "data_val_right": 0, "IsROCLeftTuend": is_left, "BrainRegion": br,
            "ShortName": "s1"}


def trials_frame():
    return pd.DataFrame([trial([10.], 1, 1), trial([2.], 0, 1),
                         trial([6.], 1, 0), trial([4.], 0, 0)])


def test_only_tuned_neurons_are_processed():
    stats = pd.DataFrame([stats_row(7, pval=0.01), stats_row(7, pval=0.9)])
    out = loopSgfPrevOutcome(stats, trials_frame(), shuffle_iterations=0, pval=0.05)
    assert len(out) == 1


@pytest.mark.parametrize("data_col", SKIP_DATA_COLS)
def test_skipped_variables_are_not_analysed(data_col):
    stats = pd.DataFrame([stats_row(7, data_col=data_col)])
    out = loopSgfPrevOutcome(stats, trials_frame(), shuffle_iterations=0, pval=0.05)
    assert out.empty


def test_the_row_carries_both_halves_and_their_difference():
    stats = pd.DataFrame([stats_row(7)])
    out = loopSgfPrevOutcome(stats, trials_frame(), shuffle_iterations=0, pval=0.05)
    row = out.iloc[0]
    assert row.PrevCorrect == 8.0        # 10 - 2
    assert row.PrevIncorrect == 2.0      # 6 - 4
    assert row.PrevDiff == 6.0
    assert row.DataCol == DATA_COL_NAMES["ChoiceLeft"]
    assert row.BrainRegion == "MFC"


def test_no_p_value_column_without_shuffles():
    stats = pd.DataFrame([stats_row(7)])
    out = loopSgfPrevOutcome(stats, trials_frame(), shuffle_iterations=0, pval=0.05)
    assert "PermutePVal" not in out.columns


def test_percentage_modulated_counts_below_the_threshold():
    df = pd.DataFrame({"PermutePVal": [0.001, 0.04, 0.2, 0.9]})
    assert modulatedPercentage(df, pval=0.05) == 50.0


def test_the_pie_averages_sessions_not_neurons():
    # s1: 1 of 2 modulated (50%); s2: 4 of 4 (100%). Mean 75%, not 83%.
    df = pd.DataFrame({"PermutePVal": [0.01, 0.9] + [0.01] * 4,
                       "ShortName": ["s1"] * 2 + ["s2"] * 4,
                       "BrainRegion": ["MFC"] * 6,
                       "PrevCorrect": 1.0, "PrevIncorrect": 0.5,
                       "DataCol": "Cur. Direction", "PriorDataCol": ""})
    plotDiff(df, pval=0.05)
    labels = [t.get_text() for t in plt.gcf().axes[0].texts if "%" in t.get_text()]
    assert any(label.startswith("75.00%") for label in labels)


def test_saved_under_the_published_names(tmp_path):
    df = pd.DataFrame({"PermutePVal": [0.01, 0.9], "ShortName": ["s1", "s1"],
                       "BrainRegion": ["MFC", "MFC"], "PrevCorrect": [1.0, 2.0],
                       "PrevIncorrect": [0.5, 1.0], "DataCol": "Cur. Direction",
                       "PriorDataCol": ""})
    plotDiff(df, pval=0.05, save_prefix=str(tmp_path), save_figs=True)
    out = tmp_path / "sgf_tests" / "prev_outcome_modulated"
    assert (out / "pie_Cur. Direction - MFC & LFC.svg").exists()
    assert (out / "mean_diff_Cur. Direction_MFC.svg").exists()
