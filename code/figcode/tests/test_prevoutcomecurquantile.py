'''Tests for the fast/slow previous-outcome panel (Figure 1I-left).

The panel asks whether a mouse's *previous* trial was more often correct when
the current trial is sampled quickly than when it is sampled slowly. Trials
arrive pre-labelled with a sampling-time tertile (``quantile_idx`` 1, 2, 3);
the panel averages ``PrevChoiceCorrect`` **per animal** within each tertile,
draws one grey line per animal, and compares the fast and slow columns with a
paired t-test across animals (the manuscript reports *p* = 0.007, n = 20).

Averaging per animal before testing is the load-bearing step: a trial-level
test would treat 63,702 trials as 63,702 independent observations. The tests
below pin that, plus the middle tertile being carried but not plotted.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
import numpy as np
import pandas as pd
import pytest

from ..prevoutcomecurquantile import (_plotPrevOutcomeCurQuantile,
                                      prevOutcomeCurQuantile,
                                      quantilePrevOutcomeCur)

PREV = "PrevChoiceCorrect"
FAST, TYPICAL, SLOW = 1, 2, 3


def animal(name: str, per_quantile: dict, n: int=20) -> pd.DataFrame:
    '''``{quantile_idx: fraction of previous trials correct}`` for one animal.'''
    rows = []
    for quantile, fraction in per_quantile.items():
        n_correct = round(n * fraction)
        rows.append(pd.DataFrame({
            "Name": name, "Date": "2024-01-01", "SessionNum": 1,
            "quantile_idx": quantile,
            "calcStimulusTime": np.linspace(.3, 2., n),
            PREV: [1.] * n_correct + [0.] * (n - n_correct)}))
    return pd.concat(rows, ignore_index=True)


def cohort(per_animal: dict, n: int=20) -> pd.DataFrame:
    return pd.concat([animal(name, quantiles, n)
                      for name, quantiles in per_animal.items()],
                     ignore_index=True)


def fastBetterThanSlow(n_animals=6, gap=.2, n=20) -> pd.DataFrame:
    '''Every animal's previous trial was more often correct before a fast one.

    The gap varies a little between animals: an exactly constant difference
    makes ``ttest_rel`` divide by a zero standard deviation, which is a
    degenerate input rather than a realistic one.
    '''
    return cohort({f"m{i}": {FAST: .55 + i * .05, TYPICAL: .5 + i * .05,
                             SLOW: .55 + i * .05 - gap + (i % 3 - 1) * .05}
                   for i in range(n_animals)}, n=n)


# --------------------------------------------------------------------------
# The per-animal average
# --------------------------------------------------------------------------

def test_animals_are_averaged_before_the_group_mean_is_taken():
    '''80% and 40% give 60%, whatever the trial counts behind them.'''
    df = cohort({"A": {FAST: .8}, "B": {FAST: .4}})
    mean, sem = quantilePrevOutcomeCur(df, PREV)
    assert mean == pytest.approx(60.)
    assert sem == pytest.approx(20.)


def test_an_animal_with_more_trials_does_not_pull_the_mean():
    '''The difference from a trial-level average, stated as a test.'''
    lopsided = pd.concat([animal("A", {FAST: .8}, n=200),
                          animal("B", {FAST: .4}, n=20)], ignore_index=True)
    assert quantilePrevOutcomeCur(lopsided, PREV)[0] == pytest.approx(60.)
    assert lopsided[PREV].mean() * 100 == pytest.approx(76.36, abs=.01)


def test_a_single_animal_reports_zero_rather_than_a_nan_error_bar():
    '''``Series.sem`` of one value is NaN; the guard turns it into 0.'''
    mean, sem = quantilePrevOutcomeCur(cohort({"A": {FAST: .8}}), PREV)
    assert mean == pytest.approx(80.)
    assert sem == 0


def test_the_result_is_a_percentage_not_a_fraction():
    assert quantilePrevOutcomeCur(cohort({"A": {FAST: .5}}), PREV)[0] == 50.


# --------------------------------------------------------------------------
# The panel
# --------------------------------------------------------------------------

def _panel(df, **kwargs):
    fig, ax = plt.subplots()
    prevOutcomeCurQuantile(df, ax=ax, **kwargs)
    return fig, ax


def _bars(ax):
    '''The bar container -- ax.containers[0] is the error bars.'''
    container, = [c for c in ax.containers if isinstance(c, BarContainer)]
    return container


def test_two_bars_are_drawn_and_the_middle_tertile_is_left_out():
    fig, ax = _panel(fastBetterThanSlow())
    assert len(_bars(ax)) == 2
    assert [label.get_text() for label in ax.get_xticklabels()] == ["Fast", "Slow"]
    plt.close(fig)


def test_the_middle_tertile_can_be_shown():
    fig, ax = plt.subplots()
    df = fastBetterThanSlow()
    _plotPrevOutcomeCurQuantile(df, df[df.quantile_idx == FAST],
                                df[df.quantile_idx == TYPICAL],
                                df[df.quantile_idx == SLOW],
                                col_prev_choice_correct=PREV, ax=ax,
                                plot_q2=True)
    assert len(_bars(ax)) == 3
    assert [label.get_text() for label in ax.get_xticklabels()] == [
        "Fast", "Typical", "Slow"]
    plt.close(fig)


def test_bar_heights_are_the_mean_of_the_per_animal_means():
    df = cohort({"A": {FAST: .8, SLOW: .5}, "B": {FAST: .6, SLOW: .1},
                 "C": {FAST: .7, SLOW: .3}})
    fig, ax = _panel(df)
    fast_bar, slow_bar = _bars(ax)
    assert fast_bar.get_height() == pytest.approx(70.)   # mean of .8 .6 .7
    assert slow_bar.get_height() == pytest.approx(30.)   # mean of .5 .1 .3
    plt.close(fig)


def test_one_grey_line_is_drawn_per_animal():
    fig, ax = _panel(fastBetterThanSlow(n_animals=6))
    grey = [line for line in ax.lines if line.get_color() == "gray"
            and line.get_marker() == "o"]
    assert len(grey) == 6
    for line in grey:
        assert len(line.get_xdata()) == 2      # fast and slow only
    plt.close(fig)


def test_the_animal_count_reaches_the_title_and_the_trial_counts_the_legend():
    fig, ax = _panel(fastBetterThanSlow(n_animals=5, n=20))
    assert "n=5 subjects" in ax.get_title()
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert any("100 Trials" in label for label in labels)   # 5 animals x 20
    plt.close(fig)


# --------------------------------------------------------------------------
# The paired test
# --------------------------------------------------------------------------

def _stars(ax):
    return [child.get_text() for child in ax.texts]


def test_a_consistent_gap_across_animals_is_significant():
    fig, ax = _panel(fastBetterThanSlow(n_animals=8, gap=.25))
    annotation, = _stars(ax)
    assert annotation.startswith("*")
    assert float(annotation.split("pval=")[1]) < 0.05
    plt.close(fig)


def test_no_gap_is_reported_as_not_significant():
    df = cohort({f"m{i}": {FAST: .5 + i * .05,
                           SLOW: .5 + i * .05 + (.05 if i % 2 else -.05)}
                 for i in range(6)})
    fig, ax = _panel(df)
    annotation, = _stars(ax)
    assert annotation.startswith("ns")
    plt.close(fig)


def test_the_test_is_paired_so_a_shared_offset_does_not_hide_the_effect():
    '''Animals differ widely in level but agree in direction.

    An unpaired test on these means would be swamped by the between-animal
    spread; the paired one sees six identical 10-point differences.
    '''
    df = cohort({f"m{i}": {FAST: .40 + i * .09,
                           SLOW: .30 + i * .09 + (i % 3 - 1) * .05}
                 for i in range(6)})
    fig, ax = _panel(df)
    annotation, = _stars(ax)
    assert annotation.startswith("*")
    plt.close(fig)


def test_a_single_star_here_means_p_under_0_05():
    '''This panel stars at 0.05; ``behavior.fastslowperf`` stars at 0.025.

    Two modules annotating panels of the same figure set with different
    cut-offs. Pinned from both sides so the discrepancy is visible rather than
    discovered later; see ``docs/repo-audit.md``.
    '''
    from ...behavior.fastslowperf import STAR_THRESHOLDS

    # Eight animals agreeing weakly: paired t gives p ~ 0.038.
    gaps = [.13, .11, .09, .05, .03, .01, -.03, .01]
    df = cohort({f"m{i}": {FAST: .55 + i * .03, SLOW: .55 + i * .03 - gap}
                 for i, gap in enumerate(gaps)}, n=100)
    fig, ax = _panel(df)
    annotation, = _stars(ax)
    pvalue = float(annotation.split("pval=")[1])
    assert 0.025 < pvalue < 0.05
    assert annotation.startswith("*")
    plt.close(fig)

    # ...but the same p would be "ns" under the other module's thresholds.
    assert [threshold for threshold, star in STAR_THRESHOLDS
            if star == "*"] == [0.025]
    assert not any(pvalue < threshold for threshold, _ in STAR_THRESHOLDS)


# --------------------------------------------------------------------------
# Filtering and saving
# --------------------------------------------------------------------------

def test_trials_without_a_previous_outcome_or_a_sampling_time_are_dropped():
    df = fastBetterThanSlow(n_animals=3)
    df.loc[df.index[:10], PREV] = np.nan
    df.loc[df.index[10:20], "calcStimulusTime"] = np.nan
    fig, ax = _panel(df)
    labels = " ".join(text.get_text() for text in ax.get_legend().get_texts())
    assert f"{len(df) // 3 - 20:,} Trials" in labels
    plt.close(fig)


def test_the_panel_saves_under_the_prefix(tmp_path):
    fig, ax = _panel(fastBetterThanSlow(), save_prefix=tmp_path, save_fig=True)
    assert (tmp_path / "prev_choice_by_quantile.svg").exists()
    plt.close(fig)


def test_saving_without_a_prefix_is_refused():
    fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="save_prefix"):
        prevOutcomeCurQuantile(fastBetterThanSlow(), ax=ax, save_fig=True)
    plt.close(fig)
