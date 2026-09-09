'''Tests for fast-vs-slow accuracy on easy trials (Figure S2M).

The panel's argument depends on three things being right: "easy" is defined
per species (humans by coherence, mice by the upstream difficulty label), the
t-test is *paired* within subject, and the three contexts are corrected
together rather than reported raw. Each is pinned here.

The human roster is also pinned. Both human contexts are iterated over the
same subject list so the accuracy and speed panels describe the same people;
taking each frame's own ``Name`` values would silently compare different
groups if a participant were missing from one context.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ..fastslowperf import (FAST, HUMAN_EASY_ABS_DV, MICE_EASY_LABEL, SLOW,
                            collectGroup, compareGroups, easyHumanTrials,
                            easyMiceTrials, fastSlowAccuracy,
                            plotFastSlowAccuracy, significanceLabel)


def _subject(name, fast_correct, slow_correct, n=50, dv=0.9, dvstr="Easy"):
    '''One subject with prescribed fast/slow accuracy (as fractions).'''
    rows = []
    for idx, correct_rate in ((FAST, fast_correct), (SLOW, slow_correct)):
        n_correct = int(round(n * correct_rate))
        rows.append(pd.DataFrame({
            "Name": name, "quantile_idx": idx, "DV": dv, "DVstr": dvstr,
            "ChoiceCorrect": np.concatenate([np.ones(n_correct),
                                             np.zeros(n - n_correct)]),
        }))
    return pd.concat(rows, ignore_index=True)


def _cohort(spec, **kwargs):
    return pd.concat([_subject(name, fast, slow, **kwargs)
                      for name, (fast, slow) in spec.items()],
                     ignore_index=True)


# --------------------------------------------------------------------------
# "Easy" is defined per species
# --------------------------------------------------------------------------

def test_human_easy_uses_absolute_coherence():
    df = pd.DataFrame({"DV": [-0.9, -0.2, -0.1, 0.0, 0.14, 0.15, 0.8]})
    kept = easyHumanTrials(df)
    assert list(kept.DV) == [-0.9, -0.2, 0.15, 0.8]
    assert HUMAN_EASY_ABS_DV == 0.15


def test_human_easy_boundary_is_inclusive():
    df = pd.DataFrame({"DV": [HUMAN_EASY_ABS_DV]})
    assert len(easyHumanTrials(df)) == 1


def test_mice_easy_uses_the_upstream_label_not_coherence():
    '''Mice are labelled upstream, so a high DV alone does not qualify.'''
    df = pd.DataFrame({"DV": [0.99, 0.99], "DVstr": ["Easy", "Med"]})
    kept = easyMiceTrials(df)
    assert list(kept.DVstr) == [MICE_EASY_LABEL]


# --------------------------------------------------------------------------
# Per-subject accuracies
# --------------------------------------------------------------------------

def test_accuracy_is_percent_correct_per_tertile():
    fast, slow = fastSlowAccuracy(_subject("M1", 0.8, 0.9, n=100))
    assert fast == pytest.approx(80.0)
    assert slow == pytest.approx(90.0)


def test_trials_without_a_choice_are_ignored():
    df = _subject("M1", 0.8, 0.9, n=100)
    missing = df.head(10).copy()
    missing["ChoiceCorrect"] = np.nan
    fast, _ = fastSlowAccuracy(pd.concat([df, missing], ignore_index=True))
    assert fast == pytest.approx(80.0)


def test_collect_group_follows_the_given_roster():
    '''Both human contexts must describe the same people.'''
    df = _cohort({"S1": (0.8, 0.9), "S2": (0.7, 0.8), "S3": (0.6, 0.7)})
    fast, slow = collectGroup(df, subjects=["S3", "S1"])
    assert fast == pytest.approx([60.0, 80.0])
    assert slow == pytest.approx([70.0, 90.0])


def test_collect_group_rejects_a_subject_with_no_trials():
    df = _cohort({"S1": (0.8, 0.9)})
    with pytest.raises(ValueError, match="no trials"):
        collectGroup(df, subjects=["S1", "Missing"])


# --------------------------------------------------------------------------
# The statistics
# --------------------------------------------------------------------------

def test_test_is_paired_within_subject():
    '''A constant within-subject gap is significant however spread the means.

    An unpaired test on these data would not be: the between-subject spread
    swamps the 5-point gap.
    '''
    # Gaps vary slightly so the paired differences are not all identical,
    # which scipy warns about as catastrophic cancellation.
    spec = {f"S{i}": (0.50 + i * 0.04, 0.55 + i * 0.04 + i * 0.002)
            for i in range(8)}
    groups = {"G": collectGroup(_cohort(spec, n=100))}
    result = compareGroups(groups)[0]
    assert result.pvalue < 1e-6


def test_correction_is_applied_across_the_three_contexts():
    spec = {f"S{i}": (0.60 + i * 0.01, 0.70 + i * 0.01 + i * 0.002)
            for i in range(10)}
    groups = {name: collectGroup(_cohort(spec, n=100))
              for name in ("A", "B", "C")}
    results = compareGroups(groups)
    assert [r.label for r in results] == ["A", "B", "C"]
    for r in results:
        assert r.pvalue_corrected >= r.pvalue
    # Holm scales the smallest p by the number of tests.
    assert results[0].pvalue_corrected == pytest.approx(
        min(3 * results[0].pvalue, 1.0))


def test_group_result_reports_its_subject_count():
    spec = {f"S{i}": (0.6, 0.7 + i * 0.002) for i in range(7)}
    result = compareGroups({"G": collectGroup(_cohort(spec))})[0]
    assert result.n_subjects == 7


def test_single_star_threshold_is_the_documented_0_025():
    '''Carried over from the notebook; elsewhere the repo uses 0.05.'''
    assert significanceLabel(0.0005) == "***"
    assert significanceLabel(0.005) == "**"
    assert significanceLabel(0.02) == "*"
    assert significanceLabel(0.03) == "ns"


# --------------------------------------------------------------------------
# The panel
# --------------------------------------------------------------------------

def _threeResults():
    spec = {f"S{i}": (0.6 + i * 0.01, 0.7 + i * 0.01 + i * 0.002)
            for i in range(5)}
    return compareGroups({name: collectGroup(_cohort(spec))
                          for name in ("Humans Accuracy",
                                       "Humans Max Outcome", "Mice")})


def test_panel_draws_a_paired_line_per_subject_per_group():
    results = _threeResults()
    fig = plotFastSlowAccuracy(results)
    paired = [ln for ln in fig.axes[0].lines if len(ln.get_xdata()) == 2]
    assert len(paired) == sum(r.n_subjects for r in results)
    plt.close(fig)


def test_panel_labels_each_group_with_its_subject_count():
    fig = plotFastSlowAccuracy(_threeResults())
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert all("(n=5)" in lbl for lbl in labels)
    plt.close(fig)


def test_panel_saves_under_the_prefix(tmp_path):
    fig = plotFastSlowAccuracy(_threeResults(), save_prefix=tmp_path,
                               save_figs=True)
    assert (tmp_path / "slow_fast_perf.svg").exists()
    plt.close(fig)


def test_panel_refuses_to_save_without_a_prefix():
    with pytest.raises(ValueError, match="save_prefix"):
        plotFastSlowAccuracy(_threeResults(), save_figs=True)
