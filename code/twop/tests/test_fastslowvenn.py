'''Tests for the fast/slow overlap Venn (Figure 4G).

Reproduced while extracting: both published SVGs are identical to the ones the
notebook cell drew.

What these pin is the arithmetic that is easy to "tidy" into something else:
the denominator is the union of the session's fast and slow trace ids, each
session contributes one percentage regardless of size, and the error bars are
the SEM across sessions.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ...common.definitions import BrainRegion
from ..fastslowvenn import (HATCH_LINEWIDTH, fastSlowOverlap,
                            loopBRPlotVennOverlap, plotVennOverlap,
                            sessionOverlaps)

MFC = int(BrainRegion.M2_Bi)
LFC = int(BrainRegion.ALM_Bi)


def frame(rows, session="s1", br=MFC):
    """rows: {trace_id: prcnt_valid}."""
    return pd.DataFrame([{"trace_id": t, "prcnt_valid": p, "ShortName": session,
                          "BrainRegion": br} for t, p in rows.items()])


def sessionFrames(n_overlap, n_fast, n_slow, n_none=0, session="s1", br=MFC):
    """A session with the given group sizes, as (fast_df, slow_df).

    The drawn percentages need three different means, so the tests below pick
    sizes that keep them apart.
    """
    fast, slow = {}, {}
    for group, (in_fast, in_slow) in [(n_overlap, (1, 1)), (n_fast, (1, 0)),
                                      (n_slow, (0, 1)), (n_none, (0, 0))]:
        for _ in range(group):
            trace_id = f"n{len(fast)}"
            fast[trace_id] = 50 * in_fast
            slow[trace_id] = 50 * in_slow
    return frame(fast, session, br), frame(slow, session, br)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_splits_one_session_into_the_three_groups():
    # 4 neurons: 1 fast-only, 1 slow-only, 1 both, 1 neither.
    fast = frame({"a": 50, "b": 0, "c": 80, "d": 0})
    slow = frame({"a": 0, "b": 50, "c": 80, "d": 0})
    overlap, fast_only, slow_only = fastSlowOverlap(fast, slow, min_prcnt_valid=10)
    assert (overlap, fast_only, slow_only) == (25.0, 25.0, 25.0)


def test_the_denominator_counts_neurons_active_in_neither():
    fast = frame({"a": 50, "b": 0})
    slow = frame({"a": 50, "b": 0})
    overlap, _, _ = fastSlowOverlap(fast, slow, min_prcnt_valid=10)
    assert overlap == 50.0       # not 100%: "b" is in the denominator


def test_the_denominator_is_the_union_of_both_frames():
    fast = frame({"a": 50})
    slow = frame({"a": 50, "b": 0})
    assert fastSlowOverlap(fast, slow, 10)[0] == 50.0


def test_threshold_is_inclusive():
    fast = frame({"a": 10})
    slow = frame({"a": 9.99})
    overlap, fast_only, slow_only = fastSlowOverlap(fast, slow, min_prcnt_valid=10)
    assert (overlap, fast_only, slow_only) == (0.0, 100.0, 0.0)


def test_a_duplicated_trace_id_is_rejected():
    dup = pd.concat([frame({"a": 50}), frame({"a": 50})])
    with pytest.raises(AssertionError):
        fastSlowOverlap(dup, frame({"a": 50}), 10)


def test_each_session_counts_once_however_many_neurons_it_has():
    # s1: 1 of 2 neurons overlap (50%). s2: 8 of 8 (100%). Mean is 75%, not the
    # 90% you would get by pooling the neurons.
    fast = pd.concat([frame({"a": 50, "b": 0}, session="s1"),
                      frame({f"n{i}": 50 for i in range(8)}, session="s2")])
    slow = fast.copy()
    overlaps = sessionOverlaps(fast, slow, ["s1", "s2"], min_prcnt_valid=10)
    assert list(overlaps.overlap) == [50.0, 100.0]
    assert overlaps.overlap.mean() == 75.0


def test_error_bars_are_the_sem_across_sessions():
    fast = pd.concat([frame({"a": 50, "b": 0}, session="s1"),
                      frame({"a": 50, "b": 50}, session="s2")])
    overlaps = sessionOverlaps(fast, fast.copy(), ["s1", "s2"], 10)
    assert overlaps.overlap.sem() == pytest.approx(25.0)


def test_labels_carry_the_percentage_and_its_sem():
    # s1 overlaps 1 of 6 (16.67%), s2 overlaps 2 of 6 (33.33%): mean 25%,
    # SEM 8.33%.
    fast1, slow1 = sessionFrames(1, 2, 3, session="s1")
    fast2, slow2 = sessionFrames(2, 2, 2, session="s2")
    plotVennOverlap(pd.concat([fast1, fast2]), pd.concat([slow1, slow2]),
                    10, ["s1", "s2"], "MFC", save_figs=False)
    labels = [t.get_text() for t in plt.gca().texts if "%" in t.get_text()]
    assert any(label.startswith("25.00%") and "8.33%" in label for label in labels)


def test_the_hatch_width_is_restored_after_drawing():
    before = mpl.rcParams["hatch.linewidth"]
    assert before != HATCH_LINEWIDTH
    fast, slow = sessionFrames(1, 2, 3)
    plotVennOverlap(fast, slow, 10, ["s1"], "MFC", save_figs=False)
    assert mpl.rcParams["hatch.linewidth"] == before


def test_identical_subsets_are_refused_rather_than_mislabelled():
    """Equal means would make the SEM lookup ambiguous."""
    fast = frame({"a": 50, "b": 0, "c": 50, "d": 0})
    slow = frame({"a": 50, "b": 50, "c": 0, "d": 0})   # all three are 25%
    with pytest.raises(AssertionError, match="share a mean"):
        plotVennOverlap(fast, slow, 10, ["s1"], "MFC", save_figs=False)


def test_one_venn_per_region_named_as_the_files_are():
    mfc_fast, mfc_slow = sessionFrames(1, 2, 3, br=MFC)
    lfc_fast, lfc_slow = sessionFrames(3, 2, 1, br=LFC)
    all_df = pd.concat([mfc_fast, lfc_fast])       # only its sessions are used
    res = loopBRPlotVennOverlap(all_df, pd.concat([mfc_fast, lfc_fast]),
                                pd.concat([mfc_slow, lfc_slow]), 10,
                                save_figs=False)
    assert sorted(res) == ["ALM", "M2"]
    assert res["M2"].loc["s1", "slow_minus_fast"] == pytest.approx(50.0)
    assert res["ALM"].loc["s1", "slow_minus_fast"] == pytest.approx(100 / 6)


def test_saving_needs_a_prefix():
    fast, slow = sessionFrames(1, 2, 3)
    with pytest.raises(AssertionError, match="fig_save_prefix"):
        plotVennOverlap(fast, slow, 10, ["s1"], "MFC", save_figs=True)


def test_saved_under_the_published_name(tmp_path):
    fast, slow = sessionFrames(1, 2, 3)
    plotVennOverlap(fast, slow, 10, ["s1"], "M2", save_figs=True,
                    fig_save_prefix=str(tmp_path))
    assert (tmp_path / "FastSlowVenn" / "valid_10%_M2.svg").exists()
