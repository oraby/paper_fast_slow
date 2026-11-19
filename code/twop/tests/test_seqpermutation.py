'''Tests for the sequence permutation tests (Figures S9G and 4J).

What these pin:

- the three per-neuron sequence tables are joined on ``trace_id``, after
  dropping neurons active on too few trials;
- the S9G statistic scores the observed pairing first and the shuffles after,
  so an identical sequence costs nothing and a reversed one the most;
- the tests run once: drawing reads the stored results and never re-runs them
  (they are unseeded, so a second run would draw different numbers);
- a figure cell can draw one session, saving to the path the loop uses, and
  asking for a session that has no test raises instead of drawing nothing.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ...common.definitions import BrainRegion
from .. import seqpermutation as sp

MFC = int(BrainRegion.M2_Bi)
LFC = int(BrainRegion.ALM_Bi)
N = 99      # permutations; the paper's 100,000 would only slow the suite


@pytest.fixture(autouse=True)
def closeFigures():
    yield
    plt.close("all")


def table(order, session="s1", br=MFC, prcnt_valid=50, jitter=0):
    '''One row per neuron; neuron ``trace_id`` peaks around sample
    ``10 * order[trace_id]``.'''
    rng = np.random.default_rng(len(order) + jitter)
    return pd.DataFrame([
        dict(trace_id=f"{session}_{i}", ShortName=session, BrainRegion=br,
             prcnt_valid=prcnt_valid,
             max_idxs=list(10 * pos + rng.integers(-1, 2, size=7)))
        for i, pos in enumerate(order)])


def three(order_fast, order_slow, session="s1", br=MFC, n=12):
    typical = table(range(n), session, br)
    return (typical, table(order_fast, session, br, jitter=1),
            table(order_slow, session, br, jitter=2))


def sessions():
    n = 12
    parts = [three(range(n), range(n)[::-1], "s1", MFC),
             three(range(n), range(n), "s2", MFC),
             three(range(n)[::-1], range(n), "s3", LFC)]
    return [pd.concat(frames, ignore_index=True) for frames in zip(*parts)]


def test_neurons_below_the_activity_floor_are_left_out():
    typical, fast, slow = three(range(6), range(6))
    fast.loc[0, "prcnt_valid"] = 5
    ok, df = sp.combDfsSortToRef(typical, fast, slow, 10, ref_name="typical")
    assert ok
    assert "s1_0" not in set(df.trace_id)
    assert len(df) == 5
    assert {"med_typical", "med_impulsive", "med_deliberate"} <= set(df.columns)


def test_an_empty_table_is_reported_not_joined():
    typical, fast, slow = three(range(6), range(6))
    ok, _ = sp.combDfsSortToRef(typical, fast.iloc[:0], slow, 10, "typical")
    assert not ok


def test_the_first_call_scores_the_observed_pairing():
    other = np.array([3., 2., 1.])
    penalty = sp.Penality(other)
    ref = np.array([1., 2., 3.])
    assert penalty._calcPenality(ref, ref) == pytest.approx(4 / 3)
    assert penalty._calcPenality(ref, ref) == 0


def test_a_reversed_sequence_costs_more_than_a_kept_one():
    typical, fast, slow = three(range(12), range(12)[::-1])
    summary, _ = sp.typicalPenaltyTests(typical, fast, slow, prcnt_valid=10,
                                        normalized=False, n_resamples=N)
    assert summary.impulsive_stat.iloc[0] == 0
    assert summary.deliberate_stat.iloc[0] > summary.impulsive_stat.iloc[0]


def test_one_summary_row_and_one_test_per_session():
    summary, tests = sp.typicalPenaltyTests(*sessions(), prcnt_valid=10,
                                            normalized=True, by_sess=True,
                                            n_resamples=N)
    # Regions come in code order, LFC (ALM) before MFC (M2).
    assert list(summary.ShortName) == ["s3", "s1", "s2"]
    assert list(summary.BrainRegion) == ["ALM", "M2", "M2"]
    assert [t["ShortName"] for t in tests] == ["s3", "s1", "s2"]
    assert len(tests[0]["typical"].null_distribution) == N
    assert set(summary.test_name) == {"Permutation"}


def test_identical_fast_and_slow_sequences_do_not_deviate():
    summary, tests = sp.deviationTests(*sessions(), prcnt_valid=10,
                                       by_sess=True, n_resamples=N)
    by_sess = dict(zip(summary.ShortName, summary.deviation_stat))
    assert by_sess["s2"].statistic == 0
    assert by_sess["s1"].statistic > 0
    assert {t["ShortName"]: t["deviation"] for t in tests}["s2"] is by_sess["s2"]


def test_drawing_does_not_rerun_the_tests(monkeypatch):
    _, penalty_tests = sp.typicalPenaltyTests(*sessions(), prcnt_valid=10,
                                              normalized=True, by_sess=True,
                                              n_resamples=N)
    _, deviation_tests = sp.deviationTests(*sessions(), prcnt_valid=10,
                                           by_sess=True, n_resamples=N)

    def rerun(*args, **kwargs):
        raise AssertionError("a drawing re-ran a permutation test")
    monkeypatch.setattr(sp.stats, "permutation_test", rerun)
    sp.plotPenaltyTests(penalty_tests, save_figs=False, show=False)
    sp.plotDeviationTests(deviation_tests, save_figs=False, show=False)


def test_the_example_saves_where_the_loop_does(tmp_path):
    _, tests = sp.deviationTests(*sessions(), prcnt_valid=10, by_sess=True,
                                 n_resamples=N)
    sp.plotDeviationTests(tests, save_figs=True, fig_save_prefix=tmp_path,
                          sessions=["s3"], show=False)
    assert [p.name for p in (tmp_path / "Sequence").iterdir()] == \
        ["deviation_rank_ALM - s3_10_prcnt_active.svg"]


def test_the_penalty_histograms_keep_their_file_names(tmp_path):
    _, tests = sp.typicalPenaltyTests(*sessions(), prcnt_valid=10,
                                      normalized=True, by_sess=True,
                                      n_resamples=N)
    sp.plotPenaltyTests(tests, save_figs=True, fig_save_prefix=tmp_path,
                        show=False)
    assert sorted(p.name for p in (tmp_path / "Sequence").iterdir()) == [
        f"seq_rank_{br} - {s}_Permutation_ Z-Scored - _10_prcnt_active.svg"
        for br, s in [("ALM", "s3"), ("M2", "s1"), ("M2", "s2")]]


def test_an_unknown_session_raises():
    _, tests = sp.deviationTests(*sessions(), prcnt_valid=10, by_sess=True,
                                 n_resamples=N)
    with pytest.raises(KeyError, match="s9"):
        sp.plotDeviationTests(tests, save_figs=False, sessions=["s9"])
