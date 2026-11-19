'''Tests for the sampling-start choice-share bars (Figure S12C).

Pinned here: the window is sampling's first 0.1 s at the data's rate, the test
is the paper's (peak activity, Mann-Whitney) split by quantile, the share
counts only the plain current-choice rows, and a bar is the mean of the
session percentages +/- their SEM.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from .. import samplingstartchoice as ssc
from ..dimreduc import DimReduc
from ..fastslowstats import splitDatTestQuantiles
from ..statstest import StatsTest

M2, ALM = int(ssc.REGIONS[0][0]), int(ssc.REGIONS[1][0])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_window_is_a_tenth_of_a_second():
    assert ssc.windowSamples(pd.DataFrame({"acq_sampling_rate": [30.043] * 2})) == 3
    assert ssc.windowSamples(pd.DataFrame({"acq_sampling_rate": [60.0]})) == 6


def test_tests_run_the_papers_choice_test_on_the_window(monkeypatch):
    names = ["-0.1s Sampling", "Sampling", "Movement to Lateral Port"]
    src = pd.DataFrame({
        "acq_sampling_rate": [30.0], "epoch": ["-0.1s Sampling"],
        "epochs_names": [names], "epochs_ranges": [[(0, 2), (3, 9), (10, 12)]],
        "trace_start_idx": [0], "trace_end_idx": [12],
        "traces_sets": [{"neuronal": {1: np.arange(13.0)}}]})
    calls = {}
    monkeypatch.setattr(ssc, "runPreprocessor",
                        lambda df, dimReducFn: calls.update(df=df, dr=dimReducFn) or "RED")
    monkeypatch.setattr(ssc, "runSplitFn",
                        lambda fn, red, st: calls.update(fn=fn, red=red, st=st) or "RES")
    assert ssc.samplingStartChoiceTests(src) == "RES"
    np.testing.assert_array_equal(calls["df"].traces_sets.iloc[0]["neuronal"][1], [3, 4, 5])
    assert calls["dr"] is DimReduc.max and calls["st"] is StatsTest.mannwhitneyu
    assert calls["fn"] is splitDatTestQuantiles and calls["red"] == "RED"


def tests_frame():
    rows = []
    # Session s1 (M2): 4 neurons, 1 significant in q1; s2 (M2): 2 neurons, 2 in q1
    for sess, n, k in [("s1", 4, 1), ("s2", 2, 2)]:
        for i in range(n):
            rows.append(dict(ShortName=sess, trace_id=i, BrainRegion=M2, quantile_idx=1,
                             data_col="ChoiceLeft", prior_data_col=None, DVstr=None,
                             pval=0.01 if i < k else 0.5))
    # Rows that must not count
    rows.append(dict(rows[0], DVstr="Easy", pval=0.5))
    rows.append(dict(rows[0], prior_data_col="PrevChoiceLeft", pval=0.5))
    rows.append(dict(rows[0], data_col="PrevChoiceLeft", pval=0.5))
    return pd.DataFrame(rows)


def test_share_counts_plain_choice_rows_per_session():
    share = ssc.choiceShareBySession(tests_frame(), pval=0.05)
    assert share[["ShortName", "n_neurons", "n_sgf", "prcnt"]].values.tolist() == \
        [["s1", 4, 1, 25.0], ["s2", 2, 2, 100.0]]


def test_bar_is_the_mean_of_sessions_with_their_sem():
    summary = ssc.summarise(ssc.choiceShareBySession(tests_frame(), pval=0.05))
    row = summary.iloc[0]
    assert row["mean"] == pytest.approx(62.5)              # not pooled 3/6 = 50
    assert row["sem"] == pytest.approx(np.std([25, 100], ddof=1)/np.sqrt(2))
    assert (row.n_sessions, row.n_neurons, row.n_sgf) == (2, 6, 3)


def test_plot_draws_and_saves_only_when_asked(tmp_path):
    share = pd.DataFrame({"BrainRegion": [M2, M2, ALM, M2], "quantile_idx": [1, 1, 1, 3],
                          "ShortName": ["a", "b", "c", "a"], "n_neurons": [10] * 4,
                          "n_sgf": [2, 4, 1, 1], "prcnt": [20.0, 40.0, 10.0, 10.0]})
    ssc.plotChoiceShare(share, save_figs=False, fig_save_prefix=str(tmp_path))
    ax = plt.gcf().axes[0]
    assert sorted(p.get_height() for p in ax.patches) == [10.0, 10.0, 30.0]
    assert [t.get_color() for t in ax.get_xticklabels()] == ["red", "yellow"]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == \
        ["MFC (2 sessions)", "LFC (1 sessions)"]
    assert not (tmp_path / ssc.SAVE_DIR).exists()
    ssc.plotChoiceShare(share, save_figs=True, fig_save_prefix=str(tmp_path))
    assert (tmp_path / ssc.SAVE_DIR / ssc.SAVE_NAME).is_file()
