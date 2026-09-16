'''Tests for the population choice decoder (Figure S12G).

Reproduced while extracting: the notebook's own table is cached in
``data/2p/svm_df.pkl`` and loaded rather than re-run, so what is checked here is
the decoder itself on data whose answer is known.

Pinned: one row per random split (so the panel's spread is across splits, not
sessions), each trial reduced to its neurons' peaks within the row's window,
and sessions kept apart — a decoder never sees two sessions' neurons, because
they were never recorded together.
'''
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ...common.definitions import BrainRegion
from ..decoders import DEFAULT_NUM_RUNS, _processDF, _runSession

MFC = int(BrainRegion.M2_Bi)


def trials(n=40, n_neurons=3, separable=True, session="s1", seed=0,
           window=(0, 9), spike_at=7):
    """Half the trials are left choices; separable ones encode that in the trace."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        left = float(i % 2)
        traces = {}
        for neuron in range(n_neurons):
            trace = rng.normal(0, 0.1, 10)
            if separable:
                trace[spike_at] += 10.0 if left else 1.0
            else:
                trace[spike_at] += 5.0
            traces[neuron] = trace
        rows.append({"ChoiceLeft": left, "traces_sets": {"neuronal": traces},
                     "trace_start_idx": window[0], "trace_end_idx": window[1],
                     "ShortName": session, "BrainRegion": MFC})
    return pd.DataFrame(rows)


def test_one_row_per_random_split():
    out = _processDF(MFC, "s1", trials(), num_runs=5)
    assert len(out) == 5
    assert list(out.iteration_idx) == [1, 2, 3, 4, 5]


def test_a_separable_choice_is_decoded():
    out = _processDF(MFC, "s1", trials(separable=True), num_runs=5)
    assert out.accuracy.mean() > 0.9


def test_an_unrelated_population_decodes_at_chance():
    out = _processDF(MFC, "s1", trials(separable=False), num_runs=20, )
    assert 0.3 < out.accuracy.mean() < 0.7


def test_only_the_rows_window_is_read():
    """The spike sits outside the window, so there is nothing to decode."""
    out = _processDF(MFC, "s1", trials(separable=True, window=(0, 4), spike_at=7),
                     num_runs=20)
    assert 0.3 < out.accuracy.mean() < 0.7


def test_trials_without_a_choice_are_dropped():
    df = trials(n=40)
    df.loc[df.index[:10], "ChoiceLeft"] = np.nan
    out = _processDF(MFC, "s1", df, num_runs=3)
    assert out is not None            # the remaining 30 still decode


def test_a_session_with_no_neurons_returns_nothing():
    df = trials(n=4)
    df["traces_sets"] = [{"neuronal": {}} for _ in range(len(df))]
    assert _processDF(MFC, "s1", df, num_runs=3) is None


def test_the_session_and_region_are_named_on_every_row():
    out = _processDF(MFC, "sess_a", trials(), num_runs=3)
    assert set(out.ShortName) == {"sess_a"}
    assert set(out.BrainRegion) == {MFC}


def test_recall_and_precision_come_back_too():
    out = _processDF(MFC, "s1", trials(), num_runs=3)
    assert {"accuracy", "recall", "precision"} <= set(out.columns)


def test_grouping_splits_the_session_into_conditions():
    df = trials(n=40)
    df["quantile_idx"] = [1] * 20 + [3] * 20
    out = _runSession(MFC, "s1", df, groupby_cols=["quantile_idx"], num_runs=3)
    assert set(out.quantile_idx) == {1, 3}
    assert len(out) == 6              # three splits per condition


def test_the_default_run_count_is_the_published_one():
    assert DEFAULT_NUM_RUNS == 50
