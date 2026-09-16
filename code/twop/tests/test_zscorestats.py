"""The recomputed statistics must be the ones NormalizeZScore actually used."""
import numpy as np
import pandas as pd
import pytest

from ...pipeline import tracesnormalize
from ..zscorestats import sessionTraceStats, traceStats


def _row(short_name, trial, traces, start=0, end=None, sole_owner=False):
    end = (len(next(iter(traces.values()))) - 1) if end is None else end
    return {"ShortName": short_name, "TrialNumber": trial, "sole_owner": sole_owner,
            "trace_start_idx": start, "trace_end_idx": end,
            "traces_sets": {"neuronal": traces}}


def _frame(rows):
    df = pd.DataFrame(rows)
    # Columns the normaliser carries along but does not use here.
    df["Name"], df["Date"], df["SessionNum"] = "GP4", "2024-01-01", 1
    return df


def test_matches_what_normalizezscore_subtracted():
    """(raw - mean) / std must reproduce the pipeline's z-scored traces."""
    rng = np.random.default_rng(0)
    raw = {1: {7: rng.normal(5, 2, 40), 9: rng.normal(-3, 4, 40)},
           2: {7: rng.normal(5, 2, 40), 9: rng.normal(-3, 4, 40)}}
    df = _frame([_row("sess_a", trial, traces) for trial, traces in raw.items()])

    zscored = tracesnormalize.NormalizeZScore(set_name="neuronal").process(df.copy())
    stats = sessionTraceStats(df)["neuronal_stats"].iloc[0]

    for (_, row), trial in zip(zscored.iterrows(), raw):
        for trace_id, trace in row.traces_sets["neuronal"].items():
            expected = (raw[trial][trace_id] - stats[trace_id]["mean"]) / stats[trace_id]["std"]
            assert np.allclose(trace, expected)


def test_uses_only_the_rows_trace_window():
    trace = np.concatenate([np.full(5, 100.0), np.arange(10.0), np.full(5, -100.0)])
    df = _frame([_row("sess_a", 1, {3: trace}, start=5, end=14)])
    stats = sessionTraceStats(df)["neuronal_stats"].iloc[0][3]
    assert stats["mean"] == pytest.approx(np.arange(10.0).mean())
    assert stats["std"] == pytest.approx(np.arange(10.0).std())


def test_sole_owner_rows_keep_the_whole_trace():
    trace = np.arange(10.0)
    df = _frame([_row("sess_a", 1, {3: trace}, start=2, end=4, sole_owner=True)])
    assert sessionTraceStats(df)["neuronal_stats"].iloc[0][3]["mean"] == pytest.approx(4.5)


def test_a_repeated_trial_number_replaces_the_earlier_segment():
    """The normaliser keys its per-trace dict by trial, so the last row wins."""
    df = _frame([_row("sess_a", 1, {3: np.zeros(4)}),
                 _row("sess_a", 1, {3: np.full(4, 8.0)})])
    stats = sessionTraceStats(df)["neuronal_stats"].iloc[0][3]
    assert stats["mean"] == pytest.approx(8.0)
    assert stats["std"] == pytest.approx(0.0)


def test_sessions_are_independent():
    df = _frame([_row("sess_a", 1, {3: np.zeros(4)}),
                 _row("sess_b", 1, {3: np.full(4, 6.0)})])
    out = sessionTraceStats(df).set_index("ShortName")["neuronal_stats"]
    assert out["sess_a"][3]["mean"] == pytest.approx(0.0)
    assert out["sess_b"][3]["mean"] == pytest.approx(6.0)


def test_trials_are_concatenated_not_averaged():
    stats = traceStats([np.zeros(4), np.full(4, 4.0)])
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["std"] == pytest.approx(2.0)


def test_nans_are_ignored():
    stats = traceStats([np.array([1.0, np.nan, 3.0])])
    assert stats["mean"] == pytest.approx(2.0)
