'''Tests for the per-trial trace reshaping helpers (Figures S12A, S12C, S12D, S12J).'''
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ..epochcrop import epochStart, keepNeurons, onlyEpoch, padToCommonLength

NAMES = ["Wait Trial Start", "Sampling", "Movement to Lateral Port"]
RANGES = [(0, 2), (3, 6), (7, 9)]


def frame():
    trace = np.arange(10.0)
    return pd.DataFrame({
        "epoch": ["Wait Trial Start"] * 2,
        "epochs_names": [NAMES, NAMES],
        "epochs_ranges": [RANGES, RANGES],
        "trace_start_idx": [0, 0], "trace_end_idx": [9, 9],
        "traces_sets": [{"neuronal": {1: trace, 2: trace + 100},
                         "neuronal_raw": {1: trace * 2}},
                        {"neuronal": {1: trace + 10}, "neuronal_raw": {}}],
    })


def test_keeps_the_epochs_inclusive_samples():
    out = onlyEpoch(frame(), "Sampling")
    np.testing.assert_array_equal(out.traces_sets[0]["neuronal"][1], [3, 4, 5, 6])
    np.testing.assert_array_equal(out.traces_sets[0]["neuronal"][2], [103, 104, 105, 106])
    np.testing.assert_array_equal(out.traces_sets[0]["neuronal_raw"][1], [6, 8, 10, 12])
    np.testing.assert_array_equal(out.traces_sets[1]["neuronal"][1], [13, 14, 15, 16])


def test_bookkeeping_describes_the_single_epoch():
    out = onlyEpoch(frame(), "Sampling")
    row = out.iloc[0]
    assert row.epochs_names == ["Sampling"]
    assert row.epochs_ranges == [(0, 3)]
    assert row.epoch == "Sampling"
    assert (row.trace_start_idx, row.trace_end_idx) == (0, 3)


def test_input_is_left_untouched():
    df = frame()
    onlyEpoch(df, "Sampling")
    assert len(df.traces_sets[0]["neuronal"][1]) == 10
    assert df.epochs_names[0] == NAMES


def test_unknown_epoch_raises():
    with pytest.raises(KeyError, match="Feedback"):
        onlyEpoch(frame(), "Feedback Start")


def test_epoch_start_keeps_the_first_samples():
    out = epochStart(frame(), "Sampling", 2)
    np.testing.assert_array_equal(out.traces_sets[0]["neuronal"][1], [3, 4])
    np.testing.assert_array_equal(out.traces_sets[1]["neuronal"][1], [13, 14])
    row = out.iloc[0]
    assert row.epochs_names == ["Sampling"] and row.epochs_ranges == [(0, 1)]
    assert (row.trace_start_idx, row.trace_end_idx) == (0, 1)


def test_epoch_start_never_runs_past_the_epoch():
    out = epochStart(frame(), "Sampling", 10)
    np.testing.assert_array_equal(out.traces_sets[0]["neuronal"][1], [3, 4, 5, 6])
    with pytest.raises(ValueError):
        epochStart(frame(), "Sampling", 0)


REAL_NAMES = ["-0.1s Sampling", "Sampling", "Movement to Lateral Port"]


def realTime():
    """Two trials: 6 and 9 samples, movement onset at 4 and 7."""
    return pd.DataFrame({
        "ShortName": ["s1", "s1"],
        "epochs_names": [REAL_NAMES, REAL_NAMES],
        "epochs_ranges": [[(0, 0), (1, 3), (4, 5)], [(0, 0), (1, 6), (7, 8)]],
        "trace_start_idx": [0, 0], "trace_end_idx": [5, 8],
        "traces_sets": [{"neuronal": {1: np.arange(6.0), 2: np.arange(6.0)}},
                        {"neuronal": {1: np.arange(9.0) + 10}}],
    })


def test_pad_keeps_the_start_and_pads_the_end():
    out = padToCommonLength(realTime())
    a, b = out.traces_sets.iloc[0]["neuronal"][1], out.traces_sets.iloc[1]["neuronal"][1]
    assert len(a) == len(b) == 9
    np.testing.assert_array_equal(a[:6], np.arange(6.0))
    assert np.isnan(a[6:]).all()
    assert out.epochs_ranges.iloc[0] == [(0, 0), (1, 3), (4, 5)]
    assert (out.trace_start_idx == 0).all() and (out.trace_end_idx == 8).all()


def test_pad_lines_up_an_epoch_onset():
    out = padToCommonLength(realTime(), align_epoch="Movement to Lateral Port")
    a, b = out.traces_sets.iloc[0]["neuronal"][1], out.traces_sets.iloc[1]["neuronal"][1]
    # The short trial is shifted by 3 so both movements start at sample 7
    assert out.epochs_ranges.iloc[0][2][0] == out.epochs_ranges.iloc[1][2][0] == 7
    assert len(a) == len(b) == 9
    assert np.isnan(a[:3]).all()
    np.testing.assert_array_equal(a[3:], np.arange(6.0))
    np.testing.assert_array_equal(b, np.arange(9.0) + 10)


def test_pad_honours_trace_start_idx():
    df = realTime().iloc[[1]].copy()
    df["trace_start_idx"] = 2
    out = padToCommonLength(df)
    np.testing.assert_array_equal(out.traces_sets.iloc[0]["neuronal"][1],
                                  np.arange(2.0, 9.0) + 10)
    assert out.epochs_ranges.iloc[0][1] == (-1, 4)


def test_keep_neurons_drops_other_traces_and_empty_trials():
    out = keepNeurons(realTime(), {"s1_2"})
    assert len(out) == 1
    assert list(out.traces_sets.iloc[0]["neuronal"]) == [2]
