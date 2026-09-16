'''Tests for the per-neuron tuning table (Figure 6B, and the input to 6E).

Reproduced while extracting: the same tuning table as the notebook cell, on
both the sampling and the feedback frames.

The pins that matter are the three filters — split rows dropped, the
ROC-preferred side taken, and empty columns removed — because each one changes
what "tuned" means downstream without changing the shape of the output.
'''
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ..tunedneurons import DEFAULT_PVAL, extractTracesPreferences


def roc_row(trace_id, data_col, pval, left=1.0, right=2.0, is_left=True,
            session="s1", br=6, prior_data_col=None, dvstr=None):
    return {"trace_id": trace_id, "ShortName": session, "BrainRegion": br,
            "data_col": data_col, "pval": pval, "data_val_left": left,
            "data_val_right": right, "IsROCLeftTuend": is_left,
            "IsLeftTunedMean": not is_left,     # deliberately the other way
            "prior_data_col": prior_data_col, "DVstr": dvstr}


def frame(rows):
    return pd.DataFrame(rows)


def test_one_row_per_neuron_one_column_per_variable():
    out = extractTracesPreferences(frame([
        roc_row("n1", "ChoiceLeft", 0.01), roc_row("n1", "PrevChoiceLeft", 0.9),
        roc_row("n2", "ChoiceLeft", 0.9), roc_row("n2", "PrevChoiceLeft", 0.01)]),
        verbose=False)
    assert list(out.trace_id) == ["s1_n1", "s1_n2"]
    assert out.set_index("trace_id").ChoiceLeft.isnull().tolist() == [False, True]
    assert out.set_index("trace_id").PrevChoiceLeft.isnull().tolist() == [True, False]


def test_an_untuned_variable_is_nan_not_zero():
    out = extractTracesPreferences(frame([roc_row("n1", "ChoiceLeft", 0.9)]),
                                   verbose=False)
    assert np.isnan(out.ChoiceLeft.iloc[0])


def test_the_value_is_the_roc_preferred_side():
    out = extractTracesPreferences(frame([
        roc_row("n1", "ChoiceLeft", 0.01, left=10., right=20., is_left=True),
        roc_row("n2", "ChoiceLeft", 0.01, left=10., right=20., is_left=False)]),
        verbose=False)
    assert list(out.ChoiceLeft) == [10.0, 20.0]     # not the larger mean twice


def test_the_threshold_is_inclusive():
    out = extractTracesPreferences(frame([roc_row("n1", "ChoiceLeft", DEFAULT_PVAL)]),
                                   verbose=False)
    assert out.ChoiceLeft.notnull().all()


def test_rows_split_by_a_prior_or_by_difficulty_are_dropped():
    out = extractTracesPreferences(frame([
        roc_row("n1", "ChoiceLeft", 0.9),
        roc_row("n1", "ChoiceLeft", 0.01, prior_data_col="PrevChoiceLeft"),
        roc_row("n1", "ChoiceLeft", 0.01, dvstr="Easy")]), verbose=False)
    assert len(out) == 1
    assert np.isnan(out.ChoiceLeft.iloc[0])   # only the unsplit row counted


def test_a_variable_no_neuron_was_tested_on_is_dropped():
    df = frame([roc_row("n1", "ChoiceLeft", 0.01),
                roc_row("n2", "Unused", 0.01, prior_data_col="PrevChoiceLeft")])
    out = extractTracesPreferences(df, verbose=False)
    assert "Unused" not in out.columns
    assert list(out.columns) == ["trace_id", "ShortName", "BrainRegion", "ChoiceLeft"]


def test_min_num_tuning_keeps_only_neurons_tuned_to_enough_variables():
    rows = [roc_row("n1", "ChoiceLeft", 0.01), roc_row("n1", "PrevChoiceLeft", 0.01),
            roc_row("n2", "ChoiceLeft", 0.01), roc_row("n2", "PrevChoiceLeft", 0.9)]
    assert len(extractTracesPreferences(frame(rows), min_num_tuning=2,
                                        verbose=False)) == 1
    assert len(extractTracesPreferences(frame(rows), verbose=False)) == 2


def test_must_tuning_col_requires_that_variable():
    rows = [roc_row("n1", "ChoiceLeft", 0.01), roc_row("n1", "PrevChoiceLeft", 0.9),
            roc_row("n2", "ChoiceLeft", 0.9), roc_row("n2", "PrevChoiceLeft", 0.01)]
    out = extractTracesPreferences(frame(rows),
                                   must_tuning_col_li=["PrevChoiceLeft"],
                                   verbose=False)
    assert list(out.trace_id) == ["s1_n2"]


def test_trace_ids_are_qualified_by_session():
    out = extractTracesPreferences(frame([
        roc_row("n1", "ChoiceLeft", 0.01, session="s1"),
        roc_row("n1", "ChoiceLeft", 0.01, session="s2")]), verbose=False)
    assert sorted(out.trace_id) == ["s1_n1", "s2_n1"]


def test_a_stricter_threshold_unlabels_a_borderline_neuron():
    df = frame([roc_row("n1", "ChoiceLeft", 0.04)])
    assert extractTracesPreferences(df, verbose=False).ChoiceLeft.notnull().all()
    assert extractTracesPreferences(df, pval_threshold=0.01,
                                    verbose=False).ChoiceLeft.isnull().all()
