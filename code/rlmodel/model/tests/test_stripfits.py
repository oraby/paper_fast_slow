"""Tests for dropping the optimiser trace from saved fits.

The load-bearing guarantees: everything a figure reads survives, and a second
run cannot destroy the backup made by the first.
"""
import os
import pickle

import numpy as np
import pandas as pd
import pytest

from .. import stripfits
from ..stripfits import TRACE_KEY, _isFitPayload, inspectFile, stripFile, stripPayload


def _subjectDf():
    return pd.DataFrame({"Name": ["m1"] * 4,
                         "DV": [0.5, -0.5, 1.0, -1.0],
                         "ChoiceLeft": [1.0, 0.0, 1.0, 0.0]})


def _entry(with_trace=True):
    df = _subjectDf()
    entry = {
        "subject_df": df,
        "fixed_params_names": np.array(["df", "biasFn"], dtype=object),
        "fixed_params_vals": [df.copy(), "None_"],
        "OptimRes": {"x": np.array([1.0, 2.0]), "fun": 3.5},
        "params_names": np.array(["DRIFT_COEF", "BOUND"], dtype=object),
        "dt": 0.005,
        "t_dur": 4.8,
    }
    if with_trace:
        entry[TRACE_KEY] = pd.DataFrame({"loss": np.arange(500.0),
                                         "DRIFT_COEF": np.arange(500.0)})
    return entry


def _fit(subjects=("m1", "m2"), with_trace=True):
    return {s: _entry(with_trace) for s in subjects}


def test_recognises_a_fit_payload_and_rejects_other_files():
    assert _isFitPayload(_fit())
    assert not _isFitPayload(_subjectDf())          # df_2p_missing.pkl
    assert not _isFitPayload({})
    assert not _isFitPayload({"a": 1, "b": 2})      # a metrics cache


def test_strip_removes_the_trace_and_nothing_else():
    fit = _fit()
    out, removed = stripPayload(fit)
    assert removed == 2
    for subject in fit:
        assert TRACE_KEY not in out[subject]
        kept = set(fit[subject]) - {TRACE_KEY}
        assert set(out[subject]) == kept
        # the values a figure reads are the same objects, not copies
        assert out[subject]["subject_df"] is fit[subject]["subject_df"]
        assert out[subject]["OptimRes"] is fit[subject]["OptimRes"]


def test_strip_does_not_mutate_the_caller_payload():
    fit = _fit()
    stripPayload(fit)
    assert TRACE_KEY in fit["m1"]


def test_roundtrip_keeps_what_the_figures_read(tmp_path):
    """OptimRes.fun drives the loss table; fixed_params_vals['df'] drives 7D."""
    src = tmp_path / "chisq_Fake_biasNone.pkl"
    with open(src, "wb") as fp:
        pickle.dump(_fit(), fp)

    res = stripFile(str(src), str(tmp_path / "model_full"), dry_run=False)
    assert res["status"] == "stripped"
    assert res["after"] < res["before"]

    back = pd.read_pickle(src)
    entry = back["m1"]
    assert TRACE_KEY not in entry
    assert entry["OptimRes"]["fun"] == 3.5
    names = list(entry["fixed_params_names"])
    df_back = entry["fixed_params_vals"][names.index("df")]
    pd.testing.assert_frame_equal(df_back, _subjectDf())
    pd.testing.assert_frame_equal(entry["subject_df"], _subjectDf())


def test_original_is_copied_to_the_backup_dir(tmp_path):
    src = tmp_path / "chisq_Fake_biasNone.pkl"
    with open(src, "wb") as fp:
        pickle.dump(_fit(), fp)
    backup_dir = tmp_path / "model_full"

    stripFile(str(src), str(backup_dir), dry_run=False)

    backup = backup_dir / "chisq_Fake_biasNone.pkl"
    assert backup.exists()
    assert TRACE_KEY in pd.read_pickle(backup)["m1"], (
        "the backup must be the full file, trace included")


def test_second_run_cannot_clobber_the_backup(tmp_path):
    """Re-running must not copy an already-stripped file over the full one."""
    src = tmp_path / "chisq_Fake_biasNone.pkl"
    with open(src, "wb") as fp:
        pickle.dump(_fit(), fp)
    backup_dir = tmp_path / "model_full"

    stripFile(str(src), str(backup_dir), dry_run=False)
    full_size = (backup_dir / "chisq_Fake_biasNone.pkl").stat().st_size

    # A second pass finds nothing to strip and leaves the backup alone.
    again = stripFile(str(src), str(backup_dir), dry_run=False)
    assert again["status"] == "already stripped"
    assert (backup_dir / "chisq_Fake_biasNone.pkl").stat().st_size == full_size
    assert TRACE_KEY in pd.read_pickle(backup_dir / "chisq_Fake_biasNone.pkl")["m1"]


def test_dry_run_changes_nothing(tmp_path):
    src = tmp_path / "chisq_Fake_biasNone.pkl"
    with open(src, "wb") as fp:
        pickle.dump(_fit(), fp)
    before = src.stat().st_size

    res = stripFile(str(src), str(tmp_path / "model_full"), dry_run=True)

    assert res["status"] == "would strip"
    assert res["after"] < res["before"]
    assert src.stat().st_size == before
    assert not (tmp_path / "model_full").exists()
    assert TRACE_KEY in pd.read_pickle(src)["m1"]


def test_inspect_reports_trace_and_duplicate_frame(tmp_path):
    src = tmp_path / "chisq_Fake_biasNone.pkl"
    with open(src, "wb") as fp:
        pickle.dump(_fit(), fp)

    info = inspectFile(str(src))
    assert info["subjects"] == 2
    assert info["trace_bytes"] > 0
    # fixed_params_vals['df'] equals subject_df, so it is counted as duplicate
    assert info["dupe_bytes"] > 0


def test_non_fit_file_is_skipped(tmp_path):
    src = tmp_path / "df_2p_missing.pkl"
    _subjectDf().to_pickle(src)
    res = stripFile(str(src), str(tmp_path / "model_full"), dry_run=False)
    assert res["status"].startswith("skipped")
    assert not (tmp_path / "model_full").exists()
