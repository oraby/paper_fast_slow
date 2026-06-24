"""Tests for ``--only-subject`` selection and the reload-merge evolve save.

``--only-subject`` filters the input dataframe to the named subjects
(``_select_only_subjects``); the per-subject save in ``fit.simulateDDM``
is now a reload-merge-write (``_merge_save_evolve``) so parallel
processes fitting different subjects don't clobber each other.
"""
from __future__ import annotations

import pickle

import pandas as pd
import pytest

from .. import fit
from ...model_runner import _select_only_subjects, _order_by_only_subjects


def _df(names):
    return pd.DataFrame({"Name": names, "x": range(len(names))})


# ---------------------------------------------------------------------------
# _select_only_subjects
# ---------------------------------------------------------------------------

def test_select_only_subjects_falsy_is_identity():
    df = _df(["S1", "S2", "S1"])
    # No selection → same object back (no copy), both for None and [].
    assert _select_only_subjects(df, None) is df
    assert _select_only_subjects(df, []) is df


def test_select_only_subjects_filters_named():
    df = _df(["S1", "S2", "S3", "S1"])
    one = _select_only_subjects(df, ["S1"])
    assert set(one["Name"]) == {"S1"}
    assert len(one) == 2
    two = _select_only_subjects(df, ["S1", "S3"])
    assert set(two["Name"]) == {"S1", "S3"}
    assert len(two) == 3


def test_select_only_subjects_unknown_name_raises_listing_available():
    df = _df(["S1", "S2"])
    with pytest.raises(ValueError, match="Nope") as exc:
        _select_only_subjects(df, ["S1", "Nope"])
    # The message names the offending subject AND the available ones so a
    # typo is diagnosable at startup.
    msg = str(exc.value)
    assert "Nope" in msg and "S1" in msg and "S2" in msg


# ---------------------------------------------------------------------------
# _order_by_only_subjects — process in CLI order
# ---------------------------------------------------------------------------

def test_order_by_only_subjects_follows_cli_order():
    """Subjects must be (re)fit in the order named on the command line,
    not the alphabetical order _extendTrials leaves the df in. Within a
    subject the trial order must be preserved (RL state propagation)."""
    # As after _extendTrials: alphabetical by Name, trial-ordered within.
    df = pd.DataFrame({
        "Name": ["S1", "S1", "S2", "S2", "S3", "S3"],
        "TrialNumber": [1, 2, 1, 2, 1, 2],
    })
    out = _order_by_only_subjects(df, ["S3", "S1", "S2"])
    # Subject blocks now follow the CLI order (this is what drives
    # simulateDDM's df.Name.unique() processing order).
    assert list(pd.unique(out["Name"])) == ["S3", "S1", "S2"]
    # Within-subject order is intact for every subject.
    for name in ("S1", "S2", "S3"):
        assert list(out[out.Name == name]["TrialNumber"]) == [1, 2]


def test_order_by_only_subjects_partial_and_noop():
    df = pd.DataFrame({"Name": ["S1", "S2", "S3"], "x": [1, 2, 3]})
    # Falsy → identity (same object, no sort).
    assert _order_by_only_subjects(df, None) is df
    assert _order_by_only_subjects(df, []) is df
    # Single subject is trivially ordered.
    one = _order_by_only_subjects(df[df.Name == "S2"], ["S2"])
    assert list(one["Name"]) == ["S2"]


# ---------------------------------------------------------------------------
# _merge_save_evolve — reload-merge-atomic-write
# ---------------------------------------------------------------------------

def test_merge_save_evolve_preserves_other_subjects(tmp_path):
    fp = tmp_path / "evolve.pkl"
    # Pre-existing pickle (e.g. from a previous full run / another process).
    with open(fp, "wb") as f:
        pickle.dump({"S1": {"a": 1}, "S2": {"a": 2}}, f)
    # This process re-fits S2 and adds a brand-new S3.
    fit._merge_save_evolve(fp, "S2", {"a": 99})
    fit._merge_save_evolve(fp, "S3", {"a": 3})
    with open(fp, "rb") as f:
        merged = pickle.load(f)
    assert set(merged) == {"S1", "S2", "S3"}
    assert merged["S1"] == {"a": 1}    # untouched
    assert merged["S2"] == {"a": 99}   # overwritten by this process
    assert merged["S3"] == {"a": 3}    # newly added
    # Atomic write leaves no temp file behind.
    assert list(tmp_path.glob("*.tmp*")) == []


def test_merge_save_evolve_creates_file_when_missing(tmp_path):
    fp = tmp_path / "evolve.pkl"
    assert not fp.exists()
    fit._merge_save_evolve(fp, "S1", {"a": 1})
    with open(fp, "rb") as f:
        assert pickle.load(f) == {"S1": {"a": 1}}


def test_merge_save_evolve_lost_update_regression(tmp_path):
    """The parallel-safety contract: a process holding a stale startup
    snapshot must not clobber another process's concurrent write.

    Emulates: process B saves S2 to disk; process A (which started with
    only {S1} in memory) then saves S1 via the reload-merge path. Because
    A reloads disk immediately before writing, S2 survives — a plain
    ``pickle.dump(A_snapshot)`` would have dropped it.
    """
    fp = tmp_path / "evolve.pkl"
    fit._merge_save_evolve(fp, "S2", {"a": 2})   # process B's write
    fit._merge_save_evolve(fp, "S1", {"a": 1})   # process A's write
    with open(fp, "rb") as f:
        merged = pickle.load(f)
    assert set(merged) == {"S1", "S2"}
