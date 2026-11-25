"""The write-time guard: refuse to save a pickle that only this repo can read.

These are the tests that actually enforce the rule, because they run on every
`uv run pytest`. The repo-wide audit (`migratepickles --audit`) checks the real
files but is too slow for the default suite.
"""
import json
import pathlib
import pickle
import re
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from ..portablepickle import (NotPortableError, assertPortable, findUnportable,
                              savePortable)


# --- things a bare reader could not import ---------------------------------
# Defined here rather than imported so the test states its own fixture, but
# their __module__ is this test module -- i.e. repo-local, exactly the shape
# the guard exists to catch.

def _repoFunction(x):
    return x


@dataclass
class _RepoConfig:
    alpha: float = 0.5


_RepoTuple = namedtuple("_RepoTuple", ["left", "right"])


def _frame():
    return pd.DataFrame({"Name": ["GP4-85", "GP4-24"], "DV": [0.5, -1.0]})


# --- what must pass ---------------------------------------------------------

def test_plain_data_is_portable():
    payload = {"df": _frame(), "loss": 1.5, "names": ["ALPHA", "BOUND"],
               "arr": np.arange(4.0), "nested": {"a": [1, 2, {"b": 3}]}}
    assert findUnportable(payload) == {}
    assertPortable(payload)


def test_traces_sets_shape_is_portable():
    traces = {"neuronal": {0: np.arange(5.0), 1: np.arange(5.0)}}
    df = pd.DataFrame({"epoch": ["Sampling"], "traces_sets": [traces]})
    assert findUnportable(df) == {}


def test_scipy_optimize_result_is_allowed():
    """``OptimRes`` in every fit payload is a scipy public class."""
    from scipy.optimize import OptimizeResult
    assert findUnportable(OptimizeResult(x=np.zeros(3), fun=1.0)) == {}


# --- what must be refused ---------------------------------------------------

def test_repo_function_is_refused():
    bad = findUnportable({"biasFn": _repoFunction})
    assert any("_repoFunction" in name for name in bad)


def test_repo_dataclass_instance_is_refused():
    bad = findUnportable({"model_config": _RepoConfig()})
    assert any("_RepoConfig" in name for name in bad)


def test_repo_namedtuple_is_refused():
    bad = findUnportable({"run": _RepoTuple(left=_frame(), right=_frame())})
    assert any("_RepoTuple" in name for name in bad)


def test_offender_inside_a_dataframe_column_is_found():
    df = _frame()
    df["cfg"] = [_RepoConfig(), _RepoConfig()]
    assert any("_RepoConfig" in name for name in findUnportable(df))


def test_offender_inside_an_object_array_is_found():
    arr = np.empty(2, dtype=object)
    arr[:] = [_repoFunction, _repoFunction]
    assert any("_repoFunction" in name for name in findUnportable(arr))


def test_pandas_string_column_is_refused():
    """It opens only under a pandas built the way the writer's was."""
    df = _frame()
    assert findUnportable(df.astype({"Name": object})) == {}
    strings = df.astype({"Name": "string"})
    for payload in (strings, {"S1": {"subject_df": strings}}, strings.Name,
                    pd.Index(strings.Name)):
        assert any("string dtype" in name for name in findUnportable(payload))


def test_error_names_the_offender_and_where_it_sits():
    with pytest.raises(NotPortableError) as excinfo:
        assertPortable({"fixed_params_vals": [1.0, _repoFunction]})
    message = str(excinfo.value)
    assert "_repoFunction" in message
    assert "fixed_params_vals" in message


# --- the writer -------------------------------------------------------------

def test_save_writes_a_file_that_plain_pickle_reads(tmp_path):
    out = tmp_path / "clean.pkl"
    savePortable({"df": _frame(), "loss": 2.0}, out)
    with open(out, "rb") as fp:
        assert set(pickle.load(fp)) == {"df", "loss"}


def test_save_refuses_and_leaves_no_file(tmp_path):
    out = tmp_path / "dirty.pkl"
    with pytest.raises(NotPortableError):
        savePortable({"biasFn": _repoFunction}, out)
    assert not out.exists()
    assert not (tmp_path / "dirty.pkl.tmp").exists()


def test_save_replaces_atomically(tmp_path):
    """A refused write must not clobber a good file that is already there."""
    out = tmp_path / "x.pkl"
    savePortable({"good": 1}, out)
    with pytest.raises(NotPortableError):
        savePortable({"bad": _repoFunction}, out)
    with open(out, "rb") as fp:
        assert pickle.load(fp) == {"good": 1}


def test_cycles_do_not_hang_the_walk():
    node = {"name": "a"}
    node["self"] = node
    assert findUnportable(node) == {}


# --- every writer goes through the guard -------------------------------------

_CODE_DIR = pathlib.Path(__file__).resolve().parents[2]
_RAW_WRITE = re.compile(r"pickle\.dump\(|\.to_pickle\(")
#: Writers allowed to pickle directly, relative to code/, with the reason.
_RAW_WRITE_ALLOWED = {
    # The guard itself, and the one-off migration tool that rewrote data/.
    "util/portablepickle.py", "util/migratepickles.py",
    # Checks with assertPortable itself, then writes through a per-process
    # temp file so concurrent cluster jobs never collide.
    "rlmodel/model/fit.py",
    # Scratch shards in a work directory, read back by the same run and merged.
    "rlmodel/model/metrics_shards.py",
    # The GUI's own cache next to the notebook, not under data/.
    "rlmodel/rlmodel_interactive.ipynb",
}


def _codeSources():
    for fp in sorted(_CODE_DIR.rglob("*")):
        rel = fp.relative_to(_CODE_DIR).as_posix()
        if "tests/" in rel or ".ipynb_checkpoints" in rel:
            continue
        if fp.suffix == ".py":
            yield rel, fp.read_text(encoding="utf-8")
        elif fp.suffix == ".ipynb":
            cells = json.loads(fp.read_text(encoding="utf-8"))["cells"]
            yield rel, "\n".join("".join(c["source"]) for c in cells
                                 if c["cell_type"] == "code")


def test_every_pickle_writer_uses_the_guard():
    raw = sorted(rel for rel, text in _codeSources()
                 if _RAW_WRITE.search(text) and rel not in _RAW_WRITE_ALLOWED)
    assert raw == [], ("these write pickles without savePortable: "
                       f"{raw}")
