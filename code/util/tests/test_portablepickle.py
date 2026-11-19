"""Round-trip tests for the portable payload format.

The point of the format is that a payload names no class outside ``numpy`` and
``builtins``, so ``test_payload_names_no_foreign_class`` is the load-bearing
test here -- the rest guard fidelity.
"""
import pickle
import functools
import pickletools
import io

import numpy as np
import pandas as pd
import pytest

from ..portablepickle import TAG, DROPPED_COLUMNS, toPortable, fromPortable


def _roundTrip(obj):
    return fromPortable(pickle.loads(pickle.dumps(toPortable(obj))))


def _globalsIn(payload):
    """Every (module, name) a pickle of ``payload`` actually imports on load.

    Recorded through ``Unpickler.find_class`` rather than by scanning opcodes:
    the opcode stream stores ``STACK_GLOBAL``'s module and name as two separate
    strings, so reading them positionally mis-pairs them.
    """
    raw = pickle.dumps(payload)
    names = set()

    class _Recorder(pickle.Unpickler):
        def find_class(self, module, name):
            names.add((module, name))
            return super().find_class(module, name)

    _Recorder(io.BytesIO(raw)).load()
    return names


class States:
    """Stands in for ``caiman...matreader.states.States``, matched by name."""
    def __init__(self, n=0):
        self.n = n


def _frame():
    return pd.DataFrame({
        "Name": ["GP4-85", "GP4-24", "GP4-85"],
        "DV": [0.5, -1.0, 0.25],
        "ChoiceLeft": [1.0, 0.0, np.nan],
        "TrialNumber": [1, 2, 3],
    })


def test_dataframe_round_trips_values_columns_and_index():
    df = _frame()
    out = _roundTrip(df)
    pd.testing.assert_frame_equal(out, df, check_dtype=False)


def test_payload_names_no_foreign_class():
    """The whole point: only numpy/builtins survive into the payload."""
    df = _frame()
    df["string_col"] = pd.array(["a", "b", "c"], dtype="string")
    df["nullable_int"] = pd.array([1, 2, None], dtype="Int64")
    df["cat"] = pd.Categorical(["x", "y", "x"])

    for module, _name in _globalsIn(toPortable(df)):
        root = module.split(".")[0]
        assert root in ("numpy", "builtins", "_codecs"), (
            f"payload still imports {module}")


def test_extension_dtypes_are_flattened_to_numpy():
    df = pd.DataFrame({
        "s": pd.array(["Easy", "Hard"], dtype="string"),
        "i": pd.array([1, 2], dtype="Int64"),
    })
    out = _roundTrip(df)
    assert out["s"].dtype == object
    assert list(out["s"]) == ["Easy", "Hard"]
    assert list(out["i"]) == [1, 2]


def test_traces_sets_object_column_survives():
    """The neural frames hold a dict-of-dict-of-ndarray per row."""
    traces = {"neuronal": {0: np.arange(5.0), 1: np.arange(5.0) * 2}}
    df = pd.DataFrame({"epoch": ["Sampling", "Feedback"],
                       "traces_sets": [traces, traces]})
    out = _roundTrip(df)
    assert set(out["traces_sets"][0]["neuronal"]) == {0, 1}
    np.testing.assert_array_equal(out["traces_sets"][0]["neuronal"][1],
                                  np.arange(5.0) * 2)


def test_stubbed_foreign_column_is_dropped_and_recorded():
    df = _frame()
    df["States"] = [States(1), States(2), States(3)]
    payload = toPortable(df)
    assert payload[DROPPED_COLUMNS] == ["States"]
    assert "States" not in fromPortable(payload).columns


class mat_struct:
    """Shape-compatible stand-in for ``scipy.io.matlab._mio5_params.mat_struct``.

    Matched structurally, not by import, because scipy keeps it behind a
    private path that has already been renamed once.
    """
    def __init__(self, **fields):
        self._fieldnames = list(fields)
        for k, v in fields.items():
            setattr(self, k, v)


def test_matstruct_becomes_a_plain_dict():
    df = _frame()
    df["drawParams"] = [mat_struct(nDots=100, apertureSize=8.0)] * 3
    out = _roundTrip(df)
    assert out["drawParams"][0] == {"nDots": 100, "apertureSize": 8.0}

    for module, _name in _globalsIn(toPortable(df)):
        assert module.split(".")[0] in ("numpy", "builtins", "_codecs")


def test_non_stub_object_column_is_kept():
    df = _frame()
    df["notes"] = [{"a": 1}, {"b": 2}, {"c": 3}]
    out = _roundTrip(df)
    assert out["notes"][1] == {"b": 2}


def test_repo_function_survives_without_being_imported_by_the_pickle():
    """The RL-model fits store bias/drift/noise functions as objects.

    Recorded by name, the payload imports nothing repo-local -- which is what
    makes it survive a renamed checkout -- yet still resolves to the same
    function.
    """
    from ...rlmodel.model.bias import _biasNone

    payload = toPortable(_biasNone)
    for module, _name in _globalsIn(payload):
        assert module.split(".")[0] in ("numpy", "builtins", "_codecs"), (
            f"payload still imports {module}")
    assert fromPortable(pickle.loads(pickle.dumps(payload))) is _biasNone


def test_partial_round_trips_with_its_bound_arguments():
    from ...rlmodel.model.drift import DRIFT_FN_DICT

    original = DRIFT_FN_DICT["DriftGain-RewardRate"]
    assert isinstance(original, functools.partial)

    out = _roundTrip(original)
    assert out.func is original.func
    assert out.keywords == original.keywords


def test_callable_payload_resolves_under_a_renamed_prefix():
    """A pickle written when the repo had a different name still resolves."""
    payload = toPortable(_frame)  # any function defined in this package
    payload["module"] = "some_other_checkout." + payload["module"]
    assert fromPortable(payload) is _frame


def test_namedtuple_becomes_field_keyed_dict():
    from collections import namedtuple
    RunData = namedtuple("RunData", ["df_src", "shortlong_df"])
    out = _roundTrip(RunData(df_src=_frame(), shortlong_df=_frame()))
    assert set(out) == {"df_src", "shortlong_df"}
    pd.testing.assert_frame_equal(out["df_src"], _frame(), check_dtype=False)


def test_nested_dict_of_frames_round_trips():
    payload = {"Real": _frame(), "Rand1": {"inner": _frame()}}
    out = _roundTrip(payload)
    pd.testing.assert_frame_equal(out["Real"], _frame(), check_dtype=False)
    pd.testing.assert_frame_equal(out["Rand1"]["inner"], _frame(),
                                  check_dtype=False)


def test_non_default_and_multi_index_round_trip():
    df = _frame().set_index("Name")
    pd.testing.assert_frame_equal(_roundTrip(df), df, check_dtype=False)

    multi = _frame().set_index(["Name", "TrialNumber"])
    pd.testing.assert_frame_equal(_roundTrip(multi), multi, check_dtype=False)


def test_series_round_trips():
    s = pd.Series([1.0, 2.0, 3.0], name="RewardRate")
    pd.testing.assert_series_equal(_roundTrip(s), s, check_dtype=False)


@pytest.mark.parametrize("value", [None, 3, "x", 2.5, True])
def test_scalars_pass_through(value):
    assert _roundTrip(value) == value or _roundTrip(value) is None


def test_interval_values_round_trip_without_importing_pandas():
    df = pd.DataFrame({"bin": pd.cut([1.0, 2.0, 3.0], bins=2)})
    df["bin"] = df["bin"].astype(object)
    payload = toPortable(df)
    for module, _name in _globalsIn(payload):
        assert module.split(".")[0] in ("numpy", "builtins", "_codecs"), (
            f"payload still imports {module}")
    out = fromPortable(payload)
    assert out["bin"][0] == df["bin"][0]


def test_dataclass_instance_round_trips_by_name():
    """``MLEModelConfig`` rides along inside the MLE fit payloads."""
    from ...rlmodel.model.mle import MLEModelConfig

    cfg = MLEModelConfig(drift_fn_str="Classic", bias_fn_str="None_",
                         noise_fn_str="Normal(0, 1)", include_Q=False,
                         include_RewardRate=False, dt=0.005, t_dur=4.8)
    payload = toPortable(cfg)
    for module, _name in _globalsIn(payload):
        assert module.split(".")[0] in ("numpy", "builtins", "_codecs")
    out = fromPortable(pickle.loads(pickle.dumps(payload)))
    assert isinstance(out, MLEModelConfig)
    assert out.drift_fn_str == "Classic" and out.t_dur == 4.8
