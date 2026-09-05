"""Fit payloads round-trip through plain pickle, naming nothing repo-local."""
import pickle

import numpy as np
import pandas as pd
import pytest

from ....util.portablepickle import findUnportable
from ..bias import BIAS_FN_DICT
from ..drift import DRIFT_FN_DICT
from ..fitio import (fromStorable, loadFit, saveFit, toStorable, _registryKey)
from ..mle import MLEModelConfig
from ..noise import NOISE_FN_DICT


def _payload(with_config=False):
    """The shape ``fit.py`` writes: names + parallel values, some callable."""
    out = {
        "fixed_params_names": np.array(
            ["df", "biasFn", "biasFn_df_cols", "biasFn_kwargs",
             "driftFn", "driftFn_df_cols", "driftFn_kwargs",
             "noiseFn", "noiseFn_df_cols"], dtype=object),
        "fixed_params_vals": np.array(
            [pd.DataFrame({"DV": [0.5]}), BIAS_FN_DICT["None_"], [], {},
             DRIFT_FN_DICT["Classic"], [], {},
             NOISE_FN_DICT["Normal(0, 1)"], []], dtype=object),
        "params_names": ["DRIFT_COEF", "BOUND"],
        "dt": 0.005,
    }
    if with_config:
        out["model_config"] = MLEModelConfig(
            drift_fn_str="Classic", bias_fn_str="None_",
            noise_fn_str="Normal(0, 1)", include_Q=False,
            include_RewardRate=False, dt=0.005, t_dur=4.8)
    return out


def test_storable_payload_names_nothing_repo_local():
    assert findUnportable(toStorable(_payload(with_config=True))) == {}


def test_live_payload_is_refused_by_the_guard():
    """Guards the guard: the untreated payload must still be caught."""
    bad = findUnportable(_payload(with_config=True))
    assert any("bias" in name for name in bad)
    assert any("MLEModelConfig" in name for name in bad)


def test_functions_survive_the_round_trip_as_the_same_objects():
    out = fromStorable(toStorable(_payload()))
    vals = list(out["fixed_params_vals"])
    assert vals[1] is BIAS_FN_DICT["None_"]
    assert vals[4] is DRIFT_FN_DICT["Classic"]
    assert vals[7] is NOISE_FN_DICT["Normal(0, 1)"]


def test_model_config_survives_as_a_dataclass():
    out = fromStorable(toStorable(_payload(with_config=True)))
    cfg = out["model_config"]
    assert isinstance(cfg, MLEModelConfig)
    assert cfg.drift_fn_str == "Classic" and cfg.t_dur == 4.8


def test_toStorable_does_not_mutate_the_caller():
    payload = _payload(with_config=True)
    toStorable(payload)
    assert callable(payload["fixed_params_vals"][1])
    assert isinstance(payload["model_config"], MLEModelConfig)


def test_fromStorable_tolerates_an_untreated_payload():
    """Old files still hold live objects; applying it must be harmless."""
    payload = _payload(with_config=True)
    out = fromStorable(payload)
    assert out["fixed_params_vals"][1] is BIAS_FN_DICT["None_"]
    assert isinstance(out["model_config"], MLEModelConfig)


def test_partial_registry_entries_resolve_by_identity():
    """``DRIFT_FN_DICT`` holds partials, which never compare equal."""
    import functools
    fn = DRIFT_FN_DICT["DriftGain-RewardRate"]
    assert isinstance(fn, functools.partial)
    assert _registryKey(fn, DRIFT_FN_DICT) == "DriftGain-RewardRate"


def test_unregistered_function_is_rejected_with_a_useful_message():
    payload = _payload()
    payload["fixed_params_vals"] = list(payload["fixed_params_vals"])
    payload["fixed_params_vals"][1] = lambda *a, **k: None
    with pytest.raises(KeyError, match="not in the registry"):
        toStorable(payload)


def test_saved_fit_reads_back_with_bare_pickle(tmp_path):
    out = tmp_path / "chisq_fake.pkl"
    saveFit({"GP4-85": _payload(with_config=True)}, out)

    with open(out, "rb") as fp:
        raw = pickle.load(fp)          # no repo import, no custom unpickler
    assert set(raw) == {"GP4-85"}
    assert isinstance(raw["GP4-85"]["model_config"], dict)
    assert raw["GP4-85"]["fixed_params_vals"][1] == "None_"


def test_loadFit_rebuilds_what_bare_pickle_leaves_plain(tmp_path):
    out = tmp_path / "chisq_fake.pkl"
    saveFit({"GP4-85": _payload(with_config=True)}, out)

    loaded = loadFit(out)["GP4-85"]
    assert loaded["fixed_params_vals"][1] is BIAS_FN_DICT["None_"]
    assert isinstance(loaded["model_config"], MLEModelConfig)


def test_a_reconstructed_partial_still_matches_the_registry():
    """An unpickled partial is a new object, so identity alone is not enough.

    This is what the real ``DriftGain-*`` fits hit: their drift function comes
    back from the pickle as a fresh ``functools.partial``.
    """
    import functools
    import pickle as _pickle
    from ..fitio import _sameCallable

    original = DRIFT_FN_DICT["DriftGain(1+r)-RewardRate"]
    revived = _pickle.loads(_pickle.dumps(original))
    assert revived is not original
    assert _sameCallable(original, revived)
    assert _registryKey(revived, DRIFT_FN_DICT) == "DriftGain(1+r)-RewardRate"


def test_a_partial_with_different_bindings_does_not_match():
    import functools
    from ..fitio import _sameCallable
    a = DRIFT_FN_DICT["DriftGain-RewardRate"]        # RR_DRIFT_MAP="2-r"
    b = DRIFT_FN_DICT["DriftGain(1+r)-RewardRate"]   # RR_DRIFT_MAP="1+r"
    assert not _sameCallable(a, b)
