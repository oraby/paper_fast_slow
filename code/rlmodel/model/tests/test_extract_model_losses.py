"""Tests for the loss-aggregation script (``rlmodel/extract_model_losses.py``)."""
from __future__ import annotations

import pickle
import types

import numpy as np

from ..mle import MLEModelConfig
from ...extract_model_losses import extract_losses, _parse_filename


def _optim(fun):
    return types.SimpleNamespace(fun=fun, x=np.array([1.0, 2.0]))


def _mle_config(**ov):
    base = dict(drift_fn_str="Classic", bias_fn_str="None_",
                noise_fn_str="Normal(0, 1)", include_Q=False,
                include_RewardRate=False, dt=0.005, t_dur=3.0)
    base.update(ov)
    return MLEModelConfig(**base)


def _write(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def test_parse_filename_weights():
    assert _parse_filename(
        "mle_Classic_biasNone__Normal(0, 1)_3s_dt0.005") == ("mle", 1.0, 0.0)
    assert _parse_filename(
        "chisq_Classic_biasNone__Normal(0, 1)_3s_dt0.005") == ("chisq", 1.0, 0.0)
    assert _parse_filename(
        "mle_Classic_biasNone__Normal(0, 1)_3s_dt0.005_mleW1_chi2W0.5") == (
            "mle", 1.0, 0.5)


def test_extract_losses_mle_chisq_joint(tmp_path):
    _write(tmp_path / "mle_Classic_biasNone__Normal(0, 1)_3s_dt0.005.pkl", {
        "S1": dict(OptimRes=_optim(120.0), model_config=_mle_config(
            mle_choice_weight=2.0, mle_condition_columns=("ChoiceLeft",)),
            fit_finish_time="2026-06-01T10:00:00", dt=0.005, t_dur=3.0),
        "S2": dict(OptimRes=_optim(140.0), model_config=_mle_config(),
                   fit_finish_time="2026-06-01T11:00:00", dt=0.005, t_dur=3.0),
    })
    _write(tmp_path / "chisq_Classic_biasNone__Normal(0, 1)_3s_dt0.005.pkl", {
        "S1": dict(OptimRes=_optim(55.0), fit_mode="chisq",
                   fit_finish_time="2026-06-02T09:00:00", dt=0.005, t_dur=3.0),
    })
    _write(tmp_path / "mle_Classic_biasNone__Normal(0, 1)_3s_dt0.005_mleW1_chi2W0.5.pkl", {
        "S1": dict(OptimRes=_optim(1.8), model_config=_mle_config(
            mle_mle_weight=1.0, mle_chi2_weight=0.5),
            total_loss=1.8, mle_part_loss=1.0, chi2_part_loss=1.6,
            ref_mle=120.0, ref_chi2=55.0,
            fit_finish_time="2026-06-03T09:00:00", dt=0.005, t_dur=3.0),
    })
    # Files that must be skipped: not a fit-result name, and a non-dict payload.
    _write(tmp_path / "model_GUI_cache.pkl", {"not": "a fit"})
    _write(tmp_path / "mle_weird.pkl", [1, 2, 3])

    df = extract_losses(tmp_path)
    assert len(df) == 4  # 2 (mle) + 1 (chisq) + 1 (joint)

    s1_mle = df[(df.model_file.str.startswith("mle_"))
                & (df.subject == "S1")
                & (~df.model_file.str.contains("chi2W"))].iloc[0]
    assert s1_mle["loss"] == 120.0
    assert s1_mle["fit_mode"] == "mle"
    assert s1_mle["drift_fn"] == "Classic"
    assert s1_mle["mle_choice_weight"] == 2.0
    assert s1_mle["mle_condition_columns"] == "('ChoiceLeft',)"
    assert s1_mle["save_time"] == "2026-06-01T10:00:00"

    chisq = df[df.fit_mode == "chisq"].iloc[0]
    assert chisq["loss"] == 55.0
    assert chisq["drift_fn"] is None  # chisq payload has no model_config

    joint = df[df.model_file.str.contains("chi2W")].iloc[0]
    assert joint["mle_chi2_weight"] == 0.5
    assert joint["total_loss"] == 1.8
    assert joint["ref_mle"] == 120.0
    assert joint["ref_chi2"] == 55.0
