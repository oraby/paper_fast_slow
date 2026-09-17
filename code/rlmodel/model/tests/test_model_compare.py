"""Tests for the model_compare engine (``rlmodel/model/compare.py``).

Covers the pure logic (filename→column classification + ordering, model-key
grouping, row gating) and a headless (Agg) grid-assembly render with the two
forward passes monkeypatched, so no behavior data or real simulation is needed.
"""
from __future__ import annotations

import pickle
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..mle_reeval import parse_fit_filename
from .. import compare
from ..compare import (classify_column, discover_fits, ColumnFit, ModelEntry,
                       plot_subject_model, _active_rows, _routed_kwargs)
from ..bias import BIAS_FN_DICT
from ..drift import DRIFT_FN_DICT
from ..noise import NOISE_FN_DICT


_BASE = "NoiseGain-RewardRate_biasQ-Val (Offset)_Normal(0, 1)_4.8s_dt0.005"


def _payload(include_Q=True, include_RewardRate=True, fun=100.0):
    return {"params_names": ["DRIFT_COEF"],
            "OptimRes": types.SimpleNamespace(x=np.array([1.0]), fun=fun),
            "include_Q": include_Q, "include_RewardRate": include_RewardRate}


# --------------------------------------------------------------------------
# Classification + ordering
# --------------------------------------------------------------------------
def test_classify_families_and_order():
    names = [
        f"chisq_NoiseGain-RewardRate_biasQ-Val (Offset)_Normal(0, 1)_4.8s_dt0.005_scaledB.pkl",
        f"mle_{_BASE}_mleW1_chi2W0.5.pkl",
        f"mle_{_BASE}.pkl",
        f"chisq_{_BASE}.pkl",
        f"mle_{_BASE}_mleW1_chi2W0.1.pkl",
    ]
    ranked = sorted((classify_column(parse_fit_filename(n)) for n in names),
                    key=lambda t: t[0])
    labels = [lbl for _, lbl in ranked]
    assert labels == ["MLE", "MLE=1, Chi²=0.1", "MLE=1, Chi²=0.5",
                      "Chi²-Noise", "Chi²-Bound"]


def test_noise_and_bound_chisq_share_model_key():
    a = parse_fit_filename(f"chisq_{_BASE}.pkl")
    b = parse_fit_filename(
        "chisq_Bound-RewardRate_biasQ-Val (Offset)_Normal(0, 1)_4.8s_dt0.005_scaledB.pkl")
    assert a.model_key == b.model_key           # grouped under one model
    assert classify_column(a)[1] == "Chi²-Noise"
    assert classify_column(b)[1] == "Chi²-Bound"


def test_mle_scaledb_and_joint_scaledb_skipped():
    assert classify_column(parse_fit_filename(f"mle_{_BASE}_scaledB.pkl")) is None
    assert classify_column(
        parse_fit_filename(f"mle_{_BASE}_scaledB_mleW1_chi2W0.5.pkl")) is None


def test_model_key_keeps_the_trailing_sym_for_the_metrics_caches():
    """The metrics caches are keyed on model_key strings (aggregate._spec_key),
    so the ``|sym`` left by the removed asymmetric-rate variants must stay."""
    assert parse_fit_filename(f"mle_{_BASE}.pkl").model_key.endswith("|sym")


# --------------------------------------------------------------------------
# Disk discovery
# --------------------------------------------------------------------------
def test_discover_groups_columns_in_order(tmp_path):
    files = {
        f"mle_{_BASE}.pkl": {"S1": _payload(), "S2": _payload()},
        f"chisq_{_BASE}.pkl": {"S1": _payload()},
        "chisq_Bound-RewardRate_biasQ-Val (Offset)_Normal(0, 1)_4.8s_dt0.005_scaledB.pkl":
            {"S1": _payload()},
        # a different model → its own entry
        "mle_Classic_biasQ-Val (Offset)_Normal(0, 1)_4.8s_dt0.005.pkl":
            {"S1": _payload()},
        # unrelated / skipped
        "loss_summary.pkl": {"not": "a fit"},
    }
    for name, obj in files.items():
        with open(tmp_path / name, "wb") as f:
            pickle.dump(obj, f)

    fits = discover_fits(tmp_path, verbose=False)
    # two models: the RewardRate model and the Classic one
    assert len(fits) == 2
    sym_key = parse_fit_filename(f"mle_{_BASE}.pkl").model_key
    entry = fits[sym_key]
    assert set(entry.subjects) == {"S1", "S2"}
    assert [c.column_label for c in entry.subjects["S1"]] == \
        ["MLE", "Chi²-Noise", "Chi²-Bound"]
    assert [c.column_label for c in entry.subjects["S2"]] == ["MLE"]


# --------------------------------------------------------------------------
# Param routing (frozen scale axis)
# --------------------------------------------------------------------------
def test_routing_noise_scaled_defaults_bound_to_one():
    fid = parse_fit_filename(f"mle_{_BASE}.pkl")
    params = {"DRIFT_COEF": 2.0, "NOISE_SIGMA": 1.5, "BIAS_COEF": 0.9,
              "ALPHA": 0.3, "BETA": 0.4, "NON_DECISION_TIME": 0.2,
              "Q_VAL_OFFSET": 0.1, "LAPSE_RATE": 0.02}
    r = _routed_kwargs(params, BIAS_FN_DICT[fid.bias], DRIFT_FN_DICT[fid.drift],
                       NOISE_FN_DICT[fid.noise])
    assert r["top_kwargs"]["NOISE_SIGMA"] == 1.5     # fitted
    assert r["top_kwargs"]["BOUND"] == 1.0           # frozen counterpart
    assert "LAPSE_RATE" not in r["top_kwargs"]       # MLE-only, not routed
    assert r["biasFn_kwargs"]["Q_VAL_OFFSET"] == 0.1


def test_routing_scaled_bound_defaults_noise_to_one():
    fid = parse_fit_filename(
        "chisq_Bound-RewardRate_biasQ-Val (Offset)_Normal(0, 1)_4.8s_dt0.005_scaledB.pkl")
    params = {"DRIFT_COEF": 2.0, "BOUND": 1.8, "BIAS_COEF": 0.9, "ALPHA": 0.3,
              "BETA": 0.4, "NON_DECISION_TIME": 0.2}
    r = _routed_kwargs(params, BIAS_FN_DICT[fid.bias], DRIFT_FN_DICT[fid.drift],
                       NOISE_FN_DICT[fid.noise])
    assert r["top_kwargs"]["BOUND"] == 1.8           # fitted
    assert r["top_kwargs"]["NOISE_SIGMA"] == 1.0     # frozen counterpart


# --------------------------------------------------------------------------
# Row gating
# --------------------------------------------------------------------------
def test_active_rows_header_always_and_learning_gates():
    rows = _active_rows({"losses_dist": False}, include_Q=True,
                        include_RewardRate=False)
    names = [r.name for r in rows]
    assert names[0] == "header"          # header always first
    assert "losses_dist" not in names     # flag respected
    assert "alpha_dist" in names          # Q-learning present
    assert "beta_dist" not in names       # no reward-rate → gated out


# --------------------------------------------------------------------------
# Headless grid assembly (forward passes monkeypatched)
# --------------------------------------------------------------------------
def _two_column_fits():
    fid_mle = parse_fit_filename("mle_Classic_biasNone__Normal(0, 1)_3s_dt0.005.pkl")
    fid_chi = parse_fit_filename("chisq_Classic_biasNone__Normal(0, 1)_3s_dt0.005.pkl")
    payload = _payload(include_Q=False, include_RewardRate=False)
    cols = [ColumnFit(fid_mle, payload, "a.pkl", "MLE", 0.0),
            ColumnFit(fid_chi, payload, "b.pkl", "Chi²-Noise", 100.0)]
    entry = ModelEntry(fid_mle.model_key, fid_mle.model_label, fid_mle,
                       {"S1": cols})
    return {fid_mle.model_key: entry}, fid_mle.model_key


def _noop_renderers():
    # Structural tests validate the grid scaffolding (axis counts / gating).
    # We patch the renderers to no-ops because this environment's matplotlib
    # crashes natively on real hist/bar drawing under the Agg backend (a
    # pre-existing delay-load DLL fault, unrelated to this code). Real drawing
    # is exercised in the notebook via the working Jupyter kernel backend.
    return {name: (lambda axes, ctx: None) for name in compare._ROW_RENDERERS}


def test_grid_axis_counts(monkeypatch):
    monkeypatch.setattr(compare, "_ROW_RENDERERS", _noop_renderers())
    monkeypatch.setattr(compare, "_compute_mle",
                        lambda *a, **k: (None, 111.0, 0.02, None))
    monkeypatch.setattr(compare, "_compute_sim",
                        lambda *a, **k: (None, 1.0, {}, None))
    fits, model_key = _two_column_fits()

    # header + losses_dist + hist_by_loss → 3 rows × 2 cols = 6 axes
    flags = {k: False for k in compare.DEFAULT_ROW_FLAGS}
    flags.update(losses_dist=True, hist_by_loss=True)
    fig = plot_subject_model(model_key, "S1", fits, None, row_flags=flags)
    assert len(fig.axes) == 6
    plt.close(fig)

    # header only → 1 row × 2 cols = 2 axes
    flags_off = {k: False for k in compare.DEFAULT_ROW_FLAGS}
    fig2 = plot_subject_model(model_key, "S1", fits, None, row_flags=flags_off)
    assert len(fig2.axes) == 2
    plt.close(fig2)


def test_grid_two_axis_row_uses_subgrid(monkeypatch):
    monkeypatch.setattr(compare, "_ROW_RENDERERS", _noop_renderers())
    monkeypatch.setattr(compare, "_compute_mle",
                        lambda *a, **k: (None, 111.0, None, None))
    monkeypatch.setattr(compare, "_compute_sim",
                        lambda *a, **k: (object(), 1.0, {}, None))
    fits, model_key = _two_column_fits()
    flags = {k: False for k in compare.DEFAULT_ROW_FLAGS}
    flags.update(rt_direction=True)
    # header (1) + rt_direction (2 sub-axes) → (1 + 2) * 2 cols = 6 axes
    fig = plot_subject_model(model_key, "S1", fits, None, row_flags=flags)
    assert len(fig.axes) == 6
    plt.close(fig)


def test_header_renderer_text_content():
    # The header renderer is text-only (no patches), so it's safe to exercise
    # for real here. Confirms the MLE-Score + λ lines are composed.
    from ..compare import _ColumnCtx, _row_header
    fid = parse_fit_filename("mle_Classic_biasNone__Normal(0, 1)_3s_dt0.005.pkl")
    fig, ax = plt.subplots()
    ctx = _ColumnCtx(subject="S1", fid=fid, column_label="MLE", include_Q=False,
                     include_RewardRate=False, t_dur=3.0, dt=0.005,
                     mle_score=123.4, lapse=0.02)
    _row_header([ax], ctx)
    joined = "\n".join(t.get_text() for t in ax.texts)
    assert "MLE-Score: 123.4" in joined
    assert "λ: 0.020" in joined
    plt.close(fig)
