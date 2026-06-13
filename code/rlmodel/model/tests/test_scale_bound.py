"""Tests for the ``--scale-bound`` flag plumbing + filename suffix grammar.

The flag is independent of Bound-RewardRate: it swaps which of
(BOUND, NOISE_SIGMA) is the fitted scale axis. These tests cover the
mechanical bits — the rescaling math has its own equivalence test
in ``test_bound_rewardrate.py``.
"""
from __future__ import annotations

import numpy as np

from ..fit import evolveFP
from ..initvals import (
    InitVals,
    _BOUND_WHEN_SCALED,
    _NOISE_WHEN_SCALED,
)
from ..state_updates import compute_starting_point_z


def test_evolveFP_suffix_grammar():
    """``_scaledB`` follows ``_asymQ/_asymRR/_asymQRR`` (or stands alone).

    Symmetric / fixed-bound fits keep the legacy filename format —
    every existing pickle on disk loads under the same path.
    """
    base = dict(drift_fn_str="Classic", bias_fn_str="None_",
                noise_fn_str="Normal(0, 1)", t_dur=3, dt=0.005,
                is_loss_no_dir=False, fit_mode="mle")
    legacy = str(evolveFP(**base))
    assert legacy.endswith("_3s_dt0.005.pkl"), legacy

    scaled_b = str(evolveFP(**base, uses_scaled_bound=True))
    assert scaled_b.endswith("_3s_dt0.005_scaledB.pkl"), scaled_b

    asym_q   = str(evolveFP(**base, uses_asym_q=True))
    assert asym_q.endswith("_3s_dt0.005_asymQ.pkl"), asym_q

    composed = str(evolveFP(**base, uses_asym_q=True,
                             uses_scaled_bound=True))
    assert composed.endswith("_3s_dt0.005_asymQ_scaledB.pkl"), composed

    composed_qrr = str(evolveFP(**base, uses_asym_q=True, uses_asym_rr=True,
                                 uses_scaled_bound=True))
    assert composed_qrr.endswith("_3s_dt0.005_asymQRR_scaledB.pkl"), (
        composed_qrr)


def test_init_val_constants_swap_makes_bound_fittable():
    """Sanity: the private constants describe a non-trivial BOUND range
    and a frozen NOISE_SIGMA. ``fit.simulateDDM`` applies them as an
    override on the ``InitVals`` dict; this test pins the contract
    those constants advertise.
    """
    assert _BOUND_WHEN_SCALED.Min < _BOUND_WHEN_SCALED.Max, (
        "BOUND must have a non-trivial fit range under --scale-bound")
    assert _NOISE_WHEN_SCALED.Min == _NOISE_WHEN_SCALED.Max, (
        "NOISE_SIGMA must be frozen (Min == Max) under --scale-bound")
    # The default value in the frozen pair sits inside [Min, Max] by
    # construction; tighten the assertion to "exactly at the pin"
    # since that's the design intent.
    assert _NOISE_WHEN_SCALED.Default == _NOISE_WHEN_SCALED.Min


def test_init_val_override_swaps_in_initvals_dict():
    """Round-trip the override through ``InitVals`` to confirm the dict
    layer accepts it. This is the exact mechanism ``fit.simulateDDM``
    uses to swap the fitted axis at fit time.
    """
    iv = InitVals()
    iv.override("BOUND", _BOUND_WHEN_SCALED)
    iv.override("NOISE_SIGMA", _NOISE_WHEN_SCALED)
    d = iv.toDict()
    assert d["BOUND"] == _BOUND_WHEN_SCALED
    assert d["NOISE_SIGMA"] == _NOISE_WHEN_SCALED


def test_compute_starting_point_z_absolute_mode():
    """``bound`` kwarg switches the clip to absolute DDM-state units —
    used under ``--scale-bound`` where BOUND varies per candidate.
    """
    # Legacy (no bound kwarg): clipped to [-1, 1].
    z_legacy = compute_starting_point_z(
        q_left=0.9, q_right=0.1, delta=2.0, offset=0.0, include_Q=True)
    assert -1.0 <= float(z_legacy) <= 1.0

    # Absolute mode with bound=0.5: clip is [-0.5, +0.5].
    z_abs = compute_starting_point_z(
        q_left=0.9, q_right=0.1, delta=2.0, offset=0.0, include_Q=True,
        bound=0.5)
    assert -0.5 <= float(z_abs) <= 0.5

    # With wider bound, the bias can span a wider absolute range.
    z_wide = compute_starting_point_z(
        q_left=0.9, q_right=0.1, delta=2.0, offset=0.0, include_Q=True,
        bound=2.0)
    assert -2.0 <= float(z_wide) <= 2.0

    # include_Q=False still short-circuits to 0 regardless of bound.
    z_off = compute_starting_point_z(
        q_left=0.9, q_right=0.1, delta=2.0, offset=0.0, include_Q=False,
        bound=2.0)
    assert z_off == 0.0


def test_compute_starting_point_z_absolute_array_bound():
    """``bound`` may be a broadcast-compatible array (per-trial under
    the vectorized population path).
    """
    q_left = np.array([0.9, 0.5, 0.1])
    q_right = np.array([0.1, 0.5, 0.9])
    bounds = np.array([0.3, 1.0, 2.0])
    z = compute_starting_point_z(
        q_left=q_left, q_right=q_right,
        delta=2.0, offset=0.0, include_Q=True, bound=bounds)
    z = np.asarray(z)
    assert np.all(np.abs(z) <= bounds)
