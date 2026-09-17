"""Tests for the ``--scale-bound`` flag plumbing + filename suffix grammar.

The flag is independent of Bound-RewardRate: it swaps which of
(BOUND, NOISE_SIGMA) is the fitted scale axis. These tests cover the
mechanical bits — the rescaling math has its own equivalence test
in ``test_bound_rewardrate.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ..bias import BIAS_FN_DICT
from ..drift import DRIFT_FN_DICT
from ..fit import evolveFP, simulateDDM
from ..initvals import InitVals
from ..mle import (MLEModelConfig, objective_from_population,
                   objective_from_vector)
from ..noise import NOISE_FN_DICT
from ..state_updates import compute_starting_point_z


def test_evolveFP_suffix_grammar():
    """``_scaledB`` is appended after ``dt``; fixed-bound fits keep the legacy
    filename format —
    every existing pickle on disk loads under the same path.
    """
    base = dict(drift_fn_str="Classic", bias_fn_str="None_",
                noise_fn_str="Normal(0, 1)", t_dur=3, dt=0.005,
                is_loss_no_dir=False, fit_mode="mle")
    legacy = str(evolveFP(**base))
    assert legacy.endswith("_3s_dt0.005.pkl"), legacy

    scaled_b = str(evolveFP(**base, uses_scaled_bound=True))
    assert scaled_b.endswith("_3s_dt0.005_scaledB.pkl"), scaled_b

    composed = str(evolveFP(**base, uses_scaled_bound=True,
                            mle_chi2_weight=0.5))
    assert composed.endswith("_3s_dt0.005_scaledB_mleW1_chi2W0.5.pkl"), composed


def test_init_val_fields_describe_scale_pair_contract():
    """Sanity: the dataclass fields describe a non-trivial fittable
    BOUND range, a frozen ``_BOUND_FIXED``, a non-trivial fittable
    NOISE_SIGMA range, and a frozen ``_NOISE_FIXED``. ``fit.simulateDDM``
    applies the *_FIXED variants as InitVal overrides depending on
    Scale-How; this test pins the contract each field advertises.
    """
    iv = InitVals()
    d = iv.toDict()
    # Fittable axes: non-trivial range.
    assert d["BOUND"].Min < d["BOUND"].Max, (
        "InitVals.BOUND must have a non-trivial fit range "
        "(used when Scale-How=Bound).")
    assert d["NOISE_SIGMA"].Min < d["NOISE_SIGMA"].Max, (
        "InitVals.NOISE_SIGMA must have a non-trivial fit range "
        "(used when Scale-How=Noise — the legacy default).")
    # Frozen counterparts: degenerate range pinned at Default.
    for name in ("_BOUND_FIXED", "_NOISE_FIXED"):
        assert d[name].Min == d[name].Max, (
            f"InitVals.{name} must be frozen (Min == Max).")
        assert d[name].Default == d[name].Min


def test_init_val_override_swaps_in_initvals_dict():
    """Round-trip the override through ``InitVals`` to confirm the dict
    layer accepts ``_BOUND_FIXED`` / ``_NOISE_FIXED`` swapped onto
    the canonical ``BOUND`` / ``NOISE_SIGMA`` keys. This is the exact
    mechanism ``fit.simulateDDM`` uses to swap the fitted axis at fit
    time.
    """
    iv = InitVals()
    iv.override("BOUND", iv._BOUND_FIXED)
    iv.override("NOISE_SIGMA", iv._NOISE_FIXED)
    d = iv.toDict()
    assert d["BOUND"] == iv._BOUND_FIXED
    assert d["NOISE_SIGMA"] == iv._NOISE_FIXED


def test_init_vals_dict_exposes_all_four_scale_pair_fields():
    """The Scale-How GUI dropdown relies on ``InitVals.toDict()``
    auto-creating sliders for all four scale-pair fields so the user
    can tweak the active and frozen values independently.
    """
    keys = set(InitVals().toDict().keys())
    assert {"BOUND", "_BOUND_FIXED", "NOISE_SIGMA", "_NOISE_FIXED"} <= keys


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


def _padded_scale_bound_df():
    """Two padded sessions, three trials each (one invalid per session).

    Matches ``_two_session_padded_df`` from ``test_mle_smoke``; duplicated
    here to keep this test self-contained.
    """
    rows = []
    for sess_num in (1, 2):
        for trial_num, choice_left, reward, rt, dv, valid in [
            (1, 1.0, 1.0, 0.12,  0.7, True),
            (2, 0.0, 0.0, 0.14, -0.5, True),
            (3, np.nan, np.nan, np.nan, 0.2, False),
        ]:
            rows.append(dict(
                Name="S1", Date=pd.Timestamp("2026-01-01"),
                SessionNum=sess_num, TrialNumber=trial_num,
                SessId=f"S1_2026-01-01_{sess_num}",
                DV=dv, DVstr=str(dv), valid=valid,
                calcStimulusTime=rt, ChoiceLeft=choice_left,
                ChoiceCorrect=reward,
            ))
    return pd.DataFrame(rows)


def test_population_path_handles_scale_bound_without_per_candidate_fallback():
    """Under ``--scale-bound``, BOUND varies across DE candidates.

    Without the path-D rescale, ``objective_from_population`` would
    detect heterogeneous BOUND, divert to a per-candidate Python loop,
    and lose the single-batch GPU payoff. With the rescale (applied
    in ``_compute_latent_population_equal_sessions``), every candidate's
    effective bound becomes 1.0 and the whole generation goes through
    one solver call. This test exercises that path and confirms it
    completes with finite losses for each candidate.
    """
    config = MLEModelConfig(
        drift_fn_str="Classic", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False, include_RewardRate=False,
        dt=0.01, t_dur=0.2, dx=0.05,
        uses_scaled_bound=True,
    )
    params_names = np.array(
        ["DRIFT_COEF", "NOISE_SIGMA", "BOUND", "NON_DECISION_TIME"])
    # Heterogeneous BOUND — exactly the configuration that used to
    # divert objective_from_population into the per-candidate loop.
    candidates = np.array([
        [1.0, 1.0, 0.5, 0.02],
        [1.0, 1.0, 1.0, 0.02],
        [1.0, 1.0, 2.0, 0.02],
    ], dtype=float).T  # (n_params, S=3)
    df = _padded_scale_bound_df()

    pop_losses = objective_from_population(
        candidates, params_names, df, config)

    assert pop_losses.shape == (3,)
    assert np.all(np.isfinite(pop_losses))


def test_population_path_matches_per_candidate_for_unit_bound_under_scale_bound():
    """When BOUND=1.0, the path-D rescale is the identity (inv_b = 1.0)
    so the population path is bit-exact with per-candidate. Confirms
    the rescale block adds no spurious arithmetic for the no-op case.
    """
    config = MLEModelConfig(
        drift_fn_str="Classic", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False, include_RewardRate=False,
        dt=0.01, t_dur=0.2, dx=0.05,
        uses_scaled_bound=True,
    )
    params_names = np.array(
        ["DRIFT_COEF", "NOISE_SIGMA", "BOUND", "NON_DECISION_TIME"])
    candidates = np.array([
        [1.0, 1.0, 1.0, 0.02],
        [0.8, 1.2, 1.0, 0.03],
        [1.5, 0.9, 1.0, 0.01],
    ], dtype=float).T
    df = _padded_scale_bound_df()

    pop_losses = objective_from_population(
        candidates, params_names, df, config)
    per_candidate = np.array([
        objective_from_vector(candidates[:, i], params_names, df, config)
        for i in range(candidates.shape[1])
    ])

    np.testing.assert_allclose(pop_losses, per_candidate, rtol=0, atol=0)


def test_population_path_approximate_equivalence_with_varying_bound():
    """For varying BOUND under ``--scale-bound``, population and
    per-candidate paths solve the same continuous PDE on DIFFERENT
    discretization grids (per-candidate has bin width dx in original
    DDM-state; population has bin width dx in rescaled coords, i.e.
    B*dx in original). The rescale identity is exact in the
    continuous limit; at any finite dx the two paths agree to within
    discretization noise that shrinks with dx. This test uses a small
    BOUND spread around 1.0 to keep that noise bounded.
    """
    config = MLEModelConfig(
        drift_fn_str="Classic", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False, include_RewardRate=False,
        dt=0.005, t_dur=0.2, dx=0.02,
        uses_scaled_bound=True,
    )
    params_names = np.array(
        ["DRIFT_COEF", "NOISE_SIGMA", "BOUND", "NON_DECISION_TIME"])
    candidates = np.array([
        [1.0, 1.0, 0.9, 0.02],
        [1.0, 1.0, 1.0, 0.02],
        [1.0, 1.0, 1.1, 0.02],
    ], dtype=float).T
    df = _padded_scale_bound_df()

    pop_losses = objective_from_population(
        candidates, params_names, df, config)
    per_candidate = np.array([
        objective_from_vector(candidates[:, i], params_names, df, config)
        for i in range(candidates.shape[1])
    ])

    # rtol generous enough to swallow the discretization mismatch
    # between bin-width-dx-in-rescaled vs bin-width-dx-in-original.
    np.testing.assert_allclose(pop_losses, per_candidate, rtol=0.05, atol=0.5)


def test_population_path_scale_bound_off_is_unaffected_by_rescale():
    """When ``--scale-bound`` is off the rescale block is skipped, so the
    legacy bit-exact behavior is preserved. BOUND is uniform across
    candidates here (``fit.simulateDDM`` freezes it upstream); this
    test simulates that by passing the same BOUND in every candidate.
    """
    config = MLEModelConfig(
        drift_fn_str="Classic", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False, include_RewardRate=False,
        dt=0.01, t_dur=0.2, dx=0.05,
        uses_scaled_bound=False,
    )
    params_names = np.array(
        ["DRIFT_COEF", "NOISE_SIGMA", "BOUND", "NON_DECISION_TIME"])
    candidates = np.array([
        [1.0, 1.0, 1.0, 0.02],
        [0.8, 1.2, 1.0, 0.03],
        [1.5, 0.9, 1.0, 0.01],
    ], dtype=float).T
    df = _padded_scale_bound_df()

    pop_losses = objective_from_population(
        candidates, params_names, df, config)
    per_candidate = np.array([
        objective_from_vector(candidates[:, i], params_names, df, config)
        for i in range(candidates.shape[1])
    ])

    np.testing.assert_allclose(pop_losses, per_candidate, rtol=0, atol=0)


def test_bound_rewardrate_without_scale_bound_raises_soft_limit():
    """Bound-RewardRate drift without ``--scale-bound`` collapses to
    NoiseGain-RewardRate semantics (BOUND frozen at 1.0) with extra
    compute and a misleading filename. ``fit.simulateDDM`` raises
    ValueError tagged ``soft-limit`` so the user has to be explicit;
    a power user wanting the combination can comment out the guard.
    """
    bias_str = "None_"
    drift_str = "Bound-RewardRate"
    noise_str = "Normal(0, 1)"
    init_vals_dict = InitVals().toDict()

    with pytest.raises(ValueError, match=r"(?i)soft-limit"):
        simulateDDM(
            df=pd.DataFrame(),
            bounds_and_defaults=init_vals_dict,
            dt=0.005, t_dur=1.0,
            biasFn=BIAS_FN_DICT[bias_str],
            driftFn=DRIFT_FN_DICT[drift_str],
            noiseFn=NOISE_FN_DICT[noise_str],
            is_loss_no_dir=False,
            num_cpus=1,
            evolvs_res={},
            fit_mode="mle",
            bias_fn_str=bias_str,
            drift_fn_str=drift_str,
            scale_bound=False,
        )


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
