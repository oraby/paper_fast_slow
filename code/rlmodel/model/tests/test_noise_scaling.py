"""Verify _noiseNormal uses sigma*sqrt(dt) scaling, not sigma*dt.

The standard DDM increment is dx = mu*dt + sigma*sqrt(dt)*epsilon, so the
noise component sigma*sqrt(dt)*epsilon has variance sigma**2 * dt. The
unit-sigma noise array returned by _noiseNormal must therefore have
empirical variance ~ dt (the multiplication by sigma happens downstream
in drift.py).
"""
import numpy as np
import pytest

from .. import noise as noise_mod


@pytest.fixture(autouse=True)
def _seed_rng():
    noise_mod.rnd_default_rng = np.random.default_rng(seed=12345)
    noise_mod._last_noise_arr = None
    noise_mod._last_norm_size = None
    yield


@pytest.mark.parametrize("dt", [0.01, 0.001, 0.0001])
def test_noise_normal_variance_matches_sqrt_dt(dt):
    arr = noise_mod._noiseNormal(size=(100_000, 1), dt=dt)
    empirical_var = float(np.var(arr))
    rel_err = abs(empirical_var - dt) / dt
    assert rel_err < 0.05, (
        f"dt={dt}: expected Var ~ {dt}, got {empirical_var} "
        f"(rel err {rel_err:.3%}). Old buggy scaling would yield ~{dt**2}.")


def test_noise_normal_not_old_dt_scaling():
    dt = 0.01
    arr = noise_mod._noiseNormal(size=(100_000, 1), dt=dt)
    empirical_var = float(np.var(arr))
    old_buggy_var = dt ** 2
    assert empirical_var > 10 * old_buggy_var, (
        f"Variance {empirical_var} is close to dt^2={old_buggy_var}; "
        f"the sqrt(dt) fix may not be applied.")


def test_noise_normal_cached_path_matches_uncached():
    dt = 0.005
    arr_uncached = noise_mod._noiseNormal(size=(50_000, 1), dt=dt)
    arr_cached = noise_mod._noiseNormal(size=(50_000, 1), dt=dt)
    for label, arr in [("uncached", arr_uncached), ("cached", arr_cached)]:
        v = float(np.var(arr))
        rel_err = abs(v - dt) / dt
        assert rel_err < 0.05, (
            f"{label} path: Var {v} != dt={dt} (rel err {rel_err:.3%})")
