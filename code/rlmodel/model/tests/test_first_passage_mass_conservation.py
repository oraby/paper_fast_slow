"""Mass-conservation tests for the first-passage solver.

For an absorbing-bound DDM, the integral of the two first-passage densities
plus the surviving mass at Tmax must equal 1:

    integral_0_Tmax f_upper(t) dt + integral_0_Tmax f_lower(t) dt + S(Tmax) = 1

The discrete propagator in diffusion_single uses exact Gaussian-CDF differences
per time step, so mass conservation holds to floating-point precision (~1e-14).
We assert 1e-6 as a comfortable margin.
"""
import numpy as np
import pytest

from ..first_passage import first_passage_density


# Common discretization (kept small so tests stay fast).
DT = 0.005
DX = 0.02
TMAX = 1.0
BOUND = 1.0


def _total_mass(result):
    dt = result.metadata["dt"]
    return (float(result.f_upper.sum()) * dt
            + float(result.f_lower.sum()) * dt
            + float(result.survival[-1]))


@pytest.mark.parametrize("z, mu, sigma", [
    (0.0, 0.5, 1.0),       # centered, positive drift
    (0.0, -0.5, 1.0),      # centered, negative drift
    (0.3, 0.0, 1.0),       # off-center, no drift
    (0.0, 0.0, 0.5),       # smaller noise
    (-0.5, 1.0, 1.5),      # off-center, strong drift, larger noise
])
def test_mass_conservation_scalar_mu(z, mu, sigma):
    r = first_passage_density(z, mu, sigma, BOUND, DT, DX, TMAX)
    total = _total_mass(r)
    assert abs(total - 1.0) < 1e-6, (
        f"z={z}, mu={mu}, sigma={sigma}: total mass {total} != 1.0 "
        f"(upper={r.f_upper.sum() * DT}, "
        f"lower={r.f_lower.sum() * DT}, "
        f"survival={r.survival[-1]})")


def test_mass_conservation_time_varying_mu():
    n_t = int(round(TMAX / DT))
    mu_array = np.linspace(0.8, 0.2, n_t)  # decaying drift
    r = first_passage_density(0.0, mu_array, 1.0, BOUND, DT, DX, TMAX)
    total = _total_mass(r)
    assert abs(total - 1.0) < 1e-6, (
        f"time-varying mu: total mass {total} != 1.0")


def test_logged_fields_populated_and_shape_consistent():
    r = first_passage_density(0.0, 0.5, 1.0, BOUND, DT, DX, TMAX)
    assert r.x_grid is not None
    assert r.p_by_t is not None
    assert r.upper_mass_by_t is not None
    assert r.lower_mass_by_t is not None

    n_t = r.times.shape[0]
    n_x = r.x_grid.shape[0]
    assert r.p_by_t.shape == (n_t, n_x), f"p_by_t shape {r.p_by_t.shape}"
    assert r.upper_mass_by_t.shape == (n_t,)
    assert r.lower_mass_by_t.shape == (n_t,)
    assert r.f_upper.shape == (n_t,)
    assert r.f_lower.shape == (n_t,)
    assert r.survival.shape == (n_t,)
    # Densities are absorbed-mass per dt.
    np.testing.assert_allclose(r.f_upper, r.upper_mass_by_t / DT, rtol=1e-12)
    np.testing.assert_allclose(r.f_lower, r.lower_mass_by_t / DT, rtol=1e-12)


def test_no_absorption_when_noise_is_tiny_and_drift_is_zero():
    """Sanity check: with tiny noise and zero drift over a short window, almost
    all mass should remain in the interior; minimal absorption."""
    r = first_passage_density(0.0, mu=0.0, sigma=0.01, bound=1.0,
                              dt=0.001, dx=0.01, tmax=0.5)
    total = _total_mass(r)
    assert abs(total - 1.0) < 1e-6
    assert r.survival[-1] > 0.99, (
        f"expected ~all mass to survive; got survival={r.survival[-1]}")


def test_drift_drives_mass_toward_upper_bound():
    """With strong positive drift, more mass should be absorbed at the upper
    bound than at the lower."""
    r = first_passage_density(0.0, mu=2.0, sigma=1.0, bound=1.0,
                              dt=0.005, dx=0.02, tmax=2.0)
    upper_total = float(r.f_upper.sum()) * 0.005
    lower_total = float(r.f_lower.sum()) * 0.005
    assert upper_total > lower_total, (
        f"upper={upper_total}, lower={lower_total}: expected upper > lower "
        f"with positive drift")
    # And total mass still conserved.
    assert abs(_total_mass(r) - 1.0) < 1e-6


def test_off_center_start_biases_absorption():
    """Start closer to upper bound -> more upper absorption with zero drift."""
    r = first_passage_density(0.5, mu=0.0, sigma=1.0, bound=1.0,
                              dt=0.005, dx=0.02, tmax=2.0)
    upper_total = float(r.f_upper.sum()) * 0.005
    lower_total = float(r.f_lower.sum()) * 0.005
    assert upper_total > lower_total, (
        f"upper={upper_total}, lower={lower_total}: starting at z=+0.5 should "
        f"absorb more at upper")
    assert abs(_total_mass(r) - 1.0) < 1e-6
