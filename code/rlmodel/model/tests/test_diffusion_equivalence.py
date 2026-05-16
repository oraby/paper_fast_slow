import numpy as np
import pytest

from ..first_passage import first_passage_density


DT = 0.005
DX = 0.02
TMAX = 0.75
BOUND = 1.0


def _assert_equivalent(actual, expected):
    np.testing.assert_allclose(
        actual.f_upper, expected.f_upper, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(
        actual.f_lower, expected.f_lower, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(
        actual.survival, expected.survival, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("z, mu, sigma", [
    (0.0, 0.0, 1.0),
    (0.2, 0.6, 1.0),
    (-0.3, -0.8, 1.2),
    (0.0, 1.5, 0.7),
])
def test_vectorized_const_mu_matches_reference(z, mu, sigma):
    reference = first_passage_density(
        z, mu, sigma, BOUND, DT, DX, TMAX, backend="reference")
    fast = first_passage_density(
        z, mu, sigma, BOUND, DT, DX, TMAX, backend="const_mu")
    auto = first_passage_density(
        z, mu, sigma, BOUND, DT, DX, TMAX, backend="auto")

    _assert_equivalent(fast, reference)
    _assert_equivalent(auto, reference)
    assert fast.metadata["backend"] == "vectorized_const_mu"
    assert auto.metadata["backend"] == "vectorized_const_mu"
    assert fast.p_by_t is None


@pytest.mark.parametrize("z, sigma, mu_start, mu_stop", [
    (0.0, 1.0, 0.8, 0.2),
    (0.2, 0.9, -0.4, 0.6),
    (-0.2, 1.3, 1.0, -0.5),
])
def test_vectorized_time_mu_matches_reference(z, sigma, mu_start, mu_stop):
    n_t = int(round(TMAX / DT))
    mu = np.linspace(mu_start, mu_stop, n_t)
    reference = first_passage_density(
        z, mu, sigma, BOUND, DT, DX, TMAX, backend="reference")
    fast = first_passage_density(
        z, mu, sigma, BOUND, DT, DX, TMAX, backend="time_mu")
    auto = first_passage_density(
        z, mu, sigma, BOUND, DT, DX, TMAX, backend="auto")

    _assert_equivalent(fast, reference)
    _assert_equivalent(auto, reference)
    assert fast.metadata["backend"] == "vectorized_time_mu"
    assert auto.metadata["backend"] == "vectorized_time_mu"
    assert fast.p_by_t is None


def test_reference_backend_keeps_logged_fields():
    r = first_passage_density(
        0.0, 0.5, 1.0, BOUND, DT, DX, TMAX, backend="reference")

    assert r.x_grid is not None
    assert r.p_by_t is not None
    assert r.upper_mass_by_t is not None
    assert r.lower_mass_by_t is not None
