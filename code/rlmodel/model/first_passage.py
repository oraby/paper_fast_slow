"""Public wrapper API for the DDM first-passage solvers."""
import numpy as np

from .diffusion import (
    FirstPassageResult,
    diffusion_single,
    diffusion_vectorized_const_mu,
    diffusion_vectorized_time_mu,
)

__all__ = ["FirstPassageResult", "first_passage_density"]


def first_passage_density(z, mu, sigma, bound, dt, dx, tmax, *,
                          backend="auto", xp=np) -> FirstPassageResult:
    """Compute the first-passage density of a DDM with absorbing bounds.

    Parameters
    ----------
    z : float
        Starting point of the accumulator, in (-bound, +bound).
    mu : float | np.ndarray
        Drift. Scalar for constant-drift variants, or 1-D array of length
        ``tmax / dt`` for time-varying drift.
    sigma, bound, dt, dx, tmax : float
        DDM and discretization parameters.
    backend : str, default "auto"
        Backend selection. ``"reference"`` always routes to the logged
        reference solver. ``"const_mu"`` and ``"time_mu"`` force the fast
        scalar-drift or time-varying-drift paths. ``"auto"`` chooses based on
        whether ``mu`` is scalar.
    xp : module, default numpy
        Array module. Placeholder for a future GPU swap (cupy); not exercised
        in the initial implementation.

    Returns
    -------
    FirstPassageResult
        Fast backends leave logging fields as ``None``. The reference backend
        populates them for debugging and visualization.
    """
    mu_is_scalar = np.isscalar(mu) or (hasattr(mu, "ndim") and mu.ndim == 0)
    if backend == "reference":
        return diffusion_single(z, mu, sigma, bound, dt, dx, tmax, xp=xp)
    if backend == "const_mu":
        return diffusion_vectorized_const_mu(
            z, mu, sigma, bound, dt, dx, tmax, xp=xp)
    if backend == "time_mu":
        return diffusion_vectorized_time_mu(
            z, mu, sigma, bound, dt, dx, tmax, xp=xp)
    if backend == "auto":
        if mu_is_scalar:
            return diffusion_vectorized_const_mu(
                z, mu, sigma, bound, dt, dx, tmax, xp=xp)
        return diffusion_vectorized_time_mu(
            z, mu, sigma, bound, dt, dx, tmax, xp=xp)
    raise ValueError(
        f"Unknown backend: {backend!r}. "
        "Expected 'auto', 'reference', 'const_mu', or 'time_mu'.")
