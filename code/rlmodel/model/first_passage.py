"""Public wrapper API for the DDM first-passage solvers.

Dispatches to a specific backend in `model.diffusion.*`. Milestone 2 only
implements the reference backend (`diffusion_single`); Milestone 3 adds the
two vectorized fast paths (`vectorized_const_mu`, `vectorized_time_mu`) and
extends the `"auto"` dispatch to pick based on the shape of `mu`.
"""
import numpy as np

from .diffusion import FirstPassageResult, diffusion_single

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
        ``tmax / dt`` for time-varying drift (Decay-Q).
    sigma, bound, dt, dx, tmax : float
        DDM and discretization parameters.
    backend : str, default "auto"
        Backend selection. In Milestone 2, both ``"reference"`` and ``"auto"``
        route to ``diffusion_single``. Milestone 3 will add ``"const_mu"`` and
        ``"time_mu"`` and have ``"auto"`` choose based on the shape of ``mu``.
    xp : module, default numpy
        Array module. Placeholder for a future GPU swap (cupy); not exercised
        in the initial implementation.

    Returns
    -------
    FirstPassageResult
        Logged result from the reference backend (in Milestone 2); future
        backends may leave logging fields as ``None``.
    """
    if backend in ("reference", "auto"):
        return diffusion_single(z, mu, sigma, bound, dt, dx, tmax, xp=xp)
    raise ValueError(
        f"Unknown backend: {backend!r}. "
        "Milestone 2 supports 'reference' or 'auto'; "
        "Milestone 3 will add 'const_mu' and 'time_mu'.")
