"""Reference DDM first-passage solver.

Solves the discrete approximation to:

    dx = mu * dt + sigma * dW_t

with absorbing bounds at +/- bound and starting point z. Probability mass that
crosses either bound during a time step is removed from the active distribution
and recorded as first-passage mass for that bound and time bin.

This is the reference implementation: clear, always-logging, intentionally
non-vectorized in spirit (uses one matmul per timestep for efficiency, but the
logic is straightforward and easy to follow). The fast paths in
`vectorized_const_mu.py` and `vectorized_time_mu.py` (Milestone 3) will use
this as their numerical oracle.
"""
import numpy as np
from scipy.special import ndtr

from .result import FirstPassageResult


def diffusion_single(z, mu, sigma, bound, dt, dx, tmax, *, xp=np):
    """Compute the first-passage density of a DDM with absorbing bounds.

    Always returns the fully logged ``FirstPassageResult``: per-step
    probability distribution over the accumulator state, plus the per-step
    absorbed mass at each bound. Intended as the equivalence-test oracle for
    the fast vectorized backends, and for visualization / debugging.

    Parameters
    ----------
    z : float
        Starting point of the accumulator. Must lie in (-bound, +bound).
    mu : float | np.ndarray
        Drift. Scalar for constant-drift variants. 1-D array of length
        ``tmax / dt`` for time-varying drift (Decay-Q variants); ``mu[i]`` is
        the drift applied during time step i.
    sigma : float
        Noise standard deviation. Stochastic increment per step is
        ``sigma * sqrt(dt) * epsilon``.
    bound : float
        Absorbing bound location; bounds are at +/- bound.
    dt : float
        Time-step size.
    dx : float
        Accumulator-axis bin width.
    tmax : float
        Maximum integration time. ``int(round(tmax / dt))`` steps are taken.
    xp : module, optional
        Array module (default: numpy). Placeholder for a future GPU swap
        (cupy). Note: the Gaussian CDF is currently computed via
        ``scipy.special.ndtr``; a non-numpy backend would need a CDF
        replacement (``0.5 * (1 + xp.erf(x / sqrt(2)))`` once xp.erf is
        available, or the backend's own ndtr).

    Returns
    -------
    FirstPassageResult
        With all logged fields (``x_grid``, ``p_by_t``, ``upper_mass_by_t``,
        ``lower_mass_by_t``) populated.
    """
    bound = float(bound)
    z = float(z)
    dt = float(dt)
    dx = float(dx)
    sigma = float(sigma)
    assert -bound < z < bound, f"z={z} must lie strictly inside (-{bound}, {bound})"
    assert sigma > 0, f"sigma={sigma} must be positive"

    sqrt_dt = float(np.sqrt(dt))
    sigma_sqrt_dt = sigma * sqrt_dt

    # Discretize the accumulator axis: n_x interior bins of width dx, centered
    # symmetrically about 0, spanning (-bound, +bound). Bin edges land exactly
    # at +/- bound (the absorbing locations).
    n_x = int(round(2.0 * bound / dx))
    assert n_x >= 2, f"n_x={n_x} (need at least 2 bins); decrease dx"
    x_grid = xp.linspace(-bound + dx / 2.0, bound - dx / 2.0, n_x)
    edges = xp.linspace(-bound, bound, n_x + 1)

    # Discretize time.
    n_t = int(round(tmax / dt))
    times = xp.arange(1, n_t + 1) * dt

    # mu can be a scalar or per-timestep array.
    mu_is_scalar = np.isscalar(mu) or (hasattr(mu, "ndim") and mu.ndim == 0)
    if mu_is_scalar:
        mu_scalar = float(mu)
        mu_arr = None
    else:
        mu_arr = np.asarray(mu, dtype=float)
        assert mu_arr.shape == (n_t,), (
            f"time-varying mu shape {mu_arr.shape} != ({n_t},)")
        mu_scalar = None

    # Initialize the probability distribution: all mass at the bin whose center
    # is closest to z.
    z_idx = int(np.argmin(np.abs(np.asarray(x_grid) - z)))
    p = xp.zeros(n_x)
    p[z_idx] = 1.0

    # Logged outputs.
    upper_mass_by_t = xp.zeros(n_t)
    lower_mass_by_t = xp.zeros(n_t)
    p_by_t = xp.zeros((n_t, n_x))

    # Main propagation loop. At each step, for every source bin s, compute the
    # discrete Gaussian transition kernel into the interior target bins and
    # into the two absorbing regions (x < -bound, x > +bound). Mass conservation
    # holds exactly per step (modulo floating-point round-off) because the CDF
    # at +/-inf is 1, 0, and bin masses are CDF differences over a partition.
    for i in range(n_t):
        mu_t = mu_scalar if mu_is_scalar else float(mu_arr[i])
        mean_shift = mu_t * dt

        # Standardize each edge relative to (source center + drift shift).
        # diff[s, k] = edges[k] - x_grid[s] - mean_shift
        diff = edges[np.newaxis, :] - x_grid[:, np.newaxis] - mean_shift
        z_edges = diff / sigma_sqrt_dt  # shape (n_x, n_x + 1)

        cdf_at_edges = ndtr(np.asarray(z_edges))  # shape (n_x, n_x + 1)

        # Probability of landing in each interior target bin from each source.
        bin_mass = cdf_at_edges[:, 1:] - cdf_at_edges[:, :-1]  # (n_x, n_x)
        # Probability of being absorbed past each bound.
        mass_above = 1.0 - cdf_at_edges[:, -1]  # (n_x,)
        mass_below = cdf_at_edges[:, 0]         # (n_x,)

        # Propagate. bin_mass[s, t] = P(target=t | source=s);
        # new_p[t] = sum_s p[s] * bin_mass[s, t].
        p_np = np.asarray(p)
        new_p = bin_mass.T @ p_np
        upper_abs = float(mass_above @ p_np)
        lower_abs = float(mass_below @ p_np)

        p = xp.asarray(new_p)
        upper_mass_by_t[i] = upper_abs
        lower_mass_by_t[i] = lower_abs
        p_by_t[i] = p

    # Densities = absorbed mass per bin divided by bin width dt.
    f_upper = upper_mass_by_t / dt
    f_lower = lower_mass_by_t / dt
    # Survival = total remaining mass over the interior at each step.
    survival = p_by_t.sum(axis=1)

    return FirstPassageResult(
        times=times,
        f_upper=f_upper,
        f_lower=f_lower,
        survival=survival,
        x_grid=x_grid,
        p_by_t=p_by_t,
        upper_mass_by_t=upper_mass_by_t,
        lower_mass_by_t=lower_mass_by_t,
        metadata={
            "n_x": n_x, "n_t": n_t, "z": z, "sigma": sigma, "bound": bound,
            "dt": dt, "dx": dx, "mu_is_scalar": mu_is_scalar,
            "backend": "single",
        },
    )
