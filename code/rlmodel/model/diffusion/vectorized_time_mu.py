"""Fast first-passage solver for time-varying-drift DDM variants."""
import numpy as np
from scipy.special import ndtr

from .result import FirstPassageResult


def diffusion_vectorized_time_mu(z, mu_array, sigma, bound, dt, dx, tmax, *, xp=np):
    """Compute first-passage densities for a time-varying-drift DDM.

    The transition matrix changes at each timestep because drift changes, so
    this backend recomputes the Gaussian kernel per step. It avoids the
    reference solver's full probability-distribution log.
    """
    bound = float(bound)
    z = float(z)
    dt = float(dt)
    dx = float(dx)
    sigma = float(sigma)
    assert -bound < z < bound, f"z={z} must lie strictly inside (-{bound}, {bound})"
    assert sigma > 0, f"sigma={sigma} must be positive"

    n_t = int(round(tmax / dt))
    mu_array = np.asarray(mu_array, dtype=float)
    assert mu_array.shape == (n_t,), (
        f"time-varying mu shape {mu_array.shape} != ({n_t},)")

    sigma_sqrt_dt = sigma * float(np.sqrt(dt))
    n_x = int(round(2.0 * bound / dx))
    assert n_x >= 2, f"n_x={n_x} (need at least 2 bins); decrease dx"
    x_grid = xp.linspace(-bound + dx / 2.0, bound - dx / 2.0, n_x)
    times = xp.arange(1, n_t + 1) * dt

    z_idx = int(np.argmin(np.abs(np.asarray(x_grid) - z)))
    p = np.zeros(n_x)
    p[z_idx] = 1.0

    upper_mass_by_t = np.zeros(n_t)
    lower_mass_by_t = np.zeros(n_t)
    survival = np.zeros(n_t)
    offsets = np.arange(-(n_x - 1), n_x, dtype=float)
    x_grid_np = np.asarray(x_grid)

    for i, mu_t in enumerate(mu_array):
        mean_shift = float(mu_t) * dt
        kernel_upper = ((offsets + 0.5) * dx - mean_shift) / sigma_sqrt_dt
        kernel_lower = ((offsets - 0.5) * dx - mean_shift) / sigma_sqrt_dt
        transition_kernel = ndtr(kernel_upper) - ndtr(kernel_lower)
        mass_above = 1.0 - ndtr((bound - x_grid_np - mean_shift) / sigma_sqrt_dt)
        mass_below = ndtr((-bound - x_grid_np - mean_shift) / sigma_sqrt_dt)

        full_conv = np.convolve(p, transition_kernel)
        new_p = full_conv[n_x - 1:2 * n_x - 1]
        upper_mass_by_t[i] = mass_above @ p
        lower_mass_by_t[i] = mass_below @ p
        survival[i] = new_p.sum()
        p = new_p

    return FirstPassageResult(
        times=xp.asarray(times),
        f_upper=xp.asarray(upper_mass_by_t / dt),
        f_lower=xp.asarray(lower_mass_by_t / dt),
        survival=xp.asarray(survival),
        metadata={
            "n_x": n_x, "n_t": n_t, "z": z, "sigma": sigma, "bound": bound,
            "dt": dt, "dx": dx, "mu_is_scalar": False,
            "backend": "vectorized_time_mu",
        },
    )
