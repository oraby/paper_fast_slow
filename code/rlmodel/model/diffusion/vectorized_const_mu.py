"""Fast first-passage solver for constant-drift DDM variants."""
import numpy as np
from scipy.special import ndtr

from .result import FirstPassageResult


def diffusion_vectorized_const_mu(z, mu_scalar, sigma, bound, dt, dx, tmax, *, xp=np):
    """Compute first-passage densities for a scalar-drift DDM.

    This fast path precomputes the Gaussian transition kernel once and reuses
    it for every timestep. It returns the same public fields as the reference
    solver, but leaves the debug logging fields as None.
    """
    bound = float(bound)
    z = float(z)
    mu_scalar = float(mu_scalar)
    dt = float(dt)
    dx = float(dx)
    sigma = float(sigma)
    assert -bound <= z <= bound, f"z={z} must lie strictly inside (-{bound}, {bound})"
    assert sigma > 0, f"sigma={sigma} must be positive"

    sigma_sqrt_dt = sigma * float(np.sqrt(dt))
    n_x = int(round(2.0 * bound / dx))
    assert n_x >= 2, f"n_x={n_x} (need at least 2 bins); decrease dx"
    x_grid = xp.linspace(-bound + dx / 2.0, bound - dx / 2.0, n_x)

    n_t = int(round(tmax / dt))
    times = xp.arange(1, n_t + 1) * dt

    z_idx = int(np.argmin(np.abs(np.asarray(x_grid) - z)))
    p = np.zeros(n_x)
    p[z_idx] = 1.0

    mean_shift = mu_scalar * dt
    offsets = np.arange(-(n_x - 1), n_x, dtype=float)
    kernel_upper = ((offsets + 0.5) * dx - mean_shift) / sigma_sqrt_dt
    kernel_lower = ((offsets - 0.5) * dx - mean_shift) / sigma_sqrt_dt
    transition_kernel = ndtr(kernel_upper) - ndtr(kernel_lower)
    x_grid_np = np.asarray(x_grid)
    mass_above = 1.0 - ndtr((bound - x_grid_np - mean_shift) / sigma_sqrt_dt)
    mass_below = ndtr((-bound - x_grid_np - mean_shift) / sigma_sqrt_dt)

    upper_mass_by_t = np.zeros(n_t)
    lower_mass_by_t = np.zeros(n_t)
    survival = np.zeros(n_t)

    for i in range(n_t):
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
            "dt": dt, "dx": dx, "mu_is_scalar": True,
            "backend": "vectorized_const_mu",
        },
    )
