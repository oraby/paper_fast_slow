"""Batched choice/RT likelihood for MLE fitting."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .array_backend import asnumpy
from .mle_likelihood import LOGLIK_FLOOR


@dataclass
class BatchedLikelihoodResult:
    loglik: np.ndarray
    choice_prob_or_density: np.ndarray
    decision_time: np.ndarray
    survival_at_tmax: np.ndarray
    upper_hit_prob_tmax: np.ndarray
    lower_hit_prob_tmax: np.ndarray
    metadata: dict = field(default_factory=dict)


@dataclass
class _BatchedSolverResult:
    """Solver output kept on the array backend (xp).

    Carries (b, n_t) densities and (b,) scalars for the ``b`` valid-solver
    trials. The caller is responsible for the single ``asnumpy`` transfer
    after gathering at the per-trial decision indices — that single transfer
    is the only sync per batch, instead of one transfer per output array.

    For ``xp=numpy`` this is just a no-op pass-through; for ``xp=cupy`` it is
    the difference between five ``(b, n_t)`` GPU->CPU copies per batch and one
    ``(b,)`` copy at the end of the gather.
    """
    upper_density_xp: object   # (b, n_t) on xp
    lower_density_xp: object   # (b, n_t) on xp
    survival_xp: object        # (b,) on xp
    upper_prob_xp: object      # (b,) on xp
    lower_prob_xp: object      # (b,) on xp
    valid_idx: np.ndarray      # (b,) numpy indices into the full batch
    metadata: dict


class BatchedDiffusionSolver:
    """Reusable batched first-passage solver.

    The solver uses FFT convolution over each bucket of trials sharing the same
    per-step drift and sigma. That avoids the previous dense transition tensor
    with shape ``(batch, n_x, n_x + 1)`` at every timestep.
    """

    def __init__(self, xp=np, normal_cdf=None):
        self.xp = xp
        self.normal_cdf = normal_cdf
        self._shape_key = None
        self.x_grid = None
        self.offsets = None
        self.fft_n = None

    def ensure_shape(self, bound, dx, n_x):
        shape_key = (float(bound), float(dx), int(n_x), self.xp.__name__)
        if shape_key == self._shape_key:
            return
        self._shape_key = shape_key
        self.x_grid = self.xp.linspace(
            -bound + dx / 2.0, bound - dx / 2.0, n_x)
        self.offsets = self.xp.asarray(
            np.arange(-(n_x - 1), n_x, dtype=float))
        self.fft_n = _next_power_of_two(3 * n_x - 2)

    def solve(self, z, mu, sigma, valid_for_loss, bound, dt, dx, tmax):
        n_trials = len(valid_for_loss)
        n_t = int(round(float(tmax) / float(dt)))
        n_x = int(round(2.0 * float(bound) / float(dx)))
        if n_x < 2:
            raise ValueError(f"n_x={n_x} (need at least 2 bins); decrease dx")
        self.ensure_shape(float(bound), float(dx), n_x)

        mu_matrix = _as_mu_matrix(mu, n_trials, n_t)
        finite_mu = np.all(np.isfinite(mu_matrix), axis=1)
        valid_solver = (
            np.asarray(valid_for_loss, dtype=bool)
            & np.isfinite(z)
            & np.isfinite(sigma)
            & finite_mu
            & (sigma > 0)
            & (z >= -bound)
            & (z <= bound)
        )

        valid_idx = np.flatnonzero(valid_solver)
        b = valid_idx.size
        base_meta = {
            "solver": "bucketed_fft",
            "bucket_count": 0,
            "batch_size": int(n_trials),
            "n_x": int(n_x),
            "n_t": int(n_t),
            "backend": self.xp.__name__,
        }
        if b == 0:
            return _BatchedSolverResult(
                upper_density_xp=self.xp.zeros((0, n_t), dtype=float),
                lower_density_xp=self.xp.zeros((0, n_t), dtype=float),
                survival_xp=self.xp.zeros(0, dtype=float),
                upper_prob_xp=self.xp.zeros(0, dtype=float),
                lower_prob_xp=self.xp.zeros(0, dtype=float),
                valid_idx=valid_idx,
                metadata=base_meta,
            )

        x_grid_cpu = np.linspace(-bound + dx / 2.0, bound - dx / 2.0, n_x)
        z_idx = np.argmin(np.abs(x_grid_cpu[None, :] - z[valid_idx, None]), axis=1)

        p = self.xp.zeros((b, n_x), dtype=float)
        p[self.xp.arange(b), self.xp.asarray(z_idx)] = 1.0
        mu_valid = np.asarray(mu_matrix[valid_idx], dtype=float)
        sigma_valid = np.asarray(sigma[valid_idx], dtype=float)

        upper_density_valid = self.xp.zeros((b, n_t), dtype=float)
        lower_density_valid = self.xp.zeros((b, n_t), dtype=float)
        upper_prob_valid = self.xp.zeros(b, dtype=float)
        lower_prob_valid = self.xp.zeros(b, dtype=float)
        bucket_count = 0

        for t_idx in range(n_t):
            new_p = self.xp.empty_like(p)
            keys = _bucket_keys(mu_valid[:, t_idx], sigma_valid)
            for key in np.unique(keys):
                local = np.flatnonzero(keys == key)
                mu_t = float(mu_valid[local[0], t_idx])
                sigma_t = float(sigma_valid[local[0]])
                kernel, mass_above, mass_below = self._transition_terms(
                    mu_t, sigma_t, bound, dt, dx)

                local_xp = self.xp.asarray(local)
                p_sub = p[local_xp]
                upper_abs = p_sub @ mass_above
                lower_abs = p_sub @ mass_below
                upper_prob_valid[local_xp] += upper_abs
                lower_prob_valid[local_xp] += lower_abs
                upper_density_valid[local_xp, t_idx] = upper_abs / dt
                lower_density_valid[local_xp, t_idx] = lower_abs / dt
                new_p[local_xp] = self._convolve_rows(p_sub, kernel, n_x)
                bucket_count += 1
            p = new_p

        survival_valid = self.xp.sum(p, axis=1)
        # NOTE: deliberately no `asnumpy` here. Keeping the (b, n_t) density
        # tensors on the xp backend lets the caller gather at decision indices
        # on-device and pay only a single (b,) GPU->CPU copy per batch instead
        # of three (b, n_t) copies. For xp=numpy this is a no-op pass-through.
        meta = dict(base_meta)
        meta["bucket_count"] = int(bucket_count)
        return _BatchedSolverResult(
            upper_density_xp=upper_density_valid,
            lower_density_xp=lower_density_valid,
            survival_xp=survival_valid,
            upper_prob_xp=upper_prob_valid,
            lower_prob_xp=lower_prob_valid,
            valid_idx=valid_idx,
            metadata=meta,
        )

    def _transition_terms(self, mu_t, sigma_t, bound, dt, dx):
        sigma_sqrt_dt = sigma_t * float(np.sqrt(dt))
        mean_shift = mu_t * dt
        kernel_upper = ((self.offsets + 0.5) * dx - mean_shift) / sigma_sqrt_dt
        kernel_lower = ((self.offsets - 0.5) * dx - mean_shift) / sigma_sqrt_dt
        transition_kernel = (
            self.normal_cdf(kernel_upper) - self.normal_cdf(kernel_lower)
        )
        mass_above = 1.0 - self.normal_cdf(
            (bound - self.x_grid - mean_shift) / sigma_sqrt_dt)
        mass_below = self.normal_cdf(
            (-bound - self.x_grid - mean_shift) / sigma_sqrt_dt)
        return transition_kernel, mass_above, mass_below

    def _convolve_rows(self, p_sub, kernel, n_x):
        p_pad = self.xp.zeros((p_sub.shape[0], self.fft_n), dtype=float)
        k_pad = self.xp.zeros(self.fft_n, dtype=float)
        p_pad[:, :n_x] = p_sub
        k_pad[:2 * n_x - 1] = kernel
        full = self.xp.fft.irfft(
            self.xp.fft.rfft(p_pad, axis=1)
            * self.xp.fft.rfft(k_pad)[None, :],
            n=self.fft_n,
            axis=1,
        )
        return full[:, n_x - 1:2 * n_x - 1]


def batched_choice_rt_loglik(observed_choice_left, observed_rt, no_choice,
                             valid_for_loss, z, mu_values, sigma, bound,
                             non_decision_time, dt, dx, tmax, *, xp=np,
                             normal_cdf=None, batch_size=None,
                             solver=None):
    """Evaluate choice/RT likelihoods for many trials on one array backend."""
    _validate_global_params(bound, dt, dx, tmax, batch_size)
    n_trials = len(valid_for_loss)
    out = _empty_result(n_trials)
    if n_trials == 0:
        return out

    observed_choice_left = np.asarray(observed_choice_left, dtype=float)
    observed_rt = np.asarray(observed_rt, dtype=float)
    no_choice = np.asarray(no_choice, dtype=bool)
    valid_for_loss = np.asarray(valid_for_loss, dtype=bool)
    z = np.asarray(z, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    non_decision_time = np.asarray(non_decision_time, dtype=float)

    decision_time = observed_rt - non_decision_time
    out.decision_time[:] = decision_time
    out.decision_time[no_choice] = np.nan

    if normal_cdf is None:
        from scipy.special import ndtr
        normal_cdf = lambda value: ndtr(np.asarray(value))
    if solver is None:
        solver = BatchedDiffusionSolver(xp=xp, normal_cdf=normal_cdf)

    metadata = {
        "solver": "bucketed_fft",
        "backend": getattr(xp, "__name__", "array_backend"),
        "batch_size": int(batch_size) if batch_size is not None else None,
        "bucket_count": 0,
    }
    print(f"Evaluating {n_trials:,} trials with batch size {batch_size:,}...")
    for start in range(0, n_trials, int(batch_size)):
        print(f"Processing batch starting at index {start}...")
        stop = min(start + int(batch_size), n_trials)
        sl = slice(start, stop)
        _evaluate_batch(
            result=out,
            observed_choice_left=observed_choice_left[sl],
            no_choice=no_choice[sl],
            valid_for_loss=valid_for_loss[sl],
            z=z[sl],
            mu_values=mu_values[sl] if getattr(mu_values, "ndim", 1) == 1 else mu_values[sl, :],
            sigma=sigma[sl],
            bound=float(bound),
            decision_time=decision_time[sl],
            dt=float(dt),
            dx=float(dx),
            tmax=float(tmax),
            offset=start,
            solver=solver,
        )
        batch_meta = out.metadata.pop("_last_batch", {})
        metadata["bucket_count"] += int(batch_meta.get("bucket_count", 0))
        metadata["n_x"] = batch_meta.get("n_x")
        metadata["n_t"] = batch_meta.get("n_t")
    out.metadata.update(metadata)
    return out


def estimate_batch_size_for_memory(memory_gb, bound, dx, tmax, dt,
                                   dtype_bytes=8):
    """Estimate a conservative batch size for the bucketed FFT solver."""
    if memory_gb is None:
        return None, None
    n_x = int(round(2.0 * float(bound) / float(dx)))
    n_t = int(round(float(tmax) / float(dt)))
    fft_n = _next_power_of_two(3 * n_x - 2)
    # Per trial: two density traces plus several active/FFT work buffers. This
    # is conservative and intentionally reader-visible because GPU memory use is
    # dominated by shape, not Python object count.
    per_trial_bytes = dtype_bytes * (2 * n_t + 6 * fft_n + 6 * n_x)
    available_bytes = float(memory_gb) * (1024 ** 3)
    batch_size = max(int(available_bytes // max(per_trial_bytes, 1)), 1)
    return batch_size, {
        "requested_memory_gb": float(memory_gb),
        "estimated_bytes_per_trial": int(per_trial_bytes),
        "estimated_total_bytes": int(batch_size * per_trial_bytes),
        "n_x": int(n_x),
        "n_t": int(n_t),
        "fft_n": int(fft_n),
    }


def _evaluate_batch(result, observed_choice_left, no_choice, valid_for_loss, z,
                    mu_values, sigma, bound, decision_time, dt, dx, tmax,
                    offset, solver):
    n_t = int(round(float(tmax) / float(dt)))
    mu_matrix = _as_mu_matrix(mu_values, len(valid_for_loss), n_t)
    valid_solver = (
        valid_for_loss
        & np.isfinite(z)
        & np.isfinite(sigma)
        & np.all(np.isfinite(mu_matrix), axis=1)
        & (sigma > 0)
        & (z >= -bound)
        & (z <= bound)
    )
    solver_result = solver.solve(
        z, mu_matrix, sigma, valid_solver, bound, dt, dx, tmax)
    valid_idx = solver_result.valid_idx
    b = valid_idx.size
    metadata = solver_result.metadata

    # All output positions default to LOGLIK_FLOOR. Trials in valid_for_loss
    # but not valid_solver (NaN inputs, sigma<=0, z out of bounds) keep this
    # default — preserves the prior semantics.
    likelihood = np.full(len(valid_for_loss), LOGLIK_FLOOR, dtype=float)
    survival_cpu_full = np.full(len(valid_for_loss), LOGLIK_FLOOR, dtype=float)
    upper_prob_full = np.full(len(valid_for_loss), np.nan, dtype=float)
    lower_prob_full = np.full(len(valid_for_loss), np.nan, dtype=float)

    if b > 0:
        # Build per-valid-trial gather inputs on CPU (small, cheap).
        decision_time_v = decision_time[valid_idx]
        choice_left_v = observed_choice_left[valid_idx]
        no_choice_v = no_choice[valid_idx]
        valid_decision_v = (
            ~no_choice_v
            & np.isfinite(choice_left_v)
            & np.isfinite(decision_time_v)
            & (decision_time_v > 0)
            & (decision_time_v <= tmax)
        )
        # decision_idx_v is bounded into [0, n_t-1]; for invalid-decision
        # rows we fill in below via xp.where, so the index value is harmless.
        decision_idx_v = np.clip(
            np.ceil(np.where(valid_decision_v,
                             decision_time_v, dt) / dt).astype(int) - 1,
            0, n_t - 1)

        # One CPU->xp transfer per small (b,) array. These are tiny compared
        # to the (b, n_t) density tensors that are no longer transferred.
        xp = solver.xp
        trial_idx_xp = xp.arange(b)
        decision_idx_xp = xp.asarray(decision_idx_v)
        is_left_xp = xp.asarray((choice_left_v == 1))
        no_choice_xp = xp.asarray(no_choice_v)
        valid_decision_xp = xp.asarray(valid_decision_v)

        # Gather densities at each trial's decision-time bin on xp.
        upper_at_t_xp = solver_result.upper_density_xp[trial_idx_xp,
                                                       decision_idx_xp]
        lower_at_t_xp = solver_result.lower_density_xp[trial_idx_xp,
                                                       decision_idx_xp]
        density_at_t_xp = xp.where(is_left_xp, upper_at_t_xp, lower_at_t_xp)

        # No-choice trials contribute the survival mass at tmax. Trials that
        # are valid_solver but have neither a valid decision nor are flagged
        # no_choice fall through to LOGLIK_FLOOR.
        like_xp = xp.where(no_choice_xp,
                           solver_result.survival_xp,
                           density_at_t_xp)
        like_xp = xp.where(no_choice_xp | valid_decision_xp,
                           like_xp,
                           LOGLIK_FLOOR)
        # Clamp non-finite/non-positive likelihoods to the floor on xp.
        like_xp = xp.where(xp.isfinite(like_xp) & (like_xp > 0.0),
                           like_xp, LOGLIK_FLOOR)

        # The ONLY GPU->CPU copies in this batch: four (b,) scalar arrays.
        # For xp=numpy these are no-op pass-throughs.
        like_v = asnumpy(xp, like_xp)
        survival_v = asnumpy(xp, solver_result.survival_xp)
        upper_prob_v = asnumpy(xp, solver_result.upper_prob_xp)
        lower_prob_v = asnumpy(xp, solver_result.lower_prob_xp)

        likelihood[valid_idx] = like_v
        survival_cpu_full[valid_idx] = np.where(
            np.isfinite(survival_v), survival_v, LOGLIK_FLOOR)
        upper_prob_full[valid_idx] = upper_prob_v
        lower_prob_full[valid_idx] = lower_prob_v

    contributing = valid_for_loss
    idx = offset + np.arange(len(valid_for_loss))
    result.choice_prob_or_density[idx[contributing]] = likelihood[contributing]
    result.loglik[idx[contributing]] = np.log(likelihood[contributing])
    result.survival_at_tmax[idx[contributing]] = survival_cpu_full[contributing]
    result.upper_hit_prob_tmax[idx[contributing]] = upper_prob_full[contributing]
    result.lower_hit_prob_tmax[idx[contributing]] = lower_prob_full[contributing]
    result.metadata["_last_batch"] = metadata


def _as_mu_matrix(mu_values, n_trials, n_t):
    mu_values = np.asarray(mu_values, dtype=float)
    if mu_values.ndim == 1:
        return np.repeat(mu_values[:, None], n_t, axis=1)
    if mu_values.shape != (n_trials, n_t):
        raise ValueError(
            f"time-varying mu shape {mu_values.shape} != ({n_trials}, {n_t})")
    return mu_values


def _bucket_keys(mu_t, sigma):
    keys = np.empty(len(mu_t), dtype=[("mu", "f8"), ("sigma", "f8")])
    keys["mu"] = np.asarray(mu_t, dtype=float)
    keys["sigma"] = np.asarray(sigma, dtype=float)
    return keys


def _empty_result(n_trials):
    return BatchedLikelihoodResult(
        loglik=np.full(n_trials, np.nan, dtype=float),
        choice_prob_or_density=np.full(n_trials, np.nan, dtype=float),
        decision_time=np.full(n_trials, np.nan, dtype=float),
        survival_at_tmax=np.full(n_trials, np.nan, dtype=float),
        upper_hit_prob_tmax=np.full(n_trials, np.nan, dtype=float),
        lower_hit_prob_tmax=np.full(n_trials, np.nan, dtype=float),
    )


def _validate_global_params(bound, dt, dx, tmax, batch_size):
    vals = [bound, dt, dx, tmax, batch_size]
    if not all(np.isfinite(float(v)) for v in vals):
        raise ValueError("bound, dt, dx, tmax, and batch_size must be finite")
    if bound <= 0 or dt <= 0 or dx <= 0 or tmax <= 0 or batch_size <= 0:
        raise ValueError("bound, dt, dx, tmax, and batch_size must be positive")


def _next_power_of_two(n):
    return 1 << (int(n) - 1).bit_length()
