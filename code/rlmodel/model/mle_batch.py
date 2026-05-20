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

    All fields are ``(b,)`` per-valid-trial arrays. The solver gathers the
    first-passage density at each trial's decision-time bin **inside** the
    timestep loop and writes only that scalar — see O1 in
    ``mle_optimization_plan.md``. The previous ``(b, n_t)`` density tensors
    are gone, freeing ~14 GB of GPU memory on a typical population fit.

    For ``xp=numpy`` this is just a no-op pass-through; for ``xp=cupy`` it
    also collapses the per-batch GPU→CPU transfer to a single ``(b,)`` copy.
    """
    upper_at_decision_xp: object   # (b,) f_upper(decision_time) / dt
    lower_at_decision_xp: object   # (b,) f_lower(decision_time) / dt
    survival_xp: object            # (b,) survival mass at tmax
    upper_prob_xp: object          # (b,) total upper absorption prob
    lower_prob_xp: object          # (b,) total lower absorption prob
    valid_idx: np.ndarray          # (b,) numpy indices into the full batch
    metadata: dict


class BatchedDiffusionSolver:
    """Reusable batched first-passage solver.

    The solver uses FFT convolution over each bucket of trials sharing the same
    per-step drift and sigma. That avoids the previous dense transition tensor
    with shape ``(batch, n_x, n_x + 1)`` at every timestep.
    """

    def __init__(self, xp=np, normal_cdf=None, show_progress=False,
                 progress_desc=None):
        self.xp = xp
        self.normal_cdf = normal_cdf
        self.show_progress = bool(show_progress)
        self.progress_desc = progress_desc
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

    def solve(self, z, mu, sigma, valid_for_loss, bound, dt, dx, tmax,
              decision_idx=None):
        """Run the bucketed-FFT first-passage solver.

        Parameters
        ----------
        decision_idx : np.ndarray | None, shape (n_trials,), int
            Per-trial decision-time bin index in ``[0, n_t-1]`` for trials
            whose density we need to gather; sentinel ``-1`` (or any value
            outside ``[0, n_t-1]``) means "skip — don't write a density for
            this trial". When provided, the solver stores only ``(b,)``
            decision-time densities instead of the full ``(b, n_t)`` tensor
            (see O1 in ``mle_optimization_plan.md``). When ``None``, no
            density is ever written and ``upper_at_decision_xp`` /
            ``lower_at_decision_xp`` come back as zeros — useful for callers
            that only need ``survival`` / ``upper_prob`` / ``lower_prob``.
        """
        n_trials = len(valid_for_loss)
        n_t = int(round(float(tmax) / float(dt)))
        n_x = int(round(2.0 * float(bound) / float(dx)))
        if n_x < 2:
            raise ValueError(f"n_x={n_x} (need at least 2 bins); decrease dx")
        self.ensure_shape(float(bound), float(dx), n_x)

        # O4: keep mu in natural shape. For constant-mu (1-D), this avoids the
        # old (b, n_t) expansion and the cupy → numpy transfer of that tensor.
        mu_cpu, mu_is_constant = _prepare_mu(mu, n_trials, n_t)
        finite_mu = _mu_isfinite_per_trial(mu_cpu, mu_is_constant)
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
            "kernel_cache_count": 0,
            "kernel_cache_hits": 0,
            "workload_trials": int(n_trials),
            "n_x": int(n_x),
            "n_t": int(n_t),
            "backend": self.xp.__name__,
            "mu_is_constant": bool(mu_is_constant),
        }
        if b == 0:
            return _BatchedSolverResult(
                upper_at_decision_xp=self.xp.zeros(0, dtype=float),
                lower_at_decision_xp=self.xp.zeros(0, dtype=float),
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
        sigma_valid = np.asarray(sigma[valid_idx], dtype=float)
        # Constant-mu: (b,) view; never expanded to (b, n_t). Time-varying:
        # (b, n_t) — required for per-timestep bucket assignment.
        if mu_is_constant:
            mu_valid = mu_cpu[valid_idx]                # (b,)
        else:
            mu_valid = mu_cpu[valid_idx]                # (b, n_t)

        # O1: per-valid-trial decision-time index on xp. Sentinel value -1
        # (or anything outside [0, n_t-1]) means "skip the gather for this
        # trial"; trials where (decision_idx_xp[i] == t_idx) at step t_idx
        # are the only positions that get a density written. The previous
        # (b, n_t) upper/lower density tensors are gone — that was ~14 GB
        # of GPU memory per evaluation on a typical population fit.
        if decision_idx is None:
            decision_idx_for_valid = np.full(b, -1, dtype=np.int64)
        else:
            decision_idx_for_valid = np.asarray(
                decision_idx, dtype=np.int64)[valid_idx]
        decision_idx_xp = self.xp.asarray(decision_idx_for_valid)

        upper_at_decision = self.xp.zeros(b, dtype=float)
        lower_at_decision = self.xp.zeros(b, dtype=float)
        upper_prob_valid = self.xp.zeros(b, dtype=float)
        lower_prob_valid = self.xp.zeros(b, dtype=float)
        bucket_count = 0

        # O2: for constant-mu the bucket structure (which trials share (mu, σ))
        # is identical at every timestep — compute it ONCE before the loop and
        # reuse the cached `local_xp` index arrays. For time-varying mu we
        # fall back to per-timestep bucketing (trajectory-based caching is a
        # candidate for a later pass once O6's factorization is in place).
        cached_buckets = None
        constant_bucket_total = 0
        kernel_cache_count = 0
        kernel_cache_hits = 0
        if mu_is_constant:
            keys_const = _bucket_keys(mu_valid, sigma_valid)
            cached_buckets = []
            for key in np.unique(keys_const):
                local = np.flatnonzero(keys_const == key)
                mu_t = float(mu_valid[local[0]])
                sigma_t = float(sigma_valid[local[0]])
                kernel, mass_above, mass_below = self._transition_terms(
                    mu_t, sigma_t, bound, dt, dx)
                cached_buckets.append({
                    "local_xp": self.xp.asarray(local),
                    "mu": mu_t,
                    "sigma": sigma_t,
                    "mass_above": mass_above,
                    "mass_below": mass_below,
                    "kernel_fft": self._kernel_fft(kernel, n_x),
                })
            # One bucket-count tally for the whole solve (instead of × n_t).
            constant_bucket_total = len(cached_buckets) * n_t
            kernel_cache_count = len(cached_buckets)
            kernel_cache_hits = len(cached_buckets) * n_t

        step_iter = _progress_iter(
            range(n_t),
            enabled=self.show_progress,
            total=n_t,
            desc=self.progress_desc or "MLE diffusion",
        )
        for t_idx in step_iter:
            new_p = self.xp.empty_like(p)
            if cached_buckets is not None:
                # Constant-mu fast path: same buckets every timestep.
                step_buckets = cached_buckets
            else:
                # Time-varying mu: re-bucket per timestep on the current mu
                # column. `mu_valid` is on CPU only because bucket assignment
                # needs `np.unique`; per-bucket scalars below are still cheap.
                keys = _bucket_keys(mu_valid[:, t_idx], sigma_valid)
                step_buckets = []
                for key in np.unique(keys):
                    local = np.flatnonzero(keys == key)
                    step_buckets.append({
                        "local_xp": self.xp.asarray(local),
                        "mu": float(mu_valid[local[0], t_idx]),
                        "sigma": float(sigma_valid[local[0]]),
                    })

            for bucket in step_buckets:
                mu_t = bucket["mu"]
                sigma_t = bucket["sigma"]
                local_xp = bucket["local_xp"]
                if "kernel_fft" in bucket:
                    mass_above = bucket["mass_above"]
                    mass_below = bucket["mass_below"]
                    kernel_fft = bucket["kernel_fft"]
                else:
                    kernel, mass_above, mass_below = self._transition_terms(
                        mu_t, sigma_t, bound, dt, dx)
                    kernel_fft = None

                p_sub = p[local_xp]
                upper_abs = p_sub @ mass_above
                lower_abs = p_sub @ mass_below
                upper_prob_valid[local_xp] += upper_abs
                lower_prob_valid[local_xp] += lower_abs
                # O1 gather: write density / dt only at trials whose
                # decision-time bin is the current t_idx. Branch-free on xp.
                hits = decision_idx_xp[local_xp] == t_idx
                upper_at_decision[local_xp] = self.xp.where(
                    hits, upper_abs / dt, upper_at_decision[local_xp])
                lower_at_decision[local_xp] = self.xp.where(
                    hits, lower_abs / dt, lower_at_decision[local_xp])
                if kernel_fft is None:
                    new_p[local_xp] = self._convolve_rows(p_sub, kernel, n_x)
                else:
                    new_p[local_xp] = self._convolve_rows_cached(
                        p_sub, kernel_fft, n_x)
                if cached_buckets is None:
                    bucket_count += 1
            p = new_p

        if cached_buckets is not None:
            bucket_count = constant_bucket_total

        survival_valid = self.xp.sum(p, axis=1)
        # NOTE: deliberately no `asnumpy` here. The caller gathers on xp
        # (with valid_decision_xp / no_choice_xp masks) and emits a single
        # (b,) GPU->CPU copy per batch. For xp=numpy this is a no-op
        # pass-through.
        meta = dict(base_meta)
        meta["bucket_count"] = int(bucket_count)
        meta["kernel_cache_count"] = int(kernel_cache_count)
        meta["kernel_cache_hits"] = int(kernel_cache_hits)
        return _BatchedSolverResult(
            upper_at_decision_xp=upper_at_decision,
            lower_at_decision_xp=lower_at_decision,
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
        return self._convolve_rows_cached(
            p_sub, self._kernel_fft(kernel, n_x), n_x)

    def _kernel_fft(self, kernel, n_x):
        """Return the padded transition-kernel FFT for convolution."""
        k_pad = self.xp.zeros(self.fft_n, dtype=float)
        k_pad[:2 * n_x - 1] = kernel
        return self.xp.fft.rfft(k_pad)

    def _convolve_rows_cached(self, p_sub, kernel_fft, n_x):
        p_pad = self.xp.zeros((p_sub.shape[0], self.fft_n), dtype=float)
        p_pad[:, :n_x] = p_sub
        full = self.xp.fft.irfft(
            self.xp.fft.rfft(p_pad, axis=1)
            * kernel_fft[None, :],
            n=self.fft_n,
            axis=1,
        )
        return full[:, n_x - 1:2 * n_x - 1]


def batched_choice_rt_loglik(observed_choice_left, observed_rt, no_choice,
                             valid_for_loss, z, mu_values, sigma, bound,
                             non_decision_time, dt, dx, tmax, *, xp=np,
                             normal_cdf=None, solver=None):
    """Evaluate choice/RT likelihoods for many trials on one array backend."""
    _validate_global_params(bound, dt, dx, tmax)
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
        "workload_trials": int(n_trials),
        "bucket_count": 0,
    }
    _evaluate_batch(
        result=out,
        observed_choice_left=observed_choice_left,
        no_choice=no_choice,
        valid_for_loss=valid_for_loss,
        z=z,
        mu_values=mu_values,
        sigma=sigma,
        bound=float(bound),
        decision_time=decision_time,
        dt=float(dt),
        dx=float(dx),
        tmax=float(tmax),
        offset=0,
        solver=solver,
    )
    batch_meta = out.metadata.pop("_last_batch", {})
    metadata["bucket_count"] += int(batch_meta.get("bucket_count", 0))
    metadata["n_x"] = batch_meta.get("n_x")
    metadata["n_t"] = batch_meta.get("n_t")
    metadata["mu_is_constant"] = batch_meta.get("mu_is_constant")
    metadata["kernel_cache_count"] = batch_meta.get("kernel_cache_count")
    metadata["kernel_cache_hits"] = batch_meta.get("kernel_cache_hits")
    out.metadata.update(metadata)
    return out


def estimate_flat_trial_capacity_for_memory(memory_gb, bound, dx, tmax, dt,
                                            dtype_bytes=8):
    """Estimate flat candidate-trial capacity for one solver call."""
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
    flat_trial_capacity = max(int(available_bytes // max(per_trial_bytes, 1)), 1)
    return flat_trial_capacity, {
        "requested_memory_gb": float(memory_gb),
        "estimated_bytes_per_trial": int(per_trial_bytes),
        "estimated_total_bytes": int(flat_trial_capacity * per_trial_bytes),
        "n_x": int(n_x),
        "n_t": int(n_t),
        "fft_n": int(fft_n),
    }


def _evaluate_batch(result, observed_choice_left, no_choice, valid_for_loss, z,
                    mu_values, sigma, bound, decision_time, dt, dx, tmax,
                    offset, solver):
    n_t = int(round(float(tmax) / float(dt)))
    # O4: keep mu in its natural shape; constant-mu stays as (n_trials,).
    mu_cpu, mu_is_constant = _prepare_mu(mu_values, len(valid_for_loss), n_t)
    valid_solver = (
        valid_for_loss
        & np.isfinite(z)
        & np.isfinite(sigma)
        & _mu_isfinite_per_trial(mu_cpu, mu_is_constant)
        & (sigma > 0)
        & (z >= -bound)
        & (z <= bound)
    )

    # O1: build the per-trial decision-time bin index BEFORE the solver call
    # so the solver can do the gather inside its timestep loop and skip
    # allocating the (b, n_t) density tensors. Sentinel -1 means "no valid
    # decision for this trial; don't gather density" — those positions get
    # LOGLIK_FLOOR (or survival mass, for no-choice trials) via the masks
    # below.
    valid_decision_full = (
        valid_solver
        & ~no_choice
        & np.isfinite(observed_choice_left)
        & np.isfinite(decision_time)
        & (decision_time > 0)
        & (decision_time <= tmax)
    )
    raw_decision_idx = np.ceil(
        np.where(valid_decision_full, decision_time, dt) / dt
    ).astype(np.int64) - 1
    decision_idx_full = np.where(
        valid_decision_full, np.clip(raw_decision_idx, 0, n_t - 1), -1)

    solver_result = solver.solve(
        z, mu_cpu, sigma, valid_solver, bound, dt, dx, tmax,
        decision_idx=decision_idx_full)
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
        # Per-valid-trial masks for the likelihood reduction. No more
        # decision_idx-based row gather here — the solver already wrote the
        # decision-time density into (b,) arrays.
        choice_left_v = observed_choice_left[valid_idx]
        no_choice_v = no_choice[valid_idx]
        valid_decision_v = valid_decision_full[valid_idx]

        xp = solver.xp
        is_left_xp = xp.asarray((choice_left_v == 1))
        no_choice_xp = xp.asarray(no_choice_v)
        valid_decision_xp = xp.asarray(valid_decision_v)

        # Already-gathered densities on xp — (b,) each, populated by the
        # solver only at decision-time hits. No row indexing needed.
        upper_at_t_xp = solver_result.upper_at_decision_xp
        lower_at_t_xp = solver_result.lower_at_decision_xp
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


def _coerce_to_numpy(value):
    """Bring an array-like (numpy or cupy) to numpy without `np.asarray`-on-cupy.

    ``np.asarray(cupy_array)`` raises ``TypeError`` on CuPy ≥ 13 and silently
    forces a host transfer on older versions; both behaviors are surprising in
    a hot loop. ``.get()`` (CuPy) and pass-through (NumPy) are explicit.
    """
    # cupy arrays expose ``.get`` and ``__cuda_array_interface__``; either is a
    # sufficient duck-type signal. We prefer ``.get`` so we don't depend on a
    # CuPy-version-specific protocol.
    if hasattr(value, "get") and hasattr(value, "__cuda_array_interface__"):
        return value.get()
    return np.asarray(value)


def _prepare_mu(mu_values, n_trials, n_t):
    """Return (mu_cpu, is_constant) without ever expanding (b,) to (b, n_t).

    O4: the previous ``_as_mu_matrix`` always returned the 2-D shape — for
    constant-mu inputs that meant materializing an (n_trials × n_t) float
    array purely to broadcast the same value per timestep. Now we keep the
    natural shape and let the solver branch.
    """
    mu_cpu = _coerce_to_numpy(mu_values).astype(float, copy=False)
    if mu_cpu.ndim == 1:
        if mu_cpu.shape != (n_trials,):
            raise ValueError(
                f"constant mu shape {mu_cpu.shape} != ({n_trials},)")
        return mu_cpu, True
    if mu_cpu.ndim == 2:
        if mu_cpu.shape != (n_trials, n_t):
            raise ValueError(
                f"time-varying mu shape {mu_cpu.shape} != ({n_trials}, {n_t})")
        return mu_cpu, False
    raise ValueError(
        f"mu must be 1-D (constant) or 2-D (time-varying); got ndim={mu_cpu.ndim}")


def _mu_isfinite_per_trial(mu_cpu, is_constant):
    """Per-trial isfinite mask. (n_trials,) bool."""
    if is_constant:
        return np.isfinite(mu_cpu)
    return np.all(np.isfinite(mu_cpu), axis=1)


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


def _validate_global_params(bound, dt, dx, tmax):
    vals = [bound, dt, dx, tmax]
    if not all(np.isfinite(float(v)) for v in vals):
        raise ValueError("bound, dt, dx, and tmax must be finite")
    if bound <= 0 or dt <= 0 or dx <= 0 or tmax <= 0:
        raise ValueError("bound, dt, dx, and tmax must be positive")


def _next_power_of_two(n):
    return 1 << (int(n) - 1).bit_length()


def _progress_iter(iterable, *, enabled, total, desc):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit="step",
        leave=False,
        dynamic_ncols=True,
    )
