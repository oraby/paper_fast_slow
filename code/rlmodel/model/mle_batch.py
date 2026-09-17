"""Batched choice/RT likelihood for MLE fitting."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .array_backend import asnumpy
from .initvals import MLE_TERMINAL_C
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
    # Terminal-C redistribution of residual interior mass at t = tmax.
    # Partition of survival_xp into three buckets by bin-center position
    # relative to the threshold ``terminal_c * bound``:
    #   x >  C·B  → terminal_upper_mass_xp
    #   x < -C·B  → terminal_lower_mass_xp
    #   |x| ≤ C·B → terminal_no_decision_mass_xp
    # Invariant: upper + lower + no_decision == survival.
    terminal_upper_mass_xp: object
    terminal_lower_mass_xp: object
    terminal_no_decision_mass_xp: object
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
              decision_idx=None, terminal_c=MLE_TERMINAL_C.Default):
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
        terminal_c : float, default ``MLE_TERMINAL_C.Default``
            Threshold C ∈ [MLE_TERMINAL_C.Min, MLE_TERMINAL_C.Max] used to
            partition residual interior mass at t = tmax into
            terminal_upper / terminal_lower / terminal_no_decision
            buckets. C=Max routes the entire interior mass to the
            no-decision bucket (legacy survival behavior).
            See ``mle_terminal_c_plan.md``.
        """
        if not (MLE_TERMINAL_C.Min <= float(terminal_c) <= MLE_TERMINAL_C.Max):
            raise ValueError(
                f"terminal_c must satisfy "
                f"{MLE_TERMINAL_C.Min} <= C <= {MLE_TERMINAL_C.Max}; "
                f"got {terminal_c}.")
        n_trials = len(valid_for_loss)
        n_t = int(round(float(tmax) / float(dt)))
        n_x = int(round(2.0 * float(bound) / float(dx)))
        if n_x < 2:
            raise ValueError(f"n_x={n_x} (need at least 2 bins); decrease dx")
        self.ensure_shape(float(bound), float(dx), n_x)

        # O4: mu is one drift per trial, (n_trials,); it is never expanded
        # to (b, n_t).
        mu_cpu = _prepare_mu(mu, n_trials)
        finite_mu = np.isfinite(mu_cpu)
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
            "mu_is_constant": True,
        }
        if b == 0:
            return _BatchedSolverResult(
                upper_at_decision_xp=self.xp.zeros(0, dtype=float),
                lower_at_decision_xp=self.xp.zeros(0, dtype=float),
                survival_xp=self.xp.zeros(0, dtype=float),
                upper_prob_xp=self.xp.zeros(0, dtype=float),
                lower_prob_xp=self.xp.zeros(0, dtype=float),
                terminal_upper_mass_xp=self.xp.zeros(0, dtype=float),
                terminal_lower_mass_xp=self.xp.zeros(0, dtype=float),
                terminal_no_decision_mass_xp=self.xp.zeros(0, dtype=float),
                valid_idx=valid_idx,
                metadata=base_meta,
            )

        x_grid_cpu = np.linspace(-bound + dx / 2.0, bound - dx / 2.0, n_x)
        z_idx = np.argmin(np.abs(x_grid_cpu[None, :] - z[valid_idx, None]), axis=1)

        p = self.xp.zeros((b, n_x), dtype=float)
        p[self.xp.arange(b), self.xp.asarray(z_idx)] = 1.0
        sigma_valid = np.asarray(sigma[valid_idx], dtype=float)
        sigma_valid_xp = self.xp.asarray(sigma_valid)
        mu_valid_xp = self.xp.asarray(mu_cpu[valid_idx])   # (b,)

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

        # O2: the bucket structure (which trials share (mu, σ)) is identical
        # at every timestep — compute it ONCE before the loop and reuse the
        # cached `local_xp` index arrays. b > 0 here, so there is at least one
        # bucket.
        cached_buckets = self._constant_mu_buckets(
            mu_valid_xp, sigma_valid_xp)

        # O8: batch the per-bucket transition + kernel-FFT setup. The old
        # per-bucket Python loop did 6 CuPy dispatches × n_buckets just to
        # populate per-bucket mass/kernel-FFT — for NoiseGain-RewardRate
        # with ~8400 buckets that was ~50k setup-time dispatches per
        # evaluation. Now we compute the whole (n_buckets, ...) batch in
        # ~7 dispatches total, then gather to (b, ...) per-trial exactly
        # as O7 needs.
        # (n_buckets,) scalars → xp arrays. The Python list
        # comprehension is unavoidable here (host-side dict access),
        # but it runs once per fit on the small bucket table, not in
        # the hot loop.
        mu_per_bucket = self.xp.asarray(
            [bk["mu"] for bk in cached_buckets], dtype=float)
        sigma_per_bucket = self.xp.asarray(
            [bk["sigma"] for bk in cached_buckets], dtype=float)

        # Batched transition_terms: one set of CDF dispatches across
        # all buckets. Returns (n_buckets, 2n-1), (n_buckets, n),
        # (n_buckets, n).
        kernels_b, mass_above_b, mass_below_b = (
            self._transition_terms_batched(
                mu_per_bucket, sigma_per_bucket, bound, dt, dx))
        # Batched kernel FFT: one rfft on (n_buckets, fft_n).
        kernel_ffts_b = self._kernel_fft_batched(kernels_b, n_x)
        del kernels_b  # no longer needed; only the FFT survives

        # Scatter bucket ids to per-trial. Still N Python iterations,
        # but each is just an int-array slice assignment; trivial vs
        # the old per-bucket transition-term loop.
        bucket_id_per_trial = self.xp.zeros(b, dtype=self.xp.int64)
        for bid, bucket in enumerate(cached_buckets):
            bucket_id_per_trial[bucket["local_xp"]] = bid

        # O7 per-trial gather (same as before, just no intermediate
        # per-bucket dict storage).
        per_trial_mass_above = mass_above_b[bucket_id_per_trial]
        per_trial_mass_below = mass_below_b[bucket_id_per_trial]
        per_trial_kernel_fft = kernel_ffts_b[bucket_id_per_trial]
        del mass_above_b, mass_below_b, kernel_ffts_b

        # Reuse one FFT padding buffer across timesteps; only the
        # leading n_x columns are touched per step, so the tail stays
        # at the initial zeros.
        p_pad_buffer = self.xp.zeros((b, self.fft_n), dtype=float)

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
            # O7 vectorized path. Six batched cupy/numpy ops per step,
            # independent of bucket count.
            upper_abs_t = (p * per_trial_mass_above).sum(axis=1)
            lower_abs_t = (p * per_trial_mass_below).sum(axis=1)
            upper_prob_valid = upper_prob_valid + upper_abs_t
            lower_prob_valid = lower_prob_valid + lower_abs_t
            hits = decision_idx_xp == t_idx
            upper_at_decision = self.xp.where(
                hits, upper_abs_t / dt, upper_at_decision)
            lower_at_decision = self.xp.where(
                hits, lower_abs_t / dt, lower_at_decision)
            # Batched FFT convolution: per-trial kernel applied in one go.
            # ``p_pad_buffer`` is reused across timesteps; only the leading
            # n_x columns change per step.
            p_pad_buffer[:, :n_x] = p
            full = self.xp.fft.irfft(
                self.xp.fft.rfft(p_pad_buffer, axis=1)
                * per_trial_kernel_fft,
                n=self.fft_n,
                axis=1,
            )
            p = full[:, n_x - 1: 2 * n_x - 1]

        survival_valid = self.xp.sum(p, axis=1)
        # Terminal-C redistribution: partition residual interior mass into
        # three buckets based on bin-center position relative to ±C·B. The
        # masks are (n_x,) on xp; broadcasting `(b, n_x) * (n_x,)` keeps the
        # whole reduction on-device. self.x_grid is set up by ensure_shape().
        thresh = float(terminal_c) * float(bound)
        assert self.x_grid is not None  # narrow for pyright
        upper_mask = self.x_grid > thresh
        lower_mask = self.x_grid < -thresh
        no_decision_mask = ~upper_mask & ~lower_mask
        terminal_upper_mass = (p * upper_mask).sum(axis=1)
        terminal_lower_mass = (p * lower_mask).sum(axis=1)
        terminal_no_decision_mass = (p * no_decision_mask).sum(axis=1)
        # NOTE: deliberately no `asnumpy` here. The caller gathers on xp
        # (with valid_decision_xp / no_choice_xp masks) and emits a single
        # (b,) GPU->CPU copy per batch. For xp=numpy this is a no-op
        # pass-through.
        meta = dict(base_meta)
        meta["bucket_count"] = int(constant_bucket_total)
        meta["kernel_cache_count"] = int(kernel_cache_count)
        meta["kernel_cache_hits"] = int(kernel_cache_hits)
        meta["terminal_c"] = float(terminal_c)
        return _BatchedSolverResult(
            upper_at_decision_xp=upper_at_decision,
            lower_at_decision_xp=lower_at_decision,
            survival_xp=survival_valid,
            upper_prob_xp=upper_prob_valid,
            lower_prob_xp=lower_prob_valid,
            terminal_upper_mass_xp=terminal_upper_mass,
            terminal_lower_mass_xp=terminal_lower_mass,
            terminal_no_decision_mass_xp=terminal_no_decision_mass,
            valid_idx=valid_idx,
            metadata=meta,
        )

    def _transition_terms_batched(self, mu_t_xp, sigma_t_xp, bound, dt, dx):
        """Compute the per-step Gaussian transition kernel and absorbed-mass
        terms for a batch of `(mu_t, sigma)` pairs.

        ``mu_t_xp`` and ``sigma_t_xp`` are ``(B,)`` arrays on ``self.xp``;
        returns ``(B, 2n_x-1)`` transition kernels and ``(B, n_x)`` absorbed-
        mass tensors in a single CuPy dispatch per CDF call, for the
        per-bucket setup (`B = n_buckets`).
        """
        # ensure_shape() ran at the top of solve(), and the solver constructor
        # always receives a normal_cdf when used from production paths. The
        # asserts here are for the type checker only — they cannot fire in
        # practice.
        assert self.offsets is not None and self.x_grid is not None, (
            "_transition_terms_batched called before ensure_shape()")
        assert self.normal_cdf is not None, (
            "BatchedDiffusionSolver requires normal_cdf")
        sigma_sqrt_dt = sigma_t_xp * float(np.sqrt(dt))            # (B,)
        mean_shift = mu_t_xp * dt                                  # (B,)
        # Broadcast offsets (2n-1,) and x_grid (n,) against the (B,) scalars.
        z_upper = (
            (self.offsets[None, :] + 0.5) * dx - mean_shift[:, None]
        ) / sigma_sqrt_dt[:, None]                                 # (B, 2n-1)
        z_lower = (
            (self.offsets[None, :] - 0.5) * dx - mean_shift[:, None]
        ) / sigma_sqrt_dt[:, None]                                 # (B, 2n-1)
        transition_kernel = (
            self.normal_cdf(z_upper) - self.normal_cdf(z_lower)
        )                                                          # (B, 2n-1)
        mass_above = 1.0 - self.normal_cdf(
            (bound - self.x_grid[None, :] - mean_shift[:, None])
            / sigma_sqrt_dt[:, None]
        )                                                          # (B, n)
        mass_below = self.normal_cdf(
            (-bound - self.x_grid[None, :] - mean_shift[:, None])
            / sigma_sqrt_dt[:, None]
        )                                                          # (B, n)
        return transition_kernel, mass_above, mass_below

    def _constant_mu_buckets(self, mu_valid_xp, sigma_valid_xp):
        """Group constant-mu trials by (mu, sigma) on the active backend.

        This replaces the older NumPy ``unique`` + ``flatnonzero`` path, which
        walked the full valid population on CPU. Only the small unique
        ``(mu, sigma)`` scalar table is copied back to host for transition
        setup; each bucket's trial indices stay on ``self.xp``.
        """
        n = int(mu_valid_xp.shape[0])
        if n == 0:
            return []
        # CuPy expects lexsort keys as a stacked ndarray, not a Python tuple.
        # NumPy accepts both forms, so use the stricter representation here.
        order = self.xp.lexsort(self.xp.stack((sigma_valid_xp, mu_valid_xp)))
        sorted_mu = mu_valid_xp[order]
        sorted_sigma = sigma_valid_xp[order]

        starts_mask = self.xp.empty(n, dtype=bool)
        starts_mask[0] = True
        starts_mask[1:] = (
            (sorted_mu[1:] != sorted_mu[:-1])
            | (sorted_sigma[1:] != sorted_sigma[:-1])
        )
        starts_xp = self.xp.nonzero(starts_mask)[0]
        stops_xp = self.xp.concatenate(
            (starts_xp[1:], self.xp.asarray([n], dtype=starts_xp.dtype)))

        starts = asnumpy(self.xp, starts_xp).astype(int, copy=False)
        stops = asnumpy(self.xp, stops_xp).astype(int, copy=False)
        unique_mu = asnumpy(self.xp, sorted_mu[starts_xp])
        unique_sigma = asnumpy(self.xp, sorted_sigma[starts_xp])
        return [
            {
                "local_xp": order[int(start):int(stop)],
                "mu": float(mu_t),
                "sigma": float(sigma_t),
            }
            for start, stop, mu_t, sigma_t
            in zip(starts, stops, unique_mu, unique_sigma)
        ]

    def _kernel_fft_batched(self, kernels, n_x):
        """Pad and FFT a batch of transition kernels.

        ``kernels`` is ``(n_buckets, 2n_x-1)``; returns ``(n_buckets, fft_n_complex)``
        in a single CuPy ``rfft`` dispatch on the padded ``(n_buckets, fft_n)``
        tensor. Used by the constant-mu setup; the time-varying hot loop
        builds its per-trial FFTs inline already.
        """
        assert self.fft_n is not None, (
            "_kernel_fft_batched called before ensure_shape()")
        n_buckets = int(kernels.shape[0])
        k_pad = self.xp.zeros((n_buckets, self.fft_n), dtype=float)
        k_pad[:, :2 * n_x - 1] = kernels
        return self.xp.fft.rfft(k_pad, axis=1)


def batched_choice_rt_loglik(observed_choice_left, observed_rt, no_choice,
                             valid_for_loss, z, mu_values, sigma, bound,
                             non_decision_time, dt, dx, tmax, *, xp=np,
                             normal_cdf=None, solver=None,
                             terminal_c=MLE_TERMINAL_C.Default,
                             lapse_rate=0.0):
    """Evaluate choice/RT likelihoods for many trials on one array backend.

    Parameters
    ----------
    terminal_c : float, default ``MLE_TERMINAL_C.Default``
        Threshold C ∈ [MLE_TERMINAL_C.Min, MLE_TERMINAL_C.Max] forwarded
        to the solver. No-choice trials use ``terminal_no_decision_mass``
        (with this threshold) as their likelihood instead of the full
        survival mass. See ``mle_terminal_c_plan.md``.
    lapse_rate : float | array-like of shape ``(n_trials,)``, default 0.0
        Contamination / lapse mixture λ ∈ [0, 1). Per-trial likelihood
        becomes ``(1-λ)·L_DDM + λ/(2·tmax)``. Accepts a scalar (broadcast
        to all trials) or a per-trial array — the population objective
        passes a per-candidate broadcast so each candidate's λ value is
        used for its block of trials. See ``mle_lapse_rate_plan.md``.
    """
    _validate_global_params(bound, dt, dx, tmax)
    n_trials = len(valid_for_loss)
    out = _empty_result(n_trials)
    if n_trials == 0:
        return out

    observed_choice_left = _coerce_to_numpy(observed_choice_left).astype(
        float, copy=False)
    observed_rt = _coerce_to_numpy(observed_rt).astype(float, copy=False)
    no_choice = _coerce_to_numpy(no_choice).astype(bool, copy=False)
    valid_for_loss = _coerce_to_numpy(valid_for_loss).astype(bool, copy=False)
    z = _coerce_to_numpy(z).astype(float, copy=False)
    sigma = _coerce_to_numpy(sigma).astype(float, copy=False)
    non_decision_time = _coerce_to_numpy(non_decision_time).astype(
        float, copy=False)
    # ``lapse_rate`` may arrive as a CuPy array (population path uses
    # ``backend.xp.repeat`` to build a per-candidate broadcast). NumPy
    # refuses to implicitly convert CuPy arrays in recent versions
    # — route through ``_coerce_to_numpy`` first, like the other inputs.
    lapse_rate = _coerce_to_numpy(lapse_rate).astype(float, copy=False)

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
        terminal_c=float(terminal_c),
        lapse_rate=lapse_rate,
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
                    offset, solver, terminal_c=MLE_TERMINAL_C.Default,
                    lapse_rate=0.0):
    n_t = int(round(float(tmax) / float(dt)))
    mu_for_solve = _prepare_mu(mu_values, len(valid_for_loss))
    finite_mu_mask = np.isfinite(mu_for_solve)
    valid_solver = (
        valid_for_loss
        & np.isfinite(z)
        & np.isfinite(sigma)
        & finite_mu_mask
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
    # Choice trials where the observed RT is in the lapse window but the
    # DDM can't reach the bound in time (RT ≤ T0 ⇒ decision_time ≤ 0).
    # The DDM density is identically 0 here, but the lapse mixture term
    # λ/(2·T_max) is well-defined, so we want them lapse-eligible instead
    # of LOGLIK_FLOOR-pinned. Solver still receives decision_idx=-1 for
    # these trials, so density_at_t_xp ends up 0 by construction. NaN RT
    # propagates through ``decision_time`` so the ``isfinite`` clause
    # excludes data-quality rejects; RT > tmax is already excluded by
    # ``valid_solver`` upstream.
    short_rt_choice_full = (
        valid_solver
        & ~no_choice
        & np.isfinite(observed_choice_left)
        & np.isfinite(decision_time)
        & (decision_time <= 0)
    )
    raw_decision_idx = np.ceil(
        np.where(valid_decision_full, decision_time, dt) / dt
    ).astype(np.int64) - 1
    decision_idx_full = np.where(
        valid_decision_full, np.clip(raw_decision_idx, 0, n_t - 1), -1)

    solver_result = solver.solve(
        z, mu_for_solve, sigma, valid_solver, bound, dt, dx, tmax,
        decision_idx=decision_idx_full, terminal_c=float(terminal_c))
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
        short_rt_choice_v = short_rt_choice_full[valid_idx]

        xp = solver.xp
        is_left_xp = xp.asarray((choice_left_v == 1))
        no_choice_xp = xp.asarray(no_choice_v)
        valid_decision_xp = xp.asarray(valid_decision_v)
        short_rt_choice_xp = xp.asarray(short_rt_choice_v)

        # Already-gathered densities on xp — (b,) each, populated by the
        # solver only at decision-time hits. No row indexing needed.
        upper_at_t_xp = solver_result.upper_at_decision_xp
        lower_at_t_xp = solver_result.lower_at_decision_xp
        density_at_t_xp = xp.where(is_left_xp, upper_at_t_xp, lower_at_t_xp)

        # No-choice trials contribute the terminal no-decision mass at tmax
        # (a function of `terminal_c`). With terminal_c → 1 this approaches
        # the full survival mass — the legacy behavior. With terminal_c = 0
        # only mass at exactly x = 0 stays; on a bin-centered grid that is
        # zero, so no-choice likelihood collapses to LOGLIK_FLOOR.
        # Trials that are valid_solver but have neither a valid decision nor
        # are flagged no_choice fall through to LOGLIK_FLOOR.
        like_xp = xp.where(no_choice_xp,
                           solver_result.terminal_no_decision_mass_xp,
                           density_at_t_xp)

        # Lapse / contamination mixture: same formula for choice and no-
        # choice trials per the user's design decision (see
        # ``mle_lapse_rate_plan.md``). λ may be a scalar or a per-trial
        # array; the population objective passes a per-candidate
        # broadcast. λ=0 reproduces the pre-mixture likelihood exactly.
        lapse_arr = np.broadcast_to(
            np.asarray(lapse_rate, dtype=float),
            (len(valid_for_loss),))
        lapse_rate_xp = xp.asarray(lapse_arr[valid_idx])
        lapse_density_xp = lapse_rate_xp / (2.0 * float(tmax))
        like_xp = (1.0 - lapse_rate_xp) * like_xp + lapse_density_xp

        # Lapse-eligible: no_choice, a valid (T0,T_max] decision, OR a
        # choice trial with RT in (0,T0] (short_rt_choice). The third case
        # lets the mixture term λ/(2·T_max) provide the floor instead of
        # LOGLIK_FLOOR — the DDM is already 0 there by construction.
        like_xp = xp.where(
            no_choice_xp | valid_decision_xp | short_rt_choice_xp,
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


def _prepare_mu(mu_values, n_trials):
    """Return mu as a host ``(n_trials,)`` float array — one drift per trial.

    O4: never expanded to (n_trials, n_t).
    """
    mu_cpu = _coerce_to_numpy(mu_values).astype(float, copy=False)
    if mu_cpu.shape != (n_trials,):
        raise ValueError(f"mu shape {mu_cpu.shape} != ({n_trials},)")
    return mu_cpu


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
    # try:
    #     from tqdm.auto import tqdm
    # except ImportError:
    #     return iterable
    from tqdm.auto import tqdm
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit="step",
        leave=False,
        dynamic_ncols=True,
    )
