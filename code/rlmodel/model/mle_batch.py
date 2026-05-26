"""Batched choice/RT likelihood for MLE fitting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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


@dataclass
class TimeVaryingMuFactors:
    """Factorized time-varying mu — avoids the ``(n_trials, n_t)`` materialization.

    For Decay-Q drift / Decaying-Q-Val noise variants, ``mu[i, t]`` is a sum
    of:

    - a constant-in-t per-trial baseline  ``base_mu[i]``
    - (optional) a drift contribution    ``q_drift_coef[i] * decay_form[c(i), t]``
    - (optional) a noise contribution    ``sigma[i] * sign(q_rel[i]) *
                                          max(|q_rel[i]| * q_coef[c(i)] -
                                              log_decay[c(i), t], 0) / dt``

    where ``c(i)`` is the candidate that owns trial ``i``. Each per-time term
    factors into a per-trial scalar times a per-candidate ``(n_t,)`` shape,
    so the (n_trials, n_t) tensor never needs to exist — instead the solver
    computes ``mu_t`` on the fly each timestep from ``(n_trials,)`` and
    ``(n_candidates, n_t)`` arrays.

    Memory for a typical Decay-Q fit (S = 60, N ≈ 5000, n_t = 3000):

    - dense ``(b, n_t)`` mu tensor: ~7 GB on GPU
    - this factored form: ~``2 * b * 8 + 2 * S * n_t * 8`` ≈ 25 MB

    See O6 in ``mle_optimization_plan.md``.
    """
    n_trials: int
    n_t: int
    dt: float
    # Per-trial constants (host arrays; solver indexes by valid_idx, then pushes
    # the subset to xp).
    base_mu: np.ndarray                              # (n_trials,) — DRIFT_COEF * DV
    candidate_id_per_trial: np.ndarray               # (n_trials,) int64
    # Decay-Q drift terms (optional pair)
    q_drift_coef_per_trial: Optional[np.ndarray] = None  # (n_trials,) — q_for_drift * Q_VAL_COEF
    decay_form_per_cand: Optional[np.ndarray] = None     # (n_candidates, n_t)
    # Decaying-Q-Val noise terms (optional triplet)
    q_abs_x_qcoef_per_trial: Optional[np.ndarray] = None  # (n_trials,) — |q_rel| * Q_VAL_COEF
    q_sign_per_trial: Optional[np.ndarray] = None         # (n_trials,) — sign(q_rel)
    sigma_per_trial: Optional[np.ndarray] = None          # (n_trials,) — sigma for noise scaling
    log_decay_per_cand: Optional[np.ndarray] = None       # (n_candidates, n_t)

    def isfinite_per_trial(self) -> np.ndarray:
        """(n_trials,) bool mask — finite across all populated factor arrays."""
        mask = np.isfinite(self.base_mu)
        for arr in (self.q_drift_coef_per_trial,
                    self.q_abs_x_qcoef_per_trial,
                    self.q_sign_per_trial,
                    self.sigma_per_trial):
            if arr is not None:
                mask &= np.isfinite(arr)
        return mask

    def select_valid(self, valid_idx: np.ndarray) -> "TimeVaryingMuFactors":
        """Return a new factors object subsetted to the given trial indices.

        The per-candidate ``(n_candidates, n_t)`` shapes are kept intact — they
        index by ``candidate_id_per_trial`` which is also subsetted here.
        """
        def _maybe(arr):
            return None if arr is None else arr[valid_idx]
        return TimeVaryingMuFactors(
            n_trials=int(valid_idx.size),
            n_t=self.n_t,
            dt=self.dt,
            base_mu=self.base_mu[valid_idx],
            candidate_id_per_trial=self.candidate_id_per_trial[valid_idx],
            q_drift_coef_per_trial=_maybe(self.q_drift_coef_per_trial),
            decay_form_per_cand=self.decay_form_per_cand,
            q_abs_x_qcoef_per_trial=_maybe(self.q_abs_x_qcoef_per_trial),
            q_sign_per_trial=_maybe(self.q_sign_per_trial),
            sigma_per_trial=_maybe(self.sigma_per_trial),
            log_decay_per_cand=self.log_decay_per_cand,
        )


class _SolverMuFactorsXP:
    """Backend-resident mirror of ``TimeVaryingMuFactors`` for inside ``solve``.

    Pushes the per-trial factors and the per-candidate decay shapes to ``xp``
    once at solver setup. Per timestep, ``mu_t(t_idx)`` computes a ``(b,)``
    drift vector entirely on-device via column-gather from the per-candidate
    shapes plus per-trial scalars. No (b, n_t) tensor materialized.
    """

    def __init__(self, factors: TimeVaryingMuFactors, xp):
        self.xp = xp
        self.dt = float(factors.dt)
        self.base_mu = xp.asarray(factors.base_mu)
        self.candidate_id = xp.asarray(factors.candidate_id_per_trial)
        self.q_drift_coef = (None if factors.q_drift_coef_per_trial is None
                             else xp.asarray(factors.q_drift_coef_per_trial))
        self.decay_form = (None if factors.decay_form_per_cand is None
                           else xp.asarray(factors.decay_form_per_cand))
        self.q_abs_x_qcoef = (None if factors.q_abs_x_qcoef_per_trial is None
                              else xp.asarray(factors.q_abs_x_qcoef_per_trial))
        self.q_sign = (None if factors.q_sign_per_trial is None
                       else xp.asarray(factors.q_sign_per_trial))
        self.sigma = (None if factors.sigma_per_trial is None
                      else xp.asarray(factors.sigma_per_trial))
        self.log_decay = (None if factors.log_decay_per_cand is None
                          else xp.asarray(factors.log_decay_per_cand))

    def mu_t(self, t_idx: int):
        """Return ``(b,)`` mu at timestep ``t_idx`` on ``xp``."""
        xp = self.xp
        result = self.base_mu
        if self.q_drift_coef is not None and self.decay_form is not None:
            # decay_form[:, t_idx] is (n_candidates,); gather to per-trial.
            decay_t = self.decay_form[:, t_idx][self.candidate_id]
            result = result + self.q_drift_coef * decay_t
        if (self.q_abs_x_qcoef is not None and self.log_decay is not None
                and self.q_sign is not None and self.sigma is not None):
            log_decay_t = self.log_decay[:, t_idx][self.candidate_id]
            decayed = xp.maximum(self.q_abs_x_qcoef - log_decay_t, 0.0)
            signed = xp.where(self.q_sign < 0, -decayed, decayed)
            result = result + self.sigma * signed / self.dt
        return result


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

        # O6: detect the factorized time-varying mu form. When present, we
        # never materialize the (n_trials, n_t) mu tensor; `mu_t` is computed
        # on the fly per timestep from per-trial scalars and per-candidate
        # (n_t,) decay shapes. The (b, n_t) GPU residency was the dominant
        # memory cost for Decay-Q population fits (~7 GB for S=60, N=5000).
        mu_is_factored = isinstance(mu, TimeVaryingMuFactors)
        if mu_is_factored:
            if mu.n_trials != n_trials or mu.n_t != n_t:
                raise ValueError(
                    f"TimeVaryingMuFactors shape ({mu.n_trials}, {mu.n_t}) "
                    f"!= solver shape ({n_trials}, {n_t})")
            mu_is_constant = False
            mu_cpu = None  # not used on factored path
            finite_mu = mu.isfinite_per_trial()
        else:
            # O4: keep mu in natural shape. For constant-mu (1-D), this avoids
            # the old (b, n_t) expansion and the cupy → numpy transfer of that
            # tensor.
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
            "mu_is_factored": bool(mu_is_factored),
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
        sigma_valid_xp = self.xp.asarray(sigma_valid)
        # Constant-mu: (b,) view; never expanded to (b, n_t). Time-varying:
        # (b, n_t) — the whole matrix lives on xp so the per-timestep slice
        # `mu_valid_xp[:, t]` is a zero-copy view feeding the batched
        # transition_terms (no per-step CPU→xp transfer). Factored mu
        # (O6) skips this entirely — `mu_t` is materialized on demand.
        if mu_is_factored:
            mu_valid_xp = None
        else:
            assert mu_cpu is not None  # narrow type for pyright
            mu_valid = mu_cpu[valid_idx]                # (b,) or (b, n_t)
            mu_valid_xp = self.xp.asarray(mu_valid)

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
        per_trial_mass_above = None
        per_trial_mass_below = None
        per_trial_kernel_fft = None
        p_pad_buffer = None
        factors_xp: Optional[_SolverMuFactorsXP] = None
        if mu_is_constant:
            cached_buckets = self._constant_mu_buckets(
                mu_valid_xp, sigma_valid_xp)

            # O8: batch the per-bucket transition + kernel-FFT setup. The old
            # per-bucket Python loop did 6 CuPy dispatches × n_buckets just to
            # populate per-bucket mass/kernel-FFT — for NoiseGain-RewardRate
            # with ~8400 buckets that was ~50k setup-time dispatches per
            # evaluation. Now we compute the whole (n_buckets, ...) batch in
            # ~7 dispatches total, then gather to (b, ...) per-trial exactly
            # as O7 needs.
            if len(cached_buckets) > 0:
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
        else:
            # Time-varying mu setup (O7 extension): mu_valid_xp is already on
            # xp from the unconditional push above; we just need a reusable
            # FFT padding buffer. No bucketing — every trial gets its own
            # kernel/mass row computed per timestep via the batched transition
            # terms below.
            p_pad_buffer = self.xp.zeros((b, self.fft_n), dtype=float)
            if mu_is_factored:
                # O6 factored time-varying: push per-trial scalars and the
                # per-candidate (n_t,) decay shapes to xp ONCE. The (b, n_t)
                # mu tensor never gets built; the timestep loop calls
                # `factors_xp.mu_t(t_idx)` to get a (b,) drift vector each
                # step from per-trial scalars + column-gather from the
                # per-candidate shapes.
                assert isinstance(mu, TimeVaryingMuFactors)  # narrow for pyright
                factors_xp = _SolverMuFactorsXP(
                    mu.select_valid(valid_idx), self.xp)

        step_iter = _progress_iter(
            range(n_t),
            enabled=self.show_progress,
            total=n_t,
            desc=self.progress_desc or "MLE diffusion",
        )
        for t_idx in step_iter:
            if (mu_is_constant
                    and per_trial_mass_above is not None
                    and p_pad_buffer is not None):
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
            else:
                # Time-varying mu (Decay-Q drift / Decaying-Q-Val noise):
                # vectorize across all valid trials per timestep using the
                # batched transition_terms. No bucketing — for population
                # fits the trajectories are usually unique per (cand, trial)
                # so bucketing wasn't compressing anything.
                assert p_pad_buffer is not None, "time-varying setup missing"
                # O6 vs O7-tv: route the per-timestep mu_t source.
                if factors_xp is not None:
                    # Factored: materialize (b,) on the fly from per-trial
                    # scalars + per-candidate column gather. No (b, n_t)
                    # tensor anywhere in this branch.
                    mu_t_xp = factors_xp.mu_t(t_idx)
                else:
                    assert mu_valid_xp is not None, (
                        "time-varying branch reached without mu_valid_xp bound")
                    mu_t_xp = mu_valid_xp[:, t_idx]           # (b,) xp view
                kernel_b, mass_above_b, mass_below_b = (
                    self._transition_terms_batched(
                        mu_t_xp, sigma_valid_xp, bound, dt, dx))
                # kernel_b: (b, 2n-1)
                # mass_above_b, mass_below_b: (b, n)
                upper_abs_t = (p * mass_above_b).sum(axis=1)   # (b,)
                lower_abs_t = (p * mass_below_b).sum(axis=1)   # (b,)
                upper_prob_valid = upper_prob_valid + upper_abs_t
                lower_prob_valid = lower_prob_valid + lower_abs_t
                hits = decision_idx_xp == t_idx
                upper_at_decision = self.xp.where(
                    hits, upper_abs_t / dt, upper_at_decision)
                lower_at_decision = self.xp.where(
                    hits, lower_abs_t / dt, lower_at_decision)
                # Build per-trial kernel-FFT freshly each timestep (kernel
                # changes with mu_t). One batched rfft on (b, fft_n).
                k_pad = self.xp.zeros((b, self.fft_n), dtype=float)
                k_pad[:, :2 * n_x - 1] = kernel_b
                kernel_fft_b = self.xp.fft.rfft(k_pad, axis=1)
                p_pad_buffer[:, :n_x] = p
                full = self.xp.fft.irfft(
                    self.xp.fft.rfft(p_pad_buffer, axis=1) * kernel_fft_b,
                    n=self.fft_n,
                    axis=1,
                )
                p = full[:, n_x - 1: 2 * n_x - 1]
                bucket_count += 1

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

    def _transition_terms_batched(self, mu_t_xp, sigma_t_xp, bound, dt, dx):
        """Compute the per-step Gaussian transition kernel and absorbed-mass
        terms for a batch of `(mu_t, sigma)` pairs.

        ``mu_t_xp`` and ``sigma_t_xp`` are ``(B,)`` arrays on ``self.xp``;
        returns ``(B, 2n_x-1)`` transition kernels and ``(B, n_x)`` absorbed-
        mass tensors in a single CuPy dispatch per CDF call. Used by both
        the constant-mu setup (`B = n_buckets`) and the time-varying hot
        loop (`B = b`, recomputed every timestep).
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
                             normal_cdf=None, solver=None):
    """Evaluate choice/RT likelihoods for many trials on one array backend."""
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
    # O6: factored time-varying mu skips the host-side coerce/expand path —
    # the solver consumes the factors object directly and computes mu_t on
    # the fly. For arrays we still go through _prepare_mu (O4) which keeps
    # constant-mu as (n_trials,) and validates time-varying shape.
    if isinstance(mu_values, TimeVaryingMuFactors):
        finite_mu_mask = mu_values.isfinite_per_trial()
        mu_for_solve = mu_values
    else:
        mu_cpu, mu_is_constant = _prepare_mu(mu_values, len(valid_for_loss), n_t)
        finite_mu_mask = _mu_isfinite_per_trial(mu_cpu, mu_is_constant)
        mu_for_solve = mu_cpu
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
    raw_decision_idx = np.ceil(
        np.where(valid_decision_full, decision_time, dt) / dt
    ).astype(np.int64) - 1
    decision_idx_full = np.where(
        valid_decision_full, np.clip(raw_decision_idx, 0, n_t - 1), -1)

    solver_result = solver.solve(
        z, mu_for_solve, sigma, valid_solver, bound, dt, dx, tmax,
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
