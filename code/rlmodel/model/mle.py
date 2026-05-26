from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from . import state_updates
from .array_backend import asnumpy, resolve_array_backend
from .mle_batch import (
    BatchedLikelihoodResult,
    BatchedDiffusionSolver,
    TimeVaryingMuFactors,
    batched_choice_rt_loglik,
    estimate_flat_trial_capacity_for_memory,
)
from .mle_likelihood import trial_choice_rt_loglik


SUPPORTED_MLE_BIASES = {"None_", "Q-Val", "Q-Val (Offset)"}
SUPPORTED_MLE_ARRAY_BACKENDS = {"auto", "numpy", "cupy"}
SUPPORTED_MLE_CUPY_FALLBACKS = {"numpy", "error"}
_PREPARED_SESSION_BACKEND_CACHE = {}


@dataclass(frozen=True)
class MLEModelConfig:
    drift_fn_str: str
    bias_fn_str: str
    noise_fn_str: str
    include_Q: bool
    include_RewardRate: bool
    dt: float
    t_dur: float
    dx: float = 0.02
    diffusion_backend: str = "auto"
    mle_array_backend: str = "numpy"
    mle_device_id: int | None = None
    mle_cupy_fallback: str = "error"
    mle_gpu_memory_gb: float | None = None
    mle_use_batched_likelihood: bool = True
    mle_show_progress: bool = False
    mle_terminal_c: float = 0.0

    @property
    def uses_q_bias(self):
        return self.bias_fn_str in {"Q-Val", "Q-Val (Offset)"}

    @property
    def uses_decay_q_drift(self):
        return "Decay Q" in self.drift_fn_str

    @property
    def uses_decay_q_noise(self):
        return self.noise_fn_str == "Decaying Q-Val"

    @property
    def requires_gpu(self):
        """True iff the user requested GPU and ruled out silent fallback.

        Set by the ``--mle-backend GPU`` CLI translation in ``model_runner.py``
        (which maps to ``mle_array_backend="cupy"`` + ``mle_cupy_fallback="error"``).
        Used by the pre-flight check in ``fit._processSubject`` to probe the
        resolved backend before differential_evolution actually runs.
        """
        return (self.mle_array_backend == "cupy"
                and self.mle_cupy_fallback == "error")


@dataclass
class MLEEvalResult:
    neg_loglik: float
    loglik: float
    n_trials_loss: int
    n_trials_total: int
    mle_df: pd.DataFrame | None = None
    backend_info: dict | None = None


@dataclass(frozen=True)
class PreparedMLEData:
    """Static trial data prepared once per subject/fit."""
    df: pd.DataFrame
    sorted_index: np.ndarray
    session_slices: tuple
    dv: np.ndarray
    valid: np.ndarray
    choice_left: np.ndarray
    reward: np.ndarray
    observed_rt: np.ndarray
    trial_number: np.ndarray

    @property
    def n_trials(self):
        return int(len(self.dv))


def validate_mle_config(model_config):
    if model_config.bias_fn_str not in SUPPORTED_MLE_BIASES:
        raise NotImplementedError(
            "MLE currently supports only bias functions: "
            f"{sorted(SUPPORTED_MLE_BIASES)}. Got {model_config.bias_fn_str!r}.")
    if model_config.mle_array_backend not in SUPPORTED_MLE_ARRAY_BACKENDS:
        raise ValueError(
            f"Unknown MLE array backend {model_config.mle_array_backend!r}. "
            f"Expected {sorted(SUPPORTED_MLE_ARRAY_BACKENDS)}.")
    if model_config.mle_cupy_fallback not in SUPPORTED_MLE_CUPY_FALLBACKS:
        raise ValueError(
            f"Unknown MLE CuPy fallback {model_config.mle_cupy_fallback!r}. "
            f"Expected {sorted(SUPPORTED_MLE_CUPY_FALLBACKS)}.")
    if model_config.mle_gpu_memory_gb is not None and model_config.mle_gpu_memory_gb <= 0:
        raise ValueError("mle_gpu_memory_gb must be positive")
    c = float(model_config.mle_terminal_c)
    if not (0.0 <= c < 1.0):
        raise ValueError(
            f"mle_terminal_c must satisfy 0 <= C < 1; got {c}.")


def params_from_vector(x, params_names):
    return {str(k).upper(): float(v) for k, v in zip(params_names, x)}


def neg_loglik(params, df, model_config):
    return evaluate_neg_loglik(params, df, model_config, return_df=False).neg_loglik


def evaluate_neg_loglik(params, df, model_config, return_df=False):
    validate_mle_config(model_config)
    data = prepare_mle_data(df)
    params = {str(k).upper(): float(v) for k, v in params.items()}

    latents = _compute_latent_arrays(data, params, model_config)
    like_result, backend_info = _evaluate_trial_likelihoods(
        data, latents, params, model_config)
    valid_mask = latents["valid_for_loss"]
    loglik_values = like_result.loglik
    finite_valid = valid_mask & np.isfinite(loglik_values)
    total_loglik = float(loglik_values[finite_valid].sum())
    n_trials_loss = int(finite_valid.sum())

    mle_df = None
    if return_df:
        mle_df = _build_mle_df(data, latents, like_result)

    return MLEEvalResult(
        neg_loglik=-float(total_loglik),
        loglik=float(total_loglik),
        n_trials_loss=n_trials_loss,
        n_trials_total=data.n_trials,
        mle_df=mle_df,
        backend_info=backend_info,
    )


def objective_from_vector(x, params_names, df, model_config):
    try:
        value = neg_loglik(params_from_vector(x, params_names), df, model_config)
    except (AssertionError, FloatingPointError, ValueError, OverflowError):
        return _objective_penalty(df)
    if not np.isfinite(value):
        return _objective_penalty(df)
    return value


def objective_from_population(x_matrix, params_names, df, model_config):
    """Vectorized DE objective for the MLE path.

    Compute population latents in candidate-major order, flatten all
    candidate-trial pairs, and make one solver call for the whole generation.

    The DE population size is chosen upstream from the memory ceiling, so the
    solver processes the supplied workload as one batch.

    Backend-agnostic: works identically on ``xp=numpy`` (no actual cost
    reduction except per-call Python overhead) and ``xp=cupy`` (collapses
    ``S`` GPU command streams into one, which is the main payoff).

    Per-candidate exceptions during latent computation are caught and replaced
    with the standard penalty, matching ``objective_from_vector``'s behavior.
    Candidates whose ``BOUND`` differs from the rest are evaluated one-at-a-time
    via ``objective_from_vector`` to avoid mixing grid sizes in the solver.
    """
    validate_mle_config(model_config)
    x_matrix = np.asarray(x_matrix, dtype=float)
    if x_matrix.ndim == 1:
        return np.asarray([objective_from_vector(
            x_matrix, params_names, df, model_config)], dtype=float)
    if x_matrix.ndim != 2:
        raise ValueError(
            f"objective_from_population expected a 2-D (n_params, S) "
            f"matrix; got shape {x_matrix.shape}")

    n_candidates = x_matrix.shape[1]
    prepared = prepare_mle_data(df) if not isinstance(df, PreparedMLEData) else df
    if n_candidates == 0:
        return np.empty(0, dtype=float)

    n_trials = prepared.n_trials
    penalty = _objective_penalty(prepared)
    losses = np.full(n_candidates, penalty, dtype=float)

    # Phase 1: compute latents for the whole population. Prepared data is
    # padded to equal session length, so we keep only a short Python loop over
    # trial position while updating all candidates and sessions at once.
    is_time_varying = (model_config.uses_decay_q_drift
                       or model_config.uses_decay_q_noise)
    n_t = int(round(model_config.t_dur / model_config.dt))
    backend = resolve_array_backend(
        model_config.mle_array_backend,
        model_config.mle_device_id,
        model_config.mle_cupy_fallback,
    )
    try:
        pop_latents = _compute_latent_population_equal_sessions(
            prepared, x_matrix, params_names, model_config, backend)
    except (FloatingPointError, ValueError, OverflowError, KeyError):
        return losses
    valid_cand_idx = np.arange(n_candidates, dtype=int)
    z_stack = pop_latents["z"]
    sigma_stack = pop_latents["sigma"]
    mu_stack = pop_latents["mu"]
    valid_stack = pop_latents["valid_for_loss"]
    no_choice_stack = pop_latents["no_choice"]
    bounds = pop_latents["bounds"]
    nondec = pop_latents["non_decision_time"]

    # Phase 2: solver requires uniform BOUND across the chunk (it sets up
    # x_grid from bound). In the default config BOUND is fixed at 1, so this
    # holds. If any candidate disagrees, evaluate those one-at-a-time and skip
    # the population path for them.
    sub_bounds = bounds[valid_cand_idx]
    if not np.allclose(sub_bounds, sub_bounds[0]):
        for i in valid_cand_idx:
            losses[i] = objective_from_vector(
                x_matrix[:, i], params_names, prepared, model_config)
        return losses
    shared_bound = float(sub_bounds[0])

    # Phase 3: configure one solver for the whole generation.
    solver = BatchedDiffusionSolver(
        xp=backend.xp,
        normal_cdf=backend.normal_cdf,
        show_progress=model_config.mle_show_progress,
        progress_desc=(
            f"MLE diffusion ({n_candidates} candidates x {n_trials} trials)"),
    )

    # Phase 4: flatten (S_valid, N) -> (S_valid * N) for ALL solver inputs.
    # Static observations are cached on the backend and broadcast across
    # candidates; per-candidate scalars are repeated for each trial.
    n_valid = valid_cand_idx.size
    backend_arrays = _prepared_session_arrays_for_backend(prepared, backend)
    flat_observed_choice = _broadcast_population_observation(
        backend.xp, backend_arrays, "choice_flat", n_valid)
    flat_observed_rt = _broadcast_population_observation(
        backend.xp, backend_arrays, "observed_rt_flat", n_valid)
    flat_nondec = backend.xp.repeat(
        backend.xp.asarray(nondec[valid_cand_idx], dtype=float), n_trials)

    flat_z = z_stack[valid_cand_idx].reshape(-1)
    flat_sigma = sigma_stack[valid_cand_idx].reshape(-1)
    flat_valid = valid_stack[valid_cand_idx].reshape(-1)
    flat_no_choice = no_choice_stack[valid_cand_idx].reshape(-1)
    if isinstance(mu_stack, TimeVaryingMuFactors):
        # O6: factored time-varying mu — pass the factors object straight
        # through. In the common case (`valid_cand_idx == arange(n_candidates)`)
        # no subsetting is needed since the factor arrays were already built
        # for the full population. If a future caller subsets candidates here,
        # this branch needs a `select_valid_candidates` helper.
        assert valid_cand_idx.size == n_candidates, (
            "factored mu doesn't yet support per-candidate subsetting at this "
            "level; ensure no candidates were dropped by upstream filters.")
        flat_mu = mu_stack
    elif is_time_varying:
        flat_mu = mu_stack[valid_cand_idx].reshape(-1, n_t)
    else:
        flat_mu = mu_stack[valid_cand_idx].reshape(-1)

    # Phase 5: single solver call across the entire flattened population.
    batch_result = batched_choice_rt_loglik(
        observed_choice_left=flat_observed_choice,
        observed_rt=flat_observed_rt,
        no_choice=flat_no_choice,
        valid_for_loss=flat_valid,
        z=flat_z,
        mu_values=flat_mu,
        sigma=flat_sigma,
        bound=shared_bound,
        non_decision_time=flat_nondec,
        dt=model_config.dt,
        dx=model_config.dx,
        tmax=model_config.t_dur,
        xp=backend.xp,
        normal_cdf=backend.normal_cdf,
        solver=solver,
        terminal_c=float(model_config.mle_terminal_c),
    )

    # Phase 6: reshape (S_valid * N,) → (S_valid, N) and aggregate.
    per_cand_loglik = batch_result.loglik.reshape(n_valid, n_trials)
    per_cand_valid = valid_stack[valid_cand_idx]
    finite_valid = per_cand_valid & np.isfinite(per_cand_loglik)
    total_loglik = np.where(finite_valid, per_cand_loglik, 0.0).sum(axis=1)
    neg_loglik = -total_loglik
    # Defensive: any candidate that somehow produced a non-finite sum gets
    # the penalty. Should not happen given the LOGLIK_FLOOR clamp inside the
    # solver, but cheap to guard.
    bad = ~np.isfinite(neg_loglik)
    neg_loglik[bad] = penalty
    losses[valid_cand_idx] = neg_loglik
    return losses


def result_payload(optim_res, params_names, params_init, params_bounds,
                   subject_df, model_config, population_info=None):
    if optim_res is None:
        x = np.asarray(params_init, dtype=float)
    else:
        x = np.asarray(optim_res.x, dtype=float)
    params = params_from_vector(x, params_names)
    data = prepare_mle_data(subject_df)
    eval_res = evaluate_neg_loglik(params, data, model_config, return_df=True)
    k = len(params_names)
    n = max(eval_res.n_trials_loss, 1)
    return dict(
        fit_mode="mle",
        mle_observation_model="choice_rt",
        subject_df=data.df,
        mle_df=eval_res.mle_df,
        dt=model_config.dt,
        dx=model_config.dx,
        t_dur=model_config.t_dur,
        noise_dt_scaling="sqrt_dt",
        include_Q=model_config.include_Q,
        include_RewardRate=model_config.include_RewardRate,
        params_names=np.asarray(params_names),
        params_init=np.asarray(params_init, dtype=float),
        params_bounds=np.asarray(params_bounds, dtype=float),
        OptimRes=optim_res,
        loglik=eval_res.loglik,
        neg_loglik=eval_res.neg_loglik,
        n_trials_loss=eval_res.n_trials_loss,
        n_trials_total=eval_res.n_trials_total,
        mle_backend_info=eval_res.backend_info,
        mle_population_info=population_info,
        aic=2 * k - 2 * eval_res.loglik,
        bic=k * np.log(n) - 2 * eval_res.loglik,
        model_config=model_config,
    )


def make_objective(params_names, df, model_config):
    return partial(objective_from_vector, params_names=params_names,
                   df=df, model_config=model_config)


def prepare_mle_data(df):
    if isinstance(df, PreparedMLEData):
        return df
    sort_cols = [col for col in ["SessId", "TrialNumber"] if col in df.columns]
    sorted_df = df.sort_values(sort_cols) if sort_cols else df
    sess_ids = sorted_df["SessId"].to_numpy()
    starts = []
    stops = []
    start = 0
    for i in range(1, len(sess_ids) + 1):
        if i == len(sess_ids) or sess_ids[i] != sess_ids[start]:
            starts.append(start)
            stops.append(i)
            start = i
    return PreparedMLEData(
        df=df,
        sorted_index=sorted_df.index.to_numpy(),
        session_slices=tuple(zip(starts, stops)),
        dv=_float_col(sorted_df, "DV"),
        valid=sorted_df["valid"].to_numpy(dtype=bool),
        choice_left=_float_col(sorted_df, "ChoiceLeft"),
        reward=_float_col(sorted_df, "ChoiceCorrect"),
        observed_rt=_float_col(sorted_df, "calcStimulusTime"),
        trial_number=_float_col(sorted_df, "TrialNumber"),
    )


def _param(params, name, default=None):
    if name in params:
        return params[name]
    if default is not None:
        return default
    raise KeyError(f"missing MLE parameter {name!r}")


def _objective_penalty(df):
    if isinstance(df, PreparedMLEData):
        n = max(df.n_trials, 1)
    else:
        n = max(int(getattr(df, "shape", [1])[0]), 1)
    return float(-np.log(1e-300) * n)


def _is_finite_number(value):
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _float_col(df, name):
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)


def _equal_session_shape(data):
    lengths = np.asarray([stop - start for start, stop in data.session_slices],
                         dtype=int)
    if lengths.size == 0:
        return 0, 0
    if not np.all(lengths == lengths[0]):
        raise AssertionError(
            "Population MLE requires padded sessions with equal length; "
            f"got session lengths {lengths.tolist()}.")
    n_sessions = int(lengths.size)
    trials_per_session = int(lengths[0])
    trial_grid = data.trial_number.reshape(n_sessions, trials_per_session)
    if not np.all(trial_grid == trial_grid[0][None, :]):
        raise AssertionError(
            "Population MLE requires every padded session to share the same "
            "TrialNumber grid.")
    return n_sessions, trials_per_session


def _prepared_session_arrays_for_backend(data, backend):
    key = (id(data), backend.actual_backend, backend.device_id)
    cached = _PREPARED_SESSION_BACKEND_CACHE.get(key)
    if cached is not None:
        return cached
    xp = backend.xp
    n_sessions, trials_per_session = _equal_session_shape(data)
    shaped = {
        "n_sessions": n_sessions,
        "trials_per_session": trials_per_session,
        "dv": xp.asarray(
            data.dv.reshape(n_sessions, trials_per_session), dtype=float),
        "valid": xp.asarray(
            data.valid.reshape(n_sessions, trials_per_session), dtype=bool),
        "choice": xp.asarray(
            data.choice_left.reshape(n_sessions, trials_per_session),
            dtype=float),
        "reward": xp.asarray(
            data.reward.reshape(n_sessions, trials_per_session), dtype=float),
        "choice_flat": xp.asarray(data.choice_left, dtype=float),
        "observed_rt_flat": xp.asarray(data.observed_rt, dtype=float),
    }
    _PREPARED_SESSION_BACKEND_CACHE[key] = shaped
    return shaped


def _broadcast_population_observation(xp, backend_arrays, name, n_candidates):
    """Return a flattened candidate-major broadcast view of a static trial array."""
    array = backend_arrays[name]
    return xp.broadcast_to(
        array[None, :], (int(n_candidates), int(array.shape[0]))
    ).reshape(-1)


def _param_population(theta, param_lookup, name, xp, default=None):
    if name in param_lookup:
        return theta[param_lookup[name]]
    if default is not None:
        return xp.full(theta.shape[1], float(default), dtype=float)
    raise KeyError(f"missing MLE parameter {name!r}")


def _compute_latent_population_equal_sessions(data, x_matrix, params_names,
                                              model_config, backend):
    xp = backend.xp
    n_candidates = int(x_matrix.shape[1])
    prepared = _prepared_session_arrays_for_backend(data, backend)
    n_sessions = prepared["n_sessions"]
    trials_per_session = prepared["trials_per_session"]
    n_trials = data.n_trials
    param_lookup = {str(name).upper(): i for i, name in enumerate(params_names)}
    theta = xp.asarray(x_matrix, dtype=float)

    dv = prepared["dv"]
    valid = prepared["valid"]
    choice = prepared["choice"]
    reward_arr = prepared["reward"]
    finite_dv = xp.isfinite(dv)
    valid_for_loss_2d = valid & finite_dv
    no_choice_2d = valid_for_loss_2d & xp.isnan(choice)

    drift_coef = _param_population(theta, param_lookup, "DRIFT_COEF", xp)
    noise_sigma = _param_population(theta, param_lookup, "NOISE_SIGMA", xp)
    bounds = _param_population(theta, param_lookup, "BOUND", xp, 1.0)
    nondec = _param_population(theta, param_lookup, "NON_DECISION_TIME", xp, 0.0)
    alpha = _param_population(theta, param_lookup, "ALPHA", xp, np.nan)
    beta = _param_population(theta, param_lookup, "BETA", xp, np.nan)
    bias_coef = _param_population(theta, param_lookup, "BIAS_COEF", xp, 0.0)
    q_offset = _param_population(theta, param_lookup, "Q_VAL_OFFSET", xp, 0.0)
    q_coef = _param_population(theta, param_lookup, "Q_VAL_COEF", xp, 0.0)
    q_decay_rate = _param_population(
        theta, param_lookup, "Q_VAL_DECAY_RATE", xp, 1.0)

    q_left = xp.full((n_candidates, n_sessions), 0.5, dtype=float)
    q_right = xp.full((n_candidates, n_sessions), 0.5, dtype=float)
    reward_rate = xp.full((n_candidates, n_sessions), 0.5, dtype=float)
    z = xp.empty((n_candidates, n_sessions, trials_per_session), dtype=float)
    sigma = xp.empty_like(z)
    time_varying = model_config.uses_decay_q_drift or model_config.uses_decay_q_noise
    n_t = int(round(model_config.t_dur / model_config.dt))
    # Per-candidate decay shapes that don't depend on the Q-state recurrence —
    # safe to compute up front. Used by O6 factored mu (and also by the
    # legacy time-varying branch where we still materialize the (..., n_t)
    # tensor, gated below).
    decay_form_per_cand = None
    log_decay_per_cand = None
    if time_varying:
        t_idx = xp.arange(n_t, dtype=float)
        decay_base = 1.0 - t_idx / float(n_t)
        if model_config.uses_decay_q_drift:
            decay_form_per_cand = decay_base[None, :] ** q_decay_rate[:, None]
        if model_config.uses_decay_q_noise:
            log_decay_per_cand = 1.0 - (
                q_decay_rate[:, None] * xp.log(t_idx[None, :] + 1.0)
                / np.log(n_t + 1.0)
            )

    # O6: for time-varying mu, accumulate ONLY the per-trial factors that the
    # Q-state recurrence touches. The (n_candidates, n_sessions,
    # trials_per_session, n_t) mu tensor is gone — it was ~7 GB on a typical
    # population GPU fit, and per-step mu can be reconstructed cheaply from
    # the (b,) per-trial scalars and the (n_candidates, n_t) decay shapes
    # above.
    if time_varying:
        mu = None
        q_drift_coef_pop = (
            xp.empty((n_candidates, n_sessions, trials_per_session), dtype=float)
            if model_config.uses_decay_q_drift else None)
        q_abs_x_qcoef_pop = (
            xp.empty((n_candidates, n_sessions, trials_per_session), dtype=float)
            if model_config.uses_decay_q_noise else None)
        q_sign_pop = (
            xp.empty((n_candidates, n_sessions, trials_per_session), dtype=float)
            if model_config.uses_decay_q_noise else None)
        sigma_for_noise_pop = (
            xp.empty((n_candidates, n_sessions, trials_per_session), dtype=float)
            if model_config.uses_decay_q_noise else None)
    else:
        mu = xp.empty_like(z)
        q_drift_coef_pop = None
        q_abs_x_qcoef_pop = None
        q_sign_pop = None
        sigma_for_noise_pop = None

    for trial_pos in range(trials_per_session):
        q_left_clip = xp.clip(q_left, state_updates.LOG_CIEL, 1.0)
        q_right_clip = xp.clip(q_right, state_updates.LOG_CIEL, 1.0)
        q_rel = xp.log(q_left_clip / q_right_clip) / state_updates.LOG_CEIL_MAX

        if model_config.uses_q_bias:
            z_t = xp.clip(
                bias_coef[:, None] * q_rel + q_offset[:, None], -1.0, 1.0)
        else:
            z_t = xp.zeros_like(q_rel)
        sigma_t = (
            reward_rate * noise_sigma[:, None]
            if model_config.include_RewardRate
            else xp.broadcast_to(noise_sigma[:, None], q_rel.shape)
        )
        base_mu = drift_coef[:, None] * dv[None, :, trial_pos]
        z[:, :, trial_pos] = z_t
        sigma[:, :, trial_pos] = sigma_t

        if time_varying:
            # O6: store the per-trial Q-state-dependent factors. The (n_t,)
            # decay shapes are per-candidate constants (computed once above);
            # mu_t at any timestep is reconstructed cheaply by the solver
            # from these factors + a column gather from decay_form_per_cand
            # / log_decay_per_cand.
            if model_config.uses_decay_q_drift:
                assert q_drift_coef_pop is not None  # narrow for pyright
                q_for_drift = xp.clip(q_rel + q_offset[:, None], -1.0, 1.0)
                q_drift_coef_pop[:, :, trial_pos] = q_for_drift * q_coef[:, None]
            if model_config.uses_decay_q_noise:
                assert q_abs_x_qcoef_pop is not None
                assert q_sign_pop is not None
                assert sigma_for_noise_pop is not None
                q_abs_x_qcoef_pop[:, :, trial_pos] = xp.abs(q_rel) * q_coef[:, None]
                q_sign_pop[:, :, trial_pos] = xp.where(
                    q_rel < 0.0, -1.0, 1.0)
                sigma_for_noise_pop[:, :, trial_pos] = sigma_t
        else:
            assert mu is not None
            mu[:, :, trial_pos] = base_mu

        valid_t = valid_for_loss_2d[None, :, trial_pos]
        reward_t = xp.nan_to_num(reward_arr[None, :, trial_pos], nan=0.0)
        choice_t = choice[None, :, trial_pos]
        if model_config.include_Q:
            left_mask = valid_t & (choice_t == 1)
            right_mask = valid_t & (choice_t == 0)
            q_left = xp.where(
                left_mask,
                q_left + alpha[:, None] * (reward_t - q_left),
                q_left,
            )
            q_right = xp.where(
                right_mask,
                q_right + alpha[:, None] * (reward_t - q_right),
                q_right,
            )
        if model_config.include_RewardRate:
            reward_rate = xp.where(
                valid_t,
                reward_rate + beta[:, None] * (reward_t - reward_rate),
                reward_rate,
            )

    flat_shape = (n_candidates, n_trials)
    valid_flat = xp.broadcast_to(
        valid_for_loss_2d[None, :, :],
        (n_candidates, n_sessions, trials_per_session),
    ).reshape(flat_shape)
    no_choice_flat = xp.broadcast_to(
        no_choice_2d[None, :, :],
        (n_candidates, n_sessions, trials_per_session),
    ).reshape(flat_shape)

    if time_varying:
        # O6 factored mu. base_mu is independent of the recurrence so we
        # build it here in one shot rather than per trial_pos. Per-trial
        # factor arrays are flattened to (n_candidates * n_trials,) so the
        # solver can index by valid_idx directly.
        base_mu_pop = drift_coef[:, None, None] * dv[None, :, :]
        flat_n = n_candidates * n_trials
        candidate_id_flat = np.broadcast_to(
            np.arange(n_candidates, dtype=np.int64)[:, None],
            (n_candidates, n_trials),
        ).reshape(-1).copy()
        mu_out = TimeVaryingMuFactors(
            n_trials=int(flat_n),
            n_t=int(n_t),
            dt=float(model_config.dt),
            base_mu=asnumpy(xp, base_mu_pop).reshape(-1).astype(float, copy=False),
            candidate_id_per_trial=candidate_id_flat,
            q_drift_coef_per_trial=(
                None if q_drift_coef_pop is None
                else asnumpy(xp, q_drift_coef_pop).reshape(-1).astype(float, copy=False)),
            decay_form_per_cand=(
                None if decay_form_per_cand is None
                else asnumpy(xp, decay_form_per_cand).astype(float, copy=False)),
            q_abs_x_qcoef_per_trial=(
                None if q_abs_x_qcoef_pop is None
                else asnumpy(xp, q_abs_x_qcoef_pop).reshape(-1).astype(float, copy=False)),
            q_sign_per_trial=(
                None if q_sign_pop is None
                else asnumpy(xp, q_sign_pop).reshape(-1).astype(float, copy=False)),
            sigma_per_trial=(
                None if sigma_for_noise_pop is None
                else asnumpy(xp, sigma_for_noise_pop).reshape(-1).astype(float, copy=False)),
            log_decay_per_cand=(
                None if log_decay_per_cand is None
                else asnumpy(xp, log_decay_per_cand).astype(float, copy=False)),
        )
    else:
        assert mu is not None
        mu_out = asnumpy(xp, mu.reshape(flat_shape))
    return {
        "z": asnumpy(xp, z.reshape(flat_shape)),
        "sigma": asnumpy(xp, sigma.reshape(flat_shape)),
        "mu": mu_out,
        "valid_for_loss": asnumpy(xp, valid_flat).astype(bool),
        "no_choice": asnumpy(xp, no_choice_flat).astype(bool),
        "bounds": asnumpy(xp, bounds),
        "non_decision_time": asnumpy(xp, nondec),
    }


def _compute_latent_arrays(data, params, model_config):
    n = data.n_trials
    q_left_before = np.full(n, 0.5, dtype=float)
    q_right_before = np.full(n, 0.5, dtype=float)
    q_rel_before = np.full(n, 0.0, dtype=float)
    reward_rate_before = np.full(n, 0.5, dtype=float)
    q_left_after = np.full(n, 0.5, dtype=float)
    q_right_after = np.full(n, 0.5, dtype=float)
    reward_rate_after = np.full(n, 0.5, dtype=float)

    alpha = _param(params, "ALPHA", np.nan)
    beta = _param(params, "BETA", np.nan)
    finite_dv = np.isfinite(data.dv)
    valid_for_loss = data.valid & finite_dv

    for start, stop in data.session_slices:
        q_left = 0.5
        q_right = 0.5
        reward_rate = 0.5
        for i in range(start, stop):
            q_left_before[i] = q_left
            q_right_before[i] = q_right
            q_rel = float(state_updates.compute_q_value(q_left, q_right))
            q_rel_before[i] = q_rel
            reward_rate_before[i] = reward_rate
            next_q_left = q_left
            next_q_right = q_right
            next_reward_rate = reward_rate
            if valid_for_loss[i]:
                reward = 0.0 if np.isnan(data.reward[i]) else float(data.reward[i])
                choice_left = data.choice_left[i]
                if model_config.include_Q and not np.isnan(choice_left):
                    if int(choice_left) == 1:
                        next_q_left = q_left + alpha * (reward - q_left)
                    else:
                        next_q_right = q_right + alpha * (reward - q_right)
                if model_config.include_RewardRate:
                    next_reward_rate = reward_rate + beta * (reward - reward_rate)
                q_left = float(next_q_left)
                q_right = float(next_q_right)
                reward_rate = float(next_reward_rate)
            q_left_after[i] = q_left
            q_right_after[i] = q_right
            reward_rate_after[i] = reward_rate

    z = _compute_z_array(q_left_before, q_right_before, q_rel_before,
                         params, model_config)
    sigma = _compute_sigma_array(reward_rate_before, params, model_config)
    mu = _compute_mu_array(data.dv, q_rel_before, sigma, params, model_config)
    return {
        "q_left_before": q_left_before,
        "q_right_before": q_right_before,
        "q_rel_before": q_rel_before,
        "reward_rate_before": reward_rate_before,
        "z": z,
        "mu": mu,
        "sigma": sigma,
        "valid_for_loss": valid_for_loss,
        "no_choice": valid_for_loss & np.isnan(data.choice_left),
        "q_left_after": q_left_after,
        "q_right_after": q_right_after,
        "reward_rate_after": reward_rate_after,
    }


def _compute_z_array(q_left, q_right, q_rel_before, params, model_config):
    if not model_config.uses_q_bias:
        return np.zeros_like(q_rel_before, dtype=float)
    z = _param(params, "BIAS_COEF") * q_rel_before + _param(
        params, "Q_VAL_OFFSET", 0.0)
    return np.clip(z, -1, 1)


def _compute_sigma_array(reward_rate_before, params, model_config):
    base_sigma = _param(params, "NOISE_SIGMA")
    if model_config.include_RewardRate:
        return reward_rate_before * base_sigma
    return np.full_like(reward_rate_before, base_sigma, dtype=float)


def _compute_mu_array(dv, q_rel_before, sigma, params, model_config):
    base_mu = _param(params, "DRIFT_COEF") * np.asarray(dv, dtype=float)
    if not model_config.uses_decay_q_drift and not model_config.uses_decay_q_noise:
        return base_mu

    n_t = int(round(model_config.t_dur / model_config.dt))
    # Keep scalar and time-varying drift in numeric arrays. Object arrays would
    # force Python loops in the batched solver and make backend transfers slow.
    mu = np.repeat(base_mu[:, None], n_t, axis=1)
    if model_config.uses_decay_q_drift:
        indices = np.arange(n_t)
        decay_form = (1 - indices / n_t) ** _param(params, "Q_VAL_DECAY_RATE")
        q_for_drift = np.clip(
            q_rel_before + _param(params, "Q_VAL_OFFSET", 0.0), -1, 1)
        mu += q_for_drift[:, None] * decay_form[None, :] * _param(
            params, "Q_VAL_COEF")
    if model_config.uses_decay_q_noise:
        mu += (
            sigma[:, None]
            * _decaying_q_noise_array(q_rel_before, params, n_t)
            / model_config.dt
        )
    return mu


def _decaying_q_noise_array(q_rel_before, params, n_t):
    indices = np.arange(n_t)
    decay = 1 - (
        _param(params, "Q_VAL_DECAY_RATE") * np.log(indices + 1)
        / np.log(n_t + 1)
    )
    q_abs = np.abs(q_rel_before)[:, None] * _param(params, "Q_VAL_COEF")
    decayed = np.maximum(q_abs - decay[None, :], 0)
    return np.where(q_rel_before[:, None] < 0, -decayed, decayed)


MIN_POPULATION_CANDIDATES = 64


def estimate_population_settings(model_config, n_trials, n_params, bound=1.0):
    """Return candidate count and SciPy popsize from the memory ceiling.

    Enforces a floor of ``MIN_POPULATION_CANDIDATES`` actual candidates so
    DE has enough diversity to explore the parameter space even when the
    memory budget would otherwise pick a smaller population.
    """
    flat_capacity, memory_estimate = estimate_flat_trial_capacity_for_memory(
        model_config.mle_gpu_memory_gb, bound, model_config.dx,
        model_config.t_dur, model_config.dt)
    if flat_capacity is None:
        target_candidates = 100 * max(int(n_params), 1)
        memory_estimate = None
    else:
        target_candidates = max(int(flat_capacity) // max(int(n_trials), 1), 1)
    n_params = max(int(n_params), 1)
    # SciPy's actual population is `scipy_popsize * n_params`. Round up so
    # `actual_candidates >= MIN_POPULATION_CANDIDATES`.
    min_popsize = -(-MIN_POPULATION_CANDIDATES // n_params)  # ceil division
    scipy_popsize = max(target_candidates // n_params, min_popsize, 1)
    actual_candidates = scipy_popsize * n_params
    population_info = {
        "target_candidates": int(target_candidates),
        "actual_candidates": int(actual_candidates),
        "scipy_popsize": int(scipy_popsize),
        "n_trials": int(n_trials),
        "n_params": int(n_params),
        "memory_estimate": memory_estimate,
    }
    return population_info


def _effective_memory_estimate(model_config, params):
    bound = _param(params, "BOUND", 1.0)
    _, estimate = estimate_flat_trial_capacity_for_memory(
        model_config.mle_gpu_memory_gb, bound, model_config.dx,
        model_config.t_dur, model_config.dt)
    return estimate


def _evaluate_trial_likelihoods(data, latents, params, model_config):
    backend = resolve_array_backend(
        model_config.mle_array_backend,
        model_config.mle_device_id,
        model_config.mle_cupy_fallback,
    )
    if model_config.mle_use_batched_likelihood:
        return _evaluate_trial_likelihoods_batched(
            data, latents, params, model_config, backend)
    likes, info = _evaluate_trial_likelihoods_rowwise(
        data, latents, params, model_config)
    info.update(backend.metadata())
    info["likelihood_evaluator"] = "rowwise"
    return _likes_to_result(likes, data.n_trials), info


def _evaluate_trial_likelihoods_rowwise(data, latents, params, model_config):
    likes = []
    bound = _param(params, "BOUND", 1.0)
    non_decision_time = _param(params, "NON_DECISION_TIME", 0.0)
    for i in range(data.n_trials):
        if not latents["valid_for_loss"][i]:
            likes.append(None)
            continue
        likes.append(trial_choice_rt_loglik(
            observed_choice_left=data.choice_left[i],
            observed_rt=data.observed_rt[i],
            z=latents["z"][i],
            mu=latents["mu"][i],
            sigma=latents["sigma"][i],
            bound=bound,
            non_decision_time=non_decision_time,
            dt=model_config.dt,
            dx=model_config.dx,
            tmax=model_config.t_dur,
            diffusion_backend=model_config.diffusion_backend,
            no_choice=latents["no_choice"][i],
        ))
    return likes, {
        "requested_backend": "rowwise",
        "actual_backend": "rowwise",
        "device_id": None,
        "warning": None,
    }


def _evaluate_trial_likelihoods_batched(data, latents, params, model_config,
                                        backend):
    n = data.n_trials
    if n == 0:
        info = backend.metadata()
        info["likelihood_evaluator"] = "batched"
        return _likes_to_result([], 0), info
    memory_estimate = _effective_memory_estimate(model_config, params)
    solver = BatchedDiffusionSolver(
        xp=backend.xp,
        normal_cdf=backend.normal_cdf,
        show_progress=False,
    )
    batch_result = batched_choice_rt_loglik(
        observed_choice_left=data.choice_left,
        observed_rt=data.observed_rt,
        no_choice=latents["no_choice"],
        valid_for_loss=latents["valid_for_loss"],
        z=latents["z"],
        mu_values=latents["mu"],
        sigma=latents["sigma"],
        bound=_param(params, "BOUND", 1.0),
        non_decision_time=np.full(n, _param(params, "NON_DECISION_TIME", 0.0)),
        dt=model_config.dt,
        dx=model_config.dx,
        tmax=model_config.t_dur,
        xp=backend.xp,
        normal_cdf=backend.normal_cdf,
        solver=solver,
        terminal_c=float(model_config.mle_terminal_c),
    )
    info = backend.metadata()
    info["likelihood_evaluator"] = "batched"
    info["memory_estimate"] = memory_estimate
    info["solver"] = batch_result.metadata
    return batch_result, info


def _array_trial_likelihood(batch_result, i):
    from .mle_likelihood import TrialLikelihood

    return TrialLikelihood(
        loglik=float(batch_result.loglik[i]),
        choice_prob_or_density=float(batch_result.choice_prob_or_density[i]),
        decision_time=float(batch_result.decision_time[i]),
        survival_at_tmax=float(batch_result.survival_at_tmax[i]),
        upper_hit_prob_tmax=float(batch_result.upper_hit_prob_tmax[i]),
        lower_hit_prob_tmax=float(batch_result.lower_hit_prob_tmax[i]),
    )


def _likes_to_result(likes, n_trials):
    result = BatchedLikelihoodResult(
        loglik=np.full(n_trials, np.nan, dtype=float),
        choice_prob_or_density=np.full(n_trials, np.nan, dtype=float),
        decision_time=np.full(n_trials, np.nan, dtype=float),
        survival_at_tmax=np.full(n_trials, np.nan, dtype=float),
        upper_hit_prob_tmax=np.full(n_trials, np.nan, dtype=float),
        lower_hit_prob_tmax=np.full(n_trials, np.nan, dtype=float),
    )
    for i, like in enumerate(likes):
        if like is None:
            continue
        result.loglik[i] = like.loglik
        result.choice_prob_or_density[i] = like.choice_prob_or_density
        result.decision_time[i] = like.decision_time
        result.survival_at_tmax[i] = like.survival_at_tmax
        result.upper_hit_prob_tmax[i] = like.upper_hit_prob_tmax
        result.lower_hit_prob_tmax[i] = like.lower_hit_prob_tmax
    return result


def _build_mle_df(data, latents, like_result):
    mu = latents["mu"]
    if np.asarray(mu).ndim == 1:
        mu_display = mu
    else:
        mu_display = mu[:, 0]
    latent_df = pd.DataFrame({
        "_row_index": data.sorted_index,
        "mle_Q_left_before": latents["q_left_before"],
        "mle_Q_right_before": latents["q_right_before"],
        "mle_Q_rel_before": latents["q_rel_before"],
        "mle_reward_rate_before": latents["reward_rate_before"],
        "mle_z": latents["z"],
        "mle_mu": mu_display,
        "mle_sigma": latents["sigma"],
        "mle_decision_time_observed": like_result.decision_time,
        "mle_choice_prob_or_density": like_result.choice_prob_or_density,
        "mle_loglik": like_result.loglik,
        "mle_valid_for_loss": latents["valid_for_loss"],
        "mle_survival_at_tmax": like_result.survival_at_tmax,
        "mle_upper_hit_prob_tmax": like_result.upper_hit_prob_tmax,
        "mle_lower_hit_prob_tmax": like_result.lower_hit_prob_tmax,
        "mle_Q_left_after": latents["q_left_after"],
        "mle_Q_right_after": latents["q_right_after"],
        "mle_reward_rate_after": latents["reward_rate_after"],
    }).set_index("_row_index")
    return data.df.copy().join(latent_df, how="left")


def _compute_z(state, params, model_config, q_rel_before):
    if not model_config.uses_q_bias:
        return 0.0
    return float(state_updates.compute_starting_point_z(
        state.q_left,
        state.q_right,
        delta=_param(params, "BIAS_COEF"),
        offset=_param(params, "Q_VAL_OFFSET", 0.0),
        include_Q=True,
    ))


def _compute_mu(coherence, params, model_config, q_rel_before, sigma):
    base_mu = _param(params, "DRIFT_COEF") * coherence
    if not model_config.uses_decay_q_drift and not model_config.uses_decay_q_noise:
        return base_mu

    n_t = int(round(model_config.t_dur / model_config.dt))
    mu = np.full(n_t, base_mu, dtype=float)
    if model_config.uses_decay_q_drift:
        indices = np.arange(n_t)
        decay_form = (1 - indices / n_t) ** _param(params, "Q_VAL_DECAY_RATE")
        q_for_drift = np.clip(q_rel_before + _param(params, "Q_VAL_OFFSET", 0.0),
                              -1, 1)
        mu += q_for_drift * decay_form * _param(params, "Q_VAL_COEF")
    if model_config.uses_decay_q_noise:
        mu += sigma * _decaying_q_noise(q_rel_before, params, n_t) / model_config.dt
    return mu


def _decaying_q_noise(q_rel_before, params, n_t):
    indices = np.arange(n_t)
    decay = 1 - (
        _param(params, "Q_VAL_DECAY_RATE") * np.log(indices + 1)
        / np.log(n_t + 1)
    )
    q_abs = abs(q_rel_before) * _param(params, "Q_VAL_COEF")
    decayed = np.maximum(q_abs - decay, 0)
    if q_rel_before < 0:
        decayed = -decayed
    return decayed


def _row_record(row_idx, state, q_rel_before, reward_rate_before, z, mu, sigma,
                trial_like, valid_for_loss, q_left_after, q_right_after,
                reward_rate_after):
    if trial_like is None:
        decision_time = np.nan
        choice_prob_or_density = np.nan
        loglik = np.nan
        survival = np.nan
        upper_prob = np.nan
        lower_prob = np.nan
    else:
        decision_time = trial_like.decision_time
        choice_prob_or_density = trial_like.choice_prob_or_density
        loglik = trial_like.loglik
        survival = trial_like.survival_at_tmax
        upper_prob = trial_like.upper_hit_prob_tmax
        lower_prob = trial_like.lower_hit_prob_tmax

    if np.isscalar(mu):
        mu_value = float(mu)
    else:
        mu_value = float(np.asarray(mu)[0])

    return {
        "_row_index": row_idx,
        "mle_Q_left_before": state.q_left,
        "mle_Q_right_before": state.q_right,
        "mle_Q_rel_before": q_rel_before,
        "mle_reward_rate_before": reward_rate_before,
        "mle_z": z,
        "mle_mu": mu_value,
        "mle_sigma": sigma,
        "mle_decision_time_observed": decision_time,
        "mle_choice_prob_or_density": choice_prob_or_density,
        "mle_loglik": loglik,
        "mle_valid_for_loss": valid_for_loss,
        "mle_survival_at_tmax": survival,
        "mle_upper_hit_prob_tmax": upper_prob,
        "mle_lower_hit_prob_tmax": lower_prob,
        "mle_Q_left_after": q_left_after,
        "mle_Q_right_after": q_right_after,
        "mle_reward_rate_after": reward_rate_after,
    }
