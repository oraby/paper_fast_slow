import datetime
import weakref
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from . import state_updates
from .array_backend import asnumpy, resolve_array_backend
from .initvals import MLE_TERMINAL_C
from .mle_batch import (
    BatchedLikelihoodResult,
    BatchedDiffusionSolver,
    batched_choice_rt_loglik,
    estimate_flat_trial_capacity_for_memory,
)
from .mle_likelihood import trial_choice_rt_loglik, LOGLIK_FLOOR


SUPPORTED_MLE_BIASES = {"None_", "Q-Val (Offset)"}
SUPPORTED_MLE_ARRAY_BACKENDS = {"auto", "numpy", "cupy"}
SUPPORTED_MLE_CUPY_FALLBACKS = {"numpy", "error"}
_PREPARED_SESSION_BACKEND_CACHE = {}

# Floor on DE actual_candidates (= scipy_popsize * n_params). Keeps the
# population diverse enough for DE to explore the parameter space even
# when the memory budget would otherwise pick a smaller batch. Used as
# the default for ``MLEModelConfig.mle_min_population_candidates``, which
# the CLI's ``--mle-min-population`` overrides per-fit.
MIN_POPULATION_CANDIDATES = 64


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
    mle_terminal_c: float = MLE_TERMINAL_C.Default
    # DE population floor — see ``MIN_POPULATION_CANDIDATES`` above for
    # the rationale. Exposed as a per-fit knob so users can dial down
    # on tight-memory GPUs (smaller floor → smaller batch → fits the
    # budget) or up for harder loss landscapes that need more
    # candidates per generation. CLI: ``--mle-min-population``.
    mle_min_population_candidates: int = MIN_POPULATION_CANDIDATES
    # Bound-RewardRate per-trial bound flag. Set True when the user
    # selects a "Bound-RewardRate*" drift; ``fit.simulateDDM`` derives
    # it from ``drift_fn_str``. The per-trial bound is
    # b_t = BOUND * (2 - r_t) = BOUND + (1 - r_t) * BOUND (see
    # state_updates.bound_scale_from_reward_rate); the MLE compute paths
    # apply the DDM rescaling identity per trial with scale s_t = 2 - r_t:
    # μ /= s_t, σ /= s_t, z = clip(absolute_bias, ±b_t) / s_t. Solver still
    # receives the scalar BOUND — no solver changes needed; see
    # scale_bound_equivalence.ipynb for the equivalence derivation.
    uses_per_trial_bound: bool = False
    # ``--scale-bound`` flag. Swaps which of (BOUND, NOISE_SIGMA) is
    # the fitted scale axis (the two are near-degenerate in the DDM
    # loss landscape). When True, bias is interpreted in absolute
    # DDM-state units (clipped to ±BOUND) rather than fraction-of-bound.
    uses_scaled_bound: bool = False
    # Per-condition sample-balancing columns for the MLE loss. Empty
    # tuple → unweighted (legacy bit-exact sum). Populated from the
    # ``--mle-conditions`` CLI flag. EXCLUDED from ``fit.evolveFP``'s
    # filename composition by design so the user can A/B test by
    # overwriting the same pickle — the column list is preserved
    # inside the saved ``model_config`` for traceability. See
    # ``prepare_mle_data._compute_trial_weights`` for the formula:
    # ``weight[i] = total_valid / (num_groups * group_size[gid(i)])``.
    mle_condition_columns: tuple = ()
    # Weighted choice-vs-RT loss (``--mle-choice-weight`` /
    # ``--mle-rt-weight``). The joint per-trial loglik is split into a
    # choice component and an RT-given-choice component (see
    # ``_apply_choice_rt_weights``) and recombined as
    # ``w_choice·log P(c) + w_rt·log p(rt|c)``. Defaults 1.0/1.0.
    # EXCLUDED from ``fit.evolveFP`` (filename invariant — A/B by
    # overwriting), preserved in the saved ``model_config``.
    mle_choice_weight: float = 1.0
    mle_rt_weight: float = 1.0
    # Choice-probability normalization for the weighted split.
    # ``"marginal"`` uses the raw bound-hit prob P_mix(c) — at weights
    # (1,1) this reproduces today's joint loss bit-exactly (legacy).
    # ``"conditional"`` divides by P_mix(L)+P_mix(R), i.e. conditions
    # on a decision being made (chat Answer 29), isolating side-bias
    # from overall decisiveness. The DATACLASS default is ``"marginal"``
    # for back-compat: existing direct-construct tests and old pickles
    # (whose saved config predates this field) keep the legacy
    # objective. The CLI / ``fit.simulateDDM`` default is
    # ``"conditional"`` (the user-facing recommended form).
    mle_choice_norm: str = "marginal"
    # Outer joint-loss weights (``--mle-mle-weight`` / ``--mle-chi2-weight``).
    # The DE objective becomes
    # ``w_mle·(MLE_negloglik/ref_mle) + w_chi2·(Chi2/ref_chi2)`` — a composite
    # of the teacher-forced per-trial MLE term and the generative
    # Ratcliff-quantile Chi² term, each divided by the subject's standalone-best
    # loss for that term (``fit._jointVectorizedObjectiveWrapper``) so each is 1 at its own
    # optimum. Default ``(1.0, 0.0)`` ⇒ pure MLE: when ``mle_chi2_weight == 0``
    # the joint path is never taken (``fit._processSubject`` keeps the
    # unchanged, byte-identical vectorized-MLE driver and never runs the Chi²
    # simulation). Joint fits get a ``_mleW{m}_chi2W{c}`` suffix from
    # ``fit.evolveFP``; pure MLE keeps the plain name so it can serve as the
    # reference. Preserved in the saved ``model_config``. Joint mode is no
    # longer pure MLE, so the
    # ``aic``/``bic`` in ``result_payload`` reflect only the MLE component.
    mle_mle_weight: float = 1.0
    mle_chi2_weight: float = 0.0

    @property
    def uses_q_bias(self):
        return "Q-Val" in self.bias_fn_str

    @property
    def uses_per_trial_drift(self):
        """DriftGain-RewardRate: the reward rate modulates the coherence
        drift (``mu *= g(r_t)``) with sigma and the bound both flat.

        Derived from ``drift_fn_str`` rather than being an explicit field
        like ``uses_per_trial_bound``: no pre-existing pickle can carry a
        ``DriftGain*`` drift string, so the derivation is total — and
        deriving it means ``mle_reeval.build_mle_config`` and
        ``visualize._evaluate_mle_loss_for_gui`` (which already thread
        ``drift_fn_str``) need no changes and cannot forget to set it.
        """
        return self.drift_fn_str.startswith("DriftGain")

    @property
    def rr_drift_map(self):
        """The ``r -> drift gain`` mapping for ``uses_per_trial_drift``.

        Encoded in the registry key (``DriftGain(1+r)-*`` vs the untagged
        ``DriftGain-*``) because the mapping is bound into the Chi² drift
        function by ``partialWithNames``; see ``drift.DRIFT_FN_DICT``.
        """
        return ("1+r" if self.drift_fn_str.startswith("DriftGain(1+r)")
                else state_updates.DEFAULT_RR_DRIFT_MAP)

    @property
    def sigma_rr_channel(self):
        """``rr_channel`` to pass to ``state_updates.compute_trial_sigma``.

        ``"drift"`` (flat sigma) only for the DriftGain family. Everything
        else — including Bound-RewardRate — stays ``"noise"``, which keeps
        the scalar helper's callers byte-identical to before; see the KNOWN
        GAP note in ``compute_trial_sigma``.
        """
        return "drift" if self.uses_per_trial_drift else "noise"

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
    # Per-trial sample-balancing weights, indexed in the same order as
    # the other arrays. ``None`` means "unweighted sum" (legacy);
    # otherwise ``(n_trials,)`` float64. Populated by
    # ``prepare_mle_data`` when ``MLEModelConfig.mle_condition_columns``
    # is non-empty. Invalid trials get weight 0 — they're masked out
    # at the sum site anyway; the 0 documents intent.
    trial_weights: np.ndarray | None = None

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
    if int(model_config.mle_min_population_candidates) < 1:
        raise ValueError(
            "mle_min_population_candidates must be a positive integer; "
            f"got {model_config.mle_min_population_candidates!r}")
    c = float(model_config.mle_terminal_c)
    if not (MLE_TERMINAL_C.Min <= c <= MLE_TERMINAL_C.Max):
        raise ValueError(
            f"mle_terminal_c must satisfy "
            f"{MLE_TERMINAL_C.Min} <= C <= {MLE_TERMINAL_C.Max}; got {c}.")
    for name in ("mle_choice_weight", "mle_rt_weight",
                 "mle_mle_weight", "mle_chi2_weight"):
        w = float(getattr(model_config, name))
        if not np.isfinite(w) or w < 0.0:
            raise ValueError(
                f"{name} must be a finite, non-negative number; got {w!r}.")
    if model_config.mle_choice_norm not in ("conditional", "marginal"):
        raise ValueError(
            "mle_choice_norm must be 'conditional' or 'marginal'; "
            f"got {model_config.mle_choice_norm!r}.")
    # DriftGain-RewardRate consistency. All three of these are impossible by
    # construction (the flags derive from one drift_fn_str), so they're really
    # guards against a hand-built config -- but a silently mis-scaled drift
    # would be near-impossible to spot in a fitted loss, hence the loud check.
    if model_config.uses_per_trial_drift:
        if model_config.rr_drift_map not in state_updates.RR_DRIFT_MAPS:
            raise ValueError(
                f"Unknown reward-rate drift map "
                f"{model_config.rr_drift_map!r}; expected one of "
                f"{list(state_updates.RR_DRIFT_MAPS)}.")
        if not model_config.include_RewardRate:
            raise ValueError(
                f"drift_fn_str {model_config.drift_fn_str!r} routes the reward "
                f"rate to the drift but include_RewardRate is False, so no "
                f"reward rate would ever be learned.")
        if model_config.uses_per_trial_bound:
            raise ValueError(
                "uses_per_trial_drift and uses_per_trial_bound are mutually "
                "exclusive reward-rate channels; got both.")


def params_from_vector(x, params_names):
    return {str(k).upper(): float(v) for k, v in zip(params_names, x)}


def neg_loglik(params, df, model_config):
    return evaluate_neg_loglik(params, df, model_config, return_df=False).neg_loglik


def evaluate_neg_loglik(params, df, model_config, return_df=False):
    validate_mle_config(model_config)
    data = prepare_mle_data(df, model_config.mle_condition_columns)
    params = {str(k).upper(): float(v) for k, v in params.items()}

    latents = _compute_latent_arrays(data, params, model_config)
    like_result, backend_info = _evaluate_trial_likelihoods(
        data, latents, params, model_config)
    valid_mask = latents["valid_for_loss"]
    # Choice-vs-RT reweighting: transform the per-trial joint loglik into
    # ``w_choice·log P(c) + w_rt·log p(rt|c)`` before any further summing.
    # Identity (returns the same array) on the marginal + (1,1) fast path,
    # so the legacy sum below stays bit-exact. See
    # ``_apply_choice_rt_weights``.
    loglik_values = _apply_choice_rt_weights(
        like_result.loglik,
        like_result.upper_hit_prob_tmax,
        like_result.lower_hit_prob_tmax,
        data.choice_left,
        latents["no_choice"],
        float(latents["lapse_rate"]),
        model_config.mle_choice_weight,
        model_config.mle_rt_weight,
        model_config.mle_choice_norm)
    finite_valid = valid_mask & np.isfinite(loglik_values)
    # ``trial_weights is None`` → unweighted legacy sum (bit-exact).
    # Otherwise multiply the per-trial loglik by the sample-balancing
    # weight before summing; see ``_compute_trial_weights`` for the
    # formula. The aggregate scale is preserved.
    if data.trial_weights is None:
        total_loglik = float(loglik_values[finite_valid].sum())
    else:
        total_loglik = float(
            (loglik_values * data.trial_weights)[finite_valid].sum())
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
    BOUND is uniform across candidates here either because it's frozen
    upstream in ``fit.simulateDDM`` (legacy path) or because
    ``_compute_latent_population_equal_sessions`` applied the path-D
    rescaling so every candidate's effective bound is 1.0
    (``--scale-bound``). Either way, one solver call handles the whole
    generation.
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
    prepared = (df if isinstance(df, PreparedMLEData)
                else prepare_mle_data(df, model_config.mle_condition_columns))
    if n_candidates == 0:
        return np.empty(0, dtype=float)

    n_trials = prepared.n_trials
    penalty = _objective_penalty(prepared)
    losses = np.full(n_candidates, penalty, dtype=float)

    # Phase 1: compute latents for the whole population. Prepared data is
    # padded to equal session length, so we keep only a short Python loop over
    # trial position while updating all candidates and sessions at once.
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
    lapse_per_cand = pop_latents["lapse_rate"]

    # Phase 2: solver requires uniform BOUND across the chunk (its x_grid
    # is set up from bound). _compute_latent_population_equal_sessions
    # guarantees uniformity by design — see the rescale block there and
    # the docstring above — so we can just read the first candidate's
    # bound and pass it through.
    shared_bound = float(bounds[valid_cand_idx[0]])

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
    flat_lapse = backend.xp.repeat(
        backend.xp.asarray(lapse_per_cand[valid_cand_idx], dtype=float),
        n_trials)

    flat_z = z_stack[valid_cand_idx].reshape(-1)
    flat_sigma = sigma_stack[valid_cand_idx].reshape(-1)
    flat_valid = valid_stack[valid_cand_idx].reshape(-1)
    flat_no_choice = no_choice_stack[valid_cand_idx].reshape(-1)
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
        lapse_rate=flat_lapse,
    )

    # Phase 6: reshape (S_valid * N,) → (S_valid, N) and aggregate.
    per_cand_loglik = batch_result.loglik.reshape(n_valid, n_trials)
    # Choice-vs-RT reweighting across the whole population at once. Static
    # observations (choice_left, no_choice) broadcast over the candidate
    # axis; the per-candidate LAPSE_RATE is brought to host and column-
    # broadcast over trials. Identity (returns the same array) on the
    # marginal + (1,1) fast path, so the legacy sums below stay bit-exact.
    per_cand_loglik = _apply_choice_rt_weights(
        per_cand_loglik,
        batch_result.upper_hit_prob_tmax.reshape(n_valid, n_trials),
        batch_result.lower_hit_prob_tmax.reshape(n_valid, n_trials),
        prepared.choice_left[None, :],
        np.isnan(prepared.choice_left)[None, :],
        asnumpy(backend.xp, lapse_per_cand[valid_cand_idx])[:, None],
        model_config.mle_choice_weight,
        model_config.mle_rt_weight,
        model_config.mle_choice_norm)
    # ``trial_weights is None`` → unweighted legacy sum (bit-exact).
    # Otherwise broadcast the per-trial sample-balancing weight across
    # the candidate axis before summing. The weights are stable across
    # DE generations (data-only), so they're computed once in
    # ``prepare_mle_data`` and live on ``prepared``.
    per_cand_valid = valid_stack[valid_cand_idx]
    finite_valid = per_cand_valid & np.isfinite(per_cand_loglik)
    if prepared.trial_weights is None:
        total_loglik = np.where(finite_valid, per_cand_loglik, 0.0).sum(axis=1)
    else:
        weighted = per_cand_loglik * prepared.trial_weights[None, :]
        total_loglik = np.where(finite_valid, weighted, 0.0).sum(axis=1)
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
    data = prepare_mle_data(subject_df, model_config.mle_condition_columns)
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
        fit_finish_time=datetime.datetime.now().isoformat(timespec="seconds"),
    )


def make_objective(params_names, df, model_config):
    return partial(objective_from_vector, params_names=params_names,
                   df=df, model_config=model_config)


def prepare_mle_data(df, condition_columns=()):
    """Build the static MLE fixture once per fit.

    ``condition_columns`` (typically sourced from
    ``MLEModelConfig.mle_condition_columns``) — when non-empty,
    per-trial sample-balancing weights are precomputed and cached on
    the returned ``PreparedMLEData.trial_weights``. The weighted-sum
    branch at the two MLE loss sites consumes them; the default
    empty tuple keeps every existing caller bit-exact.
    """
    if isinstance(df, PreparedMLEData):
        # Pass-through is the common case (callers re-prepare a
        # subject's data downstream). Hard-fail when the caller asks
        # for conditions but the prepared instance was built without
        # them — re-preparing here would either silently drop the
        # conditions or quietly disagree with the cached arrays. The
        # right fix is to pass ``condition_columns`` at the original
        # ``prepare_mle_data(raw_df, ...)`` call site (see
        # ``fit._processSubject``). This assert exists because the
        # missing thread there was the actual reason
        # ``--mle-conditions`` looked like a no-op for a release.
        assert (not condition_columns) or (df.trial_weights is not None), (
            "prepare_mle_data was called with "
            f"condition_columns={tuple(condition_columns)!r} on a "
            "PreparedMLEData with trial_weights=None — the data was "
            "prepared without conditions, so the weighted sum sites "
            "would silently fall through to the unweighted path. "
            "Re-prepare from the raw df at the original call site.")
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
    valid_mask = sorted_df["valid"].to_numpy(dtype=bool)
    return PreparedMLEData(
        df=df,
        sorted_index=sorted_df.index.to_numpy(),
        session_slices=tuple(zip(starts, stops)),
        dv=_float_col(sorted_df, "DV"),
        valid=valid_mask,
        choice_left=_float_col(sorted_df, "ChoiceLeft"),
        reward=_float_col(sorted_df, "ChoiceCorrect"),
        observed_rt=_float_col(sorted_df, "calcStimulusTime"),
        trial_number=_float_col(sorted_df, "TrialNumber"),
        trial_weights=_compute_trial_weights(
            sorted_df, tuple(condition_columns), valid_mask),
    )


def _compute_trial_weights(sorted_df, condition_columns, valid_mask):
    """Sample-balanced per-trial weights for the MLE neg-loglik sum.

    Formula (mirrors ``logic.calcLoss``'s with-direction route but in a
    single weighted sum):

        group_id(i)   = (df[col_1][i], df[col_2][i], ...) for trial i
        num_groups    = #{ distinct group ids over valid trials }
        group_size[g] = #{ valid trials with group_id == g }
        weight[i]     = total_valid / (num_groups * group_size[gid(i)])

    Sum-preserving: ``sum_i weight[i] == total_valid`` always, so the
    aggregate loss stays on the same scale as the unweighted sum;
    only the within-condition contribution is rebalanced.

    Returns ``None`` for empty ``condition_columns`` (legacy fast
    path: the sum site detects ``None`` and skips the multiply). NaN
    values in any condition column form their own group via pandas
    ``groupby(..., dropna=False)`` — the right call for no-choice
    trials under ``(ChoiceCorrect, ChoiceLeft)``.
    """
    if not condition_columns:
        return None
    missing = [c for c in condition_columns if c not in sorted_df.columns]
    if missing:
        raise KeyError(
            f"--mle-conditions references unknown df columns: {missing}; "
            f"available: {sorted(sorted_df.columns)}")
    total_valid = int(valid_mask.sum())
    if total_valid == 0:
        # Empty fixture / nothing valid. Any reasonable default works
        # since ``finite_valid`` masks the whole sum to 0.0 anyway.
        return np.ones(len(sorted_df), dtype=float)
    valid_df = sorted_df.iloc[np.flatnonzero(valid_mask)]
    group_ids_valid = valid_df.groupby(
        list(condition_columns), dropna=False, sort=False
    ).ngroup().to_numpy()
    num_groups = int(group_ids_valid.max()) + 1 if group_ids_valid.size else 0
    group_size = np.bincount(group_ids_valid, minlength=num_groups)
    per_valid_weight = (
        float(total_valid) / (num_groups * group_size[group_ids_valid]))
    weights = np.zeros(len(sorted_df), dtype=float)
    weights[valid_mask] = per_valid_weight
    return weights


def _apply_choice_rt_weights(loglik, upper_hit_prob, lower_hit_prob,
                             choice_left, no_choice, lapse_rate,
                             w_choice, w_rt, choice_norm):
    """Split the joint per-trial loglik into choice + RT components and
    recombine with per-component weights.

    Implements (lapse-exact) the decomposition of chat Answer 29:

        (1) choice component   log P(c_i)
              marginal:     log P_mix(c_i)
              conditional:  log P_mix(c_i) - log P_mix_total
        (2) RT component       log p(rt_i | c_i) = loglik_i - log P_mix(c_i)
        (3) weighted loss      w_choice * (1) + w_rt * (2)

    with the lapse-consistent masses

        P_mix(c)    = (1 - lapse) * P_DDM(c) + lapse / 2
        P_mix_total = (1 - lapse) * (P_L + P_R) + lapse

    where ``P_DDM(c)`` is the chosen side's pure-DDM bound-hit prob
    (``upper_hit_prob`` for a left/upper choice, else ``lower_hit_prob``)
    and ``loglik_i`` is the already-lapse-mixed joint log-density.

    Shapes broadcast over an optional leading candidate axis: the single
    path passes ``(n_trials,)`` arrays with scalar ``lapse_rate``; the
    population path passes ``(n_candidates, n_trials)`` hit-prob/loglik
    arrays with ``choice_left``/``no_choice`` as ``(1, n_trials)`` and
    ``lapse_rate`` as ``(n_candidates, 1)``. Pure host-numpy — the GPU
    work already happened in the solver.

    No-choice trials (NaN ``choice_left``) get the choice-only form
    ``w_choice * loglik_i`` (no RT term). In production they're already
    ``valid=False`` and excluded at the sum site; this branch is a
    defensive fallback. Returns ``loglik`` unchanged on the
    ``marginal`` + ``(1, 1)`` fast path (bit-exact legacy).
    """
    if choice_norm == "marginal" and w_choice == 1.0 and w_rt == 1.0:
        return loglik
    is_left = choice_left == 1
    p_ddm_chosen = np.where(is_left, upper_hit_prob, lower_hit_prob)
    p_mix_chosen = (1.0 - lapse_rate) * p_ddm_chosen + lapse_rate / 2.0
    log_choice = np.log(np.clip(p_mix_chosen, LOGLIK_FLOOR, None))
    log_rt = loglik - log_choice
    weighted = w_choice * log_choice + w_rt * log_rt
    if choice_norm == "conditional":
        p_mix_total = (
            (1.0 - lapse_rate) * (upper_hit_prob + lower_hit_prob) + lapse_rate)
        log_norm = np.log(np.clip(p_mix_total, LOGLIK_FLOOR, None))
        weighted = weighted - w_choice * log_norm
    # No-choice: choice-only weighting (no RT density to condition on).
    no_choice_arr = np.broadcast_to(no_choice, weighted.shape)
    weighted = np.where(no_choice_arr, w_choice * loglik, weighted)
    return weighted


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


def _evict_prepared_session_cache(data_id):
    """GC hook: drop cached session arrays for a collected ``PreparedMLEData``.

    ``id()`` is unique only among *live* objects — CPython recycles an id once
    the object it named is freed. Keying the cache by ``id(data)`` alone let a
    new PreparedMLEData allocated at a recycled address read a *previous*
    object's session arrays: a wrong-shape cache hit that surfaced as an
    order-dependent reshape crash in ``objective_from_population`` (it passed in
    isolation but failed after another test's data had been allocated and freed
    at the same address). Registered on ``data`` via ``weakref.finalize``, this
    evicts its entries the moment it is collected, so a recycled id always
    misses and recomputes.
    """
    for k in [k for k in _PREPARED_SESSION_BACKEND_CACHE if k[0] == data_id]:
        _PREPARED_SESSION_BACKEND_CACHE.pop(k, None)


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
    # Evict this data's entries on GC so a recycled id() can't alias another
    # object's arrays. Register the finalizer once per data object (the first
    # backend we cache for it); id(data) is stable while data is alive.
    if not any(k[0] == key[0] for k in _PREPARED_SESSION_BACKEND_CACHE):
        weakref.finalize(data, _evict_prepared_session_cache, key[0])
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
    # BOUND / NON_DECISION_TIME defaults documented at the call site —
    # 1.0 / 0.0 are identity / no-shift settings, not silent fallbacks.
    bounds = _param_population(theta, param_lookup, "BOUND", xp, 1.0)
    nondec = _param_population(theta, param_lookup, "NON_DECISION_TIME", xp, 0.0)
    # Flag-gated strict access: when include_Q / include_RewardRate is
    # True, the param MUST exist; KeyError on miss. Pre-shape with
    # `[:, None]` so the population (n_candidates, n_sessions) broadcast
    # happens naturally inside update_q_values / update_reward_rate.
    def _gated(name, gate):
        return theta[param_lookup[name]][:, None] if gate else None
    alpha = _gated("ALPHA", model_config.include_Q)
    beta = _gated("BETA", model_config.include_RewardRate)
    bias_coef = _param_population(theta, param_lookup, "BIAS_COEF", xp, 0.0)
    q_offset = _param_population(theta, param_lookup, "Q_VAL_OFFSET", xp, 0.0)
    # LAPSE_RATE: 0.0 reproduces the no-mixture likelihood exactly.
    lapse_rate = _param_population(
        theta, param_lookup, "LAPSE_RATE", xp, 0.0)

    q_left = xp.full((n_candidates, n_sessions), 0.5, dtype=float)
    q_right = xp.full((n_candidates, n_sessions), 0.5, dtype=float)
    reward_rate = xp.full((n_candidates, n_sessions), 0.5, dtype=float)
    z = xp.empty((n_candidates, n_sessions, trials_per_session), dtype=float)
    sigma = xp.empty_like(z)
    mu = xp.empty_like(z)

    for trial_pos in range(trials_per_session):
        # Delegate Q-value normalization + starting-point bias to
        # state_updates (same math as the scalar path).
        q_rel = state_updates.compute_q_value(q_left, q_right, xp=xp)
        # Bound-RewardRate vectorized path: apply the Phase-1-validated
        # rescaling (mu/s_t, sigma/s_t, z/s_t with s_t = 2 - r_t and z
        # clipped to ±b_t = ±BOUND*(2-r_t) in absolute units first).
        # reward_rate here is the per-(candidate, session) state BEFORE
        # the current trial — the same value the NoiseGain branch uses,
        # just consumed differently. The solver still receives the scalar
        # BOUND per candidate; no solver work.
        base_mu = drift_coef[:, None] * dv[None, :, trial_pos]
        if model_config.uses_per_trial_bound:
            bound_scale = state_updates.bound_scale_from_reward_rate(reward_rate)
            b_t = bounds[:, None] * bound_scale
            if model_config.uses_q_bias:
                z_abs = xp.clip(
                    bias_coef[:, None] * q_rel + q_offset[:, None],
                    -b_t, b_t)
                z_t = z_abs / bound_scale
            else:
                z_t = xp.zeros_like(q_rel)
            # σ /= s_t  (opposite of NoiseGain's σ *= r_t)
            sigma_t = noise_sigma[:, None] / bound_scale
            base_mu = base_mu / bound_scale       # μ /= s_t
        else:
            if model_config.uses_q_bias:
                z_t = state_updates.compute_starting_point_z(
                    q_left, q_right,
                    delta=bias_coef[:, None],
                    offset=q_offset[:, None],
                    include_Q=True, xp=xp)
            else:
                z_t = xp.zeros_like(q_rel)
            if model_config.uses_per_trial_drift:
                # DriftGain-RewardRate: mu *= g(r_t) with sigma FLAT and the
                # bound untouched. Not a rescaling identity (unlike the
                # Bound- branch above), so z keeps the standard
                # fraction-of-bound semantic and needs no adjustment.
                drift_scale = state_updates.drift_scale_from_reward_rate(
                    reward_rate, model_config.rr_drift_map)
                sigma_t = xp.broadcast_to(noise_sigma[:, None], q_rel.shape)
                base_mu = base_mu * drift_scale
            else:
                sigma_t = (
                    reward_rate * noise_sigma[:, None]
                    if model_config.include_RewardRate
                    else xp.broadcast_to(noise_sigma[:, None], q_rel.shape)
                )
        z[:, :, trial_pos] = z_t
        sigma[:, :, trial_pos] = sigma_t
        mu[:, :, trial_pos] = base_mu

        valid_t = valid_for_loss_2d[None, :, trial_pos]
        reward_t = xp.nan_to_num(reward_arr[None, :, trial_pos], nan=0.0)
        choice_t = choice[None, :, trial_pos]
        # Delegate the updates to state_updates so the population path uses
        # the same math as the scalar path. The gated alpha / beta above are
        # already pre-shaped ``[:, None]`` so update_q_values' broadcasts hit
        # (n_candidates, n_sessions) cleanly.
        if model_config.include_Q:
            new_q_left, new_q_right = state_updates.update_q_values(
                q_left, q_right, choice_t, reward_t, alpha, xp=xp)
            # update_q_values doesn't know about per-trial validity;
            # mask invalid positions here so they keep their previous
            # Q-state (matches the scalar path which guards with
            # ``if valid_for_loss[i]``).
            q_left = xp.where(valid_t, new_q_left, q_left)
            q_right = xp.where(valid_t, new_q_right, q_right)
        if model_config.include_RewardRate:
            new_rr = state_updates.update_reward_rate(
                reward_rate, reward_t, beta, xp=xp)
            reward_rate = xp.where(valid_t, new_rr, reward_rate)

    # --scale-bound: apply the path-D rescaling identity per candidate
    # (mu/B, sigma/B, z/B) so every candidate's effective bound becomes
    # 1.0. Same trick the Bound-RewardRate batched path uses, generalized
    # to per-CANDIDATE BOUND — the scale_bound_equivalence.ipynb identity
    # makes no distinction between trial-varying and candidate-varying B.
    # Without this, candidate-major BOUND heterogeneity would break the
    # solver's single-x_grid assumption and force a per-candidate Python
    # loop in objective_from_population.
    inv_b_3d = None
    if model_config.uses_scaled_bound:
        inv_b_3d = (1.0 / bounds)[:, None, None]
        z = z * inv_b_3d
        sigma = sigma * inv_b_3d
        mu = mu * inv_b_3d
        bounds = xp.ones_like(bounds)

    flat_shape = (n_candidates, n_trials)
    valid_flat = xp.broadcast_to(
        valid_for_loss_2d[None, :, :],
        (n_candidates, n_sessions, trials_per_session),
    ).reshape(flat_shape)
    no_choice_flat = xp.broadcast_to(
        no_choice_2d[None, :, :],
        (n_candidates, n_sessions, trials_per_session),
    ).reshape(flat_shape)

    return {
        "z": asnumpy(xp, z.reshape(flat_shape)),
        "sigma": asnumpy(xp, sigma.reshape(flat_shape)),
        "mu": asnumpy(xp, mu.reshape(flat_shape)),
        "valid_for_loss": asnumpy(xp, valid_flat).astype(bool),
        "no_choice": asnumpy(xp, no_choice_flat).astype(bool),
        "bounds": asnumpy(xp, bounds),
        "non_decision_time": asnumpy(xp, nondec),
        "lapse_rate": asnumpy(xp, lapse_rate),
    }


def _compute_latent_arrays(data, params, model_config):
    """Scalar per-trial Q / reward-rate / z / sigma / mu recompute.

    Thin orchestration around ``state_updates`` — the actual math (Q
    update, reward-rate update, no-choice handling, Q-value normalization)
    lives there. This function is the scalar single-subject path; the
    (n_candidates × n_sessions) vectorized counterpart is
    ``_compute_latent_population_equal_sessions``, which calls the same
    ``state_updates`` functions with ``xp=backend.xp``.

    Strict access on flag-gated params: if ``model_config.include_Q``
    is True, ``params["ALPHA"]`` must exist; same for BETA. Loud KeyError
    on miss is the contract per the consolidation plan — silently
    inventing a fallback rate is exactly what we want to prevent.
    """
    n = data.n_trials
    q_left_before = np.full(n, 0.5, dtype=float)
    q_right_before = np.full(n, 0.5, dtype=float)
    q_rel_before = np.full(n, 0.0, dtype=float)
    reward_rate_before = np.full(n, 0.5, dtype=float)
    q_left_after = np.full(n, 0.5, dtype=float)
    q_right_after = np.full(n, 0.5, dtype=float)
    reward_rate_after = np.full(n, 0.5, dtype=float)

    # Strict access when the flag says the param MUST exist. When the
    # flag is False the rate is unused.
    alpha = params["ALPHA"] if model_config.include_Q else None
    beta = params["BETA"] if model_config.include_RewardRate else None

    valid_for_loss = data.valid & np.isfinite(data.dv)
    for start, stop in data.session_slices:
        q_left, q_right, reward_rate = 0.5, 0.5, 0.5
        for i in range(start, stop):
            q_left_before[i] = q_left
            q_right_before[i] = q_right
            q_rel_before[i] = float(
                state_updates.compute_q_value(q_left, q_right))
            reward_rate_before[i] = reward_rate
            if valid_for_loss[i]:
                reward = 0.0 if np.isnan(data.reward[i]) else float(data.reward[i])
                choice_left = data.choice_left[i]
                if model_config.include_Q:
                    q_left, q_right = state_updates.update_q_values(
                        q_left, q_right, choice_left, reward, alpha)
                    # update_q_values returns 0-D arrays; downcast so
                    # subsequent loop iterations stay scalar.
                    q_left = float(q_left)
                    q_right = float(q_right)
                if model_config.include_RewardRate:
                    reward_rate = float(state_updates.update_reward_rate(
                        reward_rate, reward, beta))
            q_left_after[i] = q_left
            q_right_after[i] = q_right
            reward_rate_after[i] = reward_rate

    z = _compute_z_array(q_left_before, q_right_before, q_rel_before,
                         params, model_config)
    sigma = _compute_sigma_array(reward_rate_before, params, model_config)
    if model_config.uses_per_trial_bound or model_config.uses_per_trial_drift:
        # Non-noise reward-rate channels: sigma is CONSTANT per trial, so
        # override _compute_sigma_array's NoiseGain branch (which would scale
        # sigma by reward_rate).
        #   Bound-RewardRate: the per-trial scaling lives in bound_per_trial
        #     below, giving the path-A latents from
        #     scale_bound_equivalence.ipynb.
        #   DriftGain-RewardRate: it lives in drift_scale (mu) below.
        sigma = np.full_like(
            reward_rate_before, float(_param(params, "NOISE_SIGMA")),
            dtype=float)
    # DriftGain-RewardRate: mu = DRIFT_COEF * DV * g(r_t), applied inside
    # _compute_mu_array.
    drift_scale = drift_scale_for_config(reward_rate_before, model_config)
    mu = _compute_mu_array(data.dv, params, drift_scale=drift_scale)
    latents = {
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
        # LAPSE_RATE remains a documented optional with a meaningful
        # zero default (0.0 reproduces the pre-mixture likelihood).
        "lapse_rate": float(_param(params, "LAPSE_RATE", 0.0)),
    }
    # Bound-RewardRate: surface per-trial bound for the rowwise reference
    # path (the equivalence ground-truth proved in Phase 1). The vectorized
    # path uses the rescaled mu/sigma/z instead; see
    # _compute_latent_population_equal_sessions.
    if model_config.uses_per_trial_bound:
        bound_base = float(_param(params, "BOUND", 1.0))
        bound_per_trial = bound_base * state_updates.bound_scale_from_reward_rate(
            reward_rate_before)
        latents["bound_per_trial"] = bound_per_trial
        # Bias is in absolute DDM-state units: re-clip z to ±b_t. The
        # default _compute_z_array clip is [-1, 1] which collapses the
        # bound-dependent envelope; redo it here so the rowwise reference
        # sees the same z the math says it should.
        if model_config.uses_q_bias:
            z_raw = (_param(params, "BIAS_COEF") * q_rel_before
                     + _param(params, "Q_VAL_OFFSET", 0.0))
            latents["z"] = np.clip(z_raw, -bound_per_trial, bound_per_trial)
    return latents


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


def _compute_mu_array(dv, params, drift_scale=None):
    """Per-trial drift. ``drift_scale`` (DriftGain-RewardRate's per-trial
    ``g(r_t)``, or None) multiplies it.
    """
    base_mu = _param(params, "DRIFT_COEF") * np.asarray(dv, dtype=float)
    if drift_scale is not None:
        base_mu = base_mu * np.asarray(drift_scale, dtype=float)
    return base_mu


def estimate_population_settings(model_config, n_trials, n_params, bound=1.0):
    """Return candidate count and SciPy popsize from the memory ceiling.

    Enforces a floor of ``model_config.mle_min_population_candidates``
    actual candidates so DE has enough diversity to explore the
    parameter space even when the memory budget would otherwise pick a
    smaller population. The floor defaults to
    ``MIN_POPULATION_CANDIDATES`` (64); the ``--mle-min-population`` CLI
    flag overrides per-fit.
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
    # `actual_candidates >= mle_min_population_candidates`.
    min_floor = int(model_config.mle_min_population_candidates)
    min_popsize = -(-min_floor // n_params)  # ceil division
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
    # Optional per-trial bound override. When ``latents`` carries a
    # ``bound_per_trial`` array (1-D, length data.n_trials), use it
    # per-iteration instead of the scalar ``bound`` from params. This
    # is the entry point for the varying-bound reference path used by
    # the scale_bound_equivalence verification notebook — production
    # MLE callers never set this, so the scalar path stays bit-exact.
    bound_per_trial = latents.get("bound_per_trial")
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
            bound=(float(bound_per_trial[i])
                   if bound_per_trial is not None else bound),
            non_decision_time=non_decision_time,
            dt=model_config.dt,
            dx=model_config.dx,
            tmax=model_config.t_dur,
            diffusion_backend=model_config.diffusion_backend,
            no_choice=latents["no_choice"][i],
            lapse_rate=latents["lapse_rate"],
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
    # Bound-RewardRate fast path: apply the path-D rescaling at the
    # batched-solver boundary. ``_compute_latent_arrays`` produces
    # path-A-shaped latents (absolute mu, sigma, z; per-trial bound)
    # for the rowwise reference; the batched solver expects a single
    # scalar bound across the batch, so we rescale per-trial
    # (mu, sigma, z) by BOUND/bound_per_trial = 1/(2 - r_t). The rescaling
    # identity (proven in Phase 1) guarantees this gives identical
    # observable densities. When ``uses_per_trial_bound=False`` this
    # branch is a no-op and the production path is bit-exact.
    bound_scalar = _param(params, "BOUND", 1.0)
    mu_for_solver    = latents["mu"]
    sigma_for_solver = latents["sigma"]
    z_for_solver     = latents["z"]
    if model_config.uses_per_trial_bound:
        bpt = latents.get("bound_per_trial")
        assert bpt is not None, (
            "uses_per_trial_bound=True but _compute_latent_arrays did not "
            "produce bound_per_trial — check that flag wiring")
        rescale = bound_scalar / np.asarray(bpt, dtype=float)
        mu_for_solver = np.asarray(mu_for_solver, dtype=float) * rescale
        sigma_for_solver = np.asarray(sigma_for_solver, dtype=float) * rescale
        z_for_solver = np.asarray(z_for_solver, dtype=float) * rescale
    batch_result = batched_choice_rt_loglik(
        observed_choice_left=data.choice_left,
        observed_rt=data.observed_rt,
        no_choice=latents["no_choice"],
        valid_for_loss=latents["valid_for_loss"],
        z=z_for_solver,
        mu_values=mu_for_solver,
        sigma=sigma_for_solver,
        bound=bound_scalar,
        non_decision_time=np.full(n, _param(params, "NON_DECISION_TIME", 0.0)),
        dt=model_config.dt,
        dx=model_config.dx,
        tmax=model_config.t_dur,
        xp=backend.xp,
        normal_cdf=backend.normal_cdf,
        solver=solver,
        terminal_c=float(model_config.mle_terminal_c),
        lapse_rate=float(latents["lapse_rate"]),
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
    n_rows = len(data.sorted_index)
    # Choice / RT decomposition components (unweighted, norm-agnostic) so the
    # user can audit the split per trial and reconstruct either choice-norm.
    # See ``_apply_choice_rt_weights``. NaN on invalid/no-choice trials (their
    # hit-probs are NaN) — diagnostic only.
    _lapse = float(latents.get("lapse_rate", 0.0))
    _p_ddm_chosen = np.where(
        data.choice_left == 1,
        like_result.upper_hit_prob_tmax,
        like_result.lower_hit_prob_tmax)
    _p_mix_chosen = (1.0 - _lapse) * _p_ddm_chosen + _lapse / 2.0
    _log_choice = np.log(np.clip(_p_mix_chosen, LOGLIK_FLOOR, None))
    _p_mix_total = (
        (1.0 - _lapse)
        * (like_result.upper_hit_prob_tmax + like_result.lower_hit_prob_tmax)
        + _lapse)
    _log_choice_norm = np.log(np.clip(_p_mix_total, LOGLIK_FLOOR, None))
    latent_df = pd.DataFrame({
        "_row_index": data.sorted_index,
        "mle_Q_left_before": latents["q_left_before"],
        "mle_Q_right_before": latents["q_right_before"],
        "mle_Q_rel_before": latents["q_rel_before"],
        "mle_reward_rate_before": latents["reward_rate_before"],
        "mle_z": latents["z"],
        "mle_mu": mu_display,
        "mle_sigma": latents["sigma"],
        "mle_lapse_rate": np.full(
            n_rows, float(latents.get("lapse_rate", 0.0)), dtype=float),
        "mle_decision_time_observed": like_result.decision_time,
        "mle_choice_prob_or_density": like_result.choice_prob_or_density,
        "mle_loglik": like_result.loglik,
        # Choice / RT split components (lapse-exact, unweighted):
        # mle_log_choice = log P_mix(c); mle_rt_loglik = loglik - log P_mix(c);
        # mle_log_choice_normalizer = log(P_mix(L)+P_mix(R)) for the
        # conditional norm. log P(c)_conditional = mle_log_choice - normalizer.
        "mle_log_choice": _log_choice,
        "mle_rt_loglik": like_result.loglik - _log_choice,
        "mle_log_choice_normalizer": _log_choice_norm,
        "mle_valid_for_loss": latents["valid_for_loss"],
        "mle_survival_at_tmax": like_result.survival_at_tmax,
        "mle_upper_hit_prob_tmax": like_result.upper_hit_prob_tmax,
        "mle_lower_hit_prob_tmax": like_result.lower_hit_prob_tmax,
        "mle_Q_left_after": latents["q_left_after"],
        "mle_Q_right_after": latents["q_right_after"],
        "mle_reward_rate_after": latents["reward_rate_after"],
    }).set_index("_row_index")
    return data.df.copy().join(latent_df, how="left")


# Per-trial scalar helpers used by external callers
# (``posterior_simulate``, ``mle_visualize``, ``ddm_viewer``) to
# reconstruct mu / z for a single trial from already-fitted params.
# Not used by the MLE objective itself (the population-shaped versions
# ``_compute_z_array`` / ``_compute_mu_array`` cover that path).

def drift_scale_for_config(reward_rate, model_config):
    """Per-trial drift gain ``g(r)`` for ``model_config``, or None.

    None means "the reward rate does not modulate the drift" — the value
    ``_compute_mu`` / ``_compute_mu_array`` want in order to leave mu
    alone. Shared by the array path and the three scalar consumers so the
    DriftGain mapping can't be applied in one and forgotten in another.
    """
    if not model_config.uses_per_trial_drift:
        return None
    return state_updates.drift_scale_from_reward_rate(
        reward_rate, model_config.rr_drift_map)


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


def _compute_mu(coherence, params, drift_scale=None):
    """Scalar counterpart of ``_compute_mu_array``. ``drift_scale``
    (from ``drift_scale_for_config``) multiplies it.
    """
    base_mu = _param(params, "DRIFT_COEF") * coherence
    if drift_scale is not None:
        base_mu = base_mu * float(drift_scale)
    return base_mu

