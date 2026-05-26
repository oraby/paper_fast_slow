from dataclasses import dataclass

import numpy as np

from .first_passage import first_passage_density


LOGLIK_FLOOR = 1e-300


@dataclass
class TrialLikelihood:
    loglik: float
    choice_prob_or_density: float
    decision_time: float
    survival_at_tmax: float
    upper_hit_prob_tmax: float
    lower_hit_prob_tmax: float


def _hit_probabilities(result, dt):
    upper = float(np.asarray(result.f_upper).sum()) * dt
    lower = float(np.asarray(result.f_lower).sum()) * dt
    survival = float(np.asarray(result.survival)[-1])
    return upper, lower, survival


def _floor_likelihood(decision_time=np.nan):
    floor_loglik = float(np.log(LOGLIK_FLOOR))
    return TrialLikelihood(
        loglik=floor_loglik,
        choice_prob_or_density=LOGLIK_FLOOR,
        decision_time=decision_time,
        survival_at_tmax=LOGLIK_FLOOR,
        upper_hit_prob_tmax=np.nan,
        lower_hit_prob_tmax=np.nan,
    )


def _is_valid_solver_input(z, mu, sigma, bound, dt, dx, tmax):
    scalar_values = [z, sigma, bound, dt, dx, tmax]
    if not all(np.isfinite(float(v)) for v in scalar_values):
        return False
    if sigma <= 0 or bound <= 0 or dt <= 0 or dx <= 0 or tmax <= 0:
        return False
    if z < -bound or z > bound:
        return False
    return np.all(np.isfinite(np.asarray(mu, dtype=float)))


def trial_choice_rt_loglik(observed_choice_left, observed_rt, z, mu, sigma,
                           bound, non_decision_time, dt, dx, tmax, *,
                           diffusion_backend="auto", no_choice=False):
    """Return the choice+RT log likelihood for one observed trial.

    ``observed_choice_left=1`` maps to the upper absorbing bound, matching the
    existing simulation code. No-choice trials contribute the full survival
    mass at ``tmax``.

    NOTE: this rowwise path does **not** honor ``mle_terminal_c`` — that
    feature is only implemented for the batched path
    (``BatchedDiffusionSolver`` + ``batched_choice_rt_loglik``). The rowwise
    path is retained as a fallback/reference and is gated by
    ``MLEModelConfig.mle_use_batched_likelihood``. See
    ``mle_terminal_c_plan.md`` (Option B, deferred).
    """
    if not _is_valid_solver_input(z, mu, sigma, bound, dt, dx, tmax):
        return _floor_likelihood()

    try:
        result = first_passage_density(
            z, mu, sigma, bound, dt, dx, tmax, backend=diffusion_backend)
    except (AssertionError, FloatingPointError, ValueError, OverflowError):
        return _floor_likelihood()

    upper_hit_prob, lower_hit_prob, survival = _hit_probabilities(result, dt)

    if no_choice:
        likelihood = survival
        decision_time = np.nan
    else:
        try:
            observed_rt = float(observed_rt)
        except (TypeError, ValueError):
            observed_rt = np.nan
        if np.isnan(observed_rt):
            return _floor_likelihood(decision_time=np.nan)

        decision_time = observed_rt - float(non_decision_time)
        if decision_time <= 0 or decision_time > tmax:
            return _floor_likelihood(decision_time=decision_time)

        idx = int(np.ceil(decision_time / dt)) - 1
        idx = min(max(idx, 0), len(result.times) - 1)
        if int(observed_choice_left) == 1:
            likelihood = float(np.asarray(result.f_upper)[idx])
        else:
            likelihood = float(np.asarray(result.f_lower)[idx])

    likelihood = float(likelihood)
    if not np.isfinite(likelihood) or likelihood <= 0:
        likelihood = LOGLIK_FLOOR
    return TrialLikelihood(
        loglik=float(np.log(likelihood)),
        choice_prob_or_density=likelihood,
        decision_time=decision_time,
        survival_at_tmax=survival,
        upper_hit_prob_tmax=upper_hit_prob,
        lower_hit_prob_tmax=lower_hit_prob,
    )
