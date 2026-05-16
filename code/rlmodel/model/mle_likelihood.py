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


def trial_choice_rt_loglik(observed_choice_left, observed_rt, z, mu, sigma,
                           bound, non_decision_time, dt, dx, tmax, *,
                           diffusion_backend="auto", no_choice=False):
    """Return the choice+RT log likelihood for one observed trial.

    ``observed_choice_left=1`` maps to the upper absorbing bound, matching the
    existing simulation code. No-choice trials contribute the survival mass at
    ``tmax``.
    """
    result = first_passage_density(
        z, mu, sigma, bound, dt, dx, tmax, backend=diffusion_backend)
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
            return TrialLikelihood(
                loglik=float(np.log(LOGLIK_FLOOR)),
                choice_prob_or_density=LOGLIK_FLOOR,
                decision_time=np.nan,
                survival_at_tmax=survival,
                upper_hit_prob_tmax=upper_hit_prob,
                lower_hit_prob_tmax=lower_hit_prob,
            )

        decision_time = observed_rt - float(non_decision_time)
        if decision_time <= 0 or decision_time > tmax:
            return TrialLikelihood(
                loglik=float(np.log(LOGLIK_FLOOR)),
                choice_prob_or_density=LOGLIK_FLOOR,
                decision_time=decision_time,
                survival_at_tmax=survival,
                upper_hit_prob_tmax=upper_hit_prob,
                lower_hit_prob_tmax=lower_hit_prob,
            )

        idx = int(np.ceil(decision_time / dt)) - 1
        idx = min(max(idx, 0), len(result.times) - 1)
        if int(observed_choice_left) == 1:
            likelihood = float(np.asarray(result.f_upper)[idx])
        else:
            likelihood = float(np.asarray(result.f_lower)[idx])

    likelihood = max(float(likelihood), LOGLIK_FLOOR)
    return TrialLikelihood(
        loglik=float(np.log(likelihood)),
        choice_prob_or_density=likelihood,
        decision_time=decision_time,
        survival_at_tmax=survival,
        upper_hit_prob_tmax=upper_hit_prob,
        lower_hit_prob_tmax=lower_hit_prob,
    )
