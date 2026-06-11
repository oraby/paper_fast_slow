from dataclasses import dataclass

import numpy as np


LOG_CIEL = 0.01
LOG_CEIL_MAX = np.log(1 / LOG_CIEL)


@dataclass(frozen=True)
class LatentState:
    include_Q: bool
    include_RewardRate: bool
    q_left: float = 0.5
    q_right: float = 0.5
    reward_rate: float = 0.5


def initialize_latent_state(include_Q: bool, include_RewardRate: bool) -> LatentState:
    return LatentState(
        include_Q=include_Q,
        include_RewardRate=include_RewardRate,
    )


def compute_q_value(q_left, q_right, group_every=0):
    """Return normalized log Q-ratio in [-1, 1]."""
    q_left = np.clip(np.asarray(q_left), LOG_CIEL, 1)
    q_right = np.clip(np.asarray(q_right), LOG_CIEL, 1)
    q_value = np.log(q_left / q_right) / LOG_CEIL_MAX
    if group_every != 0:
        q_value = np.round(q_value / group_every) * group_every
    return q_value


def update_q_values(q_left, q_right, observed_choice_left, observed_reward, alpha,
                    alpha_unrewarded=None):
    """Update only the chosen side; no-choice trials leave both Q values unchanged.

    ``alpha_unrewarded`` defaults to ``alpha`` (symmetric) when omitted, when
    ``None`` is passed, or when ``NaN`` is passed. The NaN-fallback matches
    the sentinel Chisqr uses for frozen learning-rate params (see
    ``fit.py::_makeOneRunWrapper``), so callers that don't fit the
    asymmetric rate get the legacy single-alpha behavior automatically.
    """
    if alpha_unrewarded is None or (
            isinstance(alpha_unrewarded, float) and np.isnan(alpha_unrewarded)):
        alpha_unrewarded = alpha
    if observed_reward is None:
        observed_reward = 0
    else:
        observed_reward = np.nan_to_num(observed_reward, nan=0)
    if observed_choice_left is None:
        observed_choice_left = np.nan
    choice_left = np.asarray(observed_choice_left, dtype=float)
    no_choice = np.isnan(choice_left)
    learning_rate = np.where(
        no_choice | (observed_reward == 0),
        alpha_unrewarded,
        alpha,
    )
    new_q_left = np.where(
        no_choice | (choice_left == 0),
        q_left,
        # Previous trial was a left choice:
        q_left + learning_rate * (observed_reward - q_left)
    )
    new_q_right = np.where(
        no_choice | (choice_left == 1),
        q_right,
        # Previous trial was a right choice:
        q_right + learning_rate * (observed_reward - q_right)
    )
    return new_q_left, new_q_right


def update_reward_rate(reward_rate, observed_reward, beta, beta_unrewarded=None,
                       group_every=0):
    """``beta_unrewarded`` defaults to ``beta`` when omitted, ``None``, or NaN.

    NaN matches the Chisqr frozen-param sentinel; see ``update_q_values``.
    """
    if beta_unrewarded is None or (
            isinstance(beta_unrewarded, float) and np.isnan(beta_unrewarded)):
        beta_unrewarded = beta
    if observed_reward is None:
        observed_reward = 0
    else:
        observed_reward = np.nan_to_num(observed_reward, nan=0)
    learning_rate = np.where(observed_reward == 0, beta_unrewarded, beta)
    new_reward_rate = reward_rate + \
                      learning_rate * (observed_reward - reward_rate)
    if group_every != 0:
        new_reward_rate = np.round(new_reward_rate / group_every) * group_every
    return new_reward_rate


def compute_starting_point_z(q_left, q_right, delta, offset, include_Q):
    """Return normalized starting point z in current bound units, [-1, 1].

    If include_Q is False, returns 0 unconditionally. Classic and R-only MLE
    variants have no starting-point bias; offset is only meaningful when
    include_Q is True.
    """
    if not include_Q:
        return 0.0
    q_value = compute_q_value(q_left, q_right)
    return np.clip(delta * q_value + offset, -1, 1)


def compute_trial_mu(coherence, drift_coef, time_grid=None):
    mu = drift_coef * coherence
    if time_grid is None:
        return mu
    return np.full_like(time_grid, mu, dtype=float)


def compute_trial_sigma(base_sigma, reward_rate, include_RewardRate):
    if include_RewardRate:
        return reward_rate * base_sigma
    return base_sigma
