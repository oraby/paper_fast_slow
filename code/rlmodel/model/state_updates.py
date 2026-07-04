"""Per-trial state-update math.

Single source of truth for Q-value learning, reward-rate learning,
log-ratio Q normalization, and starting-point bias. Used by:

- ``logic.py`` (Chisqr forward simulator) via the scalar/NumPy path,
- ``mle.py`` per-trial recompute via the scalar/NumPy path,
- ``mle.py`` population-shaped recompute via the ``xp=backend.xp``
  path (NumPy for CPU, CuPy for GPU).

Each function takes an ``xp`` kwarg defaulted to ``np``. Pass
``xp=cupy`` (or any drop-in module) to dispatch all underlying ops on
that backend; broadcasting follows NumPy semantics so the same call
sites handle scalar, 1-D, and (n_candidates, n_sessions) inputs.

Asymmetric-rate contract (see ``update_q_values`` / ``update_reward_rate``):
``alpha_unrewarded=None`` (or omitted) means "use the symmetric
``alpha``". Sentinels like ``np.nan`` are NOT accepted — callers in the
Chisqr path now pass ``None`` for frozen-out params (see
``fit.py:_makeOneRunWrapper``).
"""
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


def compute_q_value(q_left, q_right, group_every=0, *, xp=np):
    """Return normalized log Q-ratio in [-1, 1]."""
    q_left = xp.clip(xp.asarray(q_left), LOG_CIEL, 1)
    q_right = xp.clip(xp.asarray(q_right), LOG_CIEL, 1)
    q_value = xp.log(q_left / q_right) / LOG_CEIL_MAX
    if group_every != 0:
        q_value = xp.round(q_value / group_every) * group_every
    return q_value


def update_q_values(q_left, q_right, observed_choice_left, observed_reward, alpha,
                    alpha_unrewarded=None, *, xp=np):
    """Update only the chosen side; no-choice trials leave both Q values unchanged.

    ``alpha_unrewarded=None`` (or omitted) means "use ``alpha``" — the
    symmetric/legacy single-rate behavior. Any other value enables the
    asymmetric branch: when reward == 0 or there's no choice, the
    unrewarded rate is used; otherwise the rewarded ``alpha`` fires.

    No NaN sentinel handling: the Chisqr frozen-param path now passes
    ``None`` (not ``np.nan``) so this function has exactly one fallback
    rule. Pass ``xp=cupy`` for the GPU population path; broadcasting
    follows NumPy semantics, so ``alpha[:, None]`` against
    ``(n_candidates, n_sessions)`` Q-state works the same way.
    """
    if alpha_unrewarded is None:
        alpha_unrewarded = alpha
    if observed_reward is None:
        observed_reward = 0
    else:
        observed_reward = xp.nan_to_num(observed_reward, nan=0)
    if observed_choice_left is None:
        observed_choice_left = xp.nan
    choice_left = xp.asarray(observed_choice_left, dtype=float)
    no_choice = xp.isnan(choice_left)
    learning_rate = xp.where(
        no_choice | (observed_reward == 0),
        alpha_unrewarded,
        alpha,
    )
    new_q_left = xp.where(
        no_choice | (choice_left == 0),
        q_left,
        # Previous trial was a left choice:
        q_left + learning_rate * (observed_reward - q_left)
    )
    new_q_right = xp.where(
        no_choice | (choice_left == 1),
        q_right,
        # Previous trial was a right choice:
        q_right + learning_rate * (observed_reward - q_right)
    )
    return new_q_left, new_q_right


def update_reward_rate(reward_rate, observed_reward, beta, beta_unrewarded=None,
                       group_every=0, *, xp=np):
    """``beta_unrewarded=None`` (or omitted) means "use ``beta``".

    Mirrors the ``update_q_values`` contract. Pass ``xp=cupy`` for the
    GPU population path.
    """
    if beta_unrewarded is None:
        beta_unrewarded = beta
    if observed_reward is None:
        observed_reward = 0
    else:
        observed_reward = xp.nan_to_num(observed_reward, nan=0)
    learning_rate = xp.where(observed_reward == 0, beta_unrewarded, beta)
    new_reward_rate = reward_rate + \
                      learning_rate * (observed_reward - reward_rate)
    if group_every != 0:
        new_reward_rate = xp.round(new_reward_rate / group_every) * group_every
    return new_reward_rate


def bound_scale_from_reward_rate(reward_rate):
    """Per-trial bound scale for the Bound-RewardRate ("scale-bound") drift.

    Maps the reward rate ``r`` to the factor multiplying the base bound::

        b_t = BOUND * bound_scale_from_reward_rate(r)
            = BOUND * (2 - r)
            = BOUND + (1 - r) * BOUND

    ``BOUND`` is the floor (half-width at ``r = 1``); the bound widens to
    ``2 * BOUND`` as reward rate falls toward 0. Because the reward rate is
    an EMA in ``[0, 1]``, the scale ``s_t = 2 - r`` stays in ``[1, 2]`` — so
    the path-D rescale ``1 / s_t`` (drift/noise divided by the scale) is
    always finite; no divide-by-zero floor is needed. Pure arithmetic:
    backend-agnostic (NumPy or CuPy ``xp`` arrays) and NaN-preserving, so
    NaN-padded trial slots propagate through unchanged.
    """
    return 2.0 - reward_rate


def compute_starting_point_z(q_left, q_right, delta, offset, include_Q,
                              *, xp=np, bound=None):
    """Return normalized starting point z.

    Default (``bound=None``): legacy semantic. ``z`` lives in ``[-1, 1]``,
    interpreted as a fraction of the absorbing bound. Classic and R-only
    MLE variants have no starting-point bias; offset is only meaningful
    when ``include_Q`` is True.

    Absolute mode (``bound`` provided): ``z = clip(delta*q + offset,
    -bound, +bound)`` — bias is interpreted in absolute DDM-state units
    and clipped to the active bound directly. Used by ``--scale-bound``
    fits (where BOUND varies per candidate) so the bias contribution
    doesn't implicitly track the bound. ``bound`` may be a scalar or
    a broadcast-compatible array.
    """
    if not include_Q:
        return 0.0
    q_value = compute_q_value(q_left, q_right, xp=xp)
    z = delta * q_value + offset
    if bound is None:
        return xp.clip(z, -1, 1)
    return xp.clip(z, -bound, bound)


def compute_trial_mu(coherence, drift_coef, time_grid=None, *, xp=np):
    mu = drift_coef * coherence
    if time_grid is None:
        return mu
    return xp.full_like(time_grid, mu, dtype=float)


def compute_trial_sigma(base_sigma, reward_rate, include_RewardRate):
    """Scalar / array-broadcast safe; no ``xp`` needed (pure arithmetic)."""
    if include_RewardRate:
        return reward_rate * base_sigma
    return base_sigma
