from .util import partialWithNames
from .state_updates import (DEFAULT_RR_DRIFT_MAP, RR_DRIFT_MAPS,
                            bound_scale_from_reward_rate,
                            drift_scale_from_reward_rate)
import numpy as np
import numpy.typing as npt

run_logger = None

# This should be set by the logic class
# rnd_default_rng = np.random.default_rng()
rnd_default_rng = None


def _driftClassic(starting_point : npt.NDArray,
                  nondectime : float,
                  noise : npt.NDArray,
                  drift_coef : float,
                  dvs : npt.NDArray,
                  dt : float,
                  noise_sigma : float):
    global run_logger

    drift = drift_coef * dvs * dt
    non_decsision_dt = int(nondectime / dt)
    noise *= noise_sigma
    noise[:, :non_decsision_dt] = 0
    drift = np.repeat(drift, noise.shape[1]).reshape(-1, noise.shape[1])
    drift[:, :non_decsision_dt] = 0

    isolated_drifts = drift + noise
    isolated_drifts[:,0] = starting_point
    dx = np.cumsum(isolated_drifts, axis=1)

    if run_logger is not None:
        run_logger.drift = drift
        run_logger.noise_amped = noise
        run_logger.isolated_drifts = isolated_drifts
    return dx


def _boundGainRewardRate(starting_point : npt.NDArray,
                         nondectime : float,
                         noise : npt.NDArray,
                         drift_coef : float,
                         dvs : npt.NDArray,
                         dt : float,
                         noise_sigma : float,
                         RewardRate : npt.NDArray):
    """Bound-RewardRate Chisqr drift via the path-D rescaling.

    Mathematically equivalent to simulating with per-trial bound
    ``BOUND * (2 - r_t) = BOUND + (1 - r_t) * BOUND``, but implemented as
    a fixed-bound diffusion with drift and noise divided by the scale
    ``s_t = 2 - r_t``. ``BOUND`` is the floor (half-width at r_t = 1); the
    bound widens toward ``2 * BOUND`` as reward rate falls to 0. Proven
    observationally identical to the varying-bound ground truth in Phase 1
    (scale_bound_equivalence.ipynb). NOTE: starting_point inherits
    the existing simulateDDMTrial convention (``z_norm * BOUND``) —
    bias is "fraction of base bound", not absolute. This Chisqr
    simplification differs from the MLE path's absolute-bias
    handling; documented as a known scope limitation.
    """
    global run_logger

    # Per-trial bound scale s_t = 2 - r_t (see
    # bound_scale_from_reward_rate): b_t = BOUND * (2 - r_t). The path-D
    # rescale divides drift and noise by s_t. Since r_t is an EMA in
    # [0, 1], s_t stays in [1, 2] — always finite, so no divide-by-zero
    # floor is needed. NaN-padded slots beyond actual session length in
    # logic.py still propagate (2 - NaN = NaN).
    inv_scale = 1.0 / bound_scale_from_reward_rate(RewardRate)[:, np.newaxis]
    drift = drift_coef * dvs * dt
    non_decision_dt = int(nondectime / dt)
    noise *= noise_sigma
    noise *= inv_scale            # σ / s_t per trial
    noise[:, :non_decision_dt] = 0
    drift = np.repeat(drift, noise.shape[1]).reshape(-1, noise.shape[1])
    drift *= inv_scale            # μ / s_t per trial
    drift[:, :non_decision_dt] = 0

    isolated_drifts = drift + noise
    isolated_drifts[:, 0] = starting_point
    dx = np.cumsum(isolated_drifts, axis=1)

    isolated_drifts = drift
    if run_logger is not None:
        run_logger.drift = drift
        run_logger.noise_amped = noise
        run_logger.isolated_drifts = isolated_drifts
    return dx


def _noiseGainRewardRate(starting_point : npt.NDArray,
                         nondectime : float,
                         noise : npt.NDArray,
                         drift_coef : float,
                         dvs : npt.NDArray,
                         dt : float,
                         noise_sigma : float,
                         RewardRate : npt.NDArray):
    global run_logger

    drift = drift_coef * dvs * dt
    non_decision_dt = int(nondectime / dt)
    noise *= noise_sigma
    noise *= RewardRate[:, np.newaxis]
    noise[:, :non_decision_dt] = 0
    drift = np.repeat(drift, noise.shape[1]).reshape(-1, noise.shape[1])
    drift[:, :non_decision_dt] = 0

    isolated_drifts = drift + noise
    isolated_drifts[:,0] = starting_point
    dx = np.cumsum(isolated_drifts, axis=1)

    isolated_drifts = drift # np.asarray(isolated_drifts).reshape(-1, dx.shape[1])
    # print("Drift shape:", drift.shape, "Isolated drifts shape:", isolated_drifts.shape,
    #       "Noise shape:", noise.shape, "dx shape:", dx.shape)
    if run_logger is not None:
        run_logger.drift = drift
        run_logger.noise_amped = noise
        run_logger.isolated_drifts = isolated_drifts
    return dx


def _driftGainRewardRate(starting_point : npt.NDArray,
                         nondectime : float,
                         noise : npt.NDArray,
                         drift_coef : float,
                         dvs : npt.NDArray,
                         dt : float,
                         noise_sigma : float,
                         RewardRate : npt.NDArray,
                         RR_DRIFT_MAP : str = DEFAULT_RR_DRIFT_MAP):
    """DriftGain-RewardRate: the reward rate modulates the DRIFT.

    Per-step update::

        d(t) = d(t-1) + DV * V * g(r_t) * dt + S * sqrt(dt) * eps

    with ``g(r_t) = drift_scale_from_reward_rate(r_t, RR_DRIFT_MAP)``, i.e.
    ``2 - r_t`` by default so the step is ``DV*(2V - r_t*V)*dt``.

    The third mutually-exclusive reward-rate channel: ``noise`` is scaled by
    ``noise_sigma`` ONLY (no ``*= RewardRate`` as in NoiseGain, no ``/= s_t``
    as in Bound) and the bound stays flat, so ``r_t`` acts purely on the
    evidence gain. Unlike the Bound- family this is not a rescaling of an
    equivalent model, so it is not subject to the ``--scale-bound``
    soft-limit — it runs under either scale-axis convention.

    ``RR_DRIFT_MAP`` is bound per registry key via ``partialWithNames`` (see
    DRIFT_FN_DICT), which is also what keeps ``util._fnColsAndKargs`` from
    tripping over its non-float/non-NDArray annotation.
    """
    global run_logger

    # g(r_t) in [1, 2] for r_t in [0, 1]; NaN-padded trial slots beyond the
    # actual session length in logic.py still propagate (2 - NaN = NaN).
    drift_scale = drift_scale_from_reward_rate(RewardRate, RR_DRIFT_MAP)
    drift = drift_coef * dvs * drift_scale * dt
    non_decision_dt = int(nondectime / dt)
    noise *= noise_sigma          # sigma is FLAT: no reward-rate gain
    noise[:, :non_decision_dt] = 0
    drift = np.repeat(drift, noise.shape[1]).reshape(-1, noise.shape[1])
    drift[:, :non_decision_dt] = 0

    isolated_drifts = drift + noise
    isolated_drifts[:, 0] = starting_point
    dx = np.cumsum(isolated_drifts, axis=1)

    isolated_drifts = drift
    if run_logger is not None:
        run_logger.drift = drift
        run_logger.noise_amped = noise
        run_logger.isolated_drifts = isolated_drifts
    return dx


DRIFT_FN_DICT = {
    "Classic": _driftClassic,
    "NoiseGain-RewardRate": _noiseGainRewardRate,
    # Bound-RewardRate variants implement the path-D rescaling
    # (drift / s_t, noise / s_t with s_t = 2 - r_t) so the *observable*
    # behavior is the one a fixed-noise, varying-bound
    # (b_t = BOUND * (2 - r_t) = BOUND + (1 - r_t) * BOUND) model would
    # produce — proven in Phase 1 (scale_bound_equivalence.ipynb).
    # The math is genuinely different from NoiseGain (sigma * r_t),
    # so these are separate Python functions, not aliases.
    "Bound-RewardRate": _boundGainRewardRate,
    # DriftGain-RewardRate: the reward rate modulates the coherence DRIFT
    # (mu *= g(r_t)) with sigma and the bound both flat — a third channel,
    # NOT a rescaling of either of the two above. ``RR_DRIFT_MAP`` picks the
    # r -> gain mapping and is bound here (rather than being a plain default
    # on the function) because util._fnColsAndKargs only tolerates
    # float / npt.NDArray / bool annotations on *unbound* params.
    #   g(r) = 2 - r  ("DriftGain-*"):        high reward rate => slower
    #   g(r) = 1 + r  ("DriftGain(1+r)-*"):   high reward rate => faster
    "DriftGain-RewardRate":                 partialWithNames(_driftGainRewardRate, RR_DRIFT_MAP="2-r"),
    "DriftGain(1+r)-RewardRate":             partialWithNames(_driftGainRewardRate, RR_DRIFT_MAP="1+r"),
}


# ---------------------------------------------------------------------------
# RewardRate alias layer (CLI + GUI)
# ---------------------------------------------------------------------------
#
# NoiseGain-RewardRate and Bound-RewardRate are mathematically equivalent
# under the path-D rescale (proven in scale_bound_equivalence.ipynb); the
# user shouldn't have to pick the implementation. RewardRate is the only
# name exposed at the user-facing edge (--drift CLI choices and the GUI
# Drift Fn dropdown). The active scale-bound state (--scale-bound on the
# CLI; the Scale-How dropdown in the GUI) routes the alias to the right
# DRIFT_FN_DICT key. The internal registry is unchanged so every existing
# saved-fit pickle still loads via fit.evolveFP.
# The reward-rate CHANNEL keys used inside _REWARDRATE_ALIASES below. The
# channel says which quantity the learned reward rate modulates; exactly
# one is active per fit.
RR_CHANNEL_NOISE = "noise"           # sigma *= r          (legacy default)
RR_CHANNEL_BOUND = "bound"           # b = BOUND * (2 - r) (--scale-bound)
RR_CHANNEL_DRIFT_2_R = "drift:2-r"   # mu *= (2 - r)       (--use-drift-rr)
RR_CHANNEL_DRIFT_1_R = "drift:1+r"   # mu *= (1 + r)

_REWARDRATE_ALIASES = {
    # alias -> {channel: resolved DRIFT_FN_DICT key}
    "RewardRate": {
        RR_CHANNEL_NOISE:     "NoiseGain-RewardRate",
        RR_CHANNEL_BOUND:     "Bound-RewardRate",
        RR_CHANNEL_DRIFT_2_R: "DriftGain-RewardRate",
        RR_CHANNEL_DRIFT_1_R: "DriftGain(1+r)-RewardRate",
    },
}


# Reverse map for the two SELECTABLE-alias channels (noise / bound): every
# implementation name -> the alias a user can actually pick in the dropdown.
# Used by the GUI cache-migration shim so users who reopen the notebook with
# a cache that holds an old "Drift Fn" value get auto-migrated — which is
# exactly why the DriftGain families are NOT in here: migrating a cached
# value to a non-selectable name would wedge the dropdown.
_REWARDRATE_ALIAS_FOR_INTERNAL = {
    channels[channel]: alias
    for alias, channels in _REWARDRATE_ALIASES.items()
    for channel in (RR_CHANNEL_NOISE, RR_CHANNEL_BOUND)
}


# DISPLAY aliases for the drift channel. These are not dropdown-selectable
# (the "RR as Drift" checkbox / --use-drift-rr picks the channel); they exist
# so ``FitFileId.model_key`` gives the drift-channel fits their own row in
# the model_compare / aggregate grid instead of collapsing them onto the
# NoiseGain row the way Bound- intentionally does.
_DRIFT_RR_DISPLAY_ALIAS = {
    channels[channel]: alias.replace("RewardRate", display, 1)
    for alias, channels in _REWARDRATE_ALIASES.items()
    for channel, display in ((RR_CHANNEL_DRIFT_2_R, "RewardRate (Drift)"),
                             (RR_CHANNEL_DRIFT_1_R, "RewardRate (Drift 1+r)"))
}


def display_alias_for_drift(drift_str):
    """The user-facing label for an internal ``DRIFT_FN_DICT`` key.

    NoiseGain-/Bound- collapse onto the shared ``RewardRate`` alias (they
    are the same abstract model fitted on different scale axes, so they
    belong in one model_compare row with two criterion columns).
    DriftGain- gets its own ``RewardRate (Drift[ 1+r])`` label because it is
    a genuinely different model. Non-alias drifts (``Classic``)
    pass through unchanged.
    """
    if drift_str in _REWARDRATE_ALIAS_FOR_INTERNAL:
        return _REWARDRATE_ALIAS_FOR_INTERNAL[drift_str]
    return _DRIFT_RR_DISPLAY_ALIAS.get(drift_str, drift_str)


# Inverse of resolve_drift_alias: internal key -> (alias, channel). Lets a
# consumer holding an internal name (e.g. a saved-fit filename) recover the
# control state that would select it — used by the GUI's batch save-figures
# loop, which iterates saved fits and has to drive the widgets back.
_DRIFT_CHANNEL_FOR_INTERNAL = {
    impl: (alias, channel)
    for alias, channels in _REWARDRATE_ALIASES.items()
    for channel, impl in channels.items()
}


def channel_for_drift(drift_str):
    """``(alias, channel)`` for an internal reward-rate drift key, else None.

    ``channel`` is one of the ``RR_CHANNEL_*`` constants, so
    ``resolve_drift_alias(alias, ...)`` with the matching control state
    round-trips back to ``drift_str``.
    """
    return _DRIFT_CHANNEL_FOR_INTERNAL.get(drift_str)


def is_rewardrate_alias(drift_str):
    """True iff ``drift_str`` is a user-facing R-learning drift alias.

    The gate for ``--use-drift-rr`` / the GUI's "RR as Drift" checkbox:
    routing the reward rate to the drift only means something on a model
    that actually learns a reward rate.
    """
    return drift_str in _REWARDRATE_ALIASES


def user_facing_drift_keys():
    """User-facing drift names: ``DRIFT_FN_DICT`` keys with every
    reward-rate implementation collapsed to its ``RewardRate`` alias.

    Used by argparse's ``choices=`` and the GUI dropdown so they show
    one ``RewardRate`` entry per family instead of one per channel.
    """
    suppressed = set(_REWARDRATE_ALIAS_FOR_INTERNAL) | set(_DRIFT_RR_DISPLAY_ALIAS)
    keys = [k for k in DRIFT_FN_DICT if k not in suppressed]
    keys.extend(_REWARDRATE_ALIASES)
    return keys


def resolve_drift_alias(drift_str, scale_bound, use_drift_rr=False,
                        drift_rr_map=DEFAULT_RR_DRIFT_MAP):
    """Resolve a ``RewardRate*`` alias to its canonical ``DRIFT_FN_DICT``
    registry key. No-op for non-alias drifts (``Classic``).

    ``use_drift_rr`` (``--use-drift-rr`` / the GUI checkbox) WINS over
    ``scale_bound``: routing the reward rate to the drift overrides the
    noise/bound channel entirely, and ``--scale-bound`` then only means
    what its help text says — which of (BOUND, NOISE_SIGMA) is the fitted
    scale axis. ``drift_rr_map`` selects the r -> gain mapping and is
    ignored unless ``use_drift_rr`` is set.
    """
    if drift_str not in _REWARDRATE_ALIASES:
        return drift_str
    channels = _REWARDRATE_ALIASES[drift_str]
    if use_drift_rr:
        channel = f"drift:{drift_rr_map}"
        if channel not in channels:
            raise ValueError(
                f"Unknown reward-rate drift map {drift_rr_map!r}; expected "
                f"one of {list(RR_DRIFT_MAPS)}.")
        return channels[channel]
    return channels[RR_CHANNEL_BOUND if scale_bound else RR_CHANNEL_NOISE]
