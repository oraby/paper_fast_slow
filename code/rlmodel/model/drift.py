from .util import decayingQ, partialWithNames
import numpy as np
import numpy.typing as npt
from scipy import ndimage

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


def _decayQ(starting_point : npt.NDArray,
            nondectime : float,
            noise : npt.NDArray,
            drift_coef : float,
            dvs : npt.NDArray,
            dt : float,
            noise_sigma : float,
            Q_val : npt.NDArray,
            Q_VAL_DECAY_RATE : float,
            Q_VAL_COEF : float,
            Q_VAL_OFFSET : float,
            nondectime_Q : bool = True,
            ):
    global run_logger

    drift = drift_coef * dvs * dt
    non_decsision_dt = int(nondectime / dt)
    noise *= noise_sigma
    noise[:, :non_decsision_dt] = 0
    drift = np.repeat(drift, noise.shape[1]).reshape(-1, noise.shape[1])
    drift[:, :non_decsision_dt] = 0

    Q_val = np.clip(Q_val + Q_VAL_OFFSET, -1, 1)
    indices = np.arange(noise.shape[1])
    Q_val_decay_form = (1 - indices / noise.shape[1]) ** Q_VAL_DECAY_RATE
    Q_val_noise = Q_val[:, np.newaxis] * Q_val_decay_form
    Q_val_noise *= Q_VAL_COEF * dt
    if not nondectime_Q:
        Q_val_noise = ndimage.shift(Q_val_noise, non_decsision_dt, cval=0)

    isolated_drifts = drift + noise + Q_val_noise
    isolated_drifts[:,0] = starting_point
    dx = np.cumsum(isolated_drifts, axis=1)

    isolated_drifts = drift # np.asarray(isolated_drifts).reshape(-1, dx.shape[1])
    # print("Drift shape:", drift.shape, "Isolated drifts shape:", isolated_drifts.shape,
    #       "Noise shape:", noise.shape, "dx shape:", dx.shape)
    if run_logger is not None:
        run_logger.drift = drift
        run_logger.noise_amped = noise
        run_logger.isolated_drifts = isolated_drifts
        run_logger.Q_val_decay_form = Q_val_decay_form
        run_logger.Q_val_noise = Q_val_noise
        # run_logger.drift_decaying_Q = decaying_Q
    return dx



def _noiseGainDecayingQ(starting_point: npt.NDArray,
                        nondectime: float,
                        noise: npt.NDArray,
                        drift_coef: float,
                        dvs: npt.NDArray,
                        dt: float,
                        noise_sigma: float,
                        Q_val: npt.NDArray,
                        Q_VAL_DECAY_RATE: float,
                        Q_VAL_COEF: float,
                        Q_VAL_OFFSET: float,
                        RewardRate: npt.NDArray,
                        nondectime_Q: bool = True):
    global run_logger

    non_decsision_dt = int(nondectime / dt)

    noise *= noise_sigma
    noise *= RewardRate[:, np.newaxis]
    noise[:, :non_decsision_dt] = 0

    drift = drift_coef * dvs * dt
    drift = np.repeat(drift, noise.shape[1]).reshape(-1, noise.shape[1])
    drift[:, :non_decsision_dt] = 0

    Q_val = np.clip(Q_val + Q_VAL_OFFSET, -1, 1)
    indices = np.arange(noise.shape[1])
    Q_val_decay_form = (1 - indices / noise.shape[1]) ** Q_VAL_DECAY_RATE

    Q_val_noise = Q_val[:, np.newaxis] * Q_val_decay_form
    Q_val_noise *= Q_VAL_COEF * dt

    if not nondectime_Q:
        Q_val_noise = ndimage.shift(Q_val_noise, non_decsision_dt, cval=0)

    # Combine and integrate
    isolated_drifts = drift + noise + Q_val_noise
    isolated_drifts[:, 0] = starting_point
    dx = np.cumsum(isolated_drifts, axis=1)

    # Logging (match your existing convention)
    isolated_drifts = drift
    if run_logger is not None:
        run_logger.drift = drift
        run_logger.noise_amped = noise
        run_logger.isolated_drifts = isolated_drifts
        run_logger.Q_val_decay_form = Q_val_decay_form
        run_logger.Q_val_noise = Q_val_noise

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
    ``BOUND * r_t``, but implemented as a fixed-bound diffusion with
    drift and noise divided by ``r_t``. Proven observationally
    identical to the varying-bound ground truth in Phase 1
    (scale_bound_equivalence.ipynb). NOTE: starting_point inherits
    the existing simulateDDMTrial convention (``z_norm * BOUND``) —
    bias is "fraction of base bound", not absolute. This Chisqr
    simplification differs from the MLE path's absolute-bias
    handling; documented as a known scope limitation.
    """
    global run_logger

    inv_rr = 1.0 / RewardRate[:, np.newaxis]
    drift = drift_coef * dvs * dt
    non_decision_dt = int(nondectime / dt)
    noise *= noise_sigma
    noise *= inv_rr               # σ / r_t per trial
    noise[:, :non_decision_dt] = 0
    drift = np.repeat(drift, noise.shape[1]).reshape(-1, noise.shape[1])
    drift *= inv_rr               # μ / r_t per trial
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


def _boundGainDecayingQ(starting_point: npt.NDArray,
                        nondectime: float,
                        noise: npt.NDArray,
                        drift_coef: float,
                        dvs: npt.NDArray,
                        dt: float,
                        noise_sigma: float,
                        Q_val: npt.NDArray,
                        Q_VAL_DECAY_RATE: float,
                        Q_VAL_COEF: float,
                        Q_VAL_OFFSET: float,
                        RewardRate: npt.NDArray,
                        nondectime_Q: bool = True):
    """Bound-RewardRate Decay Q variant. Same rescaling rule as
    ``_boundGainRewardRate`` applied to the drift + Q-decay noise.
    """
    global run_logger

    inv_rr = 1.0 / RewardRate[:, np.newaxis]
    non_decsision_dt = int(nondectime / dt)
    noise *= noise_sigma
    noise *= inv_rr
    noise[:, :non_decsision_dt] = 0

    drift = drift_coef * dvs * dt
    drift = np.repeat(drift, noise.shape[1]).reshape(-1, noise.shape[1])
    drift *= inv_rr
    drift[:, :non_decsision_dt] = 0

    Q_val = np.clip(Q_val + Q_VAL_OFFSET, -1, 1)
    indices = np.arange(noise.shape[1])
    Q_val_decay_form = (1 - indices / noise.shape[1]) ** Q_VAL_DECAY_RATE
    Q_val_noise = Q_val[:, np.newaxis] * Q_val_decay_form
    Q_val_noise *= Q_VAL_COEF * dt
    Q_val_noise *= inv_rr     # Q-decay drift also rescales

    if not nondectime_Q:
        Q_val_noise = ndimage.shift(Q_val_noise, non_decsision_dt, cval=0)

    isolated_drifts = drift + noise + Q_val_noise
    isolated_drifts[:, 0] = starting_point
    dx = np.cumsum(isolated_drifts, axis=1)

    isolated_drifts = drift
    if run_logger is not None:
        run_logger.drift = drift
        run_logger.noise_amped = noise
        run_logger.isolated_drifts = isolated_drifts
        run_logger.Q_val_decay_form = Q_val_decay_form
        run_logger.Q_val_noise = Q_val_noise
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
        # run_logger.drift_decaying_Q = decaying_Q
    return dx



DRIFT_FN_DICT = {
    "Classic": _driftClassic,
    "NoiseGain-RewardRate": _noiseGainRewardRate,
    # Bound-RewardRate variants implement the path-D rescaling
    # (drift / r_t, noise / r_t) so the *observable* behavior is the
    # one a fixed-noise, varying-bound (b_t = BOUND * r_t) model would
    # produce — proven in Phase 1 (scale_bound_equivalence.ipynb).
    # The math is genuinely different from NoiseGain (sigma * r_t),
    # so these are separate Python functions, not aliases.
    "Bound-RewardRate": _boundGainRewardRate,
    "Decay Q": partialWithNames(_decayQ, nondectime_Q=True, Q_VAL_OFFSET=0),
    "Decay Q (Offset)": partialWithNames(_decayQ, nondectime_Q=True),
    "NoiseGain-RewardRate Decay Q": partialWithNames(_noiseGainDecayingQ, nondectime_Q=True, Q_VAL_OFFSET=0),
    "NoiseGain-RewardRate Decay Q (Offset)": partialWithNames(_noiseGainDecayingQ, nondectime_Q=True),
    "Bound-RewardRate Decay Q":             partialWithNames(_boundGainDecayingQ, nondectime_Q=True, Q_VAL_OFFSET=0),
    "Bound-RewardRate Decay Q (Offset)":    partialWithNames(_boundGainDecayingQ, nondectime_Q=True),
}
