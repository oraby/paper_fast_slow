from .util import decayingQ, partialWithNames
import numpy as np
import numpy.typing as npt
from scipy import ndimage

run_logger = None

# This should be set by the logic class
# rnd_default_rng = np.random.default_rng()
rnd_default_rng = None

# def _driftClassic(starting_point : npt.NDArray,
#                   nondectime : float,
#                   noise : npt.NDArray, 
#                   drift_coef : float,
#                   dvs : npt.NDArray,
#                   dt : float,
#                   noise_sigma : float):
#     # num_steps = 600
#     # 15 dvs = (.3, 1, -1, -.5, .3, -.5, 1, ...)
#     # 15 starting_point = (0, 0, .3, -.2 ,, .3 ....)
#     # dt = 0.005
#     # drift_coef = 3
#     # noise = (15, 600)
#     # noise_sigma = 10
#     drift = drift_coef * dvs * dt
#     non_decsision_dt = int(nondectime / dt)
#     noise *= noise_sigma
#     noise[:, :non_decsision_dt] = 0
#     drift = np.repeat(drift, noise.shape[1]).reshape(-1, noise.shape[1])
#     drift[:, :non_decsision_dt] = 0

#     isolated_drifts = drift + noise
#     isolated_drifts[:,0] = starting_point
#     dx = np.cumsum(isolated_drifts, axis=1)
#     return dx


def _driftClassic(starting_point : npt.NDArray, 
                  nondectime : float, 
                  noise : npt.NDArray,
                  drift_coef : float,
                  dvs : npt.NDArray,
                  dt : float,
                  noise_sigma : float):
    global run_logger
    # y=sgn(x−0.5)⋅∣x−0.5∣**k + 0.5
    # abs_dv_minus_half = np.abs(dvs) - .5
    # K = 4 # k > 1 stretches the function, k < 1 compresses it
    # dvs = np.sign(dvs)*(np.sign(abs_dv_minus_half)*(np.abs(abs_dv_minus_half)**K) + .5)


    drift = drift_coef * dvs * dt #* (1 - RewardRate : npt.NDArray) # (.5 + .5*RewardRate : npt.NDArray)
    non_decsision_dt = int(nondectime / dt)
    noise *= noise_sigma
    noise[:, :non_decsision_dt] = 0
    drift = np.repeat(drift, noise.shape[1]).reshape(-1, noise.shape[1])
    drift[:, :non_decsision_dt] = 0


    isolated_drifts = drift + noise
    isolated_drifts[:,0] = starting_point
    dx = np.cumsum(isolated_drifts, axis=1)

    # print("Drift shape:", drift.shape, "Isolated drifts shape:", isolated_drifts.shape,
    #       "Noise shape:", noise.shape, "dx shape:", dx.shape)
    if run_logger is not None:
        run_logger.drift = drift
        run_logger.noise_amped = noise
        run_logger.isolated_drifts = isolated_drifts
    return dx


def _contanstQRewardRate(starting_point : npt.NDArray,
                         nondectime : float,
                         noise : npt.NDArray,
                         drift_coef : float, 
                         dvs : npt.NDArray,
                         dt : float,
                         noise_sigma : float,
                         RewardRate : npt.NDArray,
                         Q_val : npt.NDArray,
                         Q_VAL_COEF : float,
                         nondectime_Q : bool = True):

    global run_logger
    drift = drift_coef * dvs * dt #* (1 - RewardRate) # (.5 + .5*RewardRate)
    non_decsision_dt = int(nondectime / dt)
    noise *= noise_sigma
    noise[:, :non_decsision_dt] = 0
    drift = np.repeat(drift, noise.shape[1]).reshape(-1, noise.shape[1])
    drift[:, :non_decsision_dt] = 0

    # mask = np.random.uniform(0, 1, size=drift.shape) < RewardRate[:, np.newaxis]
    mask = rnd_default_rng.uniform(0, 1, size=drift.shape) < RewardRate[:, np.newaxis]
    Q_val_noise = Q_val.repeat(noise.shape[1]).reshape(-1, noise.shape[1])
    Q_val_noise *= Q_VAL_COEF * dt
    Q_val_noise[~mask] = 0
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
        run_logger.Q_val_noise = Q_val_noise
        # run_logger.drift_decaying_Q = decaying_Q
    return dx

def _decayQRewardRate(starting_point : npt.NDArray,
                      nondectime : float, 
                      noise : npt.NDArray,
                      drift_coef : float,
                      dvs : npt.NDArray,
                      dt : float,
                      noise_sigma : float,
                      RewardRate : npt.NDArray,
                      Q_val : npt.NDArray,
                      Q_VAL_DECAY_RATE : float,
                      Q_VAL_COEF : float,
                      nondectime_Q : bool = True,
                      ):
    global run_logger

    drift = drift_coef * dvs * dt
    non_decsision_dt = int(nondectime / dt)
    noise *= noise_sigma
    noise[:, :non_decsision_dt] = 0
    drift = np.repeat(drift, noise.shape[1]).reshape(-1, noise.shape[1])
    drift[:, :non_decsision_dt] = 0

    # mask = np.random.uniform(0, 1, size=drift.shape) < RewardRate[:, np.newaxis]
    mask = rnd_default_rng.uniform(0, 1, size=drift.shape) < RewardRate[:, np.newaxis]
    indices = np.arange(noise.shape[1])
    Q_val_decay_form = (1 - indices / noise.shape[1]) ** Q_VAL_DECAY_RATE
    Q_val_noise = Q_val[:, np.newaxis] * Q_val_decay_form
    Q_val_noise *= Q_VAL_COEF * dt
    Q_val_noise[~mask] = 0
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



_last_size = None
_last_drift_coef = None
_last_noise_arr = None
_last_drift_arr, _last_drift_arr2 = None, None
_last_Q_decay_rate, _last_Q_coef = None, None
_last_Q_val_noise_arr, _last_Q_val_noise_arr2 = None, None
_last_isolated_drifts_arr, _last_isolated_drifts_arr2 = None, None
def _noiseGainDecayingQ(starting_point : npt.NDArray,
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
                        RewardRate : npt.NDArray,
                        nondectime_Q : bool = True):
    global run_logger
    global _last_size, _last_noise_arr
    global _last_drift_coef, _last_drift_arr, _last_drift_arr2
    global _last_Q_decay_rate, _last_Q_coef, _last_Q_val_noise_arr, _last_Q_val_noise_arr2
    global _last_isolated_drifts_arr, _last_isolated_drifts_arr2

    same_size = _last_size == dvs.size
    _last_size = dvs.size

    non_decsision_dt = int(nondectime / dt)
    # noise *= noise_sigma
    # noise *= RewardRate[:, np.newaxis]
    if not same_size:
        _last_noise_arr = np.empty_like(noise)
    np.multiply(noise, noise_sigma, out=_last_noise_arr)
    np.multiply(_last_noise_arr, RewardRate[:, np.newaxis], out=noise)
    noise[:, :non_decsision_dt] = 0

    # drift = drift_coef * dvs * dt
    # drift = np.repeat(drift, noise.shape[1]).reshape(-1, noise.shape[1])
    # org_drift = drift
    if same_size and _last_drift_coef == drift_coef:
        drift = _last_drift_arr
    else:
        drift = drift_coef * dt
        # drift = np.repeat(drift, noise.shape[1]).reshape(-1, noise.shape[1])
        drift = np.repeat(np.asarray(drift), 
                          noise.shape[0]*noise.shape[1]).reshape(-1, noise.shape[1])
        _last_drift_arr = drift
        _last_drift_coef = drift_coef
        _last_drift_arr2 = np.empty_like(drift)
    drift = np.multiply(drift, dvs[:, np.newaxis], out=_last_drift_arr2)
    # assert drift.shape == org_drift.shape
    # assert (drift - org_drift < .001).all(), print(np.where(drift - org_drift > .001))
    drift[:, :non_decsision_dt] = 0

    # indices = np.arange(noise.shape[1])
    # Q_val_decay_form = (1 - indices / noise.shape[1]) ** Q_VAL_DECAY_RATE
    # Q_val_noise = Q_val[:, np.newaxis] * Q_val_decay_form
    # Q_val_noise *= Q_VAL_COEF * dt
    # orig_Q_val_noise = Q_val_noise
    if same_size and _last_Q_decay_rate == Q_VAL_DECAY_RATE and _last_Q_coef == Q_VAL_COEF:
        Q_val_noise = _last_Q_val_noise_arr
    else:
        indices = np.arange(noise.shape[1])
        Q_val_decay_form = (1 - indices / noise.shape[1]) ** Q_VAL_DECAY_RATE
        # Q_val_noise = Q_val[:, np.newaxis] * Q_val_decay_form
        # Q_val_noise *= Q_VAL_COEF * dt
        Q_val_noise = Q_val_decay_form * Q_VAL_COEF * dt
        # print("1. Q_val_noise:", Q_val_noise.shape)
        Q_val_noise = np.tile(Q_val_noise, (noise.shape[0], 1))
        # print("2.Q_val_noise:", Q_val_noise.shape)
        _last_Q_val_noise_arr = Q_val_noise
        _last_Q_decay_rate = Q_VAL_DECAY_RATE
        _last_Q_coef = Q_VAL_COEF
        _last_Q_val_noise_arr2 = np.empty_like(Q_val_noise)

    Q_val = np.clip(Q_val + Q_VAL_OFFSET, -1, 1)
    Q_val_noise = np.multiply(Q_val_noise, Q_val[:, np.newaxis], 
                              out=_last_Q_val_noise_arr2)
    # assert Q_val_noise.shape == orig_Q_val_noise.shape
    # assert (Q_val_noise - orig_Q_val_noise < .001).all(), print(np.where(Q_val_noise - orig_Q_val_noise > .001))

    if not nondectime_Q:
        Q_val_noise = ndimage.shift(Q_val_noise, non_decsision_dt, cval=0,
                                    output=_last_Q_val_noise_arr2)

    if not same_size:
        _last_isolated_drifts_arr = np.empty_like(noise)
        _last_isolated_drifts_arr2 = np.empty_like(noise)
    # isolated_drifts = drift + noise + Q_val_noise
    np.add(drift, noise, out=_last_isolated_drifts_arr)
    isolated_drifts = np.add(_last_isolated_drifts_arr, Q_val_noise, out=_last_isolated_drifts_arr2)
    isolated_drifts[:,0] = starting_point
    # dx = np.cumsum(isolated_drifts, axis=1)
    dx = np.cumsum(isolated_drifts, axis=1, out=_last_isolated_drifts_arr)

    isolated_drifts = drift # np.asarray(isolated_drifts).reshape(-1, dx.shape[1])
    # print("Drift shape:", drift.shape, "Isolated drifts shape:", isolated_drifts.shape,
    #       "Noise shape:", noise.shape, "dx shape:", dx.shape)
    if run_logger is not None:
        run_logger.drift = drift
        run_logger.noise_amped = noise
        run_logger.isolated_drifts = isolated_drifts
        # run_logger.drift_decaying_Q = decaying_Q
    return dx



def _driftGainRewardRate(starting_point : npt.NDArray,
                              nondectime : float,
                              noise : npt.NDArray,
                              drift_coef : float,
                              dvs : npt.NDArray,
                              dt : float, 
                              noise_sigma : float,
                              RewardRate : npt.NDArray,):
    global run_logger

    drift = drift_coef * dvs * dt * RewardRate
    non_decsision_dt = int(nondectime / dt)
    noise *= noise_sigma
    noise[:, :non_decsision_dt] = 0
    drift = np.repeat(drift, noise.shape[1]).reshape(-1, noise.shape[1])
    drift[:, :non_decsision_dt] = 0

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
    non_decsision_dt = int(nondectime / dt)
    noise *= noise_sigma
    noise *= RewardRate[:, np.newaxis]
    noise[:, :non_decsision_dt] = 0
    drift = np.repeat(drift, noise.shape[1]).reshape(-1, noise.shape[1])
    drift[:, :non_decsision_dt] = 0

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

def _driftNoiseGainRewardRate(starting_point : npt.NDArray, 
                                   nondectime : float,
                                   noise : npt.NDArray, 
                                   drift_coef : float,
                                   dvs : npt.NDArray,
                                   dt : float, 
                                   noise_sigma : float,
                                   RewardRate : npt.NDArray):
    drift = drift_coef * dvs * dt * RewardRate
    non_decsision_dt = int(nondectime / dt)
    noise *= noise_sigma
    noise *= RewardRate[:, np.newaxis]
    noise[:, :non_decsision_dt] = 0
    drift = np.repeat(drift, noise.shape[1]).reshape(-1, noise.shape[1])
    drift[:, :non_decsision_dt] = 0

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
    # "Constant Q-RewardRate": partialWithNames(_contanstQRewardRate, nondectime_Q=True),
    # "Constant Q-RewardRate (SEP NDT)": partialWithNames(_contanstQRewardRate, nondectime_Q=False),
    # "DriftGain-RewardRate": _driftGainRewardRate,
    "NoiseGain-RewardRate": _noiseGainRewardRate,
    # "DriftNoiseGain-RewardRate": _driftNoiseGainRewardRate,
    # "Decay Q-RewardRate": partialWithNames(_decayQRewardRate, nondectime_Q=True, Q_VAL_DECAY_RATE=0),
    # "Decay Q-RewardRate (Offset)": partialWithNames(_decayQRewardRate, nondectime_Q=True),
    # "Decay Q-RewardRate (SEP NDT)": partialWithNames(_decayQRewardRate, nondectime_Q=False),
    "Decay Q": partialWithNames(_decayQ, nondectime_Q=True, Q_VAL_OFFSET=0),
    "Decay Q (Offset)": partialWithNames(_decayQ, nondectime_Q=True),
    # "Decay Q (SEP NDT)": partialWithNames(_decayQ, nondectime_Q=False),
    "NoiseGain-RewardRate Decay Q": partialWithNames(_noiseGainDecayingQ, nondectime_Q=True, Q_VAL_OFFSET=0),
    "NoiseGain-RewardRate Decay Q (Offset)": partialWithNames(_noiseGainDecayingQ, nondectime_Q=True),
    # "NoiseGain-RewardRate Decay Q (SEP NDT)": partialWithNames(_noiseGainDecayingQ, nondectime_Q=False),
}
