from .util import decayingQ
import numpy as np
import numpy.typing as npt

run_logger = None

# This should be set by the logic class
# rnd_default_rng = np.random.default_rng()
rnd_default_rng = None

_last_noise_arr = None
_last_norm_size = None
def _noiseNormal(size : tuple[int, int],
                 dt : float):
    global _last_noise_arr, _last_norm_size
    if _last_norm_size == size:
        noise_arr =_last_noise_arr
        rnd_default_rng.standard_normal(size=size, out=noise_arr)
        np.multiply(noise_arr, dt, out=noise_arr)
    else:
        noise_arr = rnd_default_rng.standard_normal(size=size) * dt
        _last_noise_arr = noise_arr
        _last_norm_size = size
    return noise_arr

def _noiseQval(size : tuple[int, int],
               dt : float,
               Q_val : npt.NDArray,
               Q_VAL_DECAY_RATE : float,
               Q_VAL_COEF : float):
    global run_logger
    rndm_noise = np.random.normal(0, 1, size=size) * dt
    decaying_Q = decayingQ(size, Q_val, Q_VAL_DECAY_RATE,  Q_VAL_COEF, dt)
    noise_Q = rndm_noise + decaying_Q
    if run_logger is not None:
        run_logger.rndm_noise = rndm_noise
        run_logger.noise_decaying_Q = decaying_Q
    return noise_Q


NOISE_FN_DICT = {
    "Normal(0, 1)":_noiseNormal,
    "Decaying Q-Val":_noiseQval,
}