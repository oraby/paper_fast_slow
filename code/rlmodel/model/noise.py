import numpy as np

run_logger = None

# This should be set by the logic class
# rnd_default_rng = np.random.default_rng()
rnd_default_rng = None

_last_noise_arr = None
_last_norm_size = None
def _noiseNormal(size : tuple[int, int],
                 dt : float):
    global _last_noise_arr, _last_norm_size
    sqrt_dt = np.sqrt(dt)
    if _last_norm_size == size:
        noise_arr =_last_noise_arr
        rnd_default_rng.standard_normal(size=size, out=noise_arr)
        np.multiply(noise_arr, sqrt_dt, out=noise_arr)
    else:
        noise_arr = rnd_default_rng.standard_normal(size=size) * sqrt_dt
        _last_noise_arr = noise_arr
        _last_norm_size = size
    return noise_arr


NOISE_FN_DICT = {
    "Normal(0, 1)":_noiseNormal,
}
