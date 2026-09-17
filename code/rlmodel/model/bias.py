import numpy as np
import numpy.typing as npt

# This should be set by the logic class
# rnd_default_rng = np.random.default_rng()
rnd_default_rng = None

def _biasNone(size : int):
    return np.zeros(size)

def _biasQVal(size : int,
              BIAS_COEF : float,
              Q_val: npt.NDArray,
              Q_VAL_OFFSET: float,
              ):
    assert len(Q_val) == size
    Q_val_offsetted = np.clip(Q_val + Q_VAL_OFFSET, -1, 1)
    return Q_val_offsetted * BIAS_COEF

BIAS_FN_DICT = {
    "None_":_biasNone,
    "Q-Val (Offset)":_biasQVal,
}
