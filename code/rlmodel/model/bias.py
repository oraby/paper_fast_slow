from .util import partialWithNames
import numpy as np
import numpy.typing as npt

# This should be set by the logic class
# rnd_default_rng = np.random.default_rng()
rnd_default_rng = None

def _biasNone(size : int):
    return np.zeros(size)

def _biasMeanDir(size : int,
                 BIAS_COEF : float,
                 BIAS_MU : float,
                 BIAS_SIGMA : float, ):
    # norm = np.random.normal(BIAS_MU, BIAS_SIGMA, size=size)
    norm = rnd_default_rng.normal(BIAS_MU, BIAS_SIGMA, size=size)
    return norm * BIAS_COEF

def _biasMeanCorrIncorr(size : int,
                        BIAS_COEF : float,
                        BIAS_MU : float,
                        BIAS_SIGMA : float,
                        DV : npt.NDArray):
    norm = _biasMeanDir(size, BIAS_MU, BIAS_SIGMA)
    norm[DV < 0] = -norm[DV < 0]
    return norm * BIAS_COEF

def _biasFixedDir(size : int,
                  BIAS_COEF : float,
                  BIAS_FIXED : float):
    BIAS_FIXED = np.full(size, BIAS_FIXED)
    return BIAS_FIXED * BIAS_COEF

def _biasFixedCorrIncorr(size : int,
                         BIAS_COEF : float,
                         BIAS_FIXED : float,
                         DV : npt.NDArray):
    BIAS_FIXED = np.full(size, BIAS_FIXED)
    BIAS_FIXED[DV < 0] = -BIAS_FIXED[DV < 0]
    return BIAS_FIXED * BIAS_COEF

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
    "Fixed (Dir)":_biasFixedDir,
    "Fixed (Corr/Incorr)":_biasFixedCorrIncorr,
    "μ, σ (Dir)":_biasMeanDir,
    "μ, σ (Corr/Incorr)":_biasMeanCorrIncorr,
    "Q-Val":partialWithNames(_biasQVal, Q_VAL_OFFSET=0.0),
    "Q-Val (Offset)":_biasQVal,
}
