from dataclasses import dataclass, asdict
from typing import NamedTuple
import os

# init_vals = {
#     "DRIFT_COEF": (0, 3, 1),
#     "NOISE_SIGMA": (0, 40, 15),
#     "BOUND": (1, 1, 1),
#     "BIAS_COEF": (0, 1, .95),
#     "Bias Fixed": (-1, 1, 0),
#     "ALPHA": (0, 1, .3),
#     "BETA": (0, 1, .3),
#     # "Drift RR Coef": (0, .3, .1),
#     "Bias mu": (-1, 1, 0),
#     "Bias sigma": (0, 1, .1),
#     "NON_DECISION_TIME": (0, .6, .3),
#     "Q_VAL_DECAY_RATE": (.1, 4, 1),
#     "Q_VAL_COEF": (1, 20, 5), #.1, .03),#1, .5),
# }


class InitVal(NamedTuple):
    Min : float
    Max : float
    Default : float

@dataclass
class InitVals:
    DRIFT_COEF : InitVal        = InitVal(0, 20, 1)
    NOISE_SIGMA : InitVal       = InitVal(0, 100, 15)#InitVal(0, 40, 15)
    BOUND : InitVal             = InitVal(1, 1, 1)
    BIAS_COEF : InitVal         = InitVal(0, 1, .95)
    BIAS_FIXED : InitVal        = InitVal(-1, 1, 0)
    ALPHA : InitVal             = InitVal(0, 1, .3)
    BETA : InitVal              = InitVal(0, 1, .3)
    # Drift RR Coef : InitVal     = InitVal(0, .3, .1)
    BIAS_MU : InitVal           = InitVal(-1, 1, 0)
    BIAS_SIGMA : InitVal        = InitVal(0, 1, .1)
    NON_DECISION_TIME : InitVal = InitVal(0, 1, .3)
    Q_VAL_DECAY_RATE : InitVal  = InitVal(.1, 30, 1)#InitVal(.1, 8, 1)
    Q_VAL_COEF : InitVal        = InitVal(1, 100, 5)
    Q_VAL_OFFSET : InitVal      = InitVal(-1, 1, 0)

    def __init__(self):
        self._extras = {}
        self._removed = {}

    def get(self, val):
        return (asdict(self) | self._extras).get(val)
    
    def items(self):
        return (asdict(self) | self._extras).items()
    
    def keys(self):
        return (asdict(self) | self._extras).keys()
    
    def values(self):
        return (asdict(self) | self._extras).values()
    
    def toDict(self):
        return asdict(self) | self._extras


DT = 0.005
T_dur = 3
NUM_CPUS = os.cpu_count()