from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FirstPassageResult:
    times: np.ndarray
    f_upper: np.ndarray
    f_lower: np.ndarray
    survival: np.ndarray
    x_grid: Optional[np.ndarray] = None
    p_by_t: Optional[np.ndarray] = None
    upper_mass_by_t: Optional[np.ndarray] = None
    lower_mass_by_t: Optional[np.ndarray] = None
    metadata: Optional[dict] = None
