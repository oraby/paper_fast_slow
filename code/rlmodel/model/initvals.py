from dataclasses import dataclass, asdict
from typing import NamedTuple
import os


class InitVal(NamedTuple):
    Min : float
    Max : float
    Default : float

@dataclass
class InitVals:
    """Default (Min, Max, Default) tuples for every fittable DDM parameter.

    Scale-axis pairing — the DDM's BOUND and NOISE_SIGMA are
    near-degenerate in the loss landscape (doubling one and doubling
    the other gives ~equivalent observable behavior), so **exactly one
    of the pair is fittable at a time** and the other is frozen. The
    GUI's "Scale-How" dropdown (and the ``--scale-bound`` CLI flag)
    selects which axis is active; ``fit.simulateDDM`` calls
    ``InitVals.override(...)`` at fit time to clamp the inactive one.

    The four fields below come as two pairs:

    - ``NOISE_SIGMA`` / ``_NOISE_FIXED`` — the noise axis. ``NOISE_SIGMA``
      is the **fittable** range (used when Scale-How=Noise, the
      default); ``_NOISE_FIXED`` is the frozen counterpart (used when
      Scale-How=Bound — i.e. ``--scale-bound``).
    - ``BOUND`` / ``_BOUND_FIXED`` — the bound axis. ``BOUND`` is the
      **fittable** range (used when Scale-How=Bound); ``_BOUND_FIXED``
      is the frozen counterpart (used when Scale-How=Noise, the
      default).

    Underscore prefix marks "this is the frozen counterpart of the
    paired axis"; it does NOT mean private-do-not-touch. The GUI
    exposes both as visible sliders so the user can tweak the frozen
    value as well.
    """
    DRIFT_COEF : InitVal        = InitVal(0, 20, 1)
    NOISE_SIGMA : InitVal       = InitVal(0, 5, 1.5)
    _NOISE_FIXED : InitVal      = InitVal(1.0, 1.0, 1.0)
    BOUND : InitVal             = InitVal(0.3, 5.0, 1.0)
    _BOUND_FIXED : InitVal      = InitVal(1.0, 1.0, 1.0)
    BIAS_COEF : InitVal         = InitVal(0, 1, .95)
    ALPHA : InitVal             = InitVal(0, 1, .3)
    BETA : InitVal              = InitVal(0, 1, .3)
    NON_DECISION_TIME : InitVal = InitVal(0, 1, .3)
    Q_VAL_OFFSET : InitVal      = InitVal(-1, 1, 0)
    # MLE-only contamination / lapse mixture. λ ∈ [0, 1). Per-trial likelihood
    # becomes (1-λ)·L_DDM + λ/(2·T_max). Default initial 0.02 floors per-trial
    # loglik at log(0.02/(2·T_max)) ≈ -5.8 instead of LOGLIK_FLOOR's -691,
    # so DE isn't dominated by a handful of anticipations / fast guesses.
    # Disable for an experiment via `--init-val LAPSE_RATE=0,0,0`. Only used
    # by the MLE path — the chisq fit silently ignores it.
    LAPSE_RATE : InitVal        = InitVal(0.0, 0.1, 0.02)

    def __init__(self):
        self._extras = {}
        self._removed = {}

    def override(self, name, init_val):
        """Replace the (Min, Max, Default) tuple for a parameter.

        Routes through ``_extras`` because ``toDict`` does
        ``asdict(self) | self._extras`` (right side wins). That means the
        override takes precedence over the dataclass default without us
        having to mutate the class attribute. Name is case-insensitive but
        stored uppercase to match the rest of the codebase's convention.
        """
        if not isinstance(init_val, InitVal):
            raise TypeError(
                f"override expects an InitVal NamedTuple; got "
                f"{type(init_val).__name__}")
        self._extras[str(name).upper()] = init_val

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
T_dur = 4.8
NUM_CPUS = os.cpu_count()


# Default + valid range for ``mle_terminal_c`` (the threshold C used to
# partition residual interior mass at t = T_max). Lives outside the
# ``InitVals`` dataclass because terminal_c is a model-config knob, not a
# DE-fittable parameter. C = 1.0 is allowed and is the "survival-only"
# setting — it routes the entire interior mass to the no-decision bucket,
# so a no-choice trial's likelihood is the full survival mass at T_max.
MLE_TERMINAL_C = InitVal(Min=0.0, Max=1.0, Default=1.0)