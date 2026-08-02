"""Import smoke tests for the modules `plottraces3.ipynb` was ported onto.

The notebook itself needs multi-hundred-MB pickles to run, so these tests only
exercise the rewired import graph — which is exactly what the port changed.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]


def test_alignsampling_exposes_the_epoch_alignment_helpers():
    """The four functions ported from caiman's common/analysis/alignsampling.py."""
    from code.twop import alignsampling

    for name in ("alignSampling", "alignAroundEpoch", "getSamplingVario",
                 "getSamplingFixed", "getSamplingQuantiles"):
        assert callable(getattr(alignsampling, name)), name


def test_alignsampling_reuses_the_common_imaging_primitives():
    """They must build on common/imaging.py, not on a second copy."""
    from code.common import imaging
    from code.twop import alignsampling

    assert alignsampling._alignAroundEpoch is imaging._alignAroundEpoch
    assert alignsampling._assignTraceLen is imaging._assignTraceLen


def test_classifier_modules_import():
    """Decoders section of the notebook."""
    from code.twop import classifiercreatelabels, classifyplayground

    assert callable(classifyplayground.test)
    assert callable(classifyplayground._runClassifier)
    assert callable(classifiercreatelabels.threeDifficultiesLabels)


def test_relogit_imports():
    """Used by `_plotCorr(use_logistic=True)` in the RT-correlation section."""
    from code.twop.relogit.relogit import relogit

    assert isinstance(relogit, type)


def test_colorMapFijiArr_returns_rgb_rows_in_unit_range():
    from code.common.clr import colorMapFijiArr, colorMapFireFiji

    arr = colorMapFijiArr()
    assert arr.ndim == 2 and arr.shape[1] == 3
    assert arr.min() >= 0 and arr.max() <= 1
    # colorMapFireFiji is built straight off it, so the two must stay in sync.
    assert np.allclose(colorMapFireFiji()(0.0)[:3], arr[0])
