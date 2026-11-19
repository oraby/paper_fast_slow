import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[3]))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.pipeline.behavior import (  # noqa: E402
    CountContPrevOutcome,
    _concat_preserving_empty_columns,
)


def test_concat_preserving_empty_columns_avoids_all_na_futurewarning():
    dfs = [
        pd.DataFrame({"a": [1], "b": [np.nan]}),
        pd.DataFrame({"a": [2], "b": [3.0]}),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out = _concat_preserving_empty_columns(dfs)

    assert out["a"].tolist() == [1, 2]
    assert out["b"].tolist()[1] == 3.0
    assert list(out.columns) == ["a", "b"]


def test_count_prev_outcome_session_concat_avoids_all_na_futurewarning():
    df = pd.DataFrame({
        "Name": ["S1", "S1", "S1", "S1"],
        "SessionNum": [1, 1, 1, 1],
        "TrialNumber": [1, 1, 2, 2],
        "ChoiceCorrect": [1.0, 1.0, np.nan, np.nan],
        "ChoiceLeft": [1.0, 1.0, np.nan, np.nan],
        "LeftRewarded": [1.0, 1.0, 0.0, 0.0],
        "DV": [0.5, 0.5, np.nan, np.nan],
        "calcStimulusTime": [0.2, 0.2, np.nan, np.nan],
    })

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out = CountContPrevOutcome()._processSession(df)

    assert "PrevCalcStimulusTime3" in out.columns
    assert out["PrevOutcomeCount"].tolist() == [0, 0, 1, 1]
