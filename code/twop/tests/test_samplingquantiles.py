'''Tests for the sampling-time tercile labels (``twop/samplingquantiles.py``).

The tercile split itself is ``splitStimulusTimeByQuantile``'s and is stubbed
here; what these pin is the plumbing around it: every row gets its trial's
label, the one-row-per-trial layout keeps its index order, and the
several-rows-per-trial layout copies one decision to all of a trial's rows.
'''
from __future__ import annotations

import pandas as pd
import pytest

from .. import samplingquantiles as sq


def _stubSplit(df):
    '''Fast / typical / slow by rank of calcStimulusTime, all trials pooled.'''
    ranks = df.calcStimulusTime.rank(method="first")
    edges = len(df) / 3
    for q_idx in (1, 2, 3):
        mask = (ranks > (q_idx - 1) * edges) & (ranks <= q_idx * edges)
        yield q_idx, df[mask]


@pytest.fixture(autouse=True)
def stubSplit(monkeypatch):
    monkeypatch.setattr(sq, "splitStimulusTimeByQuantile", _stubSplit)


def _trials(times):
    return pd.DataFrame({"Name": "m", "Date": "d", "SessionNum": 1,
                         "TrialNumber": range(1, len(times) + 1),
                         "calcStimulusTime": times})


def test_one_row_per_trial_keeps_index_order():
    df = _trials([0.9, 0.1, 0.5, 0.2, 0.8, 0.6])
    out = sq.assignQuantileIdx(df)
    assert list(out.index) == list(df.index)
    assert dict(zip(out.TrialNumber, out.quantile_idx)) == {
        2: 1, 4: 1, 3: 2, 6: 2, 5: 3, 1: 3}


def test_one_row_per_trial_rejects_duplicate_index():
    df = _trials([0.1, 0.2, 0.3])
    df.index = [0, 0, 1]
    with pytest.raises(AssertionError):
        sq.assignQuantileIdx(df)


def test_by_trial_copies_the_label_to_every_epoch_row():
    trials = _trials([0.9, 0.1, 0.5])
    rows = []
    for _, trial in trials.iterrows():
        for state_id, start in ((2, 10), (1, 0)):   # deliberately unsorted
            rows.append(dict(trial, state_id=state_id, trace_start_idx=start))
    out = sq.assignQuantileIdxByTrial(pd.DataFrame(rows))
    assert len(out) == 6
    by_trial = out.groupby("TrialNumber").quantile_idx.unique()
    assert {k: list(v) for k, v in by_trial.items()} == {1: [3], 2: [1], 3: [2]}
    # Sorted by trial, then by where each epoch starts
    assert list(out.TrialNumber) == [1, 1, 2, 2, 3, 3]
    assert list(out.trace_start_idx) == [0, 10] * 3
    assert list(out.index) == list(range(6))


def test_by_trial_drops_rows_without_a_sampling_time():
    df = _trials([0.3, None, 0.1, 0.2])
    df["state_id"] = 1
    df["trace_start_idx"] = 0
    out = sq.assignQuantileIdxByTrial(df)
    assert 2 not in set(out.TrialNumber)
    assert len(out) == 3
