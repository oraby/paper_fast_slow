'''Tests for the z-scored sampling cut (``twop/zscoredsampling.py``).

What these pin: which trials survive the decision-time filter, and that the
cut runs once per session with the margins and normalisation asked for (the
cut itself is ``common.imaging.getSamplingEpoch``'s, stubbed here).
'''
from __future__ import annotations

import pandas as pd

from .. import zscoredsampling as zs
from ...pipeline.tracesnormalize import NormalizeZScore


def _rows(trial, epochs, time=1.0):
    return [dict(Name="m", Date="d", SessionNum=1, TrialNumber=trial,
                 epoch=e, calcStimulusTime=time) for e in epochs]


def test_drop_decision_time_trials():
    df = pd.DataFrame(
        _rows(1, ["Sampling", "Movement to Lateral Port"]) +       # kept
        _rows(2, ["Sampling", "Decision Time",
                  "Movement to Lateral Port"]) +                   # decision time
        _rows(3, ["Sampling"]) +                                   # never moved
        _rows(4, ["Sampling", "Movement to Lateral Port"], time=None))
    out = zs.dropDecisionTimeTrials(df)
    assert set(out.TrialNumber) == {1}
    assert list(out.index) == [0, 1]


def test_cut_runs_per_session_with_margins(monkeypatch):
    seen = []

    def fakeCut(df, **kw):
        seen.append((df.ShortName.iloc[0], kw))
        return df.assign(cut=True)

    monkeypatch.setattr(zs, "getSamplingEpoch", fakeCut)
    df = pd.DataFrame({"ShortName": ["b", "a", "b"], "x": [1, 2, 3]})
    out = zs.cutSamplingZScored(df, time_before=0.1, time_after=0.2)
    assert [s for s, _ in seen] == ["a", "b"]
    kw = seen[0][1]
    assert (kw["time_epoch_before"], kw["time_epoch_after"]) == (0.1, 0.2)
    assert isinstance(kw["normalization"], NormalizeZScore)
    assert kw["normalize_after_cutting"] and not kw["normalize_epoch_time"]
    assert out.cut.all() and list(out.index) == [0, 1, 2]
