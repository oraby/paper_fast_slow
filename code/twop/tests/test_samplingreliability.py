'''Tests for the sampling-reliability helpers (``twop/samplingreliability.py``).

The alignment and the per-neuron test are other modules' (and tested there);
they are stubbed here. What these pin is how the preprocessing and feedback
notebooks call them: which frames come back, with which normalisation, and
that the reliability run passes the figure switch and collector options on.
'''
from __future__ import annotations

import pandas as pd

from .. import samplingreliability as sr
from ...pipeline.tracesnormalize import NoNormalization, NormalizeZScore


def test_align_sampling_frames(monkeypatch):
    seen = []

    def fakeAlign(*_df, normalization=None, normalize_epoch_time=None, **kw):
        seen.append((type(normalization), normalize_epoch_time,
                     kw["time_before_sampling"], kw["time_after_sampling"]))
        return pd.DataFrame({"x": [len(seen)]})

    monkeypatch.setattr(sr, "alignSampling", fakeAlign)
    monkeypatch.setattr(sr, "sessionTraceStats", lambda df: ("stats", int(df.x.iloc[0])))
    unnormed, normed, stats = sr.alignSamplingFrames(pd.DataFrame({"a": [1]}))
    assert seen == [(NormalizeZScore, False, 0.1, 0.1),
                    (NormalizeZScore, True, 0.1, 0.1),
                    (NoNormalization, False, 0.1, 0.1)]
    assert list(unnormed.epoch) == ["Sampling"]
    assert list(normed.epoch) == ["Sampling"]
    # The statistics come from the un-normalised call
    assert stats == ("stats", 3)


def test_collect_reliability_passes_switch_and_options(monkeypatch):
    seen = {}

    class FakeCollector:
        def __init__(self, decide, **kw):
            seen["init"] = (decide, kw)
            self.res_dict = {"trace_id": ["s_1"]}

        def track(self, *a):
            pass

        def frame(self):
            return pd.DataFrame(self.res_dict)

    def fakeLoop(df, **kw):
        seen["loop"] = kw

    monkeypatch.setattr(sr, "StdDistCollector", FakeCollector)
    monkeypatch.setattr(sr, "loopSessionsExampleNeurons", fakeLoop)
    res_dict, frame = sr.collectReliability("DF", "DECIDE", "max_firing_x",
                                            save_figs=True, fig_save_prefix="out",
                                            disable_first_neg_deflect=True)
    assert seen["init"] == ("DECIDE", {"disable_first_neg_deflect": True})
    assert seen["loop"]["pdf_str"] == "max_firing_x"
    assert seen["loop"]["save_figs"] is True
    assert seen["loop"]["fig_save_prefix"] == "out"
    assert seen["loop"]["random_fraction"] == 1 and seen["loop"]["seed"] == 1
    assert res_dict == {"trace_id": ["s_1"]}
    assert list(frame.trace_id) == ["s_1"]
