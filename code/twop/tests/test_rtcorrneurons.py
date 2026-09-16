'''Tests for the per-trial peak/area table (Figures 4K, S10A-C, S11 all read it).

Reproduced while extracting: the same table as the notebook cell produced.

Pinned here: peaks are reported in **seconds from the start of the cut trace**,
each trial is measured at three smoothing levels so later panels can pick one,
and quiet trials are kept unless ``only_active_trials`` says otherwise — which
is what makes "active in x% of trials" meaningful downstream.
'''
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ...common.definitions import BrainRegion
from ..rtcorrneurons import ACQ_RATE, analyzeNeuron

MFC = int(BrainRegion.M2_Bi)


def traces(peaks_at_sec, trial_dur_sec=1.0, n=60):
    """{(trial, duration, correct, difficulty): trace} with a peak per trial."""
    out = {}
    for i, peak in enumerate(peaks_at_sec):
        trace = np.zeros(n)
        trace[int(round(peak * ACQ_RATE))] = 10.0
        out[(i, trial_dur_sec, 1.0, "Hard")] = trace
    return out


def activeRow(active_trials):
    return {"active_trials_nums": list(active_trials)}


def analyse(peaks, active=(0, 1, 2, 3), only_active_trials=False, **kwargs):
    return analyzeNeuron(traces(peaks, **kwargs), activeRow(active),
                         acq_rate=ACQ_RATE, br=MFC, sess_id="s1", trace_id=7,
                         only_active_trials=only_active_trials)


def test_the_peak_is_reported_in_seconds():
    out = analyse([0.2, 0.5])
    assert list(out.max_pos_sec) == [0.2, 0.5]


def test_the_peak_amplitude_is_the_traces_maximum():
    out = analyse([0.2])
    assert out.max_amplitude.iloc[0] == 10.0


def test_every_trial_is_measured_at_three_smoothing_levels():
    out = analyse([0.2])
    for prefix in ("", "smoothed1_", "smoothed2_"):
        assert f"{prefix}max_pos_sec" in out.columns
        assert f"{prefix}auc" in out.columns
    # Smoothing spreads the spike, so its peak is lower than the raw one.
    assert out.smoothed2_max_amplitude.iloc[0] < out.max_amplitude.iloc[0]


def test_quiet_trials_are_kept_and_marked():
    out = analyse([0.2, 0.5, 0.8], active=(0, 2))
    assert list(out.is_active) == [True, False, True]
    assert len(out) == 3


def test_quiet_trials_can_be_dropped_instead():
    out = analyse([0.2, 0.5, 0.8], active=(0, 2), only_active_trials=True)
    assert list(out.TrialNumber) == [0, 2]


def test_the_trial_duration_travels_with_the_row():
    out = analyse([0.2, 0.5], trial_dur_sec=2.5)
    assert list(out.trial_dur_sec) == [2.5, 2.5]


def test_the_neuron_and_session_are_named_on_every_row():
    out = analyse([0.2, 0.5])
    assert set(out.BrainRegion) == {MFC}
    assert set(out.ShortName) == {"s1"}
    assert set(out.trace_id) == {7}


def test_the_identifying_columns_come_first():
    out = analyse([0.2])
    assert list(out.columns)[:7] == ["BrainRegion", "ShortName", "TrialNumber",
                                     "trial_dur_sec", "trace_id",
                                     "ChoiceCorrect", "DVstr"]


def test_a_flat_trial_has_no_area():
    flat = {(0, 1.0, 1.0, "Hard"): np.zeros(30)}
    out = analyzeNeuron(flat, activeRow([0]), acq_rate=ACQ_RATE, br=MFC,
                        sess_id="s1", trace_id=7, only_active_trials=False)
    assert out.auc.iloc[0] == 0


def test_a_taller_peak_gives_more_area():
    small = {(0, 1.0, 1.0, "Hard"): np.array([0., 1., 0.])}
    large = {(0, 1.0, 1.0, "Hard"): np.array([0., 10., 0.])}
    kwargs = dict(acq_rate=ACQ_RATE, br=MFC, sess_id="s1", trace_id=7,
                  only_active_trials=False)
    small_auc = analyzeNeuron(small, activeRow([0]), **kwargs).auc.iloc[0]
    large_auc = analyzeNeuron(large, activeRow([0]), **kwargs).auc.iloc[0]
    assert large_auc > small_auc
