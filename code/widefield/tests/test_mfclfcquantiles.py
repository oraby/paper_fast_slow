'''Tests for ``widefield/mfclfcquantiles.py``.

These pin the behaviour the MFC/LFC-vs-quantile figures rely on: that the
mid-sampling window really is centred on each trial's own sampling midpoint,
and -- the property that makes the Fast/Typical/Slow comparison meaningful --
that its width in samples does not depend on how long the trial's sampling
epoch was.
'''
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ..mfclfcquantiles import (AlignTraceWithinEpoch, SamplingAnchor,
                               getTrialsMeanActivity)

ACQ_RATE = 10.0  # Hz. Keeps width_sec -> sample counts exact in these tests.
TRACE_LEN = 200


def _row(epoch="Sampling", trace_start_idx=100, trace_end_idx=120,
         trial_num=1, acq_sampling_rate=ACQ_RATE, trace=None, **overrides):
    if trace is None:
        trace = np.arange(TRACE_LEN, dtype=float)
    row = {"epoch": epoch,
           "trace_start_idx": trace_start_idx,
           "trace_end_idx": trace_end_idx,
           "acq_sampling_rate": acq_sampling_rate,
           "traces_sets": {"neuronal": {"M2_left": trace}},
           "sole_owner": False,
           "Name": "WF1",
           "ShortName": "WF1_M1",
           "TrialNumber": trial_num,
           "SamplingType": "RT",
           "Stimulus": "Stim",
           "calcStimulusTime": (trace_end_idx - trace_start_idx)
                               /acq_sampling_rate,
           "ChoiceCorrect": 1.0,
           "ChoiceLeft": 1.0,
           "DVstr": "0",
           "quantile_idx": 1}
    row.update(overrides)
    return row


def _df(*rows):
    return pd.DataFrame(list(rows))


def _window(res_row):
    return int(res_row.trace_start_idx), int(res_row.trace_end_idx)


def _width(res_row):
    return int(res_row.trace_end_idx) - int(res_row.trace_start_idx) + 1


def test_window_is_centred_on_the_epoch_midpoint():
    # Epoch [100, 120] -> midpoint 110. 0.3s at 10Hz -> 3 samples.
    proc = AlignTraceWithinEpoch(["Sampling"], fraction=0.5, width_sec=0.3)
    res = proc.process(_df(_row(trace_start_idx=100, trace_end_idx=120)))

    assert len(res) == 1
    assert _window(res.iloc[0]) == (109, 111)
    assert res.iloc[0].org_mid_idx == 110


def test_width_is_constant_across_epoch_durations():
    '''The Fast/Typical/Slow quantiles differ in sampling duration, so the
    window must not widen with the epoch or the comparison is confounded.'''
    proc = AlignTraceWithinEpoch(["Sampling"], fraction=0.5, width_sec=0.3)
    short = _row(trace_start_idx=100, trace_end_idx=110, trial_num=1)  # 11
    long_ = _row(trace_start_idx=100, trace_end_idx=180, trial_num=2)  # 81
    res = proc.process(_df(short, long_))

    assert _width(res.iloc[0]) == _width(res.iloc[1]) == 3
    # ... and each is still centred on its own midpoint, which differs.
    assert _window(res.iloc[0]) == (104, 106)
    assert _window(res.iloc[1]) == (139, 141)


@pytest.mark.parametrize("width_sec, expected_width", [
    (0.1, 1), (0.3, 3), (0.5, 5), (1.0, 10),
])
def test_width_follows_width_sec_times_acq_rate(width_sec, expected_width):
    proc = AlignTraceWithinEpoch(["Sampling"], fraction=0.5,
                                 width_sec=width_sec)
    res = proc.process(_df(_row()))
    assert _width(res.iloc[0]) == expected_width


def test_even_width_takes_the_extra_sample_before_the_anchor():
    # 0.4s at 10Hz -> 4 samples, which cannot be symmetric around one sample.
    proc = AlignTraceWithinEpoch(["Sampling"], fraction=0.5, width_sec=0.4)
    res = proc.process(_df(_row(trace_start_idx=100, trace_end_idx=120)))
    assert _window(res.iloc[0]) == (108, 111)  # anchor 110: 2 before, 1 after


def test_a_sub_sample_width_still_keeps_one_sample():
    proc = AlignTraceWithinEpoch(["Sampling"], fraction=0.5, width_sec=0.01)
    res = proc.process(_df(_row()))
    assert _width(res.iloc[0]) == 1


@pytest.mark.parametrize("fraction, expected_anchor", [
    (0.0, 100), (0.25, 105), (0.5, 110), (1.0, 120),
])
def test_fraction_anchors_along_the_epoch(fraction, expected_anchor):
    proc = AlignTraceWithinEpoch(["Sampling"], fraction=fraction,
                                 width_sec=0.3)
    res = proc.process(_df(_row(trace_start_idx=100, trace_end_idx=120)))
    assert res.iloc[0].org_mid_idx == expected_anchor


def test_a_too_short_epoch_spills_rather_than_being_clipped():
    '''Documents a deliberate choice: keeping the width fixed matters more than
    staying inside the epoch, and matches how the END window is cut. Only
    ~0.2% of trials sample for less than the window width.'''
    proc = AlignTraceWithinEpoch(["Sampling"], fraction=0.5, width_sec=0.5)
    # Epoch of 3 samples, asked for a 5-sample window.
    res = proc.process(_df(_row(trace_start_idx=100, trace_end_idx=102)))
    assert _window(res.iloc[0]) == (99, 103)


def test_window_never_starts_before_the_trace():
    proc = AlignTraceWithinEpoch(["Sampling"], fraction=0.0, width_sec=0.5)
    res = proc.process(_df(_row(trace_start_idx=0, trace_end_idx=20)))
    assert res.iloc[0].trace_start_idx == 0


def test_other_epochs_are_dropped():
    proc = AlignTraceWithinEpoch(["Sampling"], fraction=0.5, width_sec=0.3)
    res = proc.process(_df(_row(epoch="-0.1s Sampling"),
                           _row(epoch="Sampling"),
                           _row(epoch="Movement to Lateral Port")))
    assert len(res) == 1
    assert res.iloc[0].epoch == "Sampling"


def test_use_epoch_name_renames_the_kept_row():
    proc = AlignTraceWithinEpoch(["Sampling"], fraction=0.5, width_sec=0.3,
                                 use_epoch_name="Mid Sampling")
    res = proc.process(_df(_row()))
    assert res.iloc[0].epoch == "Mid Sampling"


def test_the_source_row_is_not_mutated():
    df = _df(_row(trace_start_idx=100, trace_end_idx=120))
    AlignTraceWithinEpoch(["Sampling"], fraction=0.5,
                          width_sec=0.3).process(df)
    assert (df.iloc[0].trace_start_idx, df.iloc[0].trace_end_idx) == (100, 120)


@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_fraction_outside_the_epoch_is_rejected(fraction):
    with pytest.raises(AssertionError):
        AlignTraceWithinEpoch(["Sampling"], fraction=fraction, width_sec=0.3)


def test_non_positive_width_is_rejected():
    with pytest.raises(AssertionError):
        AlignTraceWithinEpoch(["Sampling"], fraction=0.5, width_sec=0)


def test_end_anchor_keeps_the_published_figure_name():
    '''END pre-dates MID; its .svg name must not gain a suffix.'''
    assert SamplingAnchor.END.fig_suffix == ""
    assert SamplingAnchor.MID.fig_suffix == "_mid_sampling"
    assert SamplingAnchor("end") is SamplingAnchor.END
    assert SamplingAnchor("mid") is SamplingAnchor.MID


def test_getTrialsMeanActivity_averages_only_the_window():
    # trace is arange(200); window [2, 4] -> mean of 2, 3, 4.
    res = getTrialsMeanActivity(_df(_row(trace_start_idx=2, trace_end_idx=4)),
                                include_raw=True)

    assert len(res) == 1
    assert res.iloc[0].mean_activity == pytest.approx(3.0)
    assert res.iloc[0].BrainRegion == "M2"
    assert res.iloc[0].Hemisphere == "left"
    assert res.iloc[0].quantile_idx == 1
    np.testing.assert_array_equal(res.iloc[0].raw, [2.0, 3.0, 4.0])


def test_getTrialsMeanActivity_reads_unified_region_names():
    '''After UnifyBilateralRegionsTraces the traces lose their _left/_right
    suffix, and the hemisphere is reported as bilateral.'''
    row = _row(trace_start_idx=2, trace_end_idx=4)
    row["traces_sets"] = {"neuronal": {"M2": np.arange(TRACE_LEN,
                                                       dtype=float)}}
    res = getTrialsMeanActivity(_df(row), include_raw=False)

    assert res.iloc[0].BrainRegion == "M2"
    assert res.iloc[0].Hemisphere == "Bi"
    assert "raw" not in res.columns


def test_getTrialsMeanActivity_drops_trials_without_a_choice():
    res = getTrialsMeanActivity(_df(_row(ChoiceCorrect=np.nan)),
                                include_raw=False)
    assert len(res) == 0
