'''Tests for the S10A–B example neurons and their peak/AUC calculation.

What these pin: the AUC threshold at half the trace's range; the rising
segments (strictly rising, start before the rise, end exclusive when
integrated); when a trial starts on a decay, when a later rise rescues it and
where its peak then moves; that decaying trials stay in the traces panel but
leave the correlations; the two alignments' time axes; the file name; and that
the examples are 4K (left)'s neurons.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ...common.definitions import BrainRegion
from .. import rtcorrexamples as rx
from ..rtcorrpanels import EXAMPLES_4K_LEFT

MFC = int(BrainRegion.M2_Bi)
LFC = int(BrainRegion.ALM_Bi)
FS = 30.
KEY = "smoothed2_"


@pytest.fixture(autouse=True)
def closeFigures():
    yield
    plt.close("all")


def test_the_auc_sums_the_samples_above_half_the_range():
    # min 0, max 4: threshold 2, so [2, 3, 4] are integrated.
    assert rx.s10TraceAUC([0, 1, 2, 3, 4]) == pytest.approx(6.)
    # Samples above the threshold are taken as one sequence, gaps closed.
    assert rx.s10TraceAUC([4, 0, 4]) == pytest.approx(4.)


def test_rising_segments_are_strict_runs_from_the_sample_before():
    trace = [3, 1, 2, 4, 4, 5, 0, 1]
    np.testing.assert_array_equal(rx.risingSegments(trace),
                                  [[1, 3], [4, 5], [6, 7]])
    assert rx.risingSegments([3, 2, 1]).shape == (0, 2)


def test_the_auc_over_segments_leaves_out_each_end_sample():
    trace = np.array([10, 6, 2, 0, 1, 3, 4, 1.])
    assert rx.s10TraceAUC(trace, [[3, 6]]) == pytest.approx(2.5)


def test_a_maximum_after_the_second_frame_is_not_a_decay():
    decaying, new_max, segments = rx.startsOnDecay([0, 1, 5, 2], 2)
    assert not decaying and new_max is None and not len(segments)


@pytest.mark.parametrize("trace", [[10, 6, 2, 0], [10, 6, 2, 0, 1, 0]])
def test_a_trace_falling_from_its_start_with_no_big_rise_decays(trace):
    decaying, new_max, _ = rx.startsOnDecay(trace, 0)
    assert decaying and new_max is None


def test_a_later_rise_of_a_fifth_of_the_range_moves_the_peak():
    trace = np.array([10, 6, 2, 0, 1, 3, 4, 1.])
    decaying, new_max, _ = rx.startsOnDecay(trace, 0)
    assert not decaying and new_max == 6
    decaying, peak_sec, auc = rx.s10PeakAUC(trace, FS)
    assert not decaying
    assert peak_sec == pytest.approx(6 / FS - rx.TIME_BEFORE_SAMPLING)
    assert auc == pytest.approx(2.5)


def test_an_ordinary_trial_keeps_its_maximum_and_threshold_auc():
    trace = np.array([0, 1, 2, 3, 4, 1.])
    decaying, peak_sec, auc = rx.s10PeakAUC(trace, FS)
    assert not decaying
    assert peak_sec == pytest.approx(4 / FS - rx.TIME_BEFORE_SAMPLING)
    assert auc == pytest.approx(6.)


def neuronTable(session="s1", trace_id=7, br=MFC, n_trials=10, n_decaying=2):
    '''A neuron whose peak moves with its duration, plus ``n_decaying`` trials
    that only fall, and another neuron that must be ignored.'''
    rows = []
    for i in range(n_trials):
        dur = 0.5 + 0.2 * i
        n = int((dur + 0.2) * FS)
        if i < n_decaying:
            trace = np.linspace(5, 0, n)
        else:
            trace = np.exp(-((np.arange(n) - dur * FS) / 5.) ** 2)
        rows.append({"BrainRegion": br, "ShortName": session, "TrialNumber": i,
                     "trial_dur_sec": dur, "trace_id": trace_id,
                     f"{KEY}trace": trace})
    other = dict(rows[-1], trace_id=trace_id + 1)
    return pd.DataFrame(rows + [other])


def samplingFrames(session="s1"):
    return pd.DataFrame({"ShortName": [session, session],
                         "acq_sampling_rate": [FS, FS]})


def test_the_trial_table_is_one_row_per_trial_of_the_neuron():
    trials = rx.exampleTrials(neuronTable(), "s1", 7, KEY, FS)
    assert len(trials) == 10
    assert trials.starts_on_decay.sum() == 2
    assert {"TrialNumber", "trial_dur_sec", "trace", "colour",
            "starts_on_decay", "peak_sec", "auc"} <= set(trials.columns)


def test_decaying_trials_leave_the_correlations():
    trials = rx.exampleTrials(neuronTable(), "s1", 7, KEY, FS)
    stats = rx.exampleCorrelations(trials)
    # The kept peaks sit at the trial's end, so they track its duration.
    assert stats["peak"]["pearson_corr"] > 0.99
    assert stats["peak"]["slope"] == pytest.approx(45, abs=1)


def test_a_missing_neuron_or_session_raises():
    with pytest.raises(KeyError):
        rx.exampleTrials(neuronTable(), "s1", 99, KEY, FS)
    with pytest.raises(KeyError):
        rx.sessionAcqRate(samplingFrames(), "s2")


def test_the_traces_panel_has_every_trial_and_the_scatters_the_kept_ones():
    fig = rx.plotS10Example(neuronTable(), samplingFrames(), "s1", 7,
                            panel="S10A", aligned="sampling", key_prefix=KEY,
                            show=False)
    ax_tr, ax_pk, ax_auc = fig.axes
    assert sum(len(ln.get_xdata()) > 2 for ln in ax_tr.get_lines()) == 10
    for ax in (ax_pk, ax_auc):
        assert len(ax.collections[0].get_offsets()) == 8
        assert "$r$=" in ax.get_title()
    assert ax_tr.get_xlim() == pytest.approx((-0.1, 1.1))


def test_movement_alignment_ends_each_trace_just_after_movement_onset():
    t = rx.traceTimes(31, FS, "movement")
    assert t[-1] == pytest.approx(rx.TIME_AFTER_MOVEMENT)
    assert rx.traceTimes(31, FS, "sampling")[0] == pytest.approx(-0.1)
    with pytest.raises(ValueError):
        rx.traceTimes(31, FS, "reward")
    fig = rx.plotS10Example(neuronTable(), samplingFrames(), "s1", 7,
                            panel="S10B", aligned="movement", key_prefix=KEY,
                            show=False)
    assert fig.axes[0].get_xlim()[1] == pytest.approx(rx.TIME_AFTER_MOVEMENT)


def test_the_example_saves_under_its_region(tmp_path):
    rx.plotS10Example(neuronTable(br=LFC), samplingFrames(), "s1", 7,
                      panel="S10B", aligned="movement", key_prefix=KEY,
                      save_figs=True, fig_save_prefix=tmp_path, show=False)
    expected = (tmp_path / "SingleTraces" / "rt_corr_examples" / "LFC" /
                "s1_7_S10B.svg")
    assert expected.is_file()
    assert rx.savePath(tmp_path, LFC, "s1", 7, "S10B") == expected


def test_nothing_is_written_without_save_figs(tmp_path):
    rx.plotS10Example(neuronTable(), samplingFrames(), "s1", 7, panel="S10A",
                      aligned="sampling", key_prefix=KEY,
                      fig_save_prefix=tmp_path, show=False)
    assert not any(tmp_path.iterdir())


def test_the_examples_are_the_4k_left_neurons():
    assert [p for p, *_ in rx.EXAMPLES_S10] == ["S10A", "S10B"]
    assert ([(s, t) for _, s, t, _ in rx.EXAMPLES_S10] ==
            [(s, t) for _, s, t in EXAMPLES_4K_LEFT])
    assert [a for *_, a in rx.EXAMPLES_S10] == ["sampling", "movement"]


@pytest.mark.parametrize("aligned, vertical", [("sampling", True),
                                               ("movement", False)])
def test_the_traces_panel_has_a_fast_red_to_slow_yellow_key(aligned, vertical):
    fig = rx.plotS10Example(neuronTable(), samplingFrames(), "s1", 7,
                            panel="S10A", aligned=aligned, key_prefix=KEY,
                            show=False)
    (cax,) = fig.axes[0].child_axes
    assert {t.get_text() for t in cax.texts} == {"Fast", "Slow"}
    img = cax.images[0]
    rgba = img.cmap(img.norm(img.get_array()))
    first = rgba[0, 0]
    last = rgba[-1, 0] if vertical else rgba[0, -1]
    # Fast is at the first row (top) or first column (left): red.
    np.testing.assert_allclose(first[:3], [1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(last[:3], [1, 1, 0], atol=1e-6)
    assert (img.get_array().shape[0] > 1) == vertical
