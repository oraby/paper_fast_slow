'''Tests for the example single-neuron traces (Figure 4D, right).

What these pin: each window is exactly the frames asked for, on a time axis in
seconds from the session's frame rate; the traces stack in order without
overlapping; only the time axis is drawn; the file name the figure is saved
to; and that a missing session, neuron or frame range raises rather than
drawing something else.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from .. import exampletraces as et

FS = 10.


@pytest.fixture(autouse=True)
def closeFigures():
    yield
    plt.close("all")


def frame(n_frames=100):
    traces = {"neuronal": {7: np.arange(n_frames, dtype=float),
                           9: np.arange(n_frames, dtype=float) * 2,
                           11: np.sin(np.arange(n_frames))}}
    # Two rows of one session share the same traces dict, as in the real data.
    return pd.DataFrame([dict(ShortName="sA", acq_sampling_rate=FS,
                              traces_sets=traces) for _ in range(2)] +
                        [dict(ShortName="sB", acq_sampling_rate=30.,
                              traces_sets={"neuronal": {}})])


NEURONS = ((7, 10, 19, "r"), (9, 0, 4, "g"), (11, 50, 99, "b"))


def test_each_window_is_the_frames_asked_for_in_seconds():
    windows = et.exampleWindows(frame(), "sA", NEURONS)
    assert [w[0] for w in windows] == [7, 9, 11]
    neuron_id, t, dff, colour = windows[0]
    np.testing.assert_array_equal(dff, np.arange(10, 20))
    np.testing.assert_allclose(t, np.arange(10) / FS)
    assert colour == "r"
    assert len(windows[2][2]) == 50


def test_the_traces_stack_in_order_without_overlapping():
    fig = et.plotExampleTraces(frame(), "sA", NEURONS, show=False)
    lines = fig.axes[0].get_lines()
    assert [ln.get_label() for ln in lines] == ["Neuron 7", "Neuron 9",
                                                "Neuron 11"]
    lows = [ln.get_ydata().min() for ln in lines]
    highs = [ln.get_ydata().max() for ln in lines]
    assert lows[0] >= highs[1] and lows[1] >= highs[2]


def test_only_the_time_axis_is_drawn():
    ax = et.plotExampleTraces(frame(), "sA", NEURONS, show=False).axes[0]
    assert ax.spines["bottom"].get_visible()
    assert not any(ax.spines[s].get_visible() for s in ("top", "left", "right"))
    assert list(ax.get_yticks()) == []
    assert ax.get_xlabel() == "Time (s)"


def test_the_figure_saves_under_example_traces(tmp_path):
    et.plotExampleTraces(frame(), "sA", NEURONS, save_figs=True,
                         fig_save_prefix=tmp_path, show=False)
    expected = tmp_path / "ExampleTraces" / "sA_neurons_7_9_11.svg"
    assert expected.is_file()
    assert et.savePath(tmp_path, "sA", NEURONS) == expected


def test_nothing_is_written_without_save_figs(tmp_path):
    et.plotExampleTraces(frame(), "sA", NEURONS, fig_save_prefix=tmp_path,
                         show=False)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("session, neurons, error", [
    ("sC", NEURONS, KeyError),                 # no such session
    ("sA", ((8, 0, 5, "r"),), KeyError),       # no such neuron
    ("sA", ((7, 90, 100, "r"),), IndexError),  # past the last frame
    ("sA", ((7, 5, 4, "r"),), IndexError),     # last before first
])
def test_a_missing_session_neuron_or_range_raises(session, neurons, error):
    with pytest.raises(error):
        et.exampleWindows(frame(), session, neurons)


def test_the_published_example_is_three_neurons_of_one_session():
    assert et.EXAMPLE_SESSION == "GP4_23_s3_L50_D250_M2"
    assert [n[0] for n in et.EXAMPLE_NEURONS] == [37, 33, 25]
