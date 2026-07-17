'''Tests for the ``epoch_fraction_vlines`` marker in ``PlotTraces``.

These pin the one property the mid-sampling marker figure relies on: a line
requested at fraction ``f`` of a named epoch lands at that fraction of the
epoch's span in the plotted (frame) coordinates, and unmatched specs are
ignored rather than raising.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ..plottracesprocessor import PlotTraces

# Matches the Fig-2e normalized/concatenated layout the probe confirmed:
#   -0.1s Sampling (0,2) | Sampling (3,23) | Movement to Lateral Port (24,26)
EPOCHS_NAMES = ["-0.1s Sampling", "Sampling", "Movement to Lateral Port"]
EPOCHS_RANGES = [(0, 2), (3, 23), (24, 26)]
TRACE_LEN = 27
MARK_COLOR = "purple"  # distinct from the black epoch-boundary dashes


def _df():
    trace = np.arange(TRACE_LEN, dtype=float)
    return pd.DataFrame([{
        "TrialNumber": "Avg",
        "Name": "test",
        "traces_sets": {"neuronal": {"ALM": trace, "M2": -trace}},
        "epochs_names": EPOCHS_NAMES,
        "epochs_ranges": EPOCHS_RANGES,
    }])


def _run(epoch_fraction_vlines):
    fig, ax = plt.subplots()
    PlotTraces(is_avg_trc=True, x_label="x", y_label="y",
               draw_legend=False, show_plots=False,
               getAx=lambda plot_id: ax,
               epoch_fraction_vlines=epoch_fraction_vlines)._plotTraces(_df())
    return fig, ax


def _vertical_lines(ax, color=None):
    """axvline draws a Line2D with 2 equal x's; traces have TRACE_LEN points."""
    out = []
    for ln in ax.lines:
        xd = ln.get_xdata()
        if len(xd) == 2 and xd[0] == xd[1]:
            if color is None or ln.get_color() == color:
                out.append(float(xd[0]))
    return out


@pytest.mark.parametrize("fraction, expected_x", [
    (0.0, 3 + 1),          # start of Sampling, +1 for start_at_one
    (0.5, 3 + 10 + 1),     # midpoint of the (3, 23) span -> 13, +1 -> 14
    (1.0, 23 + 1),
])
def test_marker_lands_at_the_fraction_of_the_named_epoch(fraction, expected_x):
    _fig, ax = _run([{"epoch": "Sampling", "fraction": fraction,
                      "color": MARK_COLOR}])
    xs = _vertical_lines(ax, color=MARK_COLOR)
    assert xs == [expected_x]
    plt.close("all")


def test_default_fraction_is_the_midpoint():
    _fig, ax = _run([{"epoch": "Sampling", "color": MARK_COLOR}])
    assert _vertical_lines(ax, color=MARK_COLOR) == [14.0]
    plt.close("all")


def test_start_at_one_false_drops_the_offset():
    fig, ax = plt.subplots()
    PlotTraces(is_avg_trc=True, x_label="x", y_label="y", draw_legend=False,
               show_plots=False, start_at_one=False,
               getAx=lambda plot_id: ax,
               epoch_fraction_vlines=[{"epoch": "Sampling",
                                       "color": MARK_COLOR}]
               )._plotTraces(_df())
    assert _vertical_lines(ax, color=MARK_COLOR) == [13.0]
    plt.close("all")


def test_unknown_epoch_is_skipped_not_raised():
    _fig, ax = _run([{"epoch": "Nope", "fraction": 0.5, "color": MARK_COLOR}])
    assert _vertical_lines(ax, color=MARK_COLOR) == []
    plt.close("all")


def test_none_spec_adds_no_extra_lines():
    '''Only the two epoch-boundary dashes (starts of epochs 2 and 3) remain.'''
    _fig, ax = _run(None)
    assert sorted(_vertical_lines(ax)) == [4.0, 25.0]
    plt.close("all")


def test_marker_is_additive_to_the_epoch_boundary_lines():
    _fig, ax = _run([{"epoch": "Sampling", "fraction": 0.5,
                      "color": MARK_COLOR}])
    # The two black boundary dashes still there, plus our one marker.
    assert sorted(_vertical_lines(ax)) == [4.0, 14.0, 25.0]
    assert _vertical_lines(ax, color=MARK_COLOR) == [14.0]
    plt.close("all")


def test_multiple_markers_can_be_drawn():
    _fig, ax = _run([{"epoch": "Sampling", "fraction": 0.25, "color": MARK_COLOR},
                     {"epoch": "Sampling", "fraction": 0.75, "color": MARK_COLOR}])
    assert sorted(_vertical_lines(ax, color=MARK_COLOR)) == [9.0, 19.0]
    plt.close("all")


def test_styling_keys_are_forwarded_to_axvline():
    fig, ax = plt.subplots()
    PlotTraces(is_avg_trc=True, x_label="x", y_label="y", draw_legend=False,
               show_plots=False, getAx=lambda plot_id: ax,
               epoch_fraction_vlines=[{"epoch": "Sampling", "color": MARK_COLOR,
                                       "linestyle": "dotted", "linewidth": 3}]
               )._plotTraces(_df())
    marker = [ln for ln in ax.lines
              if len(ln.get_xdata()) == 2
              and ln.get_xdata()[0] == ln.get_xdata()[1]
              and ln.get_color() == MARK_COLOR][0]
    assert marker.get_linestyle() == ":"
    assert marker.get_linewidth() == 3
    plt.close("all")
