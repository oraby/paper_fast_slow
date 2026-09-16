'''Tests for the trace-reduction helpers shared by the rt/activity panels.

These two functions sit under both the active-neuron counts (S11A/S11B) and
the per-neuron correlations (4K, S10A-C), so what they call a "rise" decides
what those panels measure.

Pinned here: a flat run counts as rising, the integration threshold is a share
of each *segment's* own height, the mask keeps everything up to the peak as
well as what clears the threshold, and the areas are concatenated before being
integrated once.
'''
from __future__ import annotations

import numpy as np
import pytest

from ..traceauc import TRACE_THRESH, getPosDeflections, getTraceAUC


def test_a_rise_is_reported_from_the_sample_before_it_to_its_top():
    trace = np.array([0., 0., 5., 10., 5., 0.])
    assert getPosDeflections(trace).tolist() == [[0, 3]]


def test_a_flat_opening_belongs_to_the_first_rise():
    """diff == 0 counts as rising, so the stretch starts at 0, not at 1."""
    trace = np.array([0., 0., 1., 4., 9., 4., 1., 0.])
    assert getPosDeflections(trace).tolist() == [[0, 4]]


def test_two_separated_rises_are_reported_separately():
    trace = np.array([0., 0., 1., 4., 9., 4., 1., 0., 0., 3., 6., 3., 0.])
    assert getPosDeflections(trace).tolist() == [[0, 4], [7, 10]]


def test_a_trace_that_only_falls_has_no_rises():
    assert len(getPosDeflections(np.array([9., 6., 3., 1., 0.]))) == 0


def test_a_flat_trace_is_one_long_rise():
    assert getPosDeflections(np.zeros(5)).tolist() == [[0, 4]]


def test_the_threshold_is_a_share_of_the_segments_own_height():
    trace = np.array([0., 0., 5., 10., 5., 0.])
    _, thresholds, _, _, _ = getTraceAUC(trace, getPosDeflections(trace))
    assert thresholds[0] == pytest.approx(TRACE_THRESH * 10.0)


def test_a_small_bump_keeps_its_own_threshold():
    """Otherwise the big rise would erase it."""
    trace = np.array([0., 0., 10., 0., 0., 1., 0.])
    _, thresholds, _, _, _ = getTraceAUC(trace, getPosDeflections(trace))
    assert [round(float(t), 3) for t in thresholds] == [8.0, 0.8]


def test_the_mask_keeps_the_climb_as_well_as_the_peak():
    trace = np.array([0., 0., 5., 10., 5., 0.])
    _, _, ranges, _, masks = getTraceAUC(trace, getPosDeflections(trace))
    # Samples 0-2 lead up to the peak at 3; 4 and 5 are below the threshold.
    assert masks[0].astype(int).tolist() == [1, 1, 1, 1, 0, 0]
    assert ranges[0] == (0, len(trace))


def test_the_area_is_the_masked_samples_integrated_once():
    trace = np.array([0., 0., 10., 0.])
    auc, _, _, _, _ = getTraceAUC(trace, getPosDeflections(trace))
    assert auc == pytest.approx(np.trapezoid([0., 0., 10.]))


def test_two_equal_rises_give_more_area_than_one():
    one = np.array([0., 0., 10., 0.])
    two = np.array([0., 0., 10., 0., 0., 10., 0.])
    auc_one = getTraceAUC(one, getPosDeflections(one))[0]
    auc_two = getTraceAUC(two, getPosDeflections(two))[0]
    assert auc_two > auc_one
    # Concatenated, not summed per rise: the join contributes a trapezoid too.
    assert auc_two == pytest.approx(15.0)
    assert auc_one == pytest.approx(5.0)


def test_the_peak_index_is_reported_in_trace_coordinates():
    trace = np.array([0., 0., 1., 4., 9., 4.])
    _, _, _, min_max_idxs, _ = getTraceAUC(trace, getPosDeflections(trace))
    assert min_max_idxs[0][1] == 4


def test_a_trace_with_no_rise_returns_nothing_at_all():
    trace = np.array([9., 6., 3., 1., 0.])
    auc, thresholds, ranges, min_max_idxs, masks = getTraceAUC(
        trace, getPosDeflections(trace))
    assert auc == 0
    assert thresholds == [] and ranges == [] and masks == []
