"""Shortest interval holding a given fraction of the data -- the "shorth".

``TwoPLoad.ipynb``'s single-neuron plots use it to show how tightly a neuron's
peak times cluster across trials: given the frame index of each trial's peak,
:func:`shorth` returns the narrowest window that contains ``fraction`` of them
(``DATA_FRACTION = 0.8`` there), which the plot shades and reports as a
percentage of the trace length. A neuron that fires at the same point in
sampling on most trials has a short shorth; one whose peak wanders has a long
one.

Provenance: recovered from the ancestor project, where it was defined inline in
``OneDrive/caiman/TwoP/again/ROC_tests_new15_09_23_local.ipynb`` -- the notebook
``TwoPLoad.ipynb``'s sequence-characterisation cell was copied from -- and never
came across with it. The logic is unchanged; two initial assignments that were
always overwritten, and commented-out prints, were dropped.
"""
from __future__ import annotations

import numpy as np


def shorth(data, fraction, return_range=False):
    """Length of the shortest interval containing ``fraction`` of ``data``.

    The window spans ``round(len(data) * fraction)`` consecutive sorted values
    (never fewer than two), and the length is measured end-to-end in the units
    of ``data``. On ties the earliest window wins, as ``argmin`` does.

    Parameters
    ----------
    data : array-like
        Values to cover -- in the notebook, per-trial peak frame indices.
    fraction : float
        Share of the data the interval must hold, strictly between 0 and 1.
    return_range : bool
        Also return the interval's ``(start, end)`` values.

    Returns
    -------
    length, or ``(length, (start, end))`` when ``return_range`` is set.
    """
    assert fraction > 0
    assert fraction < 1
    total_count = len(data)
    data = np.sort(data)

    fraction_offset_idx = int(np.round(total_count * fraction)) - 1
    fraction_offset_idx = max(1, fraction_offset_idx)
    fraction_offset_idx = min(fraction_offset_idx, total_count)

    smallest_diff = data[fraction_offset_idx:] - data[:-fraction_offset_idx]
    smallest_diff_idx = smallest_diff.argmin()
    smallest_diff_start = data[smallest_diff_idx]
    smallest_diff_end = data[smallest_diff_idx + fraction_offset_idx]

    shortest_distance = smallest_diff_end - smallest_diff_start
    if return_range:
        return shortest_distance, (smallest_diff_start, smallest_diff_end)
    return shortest_distance
