"""Tests for the shorth, including equivalence with the ancestor's version."""
import numpy as np
import pytest

from ..shorth import shorth


def _ancestorShorth(data, fraction, return_range=False):
    """Verbatim from caiman/TwoP/again/ROC_tests_new15_09_23_local.ipynb.

    Kept here only as the reference the extracted version must match.
    """
    assert fraction > 0
    assert fraction < 1
    total_count = len(data)
    data = np.sort(data)
    shortest_distance = data[-1] - data[0] + 1
    shortest_distance_vals = (-1, -1)
    fraction_offset_idx = int(np.round(total_count*fraction)) - 1
    fraction_offset_idx = max(1, fraction_offset_idx)
    fraction_offset_idx = min(fraction_offset_idx, total_count)

    smallest_diff = data[fraction_offset_idx:] - data[:-fraction_offset_idx]
    smallest_diff_idx = smallest_diff.argmin()
    smallest_diff_start = data[smallest_diff_idx]
    smallest_diff_end = data[smallest_diff_idx+fraction_offset_idx]
    shortest_distance = smallest_diff_end - smallest_diff_start
    shortest_distance_vals = (smallest_diff_start, smallest_diff_end)
    ret = shortest_distance
    if return_range:
        ret = ret, shortest_distance_vals
    return ret


def test_finds_the_tight_cluster_not_the_spread_values():
    # five of ten values sit in 10..14; the other half is spread out
    data = [0, 1, 2, 10, 11, 12, 13, 14, 15, 16]
    length, rng = shorth(data, fraction=0.5, return_range=True)
    assert length == 4
    assert rng == (10, 14)


def test_identical_values_give_zero_length():
    """The case TwoPLoad widens by one frame before shading."""
    length, rng = shorth([5, 5, 5, 5], fraction=0.5, return_range=True)
    assert length == 0
    assert rng == (5, 5)


def test_order_of_the_input_does_not_matter():
    data = [14, 0, 12, 2, 16, 10, 1, 13, 11, 15]
    assert shorth(data, 0.5, return_range=True) == (4, (10, 14))


def test_tiny_fraction_still_spans_two_values():
    # round(4 * 0.1) - 1 is negative; the window is clamped to two points
    assert shorth([0, 3, 4, 10], fraction=0.1, return_range=True) == (1, (3, 4))


def test_returns_a_scalar_unless_the_range_is_asked_for():
    assert np.isscalar(shorth([0, 1, 2, 3], fraction=0.5))


@pytest.mark.parametrize("fraction", [0, 1, -0.5, 1.5])
def test_fraction_must_be_strictly_between_zero_and_one(fraction):
    with pytest.raises(AssertionError):
        shorth([0, 1, 2, 3], fraction=fraction)


def test_matches_the_ancestor_implementation():
    """Same inputs as TwoPLoad feeds it: integer frame indices, 80% coverage."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        n = int(rng.integers(3, 60))
        data = rng.integers(0, 40, size=n)
        fraction = float(rng.choice([0.8, 0.5, 0.25, 0.95, rng.uniform(0.01, 0.99)]))
        assert (shorth(data, fraction, return_range=True)
                == _ancestorShorth(data, fraction, return_range=True))
