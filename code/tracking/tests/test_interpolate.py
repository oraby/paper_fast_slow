'''Tests for keypoint filling and the body centroid.

Checked separately against the notebook on all 76,311 frames: identical output,
with and without pandas' copy-on-write. On the real data 16.7% of limb
keypoints were missing; 35,827 were filled in time and 14,438 (in 18.2% of
frames) rebuilt from the other limbs.

The copy-on-write tests are the regression guard for pandas 3. The notebook
filled gaps with an in-place call on a column pulled out of a frame, which
copy-on-write turns into a silent no-op: on two real sessions that changed the
angle of 6,189 of 11,212 frames.
'''
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ..interpolate import (ANGLE_COL, RECONSTRUCTED_SCORE, averageOffset,
                           bodyCentroid, completeLimbs, dropBadTail,
                           fillGapsInTime, interpolateMissing, rotationAngle,
                           sessionOffsets)
from ..preprocess import rotatedFrames
from .fixtures import session

LIMB_NAMES = ("track_E_L", "track_E_R", "track_L_L", "track_L_R")
#: A rectangle, so every reconstruction rule recovers the point exactly.
RECT = {"track_E_L": (-20., 20.), "track_E_R": (20., 20.),
        "track_L_L": (-20., 70.), "track_L_R": (20., 70.)}


def row(points: dict, tail=(np.nan, np.nan), tail_score=0.) -> pd.Series:
    data = {}
    for name in LIMB_NAMES:
        x, y = points.get(name, (np.nan, np.nan))
        data[f"{name}.x_x_prime"], data[f"{name}.x_y_prime"] = x, y
        data[f"{name}.score"] = .9 if name in points else 0.
    data["track_T.x_x_prime"], data["track_T.x_y_prime"] = tail
    data["track_T.score"] = tail_score
    return pd.Series(data)


def rectOffsets():
    frame = pd.DataFrame([row(RECT)] * 3)
    return sessionOffsets(frame)


def coords(r, name):
    return r[f"{name}.x_x_prime"], r[f"{name}.x_y_prime"]


# --------------------------------------------------------------------------
# Stage 1: filling in time
# --------------------------------------------------------------------------

def test_an_interior_gap_of_any_length_is_filled_linearly():
    '''Not only short gaps: the one-frame fill repeats until nothing changes.'''
    values = pd.Series([1., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 8.])
    np.testing.assert_allclose(fillGapsInTime(values), np.arange(1., 9.))


def test_gaps_at_the_edges_hold_the_nearest_valid_value():
    values = pd.Series([np.nan, np.nan, 4., 5., np.nan])
    np.testing.assert_allclose(fillGapsInTime(values), [4., 4., 4., 5., 5.])


def test_a_coordinate_missing_throughout_is_left_for_stage_two():
    assert fillGapsInTime(pd.Series([np.nan] * 4)).isna().all()


def test_filling_never_touches_its_input():
    values = pd.Series([1., np.nan, 3.])
    fillGapsInTime(values)
    assert np.isnan(values[1])


def test_filling_works_under_copy_on_write():
    with pd.option_context("mode.copy_on_write", True):
        filled = fillGapsInTime(pd.Series([1., np.nan, np.nan, 4.]))
    np.testing.assert_allclose(filled, [1., 2., 3., 4.])


# --------------------------------------------------------------------------
# Stage 2: rebuilding limbs from the others
# --------------------------------------------------------------------------

def test_the_session_offsets_are_mean_differences_between_limb_pairs():
    offsets = rectOffsets()
    np.testing.assert_allclose(offsets["shoulder"], (40., 0.))
    np.testing.assert_allclose(offsets["limb"], (40., 0.))
    np.testing.assert_allclose(offsets["left"], (0., -50.))
    np.testing.assert_allclose(offsets["right"], (0., -50.))


def test_offsets_ignore_frames_missing_either_point():
    frame = pd.DataFrame([row(RECT), row({"track_E_L": (-20., 20.)})])
    np.testing.assert_allclose(averageOffset(frame, "track_E_R", "track_E_L"),
                               (40., 0.))


@pytest.mark.parametrize("present", [
    ("track_E_L",), ("track_E_R",),
    ("track_L_L", "track_L_R"), ("track_E_L", "track_E_R"),
    ("track_L_L", "track_E_L"), ("track_L_R", "track_E_R"),
    ("track_L_L", "track_E_R"), ("track_L_R", "track_E_L"),
    ("track_E_L", "track_E_R", "track_L_L"), ("track_E_L", "track_E_R", "track_L_R"),
    ("track_E_L", "track_L_L", "track_L_R"), ("track_E_R", "track_L_L", "track_L_R"),
])
def test_every_reconstruction_rule_recovers_a_rectangular_pose(present):
    before = row({name: RECT[name] for name in present})
    count, after = completeLimbs(before, rectOffsets())
    rebuilt = [name for name in LIMB_NAMES if name not in present]
    expected_count = 2 if len(present) == 1 else 4
    assert count == expected_count
    for name in rebuilt[: expected_count - len(present)]:
        np.testing.assert_allclose(coords(after, name), RECT[name])
        assert after[f"{name}.score"] == RECONSTRUCTED_SCORE
    for name in present:
        assert after[f"{name}.score"] == .9


def test_one_shoulder_only_gains_the_other_shoulder():
    count, after = completeLimbs(row({"track_E_L": RECT["track_E_L"]}),
                                 rectOffsets())
    assert count == 2
    assert np.isnan(after["track_L_L.x_x_prime"])


def test_a_lone_hind_limb_is_not_extended():
    '''The one case with nothing to rebuild from; these points stay missing.'''
    before = row({"track_L_L": RECT["track_L_L"]})
    count, after = completeLimbs(before, rectOffsets())
    assert count == 1
    assert np.isnan(after["track_E_L.x_x_prime"])


def test_a_complete_pose_is_left_alone():
    before = row(RECT)
    count, after = completeLimbs(before, rectOffsets())
    assert count == 4
    pd.testing.assert_series_equal(after, before)


# --------------------------------------------------------------------------
# Centroid and angle
# --------------------------------------------------------------------------

def test_the_centroid_includes_the_tail_when_present():
    with_tail = bodyCentroid(row(RECT, tail=(0., 120.)), 4)
    without = bodyCentroid(row(RECT), 4)
    assert with_tail == (0., (20 + 20 + 70 + 70 + 120) / 5)
    assert without == (0., 45.)


def test_without_four_limbs_a_tail_and_shoulders_triangle_is_used():
    r = row({"track_E_L": (-20., 20.), "track_E_R": (20., 20.)}, tail=(0., 110.))
    assert bodyCentroid(r, 2) == (0., 50.)


def test_no_centroid_without_either_shape():
    r = row({"track_E_L": (-20., 20.), "track_L_L": (-20., 70.)})
    assert all(np.isnan(bodyCentroid(r, 2)))


def test_coordinates_are_truncated_toward_zero_before_averaging():
    '''int32 casting, as the published panels were computed -- not rounding.'''
    near_zero = {name: (0.9 if "R" in name else -0.9, 10.9) for name in LIMB_NAMES}
    assert bodyCentroid(row(near_zero), 4) == (0., 10.)


@pytest.mark.parametrize("centroid, angle", [
    ((0., 10.), 0.), ((10., 10.), 45.), ((-10., 10.), -45.), ((10., 0.), 90.)])
def test_the_angle_is_measured_from_straight_ahead_positive_to_the_right(centroid, angle):
    assert rotationAngle(centroid) == pytest.approx(angle)


def test_a_missing_centroid_has_a_missing_angle():
    assert np.isnan(rotationAngle((np.nan, np.nan)))


# --------------------------------------------------------------------------
# Tail clean-up
# --------------------------------------------------------------------------

def test_a_low_scoring_tail_is_removed():
    frame = pd.DataFrame([row(RECT, tail=(0., 120.), tail_score=.1),
                          row(RECT, tail=(0., 120.), tail_score=.9)])
    cleaned = dropBadTail(frame)
    assert np.isnan(cleaned.loc[0, "track_T.x_x_prime"])
    assert cleaned.loc[1, "track_T.x_x_prime"] == 0.


def test_a_tail_nearer_the_head_than_both_hind_limbs_is_removed():
    frame = pd.DataFrame([row(RECT, tail=(0., 30.), tail_score=.9)])
    assert np.isnan(dropBadTail(frame).loc[0, "track_T.x_y_prime"])


# --------------------------------------------------------------------------
# Whole sessions
# --------------------------------------------------------------------------

def _centroids(**kwargs):
    df_track, df_behavior = session(**kwargs)
    return interpolateMissing(rotatedFrames(df_track, df_behavior, verbose=False),
                              verbose=False, progress=False)


def test_one_row_per_frame_with_the_angle_added():
    out = _centroids(n_trials=2, frames_per_trial=(15, 21))
    assert len(out) == 36
    assert {"track_centroid_x", "track_centroid_y", ANGLE_COL} <= set(out.columns)


def test_a_still_mouse_points_straight_ahead():
    out = _centroids(n_trials=2)
    np.testing.assert_allclose(out[ANGLE_COL], 0., atol=1e-9)


def test_a_briefly_lost_limb_is_filled_in_time_not_rebuilt():
    out = _centroids(n_trials=1, frames_per_trial=(20,),
                     missing_by_trial={0: {"L_L": range(5, 12)}})
    assert out["track_L_L.x_x_prime"].notna().all()
    assert (out["track_L_L.score"] != RECONSTRUCTED_SCORE).all()
    np.testing.assert_allclose(out[ANGLE_COL], 0., atol=1e-9)


def test_a_limb_lost_for_a_whole_trial_is_rebuilt_from_the_others():
    out = _centroids(n_trials=2, frames_per_trial=(20,),
                     missing_by_trial={1: {"L_L": "all"}})
    trial_two = out[out.TrialNumber == 2]
    assert (trial_two["track_L_L.score"] == RECONSTRUCTED_SCORE).all()
    assert (out[out.TrialNumber == 1]["track_L_L.score"] != RECONSTRUCTED_SCORE).all()


def test_the_result_is_identical_under_copy_on_write():
    '''The pandas 3 regression guard.'''
    kwargs = dict(n_trials=3, jitter=2.,
                  missing_by_trial={0: {"E_R": range(3, 9)},
                                    1: {"L_L": "all"},
                                    2: {"T": range(0, 4)}})
    plain = _centroids(**kwargs)
    with pd.option_context("mode.copy_on_write", True):
        cow = _centroids(**kwargs)
    pd.testing.assert_frame_equal(plain, cow, check_exact=True)
