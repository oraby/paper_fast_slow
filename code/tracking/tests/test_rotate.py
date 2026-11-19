'''Tests for the apparatus coordinate frame.

The point of the frame is that angles mean the same thing in every session:
the origin is the head-post and 0 degrees is the midline, whatever the camera
did between days. The invariance tests pin exactly that.
'''
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ..preprocess import matchedFrames
from ..rotate import TRACK_COLS, analyzeRois, rotateToReference
from .fixtures import session

BASE, TOP = (405., 360.), (405., 49.)      # head-post below, reference above
LENGTH = 311.


def test_a_point_on_the_midline_has_no_sideways_offset():
    out = analyzeRois(BASE, TOP, [(405., 200.)])
    assert out["x_prime"][0] == pytest.approx(0.)
    assert out["y_prime"][0] == pytest.approx(160.)
    assert out["theta_deg"][0] == pytest.approx(0.)


def test_the_reference_point_itself_is_one_unit_along_the_midline():
    out = analyzeRois(BASE, TOP, [TOP])
    assert out["y_norm"][0] == pytest.approx(1.)
    assert out["d_norm"][0] == pytest.approx(1.)
    assert out["x_norm"][0] == pytest.approx(0.)


def test_a_point_to_the_right_is_at_plus_ninety_degrees():
    out = analyzeRois(BASE, TOP, [(415., 360.)])
    assert out["x_prime"][0] == pytest.approx(10.)
    assert out["theta_deg"][0] == pytest.approx(90.)


def test_a_point_to_the_left_is_at_minus_ninety_degrees():
    out = analyzeRois(BASE, TOP, [(395., 360.)])
    assert out["x_prime"][0] == pytest.approx(-10.)
    assert out["theta_deg"][0] == pytest.approx(-90.)


def test_normalised_values_are_in_units_of_the_reference_distance():
    out = analyzeRois(BASE, TOP, [(415., 200.)])
    assert out["x_norm"][0] == pytest.approx(10. / LENGTH)
    assert out["y_norm"][0] == pytest.approx(160. / LENGTH)


def test_the_frame_does_not_care_where_the_camera_was():
    '''Rotate and shift the whole image: every coordinate stays put.'''
    rois = np.array([(385., 340.), (425., 290.), (405., 250.)])
    reference = analyzeRois(BASE, TOP, rois)

    angle = np.radians(23.)
    turn = np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])
    move = lambda points: np.atleast_2d(points) @ turn.T + (57., -12.)
    moved = analyzeRois(move(BASE)[0], move(TOP)[0], move(rois))
    for key in ("x_prime", "y_prime", "theta_deg", "d_norm"):
        np.testing.assert_allclose(moved[key], reference[key], atol=1e-9)


def test_coincident_reference_points_are_refused():
    with pytest.raises(ValueError, match="coincide"):
        analyzeRois(BASE, BASE, [(1., 1.)])


def test_missing_keypoints_stay_missing():
    out = analyzeRois(BASE, TOP, [(np.nan, np.nan), (405., 200.)])
    assert np.isnan(out["x_prime"][0])
    assert not np.isnan(out["x_prime"][1])


# --------------------------------------------------------------------------
# Whole sessions
# --------------------------------------------------------------------------

def _rotated(**kwargs):
    df_track, df_behavior = session(**kwargs)
    return rotateToReference(matchedFrames(df_track, df_behavior), verbose=False)


def test_seven_columns_are_added_for_every_point_and_the_reference():
    out = _rotated(n_trials=1)
    for point in TRACK_COLS + ["track_TopRef"]:
        for key in ("x_prime", "y_prime", "x_norm", "y_norm",
                    "theta_rad", "theta_deg", "d_norm"):
            assert f"{point}.x_{key}" in out.columns


def test_the_head_post_is_the_session_median_of_the_screw():
    '''A few bad screw detections must not move the origin.'''
    df_track, df_behavior = session(n_trials=3)
    df_track.loc[:4, "screw.x"] = 900.
    out = rotateToReference(matchedFrames(df_track, df_behavior), verbose=False)
    np.testing.assert_allclose(out["track_T.x_x_prime"], 0., atol=1e-9)


def test_the_still_pose_comes_out_straight_ahead():
    out = _rotated(n_trials=2)
    np.testing.assert_allclose(out["track_T.x_theta_deg"], 0., atol=1e-9)
    np.testing.assert_allclose(out["track_E_L.x_x_prime"], -20., atol=1e-9)
    np.testing.assert_allclose(out["track_E_R.x_x_prime"], 20., atol=1e-9)


def test_rotation_leaves_the_input_alone():
    df_track, df_behavior = session(n_trials=1)
    frames = matchedFrames(df_track, df_behavior)
    before = frames.copy()
    rotateToReference(frames, verbose=False)
    pd.testing.assert_frame_equal(frames, before)
