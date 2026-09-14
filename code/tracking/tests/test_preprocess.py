'''End-to-end tests for the preprocessing chain.

On the real data every stage was compared with the notebook's own frames --
parsed timestamps, matched frames, rotated frames and the final 76,311-row
centroid frame -- and found identical, with and without copy-on-write.
'''
from __future__ import annotations

import numpy as np
import pandas as pd

from ..centroids import plotCentroids
from ..interpolate import ANGLE_COL
from ..preprocess import buildCentroidFrame, matchedFrames, rotatedFrames
from ..strategy import trialMeanAngles
from .fixtures import session


def test_the_chain_produces_one_angle_per_matched_frame():
    df_track, df_behavior = session(n_trials=3, frames_per_trial=(15, 21, 30))
    out = buildCentroidFrame(df_track, df_behavior, verbose=False, progress=False)
    assert len(out) == 66
    assert out[ANGLE_COL].notna().all()


def test_the_inputs_are_not_modified():
    '''The notebook wrote StimEndSysTime into the behaviour frame in place.'''
    df_track, df_behavior = session(n_trials=2)
    track_before, behavior_before = df_track.copy(), df_behavior.copy()
    buildCentroidFrame(df_track, df_behavior, verbose=False, progress=False)
    pd.testing.assert_frame_equal(df_track, track_before)
    pd.testing.assert_frame_equal(df_behavior, behavior_before)


def test_the_stages_nest():
    df_track, df_behavior = session(n_trials=2)
    matched = matchedFrames(df_track, df_behavior)
    rotated = rotatedFrames(df_track, df_behavior, verbose=False)
    final = buildCentroidFrame(df_track, df_behavior, verbose=False, progress=False)
    assert len(matched) == len(rotated) == len(final)
    assert set(matched.columns) < set(rotated.columns) < set(final.columns)


def test_the_output_is_what_the_figure_code_consumes():
    df_track, df_behavior = session(n_trials=6, frames_per_trial=(15, 21, 30))
    out = buildCentroidFrame(df_track, df_behavior, verbose=False, progress=False)
    assert len(trialMeanAngles(out)) == 6
    results = plotCentroids(out, outliers_ratio=0.1, verbose=False)
    assert results[0]["label"] == "All Tracked Subjects"
    import matplotlib.pyplot as plt
    plt.close("all")


def test_a_turned_mouse_gives_a_nonzero_angle():
    df_track, df_behavior = session(n_trials=1, frames_per_trial=(20,))
    for point in ("E_L", "L_L", "E_R", "L_R", "T"):
        df_track[f"{point}.x"] += 40.          # shift the body to the right
    out = buildCentroidFrame(df_track, df_behavior, verbose=False, progress=False)
    assert (out[ANGLE_COL] > 0).all()
    assert np.ptp(out[ANGLE_COL]) < 1e-9
