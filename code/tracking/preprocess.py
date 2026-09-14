'''From raw SLEAP output to the per-frame centroid angles (Figures S3J-M).

One call replaces the notebook's chain of preprocessing cells:

1. :mod:`tracking.sync` -- read each video's start time, pair it with the trial
   it recorded, and attach the session's hand-set midline reference;
2. :mod:`tracking.rotate` -- re-express every keypoint in the apparatus frame;
3. :mod:`tracking.interpolate` -- fill missing keypoints, take the body
   centroid, and measure its rotation angle.

The result feeds :func:`tracking.centroids.plotCentroids` and
:func:`tracking.strategy.plotStrategyComparison`.
'''
from __future__ import annotations

import pandas as pd

from .interpolate import interpolateMissing
from .rotate import rotateToReference
from .sync import (VIDEO_CLOCK_TZ, addStimulusEndTime, assignTopRef,
                   matchFramesToTrials, parseVideoTimestamps, reorderTrackCols)


def matchedFrames(df_track: pd.DataFrame, df_behavior: pd.DataFrame, *,
                  tz: str=VIDEO_CLOCK_TZ) -> pd.DataFrame:
    '''Steps 1: frames matched to trials, with the top reference attached.'''
    matched = matchFramesToTrials(parseVideoTimestamps(df_track, tz=tz),
                                  addStimulusEndTime(df_behavior, tz=tz))
    matched, _ = reorderTrackCols(matched)
    matched, _ = reorderTrackCols(assignTopRef(matched))
    return matched


def rotatedFrames(df_track: pd.DataFrame, df_behavior: pd.DataFrame, *,
                  tz: str=VIDEO_CLOCK_TZ, verbose: bool=True) -> pd.DataFrame:
    '''Steps 1-2: matched frames in the apparatus coordinate frame.'''
    rotated = rotateToReference(matchedFrames(df_track, df_behavior, tz=tz),
                                verbose=verbose)
    rotated, _ = reorderTrackCols(rotated)
    return rotated


def buildCentroidFrame(df_track: pd.DataFrame, df_behavior: pd.DataFrame, *,
                       tz: str=VIDEO_CLOCK_TZ, verbose: bool=True,
                       progress: bool=True) -> pd.DataFrame:
    '''Steps 1-3: one row per frame with ``track_centroid_rotation_angle``.

    ``df_track`` is the SLEAP export (``data/tracking/df_SLEAP_tracking.pkl``),
    ``df_behavior`` the tracked sessions' trials
    (``data/tracking/df_behavior_tracked.pkl``). Neither is modified.
    '''
    return interpolateMissing(rotatedFrames(df_track, df_behavior, tz=tz,
                                            verbose=verbose),
                              verbose=verbose, progress=progress)
