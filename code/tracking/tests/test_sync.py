'''Tests for matching video frames to behavioural trials.

The real pipeline was checked separately against the notebook's own output:
all 76,311 matched frames are identical, value for value, dtype for dtype.
These tests pin the rules that make the match work -- above all the explicit
time zone, without which the pipeline silently matched 64 frames on a UTC
machine and none on a US one.
'''
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ..sync import (MAX_CLOCK_MISMATCH_S, MAX_DURATION_MISMATCH_S,
                    SESSION_TOP_REF, VIDEO_CLOCK_OFFSET_S, addStimulusEndTime,
                    assignTopRef, clockString, matchFramesToTrials,
                    parseFilenameTime, parseVideoTimestamps, reorderTrackCols)
from .fixtures import SESSION, session


def matched(df_track, df_behavior, **kwargs):
    return matchFramesToTrials(parseVideoTimestamps(df_track),
                               addStimulusEndTime(df_behavior), **kwargs)


# --------------------------------------------------------------------------
# Reading the filename clock
# --------------------------------------------------------------------------

def test_filename_times_are_read_in_central_european_time():
    '''13:05 local on 22 October 2025 is 11:05 UTC (summer time, UTC+2).'''
    expected = datetime(2025, 10, 22, 11, 5, tzinfo=timezone.utc).timestamp()
    assert parseFilenameTime("2025-10-22 13_05_00") == expected


def test_the_winter_offset_is_applied_after_the_clocks_change():
    expected = datetime(2025, 12, 1, 11, 0, tzinfo=timezone.utc).timestamp()
    assert parseFilenameTime("2025-12-01 12_00_00") == expected


def test_the_result_does_not_depend_on_the_machine_time_zone():
    '''The regression: ``time.mktime`` used whatever zone the machine had.'''
    assert parseFilenameTime("2025-10-22 13_05_00", tz="UTC") \
        - parseFilenameTime("2025-10-22 13_05_00") == 2 * 3600


def test_the_display_string_uses_the_same_zone():
    assert clockString(parseFilenameTime("2025-10-22 13_05_00")) \
        == "Wed Oct 22 13:05:00 2025"


def test_the_timestamp_is_read_out_of_the_sleap_filename():
    df_track, _ = session(n_trials=1)
    parsed = parseVideoTimestamps(df_track)
    assert set(parsed.src_filename) == {"2025-10-22 15_05_20"}
    assert set(parsed.timestamp) == {parseFilenameTime("2025-10-22 15_05_20")}
    assert set(parsed.timestamp_str) == {"Wed Oct 22 15:05:20 2025"}


def test_a_filename_that_splits_differently_is_refused():
    df_track, _ = session(n_trials=2)
    df_track.loc[0, "source_file"] = "renamed.video_2025-10-22.csv"
    with pytest.raises(AssertionError, match="source_file"):
        parseVideoTimestamps(df_track)


def test_parsing_leaves_the_input_alone():
    df_track, _ = session(n_trials=1)
    before = df_track.copy()
    parseVideoTimestamps(df_track)
    pd.testing.assert_frame_equal(df_track, before)


# --------------------------------------------------------------------------
# Trial end times
# --------------------------------------------------------------------------

def test_stimulus_end_is_trial_start_plus_stimulus_onset_plus_sampling():
    _, df_behavior = session(n_trials=2)
    out = addStimulusEndTime(df_behavior)
    np.testing.assert_allclose(out.StimEndSysTime,
                               df_behavior.TrialStartSysTime
                               + df_behavior.StimulusStartTime
                               + df_behavior.calcStimulusTime)
    assert "StimEndSysTime" not in df_behavior.columns


def test_a_trial_without_a_sampling_time_gets_no_end_time():
    _, df_behavior = session(n_trials=2)
    df_behavior.loc[0, "calcStimulusTime"] = np.nan
    out = addStimulusEndTime(df_behavior)
    assert np.isnan(out.StimEndSysTime[0])
    assert pd.isna(out.StimEndSysTimeStr[0])
    assert isinstance(out.StimEndSysTimeStr[1], str)


# --------------------------------------------------------------------------
# Pairing videos with trials
# --------------------------------------------------------------------------

def test_every_frame_of_a_consistent_session_is_kept():
    df_track, df_behavior = session(n_trials=3)
    out = matched(df_track, df_behavior)
    assert len(out) == len(df_track)


def test_each_frame_carries_its_own_trial():
    df_track, df_behavior = session(n_trials=3, frames_per_trial=(15, 21, 30))
    out = matched(df_track, df_behavior)
    assert out.groupby("TrialNumber").size().to_dict() == {1: 15, 2: 21, 3: 30}


def test_sleap_columns_are_prefixed_and_behaviour_columns_copied():
    df_track, df_behavior = session(n_trials=1)
    out = matched(df_track, df_behavior)
    assert {"track_E_L.x", "track_frame_idx", "track_timestamp"} <= set(out.columns)
    assert {"Name", "File", "quantile_idx", "StimEndSysTime"} <= set(out.columns)
    assert {"time_diff", "num_secs", "n_rows"} <= set(out.columns)


def test_without_the_clock_offset_nothing_matches():
    df_track, df_behavior = session(n_trials=3)
    assert matched(df_track, df_behavior, clock_offset_s=0).empty


def test_a_video_starting_too_far_from_its_trial_is_dropped():
    df_track, df_behavior = session(n_trials=3)
    df_behavior.loc[1, "StimulusStartTime"] += MAX_CLOCK_MISMATCH_S + 2
    out = matched(df_track, df_behavior)
    assert sorted(out.TrialNumber.unique()) == [1, 3]


def test_a_video_whose_length_disagrees_with_the_sampling_time_is_dropped():
    '''Same stimulus end, but the trial claims a longer sampling time.'''
    df_track, df_behavior = session(n_trials=3)
    stretch = MAX_DURATION_MISMATCH_S * 2
    df_behavior.loc[2, "calcStimulusTime"] += stretch
    df_behavior.loc[2, "TrialStartSysTime"] -= stretch
    out = matched(df_track, df_behavior)
    assert sorted(out.TrialNumber.unique()) == [1, 2]


def test_reading_the_clock_in_the_wrong_zone_loses_every_video():
    '''What happened on any machine not set to Central European time.'''
    df_track, df_behavior = session(n_trials=3)
    wrong = matchFramesToTrials(parseVideoTimestamps(df_track, tz="America/New_York"),
                                addStimulusEndTime(df_behavior))
    assert wrong.empty


def test_the_offset_constant_is_four_hours_and_a_bit():
    assert VIDEO_CLOCK_OFFSET_S == pytest.approx(4 * 3600 + 2.2)


# --------------------------------------------------------------------------
# Column order and the midline reference
# --------------------------------------------------------------------------

def test_track_columns_move_to_the_front():
    df = pd.DataFrame(columns=["Name", "track_T.x", "File", "track_E_L.x"])
    reordered, track_cols = reorderTrackCols(df)
    assert track_cols == ["track_T.x", "track_E_L.x"]
    assert list(reordered.columns) == ["track_T.x", "track_E_L.x", "Name", "File"]


def test_the_session_top_reference_is_attached():
    df_track, df_behavior = session(n_trials=1)
    out = assignTopRef(matched(df_track, df_behavior))
    assert set(out["track_TopRef.x"]) == {SESSION_TOP_REF[SESSION][0]}
    assert set(out["track_TopRef.y"]) == {SESSION_TOP_REF[SESSION][1]}


def test_a_session_without_a_top_reference_is_named_in_the_error():
    '''It used to fail with a bare KeyError from inside a lambda.'''
    df = pd.DataFrame({"File": [SESSION, "day9/MLA-99/unknown.mat"]})
    with pytest.raises(KeyError, match="day9/MLA-99/unknown.mat"):
        assignTopRef(df)


def test_every_published_session_has_a_top_reference():
    assert len(SESSION_TOP_REF) == 12
    assert {path.split("/")[1] for path in SESSION_TOP_REF} == {
        "MLA-73", "MLA-74", "MLA-75", "MLA-76"}
