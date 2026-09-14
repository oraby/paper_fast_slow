'''Match SLEAP video frames to behavioural trials (Figures S3J-M, step 1).

Backend for the "Sync dfs together" section of ``Tracking.ipynb``.

The behaviour PC logs trial times as Unix timestamps. The tracking camera
writes frames into files named by the *local wall-clock* time the recording
started (``images_2025-10-22 13_05_00``). Matching the two needs three things,
all of which were implicit in the notebook:

- **a time zone to read the filenames in.** They carry none. The notebook
  parsed them with ``time.mktime``, which silently uses the time zone of
  whatever machine runs it. The pipeline therefore only worked on a machine set
  to Central European time: read as UTC, 64 of the 76,311 frames still match a
  trial; read as US Eastern time, none do, and the panels are built from
  almost nothing without any error. The zone is now explicit
  (:data:`VIDEO_CLOCK_TZ`), and on a CET machine the timestamps are
  bit-identical to before.
- **an offset** between the tracking PC's clock and the behaviour PC's
  (:data:`VIDEO_CLOCK_OFFSET_S`; the tracking PC ran about 4 h behind). The
  zone and the offset were calibrated together, so neither should be changed
  on its own.
- **a consistency check**: a video is kept only if its start lies within
  :data:`MAX_CLOCK_MISMATCH_S` of a trial's stimulus end, and its length in
  frames matches that trial's sampling time to within
  :data:`MAX_DURATION_MISMATCH_S`.

Each video is one trial's sampling epoch, so after matching every frame
carries its trial's behavioural columns. Tracking columns are prefixed
``track_``.
'''
from __future__ import annotations

from datetime import datetime
from typing import Mapping, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

#: Zone the video filename times are read in. Calibrated together with
#: VIDEO_CLOCK_OFFSET_S -- see the module docstring.
VIDEO_CLOCK_TZ = "Europe/Berlin"
#: Seconds the tracking PC's clock runs behind the behaviour PC's, as read in
#: VIDEO_CLOCK_TZ.
VIDEO_CLOCK_OFFSET_S = 60 * 60 * 4 + 2.2
#: Camera frame rate; a video's duration is its frame count over this.
FRAME_RATE = 30
#: A video must start within this many seconds of its trial's stimulus end.
MAX_CLOCK_MISMATCH_S = 3
#: ...and last within this many seconds of the trial's sampling time.
MAX_DURATION_MISMATCH_S = 0.1
FILENAME_TIME_FORMAT = "%Y-%m-%d %H_%M_%S"
TRACK_PREFIX = "track_"

#: The apparatus midline's top reference point, in image pixels, set by hand
#: for each session. Together with the head-post it defines 0 degrees.
SESSION_TOP_REF: Mapping[str, Tuple[int, int]] = {
    # day 1
    "day1/MLA-73/MLA-73_Mouse2AFC_Oct22_2025_Session1.mat": (405, 49),
    "day1/MLA-74/MLA-74_Mouse2AFC_Oct22_2025_Session2.mat": (405, 49),
    "day1/MLA-75/MLA-75_Mouse2AFC_Oct22_2025_Session2.mat": (405, 60),
    "day1/MLA-76/MLA-76_Mouse2AFC_Oct22_2025_Session3.mat": (406, 54),
    # day 2
    "day2/MLA-73/MLA-73_Mouse2AFC_Oct23_2025_Session2.mat": (405, 55),
    "day2/MLA-74/MLA-74_Mouse2AFC_Oct23_2025_Session1.mat": (405, 55),
    "day2/MLA-75/MLA-75_Mouse2AFC_Oct23_2025_Session1.mat": (405, 55),
    "day2/MLA-76/MLA-76_Mouse2AFC_Oct23_2025_Session1.mat": (405, 55),
    # day 3
    "day3/MLA-73/MLA-73_Mouse2AFC_Oct24_2025_Session1.mat": (401, 61),
    "day3/MLA-74/MLA-74_Mouse2AFC_Oct24_2025_Session1.mat": (407, 55),
    "day3/MLA-75/MLA-75_Mouse2AFC_Oct24_2025_Session1.mat": (406, 50),
    "day3/MLA-76/MLA-76_Mouse2AFC_Oct24_2025_Session1.mat": (406, 55),
}


def clockString(unix_time: float, tz: str=VIDEO_CLOCK_TZ) -> str:
    '''``time.ctime`` format, but in a fixed zone rather than the machine's.'''
    return datetime.fromtimestamp(unix_time, ZoneInfo(tz)).ctime()


def parseFilenameTime(stamp: str, tz: str=VIDEO_CLOCK_TZ) -> float:
    '''``"2025-10-22 13_05_00"`` in ``tz`` -> Unix seconds.'''
    local = datetime.strptime(stamp, FILENAME_TIME_FORMAT)
    return local.replace(tzinfo=ZoneInfo(tz)).timestamp()


def addStimulusEndTime(df_behavior: pd.DataFrame,
                       tz: str=VIDEO_CLOCK_TZ) -> pd.DataFrame:
    '''Unix time at which each trial's stimulus ended -- where its video starts.'''
    df = df_behavior.copy()
    df["StimEndSysTime"] = (df.TrialStartSysTime + df.StimulusStartTime
                            + df.calcStimulusTime)
    df["StimEndSysTimeStr"] = df.StimEndSysTime.dropna().apply(
        clockString, tz=tz)
    return df


def parseVideoTimestamps(df_track: pd.DataFrame,
                         tz: str=VIDEO_CLOCK_TZ) -> pd.DataFrame:
    '''Read each frame's recording start from its SLEAP ``source_file`` name.

    Adds ``src_filename`` (the time stamp as written), ``timestamp`` (Unix
    seconds, interpreted in ``tz``) and ``timestamp_str``.
    '''
    stamps = df_track.source_file.str.split(".").str[1]
    _assertUniformLength(stamps, "the part after the first '.'")
    stamps = stamps.str.split("images_").str[1]
    _assertUniformLength(stamps, "the part after 'images_'")
    df = df_track.copy()
    timestamps = stamps.apply(parseFilenameTime, tz=tz)
    df["src_filename"] = stamps
    df["timestamp"] = timestamps
    df["timestamp_str"] = timestamps.apply(clockString, tz=tz)
    return df


def _assertUniformLength(stamps: pd.Series, what: str) -> None:
    assert stamps.apply(len).nunique() == 1, (
        f"SLEAP source_file names do not all split the same way at {what}")


def matchFramesToTrials(df_track: pd.DataFrame,
                        df_behavior: pd.DataFrame, *,
                        clock_offset_s: float=VIDEO_CLOCK_OFFSET_S,
                        frame_rate: float=FRAME_RATE,
                        max_clock_mismatch_s: float=MAX_CLOCK_MISMATCH_S,
                        max_duration_mismatch_s: float=MAX_DURATION_MISMATCH_S
                        ) -> pd.DataFrame:
    '''One row per video frame, carrying the behavioural trial it belongs to.

    Expects the outputs of :func:`parseVideoTimestamps` and
    :func:`addStimulusEndTime`. Each video is paired with the trial whose
    stimulus end is nearest its start (after the clock offset), then kept only
    if both the start and the duration agree -- see the module docstring.
    '''
    rows = []
    df_behavior = df_behavior.reset_index(drop=True)
    for timestamp, video_df in df_track.groupby("timestamp"):
        diffs = df_behavior.StimEndSysTime - timestamp - clock_offset_s
        nearest = diffs.abs().idxmin()
        trial = df_behavior.loc[nearest]
        record = {
            "timestamp": timestamp,
            "nearest_time": trial["StimEndSysTime"],
            "nearest_time_str": clockString(trial["StimEndSysTime"]),
            "time_diff": diffs.loc[nearest],
            "n_rows": len(video_df),
            "num_secs": len(video_df) / frame_rate,
            "idx": nearest,
            "calcStimulusTime": df_behavior.loc[nearest, "calcStimulusTime"],
        }
        for _, frame in video_df.iterrows():
            row = record.copy()
            for col in video_df.columns:
                row[f"{TRACK_PREFIX}{col}"] = frame[col]
            for col in df_behavior.columns:
                row[col] = trial[col]
            rows.append(row)

    matched = pd.DataFrame(rows)
    matched = matched[matched.time_diff.abs() < max_clock_mismatch_s]
    duration_diff = matched.calcStimulusTime - matched.num_secs
    return matched[duration_diff.abs() < max_duration_mismatch_s]


def reorderTrackCols(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    '''Move the ``track_`` columns to the front; return them too.'''
    track_cols = [col for col in df.columns if col.startswith(TRACK_PREFIX)]
    other_cols = [col for col in df.columns if not col.startswith(TRACK_PREFIX)]
    return df[track_cols + other_cols], track_cols


def assignTopRef(df: pd.DataFrame,
                 top_ref: Mapping[str, Tuple[int, int]]=SESSION_TOP_REF
                 ) -> pd.DataFrame:
    '''Attach each session's hand-set midline reference as ``track_TopRef``.

    A session missing from ``top_ref`` is an error naming the session: the
    midline cannot be guessed, and without it no rotation angle exists.
    '''
    missing = sorted(set(df.File) - set(top_ref))
    if missing:
        raise KeyError(f"No top reference point for session(s): {missing}")
    df = df.copy()
    df["track_TopRef.x"] = df.File.map(lambda path: top_ref[path][0])
    df["track_TopRef.y"] = df.File.map(lambda path: top_ref[path][1])
    return df
