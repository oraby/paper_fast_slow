'''Small synthetic SLEAP exports and behaviour tables for the preprocessing tests.

Shaped like ``data/tracking/df_SLEAP_tracking.pkl`` and
``data/tracking/df_behavior_tracked.pkl``: one video per trial, named by the
local wall-clock time it started, recorded by a PC whose clock runs
``VIDEO_CLOCK_OFFSET_S`` behind the behaviour PC's.
'''
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..sync import (FRAME_RATE, SESSION_TOP_REF, VIDEO_CLOCK_OFFSET_S,
                    parseFilenameTime)

SESSION = "day1/MLA-73/MLA-73_Mouse2AFC_Oct22_2025_Session1.mat"
FIRST_VIDEO = datetime(2025, 10, 22, 15, 5, 20)
POINTS = ("screw", "E_L", "L_L", "E_R", "L_R", "T")

#: Image-pixel keypoints of a mouse lying straight along the midline. As in the
#: real videos, the body extends from the head-post *towards* the top reference
#: (405, 49): shoulders nearest the head-post, then hind limbs, then the tail.
STILL_POSE = {"screw": (405., 360.), "E_L": (385., 340.), "E_R": (425., 340.),
              "L_L": (390., 290.), "L_R": (420., 290.), "T": (405., 250.)}


def sourceFile(start: datetime) -> str:
    stamp = start.strftime("%Y-%m-%d %H_%M_%S")
    return f"labels_train_gt_0 Hatem2.003_images_{stamp}.240.analysis.csv"


def video(start: datetime, n_frames: int, pose=STILL_POSE, *,
          missing: dict=None, jitter: float=0., seed: int=0) -> pd.DataFrame:
    '''One trial's frames. ``missing`` maps a point to the frame numbers
    where SLEAP lost it (``"all"`` for every frame).'''
    rng = np.random.default_rng(seed)
    rows = []
    for frame in range(n_frames):
        row = {"track": "track_0", "frame_idx": frame, "instance.score": .9,
               "source_file": sourceFile(start)}
        for point in POINTS:
            x, y = pose[point]
            row[f"{point}.x"] = x + rng.normal(0, jitter) if jitter else x
            row[f"{point}.y"] = y + rng.normal(0, jitter) if jitter else y
            row[f"{point}.score"] = .9
        rows.append(row)
    df = pd.DataFrame(rows)
    for point, frames in (missing or {}).items():
        where = df.index if frames == "all" else list(frames)
        df.loc[where, [f"{point}.x", f"{point}.y"]] = np.nan
        df.loc[where, f"{point}.score"] = 0.
    return df


def session(n_trials: int=3, frames_per_trial=(15, 21, 30), *,
            session_file: str=SESSION, missing_by_trial: dict=None,
            jitter: float=0.):
    '''``(df_track, df_behavior)`` for one session whose videos all match.'''
    videos, trials = [], []
    for trial in range(n_trials):
        start = FIRST_VIDEO + timedelta(minutes=trial)
        n_frames = frames_per_trial[trial % len(frames_per_trial)]
        videos.append(video(start, n_frames,
                            missing=(missing_by_trial or {}).get(trial),
                            jitter=jitter, seed=trial))
        stim_end = parseFilenameTime(start.strftime("%Y-%m-%d %H_%M_%S")) \
            + VIDEO_CLOCK_OFFSET_S
        sampling = n_frames / FRAME_RATE
        trials.append({"Name": "MLA-73", "Date": FIRST_VIDEO.date(),
                       "SessionNum": 1, "TrialNumber": trial + 1,
                       "TrialStartSysTime": stim_end - 7. - sampling,
                       "StimulusStartTime": 7., "calcStimulusTime": sampling,
                       "File": session_file, "ChoiceLeft": float(trial % 2),
                       "quantile_idx": float(trial % 3 + 1)})
    return pd.concat(videos, ignore_index=True), pd.DataFrame(trials)


assert SESSION in SESSION_TOP_REF
