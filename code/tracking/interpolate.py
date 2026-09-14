'''Fill missing keypoints and reduce each frame to a body centroid (step 3).

Backend for the "Interpolate missing points" section of ``Tracking.ipynb``.
Its output, one row per frame with ``track_centroid_rotation_angle``, is what
Figures S3J-M are drawn from.

Missing SLEAP keypoints are filled in two stages, per trial:

1. **In time.** Each coordinate is interpolated linearly across the frames of
   its trial. The notebook did this by repeating a one-frame fill until nothing
   changed, which fills a gap of *any* length -- not only short ones -- and
   holds the nearest valid value across a gap at the start or end of the trial.
   :func:`fillGapsInTime` keeps exactly that algorithm, so the values are
   bit-identical.
2. **From the other limbs.** A keypoint still missing after stage 1 was
   missing in every frame of its trial. It is reconstructed from the limbs that
   are present, using that session's average offset between limb pairs
   (:func:`completeLimbs`). Reconstructed points get ``score = -1``.

A tail point with a low score, or lying below both hind limbs, is treated as a
detection error and removed before stage 1.

**The body centroid** is the mean of the four limb points plus the tail when
present; without four limbs, a triangle of tail and both shoulders is used if
available. The coordinates are cast to ``int32`` before averaging -- a leftover
from pixel drawing code -- which truncates each point toward zero. That is what
the published panels used, so it is kept.

**pandas 3.** Stage 1 used to run ``trial_df[col].interpolate(inplace=True)``,
an in-place call on a column pulled out of a frame. Under pandas 3's
copy-on-write that no longer writes back, so stage 1 would have become a silent
no-op and stage 2 would have reconstructed far more points. Stage 1 is now an
explicit assignment; see ``tracking/tests/test_interpolate.py``.
'''
from __future__ import annotations

import numpy as np
import pandas as pd

LIMBS = ["track_E_R", "track_E_L", "track_L_R", "track_L_L"]
TAIL = "track_T"
X, Y, SCORE = "x_x_prime", "x_y_prime", "score"
#: A tail detection scoring below this is discarded.
BAD_TAIL_SCORE = 0.25
#: ``score`` given to a keypoint reconstructed from the other limbs.
RECONSTRUCTED_SCORE = -1
ANGLE_COL = "track_centroid_rotation_angle"


def getCoord(col_key: str, row: pd.Series):
    '''``(x, y, score)`` of one keypoint in the apparatus frame.'''
    return (row[f"{col_key}.{X}"], row[f"{col_key}.{Y}"],
            row[f"{col_key}.{SCORE}"])


def averageOffset(session_df: pd.DataFrame, first: str, second: str) -> np.ndarray:
    '''Mean ``first - second`` over the session's frames that have both.'''
    cols = [f"{first}.{X}", f"{first}.{Y}", f"{second}.{X}", f"{second}.{Y}"]
    both = session_df[cols].dropna()
    return np.array([(both[f"{first}.{X}"] - both[f"{second}.{X}"]).mean(),
                     (both[f"{first}.{Y}"] - both[f"{second}.{Y}"]).mean()])


def sessionOffsets(session_df: pd.DataFrame) -> dict:
    '''The four limb-pair offsets :func:`completeLimbs` reconstructs from.'''
    return dict(
        shoulder=averageOffset(session_df, "track_E_R", "track_E_L"),
        limb=averageOffset(session_df, "track_L_R", "track_L_L"),
        left=averageOffset(session_df, "track_E_L", "track_L_L"),
        right=averageOffset(session_df, "track_E_R", "track_L_R"))


def fillGapsInTime(values: pd.Series) -> pd.Series:
    '''Stage 1: linear interpolation across a trial, returned, never in place.

    One-frame fills are repeated until nothing changes. Interior gaps of any
    length end up linear; leading and trailing gaps take the nearest valid
    value. A coordinate with no valid value in the trial is left missing for
    stage 2. Repeating one-frame fills (rather than one full interpolation) is
    the notebook's algorithm and is kept for bit-identical results.
    '''
    filled = values.copy()
    while True:
        missing = int(filled.isna().sum())
        filled = filled.interpolate(limit_direction="both", limit=1)
        if int(filled.isna().sum()) == missing:
            return filled


def completeLimbs(row: pd.Series, offsets: dict):
    '''Stage 2: rebuild missing limb points from the ones present.

    Returns ``(number of limb points after reconstruction, updated row)``.
    With one shoulder present the other shoulder is added; with two points,
    the other two; with three, the fourth, using the offset between the two
    points on the complete side. A single hind limb is left as it is.
    '''
    ll_x, ll_y, ll_s = getCoord("track_L_L", row)
    lr_x, lr_y, lr_s = getCoord("track_L_R", row)
    el_x, el_y, el_s = getCoord("track_E_L", row)
    er_x, er_y, er_s = getCoord("track_E_R", row)
    shoulder, limb = offsets["shoulder"], offsets["limb"]
    left, right = offsets["left"], offsets["right"]
    present = np.sum(~np.isnan([ll_x, lr_x, el_x, er_x]))
    fin = np.isfinite
    new = RECONSTRUCTED_SCORE

    if present == 1 and (fin(el_x) or fin(er_x)):
        if fin(el_x):
            er_x, er_y, er_s = el_x + shoulder[0], el_y + shoulder[1], new
        else:
            el_x, el_y, el_s = er_x - shoulder[0], er_y - shoulder[1], new
        present += 1
    elif present == 2:
        added = True
        if fin(ll_x) and fin(lr_x):
            el_x, el_y, el_s = ll_x + left[0], ll_y + left[1], new
            er_x, er_y, er_s = lr_x + right[0], lr_y + right[1], new
        elif fin(el_x) and fin(er_x):
            ll_x, ll_y, ll_s = el_x - left[0], el_y - left[1], new
            lr_x, lr_y, lr_s = er_x - right[0], er_y - right[1], new
        elif fin(ll_x) and fin(el_x):
            lr_x, lr_y, lr_s = ll_x + limb[0], ll_y + limb[1], new
            er_x, er_y, er_s = el_x + shoulder[0], el_y + shoulder[1], new
        elif fin(lr_x) and fin(er_x):
            ll_x, ll_y, ll_s = lr_x - limb[0], lr_y - limb[1], new
            el_x, el_y, el_s = er_x - shoulder[0], er_y - shoulder[1], new
        elif fin(ll_x) and fin(er_x):
            lr_x, lr_y, lr_s = ll_x + limb[0], ll_y + limb[1], new
            el_x, el_y, el_s = er_x - shoulder[0], er_y - shoulder[1], new
        elif fin(lr_x) and fin(el_x):
            ll_x, ll_y, ll_s = lr_x - limb[0], lr_y - limb[1], new
            er_x, er_y, er_s = el_x + shoulder[0], el_y + shoulder[1], new
        else:
            added = False
        if added:
            present += 2
    elif present == 3:
        if fin(ll_x) and fin(el_x):
            diff = np.array([ll_x, ll_y]) - np.array([el_x, el_y])
        else:
            assert fin(lr_x) and fin(er_x)
            diff = np.array([lr_x, lr_y]) - np.array([er_x, er_y])
        if np.isnan(ll_x):
            (ll_x, ll_y), ll_s = np.array([el_x, el_y]) + diff, new
        elif np.isnan(lr_x):
            (lr_x, lr_y), lr_s = np.array([er_x, er_y]) + diff, new
        elif np.isnan(el_x):
            (el_x, el_y), el_s = np.array([ll_x, ll_y]) - diff, new
        elif np.isnan(er_x):
            (er_x, er_y), er_s = np.array([lr_x, lr_y]) - diff, new
        present += 1

    row = row.copy()
    for point, (x, y, score) in (("track_L_L", (ll_x, ll_y, ll_s)),
                                 ("track_L_R", (lr_x, lr_y, lr_s)),
                                 ("track_E_L", (el_x, el_y, el_s)),
                                 ("track_E_R", (er_x, er_y, er_s))):
        row[f"{point}.{X}"] = x
        row[f"{point}.{Y}"] = y
        row[f"{point}.{SCORE}"] = score
    return present, row


def bodyCentroid(row: pd.Series, limb_points: int):
    '''Centroid of the limb polygon (with tail), else of tail + shoulders.

    Returns ``(nan, nan)`` when neither shape is available. Points are cast to
    ``int32`` first, as the published panels were -- see the module docstring.
    '''
    t_x, t_y, _ = getCoord(TAIL, row)
    ll_x, ll_y, _ = getCoord("track_L_L", row)
    lr_x, lr_y, _ = getCoord("track_L_R", row)
    el_x, el_y, _ = getCoord("track_E_L", row)
    er_x, er_y, _ = getCoord("track_E_R", row)
    if limb_points == 4:
        points = [(er_x, er_y), (el_x, el_y), (ll_x, ll_y),
                  *([(t_x, t_y)] if np.isfinite(t_x) else []), (lr_x, lr_y)]
    elif np.isfinite(t_x) and np.isfinite(er_x) and np.isfinite(el_x):
        points = [(t_x, t_y), (er_x, er_y), (el_x, el_y)]
    else:
        return np.nan, np.nan
    return tuple(np.mean(np.array(points, dtype=np.int32), axis=0))


def rotationAngle(centroid) -> float:
    '''Degrees from the midline, positive clockwise; 0 is straight ahead.'''
    return -(np.degrees(np.arctan2(centroid[1], centroid[0])) - 90.0)


def dropBadTail(session_df: pd.DataFrame,
                min_score: float=BAD_TAIL_SCORE) -> pd.DataFrame:
    '''Remove tail detections that score low or sit below both hind limbs.'''
    session_df = session_df.copy()
    hind_y = session_df[[f"track_L_L.{Y}", f"track_L_R.{Y}"]].min(axis=1)
    bad = ((session_df[f"{TAIL}.{SCORE}"] < min_score)
           | (session_df[f"{TAIL}.{Y}"] < hind_y))
    session_df.loc[bad, f"{TAIL}.{X}"] = np.nan
    session_df.loc[bad, f"{TAIL}.{Y}"] = np.nan
    return session_df


def interpolateMissing(df: pd.DataFrame, *, verbose: bool=True,
                       progress: bool=True) -> pd.DataFrame:
    '''One row per frame, with keypoints filled and the centroid angle added.

    Expects :func:`tracking.rotate.rotateToReference` to have run. Adds
    ``track_centroid_x``, ``track_centroid_y`` and
    ``track_centroid_rotation_angle``.
    '''
    rows = []
    for session, session_df in df.groupby("File"):
        if verbose:
            print(f"Interpolating missing limb points for session {session}")
        offsets = sessionOffsets(session_df)
        session_df = dropBadTail(session_df)
        trials = session_df.groupby("TrialNumber")
        if progress:
            from tqdm.auto import tqdm
            trials = tqdm(trials, total=session_df.TrialNumber.nunique(),
                          desc=f"Interpolating {session}")
        for _trial, trial_df in trials:
            for point in LIMBS + [TAIL]:
                for dim in (X, Y):
                    key = f"{point}.{dim}"
                    session_df.loc[trial_df.index, key] = fillGapsInTime(
                        session_df.loc[trial_df.index, key])
            for _, row in session_df.loc[trial_df.index].iterrows():
                limb_points, row = completeLimbs(row, offsets)
                centroid = bodyCentroid(row, limb_points)
                row = row.copy()
                row["track_centroid_x"] = centroid[0]
                row["track_centroid_y"] = centroid[1]
                row[ANGLE_COL] = rotationAngle(centroid)
                rows.append(row)
    return pd.DataFrame(rows)
