'''Express tracked points in the apparatus frame (Figures S3J-M, step 2).

Backend for the "Rotate to reference" section of ``Tracking.ipynb``.

Each session gets its own coordinate frame: the origin is the head-post (the
median position of the tracked ``screw`` over the session, which is fixed) and
the y-axis points along the apparatus midline towards the hand-set top
reference (see :data:`tracking.sync.SESSION_TOP_REF`). Every tracked point is
then re-expressed in that frame, so angles are comparable across sessions even
if the camera moved between days.

For a point, columns ``<point>.x_x_prime`` / ``<point>.x_y_prime`` hold its
coordinates across / along the midline; ``_x_norm`` / ``_y_norm`` the same in
units of the head-post-to-reference distance; ``_theta_rad`` / ``_theta_deg``
its signed angle from the midline; ``_d_norm`` its normalised distance. The
names keep the notebook's (slightly odd) ``.x_`` stem so downstream code reads
the same columns.
'''
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

#: The tracked limb and tail points.
TRACK_COLS = ["track_E_L", "track_L_L", "track_E_R", "track_L_R", "track_T"]
HEAD_POST = "track_screw"
TOP_REF = "track_TopRef"


def analyzeRois(base: Sequence[float], top: Sequence[float],
                rois: Iterable) -> dict:
    '''Coordinates of ``rois`` in the frame defined by ``base`` -> ``top``.

    ``base`` and ``top`` are (x, y); ``rois`` is an (N, 2) array. Raises if the
    two reference points coincide, since the direction is then undefined.
    '''
    base = np.asarray(base, dtype=float)
    top = np.asarray(top, dtype=float)
    rois = np.asarray(rois, dtype=float)

    axis = top - base
    length = np.linalg.norm(axis)
    if length == 0:
        raise ValueError("Base and Top coincide; direction is undefined.")

    along = axis / length                     # unit vector base -> top
    across = np.array([-along[1], along[0]])  # +90 degrees
    offsets = rois - base

    y_prime = offsets @ along
    x_prime = offsets @ across
    cross = axis[0] * offsets[:, 1] - axis[1] * offsets[:, 0]
    theta = np.arctan2(cross, offsets @ axis)  # signed angle from the midline

    return {
        "x_prime": x_prime, "y_prime": y_prime,
        "x_norm": x_prime / length, "y_norm": y_prime / length,
        "theta_rad": theta, "theta_deg": np.degrees(theta),
        "d_norm": np.linalg.norm(offsets, axis=1) / length,
    }


def rotateToReference(df: pd.DataFrame, *, verbose: bool=True) -> pd.DataFrame:
    '''Add the apparatus-frame columns for every tracked point, per session.

    Expects :func:`tracking.sync.assignTopRef` to have run. The head-post is
    the session's median ``track_screw`` position; the top reference is read
    from the first row, since it is constant within a session.
    '''
    df = df.copy()
    for session, session_df in df.groupby("File"):
        base = (session_df[f"{HEAD_POST}.x"].median(),
                session_df[f"{HEAD_POST}.y"].median())
        top = (session_df[f"{TOP_REF}.x"].iloc[0],
               session_df[f"{TOP_REF}.y"].iloc[0])
        if verbose:
            print(f"Session: {session}")
            print(f"  Base screw median: ({base[0]}, {base[1]})")
        for point in TRACK_COLS + [TOP_REF]:
            col_x, col_y = f"{point}.x", f"{point}.y"
            result = analyzeRois(base, top,
                                 session_df[[col_x, col_y]].to_numpy())
            for key, values in result.items():
                df.loc[session_df.index, f"{col_x}_{key}"] = values
    return df
