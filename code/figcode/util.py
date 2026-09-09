'''Shared helpers for the figure code.'''
from scipy import stats

#: Current-trial sampling time, plus ``PrevCalcStimulusTime1``, ``...2``, ...
ST_COL = "calcStimulusTime"
PREV_ST_PREFIX = "PrevCalcStimulusTime"
SUBJECT_KEY = "Name"


def stColumns(df):
    '''The sampling-time columns present in ``df``, current trial first.'''
    return [ST_COL] + [col for col in df.columns
                       if col.startswith(PREV_ST_PREFIX)]


def normalizeSTAcrossSubjects(df):
    '''Z-score every sampling-time column **within each animal**.

    Each column named by :func:`stColumns` gains a ``transformed<Column>``
    counterpart; the originals are left alone and the row order and index are
    preserved. Normalising per animal is the point: a naturally slow mouse
    must not read as a slow *trial* everywhere it appears.

    The z-score uses ``nan_policy="omit"``, so a missing sampling time stays
    missing instead of poisoning that animal's mean, and an animal with no
    spread (a single trial) yields NaN.

    Other transforms were tried on this data and not kept: winsorizing at
    5%/95%, ``RobustScaler``, ``QuantileTransformer``, Box-Cox and a plain
    log. The z-score is what the figures use.

    Read by ``stbydifficulty`` (Figures 1D, 1G, S2E-F, S3A), ``stheatmap``
    (Figure S3E) and ``behavior.stayswitchupdate`` (Figure S3G).
    '''
    df = df.copy()
    st_data_cols = stColumns(df)
    # groupby(...).transform, not .apply: apply would hand the callback the
    # grouping column too (a pandas FutureWarning), and excluding it there
    # would drop `Name` from the result, since the callback returns the whole
    # sub-frame.
    transformed = df.groupby(SUBJECT_KEY)[st_data_cols].transform(
        lambda col: stats.zscore(col, nan_policy="omit"))
    for col in st_data_cols:
        df[f"transformed{col[0].upper()}{col[1:]}"] = transformed[col]
    return df
