from .logic import makeOneRun
from ...behavior.util.splitdata import splitStimulusTimeByQuantile
import pandas as pd
import numpy as np
import numpy.typing as npt
from enum import Enum, auto
from inspect import signature
from functools import partial

class PsychometricPlot(Enum):
    _None = auto()
    All = auto()
    SlowFast = auto()


def decayingQ(size, Q_val, Q_val_decay_rate, Q_val_coef, dt):
    # Decay the Q-val by the decay rate at each time step
    # Create an array of indices
    indices = np.arange(size)
    EXPONENTIAL_DECAY = True
    if not EXPONENTIAL_DECAY:
        # Calculate the decay for each step
        decay = Q_val_decay_rate[:, np.newaxis] *  indices * dt
    else:
        #decay = Q_val_decay_rate[:, np.newaxis] / np.exp(-indices * dt)
        # Take log instead of exp
        decay = 1 - (Q_val_decay_rate[:, np.newaxis] * np.log(indices + 1) / np.log(size + 1))
    # Calculate the values
    Q_val_abs = np.abs(Q_val)
    Q_val_abs *= Q_val_coef
    decay_Q = np.maximum(Q_val_abs[:, np.newaxis] - decay, 0)
    # Get the original sign
    minus_mask = Q_val < 0
    decay_Q[minus_mask] = -decay_Q[minus_mask]
    return decay_Q



def initDF(df, include_Q, include_RewardRate):
    df = df.copy()
    df["DVabs"] = df.DV.abs()
    df["SimRT"] = np.nan
    df["SimStartingPoint"] = np.nan
    df["SimChoiceCorrect"] = np.nan
    df["SimChoiceLeft"] = np.nan
    df["SimMatchReal"] = np.nan
    df["SessId"] = df.apply(lambda x: f"{x['Name']}_{x['Date']}_{x['SessionNum']}", axis=1)
    if include_Q:
        df["Q_L"] = .5
        df["Q_R"] = .5
        df["Q_val"] = .5
    if include_RewardRate:
        df["RewardRate"] = .5
    return df


def assignQuantiles(df, ncpus=1,
                    # Some files are saved twice due to cloud sync issues
                    uniq_cols = ["Name", "Date", "SessionNum", "TrialNumber", "TrialStartSysTime"]):
    df = df.copy()
    # dups_idxs = df[uniq_cols].duplicated(keep=False).values
    # display(df[dups_idxs])
    df = df[~df[uniq_cols].duplicated(keep="first")]
    null_time = df.calcStimulusTime.isnull()
    df_null_time = df[null_time]
    df = df[~null_time]
    df = splitStimulusTimeByQuantile(df, ncpus=ncpus)
    df = pd.concat([q_df for q_idx, q_df in df] + [df_null_time])
    # Sort
    df = df.sort_values(["Name", "Date", "SessionNum", "TrialNumber"])
    return df


def extractParams(driftFn, biasFn):
    common_params = signature(makeOneRun).parameters
    drift_fn_parans = signature(driftFn).parameters
    bias_fn_parans = signature(biasFn).parameters
    params = set(common_params) | set(drift_fn_parans) | set(bias_fn_parans)
    params = {param.upper() for param in params}
    params.remove("NONDECTIME")
    params.add("NON_DECISION_TIME")
    include_RewardRate = "REWARDRATE" in params
    include_Q = ("Q_VAL" in params) or \
                ("Q_VAL_COEF" in params) or \
                ("Q_VAL_DECAY_RATE" in params)

    return params, include_RewardRate, include_Q


def driftFnColsAndKwargs(driftFn):
    COMMON_ARGS = ['STARTING_POINT', 'NONDECTIME', 'DRIFT_COEF', 'DVS',
                   'NOISE', 'NOISE_SIGMA', 'DT']
    return _fnColsAndKargs(driftFn, COMMON_ARGS)


def noiseFnColsAndKwargs(noiseFn):
    COMMON_ARGS = ['SIZE', 'DT']
    # df_cols_names, kwargs_li = _fnColsAndKargs(noiseFn, COMMON_ARGS)
    # assert not len(df_cols_names), (
    #                     f"{noiseFn}: Found unexpected df_cols: {df_cols_names}")
    return _fnColsAndKargs(noiseFn, COMMON_ARGS)

def biasFnColsAndKwargs(biasFn):
    COMMON_ARGS = ['SIZE']
    # df_cols_names, kwargs_li = _fnColsAndKargs(biasFn, COMMON_ARGS)
    # assert not len(df_cols_names), (
    #                     f"{biasFn}: Found unexpected df_cols: {df_cols_names}")
    return _fnColsAndKargs(biasFn, COMMON_ARGS)

def _fnColsAndKargs(fn, COMMON_ARGS):
    df_cols_names = []
    kwargs_li = []
    default_kwargs = {}
    if isinstance(fn, partial):
        default_kwargs = fn.keywords
        fn = fn.func
    fn_params = signature(fn).parameters
    for param, param_info in fn_params.items():
        param_uuper = param.upper()
        if param_uuper in COMMON_ARGS:
            continue
        has_default = param in default_kwargs
        if has_default:
            # print(f"{fn.__name__}:{param} has default value: {param_info.default}")
            continue
        param_type = param_info.annotation
        # print("Param:", param_info, param_info == npt.NDArray)
        assert param_type is not None, f"{fn}:{param} has no annotation"
        if param_type == npt.NDArray:
            df_cols_names.append(param)
        elif param_type == float:
            kwargs_li.append(param)
        elif param_type == bool:
            # if param_info.default is not None:
            #     default_kwargs[param] = param_info.default
            # else:
            #     raise ValueError(f"Don't know how to handle bool param: {param}"
            #                       " without a default value")
            pass
        else:
            raise ValueError(f"{fn.__name__}:{param} - Unknown type: {param_type}")
    return df_cols_names, kwargs_li #, default_kwargs

def partialWithNames(fn, **kwargs):
    fn = partial(fn, **kwargs)
    fn.__name__ = fn.func.__name__ + "_".join(f"_{key}_{val}"
                                              for key, val in kwargs.items())
    return fn