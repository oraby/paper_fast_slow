from .dimreduc import DimReduc, DimReducWrapper
from .fastslowstats import splitDatTestQuantiles, splitDatTestAll, preprocessDF
from .statstest import StatsTest
import numpy as np
from tqdm.auto import tqdm
from collections import namedtuple
import pandas as pd
from functools import partial

def makeLongTraceID(df):
    df = df.copy()
    df["long_trace_id"] = df.ShortName + "_" + df.trace_id.astype(str)
    return df

def mergeROCWithStat(stats_df, roc_df):
    stats_df, roc_df = stats_df.copy(), roc_df.copy()
    stats_df = makeLongTraceID(stats_df)
    roc_df = makeLongTraceID(roc_df)
    assert len(stats_df) == len(roc_df)
    merge_on = ["long_trace_id", "data_col", "prior_data_col", "prior_data_val",
                "DVstr"]
    if "quantile_idx" in stats_df.columns:
       merge_on += ["quantile_idx"]
    # Do we need to handle nan prior_data_col?
    stats_df = pd.merge(stats_df, roc_df, how="inner", on=merge_on,
                        suffixes=["", "_roc"])
    keep_col = ["statistic_roc", "left_mean_roc", "right_mean_roc",
                                 "left_median_roc", "right_median_roc"]
    drop_col = [col for col in stats_df.columns
                if col.endswith("_roc") and col not in keep_col]
    stats_df = stats_df.drop(columns=drop_col)
    assert len(stats_df) == len(roc_df), (
                                      f"{len(stats_df) = } != {len(roc_df) = }")
    stats_df["IsROCLeftTuend"] = stats_df.statistic_roc > 0.5
    return stats_df


def assignBrainRegion(df, df_src):
  sess_br = df_src[["ShortName", "BrainRegion", "Layer"]]
  sess_br = sess_br.drop_duplicates()
  df = df.copy()
  sess_name_col = "sess_name" if "sess_name" in df.columns else "ShortName"
  uniq_sess = df[sess_name_col].unique()
  for sess_name in uniq_sess:
    ref_sess = sess_br[sess_br.ShortName == sess_name]
    br = ref_sess.BrainRegion
    assert len(br) == 1
    br = br.iloc[0]
    layer = ref_sess.Layer
    assert len(layer) == 1
    layer = layer.iloc[0]
    cur_sess_idx = df[sess_name_col] == sess_name
    cur_sess = df[cur_sess_idx]
    if "BrainRegion" not in df.columns or cur_sess["BrainRegion"].isnull().any():
      df.loc[cur_sess_idx, "BrainRegion"] = br
    if "Layer" not in df.columns or cur_sess["Layer"].isnull().any():
      df.loc[cur_sess_idx, "Layer"] = layer
  df["BrainRegion"] = df.BrainRegion.astype(int)
  if "ShortName" not in df.columns:
    df = df.rename(columns={"sess_name":"ShortName"})
  return df


def runPreprocessor(df, dimReducFn):
  dim_reduc_wrapper = DimReducWrapper(dimReducFn=dimReducFn,
                                      set_name="neuronal")
  df_reduc = preprocessDF(df.copy(), dim_reduc_wrapper=dim_reduc_wrapper,
                          filter_df_and_layer=True)
  return df_reduc

def _shuffleQuantiles(df):
    df = df.copy()
    old_col = df.quantile_idx.copy()
    df["quantile_idx"] = np.array(old_col.sample(frac=1).values)
    return df


RunData = namedtuple("RunData", ["df_src",
                                 "df_reduc_stats",
                                 "df_reduc_roc",
                                 "shortlong_df",
                                 "shortlong_quantiled_df"])

def runSplitFn(splitFn, df_reduc, statsTestFn):
    _short_long_df, p_li = splitFn(df_reduc.copy(), statsTestFn=statsTestFn)
    _short_long_df = assignBrainRegion(_short_long_df, df_src=df_reduc)
    return _short_long_df

def _genDataForDimReduc(df_src, dimReducFn, statsTestFn, assign_name=None):
    df_reduc = runPreprocessor(df_src, dimReducFn=dimReducFn)
    shortlong_df = runSplitFn(splitDatTestAll, df_reduc, statsTestFn)
    shortlong_quantiled_df = runSplitFn(splitDatTestQuantiles, df_reduc,
                                        statsTestFn)
    if assign_name is not None:
       df_reduc["Group"] = assign_name
       shortlong_df["Group"] = assign_name
       shortlong_quantiled_df["Group"] = assign_name
    return df_reduc, shortlong_df, shortlong_quantiled_df

def genRunDataInst(df, shuffle_quantiles : bool, defaultDimReducFn,
                   defaultStatsTestFn, assign_name=None,
                   rocDimReducFn=DimReduc.mean):
    df = df.copy()
    if shuffle_quantiles:
        df = _shuffleQuantiles(df)
    if assign_name is not None:
       df["Group"] = assign_name
    df_reduc_stats, shortlong_stats_df, shortlong_quantiled_stats_df =\
                  _genDataForDimReduc(df, defaultDimReducFn, defaultStatsTestFn,
                                      assign_name)
    # Create mean-based ROC
    roc_auc_test = partial(StatsTest.AUCPermutations, skip_permutations=True)
    df_reduc_roc, shortlong_roc_df, shortlong_quantiled_roc_df =\
                            _genDataForDimReduc(df, rocDimReducFn, roc_auc_test,
                                                assign_name)
    shortlong_df = mergeROCWithStat(shortlong_stats_df, shortlong_roc_df)
    shortlong_quantiled_df = mergeROCWithStat(shortlong_quantiled_stats_df,
                                              shortlong_quantiled_roc_df)
    run_data = RunData(df_src=df,
                       df_reduc_stats=df_reduc_stats,
                       df_reduc_roc=df_reduc_roc,
                       shortlong_df=shortlong_df,
                       shortlong_quantiled_df=shortlong_quantiled_df)
    return run_data
