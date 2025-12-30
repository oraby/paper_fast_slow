from . import splitconditions as sc
from .runstattest import loopConditions
try:
    from IPython.display import display
except ModuleNotFoundError:
    pass
import numpy as np
import pandas as pd

def samplingPrevious(df, statsTestFn):
    return loopConditions(df=df,
                          epoch="Sampling",
                          statsTestFn=statsTestFn,
                          splitFns=[sc.conditionsPrevLeftRight,
                                    sc.conditionsPrevCorrectIncorrect,
                                    sc.conditionsPrevEasyDifficult
                          ],
    )

def samplingCurrent(df, statsTestFn):
    return loopConditions(df=df,
                          epoch="Sampling",
                          statsTestFn=statsTestFn,
                          splitFns=[sc.conditionsLeftRight,
                                    #    sc.conditionsCorrectIncorrect,
                                    sc.conditionsEasyDifficult
                          ],
                          # permutation_test_kargs={"n_resamples":300}
    )

def samplingCurGivenPriors(df, statsTestFn):
    df_res_li, processors_li = [], []
    for prevSplitFn in [sc.conditionsPrevLeftRight,
                        sc.conditionsPrevCorrectIncorrect,
                        sc.conditionsPrevEasyDifficult
                        ]:
        res = loopConditions(df=df.copy(),
                             epoch="Sampling",
                             statsTestFn=statsTestFn,
                             splitFns=[sc.conditionsLeftRight,
                                       # sc.conditionsCorrectIncorrect,
                                       sc.conditionsEasyDifficult],
                             priorSplitFn=prevSplitFn)
        df_res_li.append(res[0])
        processors_li.append(res[1])
    return pd.concat(df_res_li), processors_li

def samplingCurEvdGivenCurDirection(df, statsTestFn):
    df_res_li, processors_li = [], []
    for epoch in ["Sampling", "Feedback Start"]:
        res = loopConditions(df=df.copy(),
                             epoch=epoch,
                             statsTestFn=statsTestFn,
                             splitFns=[sc.conditionsEasyDifficult],
                             priorSplitFn=sc.conditionsLeftRight)
        df_res_li.append(res[0])
        processors_li.append(res[1])
    return pd.concat(df_res_li), processors_li

def _getValsMapping(test_df):
    test_df = test_df.copy()
    vals_mapping = {}
    for fnName in dir(sc):
        if fnName.startswith("conditions"):
            fn = getattr(sc, fnName)
            col_label, split_res1, split_res2 = fn(test_df)
            vals_mapping[col_label] = {split_res1.val : split_res1.label,
                                       split_res2.val : split_res2.label}
    return vals_mapping

def prepareDf(df, src_df, rename_cols=True):
    df["ShortName"] = df.sess_name
    df = df.drop(columns=["sess_name"])
    df["BrainRegion"] = None
    df["Layer"] = None
    for short_name in df.ShortName.unique():
        ex_src_row = src_df[src_df.ShortName == short_name].iloc[0]
        df.loc[df.ShortName == short_name, "BrainRegion"] = \
                                                          ex_src_row.BrainRegion
        df.loc[df.ShortName == short_name, "Layer"] = ex_src_row.Layer
    # df["trace_id"] = df.ShortName + " " + df.trace_id.astype(str)
    df["trace_id"] = df.ShortName + df.trace_id.astype(str)
    df["IsLeftTunedMean"] = df.left_mean >= df.right_mean
    df["IsLeftTunedMedian"] = df.left_median >= df.right_median
    df = df.reset_index(drop=True)
    df["statistic2"] = np.nan
    col = "IsLeftTunedStats"
    df[col] = np.nan
    mw_idxs = df.stat_test_name == "MannWhitneyU"
    df_mw = df[mw_idxs]
    df.loc[mw_idxs, "statistic2"] = \
                df_mw.num_samples_left*df_mw.num_samples_right - df_mw.statistic
    df_mw = df[mw_idxs] # Re-evaluate df_mw after adding statistic2
    df.loc[mw_idxs, col] = df_mw.statistic >= df_mw.statistic2
    auc_idxs = df.stat_test_name == "AUC_Permutations"
    df.loc[auc_idxs, col] = df.loc[auc_idxs, "statistic"] >= 0.5
    # display(df)
    assert df[col].isna().sum() == 0, display(df[df[col].isna()])
    if not rename_cols:
        return df
    col_vals_labels_mapping = _getValsMapping(test_df=src_df)
    print("col_vals_labels_mapping:", col_vals_labels_mapping)
    for col_name, vals_labels_mapping in col_vals_labels_mapping.items():
        target_rows = df.data_col == col_name
        prior_target_rows = df.prior_data_col == col_name
        for org_val, mapped_val in vals_labels_mapping.items():
            df.loc[target_rows & (df.data_val_left == org_val),
                         "data_val_left"] = mapped_val
            df.loc[target_rows & (df.data_val_right == org_val),
                         "data_val_right"] = mapped_val
            df.loc[prior_target_rows & (df.prior_data_val == org_val),
                         "prior_data_val"] = mapped_val

    for col_name in ["data_col", "data_val_left", "data_val_right",
                     "prior_data_col", "prior_data_val"]:
        for old, new in [("ChoiceLeft", "Direction"),
                         ("IsEasy", "Difficulty"),
                         ("ChoiceCorrect", "Outcome")]:
            old_str_mask = df[col_name].str.contains(old)
            old_str_mask = old_str_mask.fillna(False)
            df.loc[old_str_mask, col_name] = \
                            df.loc[old_str_mask, col_name].str.replace(old, new)
        prev_str_mask = df[col_name].str.contains("Prev")
        prev_str_mask = prev_str_mask.fillna(False)
        df.loc[prev_str_mask, col_name] = \
                df.loc[prev_str_mask, col_name].str.replace("Prev", "Previous ")
        if "data_col" in col_name:
            df.loc[~prev_str_mask, col_name] = \
                                   "Current " + df.loc[~prev_str_mask, col_name]
    display(df[df.prior_data_col.notnull()])
    display(df[["data_col", "data_val_left", "data_val_right",
                "prior_data_col", "prior_data_val"]].value_counts(dropna=False))
    return df
