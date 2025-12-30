from ..pipeline import pipeline
import numpy as np
import pandas as pd


def loopConditions(df, epoch, splitFns, priorSplitFn=None, statsTestFn=None):
    assert statsTestFn is not None, (
                            "Update your code to explicitly pass this argument")
    res_li = []
    df = df.copy()
    # df = df[df.ShortName == df.ShortName.iloc[0]]
    for idx, splitFn in enumerate(splitFns):
        print("Processing splitFn:", splitFn)
        res = _loopRunOnEpoch(df, epoch=epoch, splitFn=splitFn,
                              statsTestFn=statsTestFn,
                              priorSplitFn=priorSplitFn)
        # with open(f"intermediate_result_{epoch}{idx}.pkl", "wb") as file:
        #     pickle.dump(res, file)
        res_li.append(res)
    res_df = pd.concat([res.getStatsResults() for res in res_li])
    return res_df, res_li

def _loopRunOnEpoch(df, epoch, splitFn, statsTestFn, priorSplitFn=None):
        # if statsTestFn is None:
        #     statsTestFn = defaultStatsTestFn
        stat_test_processor = RunStatTest(dfSplitFn=splitFn,
                                          priorDfSplitFn=priorSplitFn,
                                          statsTestFn=statsTestFn)
        df = df.copy()
        df = df[df.epoch == epoch]
        chain = pipeline.Chain(
                pipeline.BySession(),
                    stat_test_processor,
                pipeline.RecombineResults(),
        )
        chain.process(df)
        return stat_test_processor

class RunStatTest(pipeline.DFProcessor):
    def __init__(self, dfSplitFn, statsTestFn, priorDfSplitFn=None,
                 statsTestFn_kargs={}, dim_reduc_info=None):
        self._splitFn = dfSplitFn
        self._priorDfSplitFn = priorDfSplitFn
        self._statsTestFn = statsTestFn
        self._statsTestFn_kargs = statsTestFn_kargs
        self._res_df = None
        self._dim_reduc_info = dim_reduc_info

    def process(self, df):
        assert df.ShortName.nunique() == 1, (
                                   "Didn't implement yet more than one session")
        assert "neuronal_reduc" in df.columns
        # print("df.ShortName:", df.ShortName.iloc[0], " - df.len:", len(df))
        # from .splitConditions import SplitResult
        org_df_len = len(df)
        if self._priorDfSplitFn is not None:
            prior_data_col, prior_split_result1, prior_split_result2 = \
                                                        self._priorDfSplitFn(df)
            df = pd.concat([prior_split_result1.df, prior_split_result2.df])
        else:
            prior_data_col = None
        data_col, split_result_left, split_result_right = self._splitFn(df)
        df1 = split_result_left.df    # split_result_left.label
        df2 = split_result_right.df # split_result_right.label
        self._data_col = data_col
        data_col_allowed_vals = [split_result_left.val, split_result_right.val]
        trace_id_li = []
        trace_data_li = []
        data_col_val_li = []
        prior_data_col_val_li = []
        dim_reduc_name_li = []
        trial_numbers = []

        for df in [df1, df2]:
            for row_idx, row in df.iterrows():
                neurons_reduc = row["neuronal_reduc"]
                dim_reduc_name = row["DimReduc"]
                for trace_id, trace_val in neurons_reduc.items():
                    if prior_data_col is not None:
                        prior_data_col_val_li.append(row[prior_data_col])
                    else:
                        prior_data_col_val_li.append(None)
                    trace_id_li.append(trace_id)
                    data_val = row[data_col]
                    assert data_val in data_col_allowed_vals
                    data_col_val_li.append(data_val)
                    trace_data_li.append(trace_val)
                    dim_reduc_name_li.append(dim_reduc_name)
                    # if trace_id == 17:
                trial_numbers.append(row.TrialNumber)

        # short_names = df.ShortName.unique()
        # print("df.ShortName:", short_names, " - data col:", self._data_col,
        #       " - prior data col:", prior_data_col," - df.len:", org_df_len,
        #       "used trials len:", len(trial_numbers))
        assert len(trial_numbers) == len(np.unique(trial_numbers))

        prepare_df = pd.DataFrame({"trace_id":trace_id_li,
                                   "data_val":data_col_val_li,
                                   "trace_data":trace_data_li,
                                   "prior_data_val":prior_data_col_val_li,
                                   "dim_reduc_name":dim_reduc_name_li})
        # print("DF len:", len(df))
        if len(df) > 0: # 10
            self._calcROC(prepare_df, prior_data_col,
                          sess_name=df.ShortName.iloc[0],
                          left_val=split_result_left.val,
                          right_val=split_result_right.val)
        return df

    def _calcROC(self, preprocessed_data_df, prior_data_col, sess_name,
                 left_val, right_val):
        results = {
            "trace_id":[],
            "data_col":[],
            "data_val_left":[],
            "data_val_right":[],
            "prior_data_col":[],
            "prior_data_val":[],
            "ShortName":[],
            "dim_reduc_name":[],
            "stat_test_name":[],
            "left_mean":[],
            "right_mean":[],
            "left_median":[],
            "right_median":[],
            # "statistic":[], # Same as auc
            # "null_distribution":[], # A list of all the null distribution values
        }
        stat_test_name = self._statsTestFn("name", None)

        data_col_allowed_vals = [left_val, right_val]
        warned_once = set()

        for trace_id, trace_df in preprocessed_data_df.groupby("trace_id"):
            for prior_val, prior_df in trace_df.groupby("prior_data_val",
                                                        dropna=False):
                dim_reduc_name = prior_df.dim_reduc_name.unique()
                assert len(dim_reduc_name) == 1
                dim_reduc_name = dim_reduc_name[0]
                data_col_vals = prior_df.data_val.unique()
                if len(data_col_vals) < 2:
                    # Add string value to support np.nan
                    if str(prior_val) not in warned_once:
                        print(f"**** ### Skipping {prior_data_col}={prior_val} "
                              f"because only {len(data_col_vals)} for "
                              f"{self._data_col}")
                        warned_once.add(str(prior_val))
                    continue
                assert all([val in data_col_allowed_vals
                            for val in data_col_vals])
                left_data = prior_df[prior_df.data_val == left_val].trace_data
                right_data = prior_df[prior_df.data_val == right_val].trace_data
                left_data = np.array(left_data.tolist()) # Do we need this here?
                # print("Lens:", [len(arr) for arr in right_data.tolist()])
                right_data = np.array(right_data.tolist())
                SUBSAMPLE = False
                if SUBSAMPLE:
                    len_left, len_right = len(left_data), len(right_data)
                    min_samples = min(len_left, len_right)
                    # print(f"{left_data.shape = } - {right_data.shape = } - "
                    #       f"{min_samples = }")
                    if min_samples < len_left:
                        idxs = np.random.choice(len_left, min_samples,
                                                replace=False)
                        left_data = left_data[idxs]
                    if min_samples < len_right:
                        idxs = np.random.choice(len_right, min_samples,
                                                replace=False)
                        right_data = right_data[idxs]
                # if trace_id == 17:
                #     print(f"prior_val= {prior_val} - "
                #           f"data_col left={len(left_data)} - "
                #           f"right={len(right_data)}")
                left_mean, right_mean = left_data.mean(), right_data.mean()
                left_median, right_median = \
                                     np.median(left_data), np.median(right_data)
                results["trace_id"].append(trace_id)
                results["data_col"].append(self._data_col)
                results["data_val_left"].append(left_val)
                results["data_val_right"].append(right_val)
                results["prior_data_col"].append(prior_data_col)
                results["prior_data_val"].append(prior_val)
                results["ShortName"].append(sess_name)
                results["dim_reduc_name"].append(dim_reduc_name)
                results["stat_test_name"].append(stat_test_name)
                results["left_mean"].append(left_mean)
                results["right_mean"].append(right_mean)
                results["left_median"].append(left_median)
                results["right_median"].append(right_median)

                if self._dim_reduc_info is not None:
                    trace_dim_reduc_info = self._dim_reduc_info[trace_id]
                    for col, col_val in trace_dim_reduc_info.items():
                        if col not in results:
                            results[col] = []
                        results[col].append(col_val)
                test_result = self._statsTestFn(left_data, right_data,
                                                **self._statsTestFn_kargs)
                for col, col_val in test_result.resultDict().items():
                    if col not in results:
                        results[col] = []
                    results[col].append(col_val)
        stats_results_df = pd.DataFrame(results)
        if self._res_df is None:
            self._res_df = stats_results_df
        else:
            self._res_df = pd.concat([self._res_df, stats_results_df])

    def getStatsResults(self):
        return self._res_df

    def descr(self):
        return "Calculating ROC and ROC p-value"
