from ..pipeline import pipeline
import numpy as np
import pandas as pd

class CalcResponseLiklihood(pipeline.DFProcessor):
    def __init__(self, threshold_val, before_sec, check_until_before_sec,
                 after_sec,  #check_until_after_sec
                 assign_cols=["BrainRegion", "Layer", "ShortName"],
                 set_name="neuronal"):
        self._threshold_val = threshold_val
        self._before_sec = before_sec
        self._check_until_before_sec = check_until_before_sec
        self._after_sec = after_sec
        # self._check_until_after_sec = check_until_after_sec
        self._assign_cols = assign_cols
        self._set_name = set_name
        self._df_res = None

    def process(self, df):
        assert df.ShortName.nunique() == 1
        # traces_liklihood_start = {}
        # traces_liklihood_already = {}
        counts = 0
        res_dict = {} # col:[] for col in self._assign_cols}
        for col in self._assign_cols:
            assert col in df.columns
            assert df[col].nunique() == 1
        # res_dict["trace_id"] = {}
        # res_dict["liklihood_start"] = []
        # res_dict["liklihood_already"] = []
        for row_idx, row in df.iterrows():
            acq_rate = row.acq_sampling_rate
            start = row.trace_start_idx
            before_idx = start - int(np.ceil(self._before_sec * acq_rate))
            after_idx = start + int(np.ceil(self._after_sec * acq_rate))
            relvant_trc_rng = np.arange(before_idx, after_idx + 1)
            check_until_before_idx = start - int(np.ceil(
                                       self._check_until_before_sec * acq_rate))
            if check_until_before_idx <= 0:
                continue # Should happen at max once
            counts += 1
            period_before_rng = np.arange(check_until_before_idx, before_idx)
            # check_until_after_idx = int(np.round(self._check_until_after_sec
            #                                       * acq_rate))
            for set_name, traces_dict in pipeline.getRowTracesSets(row).items():
                if set_name != self._set_name:
                    continue
                time_ax = pipeline.TIME_AX_LOOKUP[set_name]
                for trace_id, trace_data in traces_dict.items():
                    # traces_liklihood_start[trace_id] = \
                    #                    traces_liklihood_start.get(trace_id, 0)
                    # traces_liklihood_already[trace_id] = \
                    #                  traces_liklihood_already.get(trace_id, 0)
                    relevant_trace_data = trace_data.take(relvant_trc_rng,
                                                          axis=time_ax)
                    if relevant_trace_data.max() < self._threshold_val:
                        is_started = 0
                        is_already_started = 0
                    else:
                        period_before = trace_data.take(period_before_rng,
                                                        axis=time_ax)
                        if period_before.max() >= self._threshold_val:
                            # traces_liklihood_already[trace_id] += 1
                            is_already_started = 1
                            is_started = 0
                            # continue
                        else:
                            is_started = 1
                            is_already_started = 0
                            # traces_liklihood_start[trace_id] += 1
                # for trace_id, is_started in traces_liklihood_start.items():
                    # is_already_started = traces_liklihood_already[trace_id]
                    counts_li = res_dict.get(trace_id, [0, 0].copy())
                    counts_li[0] += is_started
                    counts_li[1] += is_already_started
                    res_dict[trace_id] = counts_li
        res_df_dict = {}
        res_df_dict["trace_id"] = []
        res_df_dict["start_count"] = []
        res_df_dict["already_started_count"] = []
        for trace_id, (started_count, already_started_count) in \
                                                               res_dict.items():
            if started_count > already_started_count:
                started_count += already_started_count
                already_started_count = 0
            else:
                already_started_count += started_count
                started_count = 0
            res_df_dict["trace_id"].append(trace_id)
            res_df_dict["start_count"].append(started_count)
            res_df_dict["already_started_count"].append(already_started_count)
        # print("Session:", df.ShortName.unique()[0])
        # for trace_id, llh in traces_liklihood_start.items():
        #     print(f"Trace:    {trace_id:04d} - Start: {llh}/{counts} "
        #           f"({100*llh/counts:.2g}%)")
        #     # llh = traces_liklihood_already.get(trace_id, 0)
        #     # print(f"    Already: {llh}/{counts} ({100*llh/counts:.2g})")
        res_df = pd.DataFrame(res_df_dict)
        res_df["trials_count"] = counts
        res_df["traces_count"] = len(res_dict)
        for col in self._assign_cols:
            res_df[col] = df[col].iloc[0]
        if self._df_res is None:
            self._df_res = res_df
        else:
            self._df_res = pd.concat([self._df_res, res_df], ignore_index=True)
        return df

    def getResults(self):
        return self._df_res

    def descr(self):
        return "TODO: Write the description"

def firingAroundEpoch(df, epoch, threshold_val, align_before_sec,
                      align_after_sec, check_until_before_sec):
        df = df.copy()
        df = df[df.epoch == epoch]
        assert all(df.sole_owner == False)
        calc_ll = CalcResponseLiklihood(threshold_val=threshold_val,
                                        before_sec=align_before_sec,
                                        after_sec=align_after_sec,
                                  check_until_before_sec=check_until_before_sec)
        chain = pipeline.Chain(
            # splitLevel(split_level),
            pipeline.BySession(),
                calc_ll,
                # tracesrestructure.AlignTraceAroundEpoch(epoch=epoch,
                #                              time_before_sec=align_before_sec,
                #                              time_after_sec=align_after_sec),
            pipeline.RecombineResults(),
        )
        chain.run(df)
        df_res = calc_ll.getResults()
        df_res["epoch"] = epoch
        df_res["threshold_val"] = threshold_val
        df_res["align_before_sec"] = align_before_sec
        df_res["align_after_sec"] = align_after_sec
        df_res["check_until_before_sec"] = check_until_before_sec
        return df_res
