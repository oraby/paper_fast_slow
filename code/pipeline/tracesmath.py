from .pipeline import (DFProcessor, TIME_AX_LOOKUP, createRowTracesSet,
                                             getRowTracesSets)
from .utils import filterNanGaussianConserving
import numpy as np
import pandas as pd
from scipy.stats import sem
from scipy.ndimage import gaussian_filter1d
from typing import List, Union, Callable

class TraceAvg(DFProcessor):
    def __init__(self, *, avg_rows: bool = True, avg_row_traces: bool = False,
                set_name=None, create_sem_set: bool = True,
                assigned_df_col_avg_row_name: str = None,
                get_name_df_cols: Union[str, List[str],
                                        Callable[["row"], str]] = None):
        '''If assigned_df_col_avg_row_name is None, then the averaged row is
        assigned no name.

        If assigned_df_col_avg_row_name is a string:
        - get_name_df_cols is None:
            * assigned_df_col_avg_row_name must be a valid df column,
            * the averaged row will be both assigned the averaged id result to
              that column, and
            * the value assigned to that column will be the column
              value + 'Avg. Trace': f"{assigned_df_col_avg_row_name} Avg. Trace"

        - get_name_df_cols is a string:
            * assigned_df_col_avg_row_name is either be a new or an existing df
              column
            * get_name_df_cols must be the name of an existing df column
            * the assigned value to assigned_df_col_avg_row_name will be the
              value of the get_name_df_cols column.

        - get_name_df_cols is a list of strings:
            * assigned_df_col_avg_row_name is either be a new or an existing df
              column
            * get_name_df_cols must consist be the names of an existing df
              columns
            * the assigned value to assigned_df_col_avg_row_name will be the
              name of of each get_name_df_cols column followed by its value:
              f"f{col1_name}_{col1_value}_{col2_name}_{col2_value}..."

        - get_name_df_cols is a callable:
            * assigned_df_col_avg_row_name is either be a new or an existing df
              column
            * get_name_df_cols should have the signature def func(row) -> str
            * the return value from calling the function will be assigned to the
              assigned_df_col_avg_row_name column.

        get_name_df_cols cannot be not None if assigned_df_col_avg_row_name is
        None
        '''
        assert avg_rows or avg_row_traces, "No average is specified"
        self._avg_rows = avg_rows
        self._avg_row_traces = avg_row_traces
        self._set_name = set_name
        self._create_sem_set = create_sem_set
        self._assigned_df_col_avg_row_name = assigned_df_col_avg_row_name
        self._get_name_df_cols = get_name_df_cols
        if get_name_df_cols is not None:
            assert assigned_df_col_avg_row_name is not None, ("Don't know what "
                "toassign get_name_df_cols into which column, "
                "assigned_df_col_avg_row_name is None ")
        self._makeRowName = (get_name_df_cols if callable(get_name_df_cols) else
                             self._defaultMakeName)

    def process(self, df):
        if self._avg_row_traces:
            res_rows = []
            for row_id, row in df.iterrows():
                set_name_to_id_to_traces = {}
                trc_rng = np.arange(row.trace_start_idx,row.trace_end_idx+1)
                for set_name, traces_dict in getRowTracesSets(row).items():
                    if (self._set_name is not None and
                        self._set_name != set_name):
                        set_name_to_id_to_traces[set_name] = traces_dict
                        continue
                    # avg_trace = np.nanmean(list(traces_dict.values()), axis=0)
                    set_axis = TIME_AX_LOOKUP[set_name]
                    traces_data = (t.take(trc_rng, axis=set_axis)
                                   for t in traces_dict.values())
                    avg_trace = np.nanmean(traces_data, axis=0)
                    set_name_to_id_to_traces[set_name] = {
                                                        "Traces Avg.":avg_trace}
                    if self._create_sem_set:
                        std_trace = sem(traces_data, axis=0, nan_policy='omit')
                        set_name_to_id_to_traces[f"{set_name}_sem"] = {
                                                        "Traces SEM": std_trace}
                row = row.copy()
                createRowTracesSet(row, set_name_to_id_to_traces)
                del set_name_to_id_to_traces # Avoid reusing the variable again
                                             # unless we are re-declaring it
                res_rows.append(row)
            df = pd.DataFrame(res_rows)
        if self._avg_rows:
            traces_sets_ids, traces_sets_ids_counts = \
                                               self._getDFTracesIdsAndCounts(df)
            set_name_to_id_to_traces = {}
            set_name_to_id_to_traces_count = {}
            for set_name, traces_ids_set in traces_sets_ids.items():
                if self._set_name is not None and self._set_name != set_name:
                    set_name_to_id_to_traces[set_name] = traces_dict
                    continue
                cur_set_traces_mean = {}
                cur_set_traces_count = {}
                if self._create_sem_set:
                    cur_set_traces_sem = {}
                for trace_id in traces_ids_set:
                    traces_li = self._getTraceInstancesAcrossDF(df, set_name,
                                                                trace_id)
                    ex_len = len(traces_li[0])
                    debug_traces_len = [len(trace) for trace in traces_li]
                    if not all([trace_len == ex_len
                                for trace_len in debug_traces_len]):
                        print("Before crash: ", df.ShortName.iloc[0])
                        [print(row[1].epochs_names, row[1].TrialNumber,
                               trace_len)
                         for trace_len, row in zip(debug_traces_len,
                                                   df.iterrows())]
                    assert all([trace_len == ex_len
                                for trace_len in debug_traces_len]), (
                        f"Not all the traces has the same length in {set_name}"
                        f"{debug_traces_len} - {df.ShortName.iloc[0]}")
                    cur_set_traces_mean[trace_id] = np.nanmean(traces_li,
                                                               axis=0)
                    if self._create_sem_set:
                        cur_set_traces_sem[trace_id] = sem(traces_li, axis=0,
                                                           nan_policy='omit')
                    # TODO: Find a way to include both information:
                    cur_count_if_exist = traces_sets_ids_counts[set_name].get(
                                                                       trace_id)
                    if cur_count_if_exist is not None:
                        cur_set_traces_count[trace_id] = cur_count_if_exist
                    else:
                        cur_set_traces_count[trace_id] = len(traces_li)
                set_name_to_id_to_traces[set_name] = cur_set_traces_mean
                if self._create_sem_set:
                    set_name_to_id_to_traces[f"{set_name}_sem"] = \
                                                              cur_set_traces_sem
                set_name_to_id_to_traces_count[set_name] = cur_set_traces_count
            ex_row = df.iloc[0].copy() # Any row would do
            ex_row["avg_traces_count"] = set_name_to_id_to_traces_count
            ex_row = createRowTracesSet(ex_row, set_name_to_id_to_traces)
            if self._assigned_df_col_avg_row_name is not None:
                res_str = self._makeRowName(ex_row)
                # print(f"{self._assigned_df_col_avg_row_name} will be "
                #       f"assigned: {res_str}")
                ex_row[self._assigned_df_col_avg_row_name] = res_str
            df = pd.DataFrame([ex_row])
        return df

    def _getDFTracesIdsAndCounts(self, df):
        set_name_to_traces_ids = {}
        traces_counts_exist = "avg_traces_count" in df
        set_name_to_traces_counts = {}
        for row_id, row in df.iterrows():
            if traces_counts_exist:
                traces_counts_dict = row.avg_traces_count
            else:
                traces_counts_dict = {}
            for set_name, trace_dict in getRowTracesSets(row).items():
                ids_set = set_name_to_traces_ids.get(set_name, set())
                ids_set.update(trace_dict.keys())
                set_name_to_traces_ids[set_name] = ids_set
                traces_counts_ids = set_name_to_traces_counts.get(set_name, {})
                traces_counts_ids.update({k:traces_counts_ids.get(k, 0) + v
                                        for k, v in traces_counts_dict.items()})
                set_name_to_traces_counts[set_name] = traces_counts_ids
        return set_name_to_traces_ids, set_name_to_traces_counts

    def _getTraceInstancesAcrossDF(self, df, set_name, trace_id):
        traces_li = []
        set_axis = TIME_AX_LOOKUP[set_name]
        for row_id, row in df.iterrows():
            traces_dict = getRowTracesSets(row)[set_name]
            trace = traces_dict.get(trace_id)
            if trace is not None:
                set_axis = TIME_AX_LOOKUP[set_name]
                cur_trace = traces_dict[trace_id]
                trc_rng = np.arange(row.trace_start_idx, row.trace_end_idx+1)
                cur_trace = cur_trace.take(trc_rng, axis=set_axis)
                traces_li.append(cur_trace)
        return traces_li

    def _defaultMakeName(self, row):
        if self._get_name_df_cols is not None:
            if isinstance(self._get_name_df_cols, list):
                name = [f"{col}_{row[col]}" for col in self._get_name_df_cols]
                name = "_".join(name)
            else:
                name = row[self._get_name_df_cols]
        else:
            name = f"{row[self._assigned_df_col_avg_row_name]} Avg. Trace"
        return name

    def descr(self) -> str:
        return ("Average traces (TODO: Describe better how).")


class DivTraces(DFProcessor):
    def __init__(self, div_field, numer_val, denom_val):
        self._div_field = div_field
        self._numer = numer_val
        self._denom = denom_val

    def process(self, df):
        numer_df = df[df[self._div_field] == self._numer]
        denom_df = df[df[self._div_field] == self._denom]
        assert len(numer_df) == len(denom_df)
        res_rows = []
        for (_idx, row_numer), (_idx, row_denom) in zip(numer_df.iterrows(),
                                                        denom_df.iterrows()):
            numer_traces_sets = getRowTracesSets(row_numer)
            denom_traces_sets = getRowTracesSets(row_denom)
            assert numer_traces_sets.keys() == denom_traces_sets.keys(), (
                                         "Not the same traces sets names exist")
            res_sets_traces = {}
            for set_name, numer_traces_dict in numer_traces_sets.items():
                denom_traces_dict = denom_traces_sets[set_name]
                assert numer_traces_dict.keys() == denom_traces_dict.keys()
                res_sets_traces[set_name] = {}
                for trace_id, numer_trace_data in numer_traces_dict.items():
                    raise NotImplementedError(
                             "Take the correct indices using trace_data.take()")
                    denom_trace_data = denom_traces_dict[trace_id]
                    div_trace = numer_trace_data/denom_trace_data
                    res_sets_traces[set_name][trace_id] = div_trace
            ex_row = row_numer.copy()
            ex_row = createRowTracesSet(ex_row, res_sets_traces)
            res_rows.append(ex_row)
        return pd.DataFrame(res_rows)

    def descr(self):
        return (f"Dividing traces: ({self._div_field}={self._numer})/"
                        f"({self._div_field}={self._denom})")


class DiffTraces(DFProcessor):
    def __init__(self, col_name, left_side, right_side, use_abs=False):
        self._col_name = col_name
        self._left_side = left_side
        self._right_side = right_side
        self._use_abs = use_abs

    def process(self, df):
        left_df = df[df[self._col_name] == self._left_side]
        right_df = df[df[self._col_name] == self._right_side]
        assert len(left_df) == len(right_df), (
                      f"Left len: {len(left_df)} - Right len: {len(right_df)}")
        res_rows = []
        for (_idx, left_row), (_idx, right_row) in zip(left_df.iterrows(),
                                                       right_df.iterrows()):
            left_traces_sets = getRowTracesSets(left_row)
            right_traces_sets = getRowTracesSets(right_row)
            assert left_traces_sets.keys() == right_traces_sets.keys(), (
                                         "Not the same traces sets names exist")
            left_trc_rng = np.arange(left_row.trace_start_idx,
                                     left_row.trace_end_idx + 1)
            right_trc_rng = np.arange(right_row.trace_start_idx,
                                      right_row.trace_end_idx + 1)
            assert left_trc_rng.shape == right_trc_rng.shape
            res_sets_traces = {}
            for set_name, left_traces_dict in left_traces_sets.items():
                right_traces_dict = right_traces_sets[set_name]
                assert left_traces_dict.keys() == right_traces_dict.keys()
                time_ax = TIME_AX_LOOKUP[set_name]
                res_sets_traces[set_name] = {}
                for trace_id, left_trace_data in left_traces_dict.items():
                    left_trace_data = left_trace_data.take(left_trc_rng,
                                                           axis=time_ax)
                    right_trace_data = right_traces_dict[trace_id]
                    right_trace_data = right_trace_data.take(right_trc_rng,
                                                             axis=time_ax)
                    # We already asserted above that both has the same length
                    diff_trace = left_trace_data - right_trace_data
                    if self._use_abs:
                        diff_trace = np.abs(diff_trace)
                    res_sets_traces[set_name][trace_id] = diff_trace
            ex_row = left_row.copy()
            ex_row = createRowTracesSet(ex_row, res_sets_traces)
            res_rows.append(ex_row)
        return pd.DataFrame(res_rows)

    def descr(self):
        c = "|" if self._use_abs else ""
        return (f"Diffing traces: {c}{self._col_name}={self._left_side} - "
                f"{self._col_name}={self._right_side}{c}")

class GaussianFilter(DFProcessor):
    def __init__(self, sigma, set_name=None, copy_data=False):
        self._sigma = sigma
        self._set_name = set_name
        self._copy_data = copy_data

    def process(self, df):
        res_rows = []
        for row_id, row in df.iterrows():
            trc_rng = np.arange(row.trace_start_idx, row.trace_end_idx + 1)
            set_name_to_id_to_traces = {}
            for set_name, traces_dict in getRowTracesSets(row).items():
                if self._set_name is not None and self._set_name != set_name:
                    set_name_to_id_to_traces[set_name] = traces_dict
                    continue
                time_ax = TIME_AX_LOOKUP[set_name]
                new_traces_dict = {}
                for trace_id, trace_data in traces_dict.items():
                    trace_data = trace_data.take(trc_rng, axis=time_ax)
                    if self._copy_data:
                        trace_data = trace_data.copy()
                    # trace_data = gaussian_filter1d(trace_data,
                    #                                sigma=self._sigma,
                    #                                axis=time_ax)
                    trace_data = filterNanGaussianConserving(trace_data,
                                                             sigma=self._sigma,
                                                             axis=time_ax)
                    new_traces_dict[trace_id] = trace_data
                set_name_to_id_to_traces[set_name] = new_traces_dict
            row = createRowTracesSet(row, set_name_to_id_to_traces)
            res_rows.append(row)
        return pd.DataFrame(res_rows)

    def descr(self):
        return (
             f"Smoothing traces with gaussian filter with sigma: {self._sigma}")
