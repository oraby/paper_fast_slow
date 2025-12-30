from .pipeline import (DFProcessor, TIME_AX_LOOKUP, createRowTracesSet,
                       getRowTracesSets)
from .utils import filterNanGaussianConserving
import numpy as np
from scipy.ndimage import gaussian_filter1d
import pandas as pd

class ByActivityRange(DFProcessor):
    def __init__(self, ascending=True, smooth_sigma=None, set_name=None):
        self._ascending = ascending
        self._smooth_sigma = smooth_sigma
        self._set_name = set_name

    def process(self, df):
        res_rows = []
        for row_id, row in df.iterrows():
            set_name_to_id_to_traces = {}
            for set_name, traces_dict in getRowTracesSets(row).items():
                if self._set_name is not None and self._set_name != set_name:
                    set_name_to_id_to_traces[set_name] = traces_dict
                    continue
                time_ax = TIME_AX_LOOKUP[set_name]
                keys, ranges, traces = zip(*[(key, trace.max() - trace.min(),
                                              trace)
                                             for key, trace in
                                                           traces_dict.items()])
                traces, keys = np.array(traces), np.array(keys)
                if self._smooth_sigma is not None:
                    traces = [gaussian_filter1d(trace_data,
                                                sigma=self._smooth_sigma,
                                                axis=time_ax)
                              for trace_data in traces]
                    traces = np.array(traces)
                new_idxs = np.argsort(ranges)
                if not self._ascending:
                    new_idxs = new_idxs[::-1]
                keys = keys[new_idxs]
                traces = traces[new_idxs]
                set_name_to_id_to_traces[set_name] = dict(zip(keys, traces))
            row = createRowTracesSet(row, set_name_to_id_to_traces)
            res_rows.append(row)
        return pd.DataFrame(res_rows)

    def descr(self):
        how = "ascendingly" if self._ascending else "descendingly"
        return f"Ordering traces {how} according to traces activity range"


class ByPeakActivityTime(DFProcessor):
    def __init__(self, ascending=True, smooth_sigma=None, set_name=None,
                 restrict_to_epochs=[]):
        self._ascending = ascending
        self._smooth_sigma = smooth_sigma
        self._set_name = set_name
        self._restrict_to_epochs = restrict_to_epochs

    def process(self, df):
        if len(self._restrict_to_epochs):
            assert "epochs_names" in df.columns
        res_rows = []
        for row_id, row in df.iterrows():
            set_name_to_id_to_traces = {}
            for set_name, traces_dict in getRowTracesSets(row).items():
                if self._set_name is not None and self._set_name != set_name:
                    set_name_to_id_to_traces[set_name] = traces_dict
                    continue
                time_ax = TIME_AX_LOOKUP[set_name]
                new_sort = ByPeakActivityTime\
                                .sortPeakActivityTime(row,
                                                    traces_dict,
                                                    time_ax,
                                                    self._ascending,
                                                    self._smooth_sigma,
                                                    self._restrict_to_epochs)
                set_name_to_id_to_traces[set_name] = new_sort
            row = createRowTracesSet(row, set_name_to_id_to_traces)
            res_rows.append(row)
        return pd.DataFrame(res_rows)

    @staticmethod
    def sortPeakActivityTime(row, traces_dict, time_ax, ascending=True,
                             smooth_sigma=None, restrict_to_epochs=[]):
        keys, traces = zip(*[(key, trace)
                             for key, trace in traces_dict.items()])
        traces, keys = np.array(traces), np.array(keys)
        if smooth_sigma is not None:
                traces = [#print(trace_data) or
                          # gaussian_filter1d(trace_data, sigma=smooth_sigma,
                          #                                     axis=time_ax)
                          filterNanGaussianConserving(trace_data,
                                                      sigma=smooth_sigma,
                                                      axis=time_ax)
                          for trace_data in traces]
                traces = np.array(traces)
        # calculate center of mass as the sum of moments of step from origin
        # zero (i.e index * value at that index), divided by the sum of masses
        # print("keys:", len(keys))
        # print("traces.shape:", traces.shape)
        # print("Range:", (1+np.arange(traces.shape[time_ax+1])))
        if len(restrict_to_epochs):
            used_ranges = [np.arange(rng[0], rng[1] + 1)
                           for name, rng in zip(row.epochs_names,
                                                row.epochs_ranges)
                           if name in restrict_to_epochs]
            used_ranges = np.concatenate(used_ranges)
            ref_traces = [trace.take(used_ranges, axis=time_ax)
                          for trace in traces]
            ref_traces = np.array(ref_traces)
        else:
            ref_traces = traces
        if np.isnan(ref_traces).all():
            return traces_dict.copy()
        BY_COM = False # Center-of-Mass
        if BY_COM:
            com = np.sum(np.nanarange(ref_traces.shape[time_ax+1]) * ref_traces,
                         axis=time_ax + 1)
            sum_of_masses = np.sum(ref_traces, axis=time_ax+1)
            # print("Numerator:", com)
            # print("sum_of_masses:", sum_of_masses)
            com = com / sum_of_masses
            com = np.round(com)
            com = com.astype(int)
            # print("COM2:", com)
            peak_activity = com    # np..(traces, axis=time_ax+1)
        else:
            # Fill completetly nan values with least values in the traces
            nan_entries = np.isnan(ref_traces)
            ref_traces[nan_entries] = ref_traces[~nan_entries].min(axis=None)
            peak_activity = np.nanargmax(ref_traces, axis=time_ax+1)
        new_idxs = np.argsort(peak_activity)
        # new_idxs = np.argsort(ranges)
        if not ascending:
            new_idxs = new_idxs[::-1]
        keys = keys[new_idxs]
        traces = traces[new_idxs]
        new_sort = dict(zip(keys, traces))
        return new_sort

    def descr(self):
        how = "ascendingly" if self._ascending else "descendingly"
        return f"Ordering activity by peak activity {how}"
