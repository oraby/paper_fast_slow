from .pipeline import (DFProcessor, updateRowTracesSet, getRowTracesSets,
                       TIME_AX_LOOKUP)
from .utils import countTransientsRatePerMin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

class WidthQuantileFilter(DFProcessor):
    def __init__(self, bottom=0.25, top=0.75):
        self._bottom = bottom
        self._top = top

    def process(self, df):
        trace_width = df.trace_end_idx - df.trace_start_idx + 1
        bottom, top = trace_width.quantile([self._bottom, self._top])
        return df[(bottom <= trace_width) & (trace_width <= top)]

    def descr(self):
        return (f"Data filtered between bottom {self._bottom} quantile and top "
                f"{self._top} quantile")


class InsignficantActivityFilter(DFProcessor):
    def __init__(self, accept_activity_below=0.25, accept_activity_above=0.75,
                 set_name="neuronal"):
        self._accept_activity_below = accept_activity_below
        self._accept_activity_above = accept_activity_above
        self._set_name = set_name

    def process(self, df):
        assert len(df) == 1, "Please pass a single-row df"
        set_name_to_id_to_traces = {}
        row = df.iloc[0].copy()
        for set_name, traces_dict in getRowTracesSets(row).items():
            if self._set_name is not None and self._set_name != set_name:
                set_name_to_id_to_traces[set_name] = traces_dict
                continue
            time_ax = TIME_AX_LOOKUP[set_name]
            traces_id_to_traces_data = {trace_id:trace_data
                for trace_id, trace_data in traces_dict.items()
                if ((np.min(trace_data, axis=time_ax) <=
                                                    self._accept_activity_below)
                    or (self._accept_activity_above <=
                                             np.max(trace_data, axis=time_ax)))}
            set_name_to_id_to_traces[set_name] = traces_id_to_traces_data
        last_set_name = set_name # Being too verbose, also I don't know what t
                                 # do if one set is empty while the other isn't.
        if len(set_name_to_id_to_traces[last_set_name]):
            row = updateRowTracesSet(row, set_name_to_id_to_traces)
            return pd.DataFrame([row])
        else:
            return pd.DataFrame(columns=df.columns) # Return an empty dataframe

    def descr(self):
        return (f"Traces with min and max activity between "
                f"{self._accept_activity_below} ->    "
                f"{self._accept_activity_above} are filtered out")

class SmallRangeFilter(DFProcessor):
    def __init__(self, _range, set_name="neuronal"):
        self._range = _range
        self._set_name = set_name

    def process(self, df):
        assert len(df) == 1, "Please pass a single-row df"
        set_name_to_id_to_traces = {}
        row = df.iloc[0].copy()
        for set_name, traces_dict in getRowTracesSets(row).items():
            if self._set_name is not None and self._set_name != set_name:
                set_name_to_id_to_traces[set_name] = traces_dict
                continue
            time_ax = TIME_AX_LOOKUP[set_name]
            traces_id_to_traces_data = {trace_id:trace_data
                for trace_id, trace_data in traces_dict.items()
                if (np.max(trace_data, axis=time_ax) -
                    np.min(trace_data, axis=time_ax)) >= self._range}
            set_name_to_id_to_traces[set_name] = traces_id_to_traces_data
        last_set_name = set_name # Being too verbose, also I don't know what to
                                 # do if one set is empty while the other isn't.
        if len(set_name_to_id_to_traces[last_set_name]):
            row = updateRowTracesSet(row, set_name_to_id_to_traces)
            return pd.DataFrame([row])
        else:
            return pd.DataFrame(columns=df.columns) # Return an empty dataframe

    def descr(self):
        return (f"Traces with activty range less than {self._range}")


class InactiveTraceFilter(DFProcessor):
    def process(self, df):
        # assert len(df) == 1, "Please pass a single-row df"
        res_rows = []
        for _, row in df.iterrows():
            set_name_to_id_to_traces = {}
            traces_stats = row.traces_stats
            for set_name, traces_dict in getRowTracesSets(row).items():
                new_traces_dict = {k:v for k,v in traces_dict.items()
                                   if traces_stats[k]["is_active"]}
                set_name_to_id_to_traces[set_name] = new_traces_dict
            row = updateRowTracesSet(row, set_name_to_id_to_traces)
            res_rows.append(row)
        return pd.DataFrame(res_rows)

    def descr(self):
        return "Removed traces that were marked as inactive"


class FilterLowRatePosTransients(DFProcessor):
    def __init__(self, min_pos_rate_per_min, valid_transients_min_dur_sec,
                 set_name=None):
        self._min_pos_rate_per_min = min_pos_rate_per_min
        self._valid_transients_min_dur_sec = valid_transients_min_dur_sec
        self._set_name = set_name

    def process(self, df):
        res_rows = []
        for _, row in df.iterrows():
            row = row.copy()
            fps = row.acq_sampling_rate
            set_name_to_id_to_traces = {}
            traces_stats = row.traces_stats
            for set_name, traces_dict in getRowTracesSets(row).items():
                if self._set_name is not None and self._set_name != set_name:
                    set_name_to_id_to_traces[set_name] = traces_dict
                    continue
                new_traces_dict = {}
                axis = TIME_AX_LOOKUP[set_name]
                accepted_count = 0
                for trace_id, trace_data in traces_dict.items():
                    stats = traces_stats[trace_id]
                    baseline, z_score = stats["baseline"], stats["zscore1"]
                    # TODO: Take along axis
                    data = trace_data.take(np.arange(row.trace_start_idx,
                                                     row.trace_end_idx+1),
                                           axis=axis)
                    (transients_rate, pos_thres, accepted_pos_transients,
                     pos_transients, mv_avg_trace, mv_avg_win, MIN_FRAMES) = \
                        countTransientsRatePerMin(trace=data,
                                                  baseline=baseline,
                                                  z_score=z_score,
                                                  frame_rate=fps,
                                                  valid_transients_min_dur_sec=\
                                             self._valid_transients_min_dur_sec)
                    is_accepted = transients_rate >= self._min_pos_rate_per_min
                    if is_accepted:
                        new_traces_dict[trace_id] = trace_data
                        accepted_count += 1
                total_count = len(traces_dict)
                print(f"{row.ShortName} - "
                      f"Accepted: {accepted_count}/{total_count} "
                      f"{100*accepted_count/total_count:.3g}%")
                set_name_to_id_to_traces[set_name] = new_traces_dict
                row[f"num_active_neurons_low_rate_{set_name}"] = accepted_count
                row[f"num_neurons_low_rate_{set_name}"] = total_count
            row = updateRowTracesSet(row, set_name_to_id_to_traces)
            res_rows.append(row)
        return pd.DataFrame(res_rows)

    def descr(self):
        return "Removed traces that were marked as inactive"


class FilterOutliersMethod:
    pass

class FilterOutliersStd(FilterOutliersMethod):
    def __init__(self, num_stds, percentile_filter=(1,99), plot=False,):
        self._num_stds = num_stds
        self._percentile_filter = percentile_filter
        self._plot = plot

    def filter(self, data, axis):
        # print("Data:", data)
        # print("Data.values:")
        # [print(k, len(val)) for k, val in data.items()]
        data_arr = np.array(list(data.values()))
        METHOD = 2
        if METHOD == 1:
            percentile = self._perctile_filter[0]
            num_removed_idxs = int(len(data) * percentile)
            if num_removed_idxs == 0:
                return list(data.keys())
            data_medians = np.nanmedian(data_arr, axis=0)
            USE_MEDIAN_TRACE = True
            if USE_MEDIAN_TRACE:
                diff_from_median_pt = np.abs(data_arr - data_medians)
                # print("diff_from_median.shape", diff_from_median_pt.shape)
                distance_from_median = np.sum(diff_from_median_pt, axis=1)
                # print("data_sum.shape", distance_from_median.shape)
                median_idx = np.argmin(distance_from_median)
                # print("median_idx", median_idx)
                median = data_arr[median_idx,:]
            else:
                median = data_medians
            median_diff = np.abs(data_arr - median)
            naughty_socres = {}
            for col in range(median_diff.shape[1]):
                # TODO: limit the out-of-bound -num_removed_idxs*2
                offenders = np.argsort(median_diff[:,col])[-num_removed_idxs*2:]
                for offender_id in offenders:
                    cur_score = naughty_socres.get(offender_id, 0)
                    naughty_socres[offender_id] = (cur_score +
                                                  median_diff[offender_id, col])
            sorted_scores = {k: v for k, v in sorted(naughty_socres.items(),
                             key=lambda item: item[1], reverse=True)}
            removed_idxs = list(sorted_scores.keys())[:num_removed_idxs]
        elif METHOD == 2:
            # Center around zero
            # TODO: Notice that this makes the plotting less comprehendable, we
            # need to plot the offsetted data, not the original one.
            traces_means = np.nanmean(data_arr, axis=1)[:,np.newaxis]
            # self._debugCentering(data_arr, traces_means)
            data_arr -= traces_means
            # data_arr = data_arr[:3,:5]
            # print("data_arr.shape", data_arr.shape)
            iqr = np.nanpercentile(data_arr, self._percentile_filter, axis=0)
            means = []
            stds = []
            stds_lims = []
            removed_idxs = set()
            for col in np.arange(data_arr.shape[1]):
                cur_col = data_arr[:,col]
                cur_iqr = iqr[:,col]
                bounds = cur_col[(cur_iqr[0] <= cur_col) &
                                 (cur_col <= cur_iqr[1])]
                cur_mean = np.nanmean(bounds)
                cur_std = np.std(bounds)
                means.append(cur_mean)
                stds.append((cur_mean - cur_std, cur_mean + cur_std))
                bot_lim = cur_mean - self._num_stds*cur_std
                top_lim = cur_mean + self._num_stds*cur_std
                stds_lims.append((bot_lim, top_lim))
                removed_idxs.update(np.where((cur_col < bot_lim) |
                                             (cur_col > top_lim))[0])
                # print("offenders", offenders)
            means = np.array(means)
            stds = np.array(stds)
            stds_lims = np.array(stds_lims)
        # print("outliers_per_point.shape", offenders_per_pt.shape)
        # print(f"Num removed: {len(removed_idxs)}/{len(data)}")
        lines = {}
        for idx, trace in enumerate(data_arr):
            if METHOD == 1 and USE_MEDIAN_TRACE and idx == median_idx:
                c = "blue" if idx in removed_idxs else "black"
                a = 1
                lw = 3
            else:
                c, a = (None,1) if idx in removed_idxs else ("green", 0.1)
                lw = 3
            # plt.plot(trace, c=c, alpha=a, lw=lw)
            lines[f"trace_{idx}"] = dict(y=trace, c=c, alpha=a, lw=lw)
            if c in ["blue", "black"]: # Repeat a second time
                lines[f"median_{idx}"] = dict(y=trace, c=c, alpha=a, lw=lw)
        if METHOD == 1:
            # plt.plot(data_medians, c="black", lw=2, ls="--")
            lines["data_medians"] = dict(y=data_medians, c="black", lw=2,
                                         ls="--")
        if METHOD == 2:
            lines["means"]   = dict(y=means, c="black", lw=2, ls="--")
            lines["std0"]    = dict(y=stds[:,0], c="blue", lw=3, ls="--",
                                    alpha=0.5)
            lines["std1"]    = dict(y=stds[:,1], c="blue", lw=3, ls="--",
                                    alpha=0.5)
            lines["stdlim1"] = dict(y=stds_lims[:,0], c="blue", lw=3, ls="-",
                                    alpha=0.5)
            lines["stdlim2"] = dict(y=stds_lims[:,1], c="blue", lw=3, ls="-",
                                    alpha=0.5)
        if self._plot:
            pass
            # for line in lines.values():
            #     plt_line = line.pop("x")
            #     plt.plot(plt_line, **line)
            # plt.show()
        # print("removed_idxs", [k for idx, k in enumerate(data.keys())
        #                        if idx in removed_idxs])
        return {k for idx, k in enumerate(data.keys())
                if idx not in removed_idxs}, lines

    def _debugCentering(self, data_arr, traces_means):
        print("Data shape:", data_arr.shape, "- Means shape:", traces_means)
        for trc_idx, trace in enumerate(data_arr):
            print("Count:", trc_idx)
            plt.plot(trace, c="k")
            plt.axhline(0, c="gray", ls="dashed")
            median = traces_means[trc_idx,0]
            plt.axhline(0, c="r", ls="dashed")
            trace = trace.copy() - median
            plt.plot(trace, c="g")
            plt.show()
            if trc_idx == 2:
                break
        print("Before:", data_arr.shape)
        plt.plot(data_arr.transpose(), c="r")
        plt.axhline(0, c="gray", ls="dashed")
        plt.show()
        data_arr = data_arr.copy() - traces_means
        print("After:")
        plt.axhline(0, c="gray", ls="dashed")
        plt.plot(data_arr.transpose(), c="g")
        plt.show()


class FilterOutlierTraces(DFProcessor):
    def __init__(self, set_name="neuronal", filter=None, epochs=[],
                 plot=True, plot_only_traces_ids=[], savePrefix=None,
                 formatRow=None):
        self._set_name = set_name
        if filter is None:
            filter = FilterOutliersStd(num_stds=3)
        self._filter = filter
        self._epochs = epochs
        self._plot = plot
        self._savePrefix = savePrefix
        self._formatRow = formatRow
        self._plot_only_traces_ids = plot_only_traces_ids

    def process(self, df):
        # print(f"Processing {df.ShortName.unique() = }")
        # Construct a set of trace ids that exist
        traces_ids = set()
        for _, row in df.iterrows():
            traces_dict = getRowTracesSets(row)[self._set_name]
            traces_ids.update(traces_dict.keys())
        # For each trace id, collect the traces from all rows that has it and run
        # the filter on to get the rows ids to remove
        row_idxs_to_traces_ids_to_remove = dict() # row_idx -> set(trace_ids)
        quick_trc_rng_lookup = dict() # row_idx -> trace_rng
        axis = TIME_AX_LOOKUP[self._set_name]
        # print("Found short names:", df.ShortName.unique())
        for trace_id in traces_ids:
            row_idx_to_sub_trace = dict()
            if self._plot:
                row_idx_to_full_trace = dict()
            for row_idx, row in df.iterrows():
                # print("Processing:", row.ShortName, row.TrialNumber,
                #       row.epoch, row.ChoiceCorrect == 1,
                #       row.trace_start_idx,row. trace_end_idx)
                traces_dict = getRowTracesSets(row)[self._set_name]
                trace_shape = None
                if trace_id in traces_dict:
                    full_trace, sub_trace = self._extractRelevantSubTrace(
                                      row_idx, row, traces_dict[trace_id], axis,
                                      quick_trc_rng_lookup)
                    if trace_shape is None:
                        trace_shape = sub_trace.shape
                    assert trace_shape == sub_trace.shape, ("All traces must "
                         f"have same shape: {trace_shape} != {sub_trace.shape}")
                    row_idx_to_sub_trace[row_idx] = sub_trace
                    if self._plot:
                        row_idx_to_full_trace[row_idx] = full_trace
            # End of row loop
            if len(row_idx_to_sub_trace):
                rows_idxs_to_keep, filter_plot_lines = self._filter.filter(
                                                row_idx_to_sub_trace, axis=axis)
                rows_idxs_to_remove = set(row_idx_to_sub_trace.keys()) - \
                                                               rows_idxs_to_keep
                # print(f"Trace id: {trace_id} removing: "
                #     f"{len(rows_idxs_to_remove)}/{len(row_idx_to_sub_trace)}")
            else:
                row_idxs_to_remove, filter_plot_lines = {}, {}
            # Mark the rows to remove for now in the dictionary
            for row_idx in rows_idxs_to_remove:
                if row_idx not in row_idxs_to_traces_ids_to_remove:
                    row_idxs_to_traces_ids_to_remove[row_idx] = set()
                row_idxs_to_traces_ids_to_remove[row_idx].add(trace_id)
            if self._plot and (not len(self._plot_only_traces_ids) or \
                               trace_id in self._plot_only_traces_ids):
                self._plotTraces(trace_id, row_idx_to_full_trace,
                                 rows_idxs_to_remove, quick_trc_rng_lookup,
                                 filter_plot_lines, # Use any row (e.g last row)
                                 ex_row=row)
        # End of trace id loop
        # Now go over the rows and remove the traces that are marked for removal
        res_rows = []
        bad_rows = []
        for row_idx, row in df.iterrows():
            if row_idx not in row_idxs_to_traces_ids_to_remove:
                continue
            new_trace_sets = getRowTracesSets(row).copy()
            cur_trace_dict = new_trace_sets[self._set_name]
            traces_ids_to_remove = row_idxs_to_traces_ids_to_remove[row_idx]
            cur_trace_dict = {k:v for k,v in cur_trace_dict.items()
                              if k not in traces_ids_to_remove}
            if len(cur_trace_dict) == 0:
                # print("cur_trace_dict.keys():", temp_keys)
                # print("Removing all traces for:", row.TrialNumber,
                #       traces_ids_to_remove)
                bad_rows.append(row_idx)
                continue
            new_trace_sets[self._set_name] = cur_trace_dict
            row = row.copy()
            row = updateRowTracesSet(row, new_trace_sets)
            res_rows.append(row)
        if not len(res_rows):
            print("No rows left after filtering - Df len:", len(df),
                  "- Bad rows len:", len(bad_rows))
            print("Using all rows. Animal name:", list(df.ShortName.unique()))
            return df
        else:
            return pd.DataFrame(res_rows)

    def _extractRelevantSubTrace(self, row_idx, row, trace_data, axis,
                                 quick_trc_rng_lookup):
        full_trc_rng, sub_trc_rng = quick_trc_rng_lookup.get(row_idx,
                                                             (None, None))
        if full_trc_rng is None:
            full_trc_rng = np.arange(row.trace_start_idx, row.trace_end_idx + 1)
            if not self._epochs:
                sub_trc_rng = full_trc_rng
            else:
                e_idxs = np.where([epoch in self._epochs
                                   for epoch in row.epochs_names])[0]
                ranges = [row.epochs_ranges[e_idx] for e_idx in e_idxs]
                sub_trc_rng = np.concatenate([np.arange(r[0], r[1]+1)
                                              for r in ranges])
            quick_trc_rng_lookup[row_idx] = full_trc_rng, sub_trc_rng
        full_trace_data = np.take(trace_data, full_trc_rng, axis=axis)
        # print("sub_trc_rng:", sub_trc_rng, "self._epochs:", self._epochs)
        sub_trace_data = np.take(trace_data, sub_trc_rng, axis=axis)
        return full_trace_data, sub_trace_data

    def _plotTraces(self, trace_id, row_idx_to_full_trace, rows_idxs_to_remove,
                    quick_trc_rng_lookup, filter_plot_lines, ex_row):
        fig, ax = plt.subplots()
        FIG_SIZE=(12,10)
        fig.set_size_inches(FIG_SIZE)
        all_lines = []
        all_labels = []
        for row_idx, full_trace in row_idx_to_full_trace.items():
            c, a = (None,1) if row_idx in rows_idxs_to_remove else \
                   ("green", 0.1)
            line2d = ax.plot(full_trace, c=c, alpha=a)[0]
            # print("line2d:", line2d.get_color())
            # Remove greens from default color cycle
            # This is a bad hack that won't work once matplotlib tweaks colors
            # values
            if line2d.get_color() == "#2ca02c":
                line2d.set_color("#8f2ca0")
            all_lines.append(line2d)
            all_labels.append(row_idx)
        # Draw the affected region with the filter calculated values
        # Use any row_idx, e.g use the last one from the loop above
        ex_trc_full, ex_trc_sub = quick_trc_rng_lookup[row_idx]
        org_x = ex_trc_sub - ex_trc_full[0]
        x = np.arange(np.amin(org_x), np.amax(org_x))
        org_x = set(org_x)
        # Make x continous for the whole period and fill the empty spaces with
        # nan
        for line_name, line in filter_plot_lines.items():
            if line_name.startswith("trace_"):
                continue
            y = line.pop("y")
            iter_y = iter(y)
            y = np.array([(next(iter_y) if idx in org_x else np.nan)
                          for idx in x])
            line2d = ax.plot(x, y, **line)
            all_lines.append(line2d)
            all_labels.append(line_name)
        # Draw vertical lines and epochs names
        epochs_start_x = [rng[0] for rng in ex_row.epochs_ranges]
        min_y = ax.get_ylim()[0]
        epochs_range = np.arange(len(ex_row.epochs_names))
        kargs = {"rotation":35, "size":"x-small", "verticalalignment":"top",
                 "horizontalalignment":"right"}
        [ax.text(epochs_start_x[idx], min_y, ex_row.epochs_names[idx], **kargs)
         for idx in epochs_range]
        [ax.axvline(x, color="k", linestyle="dashed", alpha=0.5)
         for x in epochs_start_x[1:]]
        ax.axes.xaxis.set_ticks([])
        ax.set_xlim(left=epochs_start_x[0])
        parent_dir = Path(ex_row.anlys_path)
        if self._formatRow is not None:
            row_info = self._formatRow(ex_row)
            trace_id = f"{trace_id}_{row_info}"
        ax.set_title(f"{parent_dir.name} - Trace {trace_id}")
        if self._savePrefix is not None:
            save_fp = self._savePrefix(parent_dir=parent_dir, plot_id=trace_id,
                                       plot_df=pd.DataFrame([ex_row]))
            if not isinstance(save_fp, Path):
                save_fp = Path(save_fp)
            print("Saving plot to:", save_fp)
            fig.savefig(save_fp)
        else:
            plt.show()
        plt.close()

    def descr(self):
        return "Removing outliers traces using (TODO: Add which filter)"
