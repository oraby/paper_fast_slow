from .pipeline import (DFProcessor, updateRowTracesSet, createRowTracesSet,
                                             getRowTracesSets, TIME_AX_LOOKUP)
from .utils import countTransientsRatePerMin, filterNanGaussianConserving
from scipy.ndimage import minimum_filter1d, maximum_filter1d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

class BassNormalization(DFProcessor):
    def __init__(self, normFunc, set_name, restrict_to_epochs=[]):
        self._normFunc = normFunc
        if set_name is None:
            raise NotImplementedError("Didn't have a use-case for this yet")
        if isinstance(set_name, str):
             set_name = [set_name]
        self._set_name = set_name
        self._restrict_to_epochs = restrict_to_epochs

    def process(self, df):
        if self._normFunc is None: # Shortcut for NoNormalization class
            return df
        # This simplifies the implementation, rather than creating complicated
        # nested dictionaries
        for set_name in self._set_name:
            if len(self._set_name) > 1:
                print(f"Running normalization for {set_name}")
            df = self._processForSetName(df, target_set_name=set_name)
        return df

    def _processForSetName(self, df, target_set_name):
        def getTraceRange(row):
            if not len(self._restrict_to_epochs):
                trc_rng = np.arange(row.trace_start_idx, row.trace_end_idx + 1)
            else:
                assert row.trace_start_idx == 0, "Didn't handle mid-trace start"
                trc_rng = [np.arange(rng[0], rng[1] + 1)
                           for name, rng in zip(row.epochs_names,
                                                row.epochs_ranges)
                           if name in self._restrict_to_epochs]
                trc_rng = np.concatenate(trc_rng)
            return trc_rng

        def getTrace(row, trc_rng, trace_data, time_ax):
            if not row.sole_owner:
                trace_data = trace_data.take(trc_rng, axis=time_ax)
            return trace_data

        res_rows = []
        track_across_rows = len(df) > 1
        if track_across_rows:
            all_traces_dict = {}
            first_row = True

        for row_idx, row in df.iterrows():
            trial_number = row.TrialNumber
            trc_rng = getTraceRange(row)
            set_name_to_id_to_traces = {}
            for set_name, traces_dict in getRowTracesSets(row).items():
                if target_set_name != set_name:
                    set_name_to_id_to_traces[set_name] = traces_dict
                    continue
                time_ax = TIME_AX_LOOKUP[set_name]
                if track_across_rows:
                    for trace_id, trace_data in traces_dict.items():
                        trace_data = getTrace(row, trc_rng, trace_data, time_ax)
                        trace_trials_dict = all_traces_dict.get(trace_id)
                        if trace_trials_dict is None:
                            assert first_row, "Different sessions used"
                            trace_trials_dict = {}
                        trace_trials_dict[trial_number] = trace_data
                        all_traces_dict[trace_id] = trace_trials_dict
                else:
                    new_traces_dict = {}
                    for trace_id, trace_data in traces_dict.items():
                        trace_data = getTrace(row, trc_rng, trace_data, time_ax)
                        trace_data = self._normFunc({trial_number:
                                                             trace_data.copy()},
                                                    time_ax)
                        assert isinstance(trace_data, dict)
                        assert len(trace_data) == 1
                        trace_data = next(iter(trace_data.values()))
                        # trace_data_valid = trace_data[~np.isnan(trace_data)]
                        # bot_val, top_val = np.nanpercentile(trace_data_valid,
                        #                                     [self._bot,
                        #                                            self._top],
                        #                                     axis=time_ax)
                        # trace_data = NormalizePercentile.normalizeTrace(
                        #                          trace_data, bot_val, top_val)
                        new_traces_dict[trace_id] = trace_data
                    set_name_to_id_to_traces[set_name] = new_traces_dict
            if track_across_rows:
                first_row = False
                continue
            if row.sole_owner:
                row = updateRowTracesSet(row, set_name_to_id_to_traces)
            else:
                row = createRowTracesSet(row, set_name_to_id_to_traces)
            res_rows.append(row)
        if track_across_rows:
            time_ax = TIME_AX_LOOKUP[set_name]
            trials_traces_dict = {}
            for trace_id, trace_trials_dict in all_traces_dict.items():
                org_trial_numbers = trace_trials_dict.keys()
                trace_trials_dict = self._normFunc(trace_trials_dict, time_ax)
                # bot_val, top_val = np.nanpercentile(traces_arr,
                #                                     [self._bot, self._top],
                #                                     axis=time_ax)
                # trace_bot_top_dict[trace_id] = bot_val, top_val
                assert len(org_trial_numbers) == len(trace_trials_dict)
                for trial_number, trace_data in trace_trials_dict.items():
                    assert trial_number in org_trial_numbers
                    trial_dict = trials_traces_dict.get(trial_number, {})
                    trial_dict[trace_id] = trace_data
                    trials_traces_dict[trial_number] = trial_dict

            for row_idx, row in df.iterrows():
                trc_rng = getTraceRange(row)
                set_name_to_id_to_traces = {}
                for set_name, traces_dict in getRowTracesSets(row).items():
                    if target_set_name != set_name:
                        set_name_to_id_to_traces[set_name] = traces_dict
                        continue
                    # new_traces_dict = {}
                    # for trace_id, trace_data in traces_dict.items():
                    #     trace_data = trace_data.take(trc_rng, axis=time_ax)
                    #     trace_data = trace_data.copy()
                    #     bot_val, top_val = trace_bot_top_dict[trace_id]
                    #     trace_data = NormalizePercentile.normalizeTrace(
                    #                              trace_data, bot_val, top_val)
                    #     new_traces_dict[trace_id] = trace_data
                    # set_name_to_id_to_traces[set_name] = new_traces_dict
                    set_name_to_id_to_traces[set_name] = trials_traces_dict[
                                                                row.TrialNumber]
                row = row.copy()
                row = updateRowTracesSet(row, set_name_to_id_to_traces)
                res_rows.append(row)
        return pd.DataFrame(res_rows)


class NoNormalization(BassNormalization):
    def __init__(self):
        super().__init__(normFunc=None, set_name="Any", restrict_to_epochs=None)

    # def process(self, df):
    #     return df

    def descr(self):
        return (f"Applying no normalization to trace")

    def asStr(self):
        return "without"


class NormalizeZScore(BassNormalization):
    def __init__(self, set_name, restrict_to_epochs=[]):
        super().__init__(self._normFunc, set_name, restrict_to_epochs)

    def _normFunc(self, trace_trials_dict, time_ax):
        concat_array = np.hstack(list(trace_trials_dict.values()))
        zscored_concat_array = stats.zscore(concat_array, axis=time_ax)
        res_dict = {}
        start_idx = 0
        for _id, org_trace in trace_trials_dict.items():
            end_idx = start_idx + len(org_trace)
            zscored_trace = zscored_concat_array[start_idx:end_idx]
            assert zscored_trace.shape == org_trace.shape
            res_dict[_id] = zscored_trace
            start_idx = end_idx
        return res_dict

    def descr(self):
        return (f"Normalized z-score of trace")

    def asStr(self):
        return "ZScore"


class NormalizePercentile(BassNormalization):
    def __init__(self, bot, top, set_name=None, restrict_to_epochs=[]):
        self._bot = bot
        self._top = top
        super().__init__(self._normFunc, set_name, restrict_to_epochs)

    def _normFunc(self, trace_trials_dict, time_ax):
        concat_array = np.hstack(list(trace_trials_dict.values()))
        bot_val, top_val = np.nanpercentile(concat_array,
                                            [self._bot, self._top],
                                            axis=time_ax)
        concat_array = NormalizePercentile.normalizeTrace(concat_array, bot_val,
                                                          top_val)
        res_dict = {}
        start_idx = 0
        for _id, org_trace in trace_trials_dict.items():
            end_idx = start_idx + len(org_trace)
            normalized_trace = concat_array[start_idx:end_idx]
            assert normalized_trace.shape == org_trace.shape
            res_dict[_id] = normalized_trace
            start_idx = end_idx
        return res_dict

    @staticmethod
    def normalizeTrace(trace_data, bot_val, top_val):
        trace_data[trace_data <= bot_val] = bot_val
        trace_data[top_val <= trace_data] = top_val
        trace_data = (trace_data - bot_val)/(top_val - bot_val)
        # Avoid floating point values > 1
        trace_data = np.around(trace_data, decimals=4)
        return trace_data

    def descr(self):
        return (f"Normalized trace between its {self._bot}% and {self._top}% "
                 "percentile values")
    def asStr(self):
        return "Percentile"


def normalizeMinMaxPerTrace(df, match_rows_col_name, set_name,
                            norm_min_max_val_rng=None, smooth_sigma=None):
    df = df.copy()
    df["OriginalOrder"] = np.arange(len(df))
    df_groups_li = []
    for group_name, df_group in tqdm(df.groupby(match_rows_col_name)):
        df_group = _normalizeGroupMinMax(df_group, set_name,
                                         norm_min_max_val_rng,
                                         smooth_sigma=smooth_sigma)
        df_groups_li.append(df_group)
    df = pd.concat(df_groups_li)
    df = df.sort_values(by="OriginalOrder")
    df = df.drop(columns=["OriginalOrder"])
    return df


def _normalizeGroupMinMax(df, set_name, norm_min_max_val_rng, smooth_sigma=None):
    if norm_min_max_val_rng is not None:
        min_rng, max_rng = norm_min_max_val_rng
    trace_id_min_max_dict = {}
    first_row = True
    time_ax = TIME_AX_LOOKUP[set_name]
    for _, row in df.iterrows():
        for _set_name, traces_dict in getRowTracesSets(row).items():
            if _set_name != set_name:
                continue
            for trace_id, trace_data in traces_dict.items():
                if trace_id not in trace_id_min_max_dict:
                    assert first_row, (
                         "Matching rows is broken, new trace id appeared later")
                    trace_id_min_max_dict[trace_id] = np.inf, -np.inf
                cur_min, cur_max = trace_id_min_max_dict[trace_id]
                trace_data = trace_data.astype(np.float64)
                cur_min = min(cur_min, trace_data.min())
                cur_max = max(cur_max, trace_data.max())
                trace_id_min_max_dict[trace_id] = cur_min, cur_max
    res_rows = []
    for _, row in df.iterrows():
        new_traces_set = {}
        for _set_name, traces_dict in getRowTracesSets(row).items():
            if _set_name != set_name:
                new_traces_set[_set_name] = traces_dict.copy()
                continue
            new_traces_dict = {}
            time_ax = TIME_AX_LOOKUP[_set_name]
            for trace_id, trace_data in traces_dict.items():
                min_val, max_val = trace_id_min_max_dict[trace_id]
                trace_data = trace_data.copy()
                #
                trace_data[trace_data <= min_val] = min_val
                trace_data[max_val <= trace_data] = max_val
                trace_data = (trace_data - min_val)/(max_val - min_val)
                # # Avoid floating point values > 1
                trace_data = np.around(trace_data, decimals=4)
                # # trace_data = trace_data*(max_rng - min_rng) + min_rng
                if norm_min_max_val_rng is not None:
                    trace_data = trace_data*(max_rng - min_rng) + min_rng
                if smooth_sigma is not None:
                    trace_data = filterNanGaussianConserving(trace_data,
                                                             sigma=smooth_sigma,
                                                             axis=time_ax)
                new_traces_dict[trace_id] = trace_data
            new_traces_set[_set_name] = new_traces_dict
        row = row.copy()
        row = updateRowTracesSet(row, new_traces_set)
        res_rows.append(row)
    return pd.DataFrame(res_rows)


class CalcBaseline(DFProcessor):
    def __init__(self, track_is_active : bool, min_pos_rate_per_min=None,
                             valid_transients_min_dur_sec=None):
        if track_is_active:
            assert min_pos_rate_per_min is not None
            assert valid_transients_min_dur_sec is not None
        if min_pos_rate_per_min is not None or \
            valid_transients_min_dur_sec is not None:
            assert track_is_active == True
        self._track_is_active = track_is_active
        self._min_pos_rate_per_min = min_pos_rate_per_min
        self._valid_transients_min_dur_sec = valid_transients_min_dur_sec

    def process(self, df):
        df_f_set_name = "neuronal"
        raw_set_name = "neuronal_raw"
        res_rows = []
        for row_id, row in df.iterrows():
            set_name_to_id_to_traces = {}
            row = row.copy()
            for set_name, traces_dict in getRowTracesSets(row).items():
                if df_f_set_name != set_name:
                    set_name_to_id_to_traces[set_name] = traces_dict
                    continue
                self._count = 0
                sess_info = (row.Name, row.Date, row.SessionNum)
                print(f"Processing {row.ShortName} - {sess_info}")
                fps = row.acq_sampling_rate
                traces_id_to_traces_data_df_f = {}
                traces_id_to_traces_data_raw = {}
                traces_id_to_traces_stats = {}
                for trace_id, trace_data in traces_dict.items():
                    trace_data_df_f, stats = self._calcBaseLine(trace_data, fps,
                                                                trace_id)
                    traces_id_to_traces_data_df_f[trace_id] = trace_data_df_f
                    # trace_data is unchanged, same as its original value
                    traces_id_to_traces_data_raw[trace_id] = trace_data
                    traces_id_to_traces_stats[trace_id] = stats
                set_name_to_id_to_traces[set_name] = \
                                                   traces_id_to_traces_data_df_f
                set_name_to_id_to_traces[raw_set_name] = \
                                                    traces_id_to_traces_data_raw
                num_neurons = len(traces_dict)
                if self._track_is_active:
                    assert len(traces_id_to_traces_stats) == num_neurons, (
                     f"{len(traces_id_to_traces_stats) = } != {num_neurons = }")
                    row["traces_stats"] = traces_id_to_traces_stats
                    # Print some stats
                    num_active = len([s
                                      for s in traces_id_to_traces_stats.values()
                                      if s["is_active"]])
                    print(f"{row.anlys_path} - {sess_info}:")
                    print(f"Num active neurons: {num_active}/{num_neurons} "
                          f"{(100*num_active/num_neurons):.2g}%")
                    if set_name == "neuronal":
                        row[f"num_active_neurons_df_f"] = num_active
                if set_name == "neuronal":
                    row[f"num_neurons"] = num_neurons
            row = updateRowTracesSet(row, set_name_to_id_to_traces)
            res_rows.append(row)
        return pd.DataFrame(res_rows)

    def _calcBaseLine(self, trace, frame_rate, trace_id):
        # Copied from suite2p.extraction.dcnv.preprocess()
        win_baseline = 60
        sig_baseline = 10
        fs = frame_rate
        win = int(win_baseline*fs)
        # Shouldn't matter which gaussian_filter1d, we shouldn't have nan values
        # Flow = gaussian_filter1d(trace, sigma=sig_baseline)
        # TODO: Do we need an extra filtering here if the neuron has already
        # been smoothed at origin?
        Flow = filterNanGaussianConserving(trace, sigma=sig_baseline, axis=-1)
        Flow = minimum_filter1d(Flow, win)
        Flow = maximum_filter1d(Flow,  win)
        ## End of copying
        trace_maximin = Flow

        # Find the mode value of the max_min continous (non-discrete) values by
        # using a histogram. This is the baseline value.
        num_bins = 200
        count_baseline, bins_baseline = np.histogram(trace_maximin,
                                                     bins=num_bins)
        bin_df_idx = np.argmax(count_baseline)
        # Calculate the mode mid-pint value
        baseline = \
                 (bins_baseline[bin_df_idx] + bins_baseline[bin_df_idx + 1]) / 2
        # Calculate the df/f
        trace_df_f = (trace - baseline)/baseline

        if self._track_is_active:
            # Calculate std. deviation per
            # https://doi.org/10.1016/j.neuron.2017.06.013
            less_than_mode = trace[trace < baseline]
            x = less_than_mode
            n = len(less_than_mode)
            f = baseline
            z_score = np.sqrt((2 / n) * np.sum((x - f) ** 2))
            # I know I can just do: transients_rate, * = ..., but I'm leaving
            # them as they are needed in case we re-enable the plotting function
            # below
            (transients_rate, pos_thres, accepted_pos_transients,
             pos_transients, mv_avg_trace, mv_avg_win, MIN_FRAMES) = \
                  countTransientsRatePerMin(trace, baseline, z_score,
                                            frame_rate,
                                            self._valid_transients_min_dur_sec)
            is_active_neuron = transients_rate >= self._min_pos_rate_per_min
            # if 50 < self._count <= 55:
            #     self._plotTraces()
            self._count += 1
            # mean_baseline = baseline_frames.mean()
            stats = {"is_active":is_active_neuron,
                     "baseline":baseline,
                     "zscore":z_score}
        else:
            stats = {}
        return trace_df_f, stats

    def _plotTraces(self, trace_id, trace, mv_avg_trace, baseline, z_score,
                    num_bins, count, bins, pos_transients,
                    accepted_pos_transients, frame_rate):
        raise NotImplementedError("method is out of date but almost there")
        ax = plt.axes()
        hist_count, hist_bins, _ = ax.hist(trace, color='b', histtype='step',
                                           bins=num_bins)
        assert (hist_count == count).all()
        assert (hist_bins == bins).all()
        # hist_count, hist_bins, _ = ax.hist(trace_maximin, color='g',
        #                                    histtype='step', bins=num_bins)
        # assert (hist_count == count_baseline).all()
        # assert (hist_bins == bins_baseline).all()
        bin_idx = np.argmax(count)
        mode = (bins[bin_idx] + bins[bin_idx+1])/2
        for i, c in (1, 'r'), (2, 'purple'), (3, 'yellow'):
            ax.axvline(baseline - z_score*i, color=c,
                       label=f"{i}-Z from Baseline")
            ax.axvline(baseline + z_score*i, color=c)
        ax.axvline(mode, color='k', label="Mode")
        ax.axvline(baseline, color='green', label="Baseline")
        ax.set_ylim(top=count.max())
        ax.set_title(f"Neuron: {trace_id}")
        ax.legend()
        plt.show()

        ax = plt.axes()
        # trace_test = trace
        trace_test = trace
        trace_test2 = mv_avg_trace
        sigma = 3
        trace_test3 = filterNanGaussianConserving(trace, sigma=sigma, axis=-1)
        num_frames = int(np.round(frame_rate * 60))
        num_frames = min(num_frames, len(trace_test))
        offset = 0
        trace_ex = trace_test[offset:offset+num_frames]
        trace_ex2 = trace_test2[offset:offset+num_frames]
        trace_ex3 = trace_test3[offset:offset+num_frames]
        # trace_df_f = (trace - mode_df)/mode_df
        # trace_df_f_ex = trace_df_f[:num_frames]
        ax.plot(np.arange(len(trace_ex)), trace_ex, color='b', alpha=0.3,
                label="Original")
        # ax.plot(np.arange(len(trace_ex2)), trace_ex2, color='k',
        #         label=f"Moving Average (w={mv_avg_win})")
        ax.plot(np.arange(len(trace_ex3)), trace_ex3, color='y',
                label=f"Gaussian Filter (s={sigma})")
        label_once_accepted, label_once_rejected = False, False
        for transient_idxs in pos_transients:
            accepted = transient_idxs in accepted_pos_transients
            color = 'g' if accepted else 'r'
            s, e = transient_idxs[0], transient_idxs[-1]
            s, e = s - offset, e - offset
            if s and e < 0:
                continue
            if s < 0: s = 0
            if e >= len(trace_ex): e = len(trace_ex) - 1
            # print("s:", s, "e:", e)
            sub_trace_rng = np.arange(s, e)
            sub_trace = trace_ex2[sub_trace_rng]
            if accepted and not label_once_accepted:
                label = "Accepted Transients"
                label_once_accepted = True
            elif not accepted and not label_once_rejected:
                label = "Rejected Transients"
                label_once_rejected = True
            else:
                label = None
            ax.plot(sub_trace_rng, sub_trace, color=color, label=label)
            if e == len(trace_ex):
                break
        # ax.plot(np.arange(len(trace_df_ex)), trace_df_ex, color='yellow')
        ax.axhline(mode, color='k')
        ax.axhline(baseline, color='green')
        for i, c in (1, 'r'), (2, 'purple'), (3, 'yellow'):
            ax.axhline(baseline - z_score*i, color=c)
            ax.axhline(baseline + z_score*i, color=c)
        ax.axhline(pos_thres, color="orange",
                   label="Positive Transient Threshold")
        # twix = ax.twinx()
        # twix.plot(np.arange(len(trace_df_f_ex)), trace_df_f_ex, color='r',
        #           alpha=0.3)
        ax.legend(fontsize="x-small")
        ax.set_title(f"Neuron: {trace_id} - "
                     f"Active ({transients_rate:.2g}s > "
                     f"{MIN_NUM_TRANSIENTS_PER_MIN}s)?: {is_active_neuron} - "
                     f"Pos. Transients: "
                     f"{len(accepted_pos_transients)}/{len(pos_transients)} - "
                     f"(Min #: {MIN_NUM_TRANSIENTS}) - "
                     f"Transient Min #pts: {MIN_FRAMES}",
                     fontsize="small")
        plt.show()

    def descr(self):
        return "Calculating baseline"
