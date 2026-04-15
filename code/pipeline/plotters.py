from .pipeline import DFProcessor, getRowTracesSets, TIME_AX_LOOKUP
from .utils import numTraces
from ..common.definitions import BrainRegion
from ..common.clr import colorMapParula, BrainRegion as BRClr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from collections import namedtuple
from pathlib import Path

class TracesHeatMap(DFProcessor):
    def __init__(self, is_avg_trc, set_name="neuronal", stop_after=-1,
                 restrict_to_epochs=[], heatmap_min=None, heatmap_max=None,
                 y_axis_label=None, plot_y_ids=True, title_cols=[],
                 title_cols_names=True, title_prefix=None,
                 write_title_plot_id=True, num_traces_in_title=None,
                 save_prefix=None, save_token=None, dpi=100,
                 only_traces_ids=[], color_map=colorMapParula(),
                 secondMapFn=None):
        self._is_avg_trc = is_avg_trc
        self._stop_after = stop_after
        self._set_name = set_name
        self._heatmap_min = heatmap_min
        self._heatmap_max = heatmap_max
        self._restrict_to_epochs = restrict_to_epochs
        self._title_cols = title_cols
        if title_prefix is None:
            self._title_prefix = f"{'Trial' if self._is_avg_trc else 'Trace'}: "
        else:
            self._title_prefix = title_prefix
        self._title_cols_names = title_cols_names
        self._write_title_plot_id = write_title_plot_id
        if num_traces_in_title is None:
            num_traces_in_title = is_avg_trc
        self._num_traces_in_title = num_traces_in_title
        assert num_traces_in_title or not is_avg_trc, ("Can only add number of "
                                                "traces if is_avg_trace is True")
        if save_prefix is not None and not callable(save_prefix):
            self._save_prefix = save_prefix
            save_prefix = self._defaultSavePrefix
        self._savePrefix = save_prefix
        self._save_token = save_token
        self._dpi = dpi
        self._y_axis_label = y_axis_label
        self._plot_y_ids = plot_y_ids
        assert isinstance(only_traces_ids, list)
        self._only_traces_ids = set(only_traces_ids)
        self._color_map = color_map
        self._secondMapFn = secondMapFn

    def process(self, data):
        plot_id_to_plot_rows_to_row_traces = {}
        plot_id_to_plot_titles = {}
        plot_id_to_num_traces = {}
        if "TracesMinMax" in data.columns and (self._heatmap_min is None or
                                               self._heatmap_max is None):
            plot_id_to_min_max_val = {}
            track_min_max = True
        else:
            track_min_max = False
        for row_index, row in data.iterrows():
            num_traces = 0
            if len(self._restrict_to_epochs):
                trc_rng = [np.arange(rng[0], rng[1] + 1)
                           for name, rng in zip(row.epochs_names,
                                                row.epochs_ranges)
                           if name in self._restrict_to_epochs]
                trc_rng = np.concatenate(trc_rng)
            else:
                trc_rng = np.arange(row.trace_start_idx, row.trace_end_idx + 1)
            traces = getRowTracesSets(row)[self._set_name]
            if track_min_max:
                traces_min_max = row["TracesMinMax"][self._set_name]
            for trace_id, trace in traces.items():
                if len(self._only_traces_ids) and \
                   trace_id not in self._only_traces_ids:
                     continue
                # row.TrialNumber is just "Trials Avg." in case of trace avg
                plot_id, row_id = \
                          (row.TrialNumber, trace_id) if self._is_avg_trc else \
                          (trace_id, row.TrialNumber)
                trace = trace.take(trc_rng, axis=TIME_AX_LOOKUP[self._set_name])
                # if num_traces < 3:
                #     print("trace_id:", trace_id, "trace valid:",
                #           (~np.isnan(trace)).sum())
                plot_id_traces = plot_id_to_plot_rows_to_row_traces.get(plot_id,
                                                                        {})
                plot_id_traces[row_id] = trace
                plot_id_to_plot_rows_to_row_traces[plot_id] = plot_id_traces
                if track_min_max and not self._is_avg_trc:# Each plot is a trace
                    plot_id_to_min_max_val[plot_id] = traces_min_max[plot_id]
                # TODO: If self._is_avg_trc is True, should we take the min-max
                # value across all the traces?
                if self._title_cols:
                    title = []
                    for col in self._title_cols:
                        sub_col_title = \
                                    f"{col}: " if self._title_cols_names else ""
                        title.append(f"{sub_col_title}{row[col]}")
                    title = " ".join(title)
                    # This might get overwritten multiple times if not
                    # self._is_avg_trace
                    plot_id_to_plot_titles[plot_id] = title
                num_traces += 1
            if self._num_traces_in_title and num_traces:
                assert plot_id == row.TrialNumber
                plot_id_to_num_traces[plot_id] = num_traces

        plot_id_to_epochs_names = {}
        plot_id_to_epochs_start_x = {}
        if self._is_avg_trc:
            epochs_src = [(row.TrialNumber, row)
                          for _, row in data.iterrows()]
        else:
            epochs_src = [(key, data.iloc[0]) # Use the same row for everything
                          for key in plot_id_to_plot_rows_to_row_traces.keys()]
        for plt_id, row in epochs_src:
            plot_id_to_epochs_names[plt_id] = [name for name in row.epochs_names
                                        if not len(self._restrict_to_epochs) or
                                           name in self._restrict_to_epochs]
            plot_id_to_epochs_start_x[plt_id] = [rng[0] for name, rng in zip(
                                            row.epochs_names, row.epochs_ranges)
                                            if not len(self._restrict_to_epochs)
                                            or name in self._restrict_to_epochs]
        count = 0
        for plot_id, rows_to_traces in \
                                    plot_id_to_plot_rows_to_row_traces.items():
            # fig, ax = plt.subplots(figsize=(15, 7.5))
            # fig, ax = plt.subplots(figsize=(9, 13.5))
            fig, ax = plt.subplots(figsize=(7.5, 11))
            title_plot_id = plot_id if self._write_title_plot_id else ""
            title = f"{self._title_prefix}{title_plot_id}"
            if len(title):
                    title = f"{title} - "
            if self._title_cols:
                title = f"{title}{plot_id_to_plot_titles[plot_id]}"
            if self._num_traces_in_title:
                title = f"{title} - #Traces: {plot_id_to_num_traces[plot_id]}"
            ax.set_title(title)
            heatmap_df = pd.DataFrame(list(rows_to_traces.values()),
                                      index=list(rows_to_traces.keys()))
            if self._secondMapFn is not None:
                 heatmap_df1, heatmap_df2, cmap1, cmap2 = \
                                   self._secondMapFn(heatmap_df, plot_id, title)
            else:
                 heatmap_df1, heatmap_df2, cmap1, cmap2 = \
                                         heatmap_df, None, self._color_map, None
            del heatmap_df
            vmin = self._heatmap_min
            if vmin is None and not self._is_avg_trc and track_min_max:
                vmin = plot_id_to_min_max_val.get(plot_id, [None])[0]
            vmax = self._heatmap_max
            if vmax is None and not self._is_avg_trc and track_min_max:
                vmax = plot_id_to_min_max_val.get(plot_id, [None, None])[1]
            # print("self.heatmap_df:", list(rows_to_traces.keys()))
            ax = sns.heatmap(heatmap_df1, cmap=cmap1, vmin=vmin, vmax=vmax,
                             ax=ax, yticklabels=self._plot_y_ids,
                             cbar_kws=dict(aspect=40))
            if heatmap_df2 is not None:
                sns.heatmap(heatmap_df2, cmap=cmap2, vmin=vmin, vmax=vmax,
                            ax=ax, #cbar=cbar)
                            yticklabels=self._plot_y_ids,)
            Xs_pos = plot_id_to_epochs_start_x[plot_id]
            Xs_text = plot_id_to_epochs_names[plot_id]
            [ax.axvline(x, color="k", linestyle="dashed", alpha=0.5)
             for x in Xs_pos[1:]]
            kargs = {"rotation":35, #"size":"x-small",
                     "verticalalignment":"top",
                     "horizontalalignment":"right"}
            min_y = ax.get_ylim()[0]
            [ax.text(Xs_pos[i], min_y, Xs_text[i], **kargs)
             for i in range(len(Xs_pos))]
            ax.axes.xaxis.set_ticks([])
            ax.tick_params(axis='y', rotation=0)
            ax.tick_params(axis='y', which='major', labelsize=3)
            ax.tick_params(axis='y', which='minor', labelsize=3)
            if self._y_axis_label is not None:
                ax.set_ylabel(self._y_axis_label)
            # axesPlot.text(x_pos, axesPlot.get_ylim()[0], state, **kargs)
            if self._savePrefix:
                save_prefix = self._savePrefix(plot_id=plot_id, df=data,
                                               parent_dir=data.ShortName.iloc[0],
                                               save_context=self._save_token,)
                plt.savefig(save_prefix, dpi=self._dpi, bbox_inches='tight',
                            facecolor='white', transparent=False)
            else:
                plt.show()
            plt.close()
            if count == self._stop_after:
                break
            count += 1
        return data

    def _defaultSavePrefix(self, plot_id, parent_dir, df, save_token):
        plot_id = plot_id.replace('/', '_')
        save_prefix = (f"{self._save_prefix}/{parent_dir.name}/states_avg/"
                       f"heatmap_{plot_id}.jpeg")
        return save_prefix

    def descr(self):
        return "Plotting traces data as heatmap"


class AvgTracesActivityOVerTime(DFProcessor):
    def __init__(self, set_name="neuronal", save_prefix=None, save_token=None,
                 dpi=100):
        self._set_name = set_name
        if save_prefix is not None and not callable(save_prefix):
            self._save_prefix = save_prefix
            save_prefix = self._defaultSavePrefix
        self._savePrefix = save_prefix
        self._save_token = save_token
        self._dpi = dpi

    def process(self, data):
        if "TracesMinMax" in data.columns and (self._heatmap_min is None or
                                               self._heatmap_max is None):
            plot_id_to_min_max_val = {}
            track_min_max = True
        else:
            track_min_max = False
        axis = TIME_AX_LOOKUP[self._set_name]
        fig, ax = plt.subplots()
        used_rows_names = []
        for row_index, row in data.iterrows():
            traces = getRowTracesSets(row)[self._set_name]
            traces_activity = []
            for trace_id, trace in traces.items():
                # row.TrialNumber is just "Trials Avg." in case of trace avg
                trace = trace.take(np.arange(row.trace_start_idx,
                                             row.trace_end_idx + 1),
                                   axis=axis)
                traces_activity.append(trace)
            traces_activity = np.array(traces_activity)
            assert traces_activity.min() >= 0, \
                                          "Activity not normalized between 0->1"
            assert traces_activity.max() <= 1, \
                                          "Activity not normalized between 0->1"
            traces_activity = traces_activity.sum(axis=0)
            br = BrainRegion(row.BrainRegion)
            clr = BRClr[br]
            traces_activity = 100 * traces_activity / traces_activity.max()
            ax.plot(np.arange(len(traces_activity)), traces_activity,
                    color=clr, label=f"{br} {row.Layer}")
            used_rows_names.append(row.ShortName)
        ax.axes.xaxis.set_ticks([])
        for epoch_name, epoch_range in zip(row.epochs_names, row.epochs_ranges):
            ax.axvline(epoch_range[0], linestyle="--", c="gray", alpha=0.5)
            kargs = {"rotation": 35,  # "size":"x-small",
                     "verticalalignment": "top",
                     "horizontalalignment": "right",}
            ax.text(epoch_range[0], ax.get_ylim()[0], epoch_name, **kargs)
        ax.set_xlim(left=0)
        ax.set_ylabel("Normalized Population Firing")
        ax.set_ylim(top=105)
        ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
        if len(used_rows_names) == 1:
            num_traces = numTraces(row)[self._set_name]
            title = \
              f"{used_rows_names[0]} - {br} {row.Layer} - #Traces: {num_traces}"
            ax.set_title(title)
        else:
            title = "Avg Population Activity"
        if self._savePrefix:
            save_prefix = self._savePrefix(title, Path(data.anlys_path.iloc[0]),
                                           data, save_token=self._save_token)
            plt.savefig(save_prefix, dpi=self._dpi, bbox_inches="tight",
                        facecolor="white", transparent=False)
        plt.show()
        return data

    def _defaultSavePrefix(self, plot_id, parent_dir, df, save_token):
        plot_id = plot_id.replace("/", "_")
        save_prefix = (f"{self._save_prefix}/{parent_dir.name}/states_avg/"
                       f"heatmap_{plot_id}.jpeg")
        return save_prefix

    def descr(self):
        return "Plotting activity overall traces activity over time as histogram"


class TrialHistoryActivity(DFProcessor):
    def __init__(self, stop_after=-1, save_prefix=None, set_name="neuronal"):
        self._stop_after = stop_after
        self._save_prefix = save_prefix
        self._set_name = set_name

    def process(self, df):
        Trace = namedtuple("Trace",
                           ["ID", "CurChoice", "PrevChoice", "PrevCount",
                            "Mean", "SEM", "Max", "Count"])
        res_traces = []
        max_prev_count = df.PrevOutcomeCount.abs().max()
        corr_df = df[df.ChoiceCorrect == 1]
        incorr_df = df[df.ChoiceCorrect == 0]
        for cur_choice_df, cur_str in [(corr_df, "Correct"),
                                       (incorr_df, "Incorrect")]:
            for i in range(1, max_prev_count + 1):
                prev_corr = cur_choice_df[cur_choice_df.PrevOutcomeCount == i]
                prev_incorr = cur_choice_df[
                                          cur_choice_df.PrevOutcomeCount == -i]
                for prev_df, prev_str in [(prev_corr, "Correct"),
                                          (prev_incorr, "Incorrect")]:
                    if not len(prev_df):
                        continue
                    prev_mean, prev_sem, prev_max, prev_count = \
                                                     self._tracesAvgStd(prev_df)
                    for trace_id, trace_mean in prev_mean.items():
                        trace_sem = prev_sem[trace_id]
                        trace_max = prev_max[trace_id]
                        trace_count = prev_count[trace_id]
                        trc = Trace(ID=trace_id, CurChoice=cur_str,
                                    PrevChoice=prev_str, PrevCount=i,
                                    Mean=trace_mean, SEM=trace_sem,
                                    Max=trace_max, Count=trace_count)
                        res_traces.append(trc)
        plot_df = pd.DataFrame(res_traces)
        self._plotTraces(plot_df)
        return df

    def _plotTraces(self, df):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        count = 0
        for trace_id, trace_df in df.groupby(df.ID):
            fig, axs = plt.subplots(1, 2)
            fig_size = fig.get_size_inches()
            fig.set_size_inches(fig_size[0] * 2, fig_size[1])
            y_lim_min, y_lim_max = \
                            trace_df.Mean.min() - 0.1, trace_df.Mean.max() + 0.2
            for cur_str, _ax in [("Correct", axs[1]), ("Incorrect", axs[0])]:
                cur_choice_df = trace_df[trace_df.CurChoice == cur_str]
                if not len(cur_choice_df):
                    continue
                y_maxs_min = cur_choice_df.Max.min()
                y_maxs_rng = cur_choice_df.Max.max() - y_maxs_min
                y_mean_rng = cur_choice_df.Mean.max() - cur_choice_df.Mean.min()
                y_maxs_offset = \
                    -cur_choice_df[cur_choice_df.Mean ==
                                   cur_choice_df.Mean.min()].iloc[0].Mean
                for prev_str, prev_clr in [("Correct", "g"),
                                           ("Incorrect", "r")]:
                    prev_df = cur_choice_df[
                                           cur_choice_df.PrevChoice == prev_str]
                    Xs = prev_df.PrevCount.to_numpy()
                    Ys_mean = prev_df.Mean.to_numpy()
                    Ys_sem = prev_df.SEM.to_numpy()
                    _ax.errorbar(Xs, Ys_mean, Ys_sem, c=prev_clr,
                                 label=f"Prev {prev_str} Mean")
                    num_trials = prev_df.Count.to_numpy()
                    for i, trial_count in enumerate(num_trials):
                        _ax.annotate(f"{trial_count}T",
                                     (Xs[i], Ys_mean[i] - 0.05))
                    Ys = prev_df.Max.to_numpy()
                    print("y_maxs_offset:", y_maxs_offset)
                    denom = y_maxs_rng if y_maxs_rng else 1
                    Ys = \
                      (((Ys - y_maxs_min) / denom) * y_mean_rng) - y_maxs_offset
                    # _ax.plot(Xs, Ys, c=prev_clr, linestyle="--",
                    #       label=f"Scaled Mean of Prev {prev_str} Trials Max.")
                _ax.set_title(f"Trace {trace_id} - Current Choice: {cur_str}")
                _ax.set_xlabel("Num. Previous Trials")
                _ax.set_ylabel("Activity")
                _ax.legend(loc="upper right", fontsize="x-small")
                _ax.set_xlim(_ax.get_xlim()[1], _ax.get_xlim()[0])
                _ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                _ax.set_ylim(y_lim_min, y_lim_max)
            if self._save_prefix:
                save_path = Path(f"{self._save_prefix}/Trace_{trace_id}.png")
                print("Saving to:", save_path)
                if not save_path.parent.exists():
                    save_path.parent.mkdir()
                fig.savefig(save_path, bbox_inches="tight", facecolor="white",
                            transparent=False)
                plt.close()
            else:
                plt.show()
            count += 1
            if count == self._stop_after:
                break

    def _tracesAvgStd(self, df):
        trace_id_insts = {}
        for row_idx, row in df.iterrows():
            row_traces = getRowTracesSets(row)[self._set_name]
            for trace_id, trace in row_traces.items():
                insts_li = trace_id_insts.get(trace_id, [])
                insts_li.append(trace)
                trace_id_insts[trace_id] = insts_li
        trace_id_mean, trace_id_sem, trace_id_max, trace_id_count = \
                                                                 {}, {}, {}, {}
        for trace_id, trace_insts_li in trace_id_insts.items():
            # TODO: Change next line to 1 for a trace rather than a point
            axis = None
            trace_id_mean[trace_id] = np.mean(trace_insts_li, axis=axis)
            trace_id_sem[trace_id] = stats.sem(trace_insts_li, axis=axis)
            trace_id_max[trace_id] = np.max(trace_insts_li, axis=1).mean()
            trace_id_count[trace_id] = len(trace_insts_li)
        return trace_id_mean, trace_id_sem, trace_id_max, trace_id_count

    def descr(self):
        return ("Plot each trace activity based on current and previous trials "
                "outcome.")

class TracesDistribution(DFProcessor):
    def __init__(self, set_name="neuronal", title_prefix=None, save_prefix=None,
                 dpi=100):
        self._set_name = set_name
        if title_prefix is None:
            self._title_prefix = "Session: "
        else:
            self._title_prefix = title_prefix
        if save_prefix is not None and not callable(save_prefix):
            self._save_prefix = save_prefix
            save_prefix = self._defaultSavePrefix
        self._savePrefix = save_prefix
        self._dpi = dpi

    def process(self, data):
        for row_index, row in data.iterrows():
            traces = getRowTracesSets(row)[self._set_name]
            # heatmap_df = pd.DataFrame(list(traces.values()),
            #                           index=list(traces.keys()))
            minimums, maximums, stds = \
                                   zip(*[(trace.min(), trace.max(), trace.std())
                                         for trace in traces.values()])
            minimums, maximums = np.array(minimums), np.array(maximums)
            stds = np.array(stds)
            traces_ranges = maximums - minimums
            ax = plt.axes()
            ax.violinplot([traces_ranges, minimums, maximums, stds])
            # ax.axes.xaxis.set_ticks([])
            # ax.tick_params(axis='y', rotation=0)
            # ax.tick_params(axis='y', which='major', labelsize=3)
            # ax.tick_params(axis='y', which='minor', labelsize=3)
            # axesPlot.text(x_pos, axesPlot.get_ylim()[0], state, **kargs)
            title = (f"{self._title_prefix}{row.Name} - {row.Date} - "
                     f"Sess: {row.SessionNum}")
            ax.set_title(title)
            plt.gcf().set_size_inches(15, 7.5)
            if self._savePrefix:
                title = title.replace("/", "_")
                save_prefix = self._savePrefix(title,
                                               Path(data.anlys_path.iloc[0]),
                                               data)
                plt.savefig(save_prefix, dpi=self._dpi, bbox_inches="tight",
                            facecolor="white", transparent=False)
            else:
                plt.show()
            plt.close()
            print("Map done")
        return data

    def _defaultSavePrefix(self, plot_id, parent_dir, df):
        plot_id = plot_id.replace("/", "_")
        save_prefix = (f"{self._save_prefix}/{parent_dir.name}/states_avg/"
                       f"heatmap_{plot_id}.jpeg")
        return save_prefix

    def descr(self):
        return "Plotting session data in heatmap form"
