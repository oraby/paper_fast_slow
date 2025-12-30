from ..pipeline.pipeline import DFProcessor, getRowTracesSets
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from pathlib import Path

class PlotTraces(DFProcessor):
    def __init__(self, is_avg_trc, x_label, y_label, only_traces_ids=[],
                 areas_to_colors={},
                 sem_traces=False, sem_traces_alpha=0.2, save_tiff=False,
                 dpi=100, fig_size=None, stop_after=-1, start_at_one=True,
                 set_name="neuronal", save_prefix=None, y_axis_right=False,
                 y_lims=None, max_x=True, draw_legend=True,
                 legendLabelAndLineStyle=None, save_context=None,
                 should_save_overwrite=True, plot_title_postfix="",
                 processPlotIdFn=None, show_plots=None, getAx=None,
                 set_aspect_ratio=None):
        if isinstance(areas_to_colors, dict):
            self._areas_to_colors = areas_to_colors
            areas_to_colors = self._defaultAreasToColors
        self._x_label = x_label
        self._y_label = y_label
        self._areasToColors = areas_to_colors
        self._only_traces_ids = only_traces_ids
        self._is_avg_trc = is_avg_trc
        self._sem_traces = sem_traces
        self._sem_traces_alpha = sem_traces_alpha
        self._stop_after = stop_after
        self._fig_size = fig_size
        self._start_at_one = start_at_one
        self._set_name = set_name
        self._save_tiff = save_tiff
        if not callable(save_prefix):
            self._save_prefix = save_prefix
            save_prefix = self._defaultSavePrefix
        self._savePrefix = save_prefix
        self._should_save_overwrite = should_save_overwrite
        self._show_plots = show_plots
        if self._show_plots is None:
            self._show_plots = False if self._save_tiff else True
        self._dpi = dpi
        self._y_lims = y_lims
        self._y_axis_right = y_axis_right
        self._draw_legend = draw_legend
        self._legendLabelAndLineStyle = legendLabelAndLineStyle
        self._set_aspect_ratio = set_aspect_ratio
        self._save_context = save_context
        self._plot_title_postfix = \
                          f" {plot_title_postfix}" if plot_title_postfix else ""
        self._processPlotIdFn = processPlotIdFn
        self._max_x = max_x
        self._getAx = getAx
        if self._getAx is not None:
            assert fig_size is None, "We won't own the figure to change it"
            assert save_tiff == False, "We won't own the figure to save it"

    def process(self, data):
        self._plotTraces(data)
        return data

    def _plotTraces(self, data):
        plot_id_to_plot_rows_to_row_traces = {}
        plot_id_to_plot_rows_to_row_traces_sem = {}
        plot_id_to_plot_rows_to_row_traces_count = {}
        plot_id_to_save_context = {}
        for row_index, row in data.iterrows():
            traces_dict = getRowTracesSets(row)[self._set_name]
            traces_dict_sem = None if not self._sem_traces else \
                             getRowTracesSets(row).get(f"{self._set_name }_sem")
            if "avg_traces_count" in data.columns:
                traces_dict_count = row["avg_traces_count"].get(self._set_name,
                                                                {})
            else:
                traces_dict_count = {}
            for trace_id, trace in traces_dict.items():
                # print("Trace id:", trace_id, " valid?:",
                #       trace_id not in self._only_traces_ids)
                if len(self._only_traces_ids) and \
                   trace_id not in self._only_traces_ids:
                    # print("Skipping:", trace_id)
                    continue
                # row.TrialNumber is just "Trials Avg." in case of trace avg
                plot_id, row_id = \
                        (row.TrialNumber, trace_id) if self._is_avg_trc else \
                        (trace_id, row.TrialNumber)
                plot_id_traces = plot_id_to_plot_rows_to_row_traces.get(plot_id,
                                                                        {})
                plot_id_traces[row_id] = trace
                plot_id_to_plot_rows_to_row_traces[plot_id] = plot_id_traces
                if self._sem_traces and traces_dict_sem is not None:
                    plot_id_traces_sem = \
                        plot_id_to_plot_rows_to_row_traces_sem.get(plot_id, {})
                    plot_id_traces_sem[row_id] = traces_dict_sem[trace_id]
                    plot_id_to_plot_rows_to_row_traces_sem[plot_id] = \
                                                              plot_id_traces_sem
                plot_id_counts = plot_id_to_plot_rows_to_row_traces_count.get(
                                                                    plot_id, {})
                plot_id_counts[row_id] = traces_dict_count.get(trace_id)
                plot_id_to_plot_rows_to_row_traces_count[plot_id] = \
                                                                  plot_id_counts
                if self._save_context is not None:
                    plot_id_to_save_context[plot_id] = row[self._save_context]

        plot_id_to_epochs_names = {}
        plot_id_to_epochs_start_x = {}
        if self._is_avg_trc:
            epochs_src = [(row.TrialNumber, row)
                          for _, row in data.iterrows()]
        else:
            epochs_src = [(key, data.iloc[0]) # Use the same row for everything
                          for key in plot_id_to_plot_rows_to_row_traces.keys()]
        for plt_id, row in epochs_src:
            plot_id_to_epochs_names[plt_id] = row.epochs_names
            plot_id_to_epochs_start_x[plt_id] = [
                                            rng[0] for rng in row.epochs_ranges]

        count = 0
        for plot_id, rows_to_traces in \
                                     plot_id_to_plot_rows_to_row_traces.items():
            if self._getAx is None:
                fig, ax = plt.subplots(1, 1, dpi=self._dpi)
                if self._fig_size is not None:
                    print("Fig size:", self._fig_size)
                    fig.set_size_inches(self._fig_size)
            else:
                ax = self._getAx(plot_id)
            if self._set_aspect_ratio is not None:
                ax.set_aspect(self._set_aspect_ratio)
            ax.set_xlabel(self._x_label)
            ax.set_ylabel(self._y_label)
            # ax.axhline(0, linestyle="--", c='gray') # Plot the line in the
            # background
            # print("Getting", plot_id, "from:",
            #       plot_id_to_plot_rows_to_row_traces_sem.keys())
            rows_sem = plot_id_to_plot_rows_to_row_traces_sem.get(plot_id, {})
            rows_counts = plot_id_to_plot_rows_to_row_traces_count.get(plot_id,
                                                                       {})
            for row_id, row in rows_to_traces.items():
                row_id = str(row_id)
                c = self._areasToColors(
                                  row_id.rsplit("_left")[0].rsplit("_right")[0])
                # Add one as we start from frame 1
                x_trace = np.arange(len(row))
                if self._start_at_one:
                    x_trace += 1
                if self._legendLabelAndLineStyle is not None:
                    label, ls = self._legendLabelAndLineStyle(row_id)
                else:
                    row_str = str(row_id).lower()
                    label, ls = row_id, (":" if ("left" in row_str or
                                                 "early" in row_str) else None)
                row_count = rows_counts.get(row_id)
                if row_count is not None:
                    label = f"{label} ({row_count} pts)"
                ax.plot(x_trace, row, ls=ls, label=label, c=c)
                sem_row = rows_sem.get(row_id)
                if sem_row is not None:
                    ax.fill_between(x_trace, row - sem_row, row + sem_row,
                                    color=c, alpha=self._sem_traces_alpha)
            Xs_pos = plot_id_to_epochs_start_x[plot_id]
            if self._start_at_one:
                # Add one as we start from frame 1
                Xs_pos = np.array(Xs_pos) + 1
            Xs_text = plot_id_to_epochs_names[plot_id]
            [ax.axvline(x, color="k", linestyle="dashed", alpha=0.5)
             for x in Xs_pos[1:]]
            y_trans = transforms.blended_transform_factory(ax.transData,
                                                           ax.transAxes)
            kargs = {"rotation":35, "size":"x-small",
                     "verticalalignment":"top",
                     "horizontalalignment":"right",
                     "transform":y_trans}
            [ax.text(Xs_pos[i], -0.02, Xs_text[i], **kargs)
             for i in range(len(Xs_pos))]
            ax.axes.xaxis.set_ticks([])
            if isinstance(self._max_x, bool) or self._max_x is None:
                if self._max_x == True or self._max_x is None:
                    # print(f"x-rng: {x_trace[0] = }, {x_trace[-1] = }")
                    ax.set_xlim(x_trace[0], x_trace[-1])
                else:
                    ax.set_xlim(left=Xs_pos[0])
            else:
                assert isinstance(self._max_x, (int, float))
                ax.set_xlim(x_trace[0], x_trace[0] + self._max_x)
            if self._y_lims is not None:
                ax.set_ylim(*self._y_lims)
            # TODO: Read the title from columns name that is passed in
            # constructor
            print_plot_id = plot_id
            cur_save_context = plot_id_to_save_context.get(plot_id)
            ex_row = data.iloc[0]
            name = ex_row.ShortName if "ShortName" in data.columns else \
                   ex_row.Name
            if self._processPlotIdFn is not None:
                trace_title = self._processPlotIdFn(print_plot_id,
                                                    cur_save_context,name,
                                                    ex_row)
            else:
                sub_title = "" if not len(print_plot_id) else \
                            f" - Trace:{print_plot_id}"
                trace_title = f"{name}{sub_title}{self._plot_title_postfix}"
            ax.set_title(trace_title, fontsize="small")
            ax.tick_params(axis='y', rotation=0)
            if self._draw_legend:
                ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -.2),
                          fontsize="x-small")
            ax.spines[['top', "right" ,"left"]].set_visible(False)
            if self._y_axis_right:
                ax.yaxis.tick_right()
            if self._save_tiff and self._savePrefix is not None: # and or or?
                save_fp = self._savePrefix(plot_id=print_plot_id, df=data,
                                           parent_dir=data.ShortName.iloc[0],
                                           save_context=cur_save_context)
                save_fp = Path(save_fp)
                print("Saving to:", save_fp)
                if self._should_save_overwrite or not save_fp.exists():
                    fig.savefig(save_fp, dpi=self._dpi, bbox_inches='tight')
            if self._show_plots:
                plt.show()
            if self._save_tiff:
                plt.close(fig)
            if count == self._stop_after:
                break
            count += 1
        if count > 0:
            return ax # Else we don't have ax defined

    def _defaultAreasToColors(self, trace_id):
        return self._areas_to_colors.get(trace_id)

    def _defaultSavePrefix(self, plot_id, parent_dir, df):
        if self._save_prefix is not None:
            save_prefix = f"{self._save_prefix}plot_{plot_id}.pdf"
            save_prefix.replace("/", "_")
        else:
            save_prefix = \
             f"{parent_dir}/states_avg/complete_trial_trc{parent_dir.name}.tiff"
        return save_prefix

    def descr(self):
        return "Plotting traces"