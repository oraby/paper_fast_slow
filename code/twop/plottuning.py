from ..common.definitions import BrainRegion
from ..common.clr import BrainRegion as BRCLr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from scipy import stats
from enum import Enum, auto
from typing import Literal
from functools import partial
import pathlib
import warnings


def _getSgfNeurons(stats_df, src_activitiy_df, src_col, # src_ol="quantile_idx"
                   left_tune_col, pval, data_col, prior_data_col=None,
                   src_col_val=None, # src_col_val={1|2|3]
                   stats_col_val=None):
  stats_df = stats_df[stats_df.data_col == data_col]
  if prior_data_col is not None and isinstance(prior_data_col, str):
    stats_df = stats_df[stats_df.prior_data_col ==  prior_data_col]
  else:
    assert prior_data_col is None or np.isnan(prior_data_col)
    stats_df = stats_df[stats_df.prior_data_col.isnull()]
  if src_col is not None:
    stats_has_quantiles = src_col in stats_df.columns
    if stats_col_val is not None:
      assert stats_has_quantiles
      stats_df = stats_df[stats_df[src_col] == stats_col_val]
    elif stats_has_quantiles:
      stats_df = stats_df[stats_df[src_col] == src_col_val]
  else:
    assert stats_df.long_trace_id.nunique() == len(stats_df), (
      f"Found repeated neurons in stats_df when src_col is None. Don't know "
       "which trace_id to select.")

  # quantile_neurons = stats_df[(stats_df.quantile_idx == stats_quantile_idx) &
  #                             (stats_df.data_col == "ChoiceLeft") &
  #                             (stats_df.prior_data_col.isnull())]
  if src_col_val is not None:
    quantile_src_activitiy_df = src_activitiy_df[src_activitiy_df[src_col] ==
                                                 src_col_val]
  else:
    quantile_src_activitiy_df = src_activitiy_df
  non_sgf_choice = stats_df[stats_df.pval > pval]
  sgf_choices = stats_df[stats_df.pval <= pval]
  left_sgf_choice  = sgf_choices[sgf_choices[left_tune_col] == True]
  right_sgf_choice = sgf_choices[sgf_choices[left_tune_col] == False]
  # Assert first no repeated neurons
  non_sgf_choice = non_sgf_choice.sort_values("long_trace_id")
  assert non_sgf_choice.long_trace_id.nunique() == len(non_sgf_choice), display(
         non_sgf_choice[non_sgf_choice.duplicated("long_trace_id", keep=False)])
  assert left_sgf_choice.long_trace_id.nunique() == len(left_sgf_choice)
  assert right_sgf_choice.long_trace_id.nunique() == len(right_sgf_choice)
  # Assert no overlap between the groups
  assert left_sgf_choice.long_trace_id.isin(
                                      non_sgf_choice.long_trace_id).sum() == 0,(
         display(left_sgf_choice))
  assert right_sgf_choice.long_trace_id.isin(
                                      non_sgf_choice.long_trace_id).sum() == 0
  assert left_sgf_choice.long_trace_id.isin(
                                      right_sgf_choice.long_trace_id).sum() == 0
  # Convert the other df to have the same left vs right sigf.
  quantile_src_activitiy_df = quantile_src_activitiy_df.copy()
  quantile_src_activitiy_df["is_left_choice_tuned"] = np.nan
  quantile_src_activitiy_df.loc[quantile_src_activitiy_df.long_trace_id.isin(
                                                 left_sgf_choice.long_trace_id),
                                "is_left_choice_tuned"] = 1
  quantile_src_activitiy_df.loc[quantile_src_activitiy_df.long_trace_id.isin(
                                                right_sgf_choice.long_trace_id),
                                "is_left_choice_tuned"] = 0
  display(
      quantile_src_activitiy_df.is_left_choice_tuned.value_counts(dropna=False))
  return quantile_src_activitiy_df


class BaseTuningPlot:
  # def nFigs(self):
  #   raise NotImplementedError()
  def nColsRows(self):
    raise NotImplementedError()
  def figSize(self):
    raise NotImplementedError()
  def plot(self):
    raise NotImplementedError()

class TrajectorySEM(Enum):
  BinMeanSEM = auto()
  NeuronMeanSEM = auto()

class TrajectoryTuningPlot(BaseTuningPlot):
  def __init__(self, method :Literal["mean", "sum"],  skip_qidx2, combine_sides,
               combine_brain_regions, combine_in_single_plot, label, plot_sem,
               sem_type : TrajectorySEM = TrajectorySEM.BinMeanSEM,
               plot_non_preferred=True, plot_all=True, plot_non_sgf=True,
               fig_size_col=15, y_lims=None, colors_preferred={},
               colors_non_preferred={}, bins_count=None,
               neuronFiringBinFn=None, as_weight=False,
               plot_weight_only_first_last=False,
               plt_single_pts=False, plt_linear_fit=False,
               plt_is_no_quantiles=False):
    self._method = method
    self._combine_sides = combine_sides
    self._combine_brain_regions = combine_brain_regions
    self._combine_in_single_plot = combine_in_single_plot
    self._label = label
    self._plot_non_preferred = plot_non_preferred
    self._plot_all = plot_all
    self._plot_non_sgf = plot_non_sgf
    self._plot_sem = plot_sem
    self._sem_type = sem_type
    self._y_lims = y_lims
    self._colors_preferred = colors_preferred
    self._colors_non_preferred = colors_non_preferred
    self._row_idx = 0
    self._bins_count = bins_count
    self._neuronFiringBinFn = neuronFiringBinFn
    self._skip_qidx2 = skip_qidx2
    self._as_weight = as_weight
    self._plot_weight_only_first_last = plot_weight_only_first_last
    self._plt_single_pts = plt_single_pts
    self._plt_linear_fit = plt_linear_fit
    self._plt_is_no_quantiles = plt_is_no_quantiles
    self._createFig(fig_size_col)

  def _createFig(self, fig_size_col):
    # 3 = 3 quantiles
    n_cols = 1 if (self._combine_in_single_plot or self._plt_is_no_quantiles) \
             else (2 if self._skip_qidx2 else 3)
    if not self._combine_sides:
      n_cols *= 2 # For left and right
    n_rows = 1 if self._combine_brain_regions else 2 # 2 brain-regions
    self._fig, self._axs = plt.subplots(n_rows, n_cols,
                                        figsize=(fig_size_col*n_cols,
                                                 5*n_rows))
    if n_rows == 1:
      self._axs = [self._axs]

  def _lineStyle(self, quantile_idx):
    if not self._combine_in_single_plot:
      return "solid"
    elif quantile_idx == 3:
      return "solid"
    elif quantile_idx == 2:
      return "--"
    else:
      return ":"

  def startNewBrainRegion(self, br_dscrp_str):
    self._br_dscrp_str = br_dscrp_str
    self._col_idx = 0
    self._last_dir = -1 # Any unused intial value
    self._x_ticks_side_li = {} # TODO: Map direction + quantile
    self._x_ticks_labels_side_li = {}
    self._cur_twinax = []

  def finishBrainRegion(self):
    axs = self._axs[self._row_idx]
    x_ticks_li = list(self._x_ticks_side_li.values())
    x_ticks_labels_li = list(self._x_ticks_labels_side_li.values())
    if self._combine_sides:
      # x_ticks_li = {li for li, _ in x_ticks_li}
      # x_ticks_labels_li = {li for li, _ in x_ticks_labels_li}
      _dir = ["Combined Sides"]
      if self._combine_in_single_plot or self._plt_is_no_quantiles:
        axs = [axs] # Encapsulate in an iterator
    else:
      _dir = ["Left Side", "Right Side"]
      x_ticks_li = x_ticks_li[0] + x_ticks_li[1]
      x_ticks_labels_li = x_ticks_labels_li[0] + x_ticks_labels_li[1] # Why?
      # Re-order as they are L. Imp./Del., R. Impl./Del.
      # reorder to: L./R. Imp., L./R. Imp, L./R. Del.
      x_ticks_li = x_ticks_li[::2] + x_ticks_li[1::2]
      x_ticks_labels_li = x_ticks_labels_li[::2] + x_ticks_labels_li[1::2]

    if self._combine_in_single_plot:
      x_ticks_li = [np.array([li for li in x_ticks_li ]).flatten()]
      x_ticks_labels_li = [np.array([li for li in x_ticks_labels_li]).flatten()]
    else:
      _dir = [f"{_str} - {q_str}"
              for _str in _dir
              for q_str in (("All Strategies",) if self._plt_is_no_quantiles else
                            ("Impulsive", "Deliberate") if self._skip_qidx2 else
                            ("Impulsive", "Typical ","Deliberate"))]
      if self._combine_sides:
        x_ticks_li = x_ticks_li[0] # Why?
        x_ticks_labels_li = x_ticks_labels_li[0]

    assert len(_dir) == len(axs), f"{len(_dir)} != {len(axs)}"
    assert len(x_ticks_li) == len(axs), f"{len(x_ticks_li)} != {len(axs)}"
    assert len(x_ticks_labels_li) == len(axs), (f"{len(x_ticks_labels_li)} != "
                                                f"{len(axs)}")

    axs_dir_x_lis = zip(axs, _dir, x_ticks_li, x_ticks_labels_li)
    method_str = self._method[0].upper() + self._method[1:]
    max_x = -1
    for ax, _dir, x_ticks, x_ticks_labels in axs_dir_x_lis:
      # Remove duplicates
      handles, labels = ax.get_legend_handles_labels()
      by_label = dict(zip(labels, handles))
      ax.legend(by_label.values(), by_label.keys(),
                fontsize="small", #loc=None if not self._bins_count else
                                  #    "upper left"
                                      )
      ax.set_title(f"{self._br_dscrp_str} - {_dir}")
      if self._as_weight and self._plot_weight_only_first_last:
        # Do nothing, the x-ticks and labels has already been set
        ax.set_title("Slow" if self._row_idx == 0 else "Fast")
        ax.spines['left'].set_position(('axes', 0.5))
      else:
        # Use list() to convert generators to list, so matplotlib remains happy
        x_ticks = list(x_ticks)
        x_ticks_labels = list(x_ticks_labels)
        # print("x_ticks:", x_ticks)
        ax.set_xticks(x_ticks)
        [ax.axvline(tick, color="gray", ls="--") for tick in x_ticks]
        ax.set_xticklabels(x_ticks_labels)
      # ax.set_xlim(left=-0.2)
      ax.set_xlabel("Time")
      ax.set_ylabel(f"Subpopulation {method_str} of Mean Z-Scored Activity")
      if self._y_lims is None:
        if self._method == "mean":
          ax.set_ylim(-0.2 if not self._combine_sides else -0.3, 1.5)
        elif method_str == "sum":
          # ax.set_ylim(-2000, 7000)
          ax.set_ylim(-1800, 6000)
          # ax.set_ylim(-25, 110)
      else:
        ax.set_ylim(self._y_lims)
      max_x = max(max_x, ax.get_xlim()[1])

    if not isinstance(axs, list):
      axs = list(axs)
    for ax in axs:
      ax.set_xlim(right=max_x)
    for _ax in axs + self._cur_twinax:
      _ax.spines["top"].set_visible(False)
      _ax.spines["right"].set_visible(False)
      _ax.spines["left"].set_visible(False)
      _ax.spines["bottom"].set_visible(False)
    self._row_idx += 1


  def plot(self, choice_dir,
           all_df, preferred_df, non_preferred_df, non_sgf_df,
           trace_count_all, trace_count_preferred, trace_count_non_preferred, trace_count_non_sgf,
           quantile_idx=None):
    ls = self._lineStyle(quantile_idx)
    axs = self._axs[self._row_idx]
    if (self._combine_sides and self._combine_in_single_plot) or self._plt_is_no_quantiles:
      ax =  axs
    else:
      ax = axs[self._col_idx]
      print("Col idx:", self._col_idx, "Choice dir:", choice_dir, "last_dir:", self._last_dir)
      # if not self._combine_in_single_plot and choice_dir != self._last_dir:
      if not self._combine_in_single_plot:
        # Ii that right?
        self._col_idx += 1
        self._last_dir = choice_dir

    def _getRows(df):
      return np.array(df.Raw.values.tolist())

    # print(preferred_df.Raw.iloc[0].values.notnull())
    # ex_len = pd.Series(preferred_df.Raw.iloc[0].values).count()
    # bins_raw = np.linspace(0, ex_len, self._bins_count)
    # bins = bins_raw# bins = np.around(bins_raw).astype(int)
    # epochs_ranges =
    # print("Epochs ranges:", epochs_ranges)
    before, mid, after = all_df.iloc[0].epochs_ranges[0]
    bins = [before[0]] + \
            list(np.linspace(mid[0], after[0], self._bins_count-1, endpoint=True)) + \
            [after[1]+1]
    # print(len(bins), bins)
    bins = np.array(bins)
    # print("For example length:", ex_len, "- got:", bins_raw)
    print("Rounded to:", bins)
    # neurons_firing_pos, _bins = self._neuronFiringBinFn(num_bins=self._bins_count)
    neuron_firing_iqr_df = self._neuronFiringBinFn
    # display(neurons_firing_pos)

    def getMeanActivity(df, ignore_bins=False):
      def meanNeuronTrials(sub_df):
          traces = _getRows(sub_df)
          with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            neuron_as_single_trace = np.nanmean(traces, axis=0)
          try:
            sem = stats.sem(traces, axis=0)
          except FloatingPointError: # divide by zero encountered in divide
            sem = np.full(neuron_as_single_trace.shape, np.nan)
          # print("Neuron as single trace:", neuron_as_single_trace)
          # return neuron_as_single_trace
          return pd.Series({
            'cur_neuron_trials_mean_activity_per_frame': neuron_as_single_trace,
            'cur_neuron_trials_mean_activity_sem_per_frame': sem})

      if ignore_bins:
        df_traces_grouped = df.groupby("long_trace_id",
                                         as_index=False).apply(meanNeuronTrials)
        count = len(df_traces_grouped)
        df_traces = np.array(
          df_traces_grouped.cur_neuron_trials_mean_activity_per_frame.values.tolist())
        # rows = _getRows(df)
        if self._method == "sum":
          # Don't use nansum here to avoid trailing zeros on short trials
          # print("Found", count, "rows")
          res_data = np.sum(df_traces, axis=0)/count
        else:
          assert self._method == "mean"
          res_data =  np.nanmean(df_traces, axis=0)
        return res_data

      bins_dfs = []
      df = df.copy()
      df["long_trace_id"] = df["ShortName"] + "_" + df.trace_id.astype(str)
      count = df.long_trace_id.nunique()
      max_len = bins[-1]
      bin0_r_prcnt = 100*bins[1]/max_len
      binlast_l_prcnt = 100*bins[-2]/max_len
      # print(bin0_r_prcnt)
      # print(binlast_l_prcnt)
      for bin_idx, (bin_l, bin_r) in enumerate(zip(bins[:-1], bins[1:])):
        # bin_idxs = [bin_idx-1, bin_idx, bin_idx+1]
        bin_l_prcnt, bin_r_prcnt = 100*bin_l/max_len, 100*bin_r/max_len
        bin_df = neuron_firing_iqr_df.query(f"({bin_l_prcnt} <= q1_prcnt <= {bin_r_prcnt} or "
                                            f"{bin_l_prcnt} <= q3_prcnt <= {bin_r_prcnt})"
                                            f"and med_prcnt >= {bin0_r_prcnt} and med_prcnt <= {binlast_l_prcnt}"
                                            )
        # bin_df = neurons_firing_pos[neurons_firing_pos.bin_idx.isin(bin_idxs)]
        bin_neurons_df = df[df.long_trace_id.isin(bin_df.trace_id)]
        # print(f"Mum of neurons between {bin_l_prcnt} <= (q1 pr q3) <= {bin_r_prcnt} = "
        #       f"{bin_neurons_df.trace_id.nunique()}/{df.trace_id.nunique()}")
        if not len(bin_neurons_df):
          # print("No neurons in this bin")
          continue
        bin_traces_grouped = bin_neurons_df.groupby("long_trace_id",
                                         as_index=False).apply(meanNeuronTrials)

        bin_traces_grouped_df = bin_traces_grouped.copy()

        bin_traces_mean = np.array(bin_traces_grouped.cur_neuron_trials_mean_activity_per_frame.values.tolist())
        bin_traces_sem = np.array(bin_traces_grouped.cur_neuron_trials_mean_activity_sem_per_frame.values.tolist())
        # print("Shape before:", bin_traces.shape)
        # print(bin_l, bin_r, bin_idxs, "lem bin_neurons_df:", bin_neurons_df.long_trace_id.nunique(), "len bin_traces_grouped:", len(bin_traces_grouped))
        bin_traces_mean = bin_traces_mean[:, int(bin_l):int(bin_r)]
        bin_traces_sem = bin_traces_sem[:, int(bin_l):int(bin_r)]
        # print(bin_traces.shape)

        for idx, (val, sem) in enumerate(
                                      zip(bin_traces_mean.T, bin_traces_sem.T),
                                      start=int(bin_l)):
          # val and sem are arrays
          bin_traces_grouped_df[f"m{idx}"] = val
          bin_traces_grouped_df[f"s{idx}"] = sem
        # display(bin_traces_grouped_df)
        # count = len(bin_traces_grouped)

        # bin_traces is 2d array, where each row is a neuron mean activity
        # across the subset of frames that this bin covers
        # if self._method == "sum":
        #   # Don't use nansum here to avoid trailing zeros on short trials
        #   # print("Found", count, "rows")
        #   calc_res_mean = np.sum(bin_traces_mean, axis=0)/count
        # else:
        #   assert self._method == "mean"
        #   # print("Bin idx:", bin_idx, f"l-r: {bin_l_prcnt:.1f}-{bin_r_prcnt:.1f}",
        #   #        "bin_traces:", bin_traces_mean)
        #   calc_res_mean =  np.nanmean(bin_traces_mean, axis=0)
        calc_res_mean = bin_traces_mean

        if self._sem_type == TrajectorySEM.BinMeanSEM:
          calc_res_sem = stats.sem(bin_traces_mean, axis=0)
          for idx, bin_sem in enumerate(calc_res_sem, start=int(bin_l)):
            # bin sem is a single value here repeated for all the columns
            bin_traces_grouped_df[f"frame_sem_{idx}"] = bin_sem
        else:
          assert self._sem_type == TrajectorySEM.NeuronMeanSEM
          calc_res_sem = np.nanmean(bin_traces_sem)
        # print("Calc res:", calc_res_mean.shape, "sem:", calc_res_sem.shape)
        # Now take the mean of the sem
        # print(bin_traces.shape, calc_res.shape)
        # WE will calcaulte this later
        # if not self._as_weight:
        #   calc_res_mean = np.nanmean(calc_res_mean)
        if self._method == "mean":
          calc_res_sem = np.nanmean(calc_res_sem) # Reduce to a single point
          bin_traces_grouped_df[f"frames_sem_mean"] = calc_res_sem
        else:
          assert self._method == "sum"
          calc_res_sem = np.nansum(calc_res_sem)
        # display(bin_traces_grouped_df)
        bins_dfs.append(bin_traces_grouped_df)
      return bins_dfs

    if self._plot_sem is None:
      self._plot_sem = self._method == "mean"
    if self._plot_non_sgf is None:
      self._plot_non_sgf = self._method == "sum"
    pref_bins_df_li = getMeanActivity(preferred_df)
    non_pref_bins_df_li =  getMeanActivity(non_preferred_df)
    mean_all = getMeanActivity(all_df, ignore_bins=True)

    # print("Choice dir:", choice_dir, "quantile_idx:", quantile_idx, "br:", br, "- len:", len(choice_dir_df))
    # print("preferred_dir_df:", len(preferred_dir_df), "mean:", mean_pref)
    # print("non_preferred_dir_df:", len(non_preferred_dir_df), "mean:", mean_non_pref)
    # print("non_sgf_dir_df:", len(non_sgf_dir_df), "mean:", mean_all)
    # print()
    print_labels_once = True #quantile_idx == 1 #and choice_dir
    dir_str =      (": Left" if choice_dir else " Right") if not self._combine_sides else ""
    anti_dir_str = (": Right" if choice_dir else " Left") if not self._combine_sides else ""

    q_prefix = f"Q={quantile_idx} " if quantile_idx is not None else ""
    color_pref = self._colors_preferred.get(quantile_idx, '#0C8140')
    color_non_pref = self._colors_non_preferred.get(quantile_idx, "#813219")
    xs = np.arange(len(pref_bins_df_li)) if bins is None else bins[:-1]
    # print("Xs:", xs)
    # print("mean_pref:", mean_pref)

    def getDFCols(bin_df, col_initial):
      # Reconstruct our means as 2d array
      bin_df_vals_columns = [col
                       for col in bin_df.columns if col.startswith(col_initial)]
      bin_pts_df = bin_df[["long_trace_id"] + bin_df_vals_columns]
      bin_pts_df = bin_pts_df.set_index("long_trace_id")
      # display(bin_pts_df)
      return bin_pts_df

    def binDfPtsMean(bin_df, col_initial):
      bin_pts_df = getDFCols(bin_df, col_initial)
      # Average frames together per bin
      bin_pts_df = bin_pts_df.mean(axis=1)
      # display(bin_pts_df.mean(axis=1))
      if self._method == "mean":
        pt_mean = np.nanmean(bin_pts_df)
        pt_sem = stats.sem(bin_pts_df)
      else:
        pt_mean = np.nansum(bin_pts_df)
        pt_sem = None
      return pt_mean, pt_sem
    if bins is not None:
      widths = np.diff(bins)
    if self._as_weight and self._plot_weight_only_first_last:
      plot_xs = []
      plot_ys = []
      plot_ys_sem = []
      plot_annots = []

    def binTracesMean(bin_df, col_initial):
      bin_pts_df = getDFCols(bin_df, col_initial)
      # Average frames together per trace
      bin_pts_df = bin_pts_df.mean(axis=1)
      return bin_pts_df

    x_pref_loc : float = None
    y_pref_pts : pd.Series | List[pd.Series] = None if not self._as_weight else []
    y_non_pref_pts : pd.Series | List[pd.Series] = None if not self._as_weight else []
    bins_vals_pref : List[float] = []
    bins_vals_non_pref : List[float] = []
    plotFn = partial(ax.plot, ls=ls) if bins is None else partial(ax.bar, edgecolor='k', align='edge')#, width=np.diff(bins))
    # Plot together in case we have overlapping bars
    # Adapted from here: https://stackoverflow.com/a/53276536/11996983
    labels = [f"{q_prefix}Preferred{dir_str} (#N={trace_count_preferred:,})" if print_labels_once else None,
              f"{q_prefix}Anti-Preferred{anti_dir_str} (#N={trace_count_non_preferred:,})" if print_labels_once else None]
    for i, (x,  pref_bin_df,     non_pref_bin_df) in enumerate(zip(
            xs, pref_bins_df_li, non_pref_bins_df_li)):
      if not self._as_weight:
        for j, (xx, bin_df, c, label, should_plot) in enumerate(
                                        sorted(zip([0, 1],
                                                   [pref_bin_df, non_pref_bin_df],
                                                   [color_pref, color_non_pref],
                                                   labels,
                                                   [True, self._plot_non_preferred,]))):
            if not should_plot:
              continue
            GAP_SPACE = 0.3
            extra_kwargs = dict(width=widths[i]/2 - GAP_SPACE)  if bins is not None else {}
            if self._plt_linear_fit:
              extra_kwargs["alpha"] = .3
            xx = x if bins is None else x + (xx*extra_kwargs["width"] - GAP_SPACE)
            # Reconstruct our means as 2d array
            pt_mean, pt_sem = binDfPtsMean(bin_df, col_initial="m")
            annot_str = f"{pt_mean:.2f}"
            if self._plot_sem:
              # pt_sem = BinDdPtsMean(bin_df, col_initial="s")
              extra_kwargs["yerr"] = pt_sem
              annot_str += f" ±{pt_sem:.2f}"
            plotFn(xx, pt_mean, color=c, ls=ls, zorder=-j, label=label,
                    **extra_kwargs)
            if not self._plt_linear_fit:
              ax.annotate(annot_str, (xx + extra_kwargs.get("width", 0)/2, pt_mean + 0.3),
                          #textcoords="offset points", xytext=(0, 5),
                          ha='center', fontsize=8, color=c, rotation=90)
            if self._plt_linear_fit:
              _li = bins_vals_pref if j == 0 else bins_vals_non_pref
              _li.append(pt_mean)
            if self._plt_single_pts:
              pts_vals = binTracesMean(bin_df, col_initial="m")
              if j == 0: # We are in preferred
                x_pref_loc = xx
                y_pref_pts = pts_vals
              else: # Anti-preferred
                # Plot each preferred point to its corresponding non-preferred
                # display(y_pref_pts)
                # display(pts_vals)
                x_left = x_pref_loc
                x_right = xx
                x_plot = [x_left + 0.4 + extra_kwargs.get("width", 0)/2,
                          x_right - 0.4 + extra_kwargs.get("width", 0)/2]
                for pt_idx, pt in y_pref_pts.items():
                  y_right = pts_vals[pts_vals.index == pt_idx]
                  assert len(y_right) == 1, f"Expected single point, got: {y_right}"
                  y_plot = [pt, y_right.values[0]]
                  ax.plot(x_plot, y_plot,
                          marker='o', color="gray", markerfacecolor="white",
                          markeredgecolor="black", ms=5,
                          alpha=0.3, zorder=1)
      else:
        # Terrible coding with the nexted if
        if labels[0] is not None: # IF it was reset below
          label = f"{q_prefix} Coding-Eff{dir_str} (#N={trace_count_preferred:,})" if print_labels_once else None

        assert self._method == "mean"
        # merged_df = pd.merge(pref_bin_df, non_pref_bin_df,
        #                      on="long_trace_id",
        #                      suffixes=["_pref", "_non_pref"])
        # Given our weight equation equal to:
        # 100*(pref - non_pref)/pref
        # And pref and non_pref sem as pref_sem and non_pref_sem
        # Verifying the formula against this calculator:
        # https://uncertaintycalculator.com/
        # Or this one:
        # https://nicoco007.github.io/Propagation-of-Uncertainty-Calculator/
        # Then the propogation of error approximation is:
        # sqrt(
        # (((100*pref_mean - 100*(pref_mean-non_pref_mean))/pref_mean^2)*pref_sem)^2 +
        # ((-100/pref_mean)*non_pref_sem)^2))
        # \[\begin{aligned} \Delta Error &= \sqrt{\left( \frac{\partial Error}{\partial Match} \Delta Match \right)^2 + \left( \frac{\partial Error}{\partial Mismatch} \Delta Mismatch \right)^2} \\ &= \sqrt{\left( \left( 100 / Mismatch \right) \cdot \Delta Match \right)^2 + \left( \left( -(Match * 100 / Mismatch ^ 2) \right) \cdot \Delta Mismatch \right)^2} \\ &= 0.0002554166 \\ &= 2.554166 \times 10^{-4} \\ &= 3 \times 10^{-4} \end{aligned}\]\[Error = \left( -9.99628 \pm 0.00003 \right) \times 10^{1}\]
        pref_mean, pref_sem = binDfPtsMean(pref_bin_df, col_initial="m")
        non_pref_mean, non_pref_sem = binDfPtsMean(non_pref_bin_df, col_initial="m")
        # pref_sem = BinDdPtsMean(pref_bin_df, col_initial="s")
        # non_pref_sem = BinDdPtsMean(non_pref_bin_df, col_initial="s")
        # print("Pref sem:", pref_sem, "Non pref sem:", non_pref_sem)
        bin_eff_mean = 100*(pref_mean-non_pref_mean)/pref_mean
        bin_eff_sem = np.sqrt(
          (((100*pref_mean - 100*(pref_mean-non_pref_mean))/pref_mean**2)*pref_sem)**2 +
          ((-100/pref_mean)*non_pref_sem)**2)

        pref_pts = binTracesMean(pref_bin_df, col_initial="m")
        non_pref_pts = binTracesMean(non_pref_bin_df, col_initial="m")
        y_pref_pts.append(pref_pts)
        y_non_pref_pts.append(non_pref_pts)

        annot_str = f"{bin_eff_mean:.2f} ±{bin_eff_sem:.2f}"
        if self._plot_weight_only_first_last:
          plot_xs.append(x)
          plot_ys.append(bin_eff_mean)
          plot_ys_sem.append(bin_eff_sem)
          plot_annots.append(annot_str)
          continue

        extra_kwargs = dict(width=widths[i]) if bins is not None else {}
        if self._plot_sem:
          assert self._sem_type == TrajectorySEM.BinMeanSEM
          if not np.isnan(bin_eff_sem):
            extra_kwargs["yerr"] = bin_eff_sem
        plotFn(x, bin_eff_mean, color=color_pref, ls=ls, label=label, **extra_kwargs,
               hatch="ooo", edgecolor=color_non_pref)
        ax.annotate(annot_str,
                    (x + extra_kwargs.get("width", 0)/2, bin_eff_mean + bin_eff_sem + 3),
                    #textcoords="offset points", xytext=(0, 5),
                    ha='center', fontsize=8, color=color_pref, rotation=90)
      labels = [None, None] # Print only once
      print_labels_once = False

    if self._plt_linear_fit and len(bins_vals_pref) == len(bins) - 1:
      # from sklearn.stats import LinearRegression
      from scipy.stats import linregress
      OFFSET_FIRST = 1
      _bins = np.array(bins[OFFSET_FIRST:-1]) + \
              np.diff(bins[OFFSET_FIRST:])/2
      for _bin_vals, clr in zip([bins_vals_pref, bins_vals_non_pref],
                                [color_pref, color_non_pref]):
        x = _bins[:].reshape(-1, 1) - (GAP_SPACE*2)
        # x += (np.diff(bins).mean()/2) # Center of bins
        y = np.array(_bin_vals[OFFSET_FIRST:])
        # model = LinearRegression()
        # model.fit(x, y)
        # y_pred = model.predict(x)
        slope, intercept, r_value, p_value, std_err = linregress(x.flatten(), y)
        y_pred = intercept + slope * x
        ax.plot(x.flatten(), y_pred, color=clr, ls="--", lw=3,
                label=(f"Linear fit ({'Preferred' if _bin_vals is bins_vals_pref else 'Anti-Preferred'}) "
                       f"r={r_value:.2f}, p={p_value:.3f}"))

    if self._as_weight and self._plot_weight_only_first_last:
      first_idx = 1 # Skip first bin as it's moving to port
      x_offset = 0
      x_ticks = []
      x_labels = []
      assert len(y_pref_pts) == len(plot_ys)
      assert len(y_non_pref_pts) == len(plot_ys)
      for last_idx, last_bin_label in [(-2, "before last bin"),
                                       (-1, "last bin")]:
        # A dirty hack to get the right axis: Put fast and slow together, and
        # plot slow on top row
        ax = self._axs[1 if quantile_idx == 1 else 0]
        br = BrainRegion(all_df.BrainRegion.iloc[0])
        if self._combine_in_single_plot:
          # for _ in range(quantile_idx):
          x_offset += 0.1 if br == BrainRegion.ALM_Bi else -0.1
        xs = [0 + x_offset, 3 + x_offset]
        ys = [plot_ys[first_idx], plot_ys[last_idx]]
        ys_sem = [plot_ys_sem[first_idx], plot_ys_sem[last_idx]]
        annots = [plot_annots[first_idx], plot_annots[last_idx]]
        clr = BRCLr[br]
        ax.errorbar(xs, ys, yerr=ys_sem, color=clr, #ls=ls
                    lw=3,
                    path_effects=[pe.Stroke(linewidth=4,
                                            foreground='r' if quantile_idx == 1 else 'yellow'), pe.Normal()])
        x_ticks += xs
        x_labels += ["Second bin", last_bin_label]
        for x, y, y_sem, annot in zip(xs, ys, ys_sem, annots):
          ax.annotate(annot, (x - .05, y + y_sem + 3),
                      ha='center', fontsize=8, color=clr, rotation=90)
        if self._plt_single_pts:
          for _idx, _x in [(first_idx, xs[0]), (last_idx, xs[1])]:
            cur_y_pref_pts = y_pref_pts[_idx]
            cur_y_non_pref_pts = y_non_pref_pts[_idx]
            eff = 100*(cur_y_pref_pts - cur_y_non_pref_pts)/cur_y_pref_pts
            ax.scatter([_x]*len(eff), eff, facecolor=clr, #"white",
                       #edgecolor="black",
                       s=5, alpha=0.6, zorder=1)
        x_offset += 5
      ax.set_xticks(x_ticks)
      ax.set_xticklabels(x_labels)
      self._plot_non_sgf = False # Force not plotting non-sgf

    if self._plot_non_sgf:
      if bins is not None:
        self._cur_twinax.append(ax.twinx())
        plotFn = self._cur_twinax[-1].scatter
        non_sgf_xs = np.arange(len(mean_all))
        extra_kwargs = dict()
        label = f"{q_prefix}Population (#N={trace_count_all:,})"
        non_sgf_clr = BRCLr[BrainRegion(all_df.BrainRegion.iloc[0])]
      else:
        non_sgf_xs = xs
        extra_kwargs = dict(ls=ls)
        label = f"{q_prefix}Non-Sgf (#N={trace_count_non_sgf:,})"
        non_sgf_clr = "grau"
      label = label if print_labels_once else None
      plotFn(non_sgf_xs,  mean_all, color=non_sgf_clr, label=label,
              **extra_kwargs)
    # if self._plot_sem:
    #   ax.fill_between(xs, mean_pref - sem_pref,         mean_pref + sem_pref,         alpha=0.3, color=color_pref)
    #   if self._plot_non_preferred:
    #     ax.fill_between(xs, mean_non_pref - sem_non_pref, mean_non_pref + sem_non_pref, alpha=0.3, color=color_non_pref)
    #   # if self._plot_non_sgf:
    #   #   ax.fill_between(xs, mean_all - sem_non_sgf,   mean_all + sem_non_sgf,   alpha=0.3, color="gray")

    ex_row = preferred_df.iloc[0]
    epochs_ranges = ex_row.epochs_ranges[0][1:]
    epochs_names  = ex_row.epochs_names[0][1:]
    # Append to list of lists
    x_ticks_li = self._x_ticks_side_li.get(str(choice_dir), [])
    x_ticks_labels_li = self._x_ticks_labels_side_li.get(str(choice_dir), [])
    x_ticks_li.append([rng[0] for rng in epochs_ranges])
    x_ticks_labels_li.append(epochs_names)
    self._x_ticks_side_li[str(choice_dir)] = x_ticks_li
    self._x_ticks_labels_side_li[str(choice_dir)] = x_ticks_labels_li

  '''
  def plot2(self, choice_dir,
           all_df, preferred_df, non_preferred_df, non_sgf_df,
           trace_count_all, trace_count_preferred, trace_count_non_preferred, trace_count_non_sgf,
           quantile_idx=None):
    ls = self._lineStyle(quantile_idx)
    axs = self._axs[self._row_idx]
    if self._combine_sides and self._combine_in_single_plot:
      ax =  axs
    else:
      ax = axs[self._col_idx]
      if not self._combine_in_single_plot and choice_dir != self._last_dir:
        # Ii that right?
        self._col_idx += 1
        self._last_dir = choice_dir

    def _getRows(df):
      return np.array(df.Raw.values.tolist())

    if self._bins_count is not None:
      ex_len = len(preferred_df.Raw.iloc[0].values)
      bins_raw = np.linspace(0, ex_len, self._bins_count)
      bins = np.around(bins_raw).astype(int)
      print("For example length:", ex_len, "- got:", bins_raw)
      print("Rounded to:", bins)
    else:
      bins = None
    def getMeanActivity(df, count, ignore_bins=False):
      rows = _getRows(df)
      if self._method == "sum":
        # Don't use nansum here to avoid trailing zeros on short trials
        # print("Found", count, "rows")
        res_data = np.sum(rows, axis=0) #/count
      else:
        assert self._method == "mean"
        res_data =  np.nanmean(rows, axis=0)

      if bins is not None and not ignore_bins:
        res_data_new = np.empty(len(bins)-1)
        for bin_idx, (bin, next_bin) in enumerate(zip(bins[:-1], bins[1:])):
          bin_vals_avg = res_data[bin:next_bin]
          nan_bools = np.isnan(bin_vals_avg)
          bin_vals_avg = bin_vals_avg[~nan_bools] # Avoid warn on next line
          if len(bin_vals_avg):
            bin_vals_avg = np.nanmean(bin_vals_avg)
            # res_data[bin:next_bin] = bin_vals_avg
            # nan_idxs = bin + np.where(nan_bools)[0]
            # # print("nan_idxs:", nan_idxs)
            # res_data[nan_idxs] = np.nan
            res_data_new[bin_idx] = bin_vals_avg
          else:
            res_data_new[bin_idx] = np.nan
        res_data = res_data_new
      # print("res_data:", res_data.shape)
      # print(res_data)
      return res_data

    if self._plot_sem is None:
      self._plot_sem = self._method == "mean"
    if self._plot_non_sgf is None:
      self._plot_non_sgf = self._method == "sum"
    mean_pref = getMeanActivity(preferred_df, trace_count_preferred)
    mean_non_pref =  getMeanActivity(non_preferred_df, trace_count_non_preferred)
    mean_non_sgf = getMeanActivity(non_sgf_df, trace_count_non_sgf, ignore_bins=True)
    mean_all = getMeanActivity(all_df, trace_count_all, ignore_bins=True)
    if bins is not None:
      mean_non_sgf = mean_all
      min_offset = np.nanmin(mean_pref)
      if self._plot_non_preferred:
        min_offset = min(min_offset, np.nanmin(mean_non_pref))
      if self._plot_non_sgf:
        min_offset = min(min_offset, np.nanmin(mean_non_sgf))
      # mean_pref -= min_offset
      # mean_non_pref -= min_offset
      # mean_non_sgf -= min_offset
    # Calculate SEM
    if self._plot_sem and self._method == "mean":
      # sem_all = stats.sem(all_df.MeanActivity)
      sem_pref = stats.sem(_getRows(preferred_df), axis=0)
      sem_non_pref = stats.sem(_getRows(non_preferred_df), axis=0)
      sem_non_sgf = stats.sem(_getRows(non_sgf_df), axis=0)
      # print("sem_pref:", sem_pref)
    #
    # print("Choice dir:", choice_dir, "quantile_idx:", quantile_idx, "br:", br, "- len:", len(choice_dir_df))
    # print("preferred_dir_df:", len(preferred_dir_df), "mean:", mean_pref)
    # print("non_preferred_dir_df:", len(non_preferred_dir_df), "mean:", mean_non_pref)
    # print("non_sgf_dir_df:", len(non_sgf_dir_df), "mean:", mean_non_sgf)
    # print()
    print_labels_once = True #quantile_idx == 1 #and choice_dir
    dir_str =      (": Left" if choice_dir else " Right") if not self._combine_sides else ""
    anti_dir_str = (": Right" if choice_dir else " Left") if not self._combine_sides else ""

    q_prefix = f"Q={quantile_idx} " if quantile_idx is not None else ""
    color_pref = self._colors_preferred.get(quantile_idx, 'g')
    color_non_pref = self._colors_non_preferred.get(quantile_idx, "orange")
    # print("Mean pref:", mean_pref, "len preferred_df:", len(preferred_df))
    xs = np.arange(len(mean_pref)) if bins is None else bins[:-1]
    # print("Xs:", xs)
    # print("mean_pref:", mean_pref)
    if bins is not None:
      widths = np.diff(bins)
    plotFn = partial(ax.plot, ls=ls) if bins is None else partial(ax.bar, edgecolor='k', align='edge')#, width=np.diff(bins))
    # Plot together in case we have overlapping bars
    # Adapted from here: https://stackoverflow.com/a/53276536/11996983
    labels = [f"{q_prefix}Preferred{dir_str} (#N={trace_count_preferred:,})" if print_labels_once else None,
              f"{q_prefix}Anti-Preferred{anti_dir_str} (#N={trace_count_non_preferred:,})" if print_labels_once else None]
    for i, (x, pref_pt, non_pref_pt) in enumerate(zip(xs, mean_pref, mean_non_pref)):
      for j, (_, xx, pt, c, label, should_plot) in enumerate(sorted(zip([abs(pref_pt), abs(non_pref_pt)], # Use absolute for sorting
                                                                    [0, 1],
                                                                    [pref_pt, non_pref_pt],
                                                                    [color_pref, color_non_pref],
                                                                    labels,
                                                                    [True, self._plot_non_preferred,]))):
          if should_plot:
            extra_kwargs = dict(width=widths[i]/2) if bins is not None else {}
            xx = x if bins is None else x + (xx*extra_kwargs["width"])
            plotFn(xx, pt, color=c, ls=ls, zorder=-j, label=label, **extra_kwargs)
      labels = [None, None] # Print only once
    # plotFn(xs, mean_pref, color=color_pref, ls=ls,#"solid",
    #        label=f"{q_prefix}Preferred{dir_str} (#N={trace_count_preferred:,})" if print_labels_once else None)
    # if self._plot_non_preferred:
    #   plotFn(xs, mean_non_pref, color=color_non_pref, ls=ls,#"dashed",
    #          label=f"{q_prefix}Anti-Preferred{anti_dir_str} (#N={trace_count_non_preferred:,})" if print_labels_once else None)
    if self._plot_non_sgf:
      if bins is not None:
        self._cur_twinax.append(ax.twinx())
        plotFn = self._cur_twinax[-1].scatter
        non_sgf_xs = np.arange(len(mean_non_sgf))
        extra_kwargs = dict()
        label = f"{q_prefix}Population (#N={trace_count_all:,})"
      else:
        non_sgf_xs = xs
        extra_kwargs = dict(ls=ls)
        label = f"{q_prefix}Non-Sgf (#N={trace_count_non_sgf:,})"
      label = label if print_labels_once else None
      plotFn(non_sgf_xs,  mean_non_sgf, color="gray", label=label,
             **extra_kwargs)
    if self._plot_sem:
      ax.fill_between(xs, mean_pref - sem_pref,         mean_pref + sem_pref,         alpha=0.3, color=color_pref)
      if self._plot_non_preferred:
        ax.fill_between(xs, mean_non_pref - sem_non_pref, mean_non_pref + sem_non_pref, alpha=0.3, color=color_non_pref)
      if self._plot_non_sgf:
        ax.fill_between(xs, mean_non_sgf - sem_non_sgf,   mean_non_sgf + sem_non_sgf,   alpha=0.3, color="gray")

    ex_row = preferred_df.iloc[0]
    epochs_ranges = ex_row.epochs_ranges[0][1:]
    epochs_names  = ex_row.epochs_names[0][1:]
    # Append to list of lists
    x_ticks_li = self._x_ticks_side_li.get(str(choice_dir), [])
    x_ticks_labels_li = self._x_ticks_labels_side_li.get(str(choice_dir), [])
    x_ticks_li.append([rng[0] for rng in epochs_ranges])
    x_ticks_labels_li.append(epochs_names)
    self._x_ticks_side_li[str(choice_dir)] = x_ticks_li
    self._x_ticks_labels_side_li[str(choice_dir)] = x_ticks_labels_li
  '''

  def showAndSave(self, fig_save_prefix, save_fig):
    method_str = self._method[0].upper() + self._method[1:]
    # self._fig.suptitle(f"Subpopulation {method_str} of Mean Activity by "
    #                   f"Quantile and Choice Preference {self._label}")
    if self._label != "":
      self._fig.suptitle(self._label)
    if save_fig:
      dscrp_region = "combined_regions" if self._combine_brain_regions else "sep_regions"
      strategies = "all_strategies" if self._plot_all else "quantiles"
      dscrp_plot =  f"combined_{strategies}" if self._combine_in_single_plot else f"sep_{strategies}"
      if self._plt_linear_fit:
        dscrp_plot += "_with_linear_fit"
      if self._plt_single_pts:
        dscrp_plot += "_with_single_pts"
      dscrp_side = "combined_choices" if self._combine_sides else "sep_choices"
      label = "" if not len(self._label) else f"{self._label.lower().replace(' ', '_')}_"
      if self._as_weight:
        as_weight_str = ("weight_" if not self._plot_weight_only_first_last
                         else  "eff_")
      else:
        as_weight_str = ""
      common_prefix = (f"{fig_save_prefix}/RT_Stats/traces/"
                       f"{as_weight_str}{label}{dscrp_region}_"
                       f"{dscrp_plot}_{dscrp_side}_traces.svg")
      # correct_under_3s_only_active_sampling_and_move_
      save_path = pathlib.Path(common_prefix)
      save_path.parent.parent.mkdir(exist_ok=True)
      save_path.parent.mkdir(exist_ok=True)
      self._fig.savefig(save_path)
    plt.show()



class MeanTuningPlot:
  def __init__(self, method :Literal["mean", "sum"],  combine_sides,
               combine_brain_regions, label, plot_all=True, plot_non_sgf=True,
               group_neurons_trials = False):
    self._method = method
    self._combine_sides = combine_sides
    self._combine_brain_regions = combine_brain_regions
    self._group_neurons_trials = group_neurons_trials
    self._label = label
    self._plot_all = plot_all
    self._plot_non_sgf = plot_non_sgf
    self._row_idx = 0
    self._createFig()

  def _createFig(self):
    self._stride = 3
    if self._plot_all:
      self._stride += 1
    if self._plot_non_sgf:
      self._stride += 1
    n_cols = 1 if self._combine_sides else 2
    n_rows = 1 if self._combine_brain_regions else 2
    self._fig, self._axs = plt.subplots(n_rows, n_cols,
                                        figsize=(2*self._stride*n_cols,
                                                 8*n_rows))
    if n_rows == 1:
      self._axs = [self._axs]

  def startNewBrainRegion(self, br_dscrp_str):
    self._br_dscrp_str = br_dscrp_str
    self._already_printed_count = 0
    self._x_ticks = []
    self._x_ticks_labels = []

  def finishBrainRegion(self):
    axs = self._axs[self._row_idx]
    if self._combine_sides:
      axs_dir = [(axs, "Combined Sides")]
    else:
      axs_dir = zip(axs, ["Left Side", "Right Side"])
    method_str = self._method[0].upper() + self._method[1:]
    for ax, _dir in axs_dir:
      ax.legend()
      ax.set_title(f"{self._br_dscrp_str} - {_dir}")
      ax.set_xticks(self._x_ticks)
      ax.set_xticklabels(self._x_ticks_labels)
      ax.set_xlabel("Quantile")
      ax.set_ylabel(f"{method_str} of Mean Z-Scored Activity")
      ax.spines["top"].set_visible(False)
      ax.spines["right"].set_visible(False)
      ax.spines["left"].set_visible(False)
    self._row_idx += 1

  def plot(self, choice_dir, all_df, preferred_df, non_preferred_df, non_sgf_df,
           trace_count_all, trace_count_preferred, trace_count_non_preferred, trace_count_non_sgf,
           quantile_idx):

    ax = self._axs[self._row_idx]
    dir_str = (" Left" if choice_dir else " Right") if not self._combine_sides\
              else ""
    if dir_str != "":
      assert self._combine_sides
      ax = ax[0] if dir_str == " Left" else ax[1]
    if self._group_neurons_trials:
      all_df = all_df.groupby("long_trace_id")
      preferred_df = preferred_df.groupby("long_trace_id")
      non_preferred_df = non_preferred_df.groupby("long_trace_id")
      non_sgf_df = non_sgf_df.groupby("long_trace_id")

    mean_all = getattr(all_df.MeanActivity, self._method)()
    mean_pref = getattr(preferred_df.MeanActivity, self._method)()
    mean_non_pref =  getattr(non_preferred_df.MeanActivity, self._method)()
    mean_non_sgf = getattr(non_sgf_df.MeanActivity, self._method)()
    # Calculate SEM
    if self._group_neurons_trials:
      mean_all,      sem_all = mean_all.mean(), mean_all.sem() # mean of sum or mean of mean
      mean_pref,     sem_pref = mean_pref.mean(), mean_pref.sem()
      mean_non_pref, sem_non_pref = mean_non_pref.mean(), mean_non_pref.sem()
      mean_non_sgf,  sem_non_sgf= mean_non_sgf.mean(), mean_non_sgf.sem()
    else:
      sem_all = stats.sem(all_df.MeanActivity) if self._method == "mean" else None
      sem_pref = stats.sem(preferred_df.MeanActivity) if self._method == "mean" else None
      sem_non_pref = stats.sem(non_preferred_df.MeanActivity) if self._method == "mean" else None
      sem_non_sgf = stats.sem(non_sgf_df.MeanActivity) if self._method == "mean" else None
    # print("Choice dir:", choice_dir, "quantile_idx:", quantile_idx, "br:", br, "- len:", len(choice_dir_df))
    # print("preferred_dir_df:", len(preferred_dir_df), "mean:", mean_pref)
    # print("non_preferred_dir_df:", len(non_preferred_dir_df), "mean:", mean_non_pref)
    # print("non_sgf_dir_df:", len(non_sgf_dir_df), "mean:", mean_non_sgf)
    # print()
    print_labels_once = self._already_printed_count == 0 #and choice_dir
    width = 1
    x = (self._already_printed_count+1)*self._stride
    # if not self._combine_sides:
    #   x +=  choice_dir*width
    if self._plot_all:
      ax.bar(x - 2, mean_all, yerr=sem_all,           color="w",
            label=f"All{dir_str} (#N={trace_count_all:,})" if print_labels_once else None, edgecolor="black",
            width=width)
    ax.bar(x - 1, mean_pref, yerr=sem_pref,         color="green",
          label=f"Preferred{dir_str} (#N={trace_count_preferred:,})" if print_labels_once else None, edgecolor="black",
          width=width)
    ax.bar(x,     mean_non_pref, yerr=sem_non_pref, color="orange",
          label=f"Non-Preferred{dir_str} (#N={trace_count_non_preferred:,})" if print_labels_once else None, edgecolor="black",
          width=width)
    if self._plot_non_sgf:
      ax.bar(x + 1, mean_non_sgf, yerr=sem_non_sgf,   color="gray",
            label=f"Non-Sgf{dir_str} (#N={trace_count_non_sgf:,})" if print_labels_once else None, edgecolor="black",
            width=width)
    self._already_printed_count += 1
    self._x_ticks.append(self._already_printed_count*self._stride)
    self._x_ticks_labels.append(f"Q={quantile_idx}")

  def showAndSave(self, fig_save_prefix, save_fig):
    method_str = self._method[0].upper() + self._method[1:]
    self._fig.suptitle(f"{method_str} of Mean Activity by "
                      f"Quantile and Choice Preference {self._label}")
    save_dscrp = "separate_regions" if self._combine_brain_regions else "combined_regions"
    if save_fig:
      sep_side_str = "sep_side" if self._sep_side else "comb_side"
      trials_group_str = "trials_grouped" if self._group_neurons_trials else "without_trials_grouping"
      save_fp = (f"{fig_save_prefix}/RT_Stats/bias_{self._label}_{method_str}"
                  f"_of_mean_activity_quantiles_{save_dscrp}_{sep_side_str}_"
                  f"{trials_group_str}.pdf")
      save_path = pathlib.Path(save_fp)
      save_path.parent.mkdir(exist_ok=True)
      plt.savefig(save_fp)
    plt.show()


class BallTuningPlot(MeanTuningPlot):
  def __init__(self, active_df, method: Literal['mean', 'sum'], combine_sides,
               combine_brain_regions, label, group_neurons_trials=False):
    super().__init__(method, combine_sides, combine_brain_regions, label,
                     plot_all=False, plot_non_sgf=False,
                     group_neurons_trials=group_neurons_trials)
    self._active_df = active_df

  def plot(self, choice_dir, all_df, preferred_df, non_preferred_df, non_sgf_df,
           trace_count_all, trace_count_preferred, trace_count_non_preferred, trace_count_non_sgf,
           quantile_idx):

    ax = self._axs[self._row_idx]
    dir_str = (" Left" if choice_dir else " Right") if not self._combine_sides\
              else ""
    if dir_str != "":
      assert self._combine_sides
      ax = ax[0] if dir_str == " Left" else ax[1]
    if self._group_neurons_trials:
      all_df = all_df.groupby("long_trace_id")
      preferred_df = preferred_df.groupby("long_trace_id")
      non_preferred_df = non_preferred_df.groupby("long_trace_id")
      non_sgf_df = non_sgf_df.groupby("long_trace_id")

    mean_all = getattr(all_df.MeanActivity, self._method)()
    mean_pref = getattr(preferred_df.MeanActivity, self._method)()
    mean_non_pref =  getattr(non_preferred_df.MeanActivity, self._method)()
    mean_non_sgf = getattr(non_sgf_df.MeanActivity, self._method)()
    # Calculate SEM
    if self._group_neurons_trials:
      mean_all,      sem_all = mean_all.mean(), mean_all.sem() # mean of sum or mean of mean
      mean_pref,     sem_pref = mean_pref.mean(), mean_pref.sem()
      mean_non_pref, sem_non_pref = mean_non_pref.mean(), mean_non_pref.sem()
      mean_non_sgf,  sem_non_sgf= mean_non_sgf.mean(), mean_non_sgf.sem()
    else:
      sem_all = stats.sem(all_df.MeanActivity) if self._method == "mean" else None
      sem_pref = stats.sem(preferred_df.MeanActivity) if self._method == "mean" else None
      sem_non_pref = stats.sem(non_preferred_df.MeanActivity) if self._method == "mean" else None
      sem_non_sgf = stats.sem(non_sgf_df.MeanActivity) if self._method == "mean" else None

    ball_size = trace_count_preferred / trace_count_all
    SCALE_FACTOR = 10
    ball_size *= SCALE_FACTOR
    print("ball_size:", ball_size)
    print_labels_once = self._already_printed_count == 0 #and choice_dir

    fill_prcnt = mean_pref / (mean_pref + mean_non_pref)
    print("Fill percent:", fill_prcnt)


    def arcPatch(center, radius, theta1, theta2, ax, resolution=50, **kwargs):
        # make sure ax is not empty
        # generate the points
        theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
        points = np.vstack((radius*np.cos(theta) + center[0],
                            radius*np.sin(theta) + center[1]))
        # build the polygon and add it to the axes
        poly = mpatches.Polygon(points.T, closed=True, **kwargs)
        ax.add_patch(poly)
        return poly

    def getTheta(fill_prcnt):
        fill_prcnt_theta = fill_prcnt*360
        # print("fill_prcnt_theta:", fill_prcnt_theta)
        start_theta = 0
        end_theta = fill_prcnt_theta
        while True:
            x_start = np.cos(np.radians(start_theta)) #y = #, radius*np.sin(angle)
            x_end = np.cos(np.radians(end_theta))
            if abs(x_start - x_end) < 0.001:
                break
            start_theta = (start_theta + 0.001)%360
            end_theta = (end_theta + 0.001)%360
        # print(x_start, x_end, start_theta, end_theta)
        return  start_theta, end_theta
    start_theta, end_theta = getTheta(fill_prcnt)
    # print(part_theta, end_theta, fill_prcnt_theta)

    center = self._already_printed_count + 0.5, 0.5
    USE_ONE_BALL = False
    if USE_ONE_BALL:
      arcPatch(center, ball_size, theta1=start_theta, theta2=end_theta,
              ax=ax, fill=True, color='g')
      start_theta, end_theta = getTheta(1-fill_prcnt)
      arcPatch(center, ball_size,  theta1=start_theta+180, theta2=end_theta+180,
              ax=ax, fill=True, color='orange')
    else:
      def scalePerf(perf):
        new_perf = perf/250
        print("Perf", perf, new_perf)
        return new_perf
      p_center = center[0] + 0.15, center[1]
      n_center = center[0] - 0.15, center[1]
      ax.add_patch(mpatches.Circle(p_center, radius=scalePerf(mean_pref), color="g"))
      ax.add_patch(mpatches.Circle(n_center, radius=scalePerf(mean_non_pref), color="orange"))

    ax.set_xlim(0, self._already_printed_count+1)

    # ax.pie(x + 1, mean_pref, yerr=sem_pref,         color="green",
    #       label=f"Preferred{dir_str} (#N={trace_count_preferred:,})" if print_labels_once else None, edgecolor="black",
    #       width=width)
    # ax.bar(x - 1,     mean_non_pref, yerr=sem_non_pref, color="orange",
    #       label=f"Non-Preferred{dir_str} (#N={trace_count_non_preferred:,})" if print_labels_once else None, edgecolor="black",
    #       width=width)

    self._x_ticks.append(self._already_printed_count+0.5)
    self._x_ticks_labels.append(f"Q={quantile_idx}")
    self._already_printed_count += 1


  def showAndSave(self, fig_save_prefix, save_fig):
    method_str = self._method[0].upper() + self._method[1:]
    self._fig.suptitle(f"{self._label}")
    save_dscrp = "separate_regions" if self._combine_brain_regions else "combined_regions"
    if save_fig:
      trials_group_str = "trials_grouped" if self._group_neurons_trials else "without_trials_grouping"
      plt.savefig(f"{fig_save_prefix}/RT_Stats/two_balls_{self._label}_{method_str}"
                  f"_of_mean_activity_quantiles_{save_dscrp}_"
                  f"{trials_group_str}.pdf")
    plt.show()




def samplingSlowVsFast(src_activitiy_df, stats_df,
                       tuning_plotter : BaseTuningPlot, left_tune_col, sep_side,
                       sep_brain_region : bool,  pval, save_fig,
                       fig_save_prefix=None, skip_qidx2=False,
                       data_col="ChoiceLeft", prior_data_col=None,
                       iter_col="quantile_idx", stats_col_val=None,
                       sep_2_groups_col="ChoiceLeft"):
  '''
  from . import
  src_activitiy_df = expanAlignDF(df_rt_sampling_fixed.copy())
  '''
  if save_fig:
    assert fig_save_prefix is not None
  if "DVstr" in stats_df.columns:
    stats_df = stats_df[stats_df.DVstr.isnull()]

  stats_df = stats_df.copy()
  src_activitiy_df = src_activitiy_df[src_activitiy_df.Layer == "L23"]
  src_activitiy_df = src_activitiy_df[src_activitiy_df.BrainRegion.isin(
                                       [BrainRegion.M2_Bi, BrainRegion.ALM_Bi])]
  # src_activitiy_df = src_activitiy_df.copy() # Will be copied again later
  if "long_trace_id" not in src_activitiy_df.columns:
    src_activitiy_df["long_trace_id"] = src_activitiy_df.ShortName + "_" + \
                                        src_activitiy_df.trace_id.astype(str)
  # src_activitiy_df = _setSrcDfTuning(src_activitiy_df, stats_df, left_tune_col,
  #                                    pval)]
  if iter_col is not None:
    iter_vals = sorted(src_activitiy_df[iter_col].unique())
    trials_w_tuning_by_part = {part_val:_getSgfNeurons(
                              src_col=iter_col, src_col_val=part_val,
                              stats_df=stats_df,
                              src_activitiy_df=src_activitiy_df,
                              left_tune_col=left_tune_col,
                              pval=pval,
                              data_col=data_col, prior_data_col=prior_data_col,
                              stats_col_val=stats_col_val)
                              for part_val in iter_vals}
  else:
    trials_w_tuning_by_part = {"All":_getSgfNeurons(
                              src_col=None, src_col_val=None,
                              stats_df=stats_df,
                              src_activitiy_df=src_activitiy_df,
                              left_tune_col=left_tune_col,
                              pval=pval,
                              data_col=data_col, prior_data_col=prior_data_col,
                              stats_col_val=stats_col_val)}
  del stats_df # Don't use again by mistake

  def plotFnWrap(cur_trials_w_tuning_by_part):
    for part, part_df in cur_trials_w_tuning_by_part.items():
      if skip_qidx2 and part == 2:
        continue
      _processSamplingSlowVsFast(part_df, sep_side=sep_side,
                                 tuning_plotter=tuning_plotter,
                                 quantile_idx=part,
                                 sep_2_groups_col=sep_2_groups_col)
  if sep_brain_region:
    for br, _ in src_activitiy_df.groupby("BrainRegion"):
      br_dscrp_str = str(BrainRegion(br)).split("_")[0]
      tuning_plotter.startNewBrainRegion(br_dscrp_str)
      br_quantiles_neurons = {part:part_df[part_df.BrainRegion == br]
                           for part, part_df in trials_w_tuning_by_part.items()}
      plotFnWrap(br_quantiles_neurons)
      tuning_plotter.finishBrainRegion()
  else:
    br_dscrp_str = "Brain-Regions Combined"
    tuning_plotter.startNewBrainRegion(br_dscrp_str)
    plotFnWrap(trials_w_tuning_by_part)
    tuning_plotter.finishBrainRegion()

  tuning_plotter.showAndSave(fig_save_prefix, save_fig)


def _processSamplingSlowVsFast(df, sep_side, tuning_plotter : BaseTuningPlot,
                               quantile_idx, sep_2_groups_col="ChoiceLeft"):
  if not sep_side:
    all_li, preferred_li, non_preferred_li, non_sgf_li = [], [], [], []
    trace_count_all_li, trace_count_preferred_li, trace_count_non_preferred_li, trace_count_non_sgf_li = [], [], [], []
  # Reverse so that left would show first before right
  uniq_iter_vals = sorted(df[sep_2_groups_col].dropna().unique())[::-1]
  assert len(uniq_iter_vals) == 2
  for choice_dir, non_choice_dir, choice_dir_df in [
                    (uniq_iter_vals[0], uniq_iter_vals[1], df[df[sep_2_groups_col] == uniq_iter_vals[0]]),
                    (uniq_iter_vals[1], uniq_iter_vals[0], df[df[sep_2_groups_col] == uniq_iter_vals[1]])]:
                                  #df.groupby(sep_2_groups_col):
    # trials_sgf_df = choice_dir_df[choice_dir_df.long_trace_id.isin(sgf_choice.long_trace_id)]
    # TODO: Split by ShortName
    #non_choice_dir = (choice_dir + 1) % 2
    # display(df)
    # display(choice_dir_df)
    preferred_dir_df =     choice_dir_df[choice_dir_df.is_left_choice_tuned == choice_dir]
    print("Len preferred_dir_df:", len(preferred_dir_df), "for choice_dir:", choice_dir, "For sour split:", sep_2_groups_col,
          "uniq traces:", preferred_dir_df.long_trace_id.nunique())
    non_preferred_dir_df = choice_dir_df[choice_dir_df.is_left_choice_tuned == non_choice_dir]
    non_sgf_dir_df =       choice_dir_df[choice_dir_df.is_left_choice_tuned.isnull()]
    trace_count_all           = choice_dir_df.long_trace_id.nunique()
    trace_count_preferred     = preferred_dir_df.long_trace_id.nunique()
    trace_count_non_preferred = non_preferred_dir_df.long_trace_id.nunique()
    trace_count_non_sgf       = non_sgf_dir_df.long_trace_id.nunique()
    if sep_side:
      tuning_plotter.plot(choice_dir, choice_dir_df,   preferred_dir_df,      non_preferred_dir_df,      non_sgf_dir_df,
                          trace_count_all, trace_count_preferred,             trace_count_non_preferred, trace_count_non_sgf,
                          quantile_idx=quantile_idx)
    else:
      all_li.append(choice_dir_df)
      preferred_li.append(preferred_dir_df)
      non_preferred_li.append(non_preferred_dir_df)
      non_sgf_li.append(non_sgf_dir_df)
      # Track counts as well
      trace_count_all_li.append(trace_count_all)
      trace_count_preferred_li.append(trace_count_preferred)
      trace_count_non_preferred_li.append(trace_count_non_preferred)
      trace_count_non_sgf_li.append(trace_count_non_sgf)
  if not sep_side:
    # Converts counts to np arrays and take the mean
    trace_count_all = np.array(trace_count_all_li).mean()
    trace_count_preferred = np.array(trace_count_preferred_li).mean()
    trace_count_non_preferred = np.array(trace_count_non_preferred_li).mean()
    trace_count_non_sgf = np.array(trace_count_non_sgf_li).mean()
    all_df = pd.concat(all_li)
    preferred_df = pd.concat(preferred_li)
    non_preferred_df = pd.concat(non_preferred_li)
    non_sgf_df = pd.concat(non_sgf_li)
    tuning_plotter.plot(None, all_df,          preferred_df,          non_preferred_df,          non_sgf_df,
                              trace_count_all, trace_count_preferred, trace_count_non_preferred, trace_count_non_sgf,
                        quantile_idx=quantile_idx)
