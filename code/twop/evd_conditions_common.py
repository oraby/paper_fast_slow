from ..pipeline import (
    pipeline,
    tracesmath,
    tracesfilter,
    tracesrestructure,
    traceswidth,
    tracesnormalize,
    tracessort,
    plotters,
    utils,
)
from . import behaviorcommon
from ..common._imaging import PlotTraces
import numpy as np
import pandas as pd
from enum import IntFlag, auto
from functools import partial
from itertools import chain as iterChain

class SplitLevelBy(IntFlag):
  AllAnimals = 0
  Session = auto()
  Animal = auto()
  BrainRegion = auto()
  SamplingType = auto()
  StimulusType = auto()
  CheckPointName = auto()
  SessCountFromCheckPoint = auto()
  DurationBin = auto()
  Quantile = auto()
  Direction = auto()


def _splitLevelByCols(split_level):
  by = []
  def handle(split_val, cols):
    nonlocal by, split_level
    if split_level & split_val:
       by += cols
       split_level ^= split_val
  # No need to do anything if it's AllAnimals
  handle(SplitLevelBy.Session, pipeline.BySession()._col_or_li_of_cols)
  handle(SplitLevelBy.Animal, pipeline.ByAnimal()._col_or_li_of_cols)
  handle(SplitLevelBy.BrainRegion, ["BrainRegion", "Layer"])
  handle(SplitLevelBy.SamplingType, ["SamplingType"])
  handle(SplitLevelBy.StimulusType, ["Stimulus"])
  handle(SplitLevelBy.CheckPointName, ["CheckPointName"])
  handle(SplitLevelBy.SessCountFromCheckPoint, ["SessCountFromCheckPoint"])
  handle(SplitLevelBy.DurationBin, ["dur_bin"])
  handle(SplitLevelBy.Quantile, ["quantile_idx"])
  handle(SplitLevelBy.Direction, ["ChoiceLeft"])
  if split_level:
    raise NotImplemented(f"Unhandled split level(s) {split_level}")
  return by

def splitLevel(split_level: SplitLevelBy):
  assert isinstance(split_level, SplitLevelBy), "Update your code"
  assert not ((split_level & SplitLevelBy.Session) and
             ((split_level & SplitLevelBy.Animal))), (
    "Only split by session or animal")
  by = _splitLevelByCols(split_level)
  if len(by):
    return pipeline.By(by)
  else:
    return pipeline.DoNothingPipe()

def commonTraceNormalizationInit(df, normalization,
                                 rename_feedback_epoch : bool):
  df = df.copy()
  chain = pipeline.Chain(
    pipeline.BySession(),
      pipeline.ApplyFullTrace(normalization),
    pipeline.RecombineResults(),
  )
  df = chain.run(df)
  if rename_feedback_epoch:
    def addHasFeedbackWait(sess_df):
      sess_df["HasFeedbackWait"] = any(sess_df.epoch == "Feedback Wait")
      return sess_df
    df = df.groupby("anlys_path", group_keys=False).apply(addHasFeedbackWait)
    df.loc[df.HasFeedbackWait  & (df.epoch == "Feedback Wait"),           "epoch"] = "Feedback Start"
    df.loc[~df.HasFeedbackWait & df.epoch.isin(["Reward", "Punishment"]), "epoch"] = "Feedback Start"
  return df

def normalizeTraces(df, normalization, split_level : SplitLevelBy,
                    align_feedback_time=None, limit_end_epoch=False):
  assert "epoch_time" in df.columns
  rename_feedback_epoch = align_feedback_time is not None and align_feedback_time > 0
  df = commonTraceNormalizationInit(df, normalization, rename_feedback_epoch)
  # This epoch only exists if the feedback epochs were renamed
  feedback_start_epoch = "Feedback Start"
  if rename_feedback_epoch:
    if not limit_end_epoch:
      normalize_feedback = []
      limit_to_epoch_end = []
    else:
      normalize_feedback = [traceswidth.NormalizeTime()]
      limit_to_epoch_end = ["Feedback Start"]
      print("Removing feedback epoch in sessions with feedback wait")
      def concatTrialFeedbackEnd(trial_df):
        if not trial_df.HasFeedbackWait.iloc[0] or \
           not any(trial_df.epoch == "Feedback Wait"):
          return trial_df
        feedback_idx = trial_df.epoch.isin(["Reward", "Punishment"])
        if not feedback_idx.any():
          return trial_df
        assert feedback_idx.sum() == 1, "Expected 1 feedback epoch"
        feedback_row = trial_df[feedback_idx].iloc[0]
        feedback_start_row_idx = trial_df.epoch == "Feedback Start"
        assert feedback_start_row_idx.sum() == 1, "Expected 1 feedback start epoch"
        assert trial_df[feedback_start_row_idx].trace_start_idx.iloc[0] <= \
               feedback_row.trace_start_idx
        trial_df.loc[feedback_start_row_idx,
                     "trace_end_idx"] = feedback_row.trace_end_idx
        # Remove the extra epoch
        trial_df = trial_df[~feedback_idx]
        # Update epoch time
        feedback_start_row = trial_df[feedback_start_row_idx].iloc[0]
        # Assign through .loc so trial_df itself is updated
        trial_df.loc[feedback_start_row_idx, "epoch_time"] = (
           (feedback_start_row.trace_end_idx -
            feedback_start_row.trace_start_idx + 1)/
           feedback_start_row.acq_sampling_rate)
        return trial_df
      df = df.groupby([df.ShortName,  df.TrialNumber],
                      group_keys=False).apply(concatTrialFeedbackEnd)
      print("Limiting end epoch")
    align_feedback_processors = [
      pipeline.ApplyOnSubdata("epoch",
                              lambda epoch: epoch == "Feedback Start",
          tracesrestructure.AlignTraceAroundEpoch(
              epoch_name_li=["Feedback Start"], time_before_sec=0,
              time_after_sec=align_feedback_time,
              limit_to_epoch_end=limit_to_epoch_end),
          *normalize_feedback,
          only_full_df=False,
        )
    ]
  else:
    align_feedback_processors = []
  chain = pipeline.Chain(
    splitLevel(split_level),
    # *align_feedback_processors,  # For quick testing
      pipeline.ApplyOnSubdata("epoch",
                              lambda epoch: epoch != feedback_start_epoch,
                              pipeline.By("epoch"),
                                traceswidth.NormalizeTime(),
                              only_full_df=False,),
      *align_feedback_processors,
    pipeline.RecombineResults(),
  )
  df = chain.run(df)
  return df

def avgEpochs(df,
              restrict_filtering_to_epochs=["Sampling", "Decision Time",
                                            "Movement to Lateral Port",],
              plot_traces=False, plot_only_traces_ids=[], savePrefix=None, *,
              # Must specify the following paramters
              by_split_level: SplitLevelBy = None, by_epoch: bool = False,
              filter_outlier_trials: bool = None,
              rename_traces=None, **by_processors,):
  df = df.copy()
  gen_by_processors = GenByProcessor(by_epoch=by_epoch, **by_processors)
  if filter_outlier_trials:
    filter_processor = tracesfilter.FilterOutlierTraces(
                              epochs=restrict_filtering_to_epochs,
                              filter=tracesfilter.FilterOutliersStd(num_stds=3),
                              plot=plot_traces,
                              plot_only_traces_ids=plot_only_traces_ids,
                              savePrefix=savePrefix,
                              formatRow=gen_by_processors.formatRow,)
  else:
    filter_processor = pipeline.DoNothingPipe()
  if rename_traces == True or (rename_traces is None and
                               by_split_level != SplitLevelBy.Session):
    rename_traces = [utils.RenameTraces("ShortName")]
  else:
    rename_traces = []
  chain = pipeline.Chain(
      *rename_traces,
      splitLevel(by_split_level),
      *gen_by_processors.get(),
        filter_processor,
        tracesmath.TraceAvg(avg_rows=True, avg_row_traces=False,
                            assigned_df_col_avg_row_name="TrialNumber",
                            get_name_df_cols=gen_by_processors.formatRow),
      pipeline.RecombineResults(),
  )
  return chain.run(df)


def plotHeatMap(df, by_split_level : SplitLevelBy, save_figs, savePrefix=None,
                heatMapSortSrcDf=None,  only_common_sort_not_common_rng=False,
                title_cols=[], restrict_to_epochs=[], only_traces_ids=[],
                restrict_sorting_to_epochs=[], smooth_sigma=2,
                plot_kargs={}, **by_processors,):
  gen_by_processors = GenByProcessor(**by_processors)
  def _restrictToEpochs(_df):
    if len(restrict_to_epochs):
      chain = pipeline.Chain(tracesrestructure.KeepOnlyEpochsInConcatEpochs(
                             epoch_name_li=restrict_to_epochs))
      _df = chain.run(_df)
    return _df
  def splitDf(_df, use_gen_by_processors, use_split_level=by_split_level):
    gen_li = gen_by_processors.get() if use_gen_by_processors else []
    return pipeline.Chain(
        splitLevel(use_split_level),
        *gen_li,
    ).run(_df)
  def normalizeFiringRng(_df):
    return pipeline.Chain(
        tracesnormalize.NormalizePercentile(0, 100,
                            set_name="neuronal",
                          )
    ).run(_df)
  def sortDf(_df, used_smooth_sigma=smooth_sigma):
    return pipeline.Chain(
        tracessort.ByPeakActivityTime(smooth_sigma=used_smooth_sigma,
                                 restrict_to_epochs=restrict_sorting_to_epochs),
    ).run(_df)
  print("Splitting....")
  print("gen_by_processors:", gen_by_processors._by_kws)
  df = _restrictToEpochs(df)
  df = splitDf(df, use_gen_by_processors=True)
  if heatMapSortSrcDf is None:
    df = normalizeFiringRng(df)
    df = sortDf(df)
  else:
    match_cols = _splitLevelByCols(by_split_level)
    def matchTrialsFn(row, src_df):
      matchcing = src_df[match_cols[0]] == row[match_cols[0]]
      for col in match_cols[1:]:
        matchcing &= src_df[col] == row[col]
      if matchcing.sum() != 1:
        display(row[match_cols])
        display(src_df[match_cols])
      assert matchcing.sum() == 1, f"Expected 1 row, found {matchcing.sum()}"
      src_row = src_df[matchcing]
      return src_row.iloc[0]
    sort_traces_src_df, match_cols, sort_split_level, is_one_of_them = \
                heatMapSortSrcDf(pd.concat(df) if isinstance(df, list) else df,
                                 by_split_level,
                                 gen_by_processors._by_kws, match_cols)
    sort_traces_src_df = _restrictToEpochs(sort_traces_src_df)
    sort_traces_src_df = splitDf(sort_traces_src_df,
                                 use_split_level=sort_split_level,
                                 use_gen_by_processors=is_one_of_them)
    # Sort the source first to the desired order
    sort_traces_src_df = sortDf(sort_traces_src_df,
                                used_smooth_sigma=smooth_sigma)
    sort_traces_src_df = pipeline.Chain(pipeline.RecombineResults()).run(
                                                             sort_traces_src_df)
    print("Only ids:", only_traces_ids)
    print("Matching cols in sort traces src")
    print("Continuing with normal:")
    # Then copy the order to the destination
    df = pipeline.Chain(
        pipeline.RecombineResults(),
        utils.CopyTracesSortOrderProcessor(src_df=sort_traces_src_df,
                                           set_name="neuronal",
                                           matchTrialsFn=matchTrialsFn,
                                           match_sess_cols=match_cols,
                                           only_traces_ids=[]#only_traces_ids
                                           )
    ).run(df)
    if not only_common_sort_not_common_rng:
      df = tracesnormalize.normalizeMinMaxPerTrace(df,
                                                   match_rows_col_name="ShortName",
                                                   set_name="neuronal",
                                                   smooth_sigma=smooth_sigma)
      df = splitDf(df, use_gen_by_processors=True)
    else:
      df = splitDf(df, use_gen_by_processors=True)
      df = normalizeFiringRng(df.copy())
      df = pipeline.Chain(
        tracesmath.GaussianFilter(sigma=smooth_sigma, set_name="neuronal",
                                  copy_data=True),
      ).run(df)

  chain = pipeline.Chain(
    plotters.TracesHeatMap(is_avg_trc=True, title_cols=title_cols,
                           title_prefix="", title_cols_names=False,
                           write_title_plot_id=False,
                           save_prefix=savePrefix if save_figs else None,
                           heatmap_min=0, heatmap_max=1,
                           only_traces_ids=only_traces_ids,
                           **plot_kargs),
  )
  chain.run(df)

def plotHeatMapDiff(df, by_split_level:SplitLevelBy, save_figs, diff_split_kawrg,
                    use_abs=False,  savePrefix=None, sort_traces_src_df=None,
                    sort_src_df_match_col_name="ShortName",
                    title_cols=[], restrict_to_epochs=[],
                    restrict_sorting_to_epochs=[], smooth_sigma=2,
                    **by_processors,):
  def _restrictToEpochs(_df):
    if len(restrict_to_epochs):
      chain = pipeline.Chain(
        tracesrestructure.KeepOnlyEpochsInConcatEpochs(
                                              epoch_name_li=restrict_to_epochs)
      )
      _df = chain.run(_df)
    return _df
  def splitDf(_df, without_kargs=[]):
    mod_by_processors = {k: v for k, v in by_processors.items()
                         if k not in without_kargs}
    chain = pipeline.Chain(
      splitLevel(by_split_level),
      *GenByProcessor(**mod_by_processors).get(),
    )
    return chain.run(_df)
  def normalizeAndSort(_df):
    chain = pipeline.Chain(
      tracesnormalize.NormalizePercentile(0,100,
                          ),
      tracessort.ByPeakActivityTime(smooth_sigma=smooth_sigma,
                                 restrict_to_epochs=restrict_sorting_to_epochs),
    )
    return chain.run(_df)

  assert diff_split_kawrg in by_processors

  df = _restrictToEpochs(df)
  df = splitDf(df, without_kargs=[diff_split_kawrg])

  split_col_name = GenByProcessor.kw_mapping[diff_split_kawrg]
  chain = pipeline.Chain(
    tracesmath.DiffTraces(col_name=split_col_name, left_side=True,
                          right_side=False, use_abs=use_abs),

    tracessort.ByPeakActivityTime(smooth_sigma=smooth_sigma,
                                  restrict_to_epochs=restrict_sorting_to_epochs),
    plotters.TracesHeatMap(is_avg_trc=True, title_cols=title_cols,
                           title_prefix="", title_cols_names=False,
                           write_title_plot_id=False,
                           save_prefix=savePrefix if save_figs else None,
                           color_map="PiYG" if not use_abs else "Reds",
                           heatmap_min=-1 if not use_abs else 0,
                           heatmap_max=1,
                           ),
  )
  chain.run(df)

class GenByProcessor:
  kw_mapping = {"by_epoch": "epoch",
                "by_correctness": "ChoiceCorrect",
                "by_prev_correct": "PrevChoiceCorrect",
                "by_direction": "ChoiceLeft",
                "by_prev_direction": "PrevChoiceLeft",
                "by_difficulty": "DVstr",
                "by_duration": "dur_bin",
                "by_quantile": "quantile_idx"}

  def __init__(self, by=[],  **by_kwargs,):
    self._by = by
    self._by_kws = []
    for kw, val in by_kwargs.items():
      split_str = GenByProcessor.kw_mapping.get(kw)
      assert split_str is not None, ( f"Unknown Split Processor: {kw}, "
          f"known processors: {GenByProcessor.kw_mapping.keys()}")
      if val == True:
        self._by_kws.append(split_str)

  def get(self):
    by_processors = []
    for kw in self._by_kws:
      by_processors.append(pipeline.By(kw))
    if self._by is not None and (not hasattr(self._by, "__len__") or len(self._by)):
      by_processors.append(pipeline.By(self._by))
    return by_processors

  def formatRow(self, row):
    str_li = []
    def asrtAdd(val, _str):
      nonlocal str_li
      assert val is None or np.isnan(val)
      str_li.append(_str)

    if "epoch" in self._by_kws:  # self._by_epoch:
      str_li.append(f"_epoch_{row.epoch}")
    if "PrevChoiceCorrect" in self._by_kws:  # self._by_prev_correct:
      if row.PrevChoiceCorrect == True:
        str_li.append("_PrevCorrectChoice")
      elif row.PrevChoiceCorrect == False:
        str_li.append("_PrevIncorrectChoice")
      else:
        asrtAdd(row.PrevChoiceCorrect, "_NoPrevChoice")
    if "ChoiceCorrect" in self._by_kws:  # self._by_correctness:
      if row.ChoiceCorrect == True:
        str_li.append("_CorrectChoice")
      elif row.ChoiceCorrect == False:
        str_li.append("_IncorrectChoice")
      else:
        asrtAdd(row.ChoiceCorrect, "_NoChoice")
    if "DVstr" in self._by_kws:  # self._by_difficulty:
      str_li.append(f"_{row.DVstr}")
    if "PrevChoiceLeft" in self._by_kws:  # self._by_prev_direction:
      if row.PrevChoiceLeft == True:
        str_li.append("_PrevDecisionLeft")
      elif row.PrevChoiceLeft == False:
        str_li.append("_PrevDecisionRight")
      else:
        asrtAdd(row.PrevChoiceLeft, "_PrevDecisionNone")
    if "ChoiceLeft" in self._by_kws:  # self._by_direction:
      if row.ChoiceLeft == True:
        str_li.append("_DecisionLeft")
      elif row.ChoiceLeft == False:
        str_li.append("_DecisionRight")
      else:
        asrtAdd(row.ChoiceLeft, "_DecisionNone")
    if "dur_bin" in self._by_kws:  # self._by_duration
      str_li.append(f"_Dur_{row.dur_bin.left:.1f}_{row.dur_bin.right:.1f}")
    if len(self._by):
      for col in self._by:
        val = row[col]
        _str = f"{val:.2f}" if isinstance(val, float) else str(val)
        str_li.append(f"_{col}_{_str}")
    return "".join(str_li)[1:]


def _alignAroundEpoch(df, align_to_epoch, time_before, time_after,
                      normalization, limit_to_epoch_start=[],
                      limit_to_epoch_end=[],
                      limit_trial_start=False, limit_trial_end=False):
  if normalization is not None:
    normalization_li = [pipeline.BySession(),
                         pipeline.ApplyFullTrace(normalization),
                        pipeline.RecombineResults()
    ]
  else:
    normalization_li = []
  df = df.copy()
  assert all(df.sole_owner == False), "all rows traces should be full trace"
  chain = pipeline.Chain(
    *normalization_li,
      tracesrestructure.AlignTraceAroundEpoch(epoch_name_li=[align_to_epoch],
                                              time_before_sec=time_before,
                                              time_after_sec=time_after,
                                      limit_to_epoch_start=limit_to_epoch_start,
                                      limit_to_epoch_end=limit_to_epoch_end,
                                      limit_trial_start=limit_trial_start,
                                      limit_trial_end=limit_trial_end),
    pipeline.RecombineResults(),
  )
  df = chain.run(df)
  return df


# Common plotting

def plotAreas(df_side, df_unified, split_level, use_unified, areas_to_colors,
              loopGroupTracesFn, savePrefixFn, only_traces_ids=[],
              trial_number_as_plot_id=True, multiple_trace_ids_per_plot=False,
              set_name="neuronal",
              **plot_traces_kargs
              ):
  if callable(areas_to_colors):
    local_areas_colors = areas_to_colors
  else:
    local_areas_colors = areas_to_colors.copy()
  df = df_unified if use_unified else df_side
  df = df.copy()
  if not use_unified:
    only_traces_ids = list(
          iterChain(*[[f"{el}_left", f"{el}_right"] for el in only_traces_ids]))
  print("only_traces_ids:", only_traces_ids)
  print("local_areas_colors:", local_areas_colors)
  print("multiple_trace_ids_per_plot:", multiple_trace_ids_per_plot)
  plotter_only_traces_ids = only_traces_ids if multiple_trace_ids_per_plot else\
                            []
  print("plotter only traces ids:", plotter_only_traces_ids)
  traces_plotter = PlotTraces(is_avg_trc=not trial_number_as_plot_id,
                              x_label="Time", y_label="Activity",
                              areas_to_colors=local_areas_colors,
                              only_traces_ids=plotter_only_traces_ids,
                              save_prefix=savePrefixFn,
                              set_name=set_name,
                              **plot_traces_kargs)
  if not multiple_trace_ids_per_plot:
    traces_wrap = pipeline.LoopTraces(traces_plotter, # vid_gen,
                                      set_name=set_name,
                                      groupTracesFn=loopGroupTracesFn,
                                      only_traces_ids=only_traces_ids,)
  else:
      traces_wrap = traces_plotter
  chain = pipeline.Chain(
      splitLevel(split_level), #pipeline.BySession(),  # pipeline.By("DVstr"),
          # Plot data
          # Save plotted video
          *[traces_wrap],
  )
  chain.run(df)
