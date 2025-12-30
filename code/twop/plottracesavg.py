from .evdanalysisplot import (legendLabelAndLineStyle, AllClrFn, Difficulty_clr,
                              Difficulty_Direction, DirectionClrFn,
                              ChoiceOutcomeClrFn)
from .evd_conditions_common import (avgEpochs, plotAreas, plotHeatMap,
                                    splitLevel, SplitLevelBy)
from . import behaviorcommon
from ..pipeline import pipeline
from ..common.definitions import BrainRegion
from dataclasses import dataclass, fields
from functools import partial
from inspect import signature as sig
from typing import Any, List
from pathlib import Path

def _stripOnedrivePrefix(path_str):
  one_drive_idx = path_str.lower().find("onedrive")
  if one_drive_idx != -1:
    path_onedrive = Path(path_str[one_drive_idx:])
    path = Path(*path_onedrive.parts[1:])
    return str(Path(path))
  return path_str

def updateCacheToPC(cache_df_avg):
  return {_stripOnedrivePrefix(k):v for k, v in cache_df_avg.items()}

def buildCacheKey(cur_fig_save_prefix, save_subdir, split_level,
                  align_feedback_time=None, limit_end_epoch=None,
                  cache_extra_prefix=""):
  if align_feedback_time:
    align_part = f"_align_{align_feedback_time}s"
    align_part = f"{align_part}{'_norm_limited_end' if limit_end_epoch else ''}"
  else:
    align_part = ""
  cur_fig_save_prefix = _stripOnedrivePrefix(cur_fig_save_prefix)
  cache_key = (f"{cache_extra_prefix}"
               f"{cur_fig_save_prefix}_{save_subdir}{align_part}_"
               f"split{int(split_level)}")

  return str(Path(cache_key).as_posix())

@dataclass
class Condition:
  col : str
  val : Any
  operator : str = "=="
  name : str = None

@dataclass
class PlotCombination:
  name : str
  avg_kargs: dict
  plt_clr: Any
  more_cond: Condition | List[Condition] = None

OnlyCorrectTrials = Condition("ChoiceCorrect", True,        name="correct")
WithoutDifficultTrials = Condition("DVstr", "'Hard'", "!=", name="without_difficult")

class Combinations:
  All =                 PlotCombination("All",                                  dict(),                                             AllClrFn,           )
  Evidence =            PlotCombination("Evidence",                             dict(by_difficulty=True),                           Difficulty_clr,      OnlyCorrectTrials)
  EvidenceDirection =   PlotCombination("EvidenceDirection",                    dict(by_difficulty=True, by_direction=True),        Difficulty_Direction,OnlyCorrectTrials)
  Direction =           PlotCombination("Direction",                            dict(by_direction=True),                            DirectionClrFn,      OnlyCorrectTrials)
  PrevDirCurDir =       PlotCombination("PrevDirCurDir",                        dict(by_prev_direction=True, by_direction=True),    DirectionClrFn,      OnlyCorrectTrials)
  PrevDirCurDirAll =    PlotCombination("PrevDirCurDirAll",                     dict(by_prev_direction=True, by_direction=True),    DirectionClrFn,      )
  PrevDirection =       PlotCombination("PrevDirection",                        dict(by_prev_direction=True),                       DirectionClrFn,      OnlyCorrectTrials)
  ChoiceCorrect =       PlotCombination("ChoiceCorrect",                        dict(by_correctness=True),                          ChoiceOutcomeClrFn,  WithoutDifficultTrials)
  ChoiceCorrectCurDir = PlotCombination("ChoiceCorrectCurDir",                  dict(by_correctness=True, by_direction=True),       ChoiceOutcomeClrFn,  WithoutDifficultTrials)
  PrevChoiceCorrect =   PlotCombination("PrevChoiceCorrect",                    dict(by_prev_correct=True),                         ChoiceOutcomeClrFn,  OnlyCorrectTrials)
  PrevChoiceCurDir =    PlotCombination("PrevChoiceCurDir",                     dict(by_prev_correct=True, by_direction=True),      ChoiceOutcomeClrFn,  OnlyCorrectTrials)
  PrevChoiceCurDirAll = PlotCombination("PrevChoiceCurDir",                     dict(by_prev_correct=True, by_direction=True),      ChoiceOutcomeClrFn)
  PrevChoicePrevDir =   PlotCombination("PrevChoicePrevDir",                    dict(by_prev_direction=True,  by_prev_correct=True),ChoiceOutcomeClrFn,  OnlyCorrectTrials)
  PrevChoicePrevDirCurDirLeft =  PlotCombination("PrevChoicePrevDirCurDirLeft", dict(by_prev_direction=True, by_prev_correct=True), ChoiceOutcomeClrFn,  [OnlyCorrectTrials, Condition("ChoiceLeft", False)])
  SamplingTime =        PlotCombination("SamplingTime",                         dict(by_duration=True),                             DirectionClrFn,  )
  Quantile =            PlotCombination("Quantile",                             dict(by_quantile=True),                             DirectionClrFn,  )
  QuantileOutcome =     PlotCombination("QuantileOutcome",                      dict(by_quantile=True, by_correctness=True),        ChoiceOutcomeClrFn)
  QuantileDirection =   PlotCombination("QuantileDirection",                    dict(by_quantile=True, by_direction=True),          DirectionClrFn)
  QuantilePrevDir   =   PlotCombination("QuantilePrevDir",                      dict(by_quantile=True, by_prev_direction=True),     DirectionClrFn)
  QuantileDirectionOutcome = PlotCombination("QuantileDirectionOutcome",        dict(by_quantile=True, by_direction=True, by_correctness=True), DirectionClrFn)

  @classmethod
  def getCombination(conditions=[], **filters):
    if not isinstance(conditions, list):
      conditions = [conditions]
    conditions = sorted(conditions)
    for comb in Combinations.__dict__.values():
      if not isinstance(comb, PlotCombination):
        continue
      comb_cond_li = comb.more_cond
      if not isinstance(comb_cond_li, list):
        comb_cond_li = [comb_cond_li]
      if comb.avg_kargs == filters and sorted(comb_cond_li) == conditions:
        return comb
    raise ValueError(f"Combination not found for conditions: {conditions} and "
                     f"filters: {filters}")

class Plots:
  Heatmap = "heatmap"
  Traces = "traces"
  WideFieldVid = "widefield_vid"

def plotNormalized(df_concat_norm, df_concat_without_feedback, combinations,
                   cached_df_avgs, plots : List[Plots],
                   save_figs, save_figs_outliers, fig_save_prefix,
                   normalization, split_level, limit_end_epoch,
                   filter_outlier_trials, align_feedback_time=None,
                   only_traces_ids=[], alternate_save_dir=None,
                   save_context=None, plots_kargs={},
                   onlyTraceIdsPlotProcessFn=None, cache_extra_prefix="",
                   rename_traces=None, updateDFIfUseedFn=None,
                   filter_query=None):
  # df_concat_norm = df_concat_norm[df_concat_norm.HasFeedbackWait == False]
  # df_concat_norm = df_concat_norm[df_concat_norm.ShortName == "GP4_23_S1_L50_D250_mm2"]
  assert not(len(only_traces_ids) and onlyTraceIdsPlotProcessFn is not None), (
    "Only only_traces_ids or onlyTraceIdsPlotProcessFn shoudl be specified")
  cur_fig_save_prefix_org = fig_save_prefix

  cached_df_avgs_updated = updateCacheToPC(cached_df_avgs)
  cached_df_avgs.clear()
  cached_df_avgs.update(cached_df_avgs_updated)
  del cached_df_avgs_updated
  [print(k) for k in cached_df_avgs.keys()]

  getAvgFn = partial(avgEpochsStub, cached_df_avgs=cached_df_avgs,
                     normalization=normalization,
                     cur_fig_save_prefix=cur_fig_save_prefix_org,
                     split_level=split_level,
                     save_figs_outliers=save_figs_outliers,
                     filter_outlier_trials=filter_outlier_trials,
                     only_traces_ids=only_traces_ids,
                     limit_end_epoch=limit_end_epoch,
                     align_feedback_time=align_feedback_time,
                     df_concat_without_feedback=df_concat_without_feedback,
                     cache_extra_prefix=cache_extra_prefix,
                     save_context=save_context,
                     rename_traces=rename_traces,
                     updateDFIfUseedFn=updateDFIfUseedFn,
                     filter_query=filter_query)

  for comb in combinations:
    save_subdir = comb.name
    print(f"Processing", save_subdir)
    df = df_concat_norm.copy()
    # if save_subdir == "PrevChoiceCorrect":
    # if True:
    #   df = df[df.ShortName == df.ShortName.unique()[0]]
    #   df = df[df.Name == df.Name.unique()[0]]

    # continue
    # if "PrevDirection" in save_subdir:
    #   # print(df_avg_comb.epochs_names.iloc[0])
    #   print(df.PrevChoiceLeft.unique())
    #   df_avg_comb = df_avg_comb[df_avg_comb.epoch_name.index]
    # print("Early bail out")
    # df_avg_comb = _prepareDF(df_avg_comb, save_subdir)
    if alternate_save_dir is not None:
      cur_fig_save_prefix = alternate_save_dir
    else:
      cur_fig_save_prefix = cur_fig_save_prefix_org

    if not plots: # We should just create the averages
      getAvgFn(comb, df=df)
      continue

    common_kargs = dict(comb=comb,
                        df=df,
                        getAvgFn=getAvgFn,
                        normalization=normalization,
                        cur_fig_save_prefix=cur_fig_save_prefix,
                        split_level=split_level,
                        save_figs=save_figs,
                        only_traces_ids=only_traces_ids,
                        save_context=save_context,
                        onlyTraceIdsPlotProcessFn=onlyTraceIdsPlotProcessFn,)

    if Plots.Heatmap in plots:
      heatmapStub(**common_kargs,
                  **plots_kargs.get(Plots.Heatmap, {}))

    if Plots.Traces in plots:
      plotTracesStub(**common_kargs,
                     **plots_kargs.get(Plots.Traces, {}))

    if Plots.WideFieldVid in plots:
      widefieldVidStub(**common_kargs,
                       **plots_kargs.get(Plots.WideFieldVid, {}))

def _getOnlyTracesIds(onlyTraceIdsPlotProcessFn, only_traces_ids, df,
                      split_level):
  if onlyTraceIdsPlotProcessFn is None:
    return only_traces_ids
  only_traces_ids = set()
  split_by = splitLevel(split_level)
  if isinstance(split_by, pipeline.By):
    keys = split_by._col_or_li_of_cols
    if not isinstance(keys, list):
      keys = [keys]
    SINGLE_ITEM = len(keys) == 1
    # print("Grouper:", keys)
    df.reset_index(drop=True, inplace=True)
    groupby = df.groupby(keys)
    grouper = []
    for grp_vals, grp_df in groupby:
      if not isinstance(grp_vals, list) and not isinstance(grp_vals, tuple):
        assert SINGLE_ITEM, f"Not a single item: {grp_vals}"
        grp_vals = [grp_vals]
      set_dscrp = {key:grp_val for key, grp_val in zip(keys, grp_vals)}
      grouper.append((set_dscrp, grp_df))
    # grouper = [({key:grp_val for key, grp_val in zip(keys, grp_vals)}, grp_df)
    #                          for grp_vals, grp_df in grouper]
  else:
    grouper = ({"":"All"}, df),
  for group_info, grp_df in grouper:
    for row_idx, row in grp_df.iterrows():
      row_sets = set()
      for set_name, traces_dict in pipeline.getRowTracesSets(row).items():
        traces_ids_set = onlyTraceIdsPlotProcessFn(set_name, traces_dict.keys(),
                                                   group_info=group_info)
        row_sets.update(traces_ids_set)
      if not len(row_sets):
        row_sets = {"___________________asdasdRandom"}
        df.drop(row_idx, inplace=True)
      only_traces_ids.update(row_sets)
  # print("only_traces_ids:", only_traces_ids)
  return list(only_traces_ids)

def avgEpochsStub(comb, df, cached_df_avgs, normalization,
                  cur_fig_save_prefix, split_level, save_figs_outliers,
                  filter_outlier_trials, only_traces_ids,
                  limit_end_epoch, align_feedback_time=None, save_context=None,
                  df_concat_without_feedback=None, rename_traces=None,
                  cache_extra_prefix="", updateDFIfUseedFn=None,
                  filter_query=None):
  cache_key = buildCacheKey(cur_fig_save_prefix, comb.name, split_level,
                            align_feedback_time=align_feedback_time,
                            limit_end_epoch=limit_end_epoch,
                            cache_extra_prefix=cache_extra_prefix)
  print("Cache key:", cache_key)
  df_avg_comb = cached_df_avgs.get(cache_key)
  if df_avg_comb is not None:
    print("Found in cache:", cache_key)
    plot_title_postfix, _ = _runConditions(comb, df=None)
    df_avg_comb = _prepareDF(df_avg_comb.copy(), comb_name=comb.name)
    if filter_query is not None:
      df_avg_comb = df_avg_comb.query(filter_query)
    return df_avg_comb, cache_key, plot_title_postfix

  print("Not found in cache:", cache_key)
  # We won't be able to plot a combined heatmap of correct and error as they
  # have different lengths for the feedback epoch
  if comb == Combinations.All and (align_feedback_time is None and
                                   not limit_end_epoch):
    df = df_concat_without_feedback.copy()
  if save_figs_outliers:
    savePrefix = partial(behaviorcommon.savePrefixTmplt,
                        fig_short_name="trace/trace",
                        fig_save_prefix=cur_fig_save_prefix,
                        is_unified=False, is_widefield=False,
                        normalization_str=normalization.asStr(),
                        ext=None, parent_subdir=comb.name,
                        save_context=save_context,
                        #fig_postfix=fig_postfix,
                        #alternate_animal_folder=alternate_animal_folder)
                        )
  else:
    savePrefix = None
  if updateDFIfUseedFn is not None:
    df = updateDFIfUseedFn(df)
  plot_title_postfix, df = _runConditions(comb, df.copy())
  df_avg_comb = avgEpochs(df, by_epoch=True, by_split_level=split_level,
                          **comb.avg_kargs, savePrefix=savePrefix,
                          filter_outlier_trials=filter_outlier_trials,
                          plot_only_traces_ids=[],#only_traces_ids,
                          rename_traces=rename_traces,
                          )
  if len(only_traces_ids) == 0: # We are not handling a special case
    cached_df_avgs[cache_key] = df_avg_comb
  df_avg_comb = _prepareDF(df_avg_comb.copy(), comb_name=comb.name)
  if filter_query is not None:
    df_avg_comb = df_avg_comb.query(filter_query)
  return df_avg_comb, cache_key, plot_title_postfix

def plotTracesStub(comb, df, getAvgFn, normalization, cur_fig_save_prefix,
                   split_level, save_figs, only_traces_ids=[],
                   legendLabelAndLineStyleFn=legendLabelAndLineStyle,
                   areas_to_colors=None,  onlyTraceIdsPlotProcessFn=None,
                   save_dir_prefix="", save_ext=None, save_context=None,
                   renameSaveFn=None, appendSaveFn=None, is_widefield=False,
                   fig_postfix="", **plot_kargs):
  assert not (renameSaveFn is not None and appendSaveFn is not None), (
      "can only specify one of 'renameSaveFn=' or 'appendSaveFn=' at a time")
  normalization_str = normalization.asStr()
  if normalization_str == "without":
    normalization_str = None

  if appendSaveFn is not None:
    savePrefix_kargs = dict(alternative_plot_id_col=None,
                            append_plot_id_col="SaveName")
  else: # Specify "SaveName" eve if the user didn't specify
         # alternative_plot_id_col?
    savePrefix_kargs = dict(alternative_plot_id_col="SaveName",
                            append_plot_id_col=None)
  print("save dir prefix:", save_dir_prefix)
  savePrefix = partial(behaviorcommon.savePrefixTmplt, fig_short_name="",#"avg",
                           fig_save_prefix=cur_fig_save_prefix, #df=plot_df,
                           is_unified=False, is_widefield=is_widefield,
                           fig_postfix=fig_postfix,
                           normalization_str=normalization_str,
                           ext=save_ext,
                           parent_subdir=f"{save_dir_prefix}{comb.name}",
                           save_context=save_context,
                           **savePrefix_kargs,
                           #alternate_animal_folder=alternate_animal_folder)
                          )
  # print("only_traces_ids:", only_traces_ids)
  if areas_to_colors is None:
    areas_to_colors = comb.plt_clr
  df_avg_comb, _, plot_title_postfix = getAvgFn(comb, df=df)
  cur_only_traces_ids = _getOnlyTracesIds(onlyTraceIdsPlotProcessFn,
                                          only_traces_ids, df_avg_comb,
                                          split_level=split_level)
  print("plot kargs:", plot_kargs)
  if renameSaveFn is not None or appendSaveFn is not None:
    applyFn = renameSaveFn if renameSaveFn is not None else appendSaveFn
    df_avg_comb["SaveName"] = df_avg_comb.apply(applyFn, axis=1)
  plotAreas(df_side=None, df_unified=df_avg_comb.copy(), use_unified=True,
            split_level=split_level, only_traces_ids=cur_only_traces_ids,
            loopGroupTracesFn=None, areas_to_colors=areas_to_colors,
            # sem_traces=sem_traces,
            save_context=save_context,
            save_tiff=save_figs, savePrefixFn=savePrefix,
            legendLabelAndLineStyle=legendLabelAndLineStyleFn,
            plot_title_postfix=plot_title_postfix, **plot_kargs)



def heatmapStub(comb, df, getAvgFn, normalization, cur_fig_save_prefix,
                split_level, save_figs, only_traces_ids,
                use_comb_all_seq=True, save_context=None,
                smooth_sigma=sig(plotHeatMap).parameters["smooth_sigma"].default,
                onlyTraceIdsPlotProcessFn=None, heatMapSortSrcDf='None',
                save_extra_prefix="", renameTracesFn=None, renameSaveFn=None,
                only_common_sort_not_common_rng=sig(plotHeatMap).parameters["only_common_sort_not_common_rng"].default,
                save_ext=None, fig_postfix="", **plot_kargs):
  df_avg_comb, *_ = getAvgFn(comb, df=df)
  # display_epochs, extra_prefix = ["Wait Trial Start", "Sampling"], "pre_n_sampling"
  # display_epochs, extra_prefix = ["Sampling"], "only_sampling"
  display_epochs= []
  # display_epochs, extra_prefix = ["Wait Trial Start", "Sampling", "Movement to Lateral Port"], "until_movement"
  if len(save_extra_prefix):
    save_extra_prefix = f"{save_extra_prefix}_"
  normalization_str = normalization.asStr()
  if normalization_str == "without":
    normalization_str = None
  savePrefix = partial(behaviorcommon.savePrefixTmplt,
                      fig_short_name=f"hm_{save_extra_prefix}{comb.name}_", #plot_id=plot_id,
                      fig_save_prefix=cur_fig_save_prefix, #df=plot_df,
                      is_unified=False,
                      is_widefield=False,
                      fig_postfix=fig_postfix,
                      normalization_str=normalization_str,
                      ext=save_ext, parent_subdir="summary/",
                      alternative_plot_id_col="SaveName",
                      save_context=save_context,
                      #alternate_animal_folder=alternate_animal_folder)
                      )
  if (#comb != Combinations.All and # For testing
      use_comb_all_seq and
      heatMapSortSrcDf == 'None'):
    used_split_level = getAvgFn.keywords["split_level"]
    # Remove quantile_idx fom split level if it exists
    all_split_level = used_split_level & ~SplitLevelBy.Quantile
    all_df_avg_comb, _, _ = getAvgFn(Combinations.All, df=df,
                                     split_level=all_split_level)
    def heatMapSortSrcDf(df, by_split_level, split_kws, match_cols):
      # print("All match cols:", match_cols)
      is_df_one_of_them = False
      if "quantile_idx" in match_cols:
        match_cols = match_cols.copy()
        match_cols.remove("quantile_idx")
        by_split_level = by_split_level & ~SplitLevelBy.Quantile
      return all_df_avg_comb, match_cols, by_split_level, is_df_one_of_them

  cur_only_traces_ids = _getOnlyTracesIds(onlyTraceIdsPlotProcessFn,
                                          only_traces_ids=only_traces_ids,
                                          df=df_avg_comb,
                                          split_level=split_level)
  kargs = comb.avg_kargs.copy()
  kargs["smooth_sigma"] = smooth_sigma
  kargs["only_common_sort_not_common_rng"] = only_common_sort_not_common_rng
  df_avg_comb = df_avg_comb.copy()
  if renameTracesFn is not None:
    df_avg_comb["Title"] = df_avg_comb.apply(renameTracesFn, axis=1)
  if renameSaveFn is not None:
    df_avg_comb["SaveName"] = df_avg_comb.apply(renameSaveFn, axis=1)
  plotHeatMap(df=df_avg_comb, by_epoch=True, by_split_level=split_level,
              title_cols=["Title"],
              restrict_to_epochs=display_epochs,
              restrict_sorting_to_epochs=[],#["Sampling"],
              heatMapSortSrcDf=heatMapSortSrcDf,
              # restrict_plotting_to_epochs=display_epochs,
              **kargs, savePrefix=savePrefix,
              only_traces_ids=cur_only_traces_ids,
              save_figs=save_figs, plot_kargs=plot_kargs)


def widefieldVidStub(comb, df, getAvgFn, normalization, cur_fig_save_prefix,
                     split_level, save_figs, only_traces_ids=[],
                     onlyTraceIdsPlotProcessFn=None, save_dir_prefix="",
                     renameSaveFn=None, UpathFn=None, save_ext=None,
                     activity_plot_vid=False, activity_plot_y_lims=None,
                     activity_plot_vid_max_x=None,
                     save_context=None, **plot_kargs):
  assert save_figs, "Didn't implement yet embedded videos"
  is_unified = False
  normalization_str = normalization.asStr()
  if normalization_str == "without":
    normalization_str = None
  savePrefix = partial(behaviorcommon.savePrefixTmplt, fig_short_name="",#"avg",
                       fig_save_prefix=cur_fig_save_prefix, #df=plot_df,
                       is_unified=is_unified, is_widefield=True,
                       #fig_postfix=fig_postfix,
                       normalization_str=normalization_str,
                       ext=save_ext,
                       parent_subdir=f"{save_dir_prefix}{comb.name}",
                       save_context=save_context,
                       alternative_plot_id_col="SaveName",
                       #alternate_animal_folder=alternate_animal_folder)
                      )
  # print("only_traces_ids:", only_traces_ids)
  df_avg_comb, _, plot_title_postfix = getAvgFn(comb, df=df)
  cur_only_traces_ids = _getOnlyTracesIds(onlyTraceIdsPlotProcessFn,
                                          only_traces_ids, df_avg_comb,
                                          split_level=split_level)
  if renameSaveFn is not None:
    df_avg_comb["SaveName"] = df_avg_comb.apply(renameSaveFn, axis=1)
  # print("plot kargs:", plot_kargs)
  from ..widefield import pipelineprocessors as wf_signal
  if UpathFn is None:
    df_avg_comb["U"] = df_avg_comb.anlys_path + "\\U.npy"
  else:
    df_avg_comb["U"] = UpathFn(df_avg_comb)
  if not activity_plot_vid:
    vid_gen = wf_signal.GenVid(assign_mov=False, save_prefix=savePrefix,
                              overlay_text=False)
  else:
    vid_gen = wf_signal.SaveActivityPloltVid(is_avg_trc=True, info_df_dict=plot_kargs["info_df_dict"],
                                             areas_to_colors=plot_kargs["areas_to_colors"], dpi=300,
                                             only_traces_ids=cur_only_traces_ids, save_tiff=True,
                                             plot_y_lims=activity_plot_y_lims,
                                             plot_max_x=activity_plot_vid_max_x,
                                             save_prefix=savePrefix,
                                             plot_fig_size=None)
  chain = pipeline.Chain(
      splitLevel(split_level),
          # Save plotted video
          vid_gen,
  )
  chain.run(df_avg_comb)



def _runConditions(comb : PlotCombination, df=None):
  plot_title_postfix = ""
  if comb.more_cond is not None:
    more_cond = comb.more_cond
    if not isinstance(more_cond, list):
      more_cond = [more_cond]
    for cond in more_cond:
      # print("Cond:", cond)
      if cond.name is not None:
        cond_name = cond.name
      else:
        cond_name = f"{cond.col}{cond.operator}{cond.val}"
      plot_title_postfix = f"{plot_title_postfix}, {cond_name}"
      if df is not None:
        # df = df[df[cond] == val]
        df = df.query(f"{cond.col} {cond.operator} {cond.val}")
  if len(plot_title_postfix):
    plot_title_postfix = f"(when: {plot_title_postfix[2:]})"
  return plot_title_postfix, df

def _prepareDF(df, comb_name):
  df = df.copy()
  if "BrainRegion" in df.columns:
    df = df[df.BrainRegion.isin([BrainRegion.M2_Bi, BrainRegion.ALM_Bi])]
    df["BrainRegionStr"] = df["BrainRegion"].apply(
                                  lambda br: str(BrainRegion(br)).split("_")[0])
  if "Layer" in df.columns:
    print("1. len(df):", len(df))
    df = df[df.Layer == "L23"]
    print("2. len(df):", len(df))
  # Repeat again but aftere we filtered what needs filtering
  if "BrainRegion" in df.columns:
    br_col = df.BrainRegionStr
    br_sp = " "
  else:
    br_col = ""
    br_sp = ""
  if "Layer" in df.columns:
    layer_col = df.Layer
    layer_sp = " "
  else:
    layer_col = ""
    layer_sp = ""
  if "SamplingType" in df.columns:
    sampling_col = df["SamplingType"]
    sampling_sp = " "
  else:
    sampling_col = ""
    sampling_sp = " "
  if "Stimulus" in df.columns:
    stimulus_col = df["Stimulus"]
    stimulus_sp = " "
  else:
    stimulus_col = ""
    stimulus_sp = ""
  if "quantile_idx" in df.columns:
    quantile_col = "- Quantile: " + df["quantile_idx"].astype(str)
    quantile_sp = " "
  else:
    quantile_col = ""
    quantile_sp = ""

  df["SaveName"] = df["TrialNumber"].str.replace("epoch_Wait Trial Start", "")\
                                    .str.replace("_", "").str.strip()
  df["Title"] = (df["ShortName"] + f" - {comb_name} - " + br_col + br_sp +
                 layer_col + layer_sp + stimulus_col + stimulus_sp +
                 quantile_col + quantile_sp + sampling_col + sampling_sp +
                 df["SaveName"])
  # display(df.Title)
  return df