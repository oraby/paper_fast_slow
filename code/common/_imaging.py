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
from .plottracesprocessor import PlotTraces
from .definitions import BrainRegion
try:
    from IPython import display
except ImportError:
    pass
import numpy as np
import pandas as pd
from enum import IntFlag, auto
from functools import partial
from itertools import chain as iterChain
from pathlib import Path

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
    assert isinstance(split_level, SplitLevelBy), ("Update your code or "
        "check whether auto-reload lead to old enum reference")
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
        df = df.groupby("anlys_path", group_keys=False).apply(
                                                             addHasFeedbackWait)
        df.loc[df.HasFeedbackWait & (df.epoch == "Feedback Wait"),
               "epoch"] = "Feedback Start"
        df.loc[~df.HasFeedbackWait & df.epoch.isin(["Reward", "Punishment"]),
               "epoch"] = "Feedback Start"
    return df

def normalizeTraces(df, normalization, split_level : SplitLevelBy,
                    align_feedback_time=None, limit_end_epoch=False):
    assert "epoch_time" in df.columns
    rename_feedback_epoch = (align_feedback_time is not None and
                             align_feedback_time > 0)
    df = commonTraceNormalizationInit(df, normalization, rename_feedback_epoch)
    # This epoch will only exist if we renamed the feedbacl epochs
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
                    # return trial_df[trial_df.epoch ==
                    #                 "Feedback Start"].trace_end_idx.iloc[0]
                assert feedback_idx.sum() == 1, "Expected 1 feedback epoch"
                feedback_row = trial_df[feedback_idx].iloc[0]
                feedback_start_row_idx = trial_df.epoch == "Feedback Start"
                assert feedback_start_row_idx.sum() == 1, (
                                              "Expected 1 feedback start epoch")
                assert (
                    trial_df[feedback_start_row_idx].trace_start_idx.iloc[0] <=
                    feedback_row.trace_start_idx)
                trial_df.loc[feedback_start_row_idx,
                             "trace_end_idx"] = feedback_row.trace_end_idx
                # Remove the extra epoch
                trial_df = trial_df[~feedback_idx]
                # Update epoch time
                #TODO: Include while building the dataframe
                feedback_start_row = trial_df[feedback_start_row_idx].iloc[0]
                # A bit stupid, as we are using .loc to refer to the same row.
                # I don't know how to update a row inplace.
                trial_df.loc[feedback_start_row_idx, "epoch_time"] = (
                    (feedback_start_row.trace_end_idx -
                     feedback_start_row.trace_start_idx + 1) /
                                           feedback_start_row.acq_sampling_rate)
                return trial_df
            df = df.groupby([df.ShortName, df.TrialNumber],
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
        # *align_feedback_processors,    # For quick testing
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
                              formatRow=gen_by_processors.formatRow)
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


def _debugArr(df):
    least_dur = df.dur_bin.min()
    ex_traces = df[df.dur_bin == least_dur].iloc[0].traces_sets["neuronal"]
    ex_trace = next(iter(ex_traces.values()))
    print("dur", least_dur, "arr:", ex_trace.shape, (~np.isnan(ex_trace)).sum())

def plotHeatMap(df, by_split_level : SplitLevelBy, save_figs, savePrefix=None,
                heatMapSortSrcDf=None, only_common_sort_not_common_rng=False,
                title_cols=[], restrict_to_epochs=[], only_traces_ids=[],
                restrict_sorting_to_epochs=[], smooth_sigma=2,
                plot_kargs={}, **by_processors,):
    gen_by_processors = GenByProcessor(**by_processors)
    def _restrictToEpochs(_df):
        if len(restrict_to_epochs):
            chain = pipeline.Chain(
                 tracesrestructure.KeepOnlyEpochsInConcatEpochs(
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
                                                    set_name="neuronal")
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
        # print("Before:"), _debugArr(df[0])
        df = normalizeFiringRng(df)
        # display(pd.concat(df))
        df = sortDf(df)
        # print("After:"), _debugArr(df[0])
    else:
        # print("Sorting src df")
        match_cols = _splitLevelByCols(by_split_level)
        print("Match cols:", match_cols)
        # assert len(match_cols), (
        #           "No matching cols, if on purporse then skip the loop below")
        def matchTrialsFn(row, src_df):
            # print("Row:", row[match_cols])
            # src_row = src_df[
            # src_df[sort_src_df_match_col_name] ==
            #        row[sort_src_df_match_col_name]]
            # assert len(src_row) == 1, f"Expected 1 row, found {len(src_row)}"
            matchcing = src_df[match_cols[0]] == row[match_cols[0]]
            for col in match_cols[1:]:
                matchcing &= src_df[col] == row[col]
            if matchcing.sum() != 1:
                display(row[match_cols])
                display(src_df[match_cols])
            assert matchcing.sum() == 1, (
                                     f"Expected 1 row, found {matchcing.sum()}")
            src_row = src_df[matchcing]
            # print(src_df.TrialNumber)
            return src_row.iloc[0]
        sort_traces_src_df, match_cols, sort_split_level, is_one_of_them = \
                         heatMapSortSrcDf(pd.concat(df), by_split_level,
                                          gen_by_processors._by_kws, match_cols)
        # print("New match cols:", match_cols)
        sort_traces_src_df = _restrictToEpochs(sort_traces_src_df)
        # display(sort_traces_src_df[gen_by_processors._by_kws])
        sort_traces_src_df = splitDf(sort_traces_src_df,
                                     # TODO: Probably should return the actual
                                     # extra cols (if any) to be looped on
                                     use_split_level=sort_split_level,
                                     use_gen_by_processors=is_one_of_them)
        # display(sort_traces_src_df)
        # Sort the source first to the desired order
        # sort_traces_src_df = normalizeFiringRng(sort_traces_src_df.copy())
        sort_traces_src_df = sortDf(sort_traces_src_df,
                                    used_smooth_sigma=smooth_sigma)
        sort_traces_src_df = pipeline.Chain(pipeline.RecombineResults()).run(
                                                             sort_traces_src_df)
        # print("Only ids:", only_traces_ids)
        # Then copy the order to the destination
        df = pipeline.Chain(
                pipeline.RecombineResults(),
                utils.CopyTracesSortOrderProcessor(src_df=sort_traces_src_df,
                                                   set_name="neuronal",
                                                   matchTrialsFn=matchTrialsFn,
                                                   match_sess_cols=match_cols,
                                                   only_traces_ids=[]
                                                   )
        ).run(df)
        if not only_common_sort_not_common_rng:
            df = tracesnormalize.normalizeMinMaxPerTrace(df,
                                                match_rows_col_name="ShortName",
                                                set_name="neuronal",
                                                #norm_min_max_val_rng=(0, 100),
                                                smooth_sigma=smooth_sigma)
            df = splitDf(df, use_gen_by_processors=True)
        else:
            df = splitDf(df, use_gen_by_processors=True)
            df = normalizeFiringRng(df.copy())
            df = pipeline.Chain(
                tracesmath.GaussianFilter(sigma=smooth_sigma,
                                          set_name="neuronal", copy_data=True),
            #     tracesnormalize.NormalizePercentile(0, 100,
            #                                         set_name="neuronal",
            #                          ),
            ).run(df)
        # df = splitDf(df)

    chain = pipeline.Chain(
        plotters.TracesHeatMap(is_avg_trc=True, title_cols=title_cols,
                               title_prefix="", title_cols_names=False,
                               write_title_plot_id=False,
                               save_prefix=savePrefix if save_figs else None,
                               # restrict_to_epochs=restrict_plotting_to_epochs,
                               heatmap_min=0, heatmap_max=1,
                               only_traces_ids=only_traces_ids,
                               **plot_kargs),
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
    def __init__(self, by=[], **by_kwargs,):
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
        if self._by is not None and (not hasattr(self._by, "__len__") or
                                     len(self._by)):
            by_processors.append(pipeline.By(self._by))
        return by_processors

    def formatRow(self, row):
        str_li = []
        def asrtAdd(val, _str):
            nonlocal str_li
            assert val is None or np.isnan(val)
            str_li.append(_str)
        if "epoch" in self._by_kws:
            str_li.append(f"_epoch_{row.epoch}")
        if "PrevChoiceCorrect" in self._by_kws:
            if row.PrevChoiceCorrect == True:
                str_li.append("_PrevCorrectChoice")
            elif row.PrevChoiceCorrect == False:
                str_li.append("_PrevIncorrectChoice")
            else:
                asrtAdd(row.PrevChoiceCorrect, "_NoPrevChoice")
        if "ChoiceCorrect" in self._by_kws:
            if row.ChoiceCorrect == True:
                str_li.append("_CorrectChoice")
            elif row.ChoiceCorrect == False:
                str_li.append("_IncorrectChoice")
            else:
                asrtAdd(row.ChoiceCorrect, "_NoChoice")
        if "DVstr" in self._by_kws:
            str_li.append(f"_{row.DVstr}")
        if "PrevChoiceLeft" in self._by_kws:
            if row.PrevChoiceLeft == True:
                str_li.append("_PrevDecisionLeft")
            elif row.PrevChoiceLeft == False:
                str_li.append("_PrevDecisionRight")
            else:
                asrtAdd(row.PrevChoiceLeft, "_PrevDecisionNone")
        if "ChoiceLeft" in self._by_kws:
            if row.ChoiceLeft == True:
                str_li.append("_DecisionLeft")
            elif row.ChoiceLeft == False:
                str_li.append("_DecisionRight")
            else:
                asrtAdd(row.ChoiceLeft, "_DecisionNone")
        if "dur_bin" in self._by_kws:
            str_li.append(
                         f"_Dur_{row.dur_bin.left:.1f}_{row.dur_bin.right:.1f}")
        if len(self._by):
            for col in self._by:
                val = row[col]
                _str = f"{val:.2f}" if isinstance(val, float) else str(val)
                str_li.append(f"_{col}_{_str}")
        return "".join(str_li)[1:]

def plotAreas(df_side, df_unified, split_level, use_unified,
              x_label, y_label, areas_to_colors,
              loopGroupTracesFn, savePrefixFn, only_traces_ids=[],
              trial_number_as_plot_id=True, multiple_trace_ids_per_plot=False,
              set_name="neuronal", **plot_traces_kargs):
    if callable(areas_to_colors):
        local_areas_colors = areas_to_colors
    else:
        local_areas_colors = areas_to_colors.copy()
    df = df_unified if use_unified else df_side
    df = df.copy()
    # from datetime import date
    # df = df[(df.Name.str.startswith("widefield_5")) &
    #         (df.Date == date(2021, 7, 12))]
    # df.loc[df.epoch == "Punishment","trace_end_"] =
    if not use_unified:
        only_traces_ids = list(
         iterChain(*[[f"{el}_left", f"{el}_right"] for el in only_traces_ids]))
    print("only_traces_ids:", only_traces_ids)
    print("local_areas_colors:", local_areas_colors)
    print("multiple_trace_ids_per_plot:", multiple_trace_ids_per_plot)
    plotter_only_traces_ids = \
                          only_traces_ids if multiple_trace_ids_per_plot else []
    print("plotter only traces ids:", plotter_only_traces_ids)
    traces_plotter = PlotTraces(is_avg_trc=not trial_number_as_plot_id,
                                x_label=x_label, y_label=y_label,
                                areas_to_colors=local_areas_colors,
                                only_traces_ids=plotter_only_traces_ids,
                                save_prefix=savePrefixFn, set_name=set_name,
                                **plot_traces_kargs)
    if not multiple_trace_ids_per_plot:
        traces_wrap = pipeline.LoopTraces(traces_plotter,
                                          set_name=set_name,
                                          groupTracesFn=loopGroupTracesFn,
                                          only_traces_ids=only_traces_ids,)
    else:
        traces_wrap = traces_plotter
    chain = pipeline.Chain(
            splitLevel(split_level),
                *[traces_wrap],
    )
    chain.run(df)



def savePrefixTmplt(fig_short_name, plot_id, parent_dir, df, is_unified,
                    fig_save_prefix, is_widefield, fig_postfix="",
                    normalization_str=None, ext=None, parent_subdir=None,
                    save_context=None, alternate_animal_folder=None,
                    alternative_plot_id_col=None, append_plot_id_col=None,
                    # This values here will be replaced with the corresponding
                    # variables.
                    save_fp=(
                   "{fig_save_prefix}/{session_type_path}/"
                   "{session_dir}/{parent_subdir}/"
                   "{(fig_short_name + \"_\") if len(fig_short_name) else \"\"}"
                   "{plot_id}{fig_postfix}{norm}.{ext}")
                ):
    if ext is None:
        ext = "jpeg"
    assert is_widefield != None
    # print("Plot id:", plot_id)
    # print("Parent dir:", parent_dir)
    # anlys_path = df.anlys_path
    # if len(anlys_path) and anlys_path.iloc[0] != str(parent_dir)[0]:
    #         parent_dir = str(Path(parent_dir))
    #         anlys_path = anlys_path.apply(Path).astype(str)
    animal_df = df[df.ShortName == parent_dir]

    if is_widefield:
        stimulis_kw = (
            "stimulus_type" if "stimulus_type" in animal_df.columns else
            "Stimulus")
        sampling_kw = (
            "sampling_type" if "sampling_type" in animal_df.columns else
            "SamplingType")
        stim = animal_df[stimulis_kw].iloc[0]
        sample = animal_df[sampling_kw].iloc[0]
        session_type_path = f"{stim}/{sample}"
        if is_unified:
            plot_id += "_bi"
        session_dir = f"{Path(parent_dir).name}"
    else:
        sample = animal_df["SamplingType"].iloc[0]
        layer = animal_df["Layer"].iloc[0].replace("/", "")
        brain_region = \
                f"{BrainRegion(animal_df['BrainRegion'].iloc[0])}".split("_")[0]
        session_type_path = f"{sample}/{brain_region}/{layer}"
        animal_name = animal_df["ShortName"].iloc[0]
        if alternate_animal_folder is None:
            session_dir = f"{animal_name}/"
        else:
            session_dir = f"{alternate_animal_folder}/{animal_name}_"
    if alternative_plot_id_col is not None:
        plot_id = animal_df[alternative_plot_id_col].iloc[0]
    if append_plot_id_col is not None:
        plot_id = plot_id + str(animal_df[append_plot_id_col].iloc[0])
    if save_context is not None:
        if isinstance(save_context, pd.Series):
            save_context = "_".join(f"{idx}={val}"
                                    for idx, val in save_context.items())
        plot_id = f"{plot_id}_{save_context}"
    if "dur_bin" in animal_df.columns:
        dur_bin = animal_df.dur_bin.iloc[0]
        dur_bin = f"{dur_bin.left}s_{dur_bin.right}s"
        plot_id_split = plot_id.split('_')
        plot_id = f"{plot_id_split[0]}_{dur_bin}_{'_'.join(plot_id_split[1:])}"
    # print("plot_id:", plot_id)
    if len(fig_postfix):
        fig_postfix = f"_{fig_postfix}"
    norm = f"_{normalization_str}" if normalization_str is not None else ""
    # Thanks https://stackoverflow.com/a/57597617/11996983
    save_fp = Path(eval(f"f'{save_fp}'"))
    save_fp.parent.mkdir(exist_ok=True, parents=True)
    # print(f"save fp:", save_fp)
    return save_fp
