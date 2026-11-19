'''Per-session population heatmaps and per-neuron average traces.

Backend of the heatmap and average-trace panels of the 2P notebooks, all drawn
by ``plottracesavg.plotNormalized``:

- Figures 4H and S9C (fast / typical / slow heatmaps, sorted to all trials) --
  ``2p_2_sequences``;
- Figure 5B (left: one session's heatmap; right: three neurons' left/right
  traces) -- ``2p_3_choice``;
- Figures 6A bottom and 6C (sampling + feedback heatmap and one neuron's
  traces) -- ``2p_4_feedback``.

``plotNormalized`` caches the averaged traces in ``cached_df_avgs``, a dict
carried in the plotting context, and later calls read the cache. Several
panels depend on that order: a heatmap "sorted to all trials" takes its row
order from an all-trials heatmap computed earlier on the same context. So a
notebook first runs the combinations once with ``save_figs=False`` to fill the
cache, then draws the figures.

- ``makeContext`` -- the plotting context: the cache, the cache-file prefix,
  where figures go and the trace normalization.
- ``avg`` -- one combination (heatmap or traces) for every session, or only the
  sessions / neurons in ``plot_only_ids_li``.
- ``loopHeatmapCombinations`` -- ``avg`` over sorting, firing-range and
  combination options.
- ``loopCombinationsTraces`` -- the per-neuron left/right average traces.

File names (``renameSave``) encode the options, so a figure cell and the
full-run loop that both draw a session write the same file.
'''
from __future__ import annotations

from functools import partial

import numpy as np

from ..common.definitions import BrainRegion
from ..pipeline import tracesnormalize
from .dataload import loadSgfNeurons
from .evd_conditions_common import SplitLevelBy
from .plottracesavg import Combinations, Plots
from .plottracesavg import plotNormalized as _plotNormalized


def makeContext(cache_extra_prefix, fig_save_prefix, normalization=None,
                **extra):
    '''The keyword arguments every ``avg`` call of one notebook shares.

    ``cached_df_avgs`` starts empty and is filled by the calls; keep one
    context per family of panels that must share a cache.
    '''
    if normalization is None:
        normalization = tracesnormalize.NoNormalization()
    return dict(cached_df_avgs={}, cache_extra_prefix=cache_extra_prefix,
                fig_save_prefix=fig_save_prefix, normalization=normalization,
                **extra)


def _sgfTracesSet(sgf_data_col, sgf_prior_data_col):
    '''Choice-selective neurons for ``sgf_data_col`` as "<neuron>_<session>".'''
    sgf_traces_df = loadSgfNeurons("all")
    sgf_traces_df = sgf_traces_df[sgf_traces_df.data_col == sgf_data_col]
    if sgf_prior_data_col is not None:
        sgf_traces_df = sgf_traces_df[
            sgf_traces_df.prior_data_col == sgf_prior_data_col]
    else:
        sgf_traces_df = sgf_traces_df[sgf_traces_df.prior_data_col.isnull()]
    return {f"{_id.rsplit('_', 1)[1]}_{_id.rsplit('_', 1)[0]}"
            for _id in set(sgf_traces_df.long_trace_id)}


def saveName(args, combination, *, plot_only_ids_li, split_by_session,
             split_by_quantile, only_sgf, plot_traces, sgf_data_col,
             sgf_prior_data_col, heatmap_sort_str):
    '''The file-name stem of one figure, from its group ``args``.'''
    dscrp = ""
    # A heatmap over every session pooled says so first
    is_comb_all_sess = not len(plot_only_ids_li) and not split_by_session
    if is_comb_all_sess:
        dscrp += "_all_sess"
    if split_by_quantile:
        dscrp += f"_q{args['quantile_idx']}"
    combination_kargs = combination.avg_kargs.copy()
    if combination_kargs.pop("by_difficulty", False):
        dscrp += f"_Diff{args['DVstr']}"
    if combination_kargs.pop("by_correctness", False):
        dscrp += "_Correct" if args["ChoiceCorrect"] else "_Incorrect"
    if combination_kargs.pop("by_direction", False):
        dscrp += "_Left" if args["ChoiceLeft"] else "_Right"
    if combination_kargs.pop("by_prev_direction", False):
        dscrp += "_PrevLeft" if args["PrevChoiceLeft"] else "_PrevRight"
    if combination_kargs.pop("by_prev_correct", False):
        dscrp += "_PrevIncorrect" if args["PrevChoiceCorrect"] else "_PrevCorrect"
    assert not len(combination_kargs), f"Unhandled kargs: {combination_kargs}"
    if only_sgf and not plot_traces:
        dscrp += f"_only_sgf_{sgf_data_col}"
        if sgf_prior_data_col is not None:
            dscrp += f"_prior_{sgf_prior_data_col}"
    if not is_comb_all_sess:
        dscrp += f"_{args['ShortName']}"
    if "BrainRegion" in args and ~np.isnan(args['BrainRegion']):
        br_str = str(BrainRegion(args['BrainRegion'])).split("_")[0]
        dscrp += f"_{br_str}"
    if not plot_traces:
        dscrp += f"_{heatmap_sort_str}"
        dscrp = dscrp[1:]  # Remove the first "_"
    return dscrp


def heatmapSortStr(combination, resort_heatmaps, resort_using_all,
                   split_by_quantile, only_common_sort_not_common_rng):
    '''How the rows are ordered and scaled, as it appears in the file name.'''
    if not resort_heatmaps:
        return "sorted_to_itself"
    if resort_using_all:
        sort_str = "sorted_to_all"
    else:
        sort_str = "sorted_to"
        combination_kargs = combination.avg_kargs.copy()
        if split_by_quantile:
            sort_str += "_q3"
        if combination_kargs.pop("by_difficulty", False):
            sort_str += "_Easy"
        if combination_kargs.pop("by_correctness", False):
            sort_str += "_Correct"
        if combination_kargs.pop("by_direction", False):
            sort_str += "_Left"
        if combination_kargs.pop("by_prev_direction", False):
            sort_str += "_PrevLeft"
        if combination_kargs.pop("by_prev_correct", False):
            sort_str += "_PrevIncorrect"
        assert not len(combination_kargs), f"Unhandled kargs: {combination_kargs}"
    sort_str += "_self_rng" if only_common_sort_not_common_rng else "_shared_rng"
    return sort_str


def avg(df, cached_df_avgs, cache_extra_prefix, split_by_quantile,
        split_by_session, combination: Combinations,
        resort_heatmaps: bool, resort_using_all: bool,
        only_common_sort_not_common_rng: bool,
        save_figs, fig_save_prefix, normalization,
        plot_only_ids_li=(), plot_only_ids_exact_match: bool = False,
        restrict_heatmap_sorting_to_epochs=(),
        sgf_data_col: str = None, sgf_prior_data_col: str = None,
        plot_traces: bool = False, only_sgf: bool = None,
        skip_group_infos=None):
    '''Draw one combination as heatmaps (default) or average traces.

    only_common_sort_not_common_rng : bool
        If False, the 0-1 of each row will be calculated across all generated
        heatmaps, not just relative to the current average row. I.e, when True,
        you are guaranteed that each row will go from 0 to 1.
    '''
    plot_only_ids_li = list(plot_only_ids_li)
    skip_group_infos = {} if skip_group_infos is None else skip_group_infos
    if resort_using_all:
        assert resort_heatmaps, (
            "Doesn't make sense to resort all without `resort_heatmaps = True`")
    if not len(plot_only_ids_li):
        assert split_by_session, ("Set `split_by_session = True`, otherwise "
        "session will be saved under another folder with wrong session name")
    if only_sgf:
        assert sgf_data_col is not None, (
                           "If only_sgf is True, sgf_data_col must be provided")
    sgf_traces_set = None
    if sgf_data_col is not None:
        if only_sgf is None:
            only_sgf = True
        sgf_traces_set = _sgfTracesSet(sgf_data_col, sgf_prior_data_col)

    def onlyTracesIds(set_name, traces_ids, group_info):
        good_set = set()
        for key, val in skip_group_infos.items():
            if group_info.get(key, None) == val:
                return good_set
        if set_name == "neuronal":
            for trace_id in traces_ids:
                if not len(plot_only_ids_li) or any(
                        ((key in trace_id) if not plot_only_ids_exact_match
                         else (key == trace_id)) for key in plot_only_ids_li):
                    if not only_sgf or trace_id in sgf_traces_set:
                        good_set.add(trace_id)
        return good_set

    def heatMapSortSrcDf(df, split_level, by_kws, match_cols):
        if split_by_quantile:
            df = df[df.quantile_idx == 3]
        if "DVstr" in by_kws:
            df = df[df.DVstr == "Easy"]
        if "ChoiceCorrect" in by_kws:
            df = df[df.ChoiceCorrect == True]  # noqa: E712
        if "ChoiceLeft" in by_kws:
            df = df[df.ChoiceLeft == True]  # noqa: E712
        if "PrevChoiceLeft" in by_kws:
            df = df[df.PrevChoiceLeft == True]  # noqa: E712
        if "PrevChoiceCorrect" in by_kws:
            df = df[df.PrevChoiceCorrect == False]  # noqa: E712
        match_cols = [col for col in match_cols if col != "quantile_idx"]
        is_df_one_of_them = True
        return df, match_cols, split_level, is_df_one_of_them

    heatmap_sort_str = heatmapSortStr(combination, resort_heatmaps,
                                      resort_using_all, split_by_quantile,
                                      only_common_sort_not_common_rng)
    if not resort_heatmaps:
        sort_src = None
    elif resort_using_all:
        sort_src = "None"
    else:
        sort_src = heatMapSortSrcDf

    renameSave = partial(saveName, combination=combination,
                         plot_only_ids_li=plot_only_ids_li,
                         split_by_session=split_by_session,
                         split_by_quantile=split_by_quantile, only_sgf=only_sgf,
                         plot_traces=plot_traces, sgf_data_col=sgf_data_col,
                         sgf_prior_data_col=sgf_prior_data_col,
                         heatmap_sort_str=heatmap_sort_str)

    cur_split_level = SplitLevelBy.BrainRegion
    if split_by_quantile:
        cur_split_level |= SplitLevelBy.Quantile
    if split_by_session:
        cur_split_level |= SplitLevelBy.Session

    plot_save_dir_prefix = ""
    plot_save_dir_prefix += "sgf/" if only_sgf else "sgf_and_not_sgf/"
    plot_save_dir_prefix += "quantiles/" if split_by_quantile else "no_quantiles/"
    _plotNormalized(
        df_concat_norm=df, df_concat_without_feedback=None,
        combinations=[combination], cached_df_avgs=cached_df_avgs,
        plots=[Plots.Heatmap] if not plot_traces else [Plots.Traces],
        save_figs=save_figs, save_figs_outliers=False,
        fig_save_prefix=fig_save_prefix, normalization=normalization,
        split_level=cur_split_level, align_feedback_time=None,
        limit_end_epoch=True, filter_outlier_trials=False,
        plots_kargs={Plots.Heatmap: dict(plot_y_ids=True, dpi=300,
                                         heatMapSortSrcDf=sort_src,
                                         renameSaveFn=renameSave,
                                         save_ext="svg",
                                         only_common_sort_not_common_rng=only_common_sort_not_common_rng,
                                         restrict_sorting_to_epochs=list(restrict_heatmap_sorting_to_epochs),
                                         ),
                     Plots.Traces: dict(sem_traces=True,
                                        appendSaveFn=renameSave,
                                        save_dir_prefix=plot_save_dir_prefix,
                                        save_ext="svg",)},
        only_traces_ids=[],
        onlyTraceIdsPlotProcessFn=onlyTracesIds,
        cache_extra_prefix=cache_extra_prefix,
        save_context=(["quantile_idx"] if split_by_quantile else []) +
                     (["Layer", "BrainRegion", "ChoiceLeft"] if not plot_traces else []))


class HeatmapSortingOptions:
    '''(resort_heatmaps, resort_using_all) pairs.'''
    #: Sorted by each heatmap's own peak positions, e.g. a quantile by itself.
    SORT_TO_SELF = (False, False)
    #: Sorted to another, hard-coded reference heatmap.
    RESORT_TO_ANOTHER = (True, False)
    #: Sorted to the sequence of the all-trials heatmap.
    RESORT_TO_ALL = (True, True)


class HeatmapFirinhRangeOptions:
    '''Values of ``only_common_sort_not_common_rng``.'''
    #: Each row is scaled 0-1 within itself.
    SELF_RNG = True
    #: Each row is scaled 0-1 over every heatmap of the same condition.
    SHARED_RNG = False


def loopHeatmapCombinations(df, context, split_by_quantile, save_figs,
                            plot_only_sess_li=(),
                            sgf_data_cols_comb_li=((None, None),),
                            combin_all_sess_li=(False,),
                            heatmap_sorting_options_li=(HeatmapSortingOptions.RESORT_TO_ALL,),
                            combinations_li=(Combinations.All,),
                            heatmap_firing_range_options_li=(HeatmapFirinhRangeOptions.SELF_RNG,
                                                             HeatmapFirinhRangeOptions.SHARED_RNG),
                            skip_group_infos=None):
    '''``avg`` heatmaps over every combination of the options given.'''
    plot_only_sess_li = list(plot_only_sess_li)
    for _sgf_data_col, _sgf_prior_data_col in sgf_data_cols_comb_li:
        for _combin_all_sess in combin_all_sess_li:
            if _combin_all_sess and len(plot_only_sess_li):
                continue
            for _resort_to_another, _another_is_all in heatmap_sorting_options_li:
                for not_shared_rng in heatmap_firing_range_options_li:
                    for _combination in combinations_li:
                        print(f"*****Processing: {_combination.name = } - {_combin_all_sess = } - {_resort_to_another = } - {_another_is_all = } - "
                              f"{not_shared_rng = } - {_sgf_data_col = } - {_sgf_prior_data_col = } {split_by_quantile = }")
                        sess_kargs = (dict(split_by_session=False, plot_only_ids_li=[])
                                      if _combin_all_sess else
                                      dict(split_by_session=True,
                                           plot_only_ids_li=plot_only_sess_li))
                        avg(df=df, split_by_quantile=split_by_quantile, **context,
                            combination=_combination,
                            resort_heatmaps=_resort_to_another,
                            resort_using_all=_another_is_all,
                            only_common_sort_not_common_rng=not_shared_rng,
                            **sess_kargs,
                            sgf_data_col=_sgf_data_col,
                            sgf_prior_data_col=_sgf_prior_data_col,
                            save_figs=save_figs,
                            skip_group_infos=skip_group_infos)


def loopCombinationsTraces(df, context, split_by_quantile, only_sgf, save_figs,
                           plot_only_ids_li=(), plot_only_ids_exact_match=False):
    '''Draw the per-neuron left/right average traces of every session, or only
    of the neurons named in ``plot_only_ids_li`` ("<neuron>_<session>" IDs).'''
    local_kargs = dict(context)
    local_kargs.update(dict(resort_heatmaps=False, resort_using_all=False,
                            only_common_sort_not_common_rng=False,
                            plot_traces=True, split_by_session=True))
    for combination, sgf_data_col, sgf_prior_data_col in [
            (Combinations.Direction, "ChoiceLeft", None)]:
        print(f"*****Processing: {combination.name = } -  - "
              f" {sgf_data_col = } - {sgf_prior_data_col = } {split_by_quantile = }")
        avg(df=df, split_by_quantile=split_by_quantile, **local_kargs,
            combination=combination, sgf_data_col=sgf_data_col,
            sgf_prior_data_col=sgf_prior_data_col, save_figs=save_figs,
            only_sgf=only_sgf, plot_only_ids_li=list(plot_only_ids_li),
            plot_only_ids_exact_match=plot_only_ids_exact_match)
