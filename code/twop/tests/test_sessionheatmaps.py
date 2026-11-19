'''Tests for the session heatmap / average-trace backend (``sessionheatmaps``).

``plottracesavg.plotNormalized`` does the drawing and is stubbed here. What
these pin is what this module decides before handing over: the file-name stem
of each figure (a figure cell and the full-run loop must write the same file),
the heatmap sort reference, which neurons get through the filter, and that the
plotting context's cache is the one passed on.
'''
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ...common.definitions import BrainRegion
from .. import sessionheatmaps as sh
from ..plottracesavg import Combinations, Plots

MFC = int(BrainRegion.M2_Bi)
SESS = "GP4_28_S2_L51_D250_M2L"


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(sh, "_plotNormalized", lambda **kw: seen.append(kw))
    return seen


def _context():
    return sh.makeContext(cache_extra_prefix="pfx", fig_save_prefix="out/")


def _stem(**kw):
    base = dict(plot_only_ids_li=[SESS], split_by_session=True,
                split_by_quantile=False, only_sgf=False, plot_traces=False,
                sgf_data_col=None, sgf_prior_data_col=None,
                heatmap_sort_str="sorted_to_itself")
    base.update(kw)
    args = {"ShortName": SESS, "BrainRegion": MFC, "quantile_idx": 1,
            "ChoiceLeft": True}
    return sh.saveName(args, kw.pop("combination", Combinations.All), **{
        k: v for k, v in base.items() if k != "combination"})


def test_figure_4h_stem_matches_the_published_file():
    # results/2P/Sessions/MFC/<session>/summary/hm_All__q1_<session>_M2_sorted_to_all_shared_rng.svg
    stem = _stem(split_by_quantile=True,
                 heatmap_sort_str="sorted_to_all_shared_rng")
    assert stem == f"q1_{SESS}_M2_sorted_to_all_shared_rng"


def test_traces_stem_names_the_side_and_keeps_its_leading_underscore():
    stem = _stem(combination=Combinations.Direction, plot_traces=True)
    assert stem == f"_Left_{SESS}_M2"


def test_pooled_sessions_stem_says_so_and_drops_the_session():
    stem = _stem(plot_only_ids_li=[], split_by_session=False,
                 heatmap_sort_str="sorted_to_itself")
    assert stem == "all_sess_M2_sorted_to_itself"


@pytest.mark.parametrize("resort, using_all, self_rng, expected", [
    (False, False, True, "sorted_to_itself"),
    (True, True, False, "sorted_to_all_shared_rng"),
    (True, True, True, "sorted_to_all_self_rng"),
    (True, False, False, "sorted_to_q3_shared_rng"),
])
def test_heatmap_sort_str(resort, using_all, self_rng, expected):
    assert sh.heatmapSortStr(Combinations.All, resort, using_all,
                             split_by_quantile=True,
                             only_common_sort_not_common_rng=self_rng) == expected


def test_avg_hands_over_the_context_cache_and_sort_reference(calls):
    ctx = _context()
    sh.avg(df="DF", split_by_quantile=False, split_by_session=True,
           combination=Combinations.All, resort_heatmaps=True,
           resort_using_all=True, only_common_sort_not_common_rng=False,
           save_figs=False, **ctx)
    (kw,) = calls
    assert kw["cached_df_avgs"] is ctx["cached_df_avgs"]
    assert kw["fig_save_prefix"] == "out/"
    assert kw["plots"] == [Plots.Heatmap]
    # Sorted to all trials: plotNormalized takes the string "None"
    assert kw["plots_kargs"][Plots.Heatmap]["heatMapSortSrcDf"] == "None"
    assert kw["save_context"] == ["Layer", "BrainRegion", "ChoiceLeft"]


def test_avg_without_resort_passes_no_sort_reference(calls):
    sh.avg(df="DF", split_by_quantile=True, split_by_session=True,
           combination=Combinations.All, resort_heatmaps=False,
           resort_using_all=False, only_common_sort_not_common_rng=True,
           save_figs=False, **_context())
    (kw,) = calls
    assert kw["plots_kargs"][Plots.Heatmap]["heatMapSortSrcDf"] is None
    assert kw["save_context"][0] == "quantile_idx"


def _filterFn(calls, **kw):
    base = dict(df="DF", split_by_quantile=False, split_by_session=True,
                combination=Combinations.Direction, resort_heatmaps=False,
                resort_using_all=False, only_common_sort_not_common_rng=False,
                save_figs=False, plot_traces=True)
    base.update(kw)
    sh.avg(**base, **_context())
    return calls[-1]["onlyTraceIdsPlotProcessFn"]


def test_neuron_filter_exact_and_substring(calls):
    ids = ["2_S", "20_S", "95_S"]
    exact = _filterFn(calls, plot_only_ids_li=["2_S"], plot_only_ids_exact_match=True)
    assert exact("neuronal", ids, {}) == {"2_S"}
    # Without exact matching a key matches anywhere in the ID
    loose = _filterFn(calls, plot_only_ids_li=["2"])
    assert loose("neuronal", ids, {}) == {"2_S", "20_S"}
    assert loose("other_set", ids, {}) == set()


def test_neuron_filter_keeps_only_choice_selective(calls, monkeypatch):
    sgf = pd.DataFrame({"data_col": ["ChoiceLeft", "ChoiceLeft", "PrevChoiceLeft"],
                        "prior_data_col": [np.nan, np.nan, np.nan],
                        "long_trace_id": ["S_2", "S_95", "S_20"]})
    monkeypatch.setattr(sh, "loadSgfNeurons", lambda which: sgf)
    fn = _filterFn(calls, sgf_data_col="ChoiceLeft", only_sgf=True)
    assert fn("neuronal", ["2_S", "20_S", "95_S"], {}) == {"2_S", "95_S"}


def test_neuron_filter_skips_groups(calls):
    fn = _filterFn(calls, skip_group_infos={"quantile_idx": 2})
    assert fn("neuronal", ["1_S"], {"quantile_idx": 2}) == set()
    assert fn("neuronal", ["1_S"], {"quantile_idx": 1}) == {"1_S"}


def test_loop_skips_pooled_sessions_when_sessions_are_named(calls):
    sh.loopHeatmapCombinations("DF", _context(), split_by_quantile=False,
                               save_figs=False, plot_only_sess_li=[SESS],
                               combin_all_sess_li=[True, False])
    # 1 sorting x 2 ranges x 1 combination, the pooled pass skipped
    assert len(calls) == 2
    assert {c["plots_kargs"][Plots.Heatmap]["only_common_sort_not_common_rng"]
            for c in calls} == {True, False}


def test_traces_loop_draws_the_left_right_traces(calls):
    sh.loopCombinationsTraces("DF", _context(), split_by_quantile=False,
                              only_sgf=False, save_figs=False,
                              plot_only_ids_li=("2_S",))
    (kw,) = calls
    assert kw["plots"] == [Plots.Traces]
    assert kw["combinations"] == [Combinations.Direction]
    assert kw["plots_kargs"][Plots.Traces]["save_dir_prefix"] == \
        "sgf_and_not_sgf/no_quantiles/"
