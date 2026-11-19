'''Tests for the real-time subpopulation traces (Figures S12D, S12J).

Pinned here: the sequence-bin membership the 5E / S12E bars use (now shared
with S12D), the two ways a line can be summarised (all trials pooled, which both figures
use; or each neuron's trials averaged before the across-neuron mean and SEM),
the two colour schemes,
that a sample most trials no longer reach is not drawn, and what the plotter
puts on the axes and where it saves.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from .. import subpoptraces as sp
from ..movementneurons import binEdges
from ..plottuning import (RealTimeTuningPlot, neuronsInBin,
                          pooledTrialsMeanTrace, subpopulationMeanTrace)

NAMES = ["-0.1s Sampling", "Sampling", "Movement to Lateral Port"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --- neuronsInBin -----------------------------------------------------------
def _queryReference(iqr_df, bins, bin_idx):
    """The query the 5E bars ran before it was factored out."""
    max_len = bins[-1]
    l, r = 100*bins[bin_idx]/max_len, 100*bins[bin_idx + 1]/max_len
    b0, bl = 100*bins[1]/max_len, 100*bins[-2]/max_len
    q = iqr_df.query(f"({l} <= q1_prcnt <= {r} or {l} <= q3_prcnt <= {r})"
                     f"and med_prcnt >= {b0} and med_prcnt <= {bl}")
    return set(q.trace_id)


def test_neurons_in_bin_matches_the_original_query():
    rng = np.random.default_rng(0)
    q = np.sort(rng.uniform(0, 100, (200, 3)), axis=1)
    iqr = pd.DataFrame({"trace_id": [f"s_{i}" for i in range(200)],
                        "q1_prcnt": q[:, 0], "med_prcnt": q[:, 1],
                        "q3_prcnt": q[:, 2]})
    iqr.loc[0, "q1_prcnt"] = np.nan
    bins = binEdges([(0, 2), (3, 29), (30, 32)], 7)
    for bin_idx in range(len(bins) - 1):
        assert neuronsInBin(iqr, bins, bin_idx) == _queryReference(iqr, bins, bin_idx)


def test_a_neuron_spanning_two_bins_is_in_both():
    bins = np.array([0, 10, 20, 30, 40])
    iqr = pd.DataFrame({"trace_id": ["a"], "q1_prcnt": [30.0],
                        "med_prcnt": [40.0], "q3_prcnt": [55.0]})
    assert [bool(neuronsInBin(iqr, bins, i)) for i in range(4)] == \
        [False, True, True, False]


# --- subpopulationMeanTrace -------------------------------------------------
def expanded(rows):
    """rows: (long_trace_id, trace, movement_onset)."""
    n = max(len(t) for _, t, _ in rows)
    data = {("long_trace_id", ""): [r[0] for r in rows],
            ("epochs_names", ""): [NAMES] * len(rows),
            ("epochs_ranges", ""): [[(0, 0), (1, m - 1), (m, n - 1)]
                                    for _, _, m in rows]}
    for i in range(n):
        data[("Raw", i)] = [t[i] if i < len(t) else np.nan for _, t, _ in rows]
    return pd.DataFrame(data)


def test_trials_are_averaged_within_a_neuron_first():
    # Neuron a: 3 trials at 0; neuron b: 1 trial at 4. Pooled trials -> 1,
    # neuron means -> 2.
    df = expanded([("a", [0.0, 0.0], 1)] * 3 + [("b", [4.0, 4.0], 1)])
    mean, sem = subpopulationMeanTrace(df)
    np.testing.assert_allclose(mean, [2.0, 2.0])
    np.testing.assert_allclose(sem, [2.0, 2.0])     # std([0, 4]) / sqrt(2)


def test_samples_most_trials_do_not_reach_are_dropped():
    df = expanded([("a", [1.0, 1.0, 1.0], 1), ("a", [1.0], 1), ("b", [3.0], 1)])
    mean, _ = subpopulationMeanTrace(df)
    assert mean[0] == pytest.approx(2.0)
    assert np.isnan(mean[1:]).all()                 # 1 of 3 rows left
    mean, _ = subpopulationMeanTrace(df, min_trial_fraction=0.3)
    assert mean[2] == pytest.approx(1.0)


# --- pooledTrialsMeanTrace -------------------------------------------------
def test_pooled_trials_weigh_every_trial_equally():
    # Neuron a: 3 trials at 0; neuron b: 1 trial at 4 -> pooled mean 1.
    df = expanded([("a", [0.0, 0.0], 1)] * 3 + [("b", [4.0, 4.0], 1)])
    mean, sem = pooledTrialsMeanTrace(df)
    np.testing.assert_allclose(mean, [1.0, 1.0])
    np.testing.assert_allclose(sem, [1.0, 1.0])     # std([0,0,0,4], ddof=1)/2


def test_pooled_sem_counts_only_rows_with_data():
    df = expanded([("a", [1.0, 2.0], 1), ("a", [3.0, 4.0], 1), ("b", [5.0], 1)])
    mean, sem = pooledTrialsMeanTrace(df)
    np.testing.assert_allclose(mean, [3.0, 3.0])
    np.testing.assert_allclose(sem, [np.std([1, 3, 5], ddof=1)/np.sqrt(3), 1.0])
    mean, _ = pooledTrialsMeanTrace(df, min_trial_fraction=0.9)
    assert np.isnan(mean[1])


# --- RealTimeTuningPlot -------------------------------------------------------
def _drawRegion(plotter, quantile_idx=3):
    pref = expanded([("a", [0.0, 1.0, 2.0, 3.0], 3), ("b", [0.0, 1.0, 2.0, 3.0], 3)])
    anti = expanded([("a", [0.0, 0.5, 1.0, 1.5], 2), ("b", [0.0, 0.5, 1.0, 1.5], 3)])
    plotter.startNewBrainRegion("M2")
    plotter.plot(None, pd.concat([pref, anti]), pref, anti, pref.iloc[:0],
                 2, 1.0, 1.0, 0, quantile_idx=quantile_idx)
    plotter.finishBrainRegion()


def test_plotter_labels_lines_and_marks_the_decision():
    plotter = RealTimeTuningPlot("lbl", "Sampling", sampling_rate=10,
                                 save_name="x")
    _drawRegion(plotter)
    ax = plotter._axs[0]
    labels = [l.get_label() for l in ax.get_lines() if not l.get_label().startswith("_")]
    assert labels == ["Slow Preferred (#N=1.0)", "Slow Anti-Preferred (#N=1.0)"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Sampling onset",
                                                           "Slow decision"]
    # Sampling onset at sample 1, median movement onset at sample 3 (10 Hz)
    np.testing.assert_allclose(ax.get_xticks(), [0.0, 0.2])
    assert ax.get_lines()[0].get_linestyle() == "-"


def test_plotter_pools_trials_when_asked():
    lines = {}
    for pool in (False, True):
        plotter = RealTimeTuningPlot("lbl", "Sampling", 10, save_name="x",
                                     pool_trials=pool)
        pref = expanded([("a", [0.0, 0.0], 1)] * 3 + [("b", [4.0, 4.0], 1)])
        plotter.startNewBrainRegion("M2")
        plotter.plot(None, pref, pref, pref.iloc[:0], pref.iloc[:0],
                     2, 1.0, 1.0, 0, quantile_idx=3)
        lines[pool] = plotter._axs[0].get_lines()[0].get_ydata()
    np.testing.assert_allclose(lines[False], [2.0, 2.0])
    np.testing.assert_allclose(lines[True], [1.0, 1.0])


def test_strategy_colours_and_dashed_mismatch():
    plotter = RealTimeTuningPlot("lbl", "Sampling", 10, save_name="x",
                                 color_by_strategy=True)
    pref = expanded([("a", [0.0, 1.0], 1)])
    for q in (1, 3):
        plotter.startNewBrainRegion("M2")
        plotter.plot(None, pref, pref, pref, pref.iloc[:0], 1, 1.0, 1.0, 0,
                     quantile_idx=q)
    lines = plotter._axs[0].get_lines()
    assert [(l.get_color(), l.get_linestyle()) for l in lines] == \
        [("red", "-"), ("red", "--"), ("yellow", "-"), ("yellow", "--")]


def test_mfc_on_top_lfc_below_with_the_papers_names():
    plotter = RealTimeTuningPlot("lbl", "Sampling", 10, save_name="x")
    pref = expanded([("a", [0.0, 1.0], 1)])
    for region in ("ALM", "M2"):                    # the order regions arrive in
        plotter.startNewBrainRegion(region)
        plotter.plot(None, pref, pref, pref.iloc[:0], pref.iloc[:0], 1, 1.0, 1.0, 0,
                     quantile_idx=3)
        plotter.finishBrainRegion()
    assert [ax.get_title() for ax in plotter._axs] == ["MFC - lbl", "LFC - lbl"]


def test_fast_is_dotted_and_movement_alignment_has_one_tick():
    plotter = RealTimeTuningPlot("lbl", "Movement to Lateral Port",
                                 sampling_rate=10, save_name="x")
    _drawRegion(plotter, quantile_idx=1)
    ax = plotter._axs[0]
    assert ax.get_lines()[0].get_linestyle() == ":"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Movement onset"]


def test_saves_only_when_asked(tmp_path):
    plotter = RealTimeTuningPlot("lbl", "Sampling", 10, save_name="s12x")
    _drawRegion(plotter)
    plotter.showAndSave(str(tmp_path), save_fig=False)
    assert not (tmp_path / "RT_Stats").exists()
    plotter = RealTimeTuningPlot("lbl", "Sampling", 10, save_name="s12x")
    _drawRegion(plotter)
    plotter.showAndSave(str(tmp_path), save_fig=True)
    assert (tmp_path / "RT_Stats" / "traces" / "s12x.svg").is_file()


# --- subpoptraces -------------------------------------------------------------
def test_choice_stats_keeps_significant_unsplit_choice_rows():
    df = pd.DataFrame({
        "long_trace_id": ["a", "a", "b", "c", "d"],
        "data_col": ["ChoiceLeft", "ChoiceLeft", "ChoiceLeft", "PrevChoiceLeft", "ChoiceLeft"],
        "DVstr": [None, "Easy", None, None, None],
        "pval": [0.01, 0.01, 0.2, 0.01, 0.05]})
    assert list(sp.choiceStats(df, pval=0.05).long_trace_id) == ["a", "d"]


def test_first_and_last_sampling_bins():
    assert (sp.FIRST_SAMPLING_BIN, sp.LAST_SAMPLING_BIN) == (1, 5)


def test_movement_choice_neurons_need_both():
    er = [(0, 2), (3, 29), (30, 32)]
    mf = pd.DataFrame({"BrainRegion": 15, "trace_id": ["early", "late", "late_nc", "none"],
                       "max_idxs": [[5, 6, 7], [28, 29, 31], [29, 30, 31], []]})
    assert sp.movementChoiceNeurons(mf, er, {"early", "late"}) == {"late"}


def test_figure_functions_pick_their_groups(monkeypatch):
    drawn, expanded_with = [], []
    monkeypatch.setattr(sp, "realTimeExpanded",
                        lambda df, ids, align_epoch: expanded_with.append((set(ids), align_epoch)) or "EXP")
    monkeypatch.setattr(sp, "plotSubpopulationTraces",
                        lambda exp, stats, ids, label, name, align, *a, **kw:
                        drawn.append((set(ids), name, align, kw)))
    er = [(0, 2), (3, 29), (30, 32)]          # bins at 0, 3, 8.4, ... 24.6, 30, 33
    iqr = pd.DataFrame({"trace_id": ["e", "l", "e_nc"],
                        "q1_prcnt": [12.0, 80.0, 12.0], "med_prcnt": [30.0, 85.0, 30.0],
                        "q3_prcnt": [40.0, 88.0, 40.0]})
    stats = pd.DataFrame({"long_trace_id": ["e", "l"]})
    src = pd.DataFrame({"acq_sampling_rate": [30.0, 30.1, 29.9]})
    src = pd.DataFrame({"acq_sampling_rate": [30.0, 30.1, 29.9], "ChoiceCorrect": [1, 0, 1]})
    groups = sp.plotSequenceEndsTraces(src, stats, iqr, er, 0.05, False, None)
    assert groups == {1: {"e"}, 5: {"l"}}
    assert expanded_with == [({"e", "l"}, "Sampling")]
    assert [(ids, align, kw) for ids, _, align, kw in drawn] == \
        [({"e"}, "Sampling", {}), ({"l"}, "Sampling", {})]

    drawn.clear(); expanded_with.clear()
    mf = pd.DataFrame({"BrainRegion": 15, "trace_id": ["e", "l"],
                       "max_idxs": [[5, 6, 7], [28, 29, 31]]})
    assert sp.plotMovementNeuronTraces(src, stats, mf, er, 0.05, False, None) == {"l"}
    assert expanded_with == [({"l"}, "Movement to Lateral Port")]
    assert drawn[0][1] == sp.S12J_NAME
    assert drawn[0][3] == {"color_by_strategy": True}     # fast red, slow yellow
    drawn.clear()
    sp.plotMovementNeuronTraces(src, stats, mf, er, 0.05, False, None,
                                color_by_strategy=False)
    assert drawn[0][3] == {"color_by_strategy": False}


def test_correct_only_keeps_correct_trials_and_renames(monkeypatch):
    seen, drawn = [], []
    monkeypatch.setattr(sp, "realTimeExpanded",
                        lambda df, ids, align_epoch: seen.append(list(df.ChoiceCorrect)) or "EXP")
    monkeypatch.setattr(sp, "plotSubpopulationTraces",
                        lambda exp, stats, ids, label, name, *a, **kw: drawn.append((label, name)))
    er = [(0, 2), (3, 29), (30, 32)]
    src = pd.DataFrame({"acq_sampling_rate": [30.0] * 3, "ChoiceCorrect": [1, 0, 1]})
    stats = pd.DataFrame({"long_trace_id": ["l"]})
    mf = pd.DataFrame({"BrainRegion": 15, "trace_id": ["l"], "max_idxs": [[28, 29, 31]]})
    sp.plotMovementNeuronTraces(src, stats, mf, er, 0.05, False, None, correct_only=True)
    assert seen == [[1, 1]]
    assert drawn == [(sp.S12J_LABEL + " (correct trials)", sp.S12J_NAME + "_correct_trials")]
    iqr = pd.DataFrame({"trace_id": ["l"], "q1_prcnt": [80.0], "med_prcnt": [85.0],
                        "q3_prcnt": [88.0]})
    drawn.clear()
    sp.plotSequenceEndsTraces(src, stats, iqr, er, 0.05, False, None, correct_only=True)
    assert [name for _, name in drawn] == [n + "_correct_trials" for _, _, n in sp.S12D_GROUPS]


def test_subpopulation_traces_pool_trials_by_default(monkeypatch):
    made = []
    monkeypatch.setattr(sp, "RealTimeTuningPlot",
                        lambda **kw: made.append(kw) or object())
    monkeypatch.setattr(sp, "samplingSlowVsFast", lambda **kw: None)
    sp.plotSubpopulationTraces(pd.DataFrame(), pd.DataFrame({"long_trace_id": []}),
                               set(), "l", "n", "Sampling", 30.0, 0.05, False, None)
    assert made[0]["pool_trials"] is True and made[0]["color_by_strategy"] is False
