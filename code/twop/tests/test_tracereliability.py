'''Tests for the per-neuron activity criterion (Figure S9A, and much downstream).

2pAnalysis and TwoPLoad each carried a copy of these ten functions; the module
they now share is TwoPLoad's newer one. Reproduced while extracting: both
notebooks give the same max_firing frames as before.

What matters most here is that the threshold is **relative to the neuron's own
trials** — the fifth-percentile spread times three — so a quiet neuron is not
judged against a loud one. Everything downstream (the active sets behind
Figure 4G, the reliability CDF, the sequence panels) rests on it.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ...common.definitions import BrainRegion
from ..tracereliability import (DEFAULT_ABOVE_LOWEST_PERCENTILE_ZSCORES,
                                DEFAULT_LOWEST_PERCENTILE, DecideNeurons,
                                StdDistCollector, exampleNeuron,
                                loopSessionsExampleNeurons, separateNeuron,
                                traceValidInvalid)

MFC = int(BrainRegion.M2_Bi)
LFC = int(BrainRegion.ALM_Bi)


def trials(traces, session="s1", br=MFC, quantiles=None, start=None, end=None):
    """One row per trial; traces is a list of per-trial arrays for neuron 7."""
    rows = []
    for i, trace in enumerate(traces):
        trace = np.asarray(trace, dtype=float)
        rows.append({"TrialNumber": i, "ShortName": session, "BrainRegion": br,
                     "Layer": "L23", "ChoiceLeft": i % 2, "ChoiceCorrect": 1.0,
                     "traces_sets": {"neuronal": {7: trace}},
                     "trace_start_idx": 0 if start is None else start,
                     "trace_end_idx": (len(trace) - 1) if end is None else end,
                     "epochs_ranges": [(0, 1), (2, 5)], "epochs_names": ["a", "b"]})
        if quantiles is not None:
            rows[-1]["quantile_idx"] = quantiles[i]
    return pd.DataFrame(rows)


def flat(n=8, level=0.0):
    return np.full(n, level)


def wave(n=8, amplitude=1.0):
    return amplitude * np.sin(np.linspace(0, 2 * np.pi, n))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_the_threshold_is_the_low_percentile_times_the_multiplier():
    df = trials([wave(amplitude=a) for a in [1, 2, 3, 4, 5]])
    res = DecideNeurons(smooth_sigma=0.01).getThresholds(df, 7)
    expected = np.percentile(res.all_traces_std, DEFAULT_LOWEST_PERCENTILE) \
        * DEFAULT_ABOVE_LOWEST_PERCENTILE_ZSCORES
    assert res.threshold == pytest.approx(expected)
    assert res.lower_percentile_std == pytest.approx(
        np.percentile(res.all_traces_std, DEFAULT_LOWEST_PERCENTILE))


def test_a_quiet_neuron_is_judged_against_its_own_trials():
    """The same shape at 1/100th the scale still has active trials."""
    loud = trials([wave(amplitude=a) for a in [1, 1, 1, 10]])
    quiet = trials([wave(amplitude=a / 100) for a in [1, 1, 1, 10]])
    active = lambda df: traceValidInvalid(7, df,
                                          decide_neurons=DecideNeurons(0.01)).prcnt_valid
    assert active(loud) == active(quiet)


def test_trials_above_the_threshold_are_the_active_ones():
    df = trials([flat(), flat(), flat(), wave(amplitude=5)])
    res = traceValidInvalid(7, df, decide_neurons=DecideNeurons(0.01))
    assert list(res.valid_traces_idxs) == [False, False, False, True]
    assert res.prcnt_valid == 25.0


def test_peak_position_and_value_come_from_the_active_trials():
    peak_at_two = np.array([0., 0., 9., 0., 0., 0.])
    df = trials([flat(6), peak_at_two])
    res = traceValidInvalid(7, df, decide_neurons=DecideNeurons(0.01))
    assert list(res.max_idxs) == [2]
    assert res.max_vals[0] == pytest.approx(9.0, rel=1e-3)


def test_a_query_narrows_the_active_trials_without_changing_the_threshold():
    df = trials([wave(amplitude=5), wave(amplitude=5), flat(), flat()])
    df["ChoiceCorrect"] = [1.0, 1.0, 0.0, 0.0]
    both = traceValidInvalid(7, df, decide_neurons=DecideNeurons(0.01))
    correct = traceValidInvalid(7, df, "ChoiceCorrect == 1",
                                decide_neurons=DecideNeurons(0.01))
    assert both.threshold == pytest.approx(correct.threshold)
    assert list(correct.valid_traces_idxs) == [True, True, False, False]


def test_only_the_rows_window_is_measured():
    """A spike outside trace_start_idx..trace_end_idx must not count."""
    spike_late = np.array([0., 0., 0., 0., 50.])
    windowed = trials([flat(5), spike_late], end=3)
    res = traceValidInvalid(7, windowed, decide_neurons=DecideNeurons(0.01))
    assert res.prcnt_valid == 0.0          # the spike is outside the window


def test_unequal_trial_lengths_still_work():
    df = trials([flat(6), wave(9, amplitude=5)])
    res = traceValidInvalid(7, df, decide_neurons=DecideNeurons(0.01))
    assert isinstance(res.valid_trace_smoothed, list)
    assert len(res.max_idxs) == 1


def test_the_collector_builds_one_row_per_neuron():
    df = trials([flat(), wave(amplitude=5), wave(amplitude=5)])
    collector = StdDistCollector(DecideNeurons(0.01))
    collector.track(7, df)
    out = collector.frame()
    assert len(out) == 1
    row = out.iloc[0]
    assert row.trace_id == 7
    assert row.len_all == 3 and row.len_valid == 2
    assert row.prcnt_valid == pytest.approx(200 / 3)
    assert row.std_threshold > 0


def test_the_collector_records_quantiles_when_present():
    """Active trials keep their strategy; the counts cover every trial."""
    df = trials([wave(amplitude=5), wave(amplitude=5), flat()],
                quantiles=[1, 3, 3])
    collector = StdDistCollector(DecideNeurons(0.01))
    collector.track(7, df)
    row = collector.frame().iloc[0]
    assert list(row.active_quantile_idx) == [1, 3]
    assert row.quantile_idxs_count == {1: 1, 3: 2}


def test_without_quantiles_those_fields_are_nan():
    collector = StdDistCollector(DecideNeurons(0.01))
    collector.track(7, trials([wave(amplitude=5)] * 2))
    row = collector.frame().iloc[0]
    assert np.isnan(row.active_quantile_idx) and np.isnan(row.quantile_idxs_count)


def test_two_collectors_do_not_share_rows():
    """The notebooks' module-level res_dict had to be reset by hand."""
    df = trials([wave(amplitude=5)] * 2)
    first, second = StdDistCollector(DecideNeurons(0.01)), StdDistCollector(DecideNeurons(0.01))
    first.track(7, df)
    second.track(7, df)
    assert len(first.frame()) == len(second.frame()) == 1


def test_flattening_the_leading_decay_lowers_the_measured_spread():
    """Feedback traces open on a decay; flattening it stops that counting."""
    # Decays, then turns upward at index 3: everything before that is held flat.
    decaying = np.array([10., 6., 2., 0., 4., 8.])
    df = trials([decaying])
    plain = DecideNeurons(0.01).getThresholds(df, 7)
    flattened = DecideNeurons(0.01).getThresholds(df, 7,
                                                  disable_first_neg_deflect=True)
    assert flattened.all_traces_std[0] < plain.all_traces_std[0]


def addNeuron(df, trace_id, trace):
    for _, row in df.iterrows():
        row["traces_sets"]["neuronal"][trace_id] = trace


def test_separate_neuron_keeps_the_trials_and_renames_the_trace():
    df = trials([flat(), flat()])
    addNeuron(df, 9, flat())                           # a second neuron
    out = separateNeuron(7, df, new_trace_id="s1_7")
    assert len(out) == 2
    assert list(out.iloc[0].traces_sets["neuronal"]) == ["s1_7"]


def test_every_neuron_is_processed_when_the_fraction_is_one():
    df = trials([flat(), flat()])
    addNeuron(df, 9, flat())
    seen = []
    exampleNeuron(df, processFn=lambda tid, d, **kw: seen.append(tid),
                  df_query="", random_fraction=1)
    assert sorted(seen) == ["s1_7", "s1_9"]


def test_the_loop_keeps_only_l23_mfc_and_lfc():
    mfc = trials([wave(amplitude=5)] * 2, session="mfc", br=MFC)
    lfc = trials([wave(amplitude=5)] * 2, session="lfc", br=LFC)
    other = trials([wave(amplitude=5)] * 2, session="other", br=9)
    l5 = trials([wave(amplitude=5)] * 2, session="l5", br=MFC)
    l5["Layer"] = "L5"
    collector = StdDistCollector(DecideNeurons(0.01))
    loopSessionsExampleNeurons(pd.concat([mfc, lfc, other, l5]),
                               processFn=collector.track, random_fraction=1,
                               pdf_str="x", save_figs=False, seed=1)
    assert sorted(collector.frame().ShortName) == ["lfc", "mfc"]


def test_saving_the_pdfs_needs_a_prefix():
    with pytest.raises(AssertionError, match="fig_save_prefix"):
        loopSessionsExampleNeurons(trials([flat(), flat()]),
                                   processFn=lambda *a, **k: None,
                                   random_fraction=1, pdf_str="x",
                                   save_figs=True, seed=1)
