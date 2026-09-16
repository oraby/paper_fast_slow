'''Tests for the active-neuron counts behind Figures S11A and S11B.

Reproduced while extracting: the same early and late count frames as the
notebook cells produced.

What these pin is the window, because the two panels specify it differently on
purpose: the early one is a fixed number of seconds from the trial's start, the
late one a percentage of the trial measured from its end. Mixing those up
changes what "active" means without changing the shape of the output.
'''
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ..activeneuroncount import (ACQ_RATE, TIME_BEFORE_SAMPLING,
                                 _commonNumActiveNeurons,
                                 corrTrialDurActivityCommon)

KEY = "k_"


def neuronRows(peaks_sec, trial_dur_sec=1.0, trial=0, session="s1", br=15,
               trace_len=60):
    """One row per neuron in a trial, each peaking at the given time."""
    rows = []
    for i, peak in enumerate(peaks_sec):
        trace = np.zeros(trace_len)
        idx = int(round((TIME_BEFORE_SAMPLING + peak) * ACQ_RATE))
        trace[max(idx, 1) - 1:idx + 2] = [0.0, 10.0, 0.0]
        rows.append({"BrainRegion": br, "ShortName": session, "TrialNumber": trial,
                     "trace_id": f"n{i}", "trial_dur_sec": trial_dur_sec,
                     "ChoiceCorrect": 1.0, "DVstr": "Hard",
                     KEY + "max_pos_sec": TIME_BEFORE_SAMPLING + peak,
                     "trace": trace, KEY + "trace": trace})
    return pd.DataFrame(rows)


def counts(df, **kwargs):
    kwargs.setdefault("key_prefix", KEY)
    return _commonNumActiveNeurons(df, total_neurons_count=len(df), **kwargs)


def test_the_early_window_runs_from_before_onset_to_a_fixed_cutoff():
    # Peaks at 0.05 s and 0.2 s are inside a 0.3 s window; 0.8 s is not.
    df = neuronRows([0.05, 0.2, 0.8])
    _, active, within, _ = counts(df, cut_before=TIME_BEFORE_SAMPLING,
                                  cut_before_how="FIX", cut_after=0.3,
                                  cut_after_how="FIX", cut_from="START")
    assert active == 3          # every neuron in the trial
    assert within == 2


def test_the_late_window_is_a_share_of_the_trial_measured_from_its_end():
    # A 1 s trial, last 30%: peaks at 0.8 s and 0.95 s count, 0.2 s does not.
    df = neuronRows([0.2, 0.8, 0.95], trial_dur_sec=1.0)
    _, _, within, _ = counts(df, cut_before=0.3, cut_before_how="PRCNT",
                             cut_after=0.1, cut_after_how="FIX", cut_from="END")
    assert within == 2


def test_the_late_window_scales_with_the_trial():
    """The same peak is late in a short trial and early in a long one."""
    short = neuronRows([0.8], trial_dur_sec=1.0)
    long = neuronRows([0.8], trial_dur_sec=4.0)
    late = dict(cut_before=0.3, cut_before_how="PRCNT", cut_after=0.1,
                cut_after_how="FIX", cut_from="END")
    assert counts(short, **late)[2] == 1
    assert counts(long, **late)[2] == 0


def test_a_peak_just_after_the_cutoff_is_excluded():
    df = neuronRows([0.31])
    _, _, within, _ = counts(df, cut_before=TIME_BEFORE_SAMPLING,
                             cut_before_how="FIX", cut_after=0.3,
                             cut_after_how="FIX", cut_from="START")
    assert within == 0


def test_a_percentage_window_before_the_start_is_refused():
    """There is nothing to take a percentage of before the trial begins."""
    df = neuronRows([0.1])
    with pytest.raises(AssertionError, match="Can't cut before start"):
        counts(df, cut_before=0.3, cut_before_how="PRCNT", cut_after=0.3,
               cut_after_how="FIX", cut_from="START")


def test_a_percentage_window_after_the_end_is_refused():
    df = neuronRows([0.1])
    with pytest.raises(AssertionError, match="Can't cut after end"):
        counts(df, cut_before=0.3, cut_before_how="FIX", cut_after=0.3,
               cut_after_how="PRCNT", cut_from="END")


def test_one_trial_must_have_one_duration():
    df = pd.concat([neuronRows([0.1], trial_dur_sec=1.0),
                    neuronRows([0.1], trial_dur_sec=2.0)])
    with pytest.raises(AssertionError):
        counts(df, cut_before=TIME_BEFORE_SAMPLING, cut_before_how="FIX",
               cut_after=0.3, cut_after_how="FIX", cut_from="START")


def test_the_frame_has_one_row_per_trial_and_names_the_window():
    df = pd.concat([neuronRows([0.05, 0.9], trial=0),
                    neuronRows([0.05, 0.1], trial=1)])
    out = corrTrialDurActivityCommon(df, cut_before=TIME_BEFORE_SAMPLING,
                                     cut_before_how="FIX", cut_after=0.3,
                                     cut_after_how="FIX", cut_from="START",
                                     key_prefix=KEY)
    assert len(out) == 2
    assert "trial_start_max_active_count" in out.columns
    assert list(out.trial_start_max_active_count) == [1, 2]
    assert list(out.total_count) == [2, 2]      # session neurons, both trials


def test_the_late_frame_names_its_columns_for_the_end():
    df = neuronRows([0.9])
    out = corrTrialDurActivityCommon(df, cut_before=0.3, cut_before_how="PRCNT",
                                     cut_after=0.1, cut_after_how="FIX",
                                     cut_from="END", key_prefix=KEY)
    assert "trial_end_max_active_count" in out.columns


def test_total_count_is_the_sessions_neurons_not_the_trials():
    """A neuron missing from one trial still counts in the denominator."""
    df = pd.concat([neuronRows([0.05, 0.05], trial=0),
                    neuronRows([0.05], trial=1)])
    out = corrTrialDurActivityCommon(df, cut_before=TIME_BEFORE_SAMPLING,
                                     cut_before_how="FIX", cut_after=0.3,
                                     cut_after_how="FIX", cut_from="START",
                                     key_prefix=KEY)
    assert list(out.total_count) == [2, 2]
