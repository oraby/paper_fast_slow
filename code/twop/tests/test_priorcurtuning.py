'''Tests for the priors-vs-current tuning Venn (Figure 6E).

Reproduced while extracting: the same percentages as the notebook cell for
both epochs, all three panels.

Pinned here: prior-only and current-only exclude the overlap, every session
counts once whatever its size, and the Venn's circle areas are *ratios to the
overlap* while the labels carry the percentages. That last one looks like a bug
until you read it twice, so a test says it is deliberate.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ...common.definitions import BrainRegion
from ..priorcurtuning import (HATCH_LINEWIDTH, POOLED_LABEL,
                              _plotBrainRegionTuning, plotPriorCurrentTuning,
                              tuningPercentages)

MFC = int(BrainRegion.M2_Bi)
LFC = int(BrainRegion.ALM_Bi)
NAN = np.nan


def neurons(rows, session="s1", br=MFC):
    """rows: list of (prior_value, current_value); NaN means untuned."""
    return pd.DataFrame([
        {"trace_id": f"{session}_n{i}", "ShortName": session, "BrainRegion": br,
         "PrevChoiceLeft": prior, "ChoiceLeft": current}
        for i, (prior, current) in enumerate(rows)])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_the_four_groups_partition_the_session():
    df = neurons([(1.0, NAN),      # prior only
                  (NAN, 1.0),      # current only
                  (1.0, 1.0),      # both
                  (NAN, NAN)])     # neither
    row = tuningPercentages(df)[0]
    assert (row["prior"], row["current"], row["both"], row["none"]) == \
        (25.0, 25.0, 25.0, 25.0)
    assert row["prior"] + row["current"] + row["both"] + row["none"] == 100.0


def test_prior_and_current_exclude_the_overlap():
    df = neurons([(1.0, 1.0), (1.0, 1.0), (1.0, NAN), (NAN, NAN)])
    row = tuningPercentages(df)[0]
    assert row["both"] == 50.0
    assert row["prior"] == 25.0          # not 75%
    assert row["current"] == 0.0


def test_any_prior_variable_counts():
    df = pd.DataFrame([{"trace_id": "n1", "ShortName": "s1", "BrainRegion": MFC,
                        "PrevChoiceLeft": NAN, "PrevChoiceCorrect": 1.0,
                        "ChoiceLeft": NAN}])
    assert tuningPercentages(df)[0]["prior"] == 100.0


def test_id_columns_are_not_mistaken_for_variables():
    df = neurons([(NAN, NAN)])
    assert tuningPercentages(df)[0]["none"] == 100.0


def test_each_session_contributes_one_percentage():
    small = neurons([(1.0, 1.0), (NAN, NAN)], session="s1")             # 50% both
    large = neurons([(1.0, 1.0)] * 9 + [(NAN, NAN)], session="s2")      # 90% both
    rows = tuningPercentages(pd.concat([small, large]))
    assert [r["both"] for r in rows] == [50.0, 90.0]


def test_duplicate_trace_ids_are_rejected():
    df = pd.concat([neurons([(1.0, 1.0)]), neurons([(1.0, 1.0)])])
    with pytest.raises(AssertionError, match="Trace ID mismatch"):
        tuningPercentages(df)


def test_means_and_sems_are_across_sessions():
    df = pd.concat([neurons([(1.0, 1.0), (NAN, NAN)], session="s1"),
                    neurons([(1.0, 1.0), (1.0, NAN), (NAN, 1.0), (NAN, NAN)],
                            session="s2")])
    means, sems = _plotBrainRegionTuning(df, "Sampling", "MFC")
    assert means["both"] == pytest.approx(37.5)        # (50 + 25) / 2
    assert sems["both"] == pytest.approx(12.5)


def test_labels_show_percentages_while_areas_are_ratios_to_the_overlap():
    df = neurons([(1.0, NAN), (1.0, NAN), (NAN, 1.0), (1.0, 1.0)])
    means, _ = _plotBrainRegionTuning(df, "Sampling", "MFC")
    assert (means["prior"], means["current"], means["both"]) == (50.0, 25.0, 25.0)
    labels = {t.get_text().split("%")[0] for t in plt.gca().texts if "%" in t.get_text()}
    assert {"50.0", "25.0"} <= labels        # the real percentages, not 2.0 / 1.0


def test_the_hatch_width_is_restored():
    before = mpl.rcParams["hatch.linewidth"]
    assert before != HATCH_LINEWIDTH
    _plotBrainRegionTuning(neurons([(1.0, NAN), (NAN, 1.0), (1.0, 1.0)]),
                           "Sampling", "MFC")
    assert mpl.rcParams["hatch.linewidth"] == before


def test_the_pooled_panel_comes_first_then_one_per_region():
    df = pd.concat([neurons([(1.0, NAN), (NAN, 1.0), (1.0, 1.0)], br=MFC),
                    neurons([(1.0, 1.0), (NAN, 1.0), (1.0, NAN)],
                            session="s2", br=LFC)])
    res = plotPriorCurrentTuning(df, epoch_str="Sampling")
    # Regions follow the BrainRegion enum, where ALM_Bi (6) precedes M2_Bi (15).
    assert list(res) == [POOLED_LABEL, "ALM", "M2"]


def test_saved_under_the_published_name(tmp_path):
    _plotBrainRegionTuning(neurons([(1.0, NAN), (NAN, 1.0), (1.0, 1.0)]),
                           "Feedback Start", "M2", save_figs=True,
                           fig_save_prefix=str(tmp_path))
    assert (tmp_path / "PriorCurrentTuning" /
            "M2_Feedback Start_prior_current_tuning.svg").exists()


def test_saving_needs_a_prefix():
    with pytest.raises(AssertionError, match="fig_save_prefix"):
        _plotBrainRegionTuning(neurons([(1.0, NAN), (NAN, 1.0), (1.0, 1.0)]),
                               "Sampling", "M2", save_figs=True)
