import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[3]))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.twop.sgfneurons import (  # noqa: E402
    BRACKET_TICK,
    CHOICE_DATA_COL,
    LFC,
    MFC,
    SESSION_COLUMNS,
    _topInFig,
    dataColLabel,
    drawRegionSigBracket,
    formatComparison,
    perSessionSgfPrcnt,
    regionComparison,
    significanceLabel,
)
from code.twop.statstest import (  # noqa: E402
    TEST_MWU,
    TEST_TTEST_STUDENT,
    pStars,
)

M2_BI, ALM_BI = 15, 6  # BrainRegion enum values -> MFC / LFC
PVAL = 0.05


def _neuron(sess, br, pval, data_col="ChoiceLeft", prior=None, dv=np.nan,
            roc_left=True):
    return {"ShortName": sess, "BrainRegion": br, "data_col": data_col,
            "prior_data_col": prior, "DVstr": dv, "pval": pval,
            "IsROCLeftTuend": roc_left}


def _shortlong_df():
    """2 sessions/region, 4 neurons each, with a known number significant."""
    rows = []
    # MFC session A: 2 of 4 significant (one left-tuned, one right-tuned).
    rows += [_neuron("A", M2_BI, 0.01, roc_left=True),
             _neuron("A", M2_BI, 0.02, roc_left=False),
             _neuron("A", M2_BI, 0.5), _neuron("A", M2_BI, 0.9)]
    # MFC session B: 1 of 4.
    rows += [_neuron("B", M2_BI, 0.001)] + [_neuron("B", M2_BI, 0.4)] * 3
    # LFC session C: 3 of 4, LFC session D: 4 of 4.
    rows += [_neuron("C", ALM_BI, 0.01)] * 3 + [_neuron("C", ALM_BI, 0.6)]
    rows += [_neuron("D", ALM_BI, 0.02)] * 4
    # Rows the pie charts (and we) must ignore: a DV split, a prior, and the
    # other tuning columns that live in the same shortlong_df.
    rows += [_neuron("A", M2_BI, 0.001, dv="DV>0"),
             _neuron("A", M2_BI, 0.001, prior="PrevChoiceCorrect")]
    rows += [_neuron("A", M2_BI, 0.001, data_col="OtherTuning")] * 4
    return pd.DataFrame(rows)


def test_per_session_prcnt_matches_the_pie_chart_recipe():
    sess_df = perSessionSgfPrcnt(_shortlong_df(), pval=PVAL)
    assert list(sess_df.columns) == SESSION_COLUMNS
    sess_df = sess_df.set_index("ShortName")

    # The DVstr and prior_data_col rows must not inflate A's denominator.
    assert sess_df.loc["A", "n_neurons"] == 4
    assert sess_df.loc["A", "n_sgf"] == 2
    assert sess_df.loc["A", "prcnt_sgf"] == pytest.approx(50.0)
    assert sess_df.loc["B", "prcnt_sgf"] == pytest.approx(25.0)
    assert sess_df.loc["C", "prcnt_sgf"] == pytest.approx(75.0)
    assert sess_df.loc["D", "prcnt_sgf"] == pytest.approx(100.0)
    assert list(sess_df.loc[["A", "B"], "BrainRegion"]) == [MFC, MFC]
    assert list(sess_df.loc[["C", "D"], "BrainRegion"]) == [LFC, LFC]


def test_per_session_prcnt_splits_by_selectivity_direction():
    df = _shortlong_df()
    left = perSessionSgfPrcnt(df, PVAL, check_roc_left=True).set_index("ShortName")
    right = perSessionSgfPrcnt(df, PVAL, check_roc_left=False).set_index("ShortName")
    # Session A's two significant neurons are one per direction; the split
    # wedges must add back up to the combined 50%.
    assert left.loc["A", "n_sgf"] == 1
    assert right.loc["A", "n_sgf"] == 1
    assert (left.loc["A", "prcnt_sgf"] + right.loc["A", "prcnt_sgf"]
            == pytest.approx(50.0))


def test_per_session_prcnt_ignores_the_other_tuning_columns():
    # Session A also has 4 significant rows for another data_col; none of them
    # may reach the ChoiceLeft numerator or denominator.
    sess_df = perSessionSgfPrcnt(_shortlong_df(), PVAL).set_index("ShortName")
    assert sess_df.loc["A", "n_neurons"] == 4
    assert sess_df.loc["A", "n_sgf"] == 2


def test_data_col_label_falls_back_to_the_raw_name():
    assert dataColLabel(CHOICE_DATA_COL) == "Cur. Direction"
    assert dataColLabel("Whatever") == "Whatever"


def test_region_comparison_means_sessions_and_reports_the_gate():
    comp_df = regionComparison(_shortlong_df(), pval=PVAL)
    assert len(comp_df) == 1  # one planned comparison, no multiplicity
    comp = comp_df.iloc[0]
    assert comp.data_col == CHOICE_DATA_COL
    assert comp.MFC_mean == pytest.approx(37.5)   # (50 + 25) / 2
    assert comp.LFC_mean == pytest.approx(87.5)   # (75 + 100) / 2
    assert comp.MFC_n_sessions == 2 and comp.LFC_n_sessions == 2
    assert (comp.MFC_n, comp.MFC_total) == (3, 8)
    assert (comp.LFC_n, comp.LFC_total) == (7, 8)
    # 2 sessions per region is below Shapiro-Wilk's minimum -> non-parametric.
    assert np.isnan(comp[f"{MFC}_shapiro_pval"])
    assert not comp.is_normal
    assert comp.test == TEST_MWU


def test_region_comparison_uses_the_t_test_when_sessions_look_normal():
    rng = np.random.RandomState(0)
    rows = []
    for br, prefix, sgf_rate in [(M2_BI, "M", 0.2), (ALM_BI, "L", 0.6)]:
        for sess in range(10):
            n_sgf = int(round(20 * sgf_rate + rng.normal(0, 1.5)))
            for idx in range(20):
                rows.append(_neuron(f"{prefix}{sess}", br,
                                    0.01 if idx < n_sgf else 0.5))
    comp = regionComparison(pd.DataFrame(rows), pval=PVAL).iloc[0]
    assert comp.is_normal
    assert comp.test == TEST_TTEST_STUDENT
    assert comp.pval == comp.ttest_pval
    assert comp.pval < 0.05
    assert not np.isnan(comp.mwu_pval)


# ---------------------------------------------------------------------------
# Significance bracket
# ---------------------------------------------------------------------------
def _comparison(pval=0.00478, test=TEST_TTEST_STUDENT):
    return pd.DataFrame([{"pval": pval, "test": test}])


def test_significance_label_has_stars_pvalue_and_test():
    label = significanceLabel(_comparison())
    assert label == "** p=0.00478 (t-test (Student))"
    assert significanceLabel(_comparison(0.4, TEST_MWU)) == \
        "n.s. p=0.4 (MannWhitneyU)"
    assert significanceLabel(_comparison(), show_test=False) == "** p=0.00478"


def test_p_stars_thresholds():
    assert [pStars(p) for p in (0.0001, 0.005, 0.03, 0.4)] == \
        ["***", "**", "*", "n.s."]
    assert pStars(np.nan) == "n.a."


def _pie_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for ax, label in zip(axs, [LFC, MFC]):
        ax.pie([70, 30], labels=["", "30%"], startangle=90, explode=[0, 0.1])
        ax.set_title(label, y=1.1)
    return fig, axs


def test_bracket_spans_the_two_axes_and_clears_their_titles():
    import matplotlib.pyplot as plt
    fig, axs = _pie_fig()
    line, text = drawRegionSigBracket(fig, fig.axes, _comparison())

    xs, ys = line.get_xdata(), line.get_ydata()
    centres = [ax.get_position().x0 + ax.get_position().width / 2 for ax in axs]
    assert xs == pytest.approx([centres[0], centres[0], centres[1], centres[1]])
    # Flat span with the two end ticks pointing down.
    assert ys[1] == pytest.approx(ys[2])
    assert ys[0] == pytest.approx(ys[1] - BRACKET_TICK)
    assert ys[3] == pytest.approx(ys[2] - BRACKET_TICK)

    # The ticks must stop above both titles, not run through them.
    renderer = fig.canvas.get_renderer()
    title_top = max(_topInFig(fig, ax.title, renderer) for ax in axs)
    assert ys[0] > title_top
    # Label centred over the bracket, above the line.
    assert text.get_position()[0] == pytest.approx(np.mean(centres))
    assert text.get_position()[1] > ys[1]
    assert text.get_text() == significanceLabel(_comparison())
    plt.close(fig)


def test_bracket_pushes_a_colliding_suptitle_up():
    import matplotlib.pyplot as plt
    fig, _axs = _pie_fig()
    fig.suptitle("Cur. Direction", y=0.9)  # deliberately in the bracket's way
    _line, text = drawRegionSigBracket(fig, fig.axes, _comparison())

    label_top = _topInFig(fig, text, fig.canvas.get_renderer())
    assert fig._suptitle.get_position()[1] > label_top
    plt.close(fig)


def test_bracket_leaves_a_clear_suptitle_alone():
    import matplotlib.pyplot as plt
    fig, _axs = _pie_fig()
    fig.suptitle("Cur. Direction", y=1.05)  # the notebook's placement
    drawRegionSigBracket(fig, fig.axes, _comparison())
    assert fig._suptitle.get_position()[1] == pytest.approx(1.05)
    plt.close(fig)


def test_format_comparison_mentions_the_label_and_the_gate():
    text = formatComparison(regionComparison(_shortlong_df(), pval=PVAL))
    assert "Cur. Direction" in text
    assert "MFC: 37.50% +/- 12.50% SEM" in text
    assert "Normality (Shapiro-Wilk)" in text
    assert "MannWhitneyU" in text
