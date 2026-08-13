"""Significant-neuron percentages per session, and MFC vs LFC on them.

Companion to the ``# Sgf. Neurons Pie Charts`` section of ``2pAnalysis.ipynb``.
Those pie charts already show, per brain region, the **mean +/- SEM over
sessions** of the percentage of neurons significant for a ``data_col`` -- but
they run no test *between* the regions. This module recomputes exactly that
per-session percentage (:func:`perSessionSgfPrcnt` mirrors
``loopPlotPieChart``'s ``calcSessSGFMean``) and adds the MFC-vs-LFC comparison.

**One comparison, on ``ChoiceLeft`` (current direction).** The other tuning
columns the pie charts draw are deliberately not tested: a battery of region
comparisons would be a family of tests needing a multiple-comparison
correction, and only the current-direction one is the planned question here.

The comparison is normality-gated, like the movement-neuron section: Shapiro-Wilk
on each region's per-session percentages first (the regions are independent
samples of different sessions), then a two-sample t-test if both look normal
(Levene picking Student vs Welch), otherwise Mann-Whitney U. See
:func:`~.statstest.normalityGatedTest`.
"""

import numpy as np
import pandas as pd

from ..common.definitions import BrainRegion
from .seqdeviation import region_label
from .statstest import NORMALITY_ALPHA, normalityGatedTest, pStars

# Same labels region_label produces, straight from the enum's __format__.
MFC = "{}".format(BrainRegion.M2_Bi)
LFC = "{}".format(BrainRegion.ALM_Bi)
REGIONS = (MFC, LFC)

#: The one tested tuning column, and its pie-chart title.
CHOICE_DATA_COL = "ChoiceLeft"
DATA_COL_LABELS = {CHOICE_DATA_COL: "Cur. Direction"}

SESSION_COLUMNS = ["data_col", "BrainRegion", "ShortName", "n_neurons",
                   "n_sgf", "prcnt_sgf"]

# Significance-bracket geometry, in figure fractions. The bracket goes above the
# two pies' titles -- measured, not guessed, so the end ticks stop clear of the
# title text. savefig(bbox_inches="tight") and the inline backend both grow the
# canvas to include it, the way the suptitle at y=1.05 already relies on.
BRACKET_PAD = 0.02       # above the taller of the two titles
BRACKET_TICK = 0.012     # length of the downward end ticks
BRACKET_TEXT_PAD = 0.004
BRACKET_SUPTITLE_GAP = 0.02  # kept between the label and the suptitle
BRACKET_LW = 1.2


def dataColLabel(data_col):
    """Pie-chart label for a ``data_col`` (falls back to the raw name)."""
    return DATA_COL_LABELS.get(data_col, data_col)


def perSessionSgfPrcnt(shortlong_df, pval, data_col=CHOICE_DATA_COL,
                       check_roc_left=None):
    """One row per session: % of its neurons significant for ``data_col``.

    Mirrors ``calcSessSGFMean`` inside ``loopPlotPieChart``: ``DVstr``-split rows
    dropped, only the unconditioned (no ``prior_data_col``) rows kept, and the
    denominator is every neuron of that session tested for ``data_col``.

    ``check_roc_left`` selects one selectivity direction (``True`` =
    left-tuned, ``False`` = right-tuned) the way the ``show_selectivity=True``
    pie charts split their wedge; ``None`` (default) counts both, i.e. the whole
    exploded wedge.
    """
    df = shortlong_df[shortlong_df.DVstr.isnull()]
    df = df[df.prior_data_col.isnull()]
    df = df[df.data_col == data_col]

    rows = []
    for (br, sess), sess_df in df.groupby(["BrainRegion", "ShortName"]):
        sgf_idx = sess_df.pval <= pval
        if check_roc_left is not None:
            sgf_idx &= sess_df.IsROCLeftTuend == check_roc_left
        n_neurons, n_sgf = len(sess_df), int(sgf_idx.sum())
        rows.append({"data_col": data_col,
                     "BrainRegion": region_label(br),
                     "ShortName": sess,
                     "n_neurons": n_neurons,
                     "n_sgf": n_sgf,
                     "prcnt_sgf": 100 * n_sgf / n_neurons})
    return pd.DataFrame(rows, columns=SESSION_COLUMNS)


def regionComparison(shortlong_df, pval, data_col=CHOICE_DATA_COL,
                     check_roc_left=None, alpha=NORMALITY_ALPHA):
    """MFC vs LFC on the per-session significant-neuron percentages.

    A single comparison, on ``data_col`` (current direction by default) -- see
    the module docstring on why the other tuning columns are not tested.

    Returns a one-row frame with each region's mean +/- SEM over sessions, the
    pooled neuron counts, and the normality-gated test: ``MFC_shapiro_pval`` /
    ``LFC_shapiro_pval`` and ``is_normal`` (checked first), ``levene_pval``,
    then ``test``/``statistic``/``pval`` for the selected test, plus
    ``ttest_pval`` and ``mwu_pval`` for reference.
    """
    sess_df = perSessionSgfPrcnt(shortlong_df, pval, data_col,
                                 check_roc_left=check_roc_left)
    row = {"data_col": data_col, "label": dataColLabel(data_col)}
    vals = {}
    for region in REGIONS:
        br_df = sess_df[sess_df.BrainRegion == region]
        vals[region] = br_df.prcnt_sgf.dropna()
        row[f"{region}_n_sessions"] = len(vals[region])
        row[f"{region}_mean"] = vals[region].mean()
        row[f"{region}_sem"] = vals[region].sem()
        row[f"{region}_n"] = int(br_df.n_sgf.sum())
        row[f"{region}_total"] = int(br_df.n_neurons.sum())
    row.update(normalityGatedTest(vals[MFC], vals[LFC], alpha=alpha,
                                  left_name=MFC, right_name=LFC))
    return pd.DataFrame([row])


def _asRow(comparison):
    """Accept either the one-row frame from :func:`regionComparison` or a row."""
    if isinstance(comparison, pd.DataFrame):
        assert len(comparison) == 1, \
            f"Expected a single comparison, got {len(comparison)} rows"
        return comparison.iloc[0]
    return comparison


def significanceLabel(comparison, show_test=True, pval_fmt=".3g"):
    """``"** p=0.00478 (t-test (Student))"`` -- stars, p-value and the test the
    normality gate selected, which is what goes above the bracket."""
    row = _asRow(comparison)
    label = f"{pStars(row['pval'])} p={row['pval']:{pval_fmt}}"
    if show_test:
        label = f"{label} ({row['test']})"
    return label


def _renderer(fig):
    """Renderer for a figure whose artists are in their final positions.

    The draw matters: ``pie()`` sets an equal aspect, and the axes box (hence
    the title above it) only shrinks to a square when the figure is drawn.
    Measuring before that would place the bracket against the pre-aspect
    layout, well above the titles it is supposed to span.
    """
    fig.canvas.draw()
    try:
        return fig.canvas.get_renderer()
    except AttributeError:  # canvases without a cached renderer
        return fig._get_renderer()


def _topInFig(fig, artist, renderer):
    """Top edge of an artist, in figure coordinates.

    Measured rather than derived from the title's anchor: the pie titles are
    offset (``y=1.1`` in axes coords) *and* a text height tall, and guessing
    that height is what makes a bracket land on top of the labels.
    """
    bbox = artist.get_window_extent(renderer)
    return fig.transFigure.inverted().transform((bbox.x0, bbox.y1))[1]


def drawRegionSigBracket(fig, axes, comparison, pad=BRACKET_PAD,
                         tick=BRACKET_TICK, text_pad=BRACKET_TEXT_PAD,
                         suptitle_gap=BRACKET_SUPTITLE_GAP, color="k",
                         lw=BRACKET_LW, fontsize="medium", show_test=True):
    """Span a significance bracket across two axes, above their titles.

    Drawn in figure coordinates from the centre of the first axes to the centre
    of the second (i.e. between the two pie charts' MFC / LFC titles), with
    :func:`significanceLabel` -- stars, p-value and test name -- centred above
    it. ``comparison`` is the :func:`regionComparison` result.

    Call it *after* ``fig.suptitle(...)``: an existing suptitle is pushed up if
    the label would otherwise run into it.

    Returns ``(line, text)`` so a caller can restyle or remove them.
    """
    from matplotlib.lines import Line2D
    ax_left, ax_right = axes[0], axes[1]
    renderer = _renderer(fig)

    xs = [ax.get_position().x0 + ax.get_position().width / 2
          for ax in (ax_left, ax_right)]
    y = max(_topInFig(fig, ax.title, renderer)
            for ax in (ax_left, ax_right)) + pad

    line = Line2D([xs[0], xs[0], xs[1], xs[1]],
                  [y - tick, y, y, y - tick],
                  transform=fig.transFigure, color=color, lw=lw,
                  clip_on=False)
    fig.add_artist(line)
    text = fig.text(float(np.mean(xs)), y + text_pad,
                    significanceLabel(comparison, show_test=show_test),
                    ha="center", va="bottom", color=color, fontsize=fontsize)

    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None and suptitle.get_text():
        label_top = _topInFig(fig, text, renderer)
        if suptitle.get_position()[1] < label_top + suptitle_gap:
            suptitle.set_y(label_top + suptitle_gap)
    return line, text


def formatComparison(comparison_df):
    """The :func:`regionComparison` table as printable lines."""
    lines = []
    for _, row in comparison_df.iterrows():
        lines.append(f"Sgf. neurons for {row['label']} "
                     f"({row['data_col']}) - % of all neurons")
        for region in REGIONS:
            lines.append(
                f"    {region}: {row[f'{region}_mean']:.2f}% "
                f"+/- {row[f'{region}_sem']:.2f}% SEM  "
                f"({row[f'{region}_n_sessions']} sessions, "
                f"{row[f'{region}_n']:,}/{row[f'{region}_total']:,} neurons)")
        other_pval = (row["mwu_pval"] if "t-test" in row["test"]
                      else row["ttest_pval"])
        other_test = "MannWhitneyU" if "t-test" in row["test"] else "t-test"
        lines.append(
            f"    Normality (Shapiro-Wilk): "
            f"MFC p={row[f'{MFC}_shapiro_pval']:.3g}, "
            f"LFC p={row[f'{LFC}_shapiro_pval']:.3g} -> "
            + ("normal, t-test usable" if row["is_normal"] else
               "not normal, t-test not usable"))
        lines.append(f"    MFC vs LFC: {row['test']} "
                     f"stat={row['statistic']:.3g}, p={row['pval']:.3g} "
                     f"({other_test} p={other_pval:.3g} for reference)")
    return "\n".join(lines)
