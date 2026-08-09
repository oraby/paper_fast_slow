"""MFC vs LFC bars for neurons correlated with sampling time.

Companion to the ``### Pie-Chart for rigid/streteching neurons`` section of
``plottraces3.ipynb``. The pie charts pool every neuron of a region into one
slice; here the same threshold crossing is instead computed **per session**, so
each session contributes one percentage and the two regions can be compared
with sessions as the independent unit.

Three categories are built from the same two correlation columns the pie chart
uses (``<filter_key>firing_pos_pearson_corr`` and
``<filter_key>amplitude_firing_pearson_corr``):

- ``AUC + Firing Pos``  neurons above threshold on *either* metric (the union,
                        i.e. everything that is not "Rigid").
- ``AUC``               neurons above threshold on the amplitude/AUC metric,
                        **including** those also above on firing position
                        ("Both").
- ``Firing Pos``        neurons above threshold on the firing-position metric,
                        **including** "Both".

The last two therefore overlap by exactly the "Both" set, matching how the pie
chart's ``Both`` wedge belongs to each measure, and the first is the union of
the two — so it equals ``100 - Rigid`` in the pie chart.
"""
from ...common.clr import BrainRegion as BRClr
from ...common.definitions import BrainRegion
from ..seqdeviation import _p_stars
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

CATEGORY_ALL = "AUC + Firing Pos"
CATEGORY_AUC = "AUC"
CATEGORY_FIRING_POS = "Firing Pos"
#: Order used by the figures: the pooled comparison first, then the two metrics.
CATEGORIES = (CATEGORY_ALL, CATEGORY_AUC, CATEGORY_FIRING_POS)

MFC, LFC = "MFC", "LFC"
#: The pie charts relabel the raw region names the same way.
_REGION_LABELS = {"M2": MFC, "ALM": LFC}


def regionLabel(brain_region):
    """Raw ``BrainRegion`` code/enum -> the MFC / LFC label used in the paper."""
    br_str = str(BrainRegion(brain_region)).split("_")[0]
    return _REGION_LABELS.get(br_str, br_str)


def sessionCorrelatedPrcnt(df, filter_key, corr_thresh):
    """Per-session percentage of neurons correlated with sampling time.

    ``df`` is ``res_corr_df``: one row per neuron, with ``BrainRegion``,
    ``ShortName``, ``trace_id`` and the two correlation columns. A neuron counts
    for a category when the *absolute* correlation exceeds ``corr_thresh``,
    exactly as in the pie charts.

    Returns one row per (session x category) with ``n_neurons``, ``n_above`` and
    ``prcnt``. Sessions are the unit the bars and the test below average over.
    """
    firing_pos_key = filter_key + "firing_pos_pearson_corr"
    auc_key = filter_key + "amplitude_firing_pearson_corr"
    for col in (firing_pos_key, auc_key):
        assert col in df.columns, f"{col} missing from df; wrong filter_key?"

    rows = []
    for (br, sess), sess_df in df.groupby(["BrainRegion", "ShortName"]):
        n_neurons = len(sess_df)
        assert n_neurons == sess_df.trace_id.nunique(), (
            f"{sess}: expected one row per neuron")
        is_firing_pos = sess_df[firing_pos_key].abs() > corr_thresh
        is_auc = sess_df[auc_key].abs() > corr_thresh
        counts = {CATEGORY_ALL: (is_firing_pos | is_auc).sum(),
                  CATEGORY_AUC: is_auc.sum(),
                  CATEGORY_FIRING_POS: is_firing_pos.sum()}
        for category in CATEGORIES:
            n_above = int(counts[category])
            rows.append(dict(BrainRegion=br, region=regionLabel(br),
                             ShortName=sess, category=category,
                             n_neurons=n_neurons, n_above=n_above,
                             prcnt=100 * n_above / n_neurons))
    return pd.DataFrame(rows)


def regionSignificance(prcnt_df, regions=(MFC, LFC), alpha=0.05):
    """Is the MFC vs LFC difference significant, per category?

    The two regions are *different* sessions, so this is an unpaired comparison
    with sessions as the independent unit. Shapiro-Wilk is run on each region's
    per-session percentages; if both look normal (``p > alpha``) the test is
    Welch's t-test (unequal variances, the safe default for differing session
    counts), otherwise Mann-Whitney U. Both two-sided: "do the two regions
    differ?".

    Returns a tidy table with each region's mean +/- SEM, both normality
    results, the chosen test, its statistic and p-value.
    """
    left, right = regions
    rows = []
    for category in CATEGORIES:
        cat_df = prcnt_df[prcnt_df.category == category]
        groups = {}
        for region in (left, right):
            groups[region] = cat_df[cat_df.region == region].prcnt.to_numpy()

        row = dict(category=category)
        normal = True
        for region in (left, right):
            vals = groups[region]
            row[f"n_{region}"] = len(vals)
            row[f"{region}_mean"] = vals.mean() if len(vals) else np.nan
            row[f"{region}_sem"] = (stats.sem(vals) if len(vals) > 1 else np.nan)
            # Shapiro-Wilk needs at least 3 samples; too few to judge -> assume
            # non-normal and fall back to the rank test.
            if len(vals) >= 3:
                w, p = stats.shapiro(vals)
            else:
                w, p = np.nan, np.nan
            row[f"shapiro_W_{region}"] = w
            row[f"shapiro_p_{region}"] = p
            normal = normal and (p > alpha if np.isfinite(p) else False)

        row["normality"] = "Shapiro-Wilk"
        row["normal"] = normal
        if min(len(groups[left]), len(groups[right])) < 2:
            row.update(test="n/a (too few sessions)", statistic=np.nan,
                       p_value=np.nan, sig="ns")
        else:
            if normal:
                res = stats.ttest_ind(groups[left], groups[right],
                                      equal_var=False)  # Welch, two-sided
                test = "Welch t-test"
            else:
                res = stats.mannwhitneyu(groups[left], groups[right],
                                         alternative="two-sided")
                test = "Mann-Whitney U"
            row.update(test=test, statistic=res.statistic, p_value=res.pvalue,
                       sig=_p_stars(res.pvalue))
        rows.append(row)
    return pd.DataFrame(rows)


def plotRegionBars(prcnt_df, stats_df=None, regions=(MFC, LFC),
                   corr_thresh=None, filter_key=None, plot_sessions=True,
                   save_figs=False, fig_save_prefix=None):
    """One figure, one panel per category: mean +/- SEM across sessions.

    Individual sessions are overlaid as dots (same convention as the other
    per-session bar figures in this notebook) so the spread behind the SEM stays
    visible. When ``stats_df`` is given, each panel is annotated with the chosen
    test's p-value and significance stars.
    """
    if stats_df is None:
        stats_df = regionSignificance(prcnt_df, regions=regions)

    fig, axs = plt.subplots(1, len(CATEGORIES),
                            figsize=(4 * len(CATEGORIES), 4.5), sharey=True)
    for ax, category in zip(np.atleast_1d(axs), CATEGORIES):
        cat_df = prcnt_df[prcnt_df.category == category]
        stat_row = stats_df[stats_df.category == category].iloc[0]
        panel_max = 0.0
        for x, region in enumerate(regions):
            region_df = cat_df[cat_df.region == region]
            vals = region_df.prcnt.to_numpy()
            # Raw region code -> the same color the rest of the notebook uses.
            # int() because BRClr's isinstance(.., int) check misses numpy ints.
            clr = (BRClr[int(region_df.BrainRegion.iloc[0])] if len(region_df)
                   else "gray")
            sem = stats.sem(vals) if len(vals) > 1 else None
            ax.bar(x, vals.mean() if len(vals) else np.nan, yerr=sem,
                   color=clr, alpha=.8, capsize=4, width=.6,
                   label=f"{region} ({len(vals)} sess)")
            if plot_sessions and len(vals):
                # All sessions on the bar's own x, overlapping the SEM whisker.
                ax.scatter(np.full(len(vals), x), vals,
                           color="gray", s=18, zorder=10, alpha=.7)
            if len(vals):
                panel_max = max(panel_max, vals.max(),
                                vals.mean() + (sem or 0))

        _annotateSignificance(ax, stat_row, n_groups=len(regions),
                              data_max=panel_max)
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(list(regions))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(category)
    np.atleast_1d(axs)[0].set_ylabel(
        f"Neurons correlated with sampling time (%)"
        + (f"\n|r| > {corr_thresh}" if corr_thresh is not None else ""))
    fig.suptitle("Correlation with sampling time, per session"
                 + (f" — {filter_key}" if filter_key else ""))
    fig.tight_layout()

    if save_figs:
        assert fig_save_prefix is not None, (
            "fig_save_prefix is required when save_figs is True")
        thresh_str = "" if corr_thresh is None else f"_above_{corr_thresh}"
        save_fp = (Path(fig_save_prefix) /
                   f"rt_corr_regions_bars{thresh_str}.pdf")
        print("Save fp=", save_fp)
        fig.savefig(save_fp, bbox_inches="tight")
    plt.show()
    return fig


def _annotateSignificance(ax, stat_row, n_groups, data_max):
    """Draw the significance bracket + p-value clear of every plotted point.

    ``data_max`` is the tallest thing already drawn in the panel (highest
    session dot or bar+SEM), so the bracket never lands on top of the data. The
    y-limit is then grown to leave room for the label above the bracket.
    """
    p_value = stat_row.get("p_value", np.nan)
    if not np.isfinite(p_value):
        return
    span = data_max if data_max > 0 else 1.0
    y = data_max + span * .08          # bracket sits clear of the top dot
    tick = span * .02                  # the bracket's little downward ends
    ax.plot([0, 0, n_groups - 1, n_groups - 1],
            [y, y + tick, y + tick, y], lw=1, color="black")
    ax.text((n_groups - 1) / 2, y + tick * 1.5,
            f"{stat_row['sig']}\np={p_value:.3g}\n{stat_row['test']}",
            ha="center", va="bottom", fontsize="x-small")
    # Headroom for the three-line label; shared y-axis keeps panels aligned.
    ax.set_ylim(top=max(ax.get_ylim()[1], y + span * .30))
