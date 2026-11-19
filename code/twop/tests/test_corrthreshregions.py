"""Tests for the MFC vs LFC correlated-neuron bars."""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parents[3]))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.twop.plot import corrthreshregions as ctr  # noqa: E402

FILTER_KEY = "zscore_filter_"
FIRING_POS = FILTER_KEY + "firing_pos_pearson_corr"
AUC = FILTER_KEY + "amplitude_firing_pearson_corr"

# BrainRegion codes: the enum value whose name starts with M2 / ALM.
from code.common.definitions import BrainRegion  # noqa: E402

M2_CODE = next(int(br) for br in BrainRegion if str(br).startswith("M2"))
ALM_CODE = next(int(br) for br in BrainRegion if str(br).startswith("ALM"))


def _neurons(region_code, sess, firing_pos, auc):
    """One row per neuron, mirroring res_corr_df's shape."""
    return pd.DataFrame({"BrainRegion": region_code, "ShortName": sess,
                         "trace_id": range(len(firing_pos)),
                         FIRING_POS: firing_pos, AUC: auc})


def test_regionLabel_maps_the_paper_names():
    assert ctr.regionLabel(M2_CODE) == "MFC"
    assert ctr.regionLabel(ALM_CODE) == "LFC"


def test_categories_count_both_toward_each_metric():
    # 4 neurons: [both, firing-pos only, auc only, neither]
    df = _neurons(M2_CODE, "s1",
                  firing_pos=[0.9, 0.8, 0.1, 0.0],
                  auc=[0.9, 0.1, 0.7, 0.0])
    out = ctr.sessionCorrelatedPrcnt(df, FILTER_KEY, corr_thresh=0.3)
    got = dict(zip(out.category, out.n_above))
    assert got[ctr.CATEGORY_FIRING_POS] == 2   # both + firing-pos only
    assert got[ctr.CATEGORY_AUC] == 2          # both + auc only
    assert got[ctr.CATEGORY_ALL] == 3          # the union, i.e. not "Rigid"
    assert (out.n_neurons == 4).all()
    assert got[ctr.CATEGORY_ALL] / 4 * 100 == out.query(
        "category == @ctr.CATEGORY_ALL").prcnt.iloc[0]


def test_threshold_uses_absolute_correlation():
    """Strongly *negative* correlations count too, as in the pie charts."""
    df = _neurons(M2_CODE, "s1", firing_pos=[-0.9, -0.1], auc=[0.0, 0.0])
    out = ctr.sessionCorrelatedPrcnt(df, FILTER_KEY, corr_thresh=0.3)
    assert out.query("category == @ctr.CATEGORY_FIRING_POS").n_above.iloc[0] == 1


def test_threshold_is_strict_greater_than():
    df = _neurons(M2_CODE, "s1", firing_pos=[0.3], auc=[0.0])
    out = ctr.sessionCorrelatedPrcnt(df, FILTER_KEY, corr_thresh=0.3)
    assert out.n_above.sum() == 0


def test_percentages_are_per_session_not_pooled():
    """A big session must not drown out a small one — the unit is the session."""
    df = pd.concat([
        _neurons(M2_CODE, "small", firing_pos=[0.9, 0.9], auc=[0, 0]),   # 100%
        _neurons(M2_CODE, "big", firing_pos=[0.0] * 100, auc=[0] * 100),  # 0%
    ], ignore_index=True)
    out = ctr.sessionCorrelatedPrcnt(df, FILTER_KEY, corr_thresh=0.3)
    all_df = out[out.category == ctr.CATEGORY_ALL]
    assert sorted(all_df.prcnt) == [0.0, 100.0]
    # Pooling would give 2/102 ~ 2%; the session mean is 50%.
    assert all_df.prcnt.mean() == 50.0


def test_wrong_filter_key_is_rejected():
    df = _neurons(M2_CODE, "s1", firing_pos=[0.9], auc=[0.9])
    with pytest.raises(AssertionError):
        ctr.sessionCorrelatedPrcnt(df, "iqr_filter_", corr_thresh=0.3)


def test_duplicate_neuron_rows_are_rejected():
    df = _neurons(M2_CODE, "s1", firing_pos=[0.9, 0.9], auc=[0.9, 0.9])
    df["trace_id"] = 7  # same neuron twice
    with pytest.raises(AssertionError):
        ctr.sessionCorrelatedPrcnt(df, FILTER_KEY, corr_thresh=0.3)


def _twoRegionPrcnt(mfc_vals, lfc_vals):
    """Build a prcnt_df directly, one session per supplied percentage."""
    rows = []
    for region_code, vals in ((M2_CODE, mfc_vals), (ALM_CODE, lfc_vals)):
        for i, val in enumerate(vals):
            for category in ctr.CATEGORIES:
                rows.append(dict(BrainRegion=region_code,
                                 region=ctr.regionLabel(region_code),
                                 ShortName=f"{region_code}_s{i}",
                                 category=category, n_neurons=100,
                                 n_above=val, prcnt=float(val)))
    return pd.DataFrame(rows)


def test_significance_reports_every_category_with_means_and_sem():
    prcnt_df = _twoRegionPrcnt([10, 12, 14, 11], [40, 44, 42, 39])
    out = ctr.regionSignificance(prcnt_df)
    assert list(out.category) == list(ctr.CATEGORIES)
    row = out.iloc[0]
    assert row.n_MFC == 4 and row.n_LFC == 4
    assert row.MFC_mean == pytest.approx(11.75)
    assert row.LFC_mean == pytest.approx(41.25)
    assert row.MFC_sem > 0 and row.LFC_sem > 0
    assert row.p_value < 0.05 and row.sig != "ns"


def test_normal_groups_use_welch_t_test():
    rng = np.random.default_rng(0)
    prcnt_df = _twoRegionPrcnt(rng.normal(20, 3, 30), rng.normal(22, 3, 30))
    out = ctr.regionSignificance(prcnt_df)
    assert (out.test == "Welch t-test").all()
    assert out.normal.all()


def test_non_normal_groups_fall_back_to_mann_whitney():
    # A heavy outlier breaks normality -> the rank test must be chosen.
    mfc = [10, 10.5, 11, 10.2, 10.8, 10.1, 10.4, 1000]
    lfc = [20, 20.5, 21, 20.2, 20.8, 20.1, 20.4, 20.3]
    out = ctr.regionSignificance(_twoRegionPrcnt(mfc, lfc))
    assert (out.test == "Mann-Whitney U").all()
    assert not out.normal.any()


def test_too_few_sessions_reports_no_test_rather_than_crashing():
    out = ctr.regionSignificance(_twoRegionPrcnt([10], [20]))
    assert (out.test == "n/a (too few sessions)").all()
    assert out.p_value.isna().all()


def test_plot_runs_headless_and_saves(tmp_path):
    prcnt_df = _twoRegionPrcnt([10, 12, 14, 11], [40, 44, 42, 39])
    fig = ctr.plotRegionBars(prcnt_df, corr_thresh=0.3, filter_key=FILTER_KEY,
                             save_figs=True, fig_save_prefix=tmp_path)
    assert len(fig.axes) == len(ctr.CATEGORIES)
    assert (tmp_path / "rt_corr_regions_bars_above_0.3.svg").exists()


def test_plot_without_save_writes_nothing(tmp_path):
    prcnt_df = _twoRegionPrcnt([10, 12, 14, 11], [40, 44, 42, 39])
    ctr.plotRegionBars(prcnt_df, save_figs=False)
    assert not list(tmp_path.iterdir())


def _sessionDots(ax):
    """The scatter collections only — errorbar caps are collections too."""
    from matplotlib.collections import PathCollection
    return [c for c in ax.collections if isinstance(c, PathCollection)
            and len(c.get_offsets())]


def test_session_dots_sit_on_the_bar_centre():
    """No x-jitter: every session dot shares its bar's x, over the SEM whisker."""
    prcnt_df = _twoRegionPrcnt([10, 12, 14, 11], [40, 44, 42, 39])
    fig = ctr.plotRegionBars(prcnt_df)
    ax = fig.axes[0]
    dots = _sessionDots(ax)
    assert len(dots) == 2                              # one per region
    xs_by_collection = [np.unique(c.get_offsets()[:, 0]) for c in dots]
    for xs in xs_by_collection:
        assert len(xs) == 1                            # all dots share one x
    assert sorted(x[0] for x in xs_by_collection) == [0, 1]


def test_significance_bracket_clears_the_tallest_dot():
    # LFC's tallest session (44) is well above its bar mean (41.25) + SEM.
    prcnt_df = _twoRegionPrcnt([10, 12, 14, 11], [40, 44, 42, 39])
    fig = ctr.plotRegionBars(prcnt_df)
    ax = fig.axes[0]
    dot_max = max(c.get_offsets()[:, 1].max() for c in _sessionDots(ax))
    bracket = [ln for ln in ax.lines if len(ln.get_xdata()) == 4]
    assert bracket, "no significance bracket drawn"
    assert bracket[0].get_ydata().min() > dot_max
    # ...and the label above it still fits inside the axes.
    assert ax.get_ylim()[1] > bracket[0].get_ydata().max()


def test_no_bracket_when_the_test_could_not_run():
    fig = ctr.plotRegionBars(_twoRegionPrcnt([10], [20]))
    for ax in fig.axes:
        assert not [ln for ln in ax.lines if len(ln.get_xdata()) == 4]
