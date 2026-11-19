'''Permutation tests of the firing sequence across strategies (Figures S9G, 4J).

Backend for the "Permutation test of the sequence against typical trials" and
"Seq firing deviation" sections of ``2p_2_sequences.ipynb``.

Each neuron's place in the sequence is the median of its peak-firing sample
(:func:`..twop.seqdeviation.extractIQR`), computed separately on typical,
impulsive (fast) and deliberate (slow) trials. Only neurons active on at least
``prcnt_valid`` % of the trials of all three kinds enter, and the three
sequences are joined on ``trace_id``.

- **S9G** (``typicalPenaltyTests``) -- the rank mismatch ("penalty") of the
  fast and of the slow sequence against the typical one, set against a null
  built by shuffling the typical sequence. ``plotPenaltyTests`` draws the
  per-session histograms (left panel; the published example is MFC session
  ``GP4_28_S6_L40_D250_M2L``), and the notebook's ``plotPenalityCost`` the
  summary over sessions.
- **4J** (``deviationTests``) -- the rank deviation between the fast and the
  slow sequence, against a null that re-pairs the two. ``plotDeviationTests``
  draws the per-session histograms (left panel; the published example is LFC
  session ``GP4_85_s7_L70_D250_ALM``).

**The tests are unseeded and run 100,000 permutations per session**, so the
p-values and z-scores move slightly from run to run and computing them is
slow. They are therefore computed once, and every drawing reads the stored
results: ``*Tests`` return a summary frame plus one record per session that
keeps the scipy result objects (null distributions included).

Figures are written under ``{fig_save_prefix}/Sequence/``.
'''
from __future__ import annotations

import pathlib
import statistics as stdlib_statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..common.clr import BrainRegion as BRClr
from ..common.definitions import BrainRegion
from .seqdeviation import extractIQR

#: Permutations per test, as published.
N_RESAMPLES = 100_000


def combDfsSortToRef(df_ref, df_impulsive, df_deliberate, prcnt_valid,
                     ref_name):
    '''Join the three per-neuron sequence tables on ``trace_id``.

    Returns ``(False, (ref, impulsive, deliberate))`` when a table has no
    neuron above ``prcnt_valid``, else ``(True, joined)``; the joined frame
    suffixes the reference columns ``_{ref_name}`` and the others
    ``_impulsive`` / ``_deliberate``.
    '''
    df_ref = df_ref[df_ref.prcnt_valid >= prcnt_valid]
    df_impulsive = df_impulsive[df_impulsive.prcnt_valid >= prcnt_valid]
    df_deliberate = df_deliberate[df_deliberate.prcnt_valid >= prcnt_valid]
    if not len(df_ref) or not len(df_impulsive) or not len(df_deliberate):
        return False, (df_ref, df_impulsive, df_deliberate)
    df_ref = extractIQR(df_ref)
    df_impulsive = extractIQR(df_impulsive)
    df_deliberate = extractIQR(df_deliberate)

    # Merge on trace_id in both, sorting will happen according to all as its
    # the first df we pass, and its sorted already by firing order
    df = pd.merge(df_ref, df_impulsive, how='inner', on="trace_id",
                  suffixes=(f"_{ref_name}", "_impulsive"))
    df_deliberate = df_deliberate.rename(
        columns={col: f"{col}_deliberate"
                 for col in df_deliberate.columns if col != "trace_id"})
    # suffixes has no effect here as all the columns names already differ
    df = pd.merge(df, df_deliberate, how='inner', on="trace_id")
    return True, df


def _groups(df_typical, df_impulsive, df_deliberate, by_sess):
    '''(title, ShortName, region, color, typical, impulsive, deliberate) per
    region, or per session within each region when ``by_sess``.'''
    for br, br_typical_df in df_typical.groupby("BrainRegion"):
        br = BrainRegion(br)
        br_str = str(br).split("_")[0]
        br_clr = BRClr[BrainRegion(br)]
        br_impulsive_df = df_impulsive[df_impulsive.BrainRegion == br]
        br_deliberate_df = df_deliberate[df_deliberate.BrainRegion == br]
        if not by_sess:
            yield (br_str, None, br_str, br_clr, br_typical_df,
                   br_impulsive_df, br_deliberate_df)
            continue
        for sess, sess_typical_df in br_typical_df.groupby("ShortName"):
            yield (f"{br_str} - {sess}", sess, br_str, br_clr, sess_typical_df,
                   br_impulsive_df[br_impulsive_df.ShortName == sess],
                   br_deliberate_df[br_deliberate_df.ShortName == sess])


def _pick(tests, sessions):
    if sessions is None:
        return tests
    sessions = list(sessions)
    missing = set(sessions) - {test["ShortName"] for test in tests}
    if missing:
        raise KeyError(f"Sessions without a test: {sorted(missing)}")
    return [test for test in tests if test["ShortName"] in sessions]


def _finish(fig, save_fp, save_figs, show):
    if save_figs:
        print("saving...")
        path = pathlib.Path(save_fp)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fp)
    if show is None:
        show = not save_figs
    if show:
        plt.show()
    else:
        plt.close(fig)


# --- S9G: fast / slow against typical ----------------------------------------

def _calcPenality(org_order, new_order):
    penality = np.abs((org_order - new_order)).sum() / len(org_order)
    return penality


class Penality:
    '''The test statistic, scoring the observed pairing first.

    scipy's ``permutation_test`` calls the statistic first on the unpermuted
    data, then once per permutation. Its first call therefore scores the
    reference against ``other``; the later ones score the reference against
    its own shuffles, i.e. the reference order is the one shuffled.
    '''
    def __init__(self, other) -> None:
        self._count = 0
        self._other = other

    def _calcPenality(self, org_order, new_order):
        if self._count == 0:
            assert (org_order == new_order).all()
            penalty = _calcPenality(org_order, self._other)
        else:
            penalty = _calcPenality(org_order, new_order)
        self._count += 1
        return penalty


def rankPenaltyTest(df, ref_col, other_col, normalize,
                    n_resamples=N_RESAMPLES):
    '''Permutation test of ``other_col``'s rank order against ``ref_col``'s.

    Returns ``(result, p=.025 threshold, p=.01 threshold)``; with
    ``normalize`` the statistic, the null and the thresholds are z-scored
    against the null.
    '''
    df = df.copy()
    df[f"{ref_col}_rank"] = df[ref_col].rank(method="dense").values
    df[f"{other_col}_rank"] = df[other_col].rank(method="dense").values
    x, y = df[f"{ref_col}_rank"].values, df[f"{other_col}_rank"].values
    penalty = Penality(y)
    # Not seeded: no rng= is passed, so the permutations, and this panel's
    # p-values, differ from run to run. Seeding would shift the reported values.
    stats_res = stats.permutation_test((x, x), penalty._calcPenality,
                                       n_resamples=n_resamples,
                                       vectorized=False,
                                       permutation_type="pairings")

    mean = np.mean(stats_res.null_distribution)
    std = np.std(stats_res.null_distribution)
    norm_dist = stdlib_statistics.NormalDist(mu=mean, sigma=std)
    lower_sgf_val = norm_dist.inv_cdf(.025)
    lower_v_sgf_val = norm_dist.inv_cdf(.01)

    if normalize:
        lower_sgf_zscore = norm_dist.zscore(lower_sgf_val)
        lower_v_sgf_zscore = norm_dist.zscore(lower_v_sgf_val)
        print("Mean:", mean, "- std:", std, "Cdf mean:", norm_dist.zscore(mean),
              "Lower sgf:", lower_sgf_val, "Lower sgf zscore:", lower_sgf_zscore,
              "Lower v sgf:", lower_v_sgf_val, "Lower v sgf zscore:", lower_v_sgf_zscore)
        stats_res.null_distribution = norm_dist.zscore(stats_res.null_distribution)
        stats_res.statistic = norm_dist.zscore(stats_res.statistic)

    return stats_res, (lower_sgf_zscore if normalize else lower_sgf_val), (
                       lower_v_sgf_zscore if normalize else lower_v_sgf_val)


def typicalPenaltyTests(df_typical, df_impulsive, df_deliberate, prcnt_valid,
                        normalized, by_sess=False, n_resamples=N_RESAMPLES):
    '''S9G: test the fast and the slow sequence against the typical one.

    Returns ``(summary, tests)``. ``summary`` has one row per region (or per
    session) with the statistics and thresholds ``plotPenalityCost`` reads;
    ``tests`` one record per row with the scipy results, for drawing.
    '''
    rows, tests = [], []
    groups = _groups(df_typical, df_impulsive, df_deliberate, by_sess)
    for title, sess, br_str, br_clr, typical, impulsive, deliberate in groups:
        is_good_merge, df = combDfsSortToRef(typical, impulsive, deliberate,
                                             prcnt_valid, ref_name="typical")
        if not is_good_merge:
            continue
        ref_col = "med_typical"
        res_impulsive, sgf_impulsive, v_sgf_impulsive = rankPenaltyTest(
            df, ref_col, "med_impulsive", normalized, n_resamples)
        res_deliberate, sgf_deliberate, v_sgf_deliberate = rankPenaltyTest(
            df, ref_col, "med_deliberate", normalized, n_resamples)
        res_typical, sgf_typical, v_sgf_typical = rankPenaltyTest(
            df, ref_col, "med_typical", normalized, n_resamples)
        print("stats_res_impulsive.pvalue:", res_impulsive.pvalue)
        print("stats_res_deliberate.pvalue:", res_deliberate.pvalue)
        penality = dict(impulsive_stat=res_impulsive.statistic,
                        impulsive_sgf=sgf_impulsive,
                        impulsive_v_sgf=v_sgf_impulsive,
                        deliberate_stat=res_deliberate.statistic,
                        deliberate_sgf=sgf_deliberate,
                        deliberate_v_sgf=v_sgf_deliberate,
                        typical_stat=res_typical.statistic,
                        typical_sgf=sgf_typical,
                        typical_v_sgf=v_sgf_typical,)
        if sess is None:
            print(br_str, "penality:", penality)
        else:
            print(br_str, sess, "penality:", penality)
            penality["ShortName"] = sess
        penality["BrainRegion"] = br_str
        rows.append(penality)
        tests.append(dict(title=title, ShortName=sess, BrainRegion=br_str,
                          clr=br_clr, num_cells=len(df),
                          prcnt_valid=prcnt_valid, normalized=normalized,
                          impulsive=res_impulsive, deliberate=res_deliberate,
                          typical=res_typical))

    df_res = pd.DataFrame(rows)
    df_res["normalized"] = normalized
    df_res["test_name"] = "Permutation"
    df_res["prcnt_valid"] = prcnt_valid
    return df_res, tests


def penaltyTestPath(fig_save_prefix, test) -> str:
    normed_str = " Z-Scored - " if test["normalized"] else ""
    return (f"{fig_save_prefix}/Sequence/"
            f"seq_rank_{test['title']}_Permutation_{normed_str}_"
            f"{test['prcnt_valid']}_prcnt_active.svg")


def plotPenaltyTest(test, save_figs, fig_save_prefix=None, show=None):
    '''Draw one S9G-left histogram: the typical null, with the fast, slow and
    typical statistics as dashed lines.'''
    res_impulsive, res_deliberate, res_typical = (
        test["impulsive"], test["deliberate"], test["typical"])
    fig, ax_total = plt.subplots()
    ax_total.hist([res_typical.null_distribution],
                  bins=25,
                  histtype='step',
                  label=(f"Typical ({len(res_typical.null_distribution):,} "
                         f"Permutations)"),
                  color=["orange"],
                  density=True)
    ax_total.axvline(res_impulsive.statistic, color='r', linestyle='--',
            label=f"Impulsive - rank penality={res_impulsive.statistic:.4g} - pval={res_impulsive.pvalue:.4f}")
    ax_total.axvline(res_deliberate.statistic, color='y', linestyle='--',
            label=f"Deliberate - rank penality={res_deliberate.statistic:.4g} - pval={res_deliberate.pvalue:.4f}")
    ax_total.axvline(res_typical.statistic, color='orange', linestyle='--',
            label=f"Typical - rank penality={res_typical.statistic:.4g} - pval={res_typical.pvalue:.4f}")
    ax_total.set_xlabel('Rank mismatch penality - statistic value')
    ax_total.set_ylabel('Density')
    normed_str = " Z-Scored - " if test["normalized"] else ""
    ax_total.set_title(f"Sequence Rank Permutation Test - {test['title']} - "
                       f"{normed_str}#Cells: {test['num_cells']}\n"
                       f"Min. percent of active trials per condition: "
                       f"{test['prcnt_valid']}%", color=test["clr"],
                       fontsize="small")
    ax_total.legend(fontsize="x-small")
    [ax_total.spines[_dir].set_visible(False) for _dir in ["right", "left", "top"]]
    _finish(fig, penaltyTestPath(fig_save_prefix, test) if save_figs else None,
            save_figs, show)
    return fig


def plotPenaltyTests(tests, save_figs, fig_save_prefix=None, sessions=None,
                     show=None):
    '''Draw every S9G-left histogram, or only those of ``sessions``.'''
    for test in _pick(tests, sessions):
        plotPenaltyTest(test, save_figs, fig_save_prefix, show=show)


# --- 4J: fast against slow ---------------------------------------------------

def _calcDeviation(fast_rank, slow_rank):
    deviation = np.abs((fast_rank - slow_rank)).sum() / len(fast_rank)
    return deviation


def fastSlowDeviationTest(df, fast_col, slow_col, n_resamples=N_RESAMPLES):
    '''Permutation test of the fast/slow rank deviation; returns the scipy
    result and the normal fitted to its null.'''
    df = df.copy()
    df[f"{fast_col}_rank"] = df[fast_col].rank(method="dense").values
    df[f"{slow_col}_rank"] = df[slow_col].rank(method="dense").values
    x, y = df[f"{fast_col}_rank"].values, df[f"{slow_col}_rank"].values
    # Not seeded: no rng= is passed, so the permutations, and this panel's
    # p-values, differ from run to run. Seeding would shift the reported values.
    stats_res = stats.permutation_test((x, y), _calcDeviation,
                                       n_resamples=n_resamples,
                                       permutation_type="pairings",
                                       alternative="less")

    mean = np.mean(stats_res.null_distribution)
    std = np.std(stats_res.null_distribution)
    norm_dist = stdlib_statistics.NormalDist(mu=mean, sigma=std)
    return stats_res, norm_dist


def deviationTests(df_typical, df_impulsive, df_deliberate, prcnt_valid,
                   by_sess=False, n_resamples=N_RESAMPLES):
    '''4J: test the fast sequence's deviation from the slow one.

    Returns ``(summary, tests)``: ``summary`` has one row per region (or
    session) with the ``deviation_stat`` result and ``deviation_normed_dist``
    the notebook's ``plotDeviationAgg`` reads (4J right); ``tests`` one record
    per row for drawing.
    '''
    rows, tests = [], []
    groups = _groups(df_typical, df_impulsive, df_deliberate, by_sess)
    for title, sess, br_str, br_clr, typical, impulsive, deliberate in groups:
        is_good_merge, df = combDfsSortToRef(typical, impulsive, deliberate,
                                             prcnt_valid, ref_name="typical")
        if not is_good_merge:
            continue
        stats_res, norm_dist = fastSlowDeviationTest(
            df, "med_impulsive", "med_deliberate", n_resamples)
        print("sstats_res_deviation.pvalue:", stats_res.pvalue)
        deviation = dict(deviation_stat=stats_res,
                         deviation_normed_dist=norm_dist)
        if sess is None:
            print(br_str, "deviation:", deviation)
        else:
            print(br_str, sess, "Deviation:", deviation)
            deviation["ShortName"] = sess
        deviation["BrainRegion"] = br_str
        rows.append(deviation)
        tests.append(dict(title=title, ShortName=sess, BrainRegion=br_str,
                          clr=br_clr, num_cells=len(df),
                          prcnt_valid=prcnt_valid, deviation=stats_res,
                          normed_dist=norm_dist))

    df_res = pd.DataFrame(rows)
    df_res["test_name"] = "Permutation"
    df_res["prcnt_valid"] = prcnt_valid
    return df_res, tests


def deviationTestPath(fig_save_prefix, test) -> str:
    return (f"{fig_save_prefix}/Sequence/deviation_rank_{test['title']}_"
            f"{test['prcnt_valid']}_prcnt_active.svg")


def plotDeviationTest(test, save_figs, fig_save_prefix=None, show=None):
    '''Draw one 4J-left histogram: the z-scored null and the observed
    deviation.'''
    stats_res, norm_dist = test["deviation"], test["normed_dist"]
    fig, ax_total = plt.subplots()
    normed_null_dist = norm_dist.zscore(stats_res.null_distribution)
    ax_total.hist(normed_null_dist, bins=25, histtype='step', color="k",
                  density=True,
                  label=f"Shuffle ({len(normed_null_dist):,} Permutations)")
    deviation = stats_res.statistic
    normed_deviation = norm_dist.zscore(deviation)
    ax_total.axvline(normed_deviation, color='k', linestyle='--',
                     label=f"Deviation {deviation:.4g} (z-score={normed_deviation:.4g})\n"
                           f"pval={stats_res.pvalue:.4f}")
    ax_total.set_xlabel('Rank deviation (z-score)')
    ax_total.set_ylabel('Density')
    ax_total.set_title(
        f"Sequence Deviation Permutation Test - {test['title']} - "
        f"#Cells: {test['num_cells']}\n"
        f"Min. percent of active trials per condition: {test['prcnt_valid']}%",
        color=test["clr"], fontsize="small")
    ax_total.legend(fontsize="x-small")
    [ax_total.spines[_dir].set_visible(False) for _dir in ["right", "left", "top"]]
    _finish(fig, deviationTestPath(fig_save_prefix, test) if save_figs else None,
            save_figs, show)
    return fig


def plotDeviationTests(tests, save_figs, fig_save_prefix=None, sessions=None,
                       show=None):
    '''Draw every 4J-left histogram, or only those of ``sessions``.'''
    for test in _pick(tests, sessions):
        plotDeviationTest(test, save_figs, fig_save_prefix, show=show)
