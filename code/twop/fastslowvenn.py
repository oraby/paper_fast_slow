'''Overlap between fast- and slow-strategy active neurons (Figure 4G).

Backend for the "Overlap between Active Impulsive neurons and Deliberate
neurons" section of ``2pAnalysis.ipynb``.

A neuron counts as *active* in a strategy when it fired on at least
``min_prcnt_valid`` percent of that strategy's trials (10% in the paper). The
panel asks whether the two strategies recruit the same neurons: one Venn per
region, MFC and LFC, drawn from the session-level percentages.

**The three numbers are percentages of the session's neurons, averaged across
sessions** — each session contributes one overlap, one fast-only and one
slow-only percentage, and the panel shows their mean with the SEM across
sessions. So a session with few neurons weighs as much as a large one.

**The denominator is the union of the session's fast and slow trace ids**, not
the count of recorded neurons: a neuron that never reached threshold in either
strategy still sits in the denominator as long as it was measured in both
frames, which it always is here. ``loopBRPlotVennOverlap`` does group the
all-trials frame per session, but only to enumerate the sessions — the
group sizes never enter the arithmetic.

Extracted from the notebook unchanged except that ``fig_save_prefix`` is now a
parameter (it was read from the notebook's globals) and the dead ``BY_SESS``
branch, which raised ``NotImplementedError``, is gone.
'''
from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib_venn import venn2

from ..common.definitions import BrainRegion

#: Fast gets red, slow yellow, as everywhere else in the paper.
SET_COLORS = ("red", "yellow")
#: The intersection is drawn in both colours, hatched, at this line width.
HATCH_LINEWIDTH = 4


def fastSlowOverlap(fast_df, slow_df, min_prcnt_valid):
    '''Percentages of one session's neurons that are fast-only, slow-only, both.

    Returns ``(overlap, fast_minus_slow, slow_minus_fast)``.
    '''
    assert fast_df.trace_id.nunique() == len(fast_df)
    assert slow_df.trace_id.nunique() == len(slow_df)
    total_valid_common = len(set(fast_df.trace_id) | set(slow_df.trace_id))
    fast_valid = set(fast_df[fast_df.prcnt_valid >= min_prcnt_valid].trace_id)
    slow_valid = set(slow_df[slow_df.prcnt_valid >= min_prcnt_valid].trace_id)
    prcnt = lambda ids: 100 * len(ids) / total_valid_common
    return (prcnt(fast_valid & slow_valid),
            prcnt(fast_valid - slow_valid),
            prcnt(slow_valid - fast_valid))


def sessionOverlaps(fast_df, slow_df, sessions, min_prcnt_valid):
    '''One row per session, in ``sessions`` order: the three percentages.'''
    rows = [fastSlowOverlap(fast_df[fast_df.ShortName == sess],
                            slow_df[slow_df.ShortName == sess], min_prcnt_valid)
            for sess in sessions]
    return pd.DataFrame(rows, columns=["overlap", "fast_minus_slow",
                                       "slow_minus_fast"], index=list(sessions))


def plotVennOverlap(fast_df, slow_df, min_prcnt_valid, sessions, br_str,
                    save_figs, fig_save_prefix=None):
    '''One region's Venn: mean +/- SEM across ``sessions``.'''
    overlaps = sessionOverlaps(fast_df, slow_df, sessions, min_prcnt_valid)
    means, sems = overlaps.mean(), overlaps.sem()
    print(f"{br_str} @{min_prcnt_valid}% - "
          f"Overlap Total={means.overlap:.2f}±{sems.overlap:.2f}% - "
          f"Fast-Slow={means.fast_minus_slow:.2f}±{sems.fast_minus_slow:.2f}% - "
          f"Slow-Fast={means.slow_minus_fast:.2f}±{sems.slow_minus_fast:.2f}%")

    # The label formatter only receives the subset value, so the SEM is looked
    # up by it -- which needs the three means to differ.
    assert means.nunique() == len(means), \
        f"Two subsets share a mean, so their SEMs cannot be told apart: {means}"
    means_to_sems = dict(zip(means, sems))

    fig, ax = plt.subplots(figsize=(5, 5))
    venn_obj = venn2(subsets={"10": means.fast_minus_slow,
                              "01": means.slow_minus_fast,
                              "11": means.overlap},
                     set_colors=SET_COLORS, alpha=.9,
                     subset_label_formatter=lambda x: f"{x:.2f}%\n±{means_to_sems[x]:.2f}%",
                     set_labels=("Fast", "Slow"), ax=ax)
    venn_obj.get_label_by_id("11").set_bbox(dict(facecolor='white', alpha=0.8,
                                                 linewidth=0))
    intersection = venn_obj.get_patch_by_id("11")
    intersection.set_hatch("//")
    intersection.set_facecolor(SET_COLORS[0])
    intersection.set_edgecolor(SET_COLORS[1])
    hatchlinewidth_before = mpl.rcParams['hatch.linewidth']
    mpl.rcParams['hatch.linewidth'] = HATCH_LINEWIDTH
    ax.set_title(f"{br_str} - Neurons unique and common between fast and slow"
                 f"\nfor neurons active at least {min_prcnt_valid}% trials")
    if save_figs:
        assert fig_save_prefix is not None, "Pass fig_save_prefix to save"
        save_fp = pathlib.Path(
            f"{fig_save_prefix}/FastSlowVenn/valid_{min_prcnt_valid}%_{br_str}.svg")
        print(f"Saving to: {save_fp}")
        save_fp.parent.mkdir(exist_ok=True)
        plt.savefig(save_fp, bbox_inches='tight')
    plt.show()
    mpl.rcParams['hatch.linewidth'] = hatchlinewidth_before
    return overlaps


def loopBRPlotVennOverlap(all_df, fast_df, slow_df, min_prcnt_valid, save_figs,
                          fig_save_prefix=None):
    '''Figure 4G: one Venn per brain region, MFC then LFC.'''
    res = {}
    for br, all_br_df in all_df.groupby("BrainRegion"):
        br_str = str(BrainRegion(br)).split("_")[0]
        res[br_str] = plotVennOverlap(
            fast_df[fast_df.BrainRegion == br], slow_df[slow_df.BrainRegion == br],
            min_prcnt_valid=min_prcnt_valid,
            sessions=all_br_df.groupby("ShortName").groups.keys(),
            br_str=br_str, save_figs=save_figs, fig_save_prefix=fig_save_prefix)
    return res
