'''Tuning to priors vs current-trial variables (Figure 6E).

Backend for the "Sampling & Feedback Venn Diagarams" section of
``2pAnalysis.ipynb``, drawn from the per-neuron tuning table that
:mod:`.tunedneurons` builds.

A neuron counts as **prior-tuned** when it is tuned to any variable whose name
starts with ``Prev``, and **current-tuned** when it is tuned to any of the
others. One Venn per epoch per region, plus a pooled "Both areas" panel drawn
first.

**The percentages are per session, averaged across sessions** (mean and SEM),
with each session's own neuron count as the denominator. The prior-only and
current-only numbers have the overlap subtracted, so the three add up to the
tuned fraction and the rest are untuned.

**The circles are not to scale with the labels.** ``venn2`` is handed the three
means *divided by the overlap mean*, so the overlap circle is always 1 and the
other two are relative to it; the printed labels are then overwritten with the
real percentages. The axes are widened by ``max_combined_ratio`` so the larger
circle still fits. Keep that in mind before "fixing" the areas: only the labels
carry the quantities.

Extracted from the notebook unchanged except that ``fig_save_prefix`` is a
parameter (it was a notebook global).
'''
from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_venn import venn2
from scipy.stats import sem

from ..common.definitions import BrainRegion
from .tunedneurons import ID_COLS

#: Variables measured on the previous trial share this prefix.
PRIOR_PREFIX = "Prev"
FONT_SIZE = 15
HATCH_LINEWIDTH = 3
POOLED_LABEL = "Both areas"


def tuningPercentages(pref_df):
    '''Per session: prior-only, current-only, both and untuned percentages.'''
    rows = []
    for sess, sess_df in pref_df.groupby("ShortName"):
        prior_cols = [c for c in sess_df.columns if c.startswith(PRIOR_PREFIX)]
        cur_cols = [c for c in sess_df.columns
                    if not c.startswith(PRIOR_PREFIX) and c not in ID_COLS]
        prior_tuning = sess_df[prior_cols].notnull().any(axis=1)
        cur_tuning = sess_df[cur_cols].notnull().any(axis=1)
        len_traces = len(sess_df)
        assert len_traces == sess_df.trace_id.nunique(), \
            f"Trace ID mismatch: {len_traces} != {sess_df.trace_id.nunique()} for {sess}"

        both_sum = (prior_tuning & cur_tuning).sum()
        prior_sum = prior_tuning.sum() - both_sum
        cur_sum = cur_tuning.sum() - both_sum
        pct = lambda n: n / len_traces * 100
        rows.append({"ShortName": sess, "prior": pct(prior_sum),
                     "current": pct(cur_sum), "both": pct(both_sum),
                     "none": pct(len_traces - (prior_sum + cur_sum + both_sum))})
    return rows


def _plotBrainRegionTuning(pref_df, epoch_str, br_str, save_figs=False,
                           fig_save_prefix=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Prior and Current Tuning for {br_str} during {epoch_str}")
    rows = tuningPercentages(pref_df)
    values = {key: np.array([r[key] for r in rows])
              for key in ("prior", "current", "both", "none")}
    means = {key: np.mean(arr) for key, arr in values.items()}
    sems = {key: sem(arr) for key, arr in values.items()}

    print(f"Brain Region: {br_str}, "
          f"Priors Tuning: {means['prior']:.1f}% ± {sems['prior']:.1f}%, "
          f"Current Tuning: {means['current']:.1f}% ± {sems['current']:.1f}%, "
          f"Both Tuned: {means['both']:.1f}% ± {sems['both']:.1f}%, "
          f"No Tuning: {means['none']:.1f}% ± {sems['none']:.1f}%")

    combined_priors = means["prior"] + means["both"]
    combined_current = means["current"] + means["both"]
    max_combined = max(combined_priors, combined_current)
    max_combined_ratio = 1 + (1 - max_combined / 100)
    print("Sizes of combined priors: ", combined_priors, " current: ",
          combined_current, " max_combined_ratio: ", max_combined_ratio)

    # Areas relative to the overlap; the labels below carry the real numbers.
    v = venn2(subsets=(means["prior"] / means["both"],
                       means["current"] / means["both"],
                       means["both"] / means["both"]),
              set_labels=('Priors', 'Current'), ax=ax)
    for subset_id, key in [('10', "prior"), ('01', "current"), ('11', "both")]:
        label = v.get_label_by_id(subset_id)
        label.set_text(f"{means[key]:.1f}%\n±{sems[key]:.1f}%")
        label.set_fontsize(FONT_SIZE)
    v.get_patch_by_id('10').set_color('C0')
    v.get_patch_by_id('01').set_color('C1')
    v.get_patch_by_id('11').set_color('C0')
    v.get_patch_by_id('11').set_hatch('xx')
    v.get_patch_by_id('11').set_edgecolor('C1')

    hatch_width_before = mpl.rcParams['hatch.linewidth']
    mpl.rcParams['hatch.linewidth'] = HATCH_LINEWIDTH
    v.get_patch_by_id('11').set_linewidth(1)

    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(x_lim[0] * max_combined_ratio, x_lim[1] * max_combined_ratio)
    ax.set_ylim(y_lim[0] * max_combined_ratio, y_lim[1] * max_combined_ratio)

    if save_figs:
        assert fig_save_prefix is not None, "Pass fig_save_prefix to save"
        save_fp = pathlib.Path(f"{fig_save_prefix}/PriorCurrentTuning/"
                               f"{br_str}_{epoch_str}_prior_current_tuning.svg")
        save_fp.parent.mkdir(exist_ok=True)
        fig.savefig(save_fp, dpi=300, bbox_inches="tight")
    plt.show()
    mpl.rcParams['hatch.linewidth'] = hatch_width_before
    return means, sems


def plotPriorCurrentTuning(pref_df, epoch_str, save_figs=False,
                           fig_save_prefix=None):
    '''Figure 6E: the pooled panel, then one per region.'''
    res = {POOLED_LABEL: _plotBrainRegionTuning(
        pref_df=pref_df, epoch_str=epoch_str, br_str=POOLED_LABEL,
        save_figs=save_figs, fig_save_prefix=fig_save_prefix)}
    for br, br_df in pref_df.groupby("BrainRegion"):
        br_str = str(BrainRegion(br)).split("_")[0]
        res[br_str] = _plotBrainRegionTuning(
            pref_df=br_df, epoch_str=epoch_str, br_str=br_str,
            save_figs=save_figs, fig_save_prefix=fig_save_prefix)
    return res
