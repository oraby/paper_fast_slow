'''Rigid / stretching neuron proportions (Figure 4K right, and its bars).

Backend for the "Pie-Chart for rigid/stretching neurons" section of
``2p_1_population.ipynb``.

A neuron counts as **stretching** when its peak time correlates with sampling
duration past ``corr_thresh`` (|r| > 0.3 in the paper), and as **rigid** when
it does not; the AUC correlation is treated the same way, so a neuron can be
in one group, both, or neither.

- ``plotCorrThreshPieChart`` (Figure 4K right) -- one pie per region plus a
  pooled one, over the four groups, saved per region.

The region bars at the same threshold live in
:mod:`..twop.plot.corrthreshregions` (Figure S10D).
'''
from __future__ import annotations

import matplotlib.pyplot as plt

from ..common.definitions import BrainRegion

def _processSet(br_str, br_df, filter_key, corr_thresh, save_figs, fig_save_prefix):
    firing_pos_corr_key = filter_key + "firing_pos_pearson_corr"
    auc_key_corr_key = filter_key + "amplitude_firing_pearson_corr"

    fir_pos_traces_ids = set(br_df[br_df[firing_pos_corr_key].abs() >
                             corr_thresh].long_trace_id)
    auc_traces_ids = set(br_df[br_df[auc_key_corr_key].abs() >
                        corr_thresh].long_trace_id)
    both_traces_ids = fir_pos_traces_ids.intersection(auc_traces_ids)
    fir_pos_only_traces_ids = fir_pos_traces_ids.difference(both_traces_ids)
    auc_only_traces_ids = auc_traces_ids.difference(both_traces_ids)

    # "Rigid" = correlated with neither metric, so subtract the *union* of the
    # two sets (subtracting only Both would leave every single-metric neuron in
    # Rigid and push the wedges past 100%).
    non_traces_ids = set(br_df.long_trace_id).difference(
                         fir_pos_traces_ids.union(auc_traces_ids))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([len(fir_pos_only_traces_ids), len(both_traces_ids),
            len(auc_only_traces_ids), len(non_traces_ids)],
            labels=["Firing Pos", "Both", "AUC", "Rigid"],
            colors=["C1", "olive", "C5", "gray"],
            startangle=90, counterclock=False,
            hatch=["", "xx", "", ""],
            pctdistance=1.37,
            autopct="%1.1f%%")
    ax.set_title(f"{br_str} Correlation with Sampling Time\n"
                    f"Correlation Threshold = {corr_thresh}")
    if save_figs:
        assert fig_save_prefix is not None, "Pass fig_save_prefix to save"
        fig.savefig(f"{fig_save_prefix}/{br_str}_correlation_pie_chart.pdf",
                    bbox_inches='tight')
    plt.show()

def plotCorrThreshPieChart(df, filter_key, corr_thresh, save_figs, fig_save_prefix):
    df = df.copy()
    df["long_trace_id"] = df.ShortName + "_" + df.trace_id.astype(str)

    _processSet("Both_MFC_LFC", df, filter_key, corr_thresh, save_figs,
                fig_save_prefix)
    for br, br_df in df.groupby("BrainRegion"):
        br_ = BrainRegion(br)
        br_str = str(BrainRegion(br_)).split("_")[0]
        br_str = "MFC" if br_str == "M2" else ("LFC" if br_str == "ALM" else br_str)
        _processSet(br_str, br_df, filter_key, corr_thresh, save_figs, fig_save_prefix)

