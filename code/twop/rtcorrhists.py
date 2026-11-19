'''Distributions of the rt/activity correlations (Figure S10C).

Backend for the "Correlation Hists" section of ``2p_1_population.ipynb``, drawn
from the per-neuron correlation table.

Each neuron contributes two correlations against sampling duration: when it
peaked (**firing position**) and how much area its rises covered (**AUC**).
The panels show how those are distributed, against a shuffled control built by
redrawing each neuron's trials.

- ``plotCorrDistributions`` (S10C) -- histograms of the correlation
  coefficients, real against shuffled, split by region, with the fraction past
  ``corr_thresh`` reported. It also returns the per-neuron table the pie charts
  downstream are built from.

**A neuron only enters once its trials survive filtering.** ``filter_key``
picks which filtering the upstream step applied (``"zscore_filter_"`` in the
paper), and ``time_offset`` is the AUC window's start in seconds.
'''
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common.clr import BrainRegion as BRClr
from ..common.definitions import BrainRegion

def plotCorrDistributions(df, filter_key, time_offset, incld_std, save_figs,
                          corr_thresh, res_shuffle_df=None,
                          fig_save_prefix=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    mean_str = "(mean + Std.)" if incld_std else "mean"

    corr_res_dict = {"BrainRegion":[], "ShortName":[],
                     "when":[], "metric":[], "mean":[], "std":[], "is_shuffle":[], }

    for br, br_df in df.groupby("BrainRegion"):
        # NB `br` stays the raw region code on purpose: it is what gets stored
        # in corr_res_dict below and what BRClr / BrainRegion() are given.
        # int() because BRClr's isinstance(.., int) check misses numpy ints.
        c = BRClr[int(br)]

        br_df = br_df.copy()
        br_df["long_trace_id"] = br_df.ShortName + "_" + br_df.trace_id.astype(str)

        x_key = filter_key + "firing_pos_pearson_corr"
        y_key = filter_key + "amplitude_firing_pearson_corr"

        if res_shuffle_df is not None:
            br_shuffle_df = res_shuffle_df[res_shuffle_df.BrainRegion == br]
            br_shuffle_df = br_shuffle_df.copy()
            br_shuffle_df["long_trace_id"] = br_shuffle_df.ShortName + "_" + br_shuffle_df.trace_id.astype(str)

        start_mean_col, start_std_col = filter_key + "start_max_firing_pos_mean", filter_key + "start_max_firing_pos_std"
        end_mean_col, end_std_col = filter_key + "end_max_firing_pos_mean", filter_key + "end_max_firing_pos_std"

        early_max_firing_pos_df = br_df[br_df[start_mean_col] +
                                        (br_df[start_std_col] if incld_std else 0)
                                         <= time_offset]
        late_max_firing_pos_df = br_df[br_df[end_mean_col] +
                                       (br_df[end_std_col] if incld_std else 0)
                                        <= time_offset]

        early_traces_ids = set(early_max_firing_pos_df.long_trace_id)
        late_traces_ids = set(late_max_firing_pos_df.long_trace_id)

        UNION_EARLY_LATE = False
        if UNION_EARLY_LATE:
            extra_mid_str = "Not early or late"
            early_and_late_traces_ids = early_traces_ids.union(late_traces_ids)
            mid_traces_ids = set(br_df.long_trace_id).difference(early_and_late_traces_ids)
        else:
            ADD_MID_LATE = True
            extra_mid_str = (f"with {time_offset}s > trial start {mean_str}")
            if not ADD_MID_LATE:
                extra_mid_str = (f"{extra_mid_str} and \n"
                       f"trial end {mean_str} > {time_offset}s from trial end)")
            mid_df = br_df[(br_df[start_mean_col] - (br_df[start_std_col] if incld_std else 0) > time_offset) &
                           (br_df[end_mean_col]   - (br_df[end_std_col] if incld_std else 0)   > time_offset)]
            mid_traces_ids = set(mid_df.long_trace_id)
            if ADD_MID_LATE:
                mid_traces_ids = (mid_traces_ids | late_traces_ids) - early_traces_ids

        extra_early_str = f"Early firing pos (< {time_offset}s)"

        def _calcAboveCorrThresh(data, total_data_len):
            # % of `data` itself, i.e. of the neurons in this time span.
            return 100*(abs(data) > corr_thresh).sum() / total_data_len

        def _plotHistPrcnt(ax, traces_ids_set, df_col, is_shuffle, title):
            above_corr_thresh_prcnt = above_corr_thresh_std = np.nan
            if not is_shuffle:
                data = br_df[br_df.long_trace_id.isin(traces_ids_set)][df_col]
                alpha = 1
                num_iterations = 1
                label = (f"{100*len(traces_ids_set)/len(br_df):.3g}% "
                         f"of total neurons")
                if len(data):
                    above_corr_thresh_prcnt = _calcAboveCorrThresh(data, len(br_df))
                    above_corr_thresh_std = 0
                    label += (f"\nwhere a {above_corr_thresh_prcnt:.3g}% "
                              f"subset of total neurons are correlated")
            else:
                data_shuffle = br_shuffle_df[br_shuffle_df.long_trace_id.isin(traces_ids_set)]
                data = data_shuffle[df_col]
                alpha = .3
                num_iterations = br_shuffle_df.shuffle_num.nunique()
                label = "Same neurons shuffled"
                if len(data):
                    groups = data_shuffle.groupby("shuffle_num")[df_col].apply(
                                                           _calcAboveCorrThresh, total_data_len=len(br_df))
                    above_corr_thresh_prcnt = groups.mean()
                    above_corr_thresh_std = groups.std()
                    label += (f" - correlated: "
                              f"{above_corr_thresh_prcnt:.3g}%")
            if len(data):
                pre_str = (f"{title} ({'shuffled' if is_shuffle else 'real'}) "
                           f"above correlation threshold ")
                blank_pre = " " * len(pre_str)
                print(pre_str   + f"prcnt = {above_corr_thresh_prcnt:.2f}%" )
                if is_shuffle:
                    print(blank_pre + f"  std = {above_corr_thresh_std:.2f}%")

            bins = np.arange(-1, 1.1, .1)
            hist, _ = np.histogram(data, bins)
            hist = hist.astype(float) / num_iterations
            hist = 100 * hist / len(br_df)
            ax.stairs(hist, bins, label=label, color=c, lw=2, alpha=alpha,
                      fill=False)
            if not is_shuffle and res_shuffle_df is not None:
                _plotHistPrcnt(ax, traces_ids_set, df_col, is_shuffle=True,
                               title=title)

        print("Brain Region:", str(BrainRegion(br)).split("_")[0])
        _plotHistPrcnt(ax1, early_traces_ids, y_key, is_shuffle=False,
                       title="Early seq amplitude Corr")

        _plotHistPrcnt(ax2, mid_traces_ids, y_key, is_shuffle=False,
                       title="Remaining seq amplitude Corr")
        _plotHistPrcnt(ax3, mid_traces_ids, x_key, is_shuffle=False,
                       title="Remaining seq firing pos Corr")
        for sess, sess_df in br_df.groupby("ShortName"):
            for col in (y_key, x_key):
                for when in ("early", "remaining"):
                    if when == "early":
                        whens_traces_ids = early_traces_ids
                    else:
                        whens_traces_ids = mid_traces_ids
                    when_df = sess_df[sess_df.long_trace_id.isin(whens_traces_ids)]
                    if when_df.empty:
                        continue
                    for is_shuffle in (False, True):
                        if is_shuffle and res_shuffle_df is None:
                            continue # No shuffle to compare against
                        if not is_shuffle:
                            mean = _calcAboveCorrThresh(when_df[col], len(br_df))
                            std = 0
                        else:
                            sess_shuffle_df = \
                                  br_shuffle_df[br_shuffle_df.ShortName == sess]
                            when_shuffle_df = sess_shuffle_df[
                                  sess_shuffle_df.long_trace_id.isin(whens_traces_ids)]
                            _shuffle_prcnts = when_shuffle_df.groupby(
                                "shuffle_num")[col].apply(_calcAboveCorrThresh, total_data_len=len(br_df))
                            mean = _shuffle_prcnts.mean()
                            std = _shuffle_prcnts.std()
                        corr_res_dict["BrainRegion"].append(br)
                        corr_res_dict["ShortName"].append(sess)
                        corr_res_dict["when"].append(when)
                        corr_res_dict["metric"].append(col)
                        corr_res_dict["mean"].append(mean)
                        corr_res_dict["std"].append(std)
                        corr_res_dict["is_shuffle"].append(is_shuffle)

    ax1.set_title(f"DF/F activity correlation with Sampling Time\n"
                  f"for neurons active Early (trial start {mean_str} ≤ {time_offset}s)\n{extra_early_str}")
    ax2.set_title(f"DF/F activity correlation with Sampling Time\n"
                  f"for neurons active {extra_mid_str}")
    ax3.set_title(f"Peak Firing Position correlation with Sampling Time\n"
                  f"for neurons active {extra_mid_str}")
    ax1.set_xlabel("Firing correlation with Sampling Time")
    ax2.set_xlabel("Firing correlation with Sampling Time")
    ax3.set_xlabel("Peak Firing Position Correlation with Sampling Time")
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(-1, 1)
        ax.axvline(0, ls="--", c="gray")
        ax.spines[["top", "bottom", "right", "left"]].set_visible(False)
        ax.legend(loc="upper left")
        ax.set_ylabel("% Neurons")
    if save_figs:
        assert fig_save_prefix is not None, "Pass fig_save_prefix to save"
        save_fp = f"{fig_save_prefix}/rt_corr_{filter_key}offset_{time_offset}.pdf"
        print("Save fp=", save_fp)
        plt.savefig(save_fp)
    plt.show()

    return pd.DataFrame(corr_res_dict)
