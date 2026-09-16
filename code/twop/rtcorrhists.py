'''Distributions of the rt/activity correlations (Figures S10B, S10C).

Backend for the "Correlation Hists" section of ``plottraces3.ipynb``, drawn
from the per-neuron correlation table.

Each neuron contributes two correlations against sampling duration: when it
peaked (**firing position**) and how much area its rises covered (**AUC**).
The panels show how those are distributed, against a shuffled control built by
redrawing each neuron's trials.

- ``plotCorrDistributions`` (S10C) -- histograms of the correlation
  coefficients, real against shuffled, split by region, with the fraction past
  ``corr_thresh`` reported. It also returns the per-neuron table the pie charts
  downstream are built from.
- ``plotCorrJointHist`` (S10B) -- the same two correlations against each other,
  as a stacked histogram coloured by AUC correlation (or a wireframe when
  ``PLOT_3D``).

**A neuron only enters once its trials survive filtering.** ``filter_key``
picks which filtering the upstream step applied (``"zscore_filter_"`` in the
paper), and ``time_offset`` is the AUC window's start in seconds.

Extracted from the notebook unchanged, except that ``fig_save_prefix``,
``SAVE_FIGS`` and the 3D switches are parameters.
'''
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.stats import iqr, linregress, pearsonr, sem

from ..common.clr import BrainRegion as BRClr
from ..common.definitions import BrainRegion

def plotCorrJointHist(df, filter_key, time_offset, incld_std, save_figs,
                      res_shuffle_df=None, fig_save_prefix=None,
                      plot_3d=False, combine_br=True):
    PLOT_3D, COMBINE_BR = plot_3d, combine_br
    subplots_kw = dict(projection='3d') if PLOT_3D else {}
    num_rows = 1 if COMBINE_BR else 2
    fig, all_axs = plt.subplots(num_rows, 2, figsize=(16, 6*num_rows),
                                subplot_kw=subplots_kw)
    mean_str = "(mean + Std.)" if incld_std else "mean"

    if COMBINE_BR:
        loop_axs = [all_axs]
        brs = [None]
    else:
        brs = (BrainRegion.M2_Bi, BrainRegion.ALM_Bi)
        loop_axs = all_axs
    for br, row_axs in zip(brs, loop_axs):
        print("Br:", br, type(br))
        if br is None:
            br_df = df
            c = "gray"
            br_str = "MFC + LFC"
        else:
            br_df = df[df.BrainRegion == br]
            br = BrainRegion(br)
            c = BRClr[br]
            br_str = str(br).split("_")[0]
            br_str = "MFC" if br_str == "M2" else ("LFC" if br_str == "ALM" else br_str)
        ax1, ax2 = row_axs

        x = br_df[filter_key + "firing_pos_pearson_corr"]
        y = br_df[filter_key + "amplitude_firing_pearson_corr"]

        if res_shuffle_df is not None:
            br_df = br_df.copy()
            br_df["long_trace_id"] = br_df.ShortName + "_" + br_df.trace_id.astype(str)
            br_shuffle_df = res_shuffle_df[res_shuffle_df.BrainRegion == br]
            br_shuffle_df = br_shuffle_df.copy()
            br_shuffle_df["long_trace_id"] = br_shuffle_df.ShortName + "_" + br_shuffle_df.trace_id.astype(str)
            x_shuffle = br_shuffle_df[filter_key + "firing_pos_pearson_corr"]
            y_shuffle = br_shuffle_df[filter_key + "amplitude_firing_pearson_corr"]

        start_mean_col, start_std_col = filter_key + "start_max_firing_pos_mean", filter_key + "start_max_firing_pos_std"
        end_mean_col, end_std_col = filter_key + "end_max_firing_pos_mean", filter_key + "end_max_firing_pos_std"

        early_max_firing_pos_df = br_df[br_df[start_mean_col] +
                                        (br_df[start_std_col] if incld_std else 0)
                                         <= time_offset]
        late_max_firing_pos_df = br_df[br_df[end_mean_col] +
                                       (br_df[end_std_col] if incld_std else 0)
                                        <= time_offset]
        early_index = early_max_firing_pos_df.index
        late_index = late_max_firing_pos_df.index

        if res_shuffle_df is not None:
            shuffle_early_df = br_shuffle_df[
                          br_shuffle_df.long_trace_id.isin(
                                  early_max_firing_pos_df.long_trace_id.values)]
            shuffle_late_df = br_shuffle_df[
                          br_shuffle_df.long_trace_id.isin(
                                   late_max_firing_pos_df.long_trace_id.values)]
            shuffle_early_idx = shuffle_early_df.index
            shuffle_late_idx = shuffle_late_df.index

        UNION_EARLY_LATE = False
        if UNION_EARLY_LATE:
            extra_mid_str = "Not early or late"
            early_and_late_index = early_index.union(late_index)
            mid_index = x.index[~x.index.isin(early_and_late_index)]
            if res_shuffle_df is not None:
                shuffle_mid_idx = x_shuffle.index[~x_shuffle.index.isin(
                                     shuffle_early_idx.union(shuffle_late_idx))]
        else:
            ADD_MID_LATE = True
            extra_mid_str = (f"with {time_offset}s > trial start {mean_str}")
            if not ADD_MID_LATE:
                extra_mid_str = (f"{extra_mid_str} and \n"
                       f"trial end {mean_str} > {time_offset}s from trial end)")
            # display(br_df[[start_mean_col, start_std_col, end_mean_col, end_std_col]])
            mid_index = br_df[(br_df[start_mean_col] - (br_df[start_std_col] if incld_std else 0) > time_offset) &
                              (br_df[end_mean_col]   - (br_df[end_std_col] if incld_std else 0)   > time_offset)].index
            if ADD_MID_LATE:
                mid_index = mid_index.union(late_index)
            if res_shuffle_df is not None:
                # TODO: Make a function do this
                shuffle_mid_idx = br_shuffle_df[(br_shuffle_df[start_mean_col] - (br_shuffle_df[start_std_col] if incld_std else 0) > time_offset) &
                                                (br_shuffle_df[end_mean_col]   - (br_shuffle_df[end_std_col] if incld_std else 0)   > time_offset)].index
                if ADD_MID_LATE:
                    shuffle_mid_idx = shuffle_mid_idx.union(shuffle_late_idx)



        _,            _ = _plotColorHist(y[early_index], x[early_index], ax=ax1,
                                         PLOT_3D=PLOT_3D,
                                         total_neurons=len(br_df), br_clr=c)
        color_map, norm = _plotColorHist(y[mid_index], x[mid_index], ax=ax2,
                                         PLOT_3D=PLOT_3D,
                                         total_neurons=len(br_df), br_clr=c)

        ax1.set_title(f"{br_str} - Early Seq (pos < {time_offset}s) Neurons correlations with Sampling Time\n"
                      f"for neurons active Early (trial start {mean_str} ≤ {time_offset}s)\n"
                      f"{100*len(early_index)/len(br_df):.3g}% of neurons",
                      color=c)

        ax2.set_title(f"{br_str} - Remaining seq correlation with Sampling Time\n"
                      f"for neurons active {extra_mid_str}\n"
                      f"{100*len(mid_index)/len(br_df):.3g}% of neurons",
                      color=c)


        ax1.set_xlabel("Max firing position with Sampling Time")
        ax2.set_xlabel("Max firing position with Sampling Time")

    if not PLOT_3D:
        plt.colorbar(plt.cm.ScalarMappable(cmap=color_map, norm=norm), ax=all_axs,
                    label="AUC correlation with Sampling Time", location="left",
                    fraction=.02, pad=.05)

    for ax in all_axs.flatten():
        ax.set_xlim(-1, 1)
        if PLOT_3D:
            ax.set_zlabel("% Neurons")
        else:
            ax.set_ylabel("% Neurons")
            ax.axvline(0, ls="--", c="gray")
        ax.spines[["top", "bottom", "right", "left"]].set_visible(False)
        # ax.legend(loc="upper left")
    # Set each column to the same y-lim
    if not PLOT_3D and not COMBINE_BR:
        # max_y = max([ax.get_ylim()[1] for ax in all_axs.flatten()])
        # for ax in all_axs.flatten():
        #     ax.set_ylim(0, max_y)
        for col_axs in all_axs.T:
            ax1, ax2 = col_axs
            max_y = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
            ax1.set_ylim(0, max_y)
            ax2.set_ylim(0, max_y)
    if save_figs:
        assert fig_save_prefix is not None, "Pass fig_save_prefix to save"
        save_fp = f"{fig_save_prefix}/rt_corr_{filter_key}offset_{time_offset}.svg"
        print("Save fp=", save_fp)
        plt.savefig(save_fp)
    plt.show()


def _plotColorHist(firing_pos_corr, auc_corr, total_neurons, ax, br_clr,
                   PLOT_3D=False):
    bins = np.arange(-1, 1.1, .1)
    assigned_firing_bin_idxs = np.digitize(firing_pos_corr, bins, right=True)
    assigned_auc_bin_idxs = np.digitize(auc_corr, bins, right=True)

    from ..common.clr import colorMapFijiArr
    cmap_arr = colorMapFijiArr()
    cmap_arr = np.concatenate([cmap_arr[::-1], cmap_arr])
    cmap_arr = np.c_[cmap_arr, np.ones(len(cmap_arr))]
    from matplotlib import colors as mplc
    color_map = mplc.LinearSegmentedColormap.from_list("matlab_divergance", cmap_arr)
    # color_map = plt.get_cmap("PiYG")
    norm = plt.Normalize(bins.min(), bins.max())

    bin_step = bins[1] - bins[0]
    bins_3d = np.zeros((len(bins), len(bins)))
    print("Bins shape:", bins_3d.shape)

    for firing_bin_idx in np.arange(len(bins)):
        cur_firing_idxs_mask = assigned_firing_bin_idxs == firing_bin_idx
        total_firing_count = cur_firing_idxs_mask.sum()
        height_all = 0
        height_prcnt = 0
        for auc_bin_idx in np.arange(len(bins)):
            both_idxs = (assigned_auc_bin_idxs == auc_bin_idx) & cur_firing_idxs_mask
            cur_auc_count = both_idxs.sum()
            height_all += cur_auc_count
            cur_auc_count = 100*cur_auc_count/total_neurons
            if cur_auc_count:
                auc_count_prcnt = 100*cur_auc_count/total_neurons
                # print("Auc total count=", cur_auc_count, "clr=", clr, "Height=", height)
                if PLOT_3D:
                    bins_3d[firing_bin_idx, auc_bin_idx] = cur_auc_count
                else:
                    clr = color_map(norm(bins[auc_bin_idx]))
                    ax.bar(bins[firing_bin_idx], auc_count_prcnt, bottom=height_prcnt,
                           color=clr, width=bin_step, align="edge")
                    height_prcnt += auc_count_prcnt

        assert height_all == total_firing_count

    if PLOT_3D:
        X, Y = np.meshgrid(bins, bins)
        ax.plot_wireframe(X, Y, bins_3d, color=br_clr, alpha=1)

    return color_map, norm


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
        # x = br_df[x_key]
        # y = br_df[y_key]

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
            # early_and_late_index = early_index.union(late_index)
            early_and_late_traces_ids = early_traces_ids.union(late_traces_ids)
            # mid_index = x.index[~x.index.isin(early_and_late_index)]
            mid_traces_ids = set(br_df.long_trace_id).difference(early_and_late_traces_ids)
        else:
            ADD_MID_LATE = True
            extra_mid_str = (f"with {time_offset}s > trial start {mean_str}")
            if not ADD_MID_LATE:
                extra_mid_str = (f"{extra_mid_str} and \n"
                       f"trial end {mean_str} > {time_offset}s from trial end)")
            # display(br_df[[start_mean_col, start_std_col, end_mean_col, end_std_col]])
            mid_df = br_df[(br_df[start_mean_col] - (br_df[start_std_col] if incld_std else 0) > time_offset) &
                           (br_df[end_mean_col]   - (br_df[end_std_col] if incld_std else 0)   > time_offset)]
            mid_traces_ids = set(mid_df.long_trace_id)
            if ADD_MID_LATE:
                mid_traces_ids = (mid_traces_ids | late_traces_ids) - early_traces_ids

        # extra_early_str = "Removed < 0 mean"
        extra_early_str = f"Early firing pos (< {time_offset}s)"
        # early_max_firing_pos_df = early_max_firing_pos_df[br_df[start_mean_col] > 0]
        # display(early_index)
        # display(late_index)
        # display(early_and_late_index)
        # display(mid_index)
        # display(late_max_firing_pos_df[late_max_firing_pos_df[filter_key + "firing_pos_pearson_corr"] < -.25][["BrainRegion", "ShortName", "trace_id"]])

        # print(len(early_max_firing_index))
        # s = 3
        # ax1.plot(x[early_index], y[early_index], #facecolors="k", #edgecolors=c,
        #         #markerfacecolor="none", #edgecolors=c,
        #         lw=0, ls="none", marker="o", c=c, ms=s, #alpha=.3
        #         label=f"{100*len(early_index)/len(br_df):.3g}% of neurons",
        #         )
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

        # ax2.plot(x[mid_late_idxs], y[mid_late_idxs],
        #         #markerfacecolor="none", #edgecolors=c,
        #         lw=0, ls="none", marker="o", c=c, ms=s, #alpha=.3
        #         label=f"{100*len(mid_late_idxs)/len(br_df):.3g}% of neurons",
        #         )
        # ax3.plot(x[late_index], y[late_index],
        #         #markerfacecolor="none", #edgecolors=c,
        #         lw=0, ls="none", marker="o", c=c, ms=s, #alpha=.3
        #         label=f"{100*len(late_index)/len(br_df):.3g}% of neurons",
        #         )
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
                            # std = when_shuffle_df.groupby("shuffle_num").apply(
                            #       lambda iter_df: _calcAboveCorrThresh(iter_df[col])).std()
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
    # ax3.set_title(f"Active Late (trial end {mean_str} ≤ {time_offset}s from trial end)")
    # for ax in (ax1, ax2, ax3):
    #     ax.set_xlim(-1, 1)
    #     ax.set_ylim(-1, 1)
    #     ax.set_xlabel("Max Firing poss corr")
    #     ax.set_ylabel("AUC corr")
    #     ax.axhline(0, ls="--", c="gray")
    #     ax.axvline(0, ls="--", c="gray")
    #     ax.spines[["top", "bottom", "right", "left"]].set_visible(False)
    #     ax.legend(loc="lower left")
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
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

    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.scatter(df.iqr_filter_rltv_firing_pos_slope,
    #         df.iqr_filter_amplitude_firing_slope)
    # ax.set_xlabel("Max Firing poss Slope")
    # ax.set_ylabel("Max Amplitude Slope")
    # ax.axhline(0, ls="--", c="gray")
    # ax.axvline(0, ls="--", c="gray")
    # ax.spines[["top", "bottom", "right", "left"]].set_visible(False)
    # plt.show()

    return pd.DataFrame(corr_res_dict)
