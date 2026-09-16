'''Active-neuron count against sampling duration (Figures S11A, S11B).

Backend for the "Plot Active Early then Late" section of
``plottraces3.ipynb``, drawn from the counts :mod:`.activeneuroncount` builds.

Three panels come out of the same machinery:

- **S11A middle** -- percent of neurons active early, against sampling
  duration, hard trials only (``plotEarlyActiveCount``);
- **S11B** -- the same for the last stretch of sampling
  (``plotLastActiveCount``);
- **S11A right** -- early activity against **performance** rather than
  duration (``plotEarlyActivePerf``).

**Percentages are per session**: each trial's count is divided by that
session's own neuron count, so a session with more neurons does not dominate.
Trials shorter than 0.3 s are dropped first.

The ``*Perf`` functions repeat the preprocessing and plotting rather than
sharing it -- kept as it is, since the two paths have drifted apart in small
ways and the published panels come from these exact copies.

Extracted from the notebook unchanged, except that ``fig_save_prefix`` and the
window labels are parameters.
'''
from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy import stats

from ..common.clr import BrainRegion as BRClr
from ..common.definitions import BrainRegion

def _commonPreprocess(df, start_or_end: Literal["start", "end"]):
    df = df.copy()
    df = df[df.trial_dur_sec >= .3]
    df["active_prcnt"] = 100*df.trial_active_count/df.total_count
    # total = df.total_count
    # total = df.trial_active_count
    # mean_active = df.trial_active_count.mean()
    # df["active_end_prcnt"] = 100*df.trial_end_max_active_count/total
    # df["active_extend_prcnt"] = 100*df.trial_end_extend_active_count/total
    # df["active_end_prcnt"] = 100*df.active_prcnt/df.total_count
    df_li = []
    for sess, sess_df in df.groupby("ShortName"):
        sess_df = sess_df.copy()
        # sess_mean_active = sess_df.trial_active_count.mean()
        sess_df[f"active_{start_or_end}_prcnt"] = \
            100*sess_df[f"trial_{start_or_end}_max_active_count"]/sess_df.total_count#sess_mean_active
        sess_df[f"active_extend_prcnt"] = \
            100*sess_df[f"trial_{start_or_end}_extend_active_count"]/sess_df.total_count#sess_mean_active
        df_li.append(sess_df)
    df = pd.concat(df_li)


    # df = df.copy()
    # df = df[df.trial_dur_sec >= .3]
    # df["active_prcnt"] = 100*df.trial_active_count/df.total_count
    # # total = df.total_count
    # # total = df.trial_active_count
    # # mean_active = df.trial_active_count.mean()
    # # df[f"active_{start_or_end}_prcnt"] = 100*df[f"trial_{start_or_end}_max_active_count"]/total
    # # df["active_extend_prcnt"] = 100*df[f"trial_{start_or_end}_extend_active_count"]/total
    # # df[f"active_{start_or_end}_prcnt"] = 100*df.active_prcnt/df.total_count
    # df_li = []
    # for sess, sess_df in df.groupby("ShortName"):
    #     sess_df = sess_df.copy()
    #     # sess_mean_active = sess_df.trial_active_count.mean()
    #     sess_df[f"active_{start_or_end}_prcnt"] = 100*sess_df[f"trial_{start_or_end}_max_active_count"]/sess_df.total_count#sess_mean_active
    #     sess_df["active_extend_prcnt"] = 100*sess_df.[f"trial_{start_or_end}_extend_active_count"]/sess_df.total_count#sess_mean_active
    #     df_li.append(sess_df)
    # df = pd.concat(df_li)

    return df

def _commonBRProcessing(br, br_df, ax_sess1, ax_sess2,
                        start_or_end : Literal["start", "end"],
                        PLOT_SESS=True):
    c = BRClr[br]

    xs_raw_sess = []
    ys_raw_sess = []
    y2s_raw_sess = []

    xs_z_sess = []
    ys_z_sess = []
    y2s_z_sess = []


    for sess, sess_df in br_df.groupby("ShortName"):
        sess_df = sess_df.sort_values("trial_dur_sec")
        zscored = stats.zscore(sess_df.trial_dur_sec)
        time_zscore_n2 = sess_df.trial_dur_sec.mean() - 2*sess_df.trial_dur_sec.std()
        time_zscore_n3 = sess_df.trial_dur_sec.mean() - 3*sess_df.trial_dur_sec.std()

        time_zscore_2 = sess_df.trial_dur_sec.mean() + 2*sess_df.trial_dur_sec.std()
        time_zscore_3 = sess_df.trial_dur_sec.mean() + 3*sess_df.trial_dur_sec.std()

        bins = np.arange(0, 3, .25)
        # print("Bins:", bins)
        cut_df = pd.cut(sess_df.trial_dur_sec, bins)
        sess_binned_df = sess_df.groupby(cut_df)
        x_raw = sess_binned_df.trial_dur_sec.mean()
        y_raw = sess_binned_df[f"active_{start_or_end}_prcnt"].mean()
        y2_raw = sess_binned_df.active_extend_prcnt.mean()
        xs_raw_sess.append(x_raw)
        ys_raw_sess.append(y_raw)
        y2s_raw_sess.append(y2_raw)
        if PLOT_SESS:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
            ax1.plot(x_raw[x_raw.notnull()], y_raw[y_raw.notnull()], c=c, lw=2)
            ax1.scatter(sess_df.trial_dur_sec, sess_df[f"active_{start_or_end}_prcnt"], c=c, s=1)
            # ax1.plot(x_raw[x_raw.notnull()], y2_raw[y2_raw.notnull()], c='k', lw=2)
            # ax1.scatter(sess_df.trial_dur_sec, sess_df.active_extend_prcnt, c='k', s=1)
            ax1.axvline(time_zscore_n3, ls="--", c="gray")
            ax1.axvline(time_zscore_n2, ls="--", c="gray")
            ax1.axvline(time_zscore_2, ls="--", c="gray")
            ax1.axvline(time_zscore_3, ls="--", c="gray")

        sess_df = sess_df.copy()
        sess_df["zscored_time_dur"] = zscored
        ZSCORE_CUT = 2
        sess_df = sess_df[(-ZSCORE_CUT <= sess_df.zscored_time_dur) &
                            (sess_df.zscored_time_dur <= ZSCORE_CUT)]
        #
        bins = np.arange(-1.5, 3, .5)
        # print("Bins:", bins)
        cut_df = pd.cut(sess_df.zscored_time_dur, bins)
        sess_binned_df = sess_df.groupby(cut_df)
        x_z = sess_binned_df.zscored_time_dur.mean()
        y_z = sess_binned_df[f"active_{start_or_end}_prcnt"].mean()
        y2_z = sess_binned_df.active_extend_prcnt.mean()
        xs_z_sess.append(x_z)
        ys_z_sess.append(y_z)
        y2s_z_sess.append(y2_z)
        if PLOT_SESS:
            ax2.plot(x_z[x_z.notnull()], y_z[y_z.notnull()], c=c, lw=2)
            ax2.scatter(sess_df.zscored_time_dur, sess_df.active_end_prcnt, c=c, s=1)
            # ax2.plot(x_z[x_z.notnull()], y2_z[y2_z.notnull()], c='k', lw=2)
            # ax2.scatter(sess_df.zscored_time_dur, sess_df.active_extend_prcnt, c='k', s=1)
            ax1.set_xlabel("Trial Duraton (s)")
            ax2.set_xlabel("Z-Score Time")
            ax1.set_ylabel("Percent Active")
            ax2.set_ylabel("Percent Active")
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
            # ax.set_xlim(0, 2)
            ax1.set_title(sess)
            ax2.set_title(f"Z-scored cut={ZSCORE_CUT}")
            fig.show()

    grp_raw = (xs_raw_sess, ys_raw_sess, y2s_raw_sess), ax_sess1
    grp_z = (xs_z_sess, ys_z_sess, y2s_z_sess), ax_sess2
    for (xs_sess, ys_sess, y2s_sess), ax_sess in (grp_raw, grp_z):
        xs_sess = np.array(xs_sess)
        ys_sess = np.array(ys_sess)
        y2s_sess = np.array(y2s_sess)
        x_sess_mean = np.nanmean(xs_sess, axis=0)
        ys_sess_mean = np.nanmean(ys_sess, axis=-0)
        y2s_sess_mean = np.nanmean(y2s_sess, axis=-0)

        ys_sess_sem = stats.sem(np.array(ys_sess), axis=0, nan_policy="omit")
        y2s_sess_sem = stats.sem(np.array(y2s_sess), axis=0, nan_policy="omit")
        # ys_sess_sem = np.nanstd(np.array(ys_sess), axis=-0)

        ax_sess.errorbar(x_sess_mean, ys_sess_mean, yerr=ys_sess_sem, c=c)
        # for x_sess, y_sess in zip(xs_sess, ys_sess):
        #     ax_sess.plot(x_sess, y_sess, c=c, alpha=.2)
        ax_sess.scatter(xs_sess, ys_sess, c=c, s=2)
        x_flat = np.array(xs_sess).flatten()
        y_flat = np.array(ys_sess).flatten()
        x_flat = x_flat[~np.isnan(y_flat)]
        y_flat = y_flat[~np.isnan(y_flat)]
        line_stats = stats.linregress(x_flat, y_flat)
        pearson_corr = line_stats.rvalue
        pearson_pval = line_stats.pvalue # i.e pearson p-val
        slope_angle = np.degrees(np.arctan(line_stats.slope))
        # print("Line Stats:", line_stats)
        x_fit = np.array([np.nanmin(x_sess_mean), np.nanmax(x_sess_mean)])
        ax_sess.plot(x_fit, line_stats.intercept + x_fit*line_stats.slope ,
                        c=c, ls="--", label=f"Pearson corr: {pearson_corr:.2f} - p-val: {pearson_pval:.2f}\n")

        # ax_sess.errorbar(x_sess_mean, y2s_sess_mean, yerr=y2s_sess_sem, c=c, ls="--")
        # ax_sess.plot(x_sess_mean, y2s_sess_mean-ys_sess_mean, c=c, ls=":")

def _commonPlot(df, y_label, start_or_end : Literal["start", "end"],
                PLOT_SESS=True):
    df = _commonPreprocess(df, start_or_end=start_or_end)

    fig_sess, (ax_sess1, ax_sess2) = plt.subplots(1, 2, figsize=(12, 6))
    for br, br_df in df.groupby("BrainRegion"):
        _commonBRProcessing(br, br_df, ax_sess1, ax_sess2,
                            start_or_end=start_or_end,
                            PLOT_SESS=PLOT_SESS)

    ax_sess1.set_xlabel("Trial Duration (s)")
    ax_sess2.set_xlabel("Trial Duration (z-score)")

    for ax_sess in (ax_sess1, ax_sess2):
        ax_sess.set_ylabel(y_label)
        ax_sess.spines[["top", "left", "right"]].set_visible(False)
        ax_sess.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
        ax_sess.set_ylim(bottom=0)
        ax_sess.legend()

    return fig_sess


def _commonPreprocessPerf(df, start_or_end : Literal["start", "end"]):
    df = df.copy()
    df = df[df.trial_dur_sec >= .3]
    df["active_prcnt"] = 100*df.trial_active_count/df.total_count

    df_li = []
    for sess, sess_df in df.groupby("ShortName"):
        sess_df = sess_df.copy()
        sess_df[f"active_{start_or_end}_prcnt"] = \
            100*sess_df[f"trial_{start_or_end}_max_active_count"]/sess_df.total_count
        sess_df["active_extend_prcnt"] = \
            100*sess_df[f"trial_{start_or_end}_extend_active_count"]/sess_df.total_count
        df_li.append(sess_df)
    df = pd.concat(df_li)

    df[f"active_{start_or_end}_prcnt"].hist(bins=np.arange(0, 30, 1))
    plt.show()

    return df

def _commonBRProcessingPerf(br, br_df, ax_sess1, ax_sess2,
                            start_or_end : Literal["start", "end"],
                            as_relative_perf=True, filter_outerliers=True,
                            PLOT_SESS=True):
    c = BRClr[br]

    xs_raw_sess = []
    ys_raw_sess = []

    xs_z_sess = []
    ys_z_sess = []

    for sess, sess_df in br_df.groupby("ShortName"):
        sess_df = sess_df[sess_df.DVstr == "Easy"]
        sess_df = sess_df.sort_values("trial_dur_sec")

        zscored = stats.zscore(sess_df.trial_dur_sec)
        time_zscore_n2 = sess_df.trial_dur_sec.mean() - 2*sess_df.trial_dur_sec.std()
        time_zscore_n3 = sess_df.trial_dur_sec.mean() - 3*sess_df.trial_dur_sec.std()
        time_zscore_2 = sess_df.trial_dur_sec.mean() + 2*sess_df.trial_dur_sec.std()
        time_zscore_3 = sess_df.trial_dur_sec.mean() + 3*sess_df.trial_dur_sec.std()

        bins = np.arange(0, 33, 3)
        # print("Bins:", bins)
        cut_df = pd.cut(sess_df[f"active_{start_or_end}_prcnt"], bins)
        sess_binned_df = sess_df.groupby(cut_df)
        if as_relative_perf:
            x_raw = 100*(sess_binned_df.ChoiceCorrect.mean() - sess_df.ChoiceCorrect.mean())
        else:
            x_raw = 100*sess_binned_df.ChoiceCorrect.mean()
            # print("X raw:", x_raw)
        y_raw = sess_binned_df[f"active_{start_or_end}_prcnt"].mean()
        # print("X raw:", x_raw)

        xs_raw_sess.append(x_raw)
        ys_raw_sess.append(y_raw)

        if PLOT_SESS:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
            ax1.plot(x_raw[x_raw.notnull()], y_raw[y_raw.notnull()], c=c, lw=2)
            ax1.scatter(sess_df.trial_dur_sec,
                        sess_df[f"active_{start_or_end}_prcnt"], c=c, s=1)
            # ax1.plot(x_raw[x_raw.notnull()], y2_raw[y2_raw.notnull()], c='k', lw=2)
            # ax1.scatter(sess_df.trial_dur_sec, sess_df.active_extend_prcnt, c='k', s=1)
            ax1.axvline(time_zscore_n3, ls="--", c="gray")
            ax1.axvline(time_zscore_n2, ls="--", c="gray")
            ax1.axvline(time_zscore_2, ls="--", c="gray")
            ax1.axvline(time_zscore_3, ls="--", c="gray")

        sess_df = sess_df.copy()
        sess_df["zscored_time_dur"] = zscored
        ZSCORE_CUT = 2
        sess_df = sess_df[(-ZSCORE_CUT <= sess_df.zscored_time_dur) &
                          (sess_df.zscored_time_dur <= ZSCORE_CUT)]
        #
        bins = np.arange(-1.5, 3, .5)
        # print("Bins:", bins)
        cut_df = pd.cut(sess_df.zscored_time_dur, bins)
        sess_binned_df = sess_df.groupby(cut_df)
        x_z = sess_binned_df.zscored_time_dur.mean()
        y_z = sess_binned_df[f"active_{start_or_end}_prcnt"].mean()

        xs_z_sess.append(x_z)
        ys_z_sess.append(y_z)

        if PLOT_SESS:
            ax2.plot(x_z[x_z.notnull()], y_z[y_z.notnull()], c=c, lw=2)
            ax2.scatter(sess_df.zscored_time_dur,
                        sess_df[f"active_{start_or_end}_prcnt"], c=c, s=1)

            ax1.set_xlabel("Trial Duraton (s)")
            ax2.set_xlabel("Z-Score Time")
            ax1.set_ylabel("Percent Active")
            ax2.set_ylabel("Percent Active")
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
            # ax.set_xlim(0, 2)
            ax1.set_title(sess)
            ax2.set_title(f"Z-scored cut={ZSCORE_CUT}")
            fig.show()

    grp_raw = (xs_raw_sess, ys_raw_sess), ax_sess1
    # grp_z = (xs_z_sess, ys_z_sess), ax_sess2
    for (xs_sess, ys_sess), ax_sess in (grp_raw, ):
        xs_sess = np.array(xs_sess)
        ys_sess = np.array(ys_sess)

        # above_50_perf = xs_sess > 70
        # xs_sess = xs_sess[above_50_perf]
        # ys_sess = ys_sess[above_50_perf]
        if filter_outerliers:
            if as_relative_perf:
                above_50_perf = (-20 < xs_sess) & (xs_sess < 20)
            else:
                above_50_perf = xs_sess > 70
                # above_50_perf = (50 < xs_sess) & (xs_sess < 100)
            xs_sess = xs_sess[above_50_perf]
            ys_sess = ys_sess[above_50_perf]

        # print("Xs sess:", xs_sess)
        # print("Ys sess:", ys_sess)

        x_sess_mean = np.nanmean(xs_sess, axis=0)
        ys_sess_mean = np.nanmean(ys_sess, axis=-0)

        ys_sess_sem = stats.sem(np.array(ys_sess), axis=0, nan_policy="omit")
        # ys_sess_sem = np.nanstd(np.array(ys_sess), axis=-0)
        # ax_sess.errorbar(x_sess_mean, ys_sess_mean, yerr=ys_sess_sem, c=c)

        # for x_sess, y_sess in zip(xs_sess, ys_sess):
        #     ax_sess.plot(x_sess, y_sess, c=c, alpha=.2)
        ax_sess.scatter(xs_sess, ys_sess, c=c, s=2)
        x_flat = np.array(xs_sess).flatten()
        y_flat = np.array(ys_sess).flatten()
        x_flat = x_flat[~np.isnan(y_flat)]
        y_flat = y_flat[~np.isnan(y_flat)]
        line_stats = stats.linregress(x_flat, y_flat)
        pearson_corr = line_stats.rvalue
        pearson_pval = line_stats.pvalue # i.e pearson p-val
        slope_angle = np.degrees(np.arctan(line_stats.slope))
        # print("Line Stats:", line_stats)
        x_fit = np.array([np.nanmin(x_flat), np.nanmax(x_flat)])
        ax_sess.plot(x_fit, line_stats.intercept + x_fit*line_stats.slope ,
                     c=c, ls="--",
                     label=f"Pearson corr: {pearson_corr:.2f} - p-val: {pearson_pval:.2f}\n")

        # ax_sess.errorbar(x_sess_mean, y2s_sess_mean, yerr=y2s_sess_sem, c=c, ls="--")
        # ax_sess.plot(x_sess_mean, y2s_sess_mean-ys_sess_mean, c=c, ls=":")

def _commonPlotPerf(df, y_label, start_or_end : Literal["start", "end"],
                    PLOT_SESS=True, as_relative_perf=True,
                    filter_outerliers=True):
    df = _commonPreprocessPerf(df, start_or_end=start_or_end)

    fig_sess, (ax_sess1, ax_sess2) = plt.subplots(1, 2, figsize=(12, 6))
    for br, br_df in df.groupby("BrainRegion"):
        _commonBRProcessingPerf(br, br_df, ax_sess1, ax_sess2,
                                start_or_end=start_or_end,
                                as_relative_perf=as_relative_perf,
                                filter_outerliers=filter_outerliers,
                                PLOT_SESS=PLOT_SESS)

    ax_sess1.set_xlabel("Relative Session Perf for Easy Trials (%)")

    ax_sess2.set_xlabel("Trial Duration (z-score)")

    for ax_sess in (ax_sess1, ax_sess2):
        ax_sess.set_ylabel(y_label)
        ax_sess.spines[["top", "left", "right"]].set_visible(False)
        ax_sess.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
        ax_sess.set_ylim(bottom=0)
        ax_sess.legend()

    return fig_sess


def plotEarlyActivePerf(df, save_figs, start_sampling_cutoff,
                        fig_save_prefix=None, PLOT_SESS=False):
    fig_sess = _commonPlotPerf(df,
                           y_label=f"% Active during early {start_sampling_cutoff:.2g}s Sampling / Total Session Neurons",
                           start_or_end="start",
                           as_relative_perf=False,
                           filter_outerliers=True,
                           PLOT_SESS=PLOT_SESS)
    _save(fig_sess, save_figs, fig_save_prefix,
          "rt_vs_early_active_neurons_perf_rel_corr")
    plt.show()   # plt.show(fig) only works under the inline backend
    return fig_sess


def plotEarlyActiveCount(df, save_figs, start_sampling_cutoff,
                         fig_save_prefix=None, PLOT_SESS=False):
    '''Figure S11A middle: early activity against sampling duration.'''
    fig_sess = _commonPlot(
        df, y_label=f"% Active during early {start_sampling_cutoff:.2g}s Sampling"
                    " / Total Session Neurons",
        PLOT_SESS=PLOT_SESS, start_or_end="start")
    _save(fig_sess, save_figs, fig_save_prefix, "rt_vs_early_active_neurons_corr")
    plt.show()   # plt.show(fig) only works under the inline backend
    return fig_sess


def plotLastActiveCount(df, save_figs, look_back_dur, fig_save_prefix=None,
                        PLOT_SESS=False):
    '''Figure S11B: late activity against sampling duration.'''
    fig_sess = _commonPlot(
        df, y_label=f"% Active during last {look_back_dur:.2g}% Sampling"
                    " / Total Session Neurons",
        PLOT_SESS=PLOT_SESS, start_or_end="end")
    _save(fig_sess, save_figs, fig_save_prefix, "rt_vs_last_active_neurons_corr")
    plt.show()   # plt.show(fig) only works under the inline backend
    return fig_sess


def _save(fig, save_figs, fig_save_prefix, name):
    if not save_figs:
        return
    assert fig_save_prefix is not None, "Pass fig_save_prefix to save"
    save_fp = f"{fig_save_prefix}/{name}.svg"
    print("Save fp=", save_fp)
    fig.savefig(save_fp)

