'''Single-neuron rt/activity panels and their correlations (Figures 4K left, S10A).

Backend for the plotting half of the "Correlation between rt and activity on
single cell level" section of ``plottraces3.ipynb``.

``loopNeuronsPlot`` walks every neuron, correlates its peak time and its area
against the trial's sampling duration, and -- when asked -- draws the example
panels behind Figure 4K left and S10A. It returns a **list of (keys, results)
pairs**, which ``makeDF`` downstream turns into the correlation table the
histograms and pies read.

``loopNeuronsSummary`` (the notebook's "Take 2") answers a narrower question
and returns a **DataFrame**, one row per trial, for the trial-time statistics.

**The two were both called ``loopNeuronsPlot``** in the notebook, in cells one
after another, so the second silently replaced the first: anything later that
wanted the list -- ``createShuffle``, which rebuilds the shuffled control --
would have received a DataFrame instead. They are named apart here. The same
went for ``plotNeuronCorr`` and ``_plotFilteredComb``.

``shuffle_corr`` redraws each neuron's trial order before correlating, which is
how the shuffled control in Figures S10B-C is built.

Extracted from the notebook unchanged, except for those names, and that
``KEY_PREFIX``, the sampling window and ``fig_save_prefix`` are parameters.
'''
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from pathlib import Path
from scipy.stats import iqr, linregress, pearsonr, pointbiserialr, sem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

from ..common.definitions import BrainRegion
from ..pipeline.utils import filterNanGaussianConserving
from .relogit.relogit import relogit
from .traceauc import getPosDeflections, getTraceAUC

_COLORS = [mpl.colors.to_rgb('r'), mpl.colors.to_rgb('yellow')]
FastSlow_CM = LinearSegmentedColormap.from_list("FastSlowCM", _COLORS, N=500)
#: Frames per second of the imaging.
ACQ_RATE = 30
#: Seconds kept either side of the sampling epoch, as the frames were cut.
TIME_BEFORE_SAMPLING = 0.1
TIME_AFTER_MOVEMENT = 0.1

def _plotCorr(ax, x, y, shuffle_corr=False, title_prefix="", use_logistic=False):
    if shuffle_corr:
        y = np.random.permutation(y) # Same as shuffle but shuffle() is in-place
    if use_logistic:
        STEP = .1
        MIN_SAMPLES = 2
        bins = np.arange(0, 5 + STEP, STEP)
        x_sort_idxs = np.argsort(x)
        x = x.values
        if not shuffle_corr:
            y = y.values
        x = x[x_sort_idxs]
        y = y[x_sort_idxs]

        x_idxs = np.digitize(x, bins)
        x = [bin for bin_idx, bin in enumerate(bins, 1)
             if np.sum(x_idxs == bin_idx) > MIN_SAMPLES]
        y = np.array([np.mean(y[x_idxs == i]) for i in range(1, len(bins))
                      if np.sum(x_idxs == i) > MIN_SAMPLES])

    linregress_stats = linregress(x, y)
    pearson_corr = linregress_stats.rvalue
    pearson_pval = linregress_stats.pvalue # i.e pearson p-val
    slope_angle = np.degrees(np.arctan(linregress_stats.slope))
    # pearson_stats = pearsonr(x, y)
    # pearson_corr = pearson_stats.statistic
    # pearson_pval = pearson_stats.pvalue
    # slope_angle = np.degrees(np.arctan(pearson_corr))
    if ax is not None:
        # ax.plot(x, linregress_stats.intercept + linregress_stats.slope*x, c='g')
        title_prefix = title_prefix + " - " if len(title_prefix) else ""
        ax.set_title(ax.get_title() + f"\n{title_prefix}"
                    f"$r$={pearson_corr:.2f} - $r^{{2}}$={pearson_corr**2:.2f} - "
                    f"$r_{{pval}}$={pearson_pval:.2f} - "
                    f"Slope={slope_angle:.2f}°",
                    y=0.95)
    return dict(pearson_corr=pearson_corr, pearson_pval=pearson_pval,
                slope=slope_angle)

def _plotFilteredComb(axs, neuron_df, key_prefix, max_trial_sec, colors,
                      filter_name, shuffle_corr):
    if shuffle_corr:
        assert axs is None
    if axs is not None:
        ax1, ax2, ax3, ax4 = axs
    else:
        ax1, ax2, ax3, ax4 = None, None, None, None

    # Assign color as part of df so that it would survive the filtering
    neuron_df = neuron_df.copy()
    neuron_df["color"] = colors

    Y_IS_FIRING_PROB = False
    neuron_df = neuron_df
    neuron_no_decay_active_df = neuron_df[neuron_df.is_active]

    x_corr_all = neuron_df.trial_dur_sec
    x_corr_active = neuron_no_decay_active_df.trial_dur_sec

    x = neuron_no_decay_active_df[f"{key_prefix}max_pos_sec"] - TIME_BEFORE_SAMPLING
    if Y_IS_FIRING_PROB:
        y = neuron_df.is_active
        use_logistic = True
        x_corr_for_y = x_corr_all
    else:
        y = neuron_no_decay_active_df[f"{key_prefix}auc"]
        # y = neuron_no_decay_active_df[f"{key_prefix}max_amplitude"]
        use_logistic = False
        x_corr_for_y = x_corr_active
    start_max_firing_pos_mean, start_max_firing_pos_std = x.mean(), x.std()
    start_max_firing_median, start_max_firing_iqr = x.median(), x.quantile(.75) - x.quantile(.25)
    end_max_firing_pos_mean, end_max_firing_pos_std  = (x_corr_active-x).mean(), (x_corr_active-x).std()
    end_max_firing_median, end_max_firing_iqr = (x_corr_active-x).median(), (x_corr_active-x).quantile(.75) - (x_corr_active-x).quantile(.25)
    if axs is not None:
        df_clrs = neuron_no_decay_active_df.color
        # ax1.set_title(f"Max Firing Pos - {filter_name}")
        # ax1.set_ylabel("AUC")
        ax1.set_ylabel("Max Amplitude (s)")
        ax1.set_xlabel("Max Firing Position (s)")
        ax1.scatter(x, y, color=df_clrs)
        # ax1.set_xlim(-TIME_BEFORE_SAMPLING, max(x) + TIME_AFTER_MOVEMENT)
        ax1.set_xlim(-TIME_BEFORE_SAMPLING, max_trial_sec + TIME_AFTER_MOVEMENT)
        # ax1.set_ylim(-TIME_BEFORE_SAMPLING, max(x) + TIME_AFTER_MOVEMENT)
    corr_firing_pos = _plotCorr(ax1, x_corr_active, x, shuffle_corr, "Max Firing Pos")
    corr_amplitude_firing = _plotCorr(ax1, x_corr_for_y, y, shuffle_corr, #"AUC"
                                      "Max Amplitude", use_logistic=use_logistic)

    if axs is not None:
        traces = neuron_df[f"{key_prefix}trace"].to_numpy()
        ax2.set_title(f"Sampling aligned - {filter_name}")
        ax3.set_title(f"Movement aligned - {filter_name}")

        max_trace_len = max([len(trace) for trace in traces])
        rng = np.arange(max_trace_len).astype(float) / ACQ_RATE

        traces_dict = {"trace_time": [], "trace": [], "color": []}
        for trace, clr in zip(traces, colors):
            #trace = filterNanGaussianConserving(trace, sigma=1, axis=0)
            # alpha = 0.2 if decay_start else 1
            # ls = None #if np.isnan(decay_thresh) or decay_start else "--"
            # ax2.plot(rng[:len(trace)]        - TIME_BEFORE_SAMPLING, trace, c=clr, alpha=alpha, ls=ls)
            # ax3.plot(-rng[:len(trace)][::-1] + TIME_AFTER_MOVEMENT,  trace, c=clr, alpha=alpha, ls=ls)
            # # Align around max, i.e max at 0
            # max_trace_pos_idx = np.argmax(trace)
            # shifted_x = rng[:len(trace)] - rng[max_trace_pos_idx]
            # ax4.plot(shifted_x, trace, c=clr, alpha=alpha, ls=ls)
            traces_dict["trace_time"].append(len(trace)/ACQ_RATE - TIME_BEFORE_SAMPLING - TIME_AFTER_MOVEMENT)
            traces_dict["trace"].append(trace)
            traces_dict["color"].append(clr)
        traces_df = pd.DataFrame(traces_dict)
        STEP = .4
        # display(traces_df)
        cut = pd.cut(traces_df.trace_time, bins=np.arange(0, max_trial_sec + STEP,
                                                          STEP))
        for mean_time, traces_df in traces_df.groupby(cut):
            if len(traces_df) < 5:
                continue
            min_len_trace = traces_df[traces_df.trace_time == traces_df.trace_time.min()].trace.iloc[0]
            min_trace_len = len(min_len_trace)
            traces_cut_start = traces_df.trace.apply(lambda trace:trace[:min_trace_len])
            traces_cut_end = traces_df.trace.apply(lambda trace:trace[-min_trace_len:])

            trace_start_mean = traces_cut_start.mean(axis=0)
            clrs = np.asarray(traces_df.color.to_list())
            # clrs = clrs.reshape(-1, 4)
            mean_clr = np.mean(clrs, axis=0)
            x_start = np.arange(len(trace_start_mean)) / ACQ_RATE - TIME_BEFORE_SAMPLING
            ax2.plot(x_start, trace_start_mean, c=mean_clr)
            if len(traces_cut_start) > 1:
                trace_start_sem = sem(np.asarray(traces_cut_start.to_list()),  axis=0)
                ax2.fill_between(x_start, trace_start_mean-trace_start_sem, trace_start_mean+trace_start_sem,
                                 color=mean_clr, alpha=.2)

            trace_end_mean = traces_cut_end.mean(axis=0)
            trace_end_sem = sem(np.asarray(traces_cut_end.to_list()), axis=0)
            x_end = -x_start[:len(trace_start_mean)][::-1] + TIME_AFTER_MOVEMENT
            ax3.plot(x_end, trace_end_mean, c=mean_clr)
            if len(traces_cut_end) > 1:
                ax3.fill_between(x_end, trace_end_mean-trace_end_sem, trace_end_mean+trace_end_sem,
                                 color=mean_clr, alpha=.2)

            # Try to align around max
            # max_trace_pos_idx = traces_df.traces.apply(lambda trace:np.argmax(trace))
            min_trace_max_pos_idx = min_len_trace.argmax()
            len_before_max, len_after_max = min_trace_max_pos_idx, min_trace_len - min_trace_max_pos_idx
            # Filter neurons that can't be aligned. TODO: Use median
            traces = [trace[trace.argmax()-len_before_max:trace.argmax()+len_after_max]
                      for trace in traces_df.trace
                      if trace.argmax() >= len_before_max and len(trace) - trace.argmax() >= len_after_max]
            if not len(traces):
                continue
            mean_trace = np.mean(traces, axis=0)
            x = np.arange(len(mean_trace)) / ACQ_RATE - len_before_max / ACQ_RATE
            ax4.plot(x, mean_trace, c=mean_clr)
            if len(traces) > 1:
                sem_trace = sem(traces, axis=0)
                ax4.fill_between(x, mean_trace-sem_trace, mean_trace+sem_trace,
                                 color=mean_clr, alpha=.2)

        ax1.spines[["top", "right"]].set_visible(False)

        [ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
        for ax in (ax2, ax3, ax4)]
        [ax.axhline(0, c="gray", ls="--") for ax in (     ax2, ax3, ax4)]
        [ax.axvline(0, c="gray", ls="--") for ax in (ax1, ax2, ax3, ax4)]
        # ax1.axline((0, 0), (1, 1), ls="--", c="gray")

        [ax.set_xlabel("Sampling Time (s)") for ax in (ax2, ax3, ax4)]
        [ax.set_ylabel("DF/F")              for ax in (ax2, ax3, ax4)]
        ax1.set_xlabel("Max Firing Position (s)")
        ax4.set_title("Aligned around max firing position")

    return dict(firing_pos=corr_firing_pos,
                # rltv_firing_pos=corr_rltv_firing_pos,
                amplitude_firing=corr_amplitude_firing,
                start_max_firing_pos_mean=start_max_firing_pos_mean,
                start_max_firing_pos_std=start_max_firing_pos_std,
                start_max_firing_median=start_max_firing_median,
                start_max_firing_iqr=start_max_firing_iqr,
                end_max_firing_pos_mean=end_max_firing_pos_mean,
                end_max_firing_pos_std=end_max_firing_pos_std,
                end_max_firing_median=end_max_firing_median,
                end_max_firing_iqr=end_max_firing_iqr,)

def _plotSingleTraces(neuron_df, key_prefix, neuron_iqr_idx, colors, axs,):
    # Get top 3 traces with the maxium height
    traces = neuron_df[f"{key_prefix}trace"]
    clrs = pd.Series(colors) # Only to have the same treatment as others
    trial_nums = neuron_df["TrialNumber"]

    NUM_TRACES = 3
    RAND_IDXS = True

    use_iqr_idxs = len(neuron_iqr_idx) >= NUM_TRACES

    if RAND_IDXS:
        possible_idxs = np.arange(len(traces))
        if use_iqr_idxs:
            possible_idxs = possible_idxs[neuron_iqr_idx]
        selected_idxs = np.random.choice(possible_idxs,
                                            NUM_TRACES, replace=False)
    else:
        traces_maxes = [trace.max() for trace in traces]
        # print("Traces maxes:", traces_maxes)
        selected_idxs = np.argsort(traces_maxes)[-NUM_TRACES:][::-1]

    def processSeries(s):
        s = s.to_numpy() # Otherwuse indexing not working
        # print("max_ind:", max_ind)
        s = s[selected_idxs]
        return s

    selected_traces, clrs, trial_nums = [
        processSeries(s) for s in (traces, clrs, trial_nums)]

    for trace, ax, clr, trial_num in zip(
            selected_traces, axs, clrs, trial_nums):
        rng = (np.arange(len(trace)) / ACQ_RATE) - TIME_BEFORE_SAMPLING

        ax.plot(rng, trace, c=clr)
        # print("new trace")

        positive_deflect_idxs = getPosDeflections(trace)
        (integrated_auc, traces_thresh_li, traces_rng_li,
         traces_min_max_idxs_li, masks_li) = getTraceAUC(trace,
                                                          positive_deflect_idxs)

        assert len(traces_thresh_li) == len(traces_rng_li)
        assert len(traces_thresh_li) == len(traces_min_max_idxs_li)
        assert len(traces_thresh_li) == len(masks_li)

        trace_mask = np.full_like(trace, False, dtype=bool)
        for sub_trace_thresh, (start_idx, end_idx), (min_idx, max_idx),     subtrace_mask in zip(
            traces_thresh_li,  traces_rng_li,       traces_min_max_idxs_li, masks_li):
            # print("Start:", start_idx, "End:", end_idx)
            # Draw horizontal line at threshold from start of sub-trace to end
            ax.plot(rng[start_idx:end_idx],
                    [sub_trace_thresh]*(end_idx-start_idx),
                    c="k", ls="--")
            # Draw vertical line at max-point idx from min to max
            ax.plot([rng[max_idx], rng[max_idx]],
                    [trace[min_idx], trace[max_idx]],
                    c="gray", ls="--", alpha=.5)
            # print("Start idx:", start_idx, "End idx:", end_idx)
            # print("Mask start idx:", mask_start_idx, "Mask end idx:", mask_end_idx)
            if not any(subtrace_mask):
                continue
            # print("Subtrace mask:", subtrace_mask)
            trace_mask[:] = False
            trace_mask[start_idx:end_idx] = subtrace_mask
            trc_fill_high = trace.copy()
            trc_fill_high[~trace_mask] = np.nan
            # trc_fill_low = trc_fill_high.copy()
            # trc_fill_low[trace >= sub_trace_thresh] = sub_trace_thresh
            # Anything below the threshold should be filled with the min value
            # trc_fill_low[trace < sub_trace_thresh] = trace[min_idx]
            ax.fill_between(rng, trace[min_idx], #trc_fill_low,
                            trc_fill_high, color=clr, alpha=0.3)
        # print()

        ax.axvline(0, c="gray", ls="--")
        ax.axvline(rng[-1] - TIME_AFTER_MOVEMENT, c="gray", ls="--")
        ax.spines[["top", "bottom", "right", "left"]].set_visible(False)
        ax.axhline(0, c="gray", ls="--")
        ax.set_title(f"Example Trace - Trial Number: {trial_num}\n"
                     f"Min={trace.min():.2f} - Max={trace.max():.2f} - AUC={integrated_auc:.2f}",
                      y=.95)
        ax.set_xlabel("Sampling Time (s)")
        # Check whether legend is empty before calling legend()
        handles, labels = ax.get_legend_handles_labels()
        if len(labels):
            ax.legend()

def plotNeuronCorr(br_sess_trace_id, neuron_df, key_prefix, max_trial_sec,
                   plot, save_figs, shuffle_corr, fig_save_prefix=None):
    br, sess, trace_id = br_sess_trace_id
    br = str(BrainRegion(br)).split("_")[0]

    if plot or save_figs:
        fig, rows_cols_axs = plt.subplots(5, 3, figsize=(20, 28))
        fig.suptitle(
                f"BrainRegion: {br} - Session: {sess} - Trace-id: {trace_id}\n"
                f"Num traces = {len(neuron_df)}", y=.925)

        # Skip the last row in each col
        col1_axs1, col2_axs, col3_axs = rows_cols_axs[:4].T
    else:
        col1_axs1, col2_axs, col3_axs = [None]*3

    mean, std = neuron_df.trial_dur_sec.mean(), neuron_df.trial_dur_sec.std()
    Z_SCORE = 3
    zscore_3 = Z_SCORE*std + mean
    #
    median, iqr_val = neuron_df.trial_dur_sec.median(), iqr(neuron_df.trial_dur_sec)
    iqr_1_5_val = median + 1.5*iqr_val

    if plot:
        col1_axs1[0].axvline(zscore_3, c="gray", ls="--")
        col1_axs1[0].axvline(iqr_1_5_val, c="r", ls="--")

    # colors = np.array([f"C{idx}" for idx in np.arange(len(neuron_df))])
    # display(neuron_df)
    time_lims = 0.3, 3
    trial_dur_sec = neuron_df.trial_dur_sec.copy()
    trial_dur_sec[trial_dur_sec < time_lims[0]] = time_lims[0]
    trial_dur_sec[trial_dur_sec > time_lims[1]] = time_lims[1]
    assert trial_dur_sec.min() >= time_lims[0]
    assert trial_dur_sec.max() <= time_lims[1], f"{trial_dur_sec.max()} > {time_lims[1]}"
    # Normalize trial_dur_sec between 0 and 1 based on time_lims
    trial_dur_norm = (trial_dur_sec - time_lims[0]) / (time_lims[1] - time_lims[0])
    trial_dur_norm = LogNorm()(trial_dur_norm)
    colors = FastSlow_CM(trial_dur_norm)
    colors = [tuple(clr) for clr in colors]
    _tmp_arr = np.empty(len(colors), dtype=object)
    _tmp_arr[:] = colors
    colors = _tmp_arr
    # print("Colors:", colors)

    corr_no_filtering = _plotFilteredComb(col1_axs1, neuron_df, key_prefix,
                                          max_trial_sec,
                                          colors,
                                          "No Filter",
                                          shuffle_corr=shuffle_corr)

    neuron_iqr_idx = (neuron_df.trial_dur_sec <= iqr_1_5_val).to_numpy()
    corr_IQR_filtering = _plotFilteredComb(col2_axs,
                                           neuron_df[neuron_iqr_idx],
                                           key_prefix,
                                           max_trial_sec,
                                           colors[neuron_iqr_idx],
                                           "1.5*IQR Filter",
                                           shuffle_corr=shuffle_corr)

    neuron_zscore_idx = (neuron_df.trial_dur_sec <= zscore_3).to_numpy()
    corr_zscore_filtering =  _plotFilteredComb(col3_axs,
                                               neuron_df[neuron_zscore_idx],
                                               key_prefix,
                                               max_trial_sec,
                                               colors[neuron_zscore_idx],
                                               f"Z-Score <= {Z_SCORE} Filter",
                                               shuffle_corr=shuffle_corr)

    if save_figs or plot:
        single_traces_axs = rows_cols_axs[-1, :]
        _plotSingleTraces(neuron_df=neuron_df, key_prefix=key_prefix,
                          neuron_iqr_idx=neuron_iqr_idx,
                          colors=colors,
                          axs=single_traces_axs)
        if save_figs:
            br = "MFC" if br == "M2" else ("LFC" if br == "ALM" else br)
            assert fig_save_prefix is not None, "Pass fig_save_prefix to save"
            root_dir = Path(f"{fig_save_prefix}/SingleTraces/rt_corr/{br}")
            assert root_dir.parent.exists(), f"{root_dir} doesn't exist"
            root_dir.mkdir(exist_ok=True)
            fig.savefig(f"{root_dir}/{sess}_{trace_id}_avgd.svg", dpi=150)
            plt.close()
        elif plot:
            plt.show()


    return dict(no_filter=corr_no_filtering,
                iqr_filter=corr_IQR_filtering,
                zscore_filter=corr_zscore_filtering)

def loopNeuronsPlot(df, key_prefix, plot, save_figs, shuffle_corr=False,
                    fig_save_prefix=None):
    res_data = []
    early_bailout = False
    if save_figs:
        plot = False
    if shuffle_corr:
        assert not plot
        assert not save_figs
    for br, br_df in df.groupby("BrainRegion"):
        br_str = str(BrainRegion(br)).split("_")[0]
        # print(f"Processing: {br_str}")
        #br_neurons_df = neurons_traces_df[neurons_traces_df.BrainRegion == br]
        for sess, sess_df in tqdm(br_df.groupby("ShortName"),
                                  desc=f"Processing: {br_str}",
                                  leave=not shuffle_corr):
            max_trial_sec = sess_df.trial_dur_sec.max()
            #sess_neurons_df = br_neurons_df[br_neurons_df.ShortName == sess]
            for trace_id, neuron_df in sess_df.groupby("trace_id"):
                # GP4_85_s10_L70_D250_ALM_25
                # if not (sess == "GP4_28_S1_L41_D250_M2L" and trace_id == 237):
                #     continue
                # print(f"Processing: {br_str} - {sess} - {trace_id}")
                dict_res = plotNeuronCorr((br, sess, trace_id), neuron_df,
                                          key_prefix=key_prefix, plot=plot,
                                          max_trial_sec=max_trial_sec,
                                          shuffle_corr=shuffle_corr,
                                          save_figs=save_figs,
                                          fig_save_prefix=fig_save_prefix)
                id_dict = dict(BrainRegion=br, ShortName=sess,
                               trace_id=trace_id)
                res_data.append((id_dict, dict_res))
                if not save_figs and plot:
                    early_bailout = True
                    break # Plot a sample from each session

    return res_data if not early_bailout else None


def _summariseFilteredComb(neuron_df, key_prefix, label, colors):
    neuron_df = neuron_df.copy()
    neuron_df[f"{key_prefix}max_amplitude_rank"] = neuron_df[f"{key_prefix}max_amplitude"].rank(ascending=False)
    # neuron_df[f"{key_prefix}label"] = label
    # neuron_df[f"{key_prefix}color"] = colors
    return neuron_df


def summariseNeuronCorr(neuron_df, key_prefix):
    mean, std = neuron_df.trial_dur_sec.mean(), neuron_df.trial_dur_sec.std()
    Z_SCORE = 3
    zscore_3 = Z_SCORE*std + mean
    #
    median, iqr_val = neuron_df.trial_dur_sec.median(), iqr(neuron_df.trial_dur_sec)
    iqr_1_5_val = median + 1.5*iqr_val


    # colors = np.array([f"C{idx}" for idx in np.arange(len(neuron_df))])
    # display(neuron_df)
    time_lims = 0.3, 3
    trial_dur_sec = neuron_df.trial_dur_sec.copy()
    trial_dur_sec[trial_dur_sec < time_lims[0]] = time_lims[0]
    trial_dur_sec[trial_dur_sec > time_lims[1]] = time_lims[1]
    assert trial_dur_sec.min() >= time_lims[0]
    assert trial_dur_sec.max() <= time_lims[1], f"{trial_dur_sec.max()} > {time_lims[1]}"
    # Normalize trial_dur_sec between 0 and 1 based on time_lims
    trial_dur_norm = (trial_dur_sec - time_lims[0]) / (time_lims[1] - time_lims[0])
    trial_dur_norm = LogNorm()(trial_dur_norm)
    colors = FastSlow_CM(trial_dur_norm)
    colors = [tuple(clr) for clr in colors]
    _tmp_arr = np.empty(len(colors), dtype=object)
    _tmp_arr[:] = colors
    colors = _tmp_arr
    # print("Colors:", colors)

    corr_no_filtering = _summariseFilteredComb(neuron_df, key_prefix,
                                          "No Filter",
                                          colors)

    # neuron_iqr_idx = (neuron_df.trial_dur_sec <= iqr_1_5_val).to_numpy()
    # corr_IQR_filtering = _plotFilteredComb(neuron_df[neuron_iqr_idx],
    #                                        key_prefix,
    #                                        "1.5*IQR Filter"
    #                                        colors[neuron_iqr_idx])

    # neuron_zscore_idx = (neuron_df.trial_dur_sec <= zscore_3).to_numpy()
    # corr_zscore_filtering =  _plotFilteredComb(neuron_df[neuron_zscore_idx],
    #                                            key_prefix,
    #                                            f"Z-Score <= {Z_SCORE} Filter",
    #                                            colors[neuron_zscore_idx])


    # return pd.concat([corr_no_filtering, corr_IQR_filtering, corr_zscore_filtering])
    return corr_no_filtering

def loopNeuronsSummary(df, key_prefix, shuffle_corr=False):
    res_dfs = []
    early_bailout = False

    for br, br_df in df.groupby("BrainRegion"):
        br_str = str(BrainRegion(br)).split("_")[0]
        # print(f"Processing: {br_str}")
        #br_neurons_df = neurons_traces_df[neurons_traces_df.BrainRegion == br]
        for sess, sess_df in tqdm(br_df.groupby("ShortName"),
                                  desc=f"Processing: {br_str}",
                                  leave=not shuffle_corr):
            max_trial_sec = sess_df.trial_dur_sec.max()
            #sess_neurons_df = br_neurons_df[br_neurons_df.ShortName == sess]
            for trace_id, neuron_df in sess_df.groupby("trace_id"):

                res_df = summariseNeuronCorr(neuron_df,
                                        key_prefix=key_prefix)
                res_df["BrainRegion"] = br
                res_df["ShortName"] = sess
                res_df["trace_id"] = trace_id

                res_dfs.append(res_df)

    return pd.concat(res_dfs).reset_index(drop=True) if not early_bailout else None
