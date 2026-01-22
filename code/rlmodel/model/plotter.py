from .logic import makeOneRun
from .runlogger import RunLogger
from .util import PsychometricPlot
from ...behavior.bias import calcBias
from ...behavior.rewardrate import calcAvgRewardRate
from ...common import clr
from ...figcode.psychometric import (
                       _psychAxes, _slowFastPsychSubject, _fitPsych, _getGroups,
                       getGroupsDVstr)
from ...figcode.stbydifficulty import _rtVsDifficulty
from ...pipeline.pipeline import Chain
from ...pipeline.behavior import CountContPrevOutcome
from ...figcode.prevoutcomecurquantile import (
                         _plotPrevOutcomeCurQuantile, stimulusTimeByPrevOutcome)
from itertools import chain
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
from matplotlib.ticker import AutoLocator
import numpy as np
import pandas as pd
from scipy import stats
import time
import os

_NUM_CPUS = os.cpu_count()
run_logger : RunLogger = None
last_df = None
SINGLE_WIN_LOSE_UPDATE = True


def createFig(is_small_fig_mode, fig=None):
    if not is_small_fig_mode:
        return _createFigFull(fig)
    else:
        return _createFigSmall(fig)


def runAndPlot(df, fig, axs, include_Q, include_RewardRate, biasFn, driftFn,
               noiseFn,  plot_bias_dir, psych_plot : PsychometricPlot,
               DRIFT_COEF, NOISE_SIGMA, BOUND, ALPHA, BETA,
               NON_DECISION_TIME, t_dur, dt, is_small_fig_mode,
               dvs_filter=None,
               biasFn_df_cols=[], biasFn_kwargs={},
               driftFn_df_cols=[], driftFn_kwargs={},
               noiseFn_df_cols=[], noiseFn_kwargs={},
               is_loss_no_dir=False, cached_subject_df=None,
               verbose=True):
    # print("Updating plots")

    # Save just the last run
    global run_logger
    if run_logger is not None:
        run_logger.clear()

    time_now = time.time()

    keep_cols = ["Name", "SessId", "TrialNumber", "valid", "DVstr", "DV", "DVabs",
                 "ChoiceCorrect", "ChoiceLeft", "GUI_TimeOutIncorrectChoice",
                 "SimChoiceCorrect", "SimChoiceLeft", "calcStimulusTime", "SimRT",
                 "SimStartingPoint"]
    # Keep RewardRate* columns
    keep_cols += [col for col in df.columns
                  if col.startswith("RewardRate") and col != "RewardRate"]
    if include_Q:
        keep_cols += ["Q_val", "Q_L", "Q_R"]
    if include_RewardRate:
        keep_cols += ["RewardRate"]
    # df = df[keep_cols]
    PROFILE = False
    if PROFILE:
        import cProfile
        p = cProfile.Profile()
        ret = p.runcall(makeOneRun, df, include_Q=include_Q, include_RewardRate=include_RewardRate,
                        biasFn=biasFn,   biasFn_df_cols=biasFn_df_cols,   biasFn_kwargs=biasFn_kwargs,
                        driftFn=driftFn, driftFn_df_cols=driftFn_df_cols, driftFn_kwargs=driftFn_kwargs,
                        noiseFn=noiseFn, noiseFn_df_cols=noiseFn_df_cols, noiseFn_kwargs=noiseFn_kwargs,
                        ALPHA=ALPHA, BETA=BETA, NON_DECISION_TIME=NON_DECISION_TIME,
                        DRIFT_COEF=DRIFT_COEF, NOISE_SIGMA=NOISE_SIGMA,
                        BOUND=BOUND, dt=dt, t_dur=t_dur, is_loss_no_dir=is_loss_no_dir,
                        return_df=True)
        # p.print_stats()
        p.dump_stats("/home/main/OnedriveFloatingPersonal/caiman/TwoP/again/profile.prof")
    else:
        ret = makeOneRun(df, include_Q=include_Q, include_RewardRate=include_RewardRate,
                        biasFn=biasFn,   biasFn_df_cols=biasFn_df_cols,   biasFn_kwargs=biasFn_kwargs,
                        driftFn=driftFn, driftFn_df_cols=driftFn_df_cols, driftFn_kwargs=driftFn_kwargs,
                        noiseFn=noiseFn, noiseFn_df_cols=noiseFn_df_cols, noiseFn_kwargs=noiseFn_kwargs,
                        ALPHA=ALPHA, BETA=BETA, NON_DECISION_TIME=NON_DECISION_TIME,
                        DRIFT_COEF=DRIFT_COEF, NOISE_SIGMA=NOISE_SIGMA,
                        BOUND=BOUND, dt=dt, t_dur=t_dur, is_loss_no_dir=is_loss_no_dir, return_df=True)
    loss, df = ret
    df = df[keep_cols].copy()
    if verbose:
        print("Loss:", loss)
        print(f"Running simulation time: {time.time() - time_now:.2f}"); time_now = time.time()
    # display(df.head(10))
    # display(df[df.valid].tail(10))
    df = calcAvgRewardRate(df, choice_cols=["ChoiceCorrect", "SimChoiceCorrect"],
                           rr_postfixs=["", "Sim"], groupby_cols=["SessId"])

    df = _assignPrevTrial(df)
    if verbose:
        print(f"Assigning previous trial time: {time.time() - time_now:.2f}"); time_now = time.time()
    df = df[df.valid]
    if verbose:
        print("Number of valid trials:", len(df))
    if axs is None:
        return loss, df


    subject = df.Name.iloc[0]
    fig.suptitle(f"{subject} - Loss: {loss:,.2f}")

    global last_df
    last_df = df

    plotPlots(df, axs, include_Q, include_RewardRate, plot_bias_dir, psych_plot,
                BOUND, t_dur, dt, is_small_fig_mode, dvs_filter=dvs_filter,
                biasFn_kwargs=biasFn_kwargs)
    if verbose:
        print(f"Plotting time: {time.time() - time_now:.2f}"); time_now = time.time()
    return loss, df

def plotPlots(df, axs, include_Q, include_RewardRate,
              plot_bias_dir, psych_plot : PsychometricPlot, BOUND,  t_dur, dt,
              is_small_fig_mode, dvs_filter=None,  biasFn_kwargs={},):
    time_now = time.time()
    if dvs_filter is not None:
        df = df[df.DV.isin(dvs_filter)]

    if not is_small_fig_mode:
        (ax_hist_corr_up, ax_hist_corr_down,
         ax_hist_dir_up, ax_hist_dir_down,
         ax_rt_easy, ax_rt_medium, ax_rt_hard,
         ax_psych, ax_prev_out_cur_q, ax_reward_rate, ax_motor_bias,
         ax_bias, ax_qval, ax_prev_outs_rt, *axs_win_lose_update
        ) = axs.flatten()
        if SINGLE_WIN_LOSE_UPDATE:
            ax_win_lose_update = axs_win_lose_update[0]
        else:
            (ax_win_lose_update_q1, ax_win_lose_update_q2,
                ax_win_lose_update_q3) = axs_win_lose_update

    else:
        (ax_hist_corr_up, ax_hist_corr_down,
         ax_psych, ax_reward_rate, ax_qval) = axs.flatten()
        # Needed to pass to the functions below
        ax_hist_dir_up, ax_hist_dir_down = None, None
        ax_rt_easy, ax_rt_medium, ax_rt_hard = None, None, None
        ax_bias = None
        plot_bias_dir = False

    for ax in axs.flatten():
        for line in ax.lines:
            # Skip if axvline or axhline
            if len(line.get_xdata()) == 2:
                continue
            line.remove()
        # Remove all bars
        for bar in ax.patches:
            bar.remove()
        # Also remove BarContainers to prevent lingering legend handles
        for cont in list(ax.containers):
            if isinstance(cont, BarContainer):
                ax.containers.remove(cont)
        # Remove any existing legend so it doesn't retain old entries
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    # Error bars are not remove with bar.remove() above
    ax_psych.clear()
    _psychAxes(ax=ax_psych, combine_sides=False)
    if not is_small_fig_mode:
        ax_prev_out_cur_q.clear()
        ax_prev_out_cur_q.set_title("Prev Outcome Cur Quantile")
        ax_prev_outs_rt.clear()
        ax_prev_outs_rt.set_title("Prev Outcomes RT")
        ax_motor_bias.clear()

        if SINGLE_WIN_LOSE_UPDATE:
            ax_win_lose_update.clear()
        else:
            ax_win_lose_update_q1.clear()
            ax_win_lose_update_q2.clear()
            ax_win_lose_update_q3.clear()

    print(f"Clearing time: {time.time() - time_now:.2f}"); time_now = time.time()


    _plotPsychs(df, ax_psych, psych_plot)
    print(f"Psychometric: {time.time() - time_now:.2f}"); time_now = time.time()

    _plotHists(df, ax_hist_corr_up, ax_hist_corr_down, ax_hist_dir_up, ax_hist_dir_down,
               ax_rt_easy, ax_rt_medium, ax_rt_hard, t_dur, dt,
               is_small_fig_mode=is_small_fig_mode)
    print(f"Hists: {time.time() - time_now:.2f}"); time_now = time.time()

    _plotDists(df, ax_qval, ax_reward_rate, ax_bias, plot_bias_dir,
               include_Q, include_RewardRate, BOUND, biasFn_kwargs,
               is_small_fig_mode=is_small_fig_mode)
    print(f"Dists: {time.time() - time_now:.2f}"); time_now = time.time()


    if not is_small_fig_mode:
        _plotMotorBias(df, ax_motor_bias)
        _plotPrevOutcomeQuantile(df, ax_prev_out_cur_q)
        # print(f"Prev Outcome Cur Quantile: {time.time() - time_now:.2f}"); time_now = time.time()
        _plotPrevOutcomeCount(df, ax_prev_outs_rt)
        if SINGLE_WIN_LOSE_UPDATE:
            _plotStaySwitch(df, ax_win_lose_update)
        else:
            _plotStaySwitchQ(df, ax_win_lose_update_q1,
                                 ax_win_lose_update_q2,
                                 ax_win_lose_update_q3)


def _createFigSmall(fig=None):
    if fig is None:
        fig = plt.figure(figsize=(6, 6))

    top_row_subfigs, bottom_row_subfigs = fig.subfigures(2, 1, wspace=.001)
    hist_corr_fig, rate_vs_diff_fig = top_row_subfigs.subfigures(1, 2)
    psych_fig, reward_rate_q_rate_dist_fig = bottom_row_subfigs.subfigures(1, 2)

    hist_corr_fig.suptitle("RT Hist (correct/incorrect)")
    hist_corr_axs = hist_corr_fig.subplots(2, 1, sharex=True,
                                           gridspec_kw={"hspace":0})
    hist_corr_axs[1].invert_yaxis()
    hist_corr_axs[0].set_ylabel("Correct")
    hist_corr_axs[1].set_ylabel("Incorrect")
    hist_corr_axs[0].set_title("RT Correct", alpha=0)
    hist_corr_axs[1].set_title("RT Incorrect", alpha=0)

    rate_vs_diff_ax = rate_vs_diff_fig.subplots()
    # rate_vs_diff_fig.suptitle("RT vs Difficulty")

    psych_ax = psych_fig.subplots()
    # psych_fig.suptitle("Psychometric")
    # _psychAxes(ax=psych_ax)

    reward_rate_ax, q_rate_dist_ax = reward_rate_q_rate_dist_fig.subplots(2, 1)
    reward_rate_ax.set_title("Reward Rate Dist.")
    # q_rate_dist_ax.set_title("Q-Value Dist.")
    q_rate_dist_ax.set_xlabel("Q-Value Dist.")

    axs = np.asarray(list(hist_corr_axs) +
                     [rate_vs_diff_ax, psych_ax, reward_rate_ax, q_rate_dist_ax])
    for ax in axs.flatten():
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    return fig, axs

def _createFigFull(fig=None):
    if fig is None:
        fig = plt.figure(figsize=(20, 8)) #, layout="constrained")
    # else:
    #     fig.clear()
    top_row_subfigs, bottom_row_subfigs = fig.subfigures(2, 1)#wspace=.007, hspace=.1)
    (hist_corr_fig, hist_dir_fig, psych_fig,
     prev_cur_q_fig) = top_row_subfigs.subfigures(1, 4)

    hist_corr_fig.suptitle("RT Hist (correct/incorrect)")
    hist_corr_axs = hist_corr_fig.subplots(2, 1, sharex=True, gridspec_kw={"hspace":0})
    # hist_corr_fig.suubplots_adjust(hspace=0)
    hist_corr_axs[1].invert_yaxis()
    hist_corr_axs[0].set_ylabel("Correct")
    hist_corr_axs[1].set_ylabel("Incorrect")
    # Create invisible title for the callback ti know which figure was clicked
    hist_corr_axs[0].set_title("RT Correct", alpha=0)
    hist_corr_axs[1].set_title("RT Incorrect", alpha=0)

    hist_dir_fig.suptitle("RT Hist (direction)")
    hist_dir_axs = hist_dir_fig.subplots(2, 1, sharex=True, gridspec_kw={"hspace":0})
    hist_dir_axs[1].invert_yaxis()
    hist_dir_axs[0].set_ylabel("Left")
    hist_dir_axs[1].set_ylabel("Right")
    hist_dir_axs[0].set_title("RT Left", alpha=0)
    hist_dir_axs[1].set_title("RT Right", alpha=0)

    psych_ax = psych_fig.subplots()
    psych_ax.set_title("Psychometric")
    _psychAxes(ax=psych_ax)

    if SINGLE_WIN_LOSE_UPDATE:
        prev_cur_q_fig.suptitle("Prev Outcome Cur Quantile")
        prev_out_cur_q_ax, win_lose_update_ax = prev_cur_q_fig.subplots(2, 1)
        prev_cur_q_fig.subplots_adjust(hspace=0.5)
        win_lose_update_ax.set_title("Win/Lose Stay Update")
    else:
        prev_out_cur_q_fig, win_lose_update_fig = prev_cur_q_fig.subfigures(2, 1)
        prev_out_cur_q_ax = prev_out_cur_q_fig.subplots()
        # win_lose_update_fig.suptitle("Win/Lose Stay Update")
        win_lose_update_q_axs = win_lose_update_fig.subplots(1, 3)
    prev_out_cur_q_ax.set_title("Prev Outcome Cur Quantile")



    (rt_diff_fig, reward_rate_fig, bias_qval_fig,
     prev_outs_rt_fig) = bottom_row_subfigs.subfigures(1, 4)
    rt_diff_axs = rt_diff_fig.subplots(3, 1)
    rt_diff_fig.suptitle("RT Difficulties")
    rt_diff_axs[0].set_title("Easy", fontsize="x-small")
    rt_diff_axs[1].set_title("Medium", fontsize="x-small")
    rt_diff_axs[2].set_title("Hard", fontsize="x-small")
    rt_diff_axs[0].sharex(rt_diff_axs[2])
    rt_diff_axs[1].sharex(rt_diff_axs[2])
    rt_diff_axs[2].set_xlabel("Time (s)")
    rt_diff_axs[0].get_xaxis().set_visible(False)
    rt_diff_axs[1].get_xaxis().set_visible(False)

    reward_rate_ax, motor_bias_ax  = reward_rate_fig.subplots(2, 1)
    reward_rate_ax.set_title("Reward Rate Dist.")
    motor_bias_ax.set_title("Motor Bias Dist.")

    bias_ax, qval_ax = bias_qval_fig.subplots(2, 1)
    bias_qval_fig.subplots_adjust(hspace=0.5)
    bias_ax.set_title("Bias Dist. (Fixed for Correct/Incorrect)")
    qval_ax.set_title("Q-Value Dist.")

    prev_outs_rt_fig.suptitle("Prev Outcomes RT")
    prev_outs_rt_ax = prev_outs_rt_fig.subplots()

    axs = np.asarray(list(hist_corr_axs) + list(hist_dir_axs) + list(rt_diff_axs) +
                     [psych_ax, prev_out_cur_q_ax, reward_rate_ax,
                      motor_bias_ax, bias_ax, qval_ax, prev_outs_rt_ax] +
                      ([win_lose_update_ax] if SINGLE_WIN_LOSE_UPDATE else
                       list(win_lose_update_q_axs)))
    for ax in axs.flatten():
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    return fig, axs


def _plotSimRT(df, t_dur, bins, dt):
    x = np.linspace(0, t_dur, 1000)
    if len(df):
        # kde = stats.gaussian_kde(dir_df.SimRT.values)
        # y = kde(x) * len(dir_df)
        hist, _ = np.histogram(df.SimRT, bins=bins)
        # ax.stairs(hist, bins, fill=False, color='k', alpha=0.5)
        # Treat as if a kde, smooth the bins and interpolate
        WIN = int(np.round(0.1 / dt))
        if WIN != 0:
            hist = np.convolve(hist, np.ones(WIN)/WIN, mode='same')
        y = np.interp(x, bins[:-1], hist)
    else:
        y = np.asarray([0]*len(x))
    return x, y

def _plotHist(df, ax_up, ax_down, real_col, sim_col, t_dur, dt, legend=False):
    # total = len(df)
    bins = np.arange(0, t_dur + dt, dt)
    hist_max = 0
    bin_width = bins[1] - bins[0]
    bar_kwargs = dict(x=bins[:-1], width=bin_width,  alpha=0.5, align='edge')

    legend_once = True and legend
    for ax, dir_df in zip((ax_up, ax_down), [df[df[real_col] == 1],
                                             df[df[real_col] == 0]]):
        hist, _ = np.histogram(dir_df.calcStimulusTime, bins=bins)
        # Force bars instead of patches
        # ax.stairs(hist, bins, fill=True, color='gray', alpha=0.5)
        ax.bar(height=hist, color='gray', bottom=0, **bar_kwargs,
               label='Real' if legend_once else None)
        legend_once = False
        hist_max = max(hist_max, hist.max())

    # print(df.SimRT)
    # display(subject_df2.head(60))
    legend_once = True and legend
    for ax, dir_df in zip((ax_up, ax_down), (df[df[sim_col] == 1],
                                             df[df[sim_col] == 0])):
        x, y = _plotSimRT(dir_df, t_dur, bins, dt)
        ax.plot(x, y, color='k', label='Model' if legend_once else None)
        legend_once = False
        hist_max = max(hist_max, y.max())

    ax_up.set_ylim(0, hist_max)
    ax_down.set_ylim(hist_max, 0)
    if legend:
        ax_up.legend(loc='upper right', fontsize='x-small')
    # return hist_max

def _plotHistCorrIncorr(df, ax : plt.Axes, t_dur, dt, legend=False):
    dt = .05
    bins = np.arange(0, t_dur + dt, dt)
    hist_max = 0

    hist_corr, _ = np.histogram(df[df.ChoiceCorrect == 1].calcStimulusTime, bins=bins)
    hist_incorr, _ = np.histogram(df[df.ChoiceCorrect == 0].calcStimulusTime, bins=bins)

    # ax.stairs(hist_incorr, bins, fill=True, color='r', alpha=0.5, baseline=hist_corr)
    # ax.stairs(hist_corr, bins, fill=True, color='g', alpha=0.5)
    # Stairs forces patches, but we want bars so we will do it manually
    bin_width = bins[1] - bins[0]
    bar_kwargs = dict(x=bins[:-1], width=bin_width,  alpha=0.5, align='edge')
    ax.bar(height=hist_corr, color='g', bottom=0, **bar_kwargs,
           label='Real Correct' if legend else None)
    ax.bar(height=hist_incorr, color='r', bottom=hist_corr, **bar_kwargs,
           label='Real Incorrect' if legend else None)


    x_corr, y_corr = _plotSimRT(df[df.SimChoiceCorrect == 1], t_dur, bins, dt)
    x_incorr, y_incorr = _plotSimRT(df[df.SimChoiceCorrect == 0], t_dur, bins, dt)
    ax.plot(x_corr, y_corr, color='g', label='Model Correct' if legend else None)
    ax.plot(x_incorr, y_corr+y_incorr, color='r', label='Model Incorrect' if legend else None)
    hist_max = max(hist_corr.max() + hist_incorr.max(),
                   y_corr.max() + y_incorr.max())
    ax.set_ylim(0, hist_max)
    if legend:
        ax.legend(loc='upper right', fontsize='x-small')

def _assignPrevTrial(df):
    df_tmp = df.copy()
    df_tmp["Date"] = df_tmp.SessId.str.split("_").str[-2]
    df_tmp["SessionNum"] = df_tmp.SessId.str.split("_").str[-1].astype(int)
    df_tmp["LeftRewarded"] = df_tmp.DV > 0

    for col_prefix in ["", "Sim"]:
        df_cur = df_tmp.copy()
        if len(col_prefix): # If sim
            df_cur["ChoiceLeft"] = df_tmp.SimChoiceLeft
            df_cur["ChoiceCorrect"] = df_tmp.SimChoiceCorrect
            df_cur["calcStimulusTime"] = df_tmp.SimRT
        # display(df.head())
        old_idx = df_cur.index
        df_cur = Chain(CountContPrevOutcome()).run(df_cur)
        df_cur = df_cur.loc[old_idx]
        # display(df.head())
        df_cur["PrevChoiceCorrect"] = df_cur["PrevOutcomeCount"] >= 1
        df_cur["PrevChoiceLeft"] = df_cur.PrevDirectionIsLeftCount >= 1
        df_cur.loc[df_cur.PrevOutcomeCount == 0, "PrevChoiceLeft"] = np.nan
        df_cur["Stay"] = df_cur.ChoiceLeft == df_cur.PrevChoiceLeft
        df_cur.loc[df_cur.ChoiceLeft.isnull() | (df_cur.PrevOutcomeCount == 0), "Stay"] = np.nan
        # df_cur["PrevLefRewarded"] = df.PrevLeftRewardedCount >= 1
        df_cur["StayBaseline"] = df_cur.LeftRewarded == df_cur.PrevChoiceLeft
        df_cur.loc[df_cur.PrevChoiceLeft.isnull(), "StayBaseline"] = np.nan

        # display(df.head())
        df[f"{col_prefix}PrevOutcomeCount"] = df_cur["PrevOutcomeCount"]
        df[f"{col_prefix}PrevDirectionIsLeftCount"] = df_cur["PrevDirectionIsLeftCount"]
        df[f"{col_prefix}PrevChoiceCorrect"] = df_cur["PrevChoiceCorrect"]
        df[f"{col_prefix}PrevChoiceLeft"] = df_cur["PrevChoiceLeft"]
        df[f"{col_prefix}Stay"] = df_cur["Stay"]
        df[f"{col_prefix}StayBaseline"] = df_cur["StayBaseline"]

        # if not len(col_prefix):
        #     print("len(df):", len(df))
        #     display(df[f"{col_prefix}ChoiceCorrect"][df.valid].value_counts().head(10).sort_index())

    # display(df.head())
    return df


def _dvQuantileFn(df, col, as_df=False):
    df = df.sort_values(col)
    quantile_dict = {1:{"Easy":[], "Med":[], "Hard":[]},
                     2:{"Easy":[], "Med":[], "Hard":[]},
                     3:{"Easy":[], "Med":[], "Hard":[]},
                     }
    for dv_str, dv_df in df.groupby("DVstr"):
        dv_df = dv_df[dv_df[col].notnull()]
        len_dv = len(dv_df)
        bin0, bin_3, bin_6, bin1 = 0, int(len_dv/3), int(2*len_dv/3), len_dv
        quantile_dict[1][dv_str].append(dv_df.iloc[bin0:bin_3])
        quantile_dict[2][dv_str].append(dv_df.iloc[bin_3:bin_6])
        quantile_dict[3][dv_str].append(dv_df.iloc[bin_6:bin1])
    if as_df:
        quantile_dict = {q_idx: pd.concat(list(chain(*dv_dict.values())))
                         for q_idx, dv_dict in quantile_dict.items()
                         }
    fast = quantile_dict[1]
    typical = quantile_dict[2]
    slow = quantile_dict[3]
    return fast, typical, slow


def _plotPrevOutcomeQuantile(df, ax, annotate_fast_slow=False):
    df_real = df[df.calcStimulusTime.notnull()]
    fast_real, typical_real, slow_real = _dvQuantileFn(df_real, "calcStimulusTime",
                                                       as_df=True)
    plot_q2 = True
    _plotPrevOutcomeCurQuantile(df=df_real, q1_df=fast_real, q2_df=typical_real,
                                q3_df=slow_real, ax=ax,
                                col_prev_choice_correct="PrevChoiceCorrect",
                                sess_split_cols=["SessId"],
                                width_scale=.3, plot_q2=plot_q2,
                                show_legend=False, plot_avg_line=False)

    df_sim = df[df.SimRT.notnull()]
    fast_sim, typical_sim, slow_sim = _dvQuantileFn(df_sim, "SimRT", as_df=True)
    _plotPrevOutcomeCurQuantile(df=df_sim, q1_df=fast_sim, q2_df=typical_sim,
                                q3_df=slow_sim, ax=ax,
                                col_prev_choice_correct="SimPrevChoiceCorrect",
                                sess_split_cols=["SessId"],
                                width_scale=.3, x_offset=.4, hatch="///",
                                plot_q2=plot_q2, show_legend=False,
                                plot_avg_line=False)
    if annotate_fast_slow:
        legends, labels = ax.get_legend_handles_labels()
    else:
        legends, labels = [], []
    labels += ["Real Data", "Model"]
    legends += [plt.Rectangle((0,0), 1, 1, fc="gray", ),
                plt.Rectangle((0,0), 1, 1, fc="gray", hatch='///')]
    ax.legend(legends, labels, loc="upper right", fontsize="x-small")

def _plotPrevOutcomeCount(df, ax):
    common_kwargs = dict(ax=ax, is_normalized=False, is_time=True,
                        save_figs=False, session_split_cols=["SessId"],
                        show_legend=False, width_scale=0.3)
    min_max_ys = stimulusTimeByPrevOutcome(df=df, col="calcStimulusTime",
                                           col_prefix="", **common_kwargs)
    stimulusTimeByPrevOutcome(df=df, col="SimRT", col_prefix="Sim",
                              x_offset=0.4, hatch="///", min_max_ys=min_max_ys,
                              **common_kwargs)


def _plotStaySwitchQ(df, ax_q1, ax_q2, ax_q3):
    df_real = df[df.calcStimulusTime.notnull()]
    fast_real, typical_real, slow_real = _dvQuantileFn(df_real, "calcStimulusTime",
                                                       as_df=True)
    df_sim = df[df.SimRT.notnull()]
    fast_sim, typical_sim, slow_sim = _dvQuantileFn(df_sim, "SimRT", as_df=True)

    for ax, real_df, sim_df, plot_x_label, plot_y_label in zip(
                                        (ax_q1, ax_q2, ax_q3),
                                        (fast_real, typical_real, slow_real),
                                        (fast_sim, typical_sim, slow_sim,),
                                        (False, True, False),
                                        (True, False, False)):
          _plotStaySwitch(real_df, ax, df_sim=sim_df, plot_x_label=plot_x_label,
                          plot_y_label=plot_y_label)


def _plotMotorBias(df, ax):
    if df.Name.nunique() == 1:
        grooup_cols = ["SessId"]
    else:
        grooup_cols = ["Name"]
    fast_real, typical_real, slow_real = _dvQuantileFn(df, "calcStimulusTime",
                                                         as_df=True)
    fast_sim, typical_sim, slow_sim = _dvQuantileFn(df, "SimRT", as_df=True)

    ax.axvline(0, color='gray', ls="--", alpha=0.5)
    for df_real, df_sim, color in zip((fast_real, slow_real),
                                      (fast_sim, slow_sim),
                                      ('r', '#e3e302')):
        bias_real = calcBias(df_real, "ChoiceLeft", groupby_cols=grooup_cols)
        bias_sim = calcBias(df_sim, "SimChoiceLeft", groupby_cols=grooup_cols)
        if df.Name.nunique() == 1:
            bias_real = bias_real.unstack()
            bias_sim = bias_sim.unstack()
        y = np.arange(len(bias_real))
        # print(y)
        # display(bias_real)
        # Each line
        ax.scatter(bias_real, y, color=color, marker='o', label="Real", s=10,
                   edgecolors='none')
        y = np.arange(len(bias_sim))
        ax.scatter(bias_sim, y, color=color, marker='x', label="Model", s=10,
                   )
        for sub_y in y:
            ax.axhline(sub_y, color='gray', ls="--", alpha=0.2)
    # ax.set_xlim(-.5, .5)
    # ax.set_ylim(-1, len(bias_real))
    ax.set_yticks([])
    ax.set_xlabel("Motor Bias")
    ax.set_ylabel("Subject" if df.Name.nunique() > 1 else "Session")
    ax.legend(fontsize="x-small", loc="upper left", bbox_to_anchor=(0, 1.2),
              ncols=4, borderaxespad=0)


def _plotStaySwitch(df, ax, df_sim=None, plot_x_label=True, plot_y_label=True):
    ax.axhline(0, color='k', ls="--")
    ax.axvline(0, color='k', ls="--")
    if plot_x_label:
        ax.set_xlabel("Win Stay Update (%)")
    else:
        ax.set_xticks([])
    if plot_y_label:
        ax.set_ylabel("Lose Stay Update (%)")
    else:
        ax.set_yticks([])
    ax.set_xlim(-30, 30)
    ax.set_ylim(-35, 35)
    plots_cords = df.Name.nunique() == 1

    # df = df[df.DVstr == "Med"]
    MAX_TRIALS = df.groupby("SessId").size().max() if df.Name.nunique() == 1 \
                 else df.groupby("Name").size().max()

    def _plotSubGroup(df, col_prefix, marker, s, label, c):
        win_stay_update, lose_stay_update = calcWinLoseUpdates(df, col_prefix)
        ax.scatter([win_stay_update], [lose_stay_update], c=c, marker=marker,
                    s=s, edgecolors='none', label=label, alpha=min(1,
                                                            len(df)/MAX_TRIALS))
        if plots_cords:
            ax.text(win_stay_update, lose_stay_update,
                    f"({win_stay_update:.2f}, {lose_stay_update:.2f})",
                    fontsize=8, va="top", ha="center", color=c)
        return win_stay_update, lose_stay_update

    def _plotGroup(df_real, df_sim, scale=1, c='k'):
        # for dv_str, dv_df in df.groupby("DVstr"):
        #     _plotSubGroup(dv_df, col_prefix)
        win_stay_update_real, lose_stay_update_real = _plotSubGroup(df_real, "",
                                                    "o", 20*scale, "Real", c)
        # win_stay_update_sim, lose_stay_update_sim = _plotSubGroup(df_sim, "Sim",
        #                                               "$M$", 50, "Model", c)
        # ax.plot([win_stay_update_real, win_stay_update_sim],
        #         [lose_stay_update_real, lose_stay_update_sim],
        #         c='gray', alpha=0.2, ls="--")
        return win_stay_update_real, lose_stay_update_real

    if df.Name.nunique() > 1:
        for name, subj_df in df.groupby("Name"):
            if df_sim is not None:
                subj_df_sim = df_sim[df_sim.Name == name]
            else:
                subj_df_sim = subj_df
            _plotGroup(subj_df, subj_df_sim)
    else:
        all_df_sim = df if df_sim is None else df_sim
        _plotGroup(df, all_df_sim, scale=3, c='g')
        plots_cords = False
        win_update_real_li, switch_update_real_li = [], []
        for sess, sess_df in df.groupby("SessId"):
            if df_sim is not None:
                sess_df_sim = df_sim[df_sim.SessId == sess]
            else:
                sess_df_sim = sess_df
            ret = _plotGroup(sess_df, sess_df_sim)
            win_update_real_li.append(ret[0])
            switch_update_real_li.append(ret[1])
        win_update_real_li = np.asarray(win_update_real_li)
        switch_update_real_li = np.asarray(switch_update_real_li)
        ax.errorbar(win_update_real_li.mean(), switch_update_real_li.mean(),
                    xerr=stats.sem(win_update_real_li),
                    yerr=stats.sem(switch_update_real_li),
                    c='k', ls="none", marker='o', label="Mean", zorder=10,
                    markersize=3)

def calcWinLoseUpdates(df, col_prefix):
    def _calcUpdate(df, col_prefix):
        if not len(df):
            return 0
        update = 100*(df[f"{col_prefix}Stay"].sum() -
                    df[f"{col_prefix}StayBaseline"].sum())/len(df)
        return update
    prev_win_df = df[df[f"{col_prefix}PrevOutcomeCount"] >= 1]
    prev_lose_df = df[df[f"{col_prefix}PrevOutcomeCount"] <= -1]
    win_stay_update = _calcUpdate(prev_win_df, col_prefix)
    lose_stay_update = _calcUpdate(prev_lose_df, col_prefix)
    return win_stay_update, lose_stay_update


def _plotPsychs(df, ax_psych, psych_plot : PsychometricPlot):
    if psych_plot == PsychometricPlot.All:
        # plotPsych(df, title="Real", ax=ax_psych, combine_sides=False,
        #           by_subject=False, by_session=False, default_color='gray')
        combine_sides = False
        subject = df.Name.iloc[0]
        label = f" ({len(df):,} Trials)"
        psych_kargs = dict(dv_df_bins=_getGroups(df, combine_sides=combine_sides),
                           ax=ax_psych, combine_sides=combine_sides, linewidth=2,
                           plot_points=False, ncpus=1)
        _fitPsych(**psych_kargs, label=f"Real {subject} {label}",  color='gray',
                  correct_col="ChoiceCorrect", left_col="ChoiceLeft",)
        _fitPsych(**psych_kargs, label=f"Simulated",  color='k',
                  correct_col="SimChoiceCorrect", left_col="SimChoiceLeft")
    elif psych_plot == PsychometricPlot.SlowFast:
        combine_sides = False
        psych_kwargs = dict(#plot_typical=False, is_human_subject=False,
                            #many_subjects=False, title="", #plot_points=False,
                            ax=ax_psych, combine_sides=combine_sides,
                            ncpus=1, nfits=50,
                            many_subjects=False)
        time_now = time.time()

        def _myGetGroupsDVstr(quantile_dict):
            if combine_sides:
                return {dv_str:pd.concat(df_li)
                        for dv_str, df_li in quantile_dict.items()}.items()
            else:
                df_dict_left, df_dict_zero, df_dict_right = {}, {}, {}
                for dv_str, df_li_tmp in quantile_dict.items():
                    df_concat = pd.concat(df_li_tmp)
                    df_concat_left = df_concat[df_concat.DV > 0]
                    df_concat_zero = df_concat[df_concat.DV == 0]
                    df_concat_right = df_concat[df_concat.DV < 0]
                    df_dict_left[f"{dv_str} Left"] = df_concat_left
                    if len(df_concat_zero):
                        df_dict_zero[f"{dv_str} Zero"] = df_concat_zero
                    df_dict_right[f"{dv_str} Right"] = df_concat_right
                res_dict = {}
                res_dict.update(df_dict_left)
                res_dict.update(df_dict_zero)
                res_dict.update(df_dict_right)
                return res_dict.items()
                # raise NotImplementedError("Not implemented")

        fast_real, typical_real, slow_real = _dvQuantileFn(df, "calcStimulusTime")
        _fitPsych(_myGetGroupsDVstr(fast_real), **psych_kwargs, color='r',
                  plot_points=True, label="Real Fast", linestyle="none",
                  #correct_col="ChoiceCorrect", left_col="ChoiceLeft",
                  )
        _fitPsych(_myGetGroupsDVstr(slow_real), **psych_kwargs, color='#e3e302',
                  plot_points=True, label="Real Fast", linestyle="none",
                  #correct_col="ChoiceCorrect", left_col="ChoiceLeft",
                  )

        print(f"Copy df time: {time.time() - time_now:.2f}"); time_now = time.time()

        fast_sim, typical_sim, slow_sim = _dvQuantileFn(df, "SimRT")
        _fitPsych(_myGetGroupsDVstr(fast_sim), **psych_kwargs, color='r',
                  plot_points=False, label="Sim Fast", linestyle="-",
                   correct_col="SimChoiceCorrect", left_col="SimChoiceLeft",
                  )
        _fitPsych(_myGetGroupsDVstr(slow_sim), **psych_kwargs, color='#e3e302',
                  plot_points=False, label="Sim Slow", linestyle="-",
                  correct_col="SimChoiceCorrect", left_col="SimChoiceLeft",
                  )
        if combine_sides:
            ax_psych.set_xlim(0, 1)
            ax_psych.set_ylim(40, 100)

        # legend = ax_psych.get_legend()
        # if legend is not None:
        #     legend.remove()
        ax_psych.legend(fontsize="x-small", loc="upper left")
    else:
        if psych_plot != PsychometricPlot._None:
            raise NotImplementedError("Not implemented " + str(psych_plot))
        time_now = time.time()
        legend = ax_psych.get_legend()
        if legend is not None:
            legend.remove()


def _plotHists(df, ax_hist_corr_up, ax_hist_corr_down, ax_hist_dir_up, ax_hist_dir_down,
               ax_rt_easy, ax_rt_medium, ax_rt_hard, t_dur, dt, is_small_fig_mode):
    time_now = time.time()
    _plotHist(df, ax_hist_corr_up, ax_hist_corr_down, "ChoiceCorrect", "SimChoiceCorrect",
              t_dur, dt, legend=True)
    # print(f"Correct/Incorrect: {time.time() - time_now:.2f}"); time_now = time.time()
    if not is_small_fig_mode:
        _plotHist(df, ax_hist_dir_up,  ax_hist_dir_down,  "ChoiceLeft",    "SimChoiceLeft",
                  t_dur, dt)
        # print(f"Direction: {time.time() - time_now:.2f}"); time_now = time.time()
        _plotHistCorrIncorr(df[df.DVstr == "Easy"], ax_rt_easy, t_dur, dt, legend=True)
        _plotHistCorrIncorr(df[df.DVstr == "Med"], ax_rt_medium, t_dur, dt)
        _plotHistCorrIncorr(df[df.DVstr == "Hard"], ax_rt_hard, t_dur, dt)
        # print(f"Easy/Med/Hard: {time.time() - time_now:.2f}"); time_now = time.time()

def _plotDists(df, ax_qval, ax_reward_rate, ax_bias, plot_bias_dir,
               include_Q, include_RewardRate, bound, biasFn_kwargs,
               is_small_fig_mode):
    bins_0_1 = np.linspace(0, 1, 10, endpoint=True)
    bins_m1_1 = np.linspace(-1, 1, 20, endpoint=True)

    bias_bin_size = bound/20
    start_pt_raw = df.SimStartingPoint
    start_pt_updated = start_pt_raw.copy()
    start_pt_updated[df.DV < 0] = -start_pt_updated[df.DV < 0]
    if "adapt_correct" in biasFn_kwargs: # It was fixed twice now
        start_pt_corr, start_pt_dir = start_pt_updated, start_pt_raw
    else:
        start_pt_corr, start_pt_dir = start_pt_raw, start_pt_updated


    if plot_bias_dir:
        ax_qval.hist(start_pt_dir/bound, bins=np.arange(-1, 1.1, .1), color='r')
        ax_qval.set_title("Starting Point (Direction)")
        legend = ax_qval.get_legend()
        if legend is not None:
            legend.remove()

    if include_Q:
        ax_qval.hist(df.Q_val, bins=bins_m1_1, color='g', label="Q-Value")
        ax_qval.hist(df.Q_R, bins=bins_0_1, color='r',
                     alpha=.5, histtype='step', label="Q-R")
        ax_qval.hist(df.Q_L, bins=bins_0_1, color='b',
                     alpha=.5, histtype='step', label="Q-L")
        ax_qval.legend(loc="upper left",
                       fontsize="x-small" if not is_small_fig_mode else "xx-small")
        ax_qval.xaxis.set_major_locator(AutoLocator())
        ax_qval.yaxis.set_major_locator(AutoLocator())
        if not is_small_fig_mode:
            ax_qval.set_title("Q-Value")
    else:
        ax_qval.clear()
        ax_qval.set_xticks([])
        ax_qval.set_yticks([])

    # Convert to correct/incorrect if needed
    if not is_small_fig_mode:
        ax_bias.hist(start_pt_corr,
                    bins=np.arange(-bound, bound+bias_bin_size, bias_bin_size),
                    color='brown')

    if include_RewardRate:
        ax_reward_rate.set_title("Reward Rate Dist.")
        ax_reward_rate.hist(df.RewardRate, bins=bins_0_1, color='b')
        ax_reward_rate.xaxis.set_major_locator(AutoLocator())
        ax_reward_rate.yaxis.set_major_locator(AutoLocator())
    else:
        ax_reward_rate.clear()
        ax_reward_rate.set_xticks([])
        ax_reward_rate.set_yticks([])
        # df.RewardRate.hist(ax=ax_reward_rate, bins=bins_0_1, color='b')
    # if include_Q:
    #     df.Q_val.hist(ax=ax_qval, bins=bins_m1_1, color='r')
        # df.starting_point.hist(ax=ax_bias, bins=bins_m1_1, color='brown')
