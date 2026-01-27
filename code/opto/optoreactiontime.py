from .optoprocessor import _classifyOptoConfig
from ..common.definitions import BrainRegion
from ..common.clr import BrainRegion as BrainRegionClr
from ..pipeline.tracesnormalize import filterNanGaussianConserving
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from enum import IntFlag, auto
from functools import partial
from pathlib import Path
from typing import List, Literal
import itertools

class RTPlots(IntFlag):
    AFTER_OPTO_END_DECISION_PERF = auto()
    AFTER_OPTO_END_RT_BARS = auto()
    AFTER_OPTO_END_RT_KDE = auto()
    AFTER_OPTO_DECISION_PROB_CDF = auto()
    BEFORE_OPTO_END_DECISION_PERF = auto()
    BEFORE_OPTO_END_DECISION_PROB = auto()
    BEFORE_OPTO_END_RT_KDE = auto()
    WHOLE_SAMPLING_RT_BARS = auto()
    WHOLE_SAMPLING_RT_KDE = auto()
    WHOLE_SAMPLING_RT_PERF_CORR = auto()
    AFTER_OPTO_GROUPED_DECISION_PERF = auto()
    ALL_PLOTS = 0xFFFFFFFF

def optoReactionTime(start_state, start_delay, max_dur, stimulus_time,
                     end_state, stim_type, df, min_choice_trials, subject_name,
                     z_score, plot_sem, save_prefix, save_figs=False,
                     only_brain_regions : List[BrainRegion] = [],
                     rt_plots : RTPlots = RTPlots.ALL_PLOTS):
    # print("Df len:", len(df))
    Z_SCORE_SAVE_STR = "Z_Scored_" if z_score else ""
    WITH_SEM_SAVE_SAVE_STR = "_SEM" if plot_sem else ""
    LATE_STR = "Late " if not z_score else ""
    if len(only_brain_regions):
      df = df[df.GUI_OptoBrainRegion.isin(only_brain_regions)]

    name = _classifyOptoConfig(start_state, start_delay, max_dur, stimulus_time,
                               end_state, stim_type)
    name = f"{subject_name} - {name}"
    if save_figs and subject_name != "All Mice":
        save_prefix = f"{save_prefix}/{subject_name}_"
    # Remove trials where animal stayed over-styaed, it will never equal to
    # GUI_StimulusTime due to the way we calculate, we check for - .1 sec
    df = df[df.calcStimulusTime < df.GUI_StimulusTime - 0.1]
    df_cntrl = df[df.OptoEnabled == 0]
    df_opto  = df[df.OptoEnabled == 1]

    num_bins_per_sec = 10
    max_st = df.calcStimulusTime.max()
    max_bin = int(np.ceil(max_st*num_bins_per_sec))

    if rt_plots & RTPlots.AFTER_OPTO_END_DECISION_PERF:
        _plotReactionBars(df_cntrl, df_opto, df_col="ChoiceCorrect",
                          start_delay=start_delay, max_dur=max_dur, name=name,
                          only_after_opto=True,
                          save_fp=f"{save_prefix}_perf_bars.svg",
                          save_figs=save_figs)

    if rt_plots & RTPlots.AFTER_OPTO_END_RT_BARS:
        _plotReactionBars(df_cntrl, df_opto, df_col="calcStimulusTime",
                          start_delay=start_delay, max_dur=max_dur, name=name,
                          only_after_opto=True,
                          save_fp=f"{save_prefix}_rt_bars.svg",
                          save_figs=save_figs)

    if rt_plots & RTPlots.WHOLE_SAMPLING_RT_BARS:
        _plotReactionBars(df_cntrl, df_opto, df_col="calcStimulusTime",
                          start_delay=start_delay, max_dur=max_dur, name=name,
                          only_after_opto=False,
                          save_fp=f"{save_prefix}_rt_bars_whole_sampling.svg",
                          save_figs=save_figs)

    MAX_TIME = 3
    # Reaction-Time KDE: Total, only early, only late
    if rt_plots & RTPlots.WHOLE_SAMPLING_RT_KDE:
        _plotReactionTimeKDE(df_cntrl, df_opto, max_bin=max_bin,
                             num_bins_per_sec=num_bins_per_sec,
                             max_time=MAX_TIME, start_delay=start_delay,
                             max_dur=max_dur, name=name,
                             save_fp=f"{save_prefix}total_hist.svg",
                             bw_method=0.05,
                             normalize_y_to_1=True,
                             save_figs=save_figs)

    if rt_plots & RTPlots.BEFORE_OPTO_END_RT_KDE:
        _plotReactionTimeKDE(df_cntrl, df_opto, max_bin=max_bin,
                             num_bins_per_sec=num_bins_per_sec,
                             max_time=MAX_TIME, start_delay=start_delay,
                             max_dur=max_dur, name=name,
                             x_lim=(0, start_delay + max_dur + .1),
                             save_fp=f"{save_prefix}early_hist.svg",
                             save_figs=save_figs)

    if rt_plots & RTPlots.AFTER_OPTO_END_RT_KDE:
        _plotReactionTimeKDE(df_cntrl, df_opto, max_bin=max_bin,
                             num_bins_per_sec=num_bins_per_sec,
                             max_time=MAX_TIME, start_delay=start_delay,
                             max_dur=max_dur, name=name,
                             x_lim=(start_delay + max_dur - .1, MAX_TIME),
                             save_fp=f"{save_prefix}late_hist.svg",
                             save_figs=save_figs)
    # ## Decision probability CDF
    if rt_plots & RTPlots.AFTER_OPTO_DECISION_PROB_CDF:
        _plotDecisionProbCDF(df_cntrl, df_opto,  z_score=z_score,
                             plot_sem=plot_sem, max_time=MAX_TIME,
                             start_delay=start_delay, max_dur=max_dur,
                             name=name,
                             save_fp=(f"{save_prefix}"
                                      f"{Z_SCORE_SAVE_STR}{LATE_STR}"
                                      f"decision_probability"
                                      f"{WITH_SEM_SAVE_SAVE_STR}.svg"),
                             save_figs=save_figs)

    if rt_plots & RTPlots.BEFORE_OPTO_END_DECISION_PROB:
        _plotEarlyProb(df, _calcDecisionProb, "Decision Probability",
                       "Percentage of Trials Performed",
                       opto_end=start_delay + max_dur, name=name,
                       save_fp=f"{save_prefix}early_decision_probability.svg",
                       save_figs=save_figs)

    if rt_plots & RTPlots.BEFORE_OPTO_END_DECISION_PERF:
        _plotEarlyProb(df, _calcChoicePerformance, "Performance",
                       "Decision Performance", opto_end=start_delay + max_dur,
                       name=name,
                       save_fp=f"{save_prefix}early_decision_performance.svg",
                       save_figs=save_figs)

    # TODO: Unify the deuplicated code between _plotDecisionPerf() and
    # _plotRTPerfCorr()
    ## Late Decision Performance
    if rt_plots & RTPlots.AFTER_OPTO_GROUPED_DECISION_PERF:
        _plotDecisionPerf(df_cntrl, df_opto,  z_score=z_score,
                          plot_sem=plot_sem,
                          max_time=MAX_TIME, start_delay=start_delay,
                          max_dur=max_dur, name=name,
                          save_fp=(f"{save_prefix}{Z_SCORE_SAVE_STR}{LATE_STR}"
                                   "decision_performance"
                                   f"{WITH_SEM_SAVE_SAVE_STR}.svg"),
                          save_figs=save_figs)
    ## Correlation RT/Performance
    if rt_plots & RTPlots.WHOLE_SAMPLING_RT_PERF_CORR:
        _plotRTPerfCorr(df_cntrl, df_opto, z_score=z_score, plot_sem=plot_sem,
                        max_time=MAX_TIME, start_delay=start_delay,
                        max_dur=max_dur, name=name,
                        save_fp=(f"{save_prefix}{Z_SCORE_SAVE_STR}{LATE_STR}"
                                 f"rt_performance_corr{WITH_SEM_SAVE_SAVE_STR}"
                                 ".svg"),
                        save_figs=save_figs)

    return df

def _plotReactionBars(df_cntrl, df_opto,
                      df_col : Literal["calcStimulusTime", "ChoiceCorrect"],
                      only_after_opto : bool,
                      start_delay, max_dur, name, save_fp, save_figs):
    #
    Z_SCORE = True and not df_col == "ChoiceCorrect"
    subj_rt_mean_std = {}
    for subj, subj_df in df_cntrl.groupby("Name"):
        rt_mean = subj_df.calcStimulusTime.mean()
        rt_std = subj_df.calcStimulusTime.std()
        subj_rt_mean_std[subj] = (rt_mean, rt_std)
    if only_after_opto:
        df_cntrl = df_cntrl[df_cntrl.calcStimulusTime >= start_delay + max_dur]
        df_opto  = df_opto[df_opto.calcStimulusTime >= start_delay + max_dur]
    #
    # Start all dfs after opto end
    df_opto_mfc = df_opto[df_opto.GUI_OptoBrainRegion == BrainRegion.M2_Bi]
    df_opto_lfc = df_opto[df_opto.GUI_OptoBrainRegion == BrainRegion.ALM_Bi]
    del df_opto # Don't use by mistake


    res_dict = {"Subject":[], "Control_RT":[], "MFC_RT":[], "LFC_RT":[]}
    def _calcMetrics(df, subject, brain_region):
        vals = df[df_col]
        if Z_SCORE:
            vals = (vals - subj_rt_mean_std[subject][0]) / subj_rt_mean_std[subject][1]
        val = vals.median() if df_col == "calcStimulusTime" else \
              vals.mean() * 100
        res_dict[f"{brain_region}_RT"].append(val)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    xs = [1, 2, 3]
    labels = ["Control", "Opto MFC", "Opto LFC"]

    for subj in df_cntrl.Name.unique():
        assert subj in df_opto_mfc.Name.unique(), f"{subj} not in MFC opto"
        assert subj in df_opto_lfc.Name.unique(), f"{subj} not in LFC opto"
        res_dict["Subject"].append(subj)
        _calcMetrics(df_cntrl[df_cntrl.Name == subj], subj, "Control")
        _calcMetrics(df_opto_mfc[df_opto_mfc.Name == subj], subj, "MFC")
        _calcMetrics(df_opto_lfc[df_opto_lfc.Name == subj], subj, "LFC")

    res_df = pd.DataFrame(res_dict)
    for x, res_col, clr in zip(xs,
                               ["Control_RT", "MFC_RT", "LFC_RT"],
                               ["gray", BrainRegionClr[BrainRegion.M2_Bi], BrainRegionClr[BrainRegion.ALM_Bi]]):
        ax.bar(x, res_df[res_col].mean(), yerr=res_df[res_col].sem(),
               width=0.4, color=clr, edgecolor="black")

    # ax.scatter([x[0]]*len(ys_cntrl), ys_cntrl, color="white", s=10, edgecolor="black", zorder=5)
    # ax.scatter([x[1]]*len(ys_mfc), ys_mfc, color="white", s=10, edgecolor="black", zorder=5)
    # ax.scatter([x[2]]*len(ys_lfc), ys_lfc, color="white", s=10, edgecolor="black", zorder=5)
    for _, row in res_df.iterrows():
        ys = [row.Control_RT, row.MFC_RT, row.LFC_RT]
        ax.plot(xs, ys, color="gray", alpha=0.4, marker="o",
                markerfacecolor="white", markeredgecolor="black",
                zorder=5)

    # -------- Repeated-measures ANOVA (paired) --------
    # Long format: Subject | Condition | RT
    anova_df = res_df.melt(
        id_vars="Subject",
        value_vars=["Control_RT", "MFC_RT", "LFC_RT"],
        var_name="Condition",
        value_name="RT",
    )
    # Clean condition names: Control_RT -> Control, etc.
    anova_df["Condition"] = anova_df["Condition"].str.replace("_RT", "", regex=False)
    anova_df["Subject"] = anova_df["Subject"].astype("category")
    anova_df["Condition"] = anova_df["Condition"].astype("category")

    aovrm = AnovaRM(
        data=anova_df,
        depvar="RT",
        subject="Subject",
        within=["Condition"],
    )
    anova_res = aovrm.fit()
    print("Repeated Measures ANOVA Reaction-Time Results:")
    print(anova_res)

    # -------- Post-hoc tests (paired t-tests + Holm correction) --------
    alpha = 0.05
    # p-value for main effect of Condition
    p_cond = anova_res.anova_table.loc["Condition", "Pr > F"]
    print(f"\nMain effect of Condition: p = {p_cond:.4g}")

    # helper: p -> stars
    def p_to_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''

    # map columns to x positions
    x_pos = {"Control_RT": xs[0], "MFC_RT": xs[1], "LFC_RT": xs[2]}

    if p_cond < alpha:
        print("\nPost-hoc pairwise comparisons (paired t-tests, Holm-corrected):")

        cols = ["Control_RT", "MFC_RT", "LFC_RT"]
        pairs = list(itertools.combinations(cols, 2))

        raw_pvals = []
        tvals = []
        pair_labels = []

        for c1, c2 in pairs:
            t, p = stats.ttest_rel(res_df[c1], res_df[c2])
            tvals.append(t)
            raw_pvals.append(p)
            pair_labels.append((c1, c2))

        reject, pvals_corr, _, _ = multipletests(raw_pvals, method="holm")

        for (c1, c2), tval, p_raw, p_corr, rej in zip(pair_labels, tvals, raw_pvals, pvals_corr, reject):
            name1 = c1.replace('_RT', '')
            name2 = c2.replace('_RT', '')
            print(f"{name1} vs {name2}: t={tval:.3f}, p_raw={p_raw:.4g}, p_corr={p_corr:.4g}, significant={rej}")

        # -------- Add significance stars on the plot --------
        # find top of bars to start annotations
        y_max = max(res_df.Control_RT.max(), res_df.MFC_RT.max(), res_df.LFC_RT.max(),)
        y_start = y_max * 1.05
        y_step = y_max * 0.08  # vertical spacing between lines

        def add_sig_bar(ax, x1, x2, y, text, h_frac=0.02):
            """Draws a significance bar with text between x1 and x2 at height y."""
            h = y * h_frac
            ax.plot([x1, x1, x2, x2],
                    [y, y + h, y + h, y],
                    color='black', linewidth=1)
            ax.text((x1 + x2) / 2., y + h, text,
                    ha='center', va='bottom', fontsize=12)

        level = 0
        for (c1, c2), p_corr in zip(pair_labels, pvals_corr):
            stars = p_to_stars(p_corr)
            if not stars:
                continue  # skip non-significant
            x1, x2 = x_pos[c1], x_pos[c2]
            y = y_start + level * y_step
            add_sig_bar(ax, x1, x2, y, stars)
            level += 1
    else:
        print("\nNo significant main effect of Condition; skipping post-hoc pairwise tests.")


    if Z_SCORE:
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    when_str = (" After Opto" if only_after_opto else "Whole Sampling Time")
    if df_col == "calcStimulusTime":
        ax.set_ylabel("Sampling Time " + ("(Z-Scored)" if Z_SCORE else "(s)"))
        ax.set_title(f"{name} - Sampling Time" + (" (Z-Scored)" if Z_SCORE else "")
                    + when_str)
    else:
        ax.set_ylabel("Performance %")
        ax.set_title(f"{name} - Performance " + when_str)
        ax.set_ylim(bottom=45)
    # ax.legend()
    ax.spines[["left", "top", "right"]].set_visible(False)
    if save_figs:
        save_fp = Path(save_fp)
        # Add z_scored/raw and only_after_opto/whole_sampling as a suffix
        if df_col == "calcStimulusTime":
            suffix = "_z_scored" if Z_SCORE else "_raw"
        else:
            suffix = "_perf"
        suffix += "_only_after_opto" if only_after_opto else "_whole_sampling"
        save_fp = save_fp.with_name(save_fp.stem + suffix + save_fp.suffix)
        plt.savefig(save_fp, bbox_inches="tight", dpi=300)
    plt.show()


def _plotRTPerfCorr(df_cntrl, df_opto, z_score, plot_sem, max_time, start_delay,
                    max_dur, name, save_fp, save_figs):
    fig, ax = plt.subplots(figsize=(10, 6))
    if not z_score:
        _axMarkOptoRegion(ax, start_delay, max_dur)
    else:
        ax.spines[["right", "top"]].set_visible(False)
    # df_cpy = df.copy()
    # df_cntrl = df_cpy[df_cpy.OptoEnabled == 0]
    # df_opto  = df_cpy[df_cpy.OptoEnabled == 1]
    # max_bin = int(np.ceil(df_cpy.calcStimulusTime.max()))
    # bins = np.linspace(0, max_bin/num_bins_per_sec, max_bin+1)
    # bins = np.arange(0, max_bin + 1/num_bins_per_sec, 1/num_bins_per_sec)
    # print("Bins:", bins)
    br_slope_li = {}
    ANIMAL_ALPHA = 0.2
    ST_COL = "calcStimulusTime"
    df_cntrl = df_cntrl.copy()
    df_cntrl["ST"] = df_cntrl.calcStimulusTime.copy()
    df_opto = df_opto.copy()
    df_opto["ST"] = df_opto.calcStimulusTime.copy()
    if z_score:
        br_df_li = {br:[] for br in df_opto.GUI_OptoBrainRegion.unique()}
        br_df_li["Control"] = []

    for animal_name, animal_cntrl_df in df_cntrl.groupby("Name"):
        if z_score:
            animal_cntrl_df = animal_cntrl_df.copy()
            cntrl_mean = animal_cntrl_df[ST_COL].mean()
            cntrl_std = animal_cntrl_df[ST_COL].std()
            animal_cntrl_df[ST_COL] = (animal_cntrl_df[ST_COL] -
                                       cntrl_mean)/cntrl_std
            br_df_li["Control"].append(animal_cntrl_df)

        _plotCorr(animal_cntrl_df, br="Control", ax=ax, alpha=ANIMAL_ALPHA,
                  name=animal_name, br_slope_li=br_slope_li,
                  z_score=z_score, plot_sem=plot_sem)
        df_animal_opto = df_opto[df_opto.Name == animal_name]
        for br, br_df in df_animal_opto.groupby("GUI_OptoBrainRegion"):
            if z_score:
                br_df = br_df.copy()
                br_df[ST_COL] = (br_df[ST_COL] - cntrl_mean)/cntrl_std
                br_df_li[br].append(br_df)
            _plotCorr(br_df, br=br, ax=ax, alpha=ANIMAL_ALPHA, name=animal_name,
                      br_slope_li=br_slope_li, z_score=z_score,
                      plot_sem=plot_sem)
            # for sess, sess_df in br_df.groupby(["Name", "Date","SessionNum"]):
            #     print(animal_name,
            #           f"Date: {sess[1]} /{sess[2]} - len={len(sess_df):,}")
            # print()
    if z_score:
        # df_cntrl = df_cntrl.copy()
        # cntrl_mean = df_cntrl[ST_COL].mean()
        # cntrl_std = df_cntrl[ST_COL].std()
        # df_cntrl[ST_COL] = (df_cntrl[ST_COL] - cntrl_mean)/cntrl_std
        df_cntrl = pd.concat(br_df_li["Control"])
    _plotCorr(df_cntrl, br="Control", ax=ax, alpha=1, name="All",
              br_slope_li=br_slope_li, z_score=z_score, plot_sem=plot_sem)
    for br, br_df in df_opto.groupby("GUI_OptoBrainRegion"):
        if z_score:
            # br_df = br_df.copy()
            # br_df[ST_COL] = (br_df[ST_COL] - cntrl_mean)/cntrl_std
            br_df = pd.concat(br_df_li[br])
        _plotCorr(br_df, br=br, ax=ax, alpha=1, name="All",
                  br_slope_li=br_slope_li, z_score=z_score, plot_sem=plot_sem)
    # bins = (bins[:-1] + bins[1:])/2
    # plotCorr(ax, bins, processFn=processCorr, plot_sem=True,
    #          br_slope_li=br_slope_li)
    # plotBins(ax, bins, processFn=processPerfBins, plot_sem=True,
    #          br_slope_li=br_slope_li)
    ax.set_title(f"{name} - Decision Performance")
    if not z_score:
        ax.set_xlim(start_delay + max_dur - 0.1, max_time)
    else:
        ax.set_xlim(-1, 3)
    ax.set_xlabel("Reaction Time " + ("(Normalized Z-Ssore)" if z_score else
                                      "(s)"))
    ax.axhline(50, ls="--", color="gray")
    ax.set_ylim(45, 90)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Percentage of Correct Choices")
    ax.legend(fontsize="small")
    if save_figs:
        fig.savefig(save_fp, dpi=300, bbox_inches='tight')
    plt.show()

def _axMarkOptoRegion(ax, start_delay, max_dur):
    ax.axvspan(start_delay, start_delay + max_dur, alpha=0.3,
               color="lightblue", label="Manipulation")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

def _plotDecisionProbCDF(df_cntrl, df_opto, z_score, plot_sem, max_time,
                         start_delay, max_dur, name, save_fp, save_figs):
    fig, ax = plt.subplots(figsize=(6, 6))
    if not z_score:
        _axMarkOptoRegion(ax, start_delay, max_dur)
    else:
        ax.spines[["right", "top"]].set_visible(False)
    # fig.set_size_inches(analysis.SAVE_FIG_SIZE[0], analysis.SAVE_FIG_SIZE[1])
    br_cdf_li = {}
    animals_cdf = {}
    used_x_bins = np.arange(0, 5+.1, .1) if not z_score else \
                  np.arange(-5, 5+.1, .1)
    # print("used_x_bins:", used_x_bins)
    for animal_name, animal_cntrl_df in df_cntrl.groupby("Name"):
        animal_cntrl_st = animal_cntrl_df.calcStimulusTime
        # animal_cntrl = animal_cntrl[animal_cntrl > max_dur]
        if z_score:
            cntrl_mean = animal_cntrl_st.mean()
            cntrl_std = animal_cntrl_st.std()
            animal_cntrl_st = (animal_cntrl_st - cntrl_mean)/cntrl_std
        control_st = _cumCounts(animal_cntrl_st, bins=used_x_bins,
                                plot_sem=plot_sem, br_cdf_li=br_cdf_li,
                                br="Control", ax=ax)
        animals_cdf[animal_name] = {"Control":control_st}
        animal_opto = df_opto[df_opto.Name == animal_name]
        for br, br_df in animal_opto.groupby("GUI_OptoBrainRegion"):
            # total_count = len(df_opto[(df_opto.Name == animal_name) &
            #                           (df_opto.GUI_OptoBrainRegion == br)])
            br_opto_st = br_df.calcStimulusTime
            # br_opto = br_opto[br_opto > max_dur]
            if z_score:
                br_opto_st = (br_opto_st - cntrl_mean)/cntrl_std
            animals_cdf[animal_name][br] = _cumCounts(br_opto_st,
                                                      bins=used_x_bins,
                                                      plot_sem=plot_sem,
                                                      br_cdf_li=br_cdf_li,
                                                      br=br,
                                                      ax=ax)
    # used_x_bins = x_bins[x_bins > opto_lapse]
    # xs_plot = (used_x_bins[:-1] + used_x_bins[1:])/2
    regions_ys = _processCDF(animals_src_dict=animals_cdf)
    for br, ys in regions_ys.items():
        color, label = _getBrainRegionColorAndLabel(br)
        # plotMeanSEM(ax, xs_plot, ys, color=color, label=label)
    _cumCounts(df_cntrl.calcStimulusTime, bins=used_x_bins,
               plot_sem=plot_sem, br="Control", br_cdf_li=br_cdf_li,
               is_all=True, ax=ax)
    for br, br_df in df_opto.groupby("GUI_OptoBrainRegion"):
        _cumCounts(df_cntrl.calcStimulusTime, bins=used_x_bins,
                   plot_sem=plot_sem, br=br, br_cdf_li=br_cdf_li, is_all=True,
                   ax=ax)
    late_str = "Late " if not z_score else ""
    ax.set_title(f"{name} - {late_str}Decision Probability" +
                 (" (Since Trial Start)" if z_score else ""))
    ax.set_xlabel("Time " + ("(s)" if not z_score else "(Normalized Z-Score)"))
    ax.set_ylabel("Percentage of Cumulative Trials Performed")
    if not z_score:
        ax.set_xlim(start_delay + max_dur - 0.1, max_time)
    else:
        ax.set_xlim(-1, 3)
    ax.set_ylim(0, 100)
    ax.legend()
    if save_figs:
        fig.savefig(save_fp, dpi=300, bbox_inches='tight')
    plt.show()

def _plotDecisionPerf(df_cntrl, df_opto, z_score, plot_sem, max_time,
                      start_delay, max_dur, name, save_fp, save_figs):
    fig, ax = plt.subplots(figsize=(6, 6))
    if not z_score:
        _axMarkOptoRegion(ax, start_delay, max_dur)
    else:
        ax.spines[["right", "top"]].set_visible(False)

    num_bins_per_sec = 4 #if not Z_SCORE else 2 # Z-SCore spans longer
    # df = df.copy()
    # Make sure that there is emough data for the gaussian filter
    # LIM = max_time + 2*(1/num_bins_per_sec)
    # df.loc[df.calcStimulusTime > LIM + 2*(1/num_bins_per_sec),
    #        "calcStimulusTime"] = LIM
    #display(df_cpy.calcStimulusTime.describe())
    max_st = max(df_cntrl.calcStimulusTime.max(),
                 df_opto.calcStimulusTime.max())
    max_bin = int(np.ceil(max_st))
    # bins = np.linspace(0, max_bin/num_bins_per_sec, max_bin+1)
    bins = np.arange(-1.5 if z_score else 0,
                     max_bin + 1/num_bins_per_sec, 1/num_bins_per_sec)
    ANIMAL_ALPHA = 0.2
    br_perf_li = {}
    ST_COL = "calcStimulusTime"
    df_cntrl = df_cntrl.copy()
    df_cntrl["ST"] = df_cntrl.calcStimulusTime.copy()
    df_opto = df_opto.copy()
    df_opto["ST"] = df_opto.calcStimulusTime.copy()
    if z_score:
        br_df_li = {br:[] for br in df_opto.GUI_OptoBrainRegion.unique()}
        br_df_li["Control"] = []

    for animal_name, animal_cntrl_df in df_cntrl.groupby("Name"):
        if z_score:
            animal_cntrl_df = animal_cntrl_df.copy()
            cntrl_mean = animal_cntrl_df[ST_COL].mean()
            cntrl_std = animal_cntrl_df[ST_COL].std()
            animal_cntrl_df[ST_COL] = (animal_cntrl_df[ST_COL] -
                                       cntrl_mean)/cntrl_std
            br_df_li["Control"].append(animal_cntrl_df)
        _plotPerf(animal_cntrl_df, br="Control", ax=ax, alpha=ANIMAL_ALPHA,
                  name=animal_name, br_perf_li=br_perf_li, bins=bins,
                  z_score=z_score, plot_sem=plot_sem)
        df_animal_opto = df_opto[df_opto.Name == animal_name]
        for br, br_df in df_animal_opto.groupby("GUI_OptoBrainRegion"):
            if z_score:
                br_df = br_df.copy()
                br_df[ST_COL] = (br_df[ST_COL] - cntrl_mean)/cntrl_std
                br_df_li[br].append(br_df)
            _plotPerf(br_df, br=br, ax=ax, alpha=ANIMAL_ALPHA, name=animal_name,
                      br_perf_li=br_perf_li, bins=bins, z_score=z_score,
                      plot_sem=plot_sem)
            # for sess, sess_df in br_df.groupby(["Name", "Date","SessionNum"]):
            #     print(animal_name,
            #           f"Date: {sess[1]} /{sess[2]} - len={len(sess_df):,}")
            # print()
    if z_score:
        # df_cntrl = df_cntrl.copy()
        # cntrl_mean = df_cntrl[ST_COL].mean()
        # cntrl_std = df_cntrl[ST_COL].std()
        # df_cntrl[ST_COL] = (df_cntrl[ST_COL] - cntrl_mean)/cntrl_std
        df_cntrl = pd.concat(br_df_li["Control"])
    _plotPerf(df_cntrl, br="Control", ax=ax, alpha=1, name="All",
              br_perf_li=br_perf_li, bins=bins, z_score=z_score,
              plot_sem=plot_sem)
    for br, br_df in df_opto.groupby("GUI_OptoBrainRegion"):
        if z_score:
            # br_df = br_df.copy()
            # br_df[ST_COL] = (br_df[ST_COL] - cntrl_mean)/cntrl_std
            br_df = pd.concat(br_df_li[br])
        _plotPerf(br_df, br=br, ax=ax, alpha=1, name="All",
                  br_perf_li=br_perf_li, bins=bins, z_score=z_score,
                  plot_sem=plot_sem)
    # bins = (bins[:-1] + bins[1:])/2
    # plotCorr(ax, bins, processFn=processCorr, plot_sem=True)
    # plotBins(ax, bins, processFn=processPerfBins, plot_sem=True)
    late_str = "Late " if not z_score else ""
    ax.set_title(f"{name} - {late_str}Decision Performance" +
                 (" (Since Trial Start)" if z_score else ""))
    if not z_score:
        ax.set_xlim(start_delay + max_dur - 0.1, max_time)
    else:
        ax.set_xlim(-1, 3)
    ax.set_xlabel("Reaction Time " + ("(Normalized Z-Ssore)" if z_score else
                                      "(s)"))
    ax.axhline(50, ls="--", color="gray")
    ax.set_ylim(45, 90)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Percentage of Correct Choices")
    ax.legend(fontsize="small")
    if save_figs:
        fig.savefig(save_fp, dpi=300, bbox_inches='tight')
    plt.show()

def _plotReactionTimeKDE(df_cntrl, df_opto, max_bin, num_bins_per_sec, max_time,
                         start_delay, max_dur, name, save_fp, save_figs,
                         normalize_y_to_1=False,
                         x_lim=None, bw_method=0.1):
    fig, ax = plt.subplots()
    _axMarkOptoRegion(ax, start_delay, max_dur)
    # fig.set_size_inches(analysis.SAVE_FIG_SIZE[0], analysis.SAVE_FIG_SIZE[1])
    animals_kde = {}
    animals_num_pts = {}
    br_std = {br:[] for br in df_opto.GUI_OptoBrainRegion.unique()}
    br_std["Control"] = []
    for animal_name, animal_cntrl_df in df_cntrl.groupby("Name"):
        st_cntrl_col = animal_cntrl_df.calcStimulusTime
        br_std["Control"].append(st_cntrl_col.std())
        controlKDE = _getKDEFn(st_cntrl_col, bw_method=bw_method)
        animals_kde[animal_name] = {"Control":controlKDE}
        animals_num_pts[animal_name] = {"Control":len(st_cntrl_col)}
        for br, br_df in df_opto[df_opto.Name == animal_name].groupby(
                                                         "GUI_OptoBrainRegion"):
            st_br_col = br_df.calcStimulusTime
            br_std[br].append(st_br_col.std())
            optoKDE = _getKDEFn(st_br_col, bw_method=bw_method)
            animals_kde[animal_name][br] = optoKDE
            animals_num_pts[animal_name][br] = len(st_br_col)


    br_mode_width = {br:[] for br in df_opto.GUI_OptoBrainRegion.unique()}
    br_mode_width["Control"] = []

    x_bins = np.linspace(0, max_bin/num_bins_per_sec, 1000)
    processXbinsFn = partial(processXbins, animals_kde=animals_kde,
                             br_std=br_std, animals_num_pts=animals_num_pts,
                             max_dur=max_dur)
    _plotBins(ax, x_bins, processFn=processXbinsFn, br_mode_width=br_mode_width,
              normalize_y_to_1=normalize_y_to_1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Probability")
    if x_lim is None:
        x_lim = 0, max_time
    ax.set_xlim(*x_lim)
    ax.set_ylim(bottom=0)
    ax.set_title(f"{name} - Total Reaction Time")
    ax.legend()

    if save_figs:
        fig.savefig(save_fp, dpi=300, bbox_inches='tight')
    plt.show()

def _getKDEFn(df_col, bw_method=0.1):
    kde = stats.gaussian_kde(df_col, bw_method=bw_method)
    return kde

def _plotMeanSEM(ax, x, ys, color, label, plot_sem=True,
                 normalize_y_to_1=False):
    if normalize_y_to_1:
        ys = ys / np.nanmax(ys)
    y = np.nanmean(ys, axis=0)
    ax.plot(x, y, color=color, label=label)
    if plot_sem:
      y_sem = stats.sem(ys, axis=0, nan_policy="omit")
      ax.fill_between(x, y-y_sem, y+y_sem, color=color, alpha=0.2)

def _getBrainRegionColorAndLabel(br):
    if isinstance(br, str) and br.split(" ", 1)[0] == "Control":
      sec_br = br.split(" ")
      sec_br = "" if len(sec_br) == 1 else f" {BrainRegion(float(sec_br[1]))}"
      return "black", f"Control{sec_br}"
    else:
      br = BrainRegion(br)
      postfix = " inhibition" if br != BrainRegion.Pulses else ""
      return BrainRegionClr[str(br)], f"{str(br).split('_')[0]}{postfix}"

def _plotBins(ax, used_x_bins, processFn, br_mode_width, plot_sem=True,
              normalize_y_to_1=False):
    regions_ys_num_pts = processFn(used_x_bins, br_mode_width=br_mode_width)
    if isinstance(regions_ys_num_pts, tuple):
        regions_ys, regions_num_pts, region_num_subjects = regions_ys_num_pts
    else:
        regions_ys, regions_num_pts, region_num_subjects = \
                                                  regions_ys_num_pts, None, None
    for br, ys in regions_ys.items():
        color, label = _getBrainRegionColorAndLabel(br)
        if regions_num_pts is not None:
            assert region_num_subjects[br] == len(regions_ys[br]), (
                                               "Mismatch in number of subjects")
            label = f"{label} (n={np.sum(regions_num_pts[br]):,}, "
            label += f"{region_num_subjects[br]} subjects)"
        _plotMeanSEM(ax, used_x_bins, ys, color=color, label=label,
                     plot_sem=plot_sem, normalize_y_to_1=normalize_y_to_1)

def _processCDF(animals_src_dict):
    regions_ys = {}
    for cdf_dict in animals_src_dict.values():
      for br, ys in cdf_dict.items():
        arr_ys = regions_ys.get(br, [])
        arr_ys.append(ys)
        regions_ys[br] = arr_ys
    return regions_ys


def _findKdeModeWidth(ys, used_x_bins, br, br_mode_width):
    THRESHOLD_FACTOR = 0.5
    y_max = ys.max() # Treat it as a histogram
    threshold_y = THRESHOLD_FACTOR*y_max
    max_y_idx = np.where(ys == y_max)[0][0]
    ys_threshold_idxs = np.array(np.where(ys < threshold_y)[0])
    # print("max_x_idxs:", max_y_idx)
    ys_before = ys_threshold_idxs[ys_threshold_idxs < max_y_idx][-1]
    ys_after = ys_threshold_idxs[ys_threshold_idxs > max_y_idx][0]
    start_sec = used_x_bins[ys_before]
    mode_sec = used_x_bins[max_y_idx]
    end_sec = used_x_bins[ys_after]
    br_mode_width[br].append(end_sec - start_sec)
    if br != "Control":
        br = str(BrainRegion(br))
    # print(f"{br} - y-Before -> y-Mode -> y-After -   "
    #       f"{start_sec:.2f} -> {mode_sec:.2f} -> {end_sec:.2f} Sec -   y = "
    #       f" {ys[ys_before]:.2f} -> {ys[max_y_idx]:.2f} -> {ys[ys_after]:.2f}"
    #       f" - Span: {end_sec - start_sec:.2f} Sec")

def processXbins(used_x_bins, br_mode_width, animals_kde, br_std,
                 animals_num_pts, max_dur):
    regions_ys = {}
    regions_num_pts = {}
    region_num_subjects = {}
    br_max_y_x_pos = {} # Used to find out who fires during opto inhibtion
    br_cntrl_time_diff = {}
    for animal_name, dict_KDEs in animals_kde.items():
        control_max_y_x_pos = None
        for br, kde in dict_KDEs.items():
            if br not in regions_ys:
                regions_ys[br] = []
                regions_num_pts[br] = []
                br_max_y_x_pos[br] = []
                br_cntrl_time_diff[br] = []
                br_std[br] = []
            ys = kde(used_x_bins)
            y_max = ys.max() # Treat it as a histogram
            max_y_x_pos = used_x_bins[ys == y_max][0]
            # print("AnimalL", animal_name, "Br:", br,
            #       "- y-max-x-pos:", used_x_bins[ys == ys.max()])
            br_max_y_x_pos[br].append(max_y_x_pos)
            _findKdeModeWidth(ys, used_x_bins, br, br_mode_width)
            if br == "Control":
                control_max_y_x_pos = max_y_x_pos
            else:
                assert control_max_y_x_pos is not None, (
                                      "Implement it rather than assuming order")
                dist_from_contrl = max_y_x_pos - control_max_y_x_pos
                br_cntrl_time_diff[br].append(dist_from_contrl)
            # ys = ys/y_max # Normalize
            regions_ys[br].append(ys)
            regions_num_pts[br].append(animals_num_pts[animal_name][br])
            region_num_subjects[br] = region_num_subjects.get(br, 0) + 1
    # Print time difference
    def printDiff(arr, total_count, dscrp):
        arr_mean = arr.mean()
        arr_sem = stats.sem(arr)
        cur_count = len(arr)
        # print(f"For {dscrp}: {cur_count}/{total_count} - "
        #       f"mean: {arr_mean.mean():.3f} ±{arr_sem:.3f} sem")

    for br, br_dist_mode_li in br_max_y_x_pos.items():
        br_str = str(BrainRegion(br)) if br != "Control" else br
        br_dist_mode_li = np.array(br_dist_mode_li)
        mode_before_opto = br_dist_mode_li[br_dist_mode_li <= max_dur]
        mode_after_opto = br_dist_mode_li[br_dist_mode_li > max_dur]
        total_len = len(br_dist_mode_li)
        printDiff(mode_before_opto, total_len, f"{br_str} Mode during opto")
        printDiff(mode_after_opto,  total_len, f"{br_str} Mode after opto")

        mode_width_li = np.array(br_mode_width[br])
        printDiff(mode_width_li, total_len, f"{br_str} Mode width")

        br_time_diff_li = br_cntrl_time_diff[br]
        if br == "Control":
            assert len(br_time_diff_li) == 0
            continue
        br_time_diff_li = np.array(br_time_diff_li)
        assert total_len == len(br_time_diff_li), "Shouldn't happen"
        printDiff(br_time_diff_li, total_len,
                  f"{br_str} Allt time diff from control")
        # Now split it to distribution mode before and after cntrl mode:
        mode_before_cntrl = br_time_diff_li[br_time_diff_li < 0]
        mode_after_cntrl = br_time_diff_li[br_time_diff_li > 0]
        printDiff(mode_before_cntrl, total_len,
                  f"{br_str} Only -ve time diff from control")
        printDiff(mode_after_cntrl, total_len,
                  f"{br_str} Only +ve time diff from control")
        # print(f"br: {br_str} - Time diff: {br_time_diff_li.mean():.2f}",
        #       f" - SEM:: {stats.sem(br_time_diff_li):.2f}")
      # print("Vals:", br_time_diff_li)
    return regions_ys, regions_num_pts, region_num_subjects

def _cumCounts(col_data, bins, br_cdf_li, br, plot_sem, is_all=False, ax=None):
    if br not in br_cdf_li:
        br_cdf_li[br] = []
    if not is_all:
        col_data = col_data[col_data.notnull()]
        total_count = len(col_data)
        counts, _ = np.histogram(col_data, bins=bins)
        y = np.cumsum(counts)
        offset = total_count - y[-1]
        y += offset
        y = 100*y/total_count # (total_count - len(col_data)zzzzz
        # y += (1-len(col_data)/total_count)*100
        br_cdf_li[br].append(y)
    else:
        y = np.mean(br_cdf_li[br], axis=0)
        sem = stats.sem(br_cdf_li[br], axis=0)
        # print("Sem:", sem)
    if not plot_sem or is_all:
        assert ax is not None
        color, label = _getBrainRegionColorAndLabel(br)
        x = (bins[:-1]+bins[1:])/2
        ax.plot(x, y, alpha=0.3 if not is_all else 1,
                lw=3 if is_all else 1,
                color=color, label=label if is_all else None)
    if is_all and plot_sem:
        assert ax is not None
        ax.fill_between(x, y-sem, y+sem, color=color, alpha=.2)
    return y

def _calcDecisionProb(df, cutoff):
    return 100*len(df[df.calcStimulusTime <= cutoff])/len(df)

def _calcChoicePerformance(df, cutoff):
    df = df[df.calcStimulusTime <= cutoff]
    # _br = df.GUI_OptoBrainRegion.unique()
    # print("For cut-off:", cutoff, "len:", len(df), "Br:", _br)
    if len(df):
        return 100*len(df[df.ChoiceCorrect == 1])/len(df)
    else:
        return np.nan

def _plotEarlyProb(df, metricFn, metric_name, y_label, opto_end, name,
                   save_fp, save_figs):
    fig, ax = plt.subplots()
    # fig.set_size_inches(analysis.SAVE_FIG_SIZE[0]/2, analysis.SAVE_FIG_SIZE[1])
    animals_prob = {}
    for animal_name, animal_df in df.groupby("Name"):
        animals_prob[animal_name] = {}
        for br, region_df in animal_df.groupby("GUI_OptoBrainRegion"):
          region_cntrl = region_df[region_df.OptoEnabled == 0]
          cntrl_perf = metricFn(region_cntrl, opto_end)
          region_opto = region_df[region_df.OptoEnabled == 1]
          opto_perf = metricFn(region_opto, opto_end)
          if not np.isnan(cntrl_perf) and not np.isnan(opto_perf):
              animals_prob[animal_name][br] = (cntrl_perf, opto_perf,
                                            len(region_cntrl), len(region_opto))
          else:
              print("SKipping:", animal_name, "for br:", BrainRegion(br))

    br_perf = {}
    for animal_name, animal_br_perf in animals_prob.items():
        for br, (cntrl_perf, opto_perf,
                cntrl_len, opto_len) in animal_br_perf.items():
          color, label = _getBrainRegionColorAndLabel(br)
          ax.plot([0, 1], [cntrl_perf, opto_perf], color=color, alpha=0.1)
          if br not in br_perf:
              br_perf[br] = []
          br_perf[br].append((cntrl_perf, opto_perf, cntrl_len, opto_len))
    offset = 0
    for br, zipped_perf in br_perf.items():
        zipped_perf = np.array(zipped_perf)
        cntrl_perf = zipped_perf[:, 0]
        opto_perf = zipped_perf[:, 1]
        cntrl_len = zipped_perf[:, 2]
        opto_len = zipped_perf[:, 3]
        color, label = _getBrainRegionColorAndLabel(br)
        num_subjects = len(zipped_perf)
        cntrl_len_sum = int(np.sum(cntrl_len))
        opto_len_sum = int(np.sum(opto_len))
        cntrl_mean = np.nanmean(cntrl_perf)
        cntrl_sem = 0 if len(cntrl_perf) == 1 else stats.sem(cntrl_perf)
        opto_mean = np.nanmean(opto_perf)
        opto_sem = 0 if len(opto_perf) == 1 else stats.sem(opto_perf)
        label = (f"{label} Cntrl: {cntrl_mean:.2f}% ±{cntrl_sem:.2f}- "
                f"Opto: {opto_mean:.2f}% ±{opto_sem:.2f}\n"
                f"{num_subjects} Subjects "
                f"({opto_len_sum:,} Opto - {cntrl_len_sum:,} Control)")
        ax.errorbar([0 + offset, 1 + offset], [cntrl_mean, opto_mean],
                    [cntrl_sem, opto_sem], color=color, alpha=1, label=label)
        offset = 0.01
    ax.set_xticks([0, 1], ["Control", "Opto"])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.set_xlim(-0.2, 1.2)
    ax.set_title(f"{name} - Early {metric_name}")
    ax.set_ylabel(y_label)
    ax.legend()
    if save_figs:
        fig.savefig(save_fp, dpi=300, bbox_inches='tight')
    plt.show()


def _plotPerf(df, br, ax, alpha, name, br_perf_li, bins, z_score, plot_sem):
    if br not in br_perf_li:
        br_perf_li[br] = []
    if name != "All":
        df = df[df.ChoiceCorrect.notnull()]
        # num_pts = len(df)
        counts_total, _ = np.histogram(df.calcStimulusTime, bins=bins)
        counts_total = counts_total.astype(float)
        counts_corr, _ = np.histogram(
                          df[df.ChoiceCorrect == 1].calcStimulusTime, bins=bins)
        # Disable points with zero entries
        counts_total[counts_total == 0] = np.nan
        perf = 100*counts_corr/counts_total
        perf = filterNanGaussianConserving(perf, sigma=2 if z_score else 1,
                                           axis=0)
        br_perf_li[br].append(perf)
        y = perf
    else:
        y_all = np.array(br_perf_li[br])
        y = np.nanmean(y_all, axis=0)
        y_sem = stats.sem(y_all, axis=0, nan_policy="omit")

    if name == "All" or not plot_sem:
        x = (bins[:-1] + bins[1:])/2
        color, label = _getBrainRegionColorAndLabel(br)
        if name != "All":
            label = None
        lw = 3 if alpha == 1 else 1
        ax.plot(x, y, c=color, alpha=alpha, lw=lw, label=label)

    if name == "All" and plot_sem:
        ax.fill_between(x, y-y_sem, y+y_sem, color=color, alpha=0.2)

def _plotCorr(df, br, ax, alpha, name, br_slope_li, z_score, plot_sem):
    df = df[df.ChoiceCorrect.notnull()]
    num_pts = len(df)
    color, label = _getBrainRegionColorAndLabel(br)
    BIN_EVERY = 50#30
    ST_COL = "calcStimulusTime"
    df = df.sort_values(ST_COL).reset_index(drop=True)
    grp_by = df.groupby(df.index // BIN_EVERY)
    grp_by = grp_by.agg(calcStimulusTime=(ST_COL, "mean"),
                        Size=(ST_COL, "size"),
                        ST=("ST", "mean"),
                        ChoiceCorrect=("ChoiceCorrect", "mean"))
    grp_by["ChoiceCorrect"] = grp_by["ChoiceCorrect"]*100 # Change to percentage
    grp_by = grp_by[grp_by.Size >= BIN_EVERY] #*2/3] # Remove small last groups
    # if name == "vgat-46":
    #   display(grp_by)
    # print("Min:", grp_by.calcStimulusTime.min())
    if name == "All" and not plot_sem:
        ax.scatter(grp_by.calcStimulusTime, grp_by.ChoiceCorrect, color=color,
                  alpha=.5, s=2)
    # Force a common intercept but with different origin than x=0, y=0
    INTERCEPT = 50
    REG_X_OFFSET = -1.5 if z_score else 0
    x = grp_by.calcStimulusTime.values - REG_X_OFFSET
    x = x[:, np.newaxis]
    y = grp_by.ChoiceCorrect.values - INTERCEPT
    if len(x) >= 2:
        regression = LinearRegression(fit_intercept=False)
        regression.fit(x, y)
        slope = regression.coef_[0]
    else:
        slope = np.nan
    # intercept = regression.intercept_ + INTERCEPT
    # print("regression.intercept_", regression.intercept_)
    if br not in br_slope_li:
        br_slope_li[br] = []
    # IF it's Z-SCore and "All" then below, we below we already concatenated the
    # different z-score animals and no need to take the mean and SEM
    if name != "All" or z_score: # alpha != 1:
        br_slope_li[br].append(slope)
        slope_sem = 0
        if name == "All":
            label = f"{label} {name} - Slope={slope:.3f} ({num_pts:,} pts)"
        else:
            label = None
    else:
        # print("Using mean slope")
        slope_li = np.array(br_slope_li[br])
        slope_li = slope_li[~np.isnan(slope_li)]
        slope, slope_sem = np.mean(slope_li), stats.sem(slope_li)
        # label = f"{label} (n={num_pts:,}) - "
        label = (f"{label} {name} - Slope={slope:.3f} +/-{slope_sem:.3f} "
                 f"({len(slope_li)} Animals)")

    x = [REG_X_OFFSET if z_score else 0, df[ST_COL].max()]
    y = [INTERCEPT + (x[0]-REG_X_OFFSET)*slope,
         INTERCEPT + (x[1]-REG_X_OFFSET)*slope]
    if name == "All" or not plot_sem:
        ax.plot(x, y, c=color, alpha=alpha, lw=3 if alpha == 1 else 1,
                # label=f"{label} "#r={lin_reg.rvalue:.3f} - "
                #       #f"p={lin_reg.pvalue:.3f} - "
                #       f"Slope: {intercept},{slope:.2f}"
                label=label)
    if plot_sem and name == "All":
        ax.fill_between(x, y-slope, y+slope, color=color, alpha=0.2)
