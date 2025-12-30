from .util import normalizeSTAcrossSubjects
import matplotlib.cm as mplcm
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.ticker as plticker
import numpy as np
import seaborn as sns
from math import floor, ceil
from pathlib import Path
from scipy import stats


class DifficultyClr:
    # Easy = "#8a9400"
    # Med = "#009994"
    # Hard = "#680094"
    Easy = "k"
    Med = "gray"
    Hard = "lightgray"

def stByDifficulty(df, subjs_dscrp="All Mice", plot_single_subjects=True,
                   CDF_xlim=None, save_prefix="", save_figs=False):
    df = df[df.calcStimulusTime.notnull()]
    df = normalizeSTAcrossSubjects(df)

    _processSubject(df, subjs_dscrp, many_animals=True, CDF_xlim=CDF_xlim,
                   save_prefix=save_prefix, save_figs=save_figs,)
    if not plot_single_subjects:
        return
    for subject_name, subject_df in df.groupby("Name"):
        print(subject_name)
        _processSubject(subject_df, subject_name, many_animals=False,
                        CDF_xlim=CDF_xlim, save_prefix=save_prefix,
                        save_figs=save_figs)
        # break

def stDistOnly(df, plot_combined_subjects=True, plot_single_subjects=True, 
               descrp="Mice", save_prefix="", save_figs=False,
               as_kde=False):
    df = df[df.calcStimulusTime.notnull()]
    df = normalizeSTAcrossSubjects(df)
    def _processSubjectST(df, subject_name, many_animals):
        fig, axs = plt.subplots(3, 1, figsize=(8, 6), tight_layout=True)
        ax_easy, ax_med, ax_hard = axs
        total_num_trials = 0
        fig.suptitle(f"Stimulus Time Distribution - {subject_name}")
        for dv, clr, ax_dv in [("Easy", DifficultyClr.Easy, ax_easy),
                               ("Med", DifficultyClr.Med, ax_med),
                               ("Hard", DifficultyClr.Hard, ax_hard)]:
            dv_df = df[df.DVstr == dv]
            num_trials = len(dv_df)
            total_num_trials += num_trials
            # print(f"Number of trials for {dv}: {num_trials}")
            if not many_animals and not len(dv_df):
                continue
            num_trials = len(dv_df)
            num_sessions = len(
                            dv_df[["Name", "Date", "SessionNum"]].drop_duplicates())
            num_subjects = len(dv_df.Name.unique()) if many_animals else None
            _setupAxes(ax_dv, many_animals, is_CDF=False,
                       subject_name=dv, num_trials=num_trials,
                       num_sessions=num_sessions, num_subjects=num_subjects)
            _handleDifficultyDf(dv, dv_df, ax_dv,  clr, CDF=False,
                                many_animals=many_animals, plot_median=False)
            # ax_dv.legend()
            [ax_dv.spines[_dir].set_visible(False) 
             for _dir in ["top", "right", "left"]]
        
        
        # ax.legend(loc="upper right")
        if save_figs:
            kde_str = "kde_"  if as_kde else ""
            save_fp = Path(
                  f"{save_prefix}st_only/{descrp}_{kde_str}{subject_name}.svg")
            save_fp.parent.mkdir(exist_ok=True)
            fig.savefig(save_fp)
        plt.show()
    if plot_combined_subjects:
        _processSubjectST(df, f"All {descrp}", many_animals=True)

    if not plot_single_subjects:
        return
    for subject_name, subject_df in df.groupby("Name"):
        print(subject_name)
        _processSubjectST(subject_df, subject_name, many_animals=False)


def loopstVsDiffOnly(df, title, plot_combined_subjects=True, 
                     plot_single_subjects=False,  save_prefix="",
                     save_figs=False):
    print("TODO: Rename this function and the function below to: "
          "fastSlowStVsDIffOnly")
    df = df[df.calcStimulusTime.notnull()]
    df = normalizeSTAcrossSubjects(df)
    # ax.legend(loc="upper right")
    def _processSubject(df, subject_name, many_animals, sep_corr_incorr):
        df = df.copy()
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        num_trials = len(df)
        num_sessions = len(df[["Name", "Date", "SessionNum"]].drop_duplicates())
        num_subjects = len(df.Name.unique())
        # clr =
        _rtVsDifficulty(df, subject_name, ax=ax,
                        many_animals=many_animals,
                        sep_corr_incorr=sep_corr_incorr,
                        clr=None,
                        #legend_loc=[1.15, 0.1 if q_idx == 1 else 0.7],
                        #x_offset=-0.02 if q_idx == 1 else 0.02,
                        linfit=False)

        _setupAxes(ax, many_animals, is_CDF=False, subject_name=subject_name,
                   num_trials=num_trials, num_sessions=num_sessions,
                   num_subjects=num_subjects if many_animals else None,
                   set_xlim=None)
        [ax.spines[_dir].set_visible(False) for _dir in ["left", "right"]]
        if save_figs:
            mid_str = "_all" if sep_corr_incorr else "_corr_incorr"
            fig.savefig(f"{save_prefix}st_vs_diff_only/"
                        f"overall_{title}_{subject_name}{mid_str}"
                        # f"{'_including_tpical' if plot_typical else ''}"
                        f"_st_vs_diff.svg")
        plt.show()

    if plot_combined_subjects:
        # _processSubject(df, title, many_animals=True, sep_corr_incorr=False)
        _processSubject(df, title, many_animals=True, sep_corr_incorr=True)
    if not plot_single_subjects:
        return
    for subject_name, subject_df in df.groupby("Name"):
        print(subject_name)
        # _processSubject(subject_df, subject_name, many_animals=False,
        #                 sep_corr_incorr=False)
        _processSubject(subject_df, subject_name, many_animals=False,
                        sep_corr_incorr=True)

def stVsDiffOnly(df, plot_combined_subjects=True, 
                 plot_single_subjects=True, y_fast_min=None,
                 y_slow_min=None, save_prefix="", save_figs=False,
                 all_dscrp="All Mice"):
    df = df[df.calcStimulusTime.notnull()]
    df = normalizeSTAcrossSubjects(df)
    # ax.legend(loc="upper right")
    def _processSubject(df, subject_name, many_animals, sep_corr_incorr):
        df = df.copy()
        fig, main_ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        twin_ax = main_ax.twinx()
        num_trials = len(df)
        num_sessions = len(df[["Name", "Date", "SessionNum"]].drop_duplicates())
        num_subjects = len(df.Name.unique())
        for q_idx, q_df in reversed(list(df.groupby("quantile_idx"))):
            assert q_idx in (1, 2, 3)
            if q_idx == 2:# and not plot_typical:
                continue
            ax = main_ax if q_idx == 3 else twin_ax
            q_str = "Fast" if q_idx == 1 else ("Typical" if q_idx == 2 else
                                               "Slow")
            clr = "red" if q_idx == 1 else ("orange" if q_idx == 2 else
                                            "yellow")
            _rtVsDifficulty(q_df, f"{subject_name} - {q_str}", ax=ax,
                            many_animals=many_animals,
                            sep_corr_incorr=sep_corr_incorr,
                            clr=clr,
                            legend_loc=[1.15, 0.1 if q_idx == 1 else 0.7],
                            x_offset=-0.02 if q_idx == 1 else 0.02,
                            linfit=False, plot_incorr=False)

        _setupAxes(main_ax, many_animals, is_CDF=False,
                   subject_name=subject_name, num_trials=None, #num_trials,
                   num_sessions=num_sessions,
                   num_subjects=num_subjects if many_animals else None,
                   set_xlim=None)

        twin_ax.set_title("")
        twin_ax.yaxis.tick_left()
        twin_ax.yaxis.set_label_position("left")
        # twin_ax.tick_params(axis='y', colors='red')
        twin_ax.spines["left"].set_color("red")
        # Set paddding
        twin_ax.spines["left"].set_position(("axes", -0.12))
        main_ax.spines["left"].set_color("yellow")
        # main_ax.tick_params(axis='y', colors='gold')
        main_ax.set_ylabel("")
        biggest_rng = 0
        for ax in [main_ax, twin_ax]:
            [ax.spines[_dir].set_visible(False) for _dir in [#"left",
                                                             "right"]]
            ylim = ax.get_ylim()
            # ax.set_ylim((ylim[0] - 0.05, ylim[1]+0.05))
            biggest_rng = max(biggest_rng, abs(ylim[1] - ylim[0]))
        # Round it to the nearest .1
        biggest_rng = round(biggest_rng * 10) / 10
        print(f"biggest_rng: {biggest_rng}")
        # Make both axes span the same range
        for is_fast, ax in [(False, main_ax), (True, twin_ax)]:
            ylim = ax.get_ylim()
            ylim = (floor(ylim[0] * 10) / 10, ceil(ylim[1] * 10) / 10)
            if is_fast and y_fast_min is not None:
                ylim = (y_fast_min, y_fast_min + biggest_rng)
            elif not is_fast and y_slow_min is not None:
                ylim = (y_slow_min, y_slow_min + biggest_rng)
            else:
                mid = ylim[0] + (ylim[1] - ylim[0]) / 2
                ylim = mid - biggest_rng/2, mid + biggest_rng/2
                # if is_fast:
                # else:
                #     ylim = ylim[0] , ylim[0] + biggest_rng
                
            ax.set_ylim(ylim)
            multiple_locator = 0.1 if biggest_rng < 2 else 0.5
            if (is_fast and y_fast_min is not None) or \
               (not is_fast and y_slow_min is not None):
                ticks = np.arange(ylim[0], ylim[0] + biggest_rng, 
                                  multiple_locator)    
                ax.yaxis.set_ticks(ticks)
            else:
                ax.yaxis.set_major_locator(plticker.MultipleLocator(multiple_locator))
            ax.yaxis.set_major_formatter(plticker.FormatStrFormatter('%.1f'))

        if save_figs:
            mid_str = "_all" if sep_corr_incorr else "_corr_incorr"
            save_fp = Path(f"{save_prefix}st_vs_diff_fast_slow/"
                            f"{subject_name}{mid_str}_st_vs_diff.svg")
            save_fp.parent.mkdir(exist_ok=True)
            fig.savefig(save_fp)
        plt.show()

    if plot_combined_subjects:
        # _processSubject(df, all_dscrp, many_animals=True, sep_corr_incorr=False)
        _processSubject(df, all_dscrp, many_animals=True, sep_corr_incorr=True)
    if not plot_single_subjects:
        return
    for subject_name, subject_df in df.groupby("Name"):
        print(subject_name)
        # _processSubject(subject_df, subject_name, many_animals=False,
        #                 sep_corr_incorr=False)
        _processSubject(subject_df, subject_name, many_animals=False,
                        sep_corr_incorr=True)


def _setupAxes(ax, many_animals, is_CDF, subject_name, num_trials=None,
               num_sessions=None, num_subjects=None, x_lim=None,
               set_xlim=True):
    ax.set_xlabel("Stimulus Time " +
                  ("(normalized)" if many_animals else "(s)"))
    if set_xlim:
        if x_lim is not None:
            ax.set_xlim(x_lim)
        else:
            if many_animals:
                ax.set_xlim(-1.3, 1.7)
            else:
                ax.set_xlim(0.3, 3)

    if num_trials is not None or num_subjects is not None or \
       num_sessions is not None:
        num_trials_str = " ("
        if num_subjects is not None:
            num_trials_str += f"{num_subjects} subjects / "
        if num_sessions is not None:
            num_trials_str += f"{num_sessions} sessions / "
        if num_trials is not None:
            num_trials_str += f"{num_trials:,} trials)"
        else:
            num_trials_str = num_trials_str[:-3] + ")"
    else:
        num_trials_str = ""
    ax.set_title(f"{subject_name}{num_trials_str}")
    if not is_CDF:
        ax.title.set_position([.85, 1])

def _handleDifficultyDf(dv, dv_df, ax, clr, CDF, many_animals=False,
                        plot_median=True, as_kde=False):
    # print(f"Number of trials for {dv}: {len(dv_df)}")
    assert len(dv_df) > 0, f"No data for {dv}"
    # print(f"Number of trials for {dv}: {len(dv_df)}")
    col = ("calcStimulusTime" if not many_animals else
           "transformedCalcStimulusTime")
    if CDF:
        # sns.ecdfplot(data=dv_df, x=col, ax=ax, color=clr,
        #              label=f"{dv} (n={len(dv_df):,} trials)")
        sns.kdeplot(data=dv_df, x=col, fill=False, ax=ax, color=clr,
                    label=f"{dv} (n={len(dv_df):,} trials)")
    else:
        # display(
        #     dv_df[dv_df.AuditoryTrial == 1].GUI_ExperimentType.value_counts())
        if as_kde:
            assert not CDF
            sns.kdeplot(data=dv_df, x=col, fill=True, #
                        ax=ax, bw_adjust=0.3)
            path = ax.collections[0].get_paths()[0]
            patch = mpatches.PathPatch(path, transform=ax.transData)
        else:
            BIN_STEP = 0.04
            min_bin, max_bin = ax.get_xlim()
            min_bin = min_bin - BIN_STEP
            max_bin = max_bin + BIN_STEP
            # print(f"min_bin: {min_bin}, max_bin: {max_bin}")
            # bins = np.arange(-2, 3.1, 0.04)
            bins = np.arange(min_bin, max_bin, 0.04)
            assigned_rt_bin_idxs = np.digitize(dv_df[col], bins, right=True)
            color_map = mplcm.get_cmap("autumn", 1024)

            for rt_bin_idx in np.arange(len(bins)):
                cur_rt_idxs_mask = assigned_rt_bin_idxs == rt_bin_idx
                total_rt_count = cur_rt_idxs_mask.sum()
                bin_val = bins[rt_bin_idx] + BIN_STEP / 2
                cdf_rt_prcnt = (dv_df[col] <= bin_val).sum() / len(dv_df[col])
                clr = color_map(int(cdf_rt_prcnt * 1023))
                ax.bar(bins[rt_bin_idx], total_rt_count, color=clr,
                       width=BIN_STEP, align="edge")


        q1_lim = dv_df.groupby("Name")[col].quantile(1/3)
        q3_lim = dv_df.groupby("Name")[col].quantile(2/3)
        ax.axvline(q1_lim.mean(), color="gray", alpha=.5, linestyle="--", zorder=20)
        ax.axvline(q3_lim.mean(), color="gray", alpha=.5, linestyle="--", zorder=20)
        x_lim = ax.get_xlim()
        ax.axvspan(x_lim[0], q1_lim.mean(), color="r", alpha=0.1, zorder=20)
        ax.axvspan(q1_lim.mean(), q3_lim.mean(), color="orange", alpha=0.1, zorder=20)
        ax.axvspan(q3_lim.mean(), x_lim[1], color="yellow", alpha=0.1, zorder=20)
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        text_kwargs = dict(fontsize=13, fontweight="normal",
                           transform=trans, ha="center")
        t_fast = ax.text(x_lim[0] + (q1_lim.mean()-x_lim[0])/2, .8, "Fast",
                         color='r', **text_kwargs)
        t_fast_pos = t_fast.get_position()
        # Draw left pointing arrow below the text
        ax.annotate("", xy=(t_fast_pos[0] + .1, 0.75),
                    xytext=(t_fast_pos[0] - 0.1, 0.75),
                    arrowprops=dict(arrowstyle="<-", color='r'),
                    xycoords=trans)
        ax.text(.1 + q1_lim.mean() + (q3_lim.mean()-q1_lim.mean())/2, .8,
        # ax.text(.5, .8, color='orange',
                "Typical", color='orange', **text_kwargs)
        t_slow = ax.text(q3_lim.mean() + (x_lim[1]-q3_lim.mean())/2, .8, "Slow",
                         bbox=dict(facecolor='gray', alpha=.6, edgecolor='none'),
                         color='yellow', **text_kwargs)
        # Draw right pointing arrow below the text
        t_slow_pos = t_slow.get_position()
        ax.annotate("", xy=(t_slow_pos[0] + .1, 0.75),
                    xytext=(t_slow_pos[0] - 0.1, 0.75),
                    arrowprops=dict(arrowstyle="->", color='yellow'),
                    xycoords=trans)
    if plot_median:
        mean = dv_df.groupby("Name")[col].quantile(0.5).mean()
        # mean = dv_df[col].mode()[0]
        # print(f"Mean {dv}: {mean}")
        ax.axvline(mean, color=clr, linestyle="--" if CDF else None, zorder=20)


def _rtVsDifficulty(df, animal_name, ax, many_animals, sep_corr_incorr=True,
                    clr=None, legend_loc=None, x_offset=0, linfit=False,
                    ls="-", marker=None, plot_legend=True, legend_fontsize=None,
                    col_rt="calcStimulusTime", col_corr="ChoiceCorrect",
                    plot_incorr=True):
    col_rt = (col_rt if not many_animals else
             "transformedCalcStimulusTime")
    df = df[df[col_rt].notnull()]
    if sep_corr_incorr:
        clr_corr = 'g' if clr is None else clr
        clr_incorr = 'r' if clr is None else clr
        ls_incorr = ls if clr is None else "--"
        loop_li = [(df[df[col_corr] == 1], "Correct", clr_corr, ls)]
        if plot_incorr:
            loop_li.append(
                   (df[df[col_corr] == 0], "Incorrect", clr_incorr, ls_incorr))
    else:
        loop_li = [
            (df, f"{animal_name}", 'k' if clr is None else clr, ls),
        ]

    for sub_df, label, c, _ls in loop_li:
        x_data = []
        y_data = []
        y_data_sem = []
        count = 0
        for dv_df in (sub_df[sub_df.DVstr == "Hard"],
                      sub_df[sub_df.DVstr == "Med"],
                      sub_df[sub_df.DVstr == "Easy"]):
            count += 1
            if not len(dv_df):
                continue
            x_data.append(count)
            if many_animals:
                st = dv_df.groupby("Name")[col_rt].mean()
                y_mean = st.mean()
                y_sem = st.sem()
                label_ = f"{y_mean:.2f}±{y_sem:.2f} SEM\n" + \
                         f"n={len(st):,} Subjects, {len(dv_df):,} Trials"
            else:
                if "SessId" in dv_df.columns:
                    group_cols = ["SessId"]
                else:
                    group_cols = ["Date", "SessionNum"]
                st = dv_df.groupby(group_cols)[col_rt].mean()
                y_mean = st.mean()
                y_sem = st.sem()
                label_ = f"{y_mean:.2f}±{y_sem:.2f} SEM\n" + \
                         f"n={len(st):,} Sessions, {len(dv_df):,} Trials"
            # Annotate below
            ax.annotate(label_, (count + 0.01, y_mean - 0.01), color=c)
            y_data.append(y_mean)
            y_data_sem.append(y_sem)
            # st = dv_df[col_rt]
            # y_data.append(st.mean())
            # y_data_sem.append(st.sem())
        x_data = np.array(x_data)
        n_subjects = len(sub_df.Name.unique())
        ax.errorbar(x_data + x_offset, y_data, yerr=y_data_sem, color=c,
                    label=f"{label} (n={n_subjects} Subjects, "
                          f"{len(sub_df):,} trials)",
                    linestyle=_ls, marker=marker)
        if linfit:
            x_data = np.array(x_data)
            y_data = np.array(y_data)
            reg_res = stats.linregress(x_data, y_data)
            ax.plot(x_data, reg_res.intercept + reg_res.slope * x_data,
                    color=c, linestyle=":", alpha=0.3)
            slope = -reg_res.slope
            # Annotate on top of the first value
            slope_angle = np.rad2deg(np.arctan(slope))
            ax.annotate(f"Slope={slope:.2f} - θ={slope_angle:.2f}°",
                        (x_data[0] + 0.1, y_data[0] + 0.02),
                        color=c)#'k' if c == "yellow" else c),

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Hard", "Med", "Easy"])
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Stimulus Time " +
                  ("(Normalized)" if many_animals else "(s)"))
    ax.set_title(f"{animal_name} (n={len(df):,} trials)")
    if plot_legend:
        foctsize_dict = {} if legend_fontsize is None else \
                        {"fontsize": legend_fontsize}
        ax.legend(loc=legend_loc, **foctsize_dict)
    [ax.spines[_dir].set_visible(False) for _dir in ["top", "right"]]

def _processSubject(df, subject_name, many_animals=False,
                    save_prefix="", CDF_xlim=None, save_figs=False):
    # df = df[df.ChoiceCorrect == False]
    grid_size = 3, 3
    fig = plt.figure(figsize=(grid_size[1]*9, grid_size[0]*2),
                     tight_layout=True)
    gs = GridSpec(grid_size[0], grid_size[1], figure=fig)
    ax_cdf = fig.add_subplot(gs[:, 0])
    ax_easy = fig.add_subplot(gs[0, 1])
    ax_med =  fig.add_subplot(gs[1, 1])
    ax_hard = fig.add_subplot(gs[2, 1])
    ax_diff_vs_rt = fig.add_subplot(gs[:, 2])

    total_num_trials = 0
    for dv, clr, ax_dv in [("Easy", DifficultyClr.Easy, ax_easy),
                           ("Med", DifficultyClr.Med, ax_med),
                           ("Hard", DifficultyClr.Hard, ax_hard)]:
        dv_df = df[df.DVstr == dv]
        num_trials = len(dv_df)
        total_num_trials += num_trials
        # print(f"Number of trials for {dv}: {num_trials}")
        if not many_animals and not len(dv_df):
            continue
        num_trials = len(dv_df)
        num_sessions = len(
                        dv_df[["Name", "Date", "SessionNum"]].drop_duplicates())
        num_subjects = len(dv_df.Name.unique()) if many_animals else None
        _setupAxes(ax_dv, many_animals, is_CDF=False,
                   subject_name=subject_name, num_trials=num_trials,
                   num_sessions=num_sessions, num_subjects=num_subjects)
        _handleDifficultyDf(dv, dv_df, ax_cdf, clr, CDF=True,
                            many_animals=many_animals)
        _handleDifficultyDf(dv, dv_df, ax_dv,  clr, CDF=False,
                            many_animals=many_animals, plot_median=False)
        # ax_dv.legend()

    for ax_diff in [ax_easy, ax_med, ax_hard]:
        [ax_diff.spines[_dir].set_visible(False)
         for _dir in ["left", "right", "top"]]

    for ax_diff in [ax_cdf]:
        [ax_diff.spines[_dir].set_visible(False) for _dir in ["left", "top"]]
    # Put ax_cdf y-axis on the right
    ax_cdf.yaxis.tick_right()
    ax_cdf.yaxis.set_label_position("right")

    _setupAxes(ax_cdf, many_animals, is_CDF=True, subject_name=subject_name,
               num_trials=len(df), x_lim=CDF_xlim)
    ax_cdf.axhline(0.5, color="gray", linestyle="--")
    ax_cdf.legend()
    _rtVsDifficulty(df, subject_name, ax_diff_vs_rt, many_animals=many_animals)
    if save_figs:
        if subject_name == "All Mice":
            subject_name = "_" + subject_name # Make it show up first
        save_fp = Path(f"{save_prefix}rt_by_difficulty/{subject_name}.svg")
        save_fp.parent.mkdir(exist_ok=True)
        fig.savefig(save_fp)
    plt.show()
