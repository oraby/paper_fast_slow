import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy import stats
import scikit_posthocs as sp

def _calcSessAvgRewardRate(sess_df, choice_col, rr_postfix=""):
    choice_correct_no_nan = sess_df[choice_col].fillna(0)
    sess_df = sess_df.copy()
    rr_cols = []
    for num_past_trials in range(2, 11):
        rr_col = f"RewardRate{rr_postfix}{num_past_trials}"
        sess_df[rr_col] = choice_correct_no_nan.rolling(num_past_trials).mean().shift(1)
        rr_cols.append(rr_col)
    # display(sess_df.head(20)[["Name", "TrialNumber", choice_col] + rr_cols])
    return sess_df

def calcAvgRewardRate(df, choice_cols=["ChoiceCorrect"], rr_postfixs=[""],
                      groupby_cols=["Name", "Date", "SessionNum"]):
    df = df.copy()
    df_li = []
    # Assert that the df is already sorted in at least at groupby_cols level by
    # TrialNumber
    assert df.groupby(groupby_cols).apply(
                    lambda sess_df:sess_df.TrialNumber.diff()[1:].gt(0).all()
                    or (display(sess_df.TrialNumber.diff()) is False)).all(), (
        "If not sorted by TrialNumber, how can we calculate the reward rate?")
    # Supress pandas warning
    groupby_cols_iter = groupby_cols[0] if len(groupby_cols) == 1 else groupby_cols
    for sess, sess_df in df.groupby(groupby_cols_iter):
        for choice_col, rr_postfix in zip(choice_cols, rr_postfixs):
            sess_df = _calcSessAvgRewardRate(sess_df, choice_col, rr_postfix)
        df_li.append(sess_df)
    res_df = pd.concat(df_li)
    res_df = res_df.sort_values(groupby_cols + ["TrialNumber"])
    res_df = res_df.reset_index(drop=True)
    return res_df



def plotSubjectRewardRateRt(subject, subject_df, col_postfix, rt_col, RT_ZSCORE,
                            num_past_trials, BY_SESS, min_trials_per_sess_rr,
                            plot=True, save_figs=False,
                            descrp="", save_prefix=None):
    if save_figs:
        assert save_prefix is not None
    # print("Subject:", subject, "Num Trials:", len(subject_df))
    if "valid" in subject_df.columns:
        subject_df = subject_df[subject_df.valid]
    if RT_ZSCORE:
        subject_df = subject_df.copy()
        subject_df[rt_col] = stats.zscore(subject_df[rt_col], nan_policy="omit")

    res_dict = {"GUI_TimeOutIncorrectChoice": [],
                "bin": [],
                "RewardRateRTAvg": [],
                "RewardRateRTSEM": [],
                "RewardRateCount": [],
                "TotalNumTrials":[]}
    kwargs = dict(subject=subject, col_postfix=col_postfix, rt_col=rt_col,
                  RT_ZSCORE=RT_ZSCORE, num_past_trials=num_past_trials,
                  min_trials_per_sess_rr=min_trials_per_sess_rr,
                  BY_SESS=BY_SESS, plot=plot, save_figs=save_figs,
                  descrp=descrp, save_prefix=save_prefix)

    for timeout, timeout_df in subject_df.groupby("GUI_TimeOutIncorrectChoice"):
        timeout_res_dict = _plotRewardRateRT(df=timeout_df, timeout=timeout,
                                             **kwargs)
        for key, val in timeout_res_dict.items():
            res_dict[key].extend(val)
    all_res_dict = _plotRewardRateRT(df=subject_df, timeout="All",
                                     **kwargs)
    for key, val in all_res_dict.items():
        res_dict[key].extend(val)
    res_df = pd.DataFrame(res_dict)
    res_df["Name"] = subject
    res_df["col_postfix"] = col_postfix
    res_df["num_trials"] = len(subject_df)
    return res_df

def _plotRewardRateRT(subject, df, col_postfix, rt_col, RT_ZSCORE,
                      num_past_trials, BY_SESS,  timeout,
                      min_trials_per_sess_rr, plot=True,
                      save_figs=False, descrp="", save_prefix=None):
    if save_figs:
        assert save_prefix is not None
    RewardRateCol = f"RewardRate{col_postfix}{num_past_trials}"
    CUT_SIZE = 1/num_past_trials
    ratios_bins = np.arange(0, 1 + CUT_SIZE + .001, CUT_SIZE)
    res_dict = {"GUI_TimeOutIncorrectChoice": [],
                "bin": [],
                "RewardRateRTAvg": [],
                "RewardRateRTSEM": [],
                "RewardRateCount": [],
                "TotalNumTrials":[]}
    # print("Ratio Bins:", ratios_bins)
    rr_vals = df[RewardRateCol]
    # Round to the nearest 5th decimal to avoid floating point errors
    rr_vals += 1e-5
    bins = pd.cut(rr_vals, ratios_bins, right=False)
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        twin_ax = ax.twinx()
    groupby_obj = df.groupby(bins)
    bin_xs = []
    ys = []
    ys_err = []
    for bin_name, bin_df in groupby_obj:
        if not len(bin_df):
            continue
        total_num_trials = len(bin_df)
        if BY_SESS:
            bin_df = bin_df.groupby("SessId")
            # Print each sess size
            bin_df = bin_df.filter(lambda x: len(x) > min_trials_per_sess_rr)
            if not len(bin_df):
                continue
            bin_df = bin_df.groupby("SessId")
            # Recompute total_num_trials after filtering
            total_num_trials = bin_df.size().sum()
        bin_x = bin_name.left #+ CUT_SIZE/2
        bin_xs.append(bin_x)
        if BY_SESS:
            y = bin_df[rt_col].mean()
            # print("y:", y)
            y_avg = y.mean()
            y_err = y.sem()
        else:
            y = bin_df[rt_col]
            y_avg = y.median() if not RT_ZSCORE else y.mean()
            y_err = y.sem()
        ys.append(y_avg)
        ys_err.append(y_err if len(bin_df) > 1 else 0)
        res_dict["GUI_TimeOutIncorrectChoice"].append(timeout)
        res_dict["bin"].append(bin_name)
        res_dict["RewardRateRTAvg"].append(y_avg)
        res_dict["RewardRateRTSEM"].append(y_err)
        res_dict["RewardRateCount"].append(len(bin_df))
        res_dict["TotalNumTrials"].append(total_num_trials)

        if plot:
            twin_ax.bar(bin_x, len(bin_df), label=bin_name,
                        width=CUT_SIZE - CUT_SIZE/5, color="gray", alpha=.2)

    if plot:
        # ys = groupby_obj[rt_col].mean()
        # ys_err = groupby_obj[rt_col].sem() if len(bin_df) > 1 else 0
        # print("Y:", ys)
        ax.errorbar(bin_xs, ys, yerr=ys_err, label=bin_name)
        ax.set_xlabel(f"Reward Rate (Past {num_past_trials} Trials)")
        ax.set_ylabel("Reaction-Time" + (" (Z-Score)" if RT_ZSCORE else ""))
        ax.set_title((descrp + '\n' if len(descrp) else "") +
                     f"Subject: {subject} - Incorrect Timeout: ~{timeout}s")
        ax.spines[['top', 'right']].set_visible(False)
        twin_ax.spines[['top', 'right']].set_visible(False)
        twin_ax.set_ylabel("Trials Count")
        if save_figs:
            save_fp = _buildSaveFP(save_prefix, subject, descrp, timeout,
                                   BY_SESS, RT_ZSCORE, num_past_trials)
            fig.savefig(save_fp,  bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    return res_dict


def loopRewardRateAnalysis(df, plot_subjects,
                           plot_singles_on_all=True,
                           save_figs=False, descrp="",
                           save_prefix=None, ax_all=None,
                           ax_all_used_timeout_time=None,
                           min_trials_per_sess_rr=5,
                           min_num_trials=500):
    if save_figs:
        assert save_prefix is not None
    if ax_all is not None:
        assert ax_all_used_timeout_time is not None
    df_li = []
    REWARD_RATE_NUM_PAST_TRIALS = 5
    BY_SESS = True
    RT_ZSCORE = True

    postfix_rt_col_tups = [("", "calcStimulusTime")]
    if "SimRT" in df.columns:
        postfix_rt_col_tups.append(("Sim", "SimRT"))
    for name, subject_df in df.groupby("Name"):
        # print("Subject DF:", subject_df)
        for col_postfix, col_rt in postfix_rt_col_tups:
            # print("Col Postfox:", col_postfix)
            res_df = plotSubjectRewardRateRt(name, subject_df, col_postfix,
                                             col_rt,
                                             num_past_trials=REWARD_RATE_NUM_PAST_TRIALS,
                                             BY_SESS=BY_SESS,
                                             min_trials_per_sess_rr=min_trials_per_sess_rr,
                                             save_figs=save_figs,
                                             RT_ZSCORE=RT_ZSCORE,
                                             plot=False, descrp=descrp,
                                             save_prefix=save_prefix)

            # plotSubjectSessQuantiles(name, subject_processed_df, col_prefix,
            #                          col_rt, save_figs=save_figs)
            df_li.append(res_df)
    res_df = pd.concat(df_li)
    # display(res_df[["Name", "col_postfix", "num_trials"]].value_counts())
    if plot_subjects:
        for subject, subject_df in res_df.groupby("Name"):
            _plotLoopResults(subject, subject_df,
                             RT_ZSCORE, REWARD_RATE_NUM_PAST_TRIALS, BY_SESS,
                             descrp=descrp, save_figs=save_figs,
                             save_prefix=save_prefix)
    res_df = res_df[res_df.num_trials >= min_num_trials]
    if ax_all_used_timeout_time is not None:
        res_df = res_df[res_df.GUI_TimeOutIncorrectChoice ==
                        ax_all_used_timeout_time]
    _plotLoopResults("All Subjects", res_df, RT_ZSCORE,
                     REWARD_RATE_NUM_PAST_TRIALS, BY_SESS,
                     descrp=descrp, save_figs=save_figs,
                     plot_singles_on_all=plot_singles_on_all,
                     save_prefix=save_prefix, use_ax=ax_all)


def _plotLoopResults(subject, res_df, RT_ZSCORE, REWARD_RATE_NUM_PAST_TRIALS,
                     BY_SESS, save_figs=False, descrp="", save_prefix=None,
                     use_ax=None, plot_singles_on_all=True):
    if save_figs:
        assert save_prefix is not None
    if use_ax is not None:
        assert res_df.GUI_TimeOutIncorrectChoice.nunique() == 1
    for timeout, timeout_both_df in res_df.groupby("GUI_TimeOutIncorrectChoice"):
        if use_ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        else:
            ax = use_ax
        groups_ys = {}
        for col_postfix, timeout_df in timeout_both_df.groupby("col_postfix"):
            groupby_obj = timeout_df.groupby(f"bin")
            xs, ys, ys_err = [], [], []
            # Map each subject its list of ys
            ys_singles = {subject_: []
                          for subject_ in timeout_both_df["Name"].unique()}
            for bin_name, bin_df in groupby_obj:
                if not len(bin_df):
                    print("Skipping:", bin_name, "for timeout:", timeout,
                          "for subject:", subject)
                    continue
                y = bin_df["RewardRateRTAvg"].mean()
                y_err = bin_df["RewardRateRTAvg"].sem()
                x = bin_name.left
                # print("Bin Name:", bin_name, "X:", x, "Y:", y, "Y Err:", y_err)
                xs.append(x)
                ys.append(y)
                ys_err.append(y_err)
                if bin_name not in groups_ys:
                    groups_ys[bin_name] = []
                for subject_ in ys_singles.keys():
                    subject_bin_df = bin_df[bin_df["Name"] == subject_]
                    if not len(subject_bin_df):
                        y_single = np.nan
                    else:
                        y_single = subject_bin_df["RewardRateRTAvg"]
                        assert len(y_single) == 1
                        y_single = y_single.iloc[0]
                        groups_ys[bin_name].append(y_single)
                    ys_singles[subject_].append(y_single)
                # Annotate each point with the number of subjects, mean and SEM
                num_subj = len(bin_df)
                num_trials = bin_df["TotalNumTrials"].sum()
                ax.annotate(f" n={num_subj} Subj ({num_trials:,} Trials)\n"
                            f"mean={y:.2f} ±{y_err:.2f}SEM",
                            (x, y), textcoords="offset points", xytext=(0, 10),
                             ha='left', va='bottom')
            # print("Col Postfix:", col_postfix, "Timeout:", timeout)
            total_num_trials = timeout_df["TotalNumTrials"].sum()
            err_bars = ax.errorbar(xs, ys, yerr=ys_err,
                                   label=(timeout if plot_singles_on_all else descrp) +
                                   f"\n({total_num_trials:,} trials)")
            color = err_bars.lines[0].get_color()
            if plot_singles_on_all:
                for subject_, ys_single in ys_singles.items():
                    ax.plot(xs, ys_single, color="gray", alpha=.2)

        # Check normality of each ys-group
        if plot_singles_on_all:
            all_normal = True
            for group, group_ys in groups_ys.items():
                print("Group:", group)
                # print("Group Ys:", list(reversed(sorted(group_ys))))
                stat, pval = stats.shapiro(group_ys)
                is_normal = pval >= 0.05
                print("Shapiro-Wilk Test for normality - Subject/Group:", subject,
                      "Timeout:", timeout, "Stat:", stat, "Pval:", pval,
                      "Is Normal:" , is_normal)
                all_normal = all_normal and is_normal
            if all_normal:
                print("All groups normal - Proceeding with ANOVA")
                fstat, pval = stats.f_oneway(*groups_ys.values())
                print("One-Way ANOVA - Subject/Group:", subject,
                    "Timeout:", timeout, "Fstat:", fstat, "Pval:", pval)
                posthoc_fn = sp.posthoc_tukey
                posthoc_fn_str = "Tukey's"
                posthoc_kwargs = {}
            else:
                print("Not all groups normal - Proceeding with Kruskal-Wallis Test")
                stat, pval = stats.kruskal(*groups_ys.values())
                print("Kruskal-Wallis Test - Subject/Group:", subject,
                    "Timeout:", timeout, "Stat:", stat, f"Pval: {pval:.4f}")
                posthoc_fn = sp.posthoc_dunn
                posthoc_fn_str = "Dunn's"
                posthoc_kwargs = {"p_adjust": "holm"}
            # Run post-hoc test correction if significant
            stats_str = "Stats Method: " + \
                        ("ANOVA with Tukey's post-hoc" if all_normal else
                         "Kruskal-Wallis with Dunn's post-hoc")
            if pval < 0.05:
                print("Significant differences found - Proceeding with post-hoc tests")
                data_for_posthoc = []
                group_labels = []
                for group, group_ys in groups_ys.items():
                    for y in group_ys:
                        if not np.isnan(y):
                            data_for_posthoc.append(y)
                            group_labels.append(str(group))
                posthoc_res = posthoc_fn(pd.DataFrame({"Value": data_for_posthoc,
                                                    "Group": group_labels}),
                                        val_col="Value", group_col="Group",
                                        **posthoc_kwargs)
                print(f"Post-hoc {posthoc_fn_str} test results:\n",
                      str(posthoc_res).replace("    ", "\t"))
                # Plot significance connections with stars for p < 0.05
                y_offset = 0.1
                for i, group1 in enumerate(groups_ys.keys()):
                    for j, group2 in enumerate(groups_ys.keys()):
                        if j <= i:
                            continue
                        pval_posthoc = posthoc_res.loc[str(group1), str(group2)]
                        if pval_posthoc < 0.05:
                            x1 = group1.left
                            x2 = group2.left
                            y_max = max(ys + ys_err) + 0.1 + y_offset
                            ax.plot([x1, x1, x2, x2],
                                    [y_max - 0.05, y_max, y_max, y_max - 0.05],
                                    color=color)
                            stars_str = "***" if pval_posthoc < 0.001 else (
                                        "**" if pval_posthoc < 0.01 else (
                                        "*" if pval_posthoc < 0.05 else "n.s."))
                            ax.text((x1 + x2)/2, y_max - 0.01,
                                    stars_str, ha='center', va='bottom')
                            y_offset += 0.15
        else:
            stats_str = ""

        ax.set_xlabel(f"Reward Rate (Past {REWARD_RATE_NUM_PAST_TRIALS} Trials)")
        ax.set_ylabel("Reaction-Time" + (" (Z-Score)" if RT_ZSCORE else ""))
        ax.set_title((descrp + '\n' if len(descrp) else "") +
                     f"{subject} - Reward Rate vs RT - Timeout: ~{timeout}s"
                     + ("\n" + stats_str if len(stats_str) else ""))
        ax.spines[['top', 'right']].set_visible(False)
        if save_figs:
            if use_ax:
                ax.legend()
                fig = ax.get_figure()
            save_fp = _buildSaveFP(save_prefix, subject, descrp, timeout,
                                   BY_SESS,
                                   RT_ZSCORE, REWARD_RATE_NUM_PAST_TRIALS)
            fig.savefig(save_fp,  bbox_inches='tight')
            plt.close(fig)
        elif use_ax is None:
            plt.show()

def _buildSaveFP(save_prefix, subject, descrp, timeout, BY_SESS, RT_ZSCORE,
                 num_past_trials):
    label = f"RewardRate_RT"
    print("Saving:", label)
    by_sess_str = "BySess/" if BY_SESS else "AllTrials/"
    zscore_str = "ZScoreRT/" if RT_ZSCORE else "RawRT/"
    save_fp = Path(f"{save_prefix}/reward_rate{num_past_trials}/")
    save_fp.mkdir(exist_ok=True)
    save_fp = save_fp / f"{descrp}/"
    save_fp.mkdir(exist_ok=True)
    save_fp = save_fp / f"{zscore_str}"
    save_fp.mkdir(exist_ok=True)
    save_fp = save_fp / f"{by_sess_str}"
    save_fp.mkdir(exist_ok=True)
    save_fp = save_fp / f"{subject}/"
    save_fp.mkdir(exist_ok=True)
    save_fp = save_fp / f"{label}_timeout{timeout}s.svg"
    return save_fp
