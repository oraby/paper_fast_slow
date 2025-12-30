import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

_CHOICE_LEFT_COL = "ChoiceLeft"
_GROUPBY_COLS = ["Name", "Date", "SessionNum", "quantile_idx"]
def calcSessionBias(sess_df, choice_left_col=_CHOICE_LEFT_COL):
    choices = sess_df[choice_left_col]
    choices = choices[choices.notnull()]
    # bias = choices.mean() - 0.5
    left_rewarded = sess_df.DV > 0
    bias = 100 * (choices.mean() - left_rewarded.mean())
    return bias

def calcSubjQuantileBias(subj_df, choice_left_col=_CHOICE_LEFT_COL,
                         groupby_cols=_GROUPBY_COLS):
    return subj_df.groupby(groupby_cols).apply(calcSessionBias,
                                               choice_left_col=choice_left_col)

def calcBias(df, choice_left_col=_CHOICE_LEFT_COL, groupby_cols=_GROUPBY_COLS):
    return df.groupby("Name").apply(calcSubjQuantileBias,
                                    choice_left_col=choice_left_col,
                                    groupby_cols=groupby_cols)


def plotBias(df, as_abs, plot_single_subjects, save_figs=False,
             save_prefix=None, getAxFn=None,
             choice_left_col=_CHOICE_LEFT_COL, groupby_cols=_GROUPBY_COLS):
    bias_df = calcBias(df, choice_left_col=choice_left_col,
                       #groupby_cols=["Name", "quantile_idx"],
                       groupby_cols=groupby_cols)
    # display(bias_df)
    def setupAx(ax):
        ax.grid(True, axis="x", which="both", color="gray", linestyle="--", alpha=0.3)
        minor_ticks = np.arange(-0.5 if not as_abs else 0, 0.6, 0.05)
        ax.set_xticks(minor_ticks, minor=True)
        ax.invert_yaxis()
        ax.set_xlabel("Bias" + (" (Absolute)" if as_abs else " Left"))
        ax.set_xlim(-0.4 if not as_abs else -.01, 0.41)
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.3)

    all_subjects, subjects_biases, all_qs_sems = [], [], []
    for subj, subj_df in bias_df.groupby("Name"):
        if as_abs:
            subj_df = subj_df.abs()
        #     print(subj)
        #     display(subj_df)
        subj_df = subj_df.unstack()
        q1 = subj_df[1]
        q2 = subj_df[2]
        q3 = subj_df[3]
        subject_bias = q1.mean(), q2.mean(), q3.mean()
        qs_sems = q1.sem(), q2.sem(), q3.sem()
        subjects_biases.append(subject_bias)
        all_qs_sems.append(qs_sems)
        all_subjects.append(subj)
        if not plot_single_subjects:
            continue
        if getAxFn is not None:
            ax = getAxFn()
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        # ax.scatter(q1.values, np.arange(len(q1)), label="Fast", c="r")
        # ax.scatter(q2.values, np.arange(len(q2)), label="Typical", c="orange")
        # ax.scatter(q3.values, np.arange(len(q3)), label="Slow", c="yellow")
        q1_mean, q2_mean, q3_mean = subject_bias
        q1_sem, q2_sem, q3_sem = qs_sems
        ax.errorbar(q1_mean, len(q1), xerr=q1_sem, fmt="o", c="r")
        ax.errorbar(q2_mean, len(q2), xerr=q2_sem, fmt="o", c="orange")
        ax.errorbar(q3_mean, len(q3), xerr=q3_sem, fmt="o", c="yellow")
        ax.set_title(f"Bias for {subj}")
        ax.set_ylabel("Session Count")
        setupAx(ax)
        if save_figs:
            abs_str = "abs_" if as_abs else ""
            plt.savefig(f"{save_prefix}/bias/{abs_str}{subj}_bias.svg")
        if getAxFn is not None:
            plt.show()
        # break
    if getAxFn is not None:
        ax = getAxFn()
    else:
        num_subjects = len(all_subjects)
        if not as_abs:
            fig, ax = plt.subplots(1, 1, figsize=(6, 1 + num_subjects/5))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    subjects_biases = np.array(subjects_biases)
    # all_qs_sems = np.array(all_qs_sems)
    ys = np.arange(len(subjects_biases))

    # print("all_qs_means", all_qs_means)
    all_fast_biases = subjects_biases[:, 0]
    all_slow_biases = subjects_biases[:, 2]
    # print("q_fast", all_fast_biases)
    # print("q_slow", all_slow_biases)
    test_res = stats.ttest_rel(all_fast_biases, all_slow_biases)
    sgf_str = "***" if test_res.pvalue < 0.001 else (
              "**" if test_res.pvalue < 0.01 else (
              "*" if test_res.pvalue < 0.05 else
              "ns"))
    sgf_str= f" {sgf_str} - pval={test_res.pvalue:.3f}"
    # print("Fast vs Slow t-test", test_res)
    CONNECT = True #and as_abs
    if CONNECT:
        ys *= 2
        for idx, subject_bias in enumerate(subjects_biases):
            idx *= 2
            fast_bias, typical_bias, slow_bias = subject_bias
            if not as_abs:
                ax.barh(idx - .3, fast_bias, height=0.3, color="r",
                        label="Fast" if idx == 0 else None)
                # ax.barh(idx, typical, height=0.3, color="orange", label="Typical")
                ax.barh(idx + .3, slow_bias, height=0.3, color="yellow",
                        label="Slow" if idx == 0 else None)
            else:
                ax.plot([1.03, 1.97], [fast_bias, slow_bias], c="gray",
                        alpha=0.3, marker="o")
            # ax.plot([0, Fast], [idx, idx-.3],  c='r', ls="--")
            # # ax.plot([0, typical], [idx, idx], c='orange', ls="--")
            # ax.plot([0, Slow], [idx, idx+.3], c='yellow', ls="--")
        if as_abs:
            fast_mean = all_fast_biases.mean()
            fast_sem = stats.sem(all_fast_biases)
            df = df.copy()
            df["ShortName"] = df.Name + "_" + df.Date.astype(str) + "_" + \
                              df.SessionNum.astype(str)
            num_sessions = df[df.quantile_idx == 1].ShortName.nunique()
            ax.bar(1, fast_mean, yerr=fast_sem,
                   label=(f"Fast {fast_mean:.2f}% ±{fast_sem:.2f}%"
                          f" - n={len(all_fast_biases)} Subjects, {num_sessions} Sessions"),
                   color="r")

            slow_mean = all_slow_biases.mean()
            slow_sem = stats.sem(all_slow_biases)
            num_sessions = df[df.quantile_idx == 3].ShortName.nunique()
            ax.bar(2, slow_mean, yerr=slow_sem,
                   label=(f"Slow {slow_mean:.2f}% ±{slow_sem:.2f}% "
                          f"- n={len(all_slow_biases)} Subjects, {num_sessions} Sessions"),
                   color="yellow")
            max_y = max(all_fast_biases.max(), all_slow_biases.max())
            ax.annotate(sgf_str, xy=(1.5, max_y),
                        fontsize=12, ha="center", va="bottom")
        # ax.scatter(all_qs_means[:, 0], ys-.3, label="Fast", c="r")
        # # ax.scatter(all_qs_means[:, 1], ys, label="Typical", c="orange")
        # ax.scatter(all_qs_means[:, 2], ys+.3, label="Slow", c="yellow")
    else:
        ax.errorbar(subjects_biases[:, 0], ys,
                    xerr=all_qs_sems[:, 0], fmt="o", label="Fast", c="r")
        ax.errorbar(subjects_biases[:, 1], ys,
                    xerr=all_qs_sems[:, 1], fmt="o", label="Typical", c="orange")
        ax.errorbar(subjects_biases[:, 2], ys,
                    xerr=all_qs_sems[:, 2], fmt="o", label="Slow", c="yellow")

    ax.set_title("Bias for all subjects" + (" (Absolute)" if as_abs else ""))
    if not as_abs:
        ax.set_yticks(ys)
        ax.set_yticklabels(all_subjects)
        ax.tick_params(axis="y", which="both", length=0)
        setupAx(ax)
        ax.set_xlim(-0.1 if not as_abs else -.01, 0.1 if not as_abs else 0.15)
    else:
        ax.spines[["right", "top"]].set_visible(False)
        x_labels = ["Fast", "Slow"]
        ax.set_xticks([1, 2])
        ax.set_xticklabels(x_labels)
        ax.set_xlim(0, len(x_labels) + 1)
        ax.set_ylabel("Bias (Absolute)")
        # Set y formatter as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
                                                     lambda x, _: f"{int(x)}%"))

    ax.legend(fontsize="x-small")
    if save_figs:
        abs_str = "abs_" if as_abs else ""
        save_fp = Path(f"{save_prefix}/bias/{abs_str}all_bias.svg")
        save_fp.parent.mkdir(exist_ok=True)
        plt.savefig(save_fp, bbox_inches="tight")

    if getAxFn is not None:
        plt.show()
