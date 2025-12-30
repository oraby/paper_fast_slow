import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

def prevOutcomeCurQuantile(df, col_prev_choice_correct="PrevChoiceCorrect",
                           col_rt="calcStimulusTime",
                           col_quantile_idx="quantile_idx", ax=None,
                           save_prefix=None, save_fig=False):
    if save_fig:
        assert save_prefix is not None, "Please provide save_prefix"

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.get_figure()

    df = df[df[col_prev_choice_correct].notnull()]
    df = df[df[col_rt].notnull()]
    q1_df = df[df[col_quantile_idx] == 1]
    q2_df = df[df[col_quantile_idx] == 2]
    q3_df = df[df[col_quantile_idx] == 3]
    _plotPrevOutcomeCurQuantile(df, q1_df=q1_df, q2_df=q2_df,
                                q3_df=q3_df,
                                col_prev_choice_correct=col_prev_choice_correct,
                                ax=ax)
    if save_fig:
        fig.savefig(f"{save_prefix}/prev_choice_by_quantile.svg",
                    dpi=300, bbox_inches='tight')
    plt.show()

_Q_GRP_BY_COLS = ["Name"]
def quantilePrevOutcomeCur(q_df, col_prev_choice_correct):
    val = q_df.groupby(_Q_GRP_BY_COLS)[col_prev_choice_correct].mean()
    return val.mean()*100, val.sem()*100 if len(val) > 1 else 0

def _plotPrevOutcomeCurQuantile(df, q1_df, q2_df, q3_df,
                                col_prev_choice_correct, ax,
                                sess_split_cols=["Name", "Date", "SessionNum"],
                                width_scale=1, x_offset=0, hatch=None,
                                plot_q2=False, show_legend=True):
    WIDTH = .8 # Default ax.bar() width
    mean = df.groupby(_Q_GRP_BY_COLS)[col_prev_choice_correct].mean()
    # display(mean.mean())
    ax.axhline(100*mean.mean(), color="gray", label="Mean",
               linestyle="--"  if hatch is None else "-.")
    count = 1
    x_labels = []
    iter_li = [("r", q1_df, "Impulsive"),
                ("orange", q2_df, "Typical"),
                ("yellow", q3_df, "Deliberate")]
    if not plot_q2:
        iter_li.pop(1)
    for clr, q_df, dscrp in iter_li:
        num_trials = len(q_df)
        num_sessions = len(q_df[sess_split_cols].drop_duplicates())
        num_subjects = len(q_df.Name.unique())
        val_mean, err_bar = quantilePrevOutcomeCur(q_df,
                                                   col_prev_choice_correct)
        # print("Sem:", err_bar)
        label = (f"{dscrp} {val_mean:.2f}% ±{err_bar:.2f}%\n"
                 f"(N={num_subjects} subjects, {num_sessions:,} sessions, {num_trials:,} Trials")
        ax.bar(count + x_offset, val_mean, yerr=err_bar, color=clr, label=label,
               width=WIDTH*width_scale, hatch=hatch)
        # ax.errorbar(count, val.mean(), val.sem(), fmt="o", color="k")
        count += 1
        x_labels.append(dscrp)

    if hatch is not None:
        return
    ax.set_xticks(np.arange(1, len(x_labels) + 1))
    ax.set_xticklabels(x_labels)
    ax.set_ylim(bottom=50, top=100)
    ax.set_xlim(0, len(x_labels) + 1)
    ax.set_ylabel("Prev. Trial Correct (%)")
    if show_legend:
        ax.legend(loc="upper right", fontsize="xx-small")
    ax.set_title("Strategy by prev. trial outcome (Err bar: Subject)")
    [ax.spines[_dir].set_visible(False) for _dir in ["top", "right"]]


def stimulusTimeByPrevOutcome(df, col, is_normalized, save_figs,
                              save_prefix=None):
    if save_figs:
        assert save_prefix is not None

    df = df[df.PrevOutcomeCount.notnull()]
    df = df[df.PrevOutcomeCount != 0]
    is_time = "Time" in col
    if is_time:
        if is_normalized:
            assert col == "transformedCalcStimulusTime"
    df = df.copy()
    LIMIT_AT = 2
    # df.loc[df.PrevOutcomeCount > LIMIT_AT, "PrevOutcomeCount"] = LIMIT_AT
    # df.loc[df.PrevOutcomeCount < -LIMIT_AT, "PrevOutcomeCount"] = -LIMIT_AT
    df = df[(-LIMIT_AT <= df.PrevOutcomeCount) & (df.PrevOutcomeCount <= LIMIT_AT)]

    prev_outcome = sorted(df.PrevOutcomeCount.unique())
    print(prev_outcome)
    fig, ax = plt.subplots(figsize=(10, 5))
    count = 0
    x_ticks = []
    x_ticks_labels = []
    GROUPBY_COL = "Name"
    FILTER = False
    if FILTER:
        df_li = []
        num_prev_outcomes = len(prev_outcome)
        subjects_before = len(df.Name.unique())
        for subject, subject_df in df.groupby(GROUPBY_COL):
            filtered_df = subject_df.groupby("PrevOutcomeCount").filter(
                               lambda prev_outcome_df:len(prev_outcome_df) > 30)
            # See if we have all the prev_outcomes
            if len(filtered_df.PrevOutcomeCount.unique()) == num_prev_outcomes:
                df_li.append(filtered_df)
        df = pd.concat(df_li)
        subjects_after = len(df.Name.unique())
        print(f"Filtered subjects: {subjects_before} -> {subjects_after}")
    for prev_outcome_val in prev_outcome:
        prev_outcome_df = df[df.PrevOutcomeCount == prev_outcome_val]
        # ax.violinplot(prev_outcome_df[ST_COL], positions=[count],
        #               showmeans=True)
        # ax.boxplot(prev_outcome_df[ST_COL], positions=[count],
        #            showmeans=True, showfliers=False)
        num_trials = len(prev_outcome_df)
        num_sessions = len(
              prev_outcome_df[["Name", "Date", "SessionNum"]].drop_duplicates())
        num_subjects = len(prev_outcome_df.Name.unique())
        # st_mean = prev_outcome_df[col].mean()
        # st_sem = prev_outcome_df[col].sem()
        groupby = prev_outcome_df.groupby(GROUPBY_COL)
        st = groupby[col]
        st_mean = st.mean().mean()
        st_sem = st.mean().sem()
        color = 'r' if prev_outcome_val < 0 else 'g'
        # Plus-Minus sign: ±
        letter = 'I' if prev_outcome_val < 0 else 'C'
        label = (f"{letter}={abs(prev_outcome_val)} - RT: {st_mean:.2f}s ±{st_sem:.2f}s\n"
                 f"(N= {num_subjects} Subjects / {num_sessions} Sess / "
                 f"{num_trials:,} Trials)")
        # COL = "ChoiceCorrect"
        # COL = ST_COL
        ax.bar(count, st_mean, yerr=st_sem, label=label, color=color)
        x_ticks.append(count)
        x_ticks_labels.append(f"{abs(prev_outcome_val)}\nPrev. " +
                           ("Incorrect" if prev_outcome_val < 0 else "Correct"))
        count += 1

    for idx, (subject, subject_df) in enumerate(df.groupby(GROUPBY_COL)):
        st = subject_df.groupby("PrevOutcomeCount")[col]
        st_mean = st.mean()
        ax.plot(x_ticks, st_mean, color=f'C{idx}', alpha=0.1, marker='o')

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_labels)
    ax.set_xlabel("Previous Outcome")
    if is_time:
        ax.set_title("Stimulus Time by Prev. Outcome")
        ax.set_ylabel("Stimulus Time " +
                      ("(Z-Scored Normalized)" if is_normalized else "(s)"))
        if not is_normalized:
            # ax.set_ylim(1.0, 1.2)
            ax.set_ylim(1., 1.32)
            # ax.set_ylim(1.0, 1.25) # Hard
        # Draw legent a bit outside on the top right
        ax.legend(loc="upper right", fontsize="x-small",
                #   bbox_to_anchor=(1.1, 1.1)
                  )
    else:
        ax.set_title("Performance by Prev. Outcome")
        ax.set_ylabel("Performance")
        ax.legend(loc="upper left", fontsize="x-small")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
        ax.set_ylim(0.69, 0.8) # All and Easy
        # ax.set_ylim(0.76, 0.9) # Easy
        # ax.set_ylim(0.55, 0.65) # Hard

    ax.spines[["top", "right", "left"]].set_visible(False)
    if save_figs:
        if is_time:
            main_str = "stimulus_time"
            if is_normalized:
                main_str += "_norm"
        else:
            main_str = "performance"
        plt.savefig(f"{save_prefix}/{main_str}_by_prev_outcome.svg",
                    bbox_inches="tight")
    plt.show()

