from .util import normalizeSTAcrossSubjects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp
import statsmodels.api as sm
from statsmodels.formula.api import ols
from typing import Literal

def stHeatmap(df : pd.DataFrame,
              mean_or_median : Literal["mean", "median"],
              dscrp : str, df_query="",
              plot_combined_subjects=True,
              plot_single_subjects=True,
              save_prefix="", save_fig=False):
    df = df[df.calcStimulusTime.notnull()]
    df_org = df.copy()

    if plot_combined_subjects:
        # TODO: Make the function handle the transformedCalcStimulusTime
        df = normalizeSTAcrossSubjects(df)
        df["calcStimulusTime"] = df["transformedCalcStimulusTime"]
        _gropuDataByPriorAndCurrentSubject(df, is_many_subjects=True,
                                        df_query=df_query,
                                        dscrp=dscrp + " All Subjects",
                                        mean_or_median=mean_or_median,
                                        append_sec_str=False,
                                        save_fig=save_fig,
                                        save_prefix=save_prefix)
        plt.show()

    if not plot_single_subjects:
        return
    for subject_name, subject_df in df_org.groupby("Name"):
        print(subject_name)
        _gropuDataByPriorAndCurrentSubject(subject_df, is_many_subjects=False,
                                           df_query=df_query,
                                           dscrp=dscrp + f" - {subject_name}",
                                           mean_or_median=mean_or_median,
                                           save_fig=save_fig,
                                           save_prefix=save_prefix)
        plt.show()

#: Heatmap row order: previous trial rewarded first.
PREV_TRIAL_ORDER = ("Rewarded", "Not-Rewarded")
#: Heatmap column order: easiest current trial first.
CUR_DIFFICULTY_ORDER = ("Easy", "Med", "Hard")


def _inOrder(labels, order):
    """The labels of ``order`` that are present in ``labels``, in that order."""
    present = set(labels)
    return [label for label in order if label in present]


def _asTuple(group_key):
    """Normalise a groupby key to a tuple.

    ``df.groupby(["one_col"])`` yields a 1-tuple in pandas >= 2 and yielded the
    bare value before. Both grouping keys here are single-element tuples in the
    shipped configuration, so the callers index them positionally and rely on
    this.
    """
    return group_key if isinstance(group_key, tuple) else (group_key,)


def _gropuDataByPriorAndCurrentSubject(df : pd.DataFrame,
                            is_many_subjects : bool,
                            mean_or_median : Literal["mean", "median"],
                            dscrp : str, df_query="", append_sec_str=True,
                            save_fig=False, save_prefix=""):
    assert mean_or_median in ["mean", "median"]
    if save_fig:
        assert len(save_prefix), (
                          "save_prefix must be specified when save_fig is True")
    if len(df_query):
        len_before = len(df)
        df = df.query(df_query)
        len_after = len(df)
        print(f"Len before: {len_before:,} - after: {len_after:,}")

    df = df[df.ChoiceCorrect.notnull()]
    df = df[df.Stay.notnull()]
    df = df[df.PrevChoiceCorrect.notnull()]
    # Filter out medium trials
    df_original = df.copy()
    TWO_DIFFICULTIES = False
    if TWO_DIFFICULTIES:
        df = df[df.DVstr.isin(["Easy", "Hard"])]
        df = df[df.PrevDVstr.isin(["Easy", "Hard"])]

    cols = ["PrevChoiceCorrect", "PrevDVstr", "Stay", "DVstr",
            "calcStimulusTime",
            "Name", "Date", "SessionNum"]
    df = df[cols]
    # df["Strategy"] = np.nan
    # df.loc[df.Stay == True, "Strategy"] = "Stay"
    # df.loc[df.Stay == False, "Strategy"] = "Switch"
    # Object dtype up front: starting from a float NaN column and assigning
    # strings into it is deprecated (pandas will raise on the silent upcast).
    # The resulting column is the same object column either way.
    df["PrevTrial"] = pd.Series(np.nan, index=df.index, dtype=object)
    df.loc[df.PrevChoiceCorrect == True, "PrevTrial"] = "Rewarded"
    df.loc[df.PrevChoiceCorrect == False, "PrevTrial"] = "Not-Rewarded"
    df["CurDifficulty"] = df.DVstr
    df["PrevDifficulty"] = df.PrevDVstr
    df = df.drop(columns=["Stay", "PrevChoiceCorrect", "DVstr", "PrevDVstr"])
    PREV_DIFFICULTY = False
    prev_trial = "PrevTrial",
    if PREV_DIFFICULTY:
        prev_trial =  tuple(list(prev_trial) + ["PrevDifficulty"])
    cur_trial = "CurDifficulty", #"Strategy",

    # Create the heatmap
    fig1 = _stHeatmap(df.copy(), is_many_subjects=is_many_subjects,
                      prev_trial=prev_trial, cur_trial=cur_trial,
                      mean_or_median=mean_or_median, dscrp=dscrp,
                      df_query=df_query, append_sec_str=append_sec_str,
                      PREV_DIFFICULTY=PREV_DIFFICULTY)
    if save_fig:
        if len(dscrp) and len(df_query):
            dscrp = f"{dscrp}_"
        save_fp = \
               f"{save_prefix}/prior_cur_{dscrp}{df_query}_{mean_or_median}.svg"
        save_fp = save_fp.replace('=', '_')
        save_fp = Path(save_fp)
        save_fp.parent.mkdir(exist_ok=True)
        fig1.savefig(save_fp, dpi=300, bbox_inches='tight')


    if is_many_subjects:
        fig2, ax_violin = plt.subplots(1, figsize=(20, 8))
        _violinPlot(df.copy(), ax_violin)
        fig2.tight_layout()
        if save_fig:
            save_fp = f"{save_prefix}/prior_cur_violin_{dscrp}{df_query}.svg"
            save_fp = save_fp.replace('=', '_')
            save_fp = Path(save_fp)
            save_fp.parent.mkdir(exist_ok=True)
            fig2.savefig(save_fp, dpi=300, bbox_inches='tight')
        plt.show()

def _stHeatmap(df, is_many_subjects, prev_trial, cur_trial, mean_or_median,
               dscrp, df_query, append_sec_str, PREV_DIFFICULTY):
    df_li = []
    num_trials = len(df)
    num_sessions = len(df[["Name", "Date", "SessionNum"]].drop_duplicates())
    num_subjects_str = df.Name.nunique()
    if num_subjects_str > 1:
        num_subjects_str = f"{num_subjects_str} Subjects / "
    else:
        num_subjects_str = ""

    for sub_index, sub_df in  df.groupby(list(prev_trial)):
        # pandas >= 2 hands back a 1-tuple when grouping by a single-element
        # list, where it used to hand back the bare value. Both `prev_trial`
        # and `cur_trial` are single-element tuples unless PREV_DIFFICULTY is
        # on, so normalise the key rather than depending on the pandas
        # version. Without this the panel raises
        # "Length of values (1) does not match length of index (3)".
        sub_index = _asTuple(sub_index)
        sub_df = sub_df.copy()
        indx_groups = sub_df.groupby(list(cur_trial), as_index=False)
        if is_many_subjects:
            # It's a group of subjects)
            avg_dict = {k:[] for k in cur_trial}
            avg_dict["calcStimulusTime"] = []
            avg_dict["calcStimulusTimeSEM"] = []
            for grp_name, grp_st in indx_groups:
                grp_name = _asTuple(grp_name)
                vals = grp_st.groupby("Name").calcStimulusTime
                # avg_dict[grp_name] = pd.Series(vals, name=grp_name)
                avg = vals.mean() if mean_or_median == "mean" else vals.median()
                avg_dict["calcStimulusTime"].extend(avg)
                avg_dict["calcStimulusTimeSEM"].extend(vals.sem())
                for position, k in enumerate(cur_trial):
                    avg_dict[k].extend([grp_name[position]]*len(vals))
            indx_groups_st = pd.DataFrame(avg_dict).set_index(list(cur_trial))
            indx_groups_st = indx_groups_st.reset_index()
            # display(indx_groups_st.head())
            # print("****************")
            # indx_groups_st = indx_groups_st.groupby(indx_groups_st.index)
            # for grp_name, grp_st in indx_groups_st:
            #     print("grp_name:", grp_name)
            #     print("grp_st:", grp_st)
            # display(indx_groups_st)
            indx_groups_st = indx_groups_st.groupby(list(cur_trial),
                                                    as_index=False)
        else:
            indx_groups_st = indx_groups.calcStimulusTime
            # for grp_name, grp_st in indx_groups_st:
            #     print("grp_name:", grp_name)
            #     print("grp_st:", grp_st)
            # display(indx_groups_st)
        if mean_or_median == "mean":
            idx_groups_avgs = indx_groups_st.mean()
            # print("**idx_groups_avgs:", idx_groups_avgs)
        else:
            if not is_many_subjects:
                idx_groups_avgs = indx_groups_st.median()
            else:
                # We already calculated the median per subject above
                idx_groups_avgs = indx_groups_st.mean()
        if PREV_DIFFICULTY:
            was_rewarded = sub_index[0]
            prev_diff = sub_index[1]
            # Treat as data-frame
            idx_groups_avgs["PrevTrial"] = was_rewarded
            idx_groups_avgs["PrevDifficulty"] = prev_diff
        else:
            idx_groups_avgs["PrevTrial"] = sub_index[0]
        df_li.append(idx_groups_avgs)

    df_summary =  pd.concat(df_li)
    # display(df)
    # df = df.set_index(index)
    df_summary = df_summary.pivot(columns=list(cur_trial),
                                  index=list(prev_trial))
    # display(df_summary)
    if is_many_subjects:
        df_summary_sem = df_summary.copy()
        df_summary = df_summary.drop(columns="calcStimulusTimeSEM")
        df_summary_sem = df_summary_sem.drop(columns="calcStimulusTime")
        # display(df_summary_sem)

    # Fixed, published order. These used to be `sort_index(key=...)` callbacks
    # that ignored the index they were handed and returned a hardcoded
    # permutation of the alphabetical pivot order, which only worked while the
    # index held exactly the expected labels. Selecting by label gives the same
    # order and also survives a `df_query` that leaves out a level.
    df_summary = df_summary.droplevel(level=0, axis="columns")
    df_summary = df_summary.loc[_inOrder(df_summary.index, PREV_TRIAL_ORDER),
                                _inOrder(df_summary.columns,
                                         CUR_DIFFICULTY_ORDER)]
    if is_many_subjects:
        df_summary_sem = df_summary_sem.droplevel(level=0, axis="columns")
        df_summary_sem = df_summary_sem.loc[df_summary.index,
                                            df_summary.columns]
    min_val, max_val = df_summary.values.min(), df_summary.values.max()

    print("df.min():", min_val)
    def format(val, pos):
        if abs(val - min_val) < 0.000001:
            return "Fast"
        else:
            return "Slow"

    labels = np.array([[f"{cell:.2f}" + ('s' if append_sec_str else "")
                        for cell in row] for row in  df_summary.values])
    if is_many_subjects:
        labels_sem = np.array([[f"±{cell:.2f}" + ('s' if append_sec_str else "")
                            for cell in row] for row in  df_summary_sem.values])
        labels_li = [f"{val} {sem}" for val, sem in zip(labels.flatten(),
                                                        labels_sem.flatten())]
        labels = np.array(labels_li).reshape(labels.shape)

    fig, axs = plt.subplots(1, 2, figsize=(7, 6), width_ratios=[1, 0.05])
    fig.subplots_adjust(wspace=0.1)
    ax_heatmap, cmap_ax = axs
    sns.heatmap(df_summary,
                #  cmap="YlOrRd_r",
                cmap="autumn",
                # cmap="viridis",
                annot=labels,
                fmt="",
                cbar_kws=dict(ticks=[min_val, max_val], format=format),
                ax=ax_heatmap,
                cbar_ax=cmap_ax,
                )
    ax_heatmap.xaxis.set_label_position('top')
    ax_heatmap.xaxis.tick_top()
    ax_heatmap.xaxis.set_tick_params(rotation=45)
    # cbar = ax_heatmap.collections[0].colobar
    # cbar.set_ticks([df.min(),df.max()])
    # cbar.set_ticklabels(["Slow", "Fast"])
    if len(dscrp) or len(df_query):
        title_dscrp = dscrp
        if len(title_dscrp) and len (df_query):
            title_dscrp = f"{title_dscrp}  - where: "
        title_post = " (Subject SEM)" if is_many_subjects else ""
        ax_heatmap.set_title(f"{title_dscrp}{df_query} ({mean_or_median} value)"
                             f"\nn={num_subjects_str}{num_sessions:,} Sessions "
                             f"/ {num_trials:,} Trials{title_post}")
    cmap_ax.invert_yaxis()

    ax_heatmap.set_xlabel("Current Trial")
    ax_heatmap.set_ylabel("Previous Trial")

    return fig



def _violinPlot(df, ax):
    num_subjects = len(df.Name.unique())
    df = df.copy()
    df["StimulusTime"] = df.calcStimulusTime

    # keep a copy of the original (trial-level) data for the 3-way ANOVA
    orig_df = df.copy()

    # ---- log transform (shared for everything) ----
    eps = 1e-6
    min_stim = orig_df["StimulusTime"].min()
    shift = -min_stim + eps if min_stim <= 0 else 0.0  # ensure positivity
    orig_df["StimulusTime_log"] = np.log(orig_df["StimulusTime"] + shift)
    df["StimulusTime_log"] = np.log(df["StimulusTime"] + shift)

    # ---- keep your slicing as-is ----
    df_win = df[df.PrevTrial == "Rewarded"]
    df_win_hard = df_win[df_win.CurDifficulty == "Hard"]
    df_win_easy = df_win[df_win.CurDifficulty == "Easy"]
    df_lose = df[df.PrevTrial == "Not-Rewarded"]
    df_lose_hard = df_lose[df_lose.CurDifficulty == "Hard"]
    df_lose_easy = df_lose[df_lose.CurDifficulty == "Easy"]

    data_li = [
        df_lose_hard.query("PrevDifficulty == 'Easy'"),
        df_lose_hard.query("PrevDifficulty == 'Hard'"),
        df_lose_easy.query("PrevDifficulty == 'Easy'"),
        df_lose_easy.query("PrevDifficulty == 'Hard'"),
        df_win_hard.query("PrevDifficulty == 'Easy'"),
        df_win_hard.query("PrevDifficulty == 'Hard'"),
        df_win_easy.query("PrevDifficulty == 'Easy'"),
        df_win_easy.query("PrevDifficulty == 'Hard'"),
    ]

    # ---- per-cell subject means + normality (raw vs log) ----
    data_li_ = []
    all_normal = True  # this will refer to the log-transformed data
    for _df in data_li:
        _df = _df.copy()
        _df = _df.groupby("Name", as_index=False).agg({
            "StimulusTime": "mean",
            "StimulusTime_log": "mean",
            "Name": "count",
            "PrevTrial": "first",
            "CurDifficulty": "first",
            "PrevDifficulty": "first",
        })
        _df["n_trials"] = _df["Name"]
        _df = _df.drop(columns="Name")
        _df["n_trials_sem"] = _df["n_trials"].sem()
        _df["n_trials"] = _df["n_trials"].mean()
        data_li_.append(_df)

        normality_raw = stats.shapiro(_df.StimulusTime)
        normality_log = stats.shapiro(_df.StimulusTime_log)

        # print("Normality (raw) for", _df.PrevTrial.unique()[0],
        #       "- Prev", _df.PrevDifficulty.unique()[0],
        #       "- Cur:", _df.CurDifficulty.unique()[0],
        #       ":", normality_raw)
        # print("Normality (log) for", _df.PrevTrial.unique()[0],
        #       "- Prev", _df.PrevDifficulty.unique()[0],
        #       "- Cur:", _df.CurDifficulty.unique()[0],
        #       ":", normality_log)

        # decide ANOVA vs Kruskal based on log-normality
        if normality_log.pvalue < 0.05:
            all_normal = False

    data_li = data_li_
    del data_li_

    # separate raw vs log data for stats
    stats_data_raw = [dt.StimulusTime for dt in data_li]
    stats_data_log = [dt.StimulusTime_log for dt in data_li]

    # ---- plotting (unchanged except that df now also has StimulusTime_log) ----
    df_plot = pd.concat(data_li)  # avoid overwriting orig_df
    df_plot["Comb"] = ("Prev. " + df_plot.PrevTrial.astype(str) +
                       "\nCur. " + df_plot.CurDifficulty.astype(str) +
                       "\nPrev." + df_plot.PrevDifficulty.astype(str) +
                       "\nTrials n=" + df_plot.apply(
                           lambda row: f"{row.n_trials:,} ±{row.n_trials_sem:.2f}",
                           axis=1)
                       )
    org_order = df_plot["Comb"].unique()
    df_pivot = df_plot[["Comb", "StimulusTime"]].pivot(columns="Comb",
                                                       values="StimulusTime")
    df_pivot = df_pivot[org_order]

    sns.violinplot(data=df_pivot, ax=ax, density_norm="count", common_norm=True)

    ax.set_ylim(df_pivot.min().min(), df_pivot.max().max())
    y_ticks, y_tick_labels = ax.get_yticks(), ax.get_yticklabels()
    max_y = df_pivot.max().max() + 0.5

    # ---- omnibus test: ANOVA on log data if normal, else Kruskal on raw ----
    if all_normal:
        STtest = stats.f_oneway
        statistics, p_val = STtest(*stats_data_log)
        stats_data_for_posthoc = stats_data_log
        transform_label = "log-transformed"
    else:
        STtest = stats.kruskal
        statistics, p_val = STtest(*stats_data_raw)
        stats_data_for_posthoc = stats_data_raw
        transform_label = "raw"

    # print(f"{STtest.__name__} ({transform_label} data) statistics:",
    #       statistics, "p_val:", p_val)

    # ---- post-hoc tests (kept as close as possible to your original) ----
    if STtest.__name__ == "f_oneway":
        p_adjust = "bonferroni"
        corr_res = sp.posthoc_tukey(stats_data_for_posthoc)
        post_hoc_str = "Tukey's test (on log data)"
    elif STtest.__name__ == "kruskal":
        p_adjust = "holm"
        corr_res = sp.posthoc_dunn(stats_data_for_posthoc, p_adjust)
        post_hoc_str = f"Dunn's test with {p_adjust} correction"

    corr_res_named = corr_res.copy()
    corr_res_named.columns = org_order
    corr_res_named.index = org_order
    # if "display" in globals():
    #     print("With correction:")
    #     display(corr_res_named <= 0.05)
    # if "summary_data" in locals():
    #     similar_df = summary_data == (corr_res_named <= 0.05)
    #     if "display" in globals():
    #         display(similar_df)
    #     print("All similar?:", similar_df.all().all())

    # Print every value pair first
    print("\t", end="")
    for i, (_, row) in enumerate(corr_res.iterrows()):
        dscrp = ", ".join(org_order[i].split('\n')[:3])
        print(dscrp, end="\t")
    print()
    for i, (_, row) in enumerate(corr_res.iterrows()):
        dscrp = ", ".join(org_order[i].split('\n')[:3])
        print(dscrp, end="\t")
        for j, val in enumerate(row):
            print(f"{val:.5f}", end="\t")
        print()

    for i, (_, row) in enumerate(corr_res.iterrows()):
        j_start = i + 1  # skip the mirrored part
        for j, val in enumerate(row[j_start:], j_start):
            if val <= 0.05:
                # print(f"({i+1}, {j+1}) is significant")
                star = "***" if val <= 0.001 else "**" if val <= 0.01 else "*"
                ax.plot([i, i, j, j],
                        [max_y, max_y+0.05, max_y+0.05, max_y],
                        lw=.5, c='k')
                ax.text((i+j)/2, max_y+0.04, star,
                        ha='center', va='bottom', c='k')
                max_y += 0.125

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_ylim(ax.get_ylim()[0] - .5, max_y)
    ax.set_ylabel("Stimulus Time (z-score)")
    ax.set_xlabel("")
    [ax.spines[_dir].set_visible(False) for _dir in ["left", "top", "right"]]

    ax.set_title("Stimulus Time Distribution (Error bar: "
                 f"Group Subjects' (n={num_subjects}) SEM)\n"
                 f"{STtest.__name__} ({transform_label} data) p-value: {p_val:.5f} - "
                 f"Pairwise comparison: {post_hoc_str}")

    # ------------------------------------------------------------------
    # 3-way factorial ANOVA (trial-level, log-transformed) for reporting
    # ------------------------------------------------------------------
    # print("\n=== 3-way factorial ANOVA on log-transformed StimulusTime ===")
    # model = ols("StimulusTime_log ~ C(PrevTrial)*C(CurDifficulty)*C(PrevDifficulty)",
    #              data=orig_df).fit()
    # anova_table = sm.stats.anova_lm(model, typ=2)
    # print(anova_table)

    # resid_norm = stats.shapiro(model.resid)
    # print("Shapiro normality test on residuals (log scale):", resid_norm)

def _violinPlot2(df, ax):
    num_subjects = len(df.Name.unique())
    df["StimulusTime"] = df.calcStimulusTime
    # display(df)
    df_win = df[df.PrevTrial == "Rewarded"]
    df_win_hard = df_win[df_win.CurDifficulty == "Hard"]
    df_win_easy = df_win[df_win.CurDifficulty == "Easy"]
    df_lose = df[df.PrevTrial == "Not-Rewarded"]
    df_lose_hard = df_lose[df_lose.CurDifficulty == "Hard"]
    df_lose_easy = df_lose[df_lose.CurDifficulty == "Easy"]
    # def sortWin(df):
    #     return df.sort_values(by=["Strategy", "PrevDifficulty"],
    #                           ascending=[False, False])
    # def sortLose(df):
    #     return df.sort_values(by=["Strategy", "PrevDifficulty"],
    #                           ascending=[True, False])
    # df = pd.concat([sortLose(df_lose_hard), sortLose(df_lose_easy),
    #                 sortWin(df_win_hard), sortWin(df_win_easy)])

    # Decide on how to break down the data
    # data_li = [
    #            df_lose_hard.query("Strategy == 'Stay'"),
    #            df_lose_hard.query("Strategy == 'Switch'"),
    #            df_lose_easy.query("Strategy == 'Stay'"),
    #            df_lose_easy.query("Strategy == 'Switch'"),
    #            df_win_hard.query("Strategy == 'Switch'"),
    #            df_win_hard.query("Strategy == 'Stay'"),
    #            df_win_easy.query("Strategy == 'Switch'"),
    #            df_win_easy.query("Strategy == 'Stay'"),
    #            ]
    data_li = [
               df_lose_hard.query("PrevDifficulty == 'Easy'"),
               df_lose_hard.query("PrevDifficulty == 'Hard'"),
               df_lose_easy.query("PrevDifficulty == 'Easy'"),
               df_lose_easy.query("PrevDifficulty == 'Hard'"),
               df_win_hard.query("PrevDifficulty == 'Easy'"),
               df_win_hard.query("PrevDifficulty == 'Hard'"),
               df_win_easy.query("PrevDifficulty == 'Easy'"),
               df_win_easy.query("PrevDifficulty == 'Hard'"),
               ]
    data_li_ = []
    all_normal = True
    for _df in data_li:
        _df = _df.copy()
        _df = _df.groupby("Name", as_index=False)
        # _df["n_trials"] = len(_df)
        _df = _df.agg({"StimulusTime":"mean", "Name":"count",
                       "PrevTrial":"first", "CurDifficulty":"first",
                       "PrevDifficulty":"first"})
        _df["n_trials"] = _df["Name"]
        _df = _df.drop(columns="Name")
        _df["n_trials_sem"] = _df["n_trials"].sem()
        _df["n_trials"] = _df["n_trials"].mean()
        data_li_.append(_df)
        normality_res = stats.shapiro(_df.StimulusTime)
        print("Normality test for", _df.PrevTrial.unique()[0],
              "- Prev", _df.PrevDifficulty.unique()[0],
              "- Cur:", _df.CurDifficulty.unique()[0],
              ":", normality_res)
        if normality_res.pvalue < 0.05:
            all_normal = False
    data_li = data_li_
    del data_li_
    stats_data = [dt.StimulusTime for dt in data_li]

    df = pd.concat(data_li)
    # display(df)

    df["Comb"] = ("Prev. " + df.PrevTrial.astype(str) +
                  "\nCur. " + df.CurDifficulty.astype(str) +
                  #"\n" + df.Strategy.astype(str) +
                  "\nPrev." + df.PrevDifficulty.astype(str) +
                  "\nTrials n=" + df.apply(
                       lambda row: f"{row.n_trials:,} ±{row.n_trials_sem:.2f}",
                       axis=1)
                 )
    org_order = df["Comb"].unique()
    complete_df = df.copy()
    df = df[["Comb", "StimulusTime"]]
    # df = df.set_index("Comb")#.transpose()
    # df = df.melt(id_vars="Comb", var_name="StimulusTime")
    df = df.pivot(columns="Comb", values="StimulusTime")
    # df = df.sample(10)
    df = df[org_order]
    # print("df now:")
    # display(df)
    sns.violinplot(data=df, ax=ax, #orient='v', inner=None, #linewidth=0.5
                   #x="Comb", y="StimulusTime", # #order=df.index.unique()
                   density_norm="count", common_norm=True,
                   )


    ax.set_ylim(df.min().min(), df.max().max())
    y_ticks, y_tick_labels = ax.get_yticks(), ax.get_yticklabels()
    max_y = df.max().max() + 0.5

    STtest = stats.f_oneway if all_normal else stats.kruskal
    statistics, p_val = STtest(*stats_data)
    print(STtest.__name__, "statistics:", statistics, "p_val:", p_val)
    if STtest.__name__ == "f_oneway":
        p_adjust = "bonferroni"
        corr_res = sp.posthoc_tukey(stats_data)
        post_hoc_str = "Tukey's test"
    elif STtest.__name__ == "kruskal":
        p_adjust = "holm"
        corr_res = sp.posthoc_dunn(stats_data, p_adjust)
        post_hoc_str = f"Dunn's test with {p_adjust} correction"

    corr_res_named = corr_res.copy()
    corr_res_named.columns = org_order
    corr_res_named.index = org_order
    if "display" in globals():
        print("With correction:")
        display(corr_res_named <= 0.05)
    if "summary_data" in locals():
        similar_df = summary_data == (corr_res_named <= 0.05)
        if "display" in globals():
            display(similar_df)
        print("All similar?:", similar_df.all().all())

    for i, (_, row) in enumerate(corr_res.iterrows()):
        j_start = i + 1 # skip the mirrored part
        for j, val in enumerate(row[j_start:], j_start):
            if val <= 0.05:
                print(f"({i+1}, {j+1}) is significant")
                star = "***" if val <= 0.001 else "**" if val <= 0.01 else "*"
                ax.plot([i, i, j, j], [max_y, max_y+0.05, max_y+0.05, max_y],
                        lw=.5, c='k')
                ax.text((i+j)/2, max_y+0.04, star, ha='center', va='bottom',
                        c='k')
                max_y += 0.125

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_ylim(ax.get_ylim()[0] - .5, max_y)
    ax.set_ylabel("Stimulus Time (s)")
    ax.set_xlabel("")
    [ax.spines[_dir].set_visible(False) for _dir in ["left", "top", "right"]]
    ax.set_title("Stimulus Time Distribution (Error bar: "
                 f"Group Subjects' (n={num_subjects}) SEM)\n"
                 f"{STtest.__name__} p-value: {p_val:.5f} - "
                 f"Pairwise comparison: {post_hoc_str}")
