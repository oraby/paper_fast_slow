import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize as ColorNormalize
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def staySwitchUpdate(df : pd.DataFrame, total_trials=None, df_query="",
                     dscrp="",  plot_single_subjects=True,
                     plot_combined_subjects=True,
                     save_prefix="",
                     save_fig=False):


    if plot_combined_subjects:
        _staySwitchUpdateSubject(df.copy(), is_many_subjects=True,
                                df_query=df_query,
                                dscrp=dscrp  + " - All subjects",
                                total_trials=total_trials,
                                save_prefix=save_prefix,
                                save_fig=save_fig)
    if not plot_single_subjects:
        return
    for subject, subject_df in df.groupby("Name"):
        if len(subject_df) < 500:
            print(f"Skipping Subject: {subject} - len: {len(subject_df)}")
            continue
        print(f"Subject: {subject} - len: {len(subject_df)}")
        try:
            _staySwitchUpdateSubject(subject_df, is_many_subjects=False,
                                     df_query=df_query,
                                     dscrp=dscrp + f" - {subject}",
                                     total_trials=total_trials,
                                     save_prefix=save_prefix,
                                     save_fig=save_fig)
        except Exception as e:
            print(f"Failed for {subject} - {e}")

def _staySwitchUpdateSubject(df : pd.DataFrame, is_many_subjects : bool,
                             total_trials=None, df_query="", dscrp="",
                             save_fig=False, save_prefix=""):
    if save_fig:
        assert save_prefix, "save_prefix must be provided when save_fig is True"

    len_before_filtering = len(df)
    # This is useful if we are operating only part of the data but we want to
    # compare the number to the total number of trials
    assert total_trials is None, "TODO: Pass total trial per subject"

    if len(df_query):
        sess_df_li = []
        for sess, sess_df in df.groupby(["Name", "Date", "SessionNum"]):
            sess_df = sess_df.query(df_query)
            sess_df_li.append(sess_df)
        df = pd.concat(sess_df_li)
    len_after = len(df)

    cols = ["Name",
            "PrevChoiceCorrect", "PrevDVstr",
            "DVstr", "ChoiceCorrect",
            "StayBaseline", "Stay"]
    df = df[cols].copy()
    df["PrevTrial"] = np.nan
    df.loc[df.PrevChoiceCorrect == True, "PrevTrial"] = "Rewarded"
    df.loc[df.PrevChoiceCorrect == False, "PrevTrial"] = "Not-Rewarded"
    df["CurDifficulty"] = df.DVstr
    df["PrevDifficulty"] = df.PrevDVstr
    df = df.drop(columns=["PrevChoiceCorrect", "DVstr", "PrevDVstr",
                          ])
    prev_trial_cols = "PrevTrial",
    cur_trial_cols = "CurDifficulty",

    df_li = []
    num_trials = len(df)
    if is_many_subjects:
        num_trials_subject = df.groupby("Name").size()
    else:
        num_trials_subject = None
    # A scalar key for a single column suppresses a groupby() warning
    for sub_index, sub_df in  df.groupby(list(prev_trial_cols)
                                         if len(prev_trial_cols) > 1 else
                                         prev_trial_cols[0]):
        prev_trial_val = sub_index
        assert isinstance(prev_trial_val, str), "Did you update the code here?"
        sub_df = sub_df.copy()
                                    # Again, suppress the warning
        indx_groups = sub_df.groupby(list(cur_trial_cols)
                                     if len(cur_trial_cols) > 1 else
                                     cur_trial_cols[0],  as_index=False)
        res_grp = indx_groups.apply(_calcGroupUpdate,
                                    is_many_subjects=is_many_subjects,
                                    prev_trial_val=prev_trial_val,
                                    num_trials_subject=num_trials_subject,
                                    num_trials=num_trials)
        total_row = _calcGroupUpdate(sub_df,
                                     is_many_subjects=is_many_subjects,
                                     prev_trial_val=prev_trial_val,
                                     num_trials_subject=num_trials_subject,
                                     num_trials=num_trials)
        assert len(cur_trial_cols) == 1, "Did you update the code here?"
        total_row[cur_trial_cols[0]] = "Total"
        res_grp = pd.concat([res_grp, total_row.to_frame().T])
        idx_groups_avgs = res_grp
        idx_groups_avgs["PrevTrial"] = prev_trial_val
        df_li.append(idx_groups_avgs)

    df_org = df
    df = pd.concat(df_li)

    df = df.pivot(columns=list(cur_trial_cols), index=list(prev_trial_cols))
    df_update = df[["StayUpdate"]].droplevel(level=0, axis="columns")
    df_nTrials_prcnt = df[["nTrialsPrnct"]].droplevel(level=0, axis="columns")
    df_performance = df[["Performance"]].droplevel(level=0, axis="columns")
    if is_many_subjects:
        df_update_sem = df[["StayUpdate_SEM"]].droplevel(level=0,
                                                         axis="columns")
        df_performance_sem = df[["Performance_SEM"]].droplevel(level=0,
                                                               axis="columns")
        df_nTrial_prcnt_sem = df[["nTrialsPrnct_SEM"]].droplevel(level=0,
                                                                 axis="columns")

    def sortDiff(diff):
        print("diff:", diff)
        cur_order = [
                "Not-Rewarded-Easy", "Not-Rewarded-Hard", "Not-Rewarded-Med",
                "Rewarded-Easy", "Rewarded-Hard", "Rewarded-Med"
                ]
        new_order = [
                "Rewarded-Hard", "Rewarded-Med", "Rewarded-Easy",
                "Not-Rewarded-Easy", "Not-Rewarded-Med", "Not-Rewarded-Hard",
                ]
        new_sort = [new_order.index(key) for key in cur_order]
        return new_sort
    df_update = df_update.iloc[::-1]

    def sortDiffols(diff):
        cur_order = ["Easy", "Hard", "Med", "Total"
                     ]
        new_order = ["Easy", "Med", "Hard", "Total"
                    ]
        new_sort = [new_order.index(key) for key in cur_order]
        return new_sort
    df_update = df_update.sort_index(axis='columns', key=sortDiffols, level=1)

    fig1, axs = plt.subplots(1, 2, figsize=(7, 6), width_ratios=[1, 0.05])
    fig1.subplots_adjust(wspace=0.15)
    ax_heatmap, cmap_ax = axs

    # sns.heatmap cannot draw circles, so the cells are drawn by hand below.


    x_cur_labels = df_update.columns.to_list()
    y_prev_labels = df_update.index.get_level_values(0).to_list()

    vmin, vmax = -20, 20
    x_cur_rng = np.arange(len(x_cur_labels))
    y_prev_rng = np.arange(len(y_prev_labels))
    x_cur, y_prev = np.meshgrid(x_cur_rng, y_prev_rng)
    updates = df_update.values.flatten()
    def getCorrespondigDF(other_df):
        other_df = other_df.copy()
        other_df = other_df[df_update.columns]
        other_df = other_df.reindex(df_update.index)
        return other_df.values.flatten()
    updates_perf = getCorrespondigDF(df_performance)
    updates_ntrials_prcnt = getCorrespondigDF(df_nTrials_prcnt)

    zip_li = [x_cur.flat, y_prev.flat, updates,
              updates_ntrials_prcnt, updates_perf]
    if is_many_subjects:
        updates_sem = getCorrespondigDF(df_update_sem)
        updates_perfs_sem = getCorrespondigDF(df_performance_sem)
        updates_ntrials_prcnt_sem = getCorrespondigDF(df_nTrial_prcnt_sem)
        zip_li.extend([updates_sem, updates_perfs_sem,
                       updates_ntrials_prcnt_sem])


    cmap = cm.ScalarMappable(norm=ColorNormalize(vmin, vmax), cmap="PuOr")
    for zip_iter in zip(*zip_li):
        i_cur, j_prev, update, n_trials_prcnt, perf, *sems = zip_iter
        if is_many_subjects:
            update_sem, perf_sem, n_trials_prcnt_sem = sems
        MAX_RADIUS = 0.5
        radius = MAX_RADIUS*n_trials_prcnt / updates_ntrials_prcnt.max()
        bg_clr = cmap.to_rgba(update)
        ax_heatmap.add_artist(plt.Rectangle((i_cur-0.5,j_prev-0.5),
                                            width=1, height=1,
                                            color=bg_clr,))

        j_offset = 0
        j_offset2 = 0.15
        # Contrast based on background color:
        # https://stackoverflow.com/a/3943023/11996983
        txt_clr = "black" if (bg_clr[0]*0.299 + bg_clr[1]*0.587 +
                              bg_clr[2]*0.114) > 0.7265 else "white"
        txt_2nd_clr = "#6b6b6b" if txt_clr == "black" else "#e0e0e0"
        font_size = max(10, radius*15)
        va = "center"
        update_str = f"{update:.2f}%"
        if is_many_subjects:
            update_str += f" ±{update_sem:.2f}%"
        ax_heatmap.text(i_cur, j_prev + j_offset, update_str, color=txt_clr,
                        fontsize=font_size, ha='center', va=va, zorder=10)
        trials_prcnt = n_trials_prcnt
        perf_str = f"{perf:.2f}%"
        trials_prcnt_str = f"{trials_prcnt:.2f}%"
        if is_many_subjects:
            perf_str += f"\n±{perf_sem:.2f}%"
            trials_prcnt_str += f"\n±{n_trials_prcnt_sem:.2f}%"
        ax_heatmap.text(i_cur, j_prev + j_offset2,
                        f"Perf={perf_str}\n"
                        f"Trials={trials_prcnt_str}",
                        color=txt_2nd_clr, fontsize=9, ha='center', va='top',
                        zorder=10)



    MAX_RADIUS += (0.003*2*np.pi) # margin around the cells
    ax_heatmap.tick_params(axis="both", which="both", length=0, pad=15)
    ax_heatmap.xaxis.tick_top()
    ax_heatmap.xaxis.set_label_position('top')
    ax_heatmap.set_xlabel("Current Trial")
    ax_heatmap.set_xticks(x_cur_rng)
    ax_heatmap.set_xticklabels(x_cur_labels, minor=False)
    ax_heatmap.set_xlim(-MAX_RADIUS, len(x_cur_labels) - MAX_RADIUS)
    ax_heatmap.set_ylabel("Previous Trial")
    ax_heatmap.set_yticks(y_prev_rng, minor=False)
    ax_heatmap.set_yticklabels(y_prev_labels)
    ax_heatmap.set_ylim(len(y_prev_labels) - MAX_RADIUS, -MAX_RADIUS)


    if len(dscrp) or len(df_query):
        title_dscrp = dscrp
        if len(title_dscrp) and len (df_query):
            title_dscrp = f"{title_dscrp}  - where: "
        if is_many_subjects:
            num_subjects = df_org.Name.nunique()
            count_str = (f"n={num_subjects:,} Subjects - {num_trials:,} Trials"
                           " - Err bars: Subjects SEM")
        else:
            count_str = f"n={num_trials:,} Trials"
        ax_heatmap.set_title(f"{title_dscrp}{df_query}"
                             f"\n{count_str}", pad=20)

    plt.colorbar(cmap,  cax=cmap_ax, orientation="vertical")
    cmap_ax.yaxis.set_label_position("right")
    cmap_ax.set_ylabel("Update", rotation=270, labelpad=-50, fontsize=12)
    cmap_ax.yaxis.tick_right()
    cmap_ax.set_yticks([vmin, vmax])
    cmap_ax.set_yticklabels([f"{abs(vmin)}% Switch", f"{vmax}% Stay"])

    [ax.spines[_dir].set_visible(False)
     for _dir in ["left", "top", "right", "bottom"]
     for ax in [ax_heatmap, cmap_ax]]

    if save_fig:
        if len(dscrp) and len(df_query):
            dscrp = f"{dscrp}_" + "".join((c if c.isalnum() else '_')
                                          for c in df_query)
        save_fp = f"{save_prefix}/StaySwitch/{dscrp}{df_query}_switch.svg"
        save_fp = save_fp.replace('=', '_')
        save_fp = Path(save_fp)
        save_fp.parent.mkdir(exist_ok=True)
        fig1.savefig(save_fp, dpi=300, bbox_inches='tight')
    plt.show()


def _calcGroupUpdate(grp_df, is_many_subjects, prev_trial_val,
                     num_trials_subject=None, num_trials=None):

    if hasattr(grp_df, "name"):
        name = grp_df.name
    else:
        name = "All"
    grp_df = grp_df.copy()
    grp_df = grp_df[grp_df.Stay.notnull()]

    more_cols = ["StayUpdate", "StayMean", "RefStayMean", "Performance",
                 "nTrials", "nTrialsPrnct"]
    if is_many_subjects:
        assert num_trials_subject is not None, (
                        "num_trials_subject must be provided for many subjects")
        more_cols_n_subj = more_cols + ["Name"]

        # An explicit loop rather than groupby(...).apply: _calcUpdate both
        # reads and returns "Name", so the grouping column cannot be excluded
        # from it (which is what pandas 3 does by default). Same values, same
        # order -- groupby sorts by key either way.
        subject_updates = [_calcUpdate(subject_df)
                           for _, subject_df in grp_df.groupby("Name")]
        local_dict = {}
        for col in more_cols_n_subj:
            if col == "nTrialsPrnct": # We will calculate it later
                continue
            local_dict[col] = [subject_vals[col]
                               for subject_vals in subject_updates]
        subject_grps_df = pd.DataFrame(local_dict)
        subject_grps_df["nTrialsPrnct"] = subject_grps_df.apply(
            lambda row: 100*row["nTrials"]/num_trials_subject[row.Name],
            axis=1)

        res_dict = {}
        for col in more_cols:
            vals = subject_grps_df[col]
            avg = vals.mean()
            sem = vals.sem()
            res_dict[col] = avg
            res_dict[f"{col}_SEM"] = sem
    else:
        assert num_trials is not None, (
                               "num_trials must be provided for single subject")
        res_dict = _calcUpdate(grp_df)
        res_dict["nTrialsPrnct"] = 100*len(grp_df)/num_trials
    for col in more_cols:
        grp_df[col] = res_dict[col]
        if is_many_subjects:
            grp_df[f"{col}_SEM"] = res_dict[f"{col}_SEM"]
        assert grp_df[col].nunique() == 1
    grp_df["Description"] = name
    return grp_df.iloc[0] # return first row as all rows are the same metrics


def _calcUpdate(grp_df):
    stay_update = 100*(grp_df.Stay.sum() -
                       grp_df.StayBaseline.sum())/len(grp_df)
    assert grp_df.Name.nunique() == 1
    stay_mean = 100*grp_df.Stay.mean()
    ref_stay_mean = 100*grp_df.StayBaseline.mean()
    performance = 100*grp_df.ChoiceCorrect.mean()
    n_trials = len(grp_df)
    return {"StayUpdate": stay_update,
            "StayMean": stay_mean,
            "RefStayMean": ref_stay_mean,
            "Performance": performance,
            "nTrials": n_trials,
            "Name": grp_df.Name.iloc[0] if grp_df.Name.nunique() == 1 else "Many",
            }
