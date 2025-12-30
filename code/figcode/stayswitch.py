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
    # if total_trials is None:
    #     total_trials = len(df)

    # grand_baseline = df.StayBaseline.mean()
    # print("grand_baseline:", 100*grand_baseline)
    if len(df_query):
        sess_df_li = []
        for sess, sess_df in df.groupby(["Name", "Date", "SessionNum"]):
            sess_df = sess_df.query(df_query)
            sess_df_li.append(sess_df)
        df = pd.concat(sess_df_li)
    len_after = len(df)
    # print(f"Len before: {len_before_filtering:,} - after: {len_after:,}")

    cols = ["Name", #"Date", "SessionNum",
            "PrevChoiceCorrect", "PrevDVstr",
            "DVstr", "ChoiceCorrect",
            "StayBaseline", "Stay"]
    df = df[cols].copy()
    df["PrevTrial"] = np.nan
    # df.loc[df.ChoiceCorrect == True, "CurOutcome"] = "Correct"
    # df.loc[df.ChoiceCorrect == False, "CurOutcome"] = "Incorrect"
    df.loc[df.PrevChoiceCorrect == True, "PrevTrial"] = "Rewarded"
    df.loc[df.PrevChoiceCorrect == False, "PrevTrial"] = "Not-Rewarded"
    df["CurDifficulty"] = df.DVstr
    df["PrevDifficulty"] = df.PrevDVstr
    df = df.drop(columns=["PrevChoiceCorrect", "DVstr", "PrevDVstr",
                          #"ChoiceCorrect"
                          ])
    prev_trial_cols = "PrevTrial", #"PrevDifficulty"
    cur_trial_cols = "CurDifficulty",

    df_li = []
    num_trials = len(df)
    if is_many_subjects:
        num_trials_subject = df.groupby("Name").size()
    else:
        num_trials_subject = None
    # display(df)                        # Supress groupby() wawrning
    for sub_index, sub_df in  df.groupby(list(prev_trial_cols)
                                         if len(prev_trial_cols) > 1 else
                                         prev_trial_cols[0]):
        # print("sub_index:", sub_index, "- len:", len(sub_df))
        prev_trial_val = sub_index
        assert isinstance(prev_trial_val, str), "Did you update the code here?"
        sub_df = sub_df.copy()
        # print("Columns: ", sub_df.columns)
                                    # Again, supress warning
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
        # display(res_grp)
        # res_grp = res_grp.rename_axis("StayUpdate")
        # display(indx_groups.apply(lambda grp_df:len(grp_df)))
        # display(indx_groups.Stay.mean())
        # display(indx_groups.StayBaseline.mean())
        idx_groups_avgs = res_grp
        # was_rewarded = sub_index[0]
        # prev_diff = sub_index[1]
        # # Treat as data-frame
        # idx_groups_avgs["PrevTrial"] = was_rewarded
        # idx_groups_avgs["PrevDifficulty"] = prev_diff
        idx_groups_avgs["PrevTrial"] = prev_trial_val
        df_li.append(idx_groups_avgs)

    df_org = df
    df = pd.concat(df_li)

    # df = df.set_index(index)
    # display(df)
    # print("columns:", list(cur_trial_cols))
    # print("index:", list(prev_trial_cols))
    df = df.pivot(columns=list(cur_trial_cols), index=list(prev_trial_cols))
    # display(df)
    df_update = df[["StayUpdate"]].droplevel(level=0, axis="columns")
    # df_nTrials = df[["nTrials"]].droplevel(level=0, axis="columns")
    df_nTrials_prcnt = df[["nTrialsPrnct"]].droplevel(level=0, axis="columns")
    df_performance = df[["Performance"]].droplevel(level=0, axis="columns")
    if is_many_subjects:
        df_update_sem = df[["StayUpdate_SEM"]].droplevel(level=0,
                                                         axis="columns")
        df_performance_sem = df[["Performance_SEM"]].droplevel(level=0,
                                                               axis="columns")
        # df_nTrial_sem = df[["nTrials_SEM"]].droplevel(level=0, axis="columns")
        df_nTrial_prcnt_sem = df[["nTrialsPrnct_SEM"]].droplevel(level=0,
                                                                 axis="columns")

    # print("Columns:", df.columns)
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
    # df_update = df_update.sort_index(axis='index', key=sortDiff, level=1)
    df_update = df_update.iloc[::-1]

    def sortDiffols(diff):
        # print("cols diff:", diff)
        cur_order = ["Easy", "Hard", "Med", "Total"
                     ]
        new_order = ["Easy", "Med", "Hard", "Total"
                    #  "Stay-Med", "Switch-Med",
                    ]
        # new_order = cur_order
        new_sort = [new_order.index(key) for key in cur_order]
        # print("Cur sort:", [cur_order[i] for i in new_sort])
        # print("new sort:", new_sort)
        return new_sort
    # display(df_update)
    df_update = df_update.sort_index(axis='columns', key=sortDiffols, level=1)
    # df_update = df_update.sort_index(axis='columns', ascending=[True, True])
    # df_update = df_update.droplevel(level=0, axis="columns")
    # display(df_update)

    fig1, axs = plt.subplots(1, 2, figsize=(7, 6), width_ratios=[1, 0.05])
    fig1.subplots_adjust(wspace=0.15)
    ax_heatmap, cmap_ax = axs

    # Unfortunatly, sns.heatmap doesn't give us the flexibility to draw circles
    # labels = np.array([[f"{cell:.2f}%" for cell in row]
    #                    for row in df_update.values])
    # min_val, max_val = df_update.values.min(), df_update.values.max()
    # def format(val, pos):
    #     if abs(val - min_val) < 0.000001:
    #         return "Fast"
    #     else:
    #         return "Slow"
    # sns.heatmap(df_update,
    #             #  cmap="YlOrRd_r",
    #             # cmap="autumn",
    #             cmap="PuOr",
    #             annot=labels,
    #             fmt="",
    #             cbar_kws=dict(#ticks=[0.4, 0.6],
    #                           #format=format
    #                             ),
    #             ax=ax_heatmap,
    #             cbar_ax=cmap_ax,
    #             vmin=-20, vmax=20,
    #             )


    x_cur_labels = df_update.columns.to_list()
    y_prev_labels = df_update.index.get_level_values(0).to_list()
    # print("x_labels:", x_cur_labels)
    # print("y_labels:", y_prev_labels)

    vmin, vmax = -20, 20
    x_cur_rng = np.arange(len(x_cur_labels))
    y_prev_rng = np.arange(len(y_prev_labels))
    x_cur, y_prev = np.meshgrid(x_cur_rng, y_prev_rng)
    updates = df_update.values.flatten()
    # display(z)
    # z_colors = (z - vmin) / (vmax - vmin)
    # display(z_colors)
    def getCorrespondigDF(other_df):
        other_df = other_df.copy()
        other_df = other_df[df_update.columns]
        other_df = other_df.reindex(df_update.index)
        return other_df.values.flatten()
    updates_perf = getCorrespondigDF(df_performance)
    # updates_ntrials = getCorrespondigDF(df_nTrials)
    updates_ntrials_prcnt = getCorrespondigDF(df_nTrials_prcnt)

    zip_li = [x_cur.flat, y_prev.flat, updates, #updates_ntrials,
              updates_ntrials_prcnt, updates_perf]
    if is_many_subjects:
        updates_sem = getCorrespondigDF(df_update_sem)
        updates_perfs_sem = getCorrespondigDF(df_performance_sem)
        # updates_ntrials_sem = getCorrespondigDF(df_nTrial_sem)
        updates_ntrials_prcnt_sem = getCorrespondigDF(df_nTrial_prcnt_sem)
        zip_li.extend([updates_sem, updates_perfs_sem, #updates_ntrials_sem
                       updates_ntrials_prcnt_sem])

    # print("z:", z)

    cmap = cm.ScalarMappable(norm=ColorNormalize(vmin, vmax), cmap="PuOr")
    for zip_iter in zip(*zip_li):
        i_cur, j_prev, update, n_trials_prcnt, perf, *sems = zip_iter
        # print("Update:", update, "nTrials:", n_trials, "Perf:", perf)
        if is_many_subjects:
            update_sem, perf_sem, n_trials_prcnt_sem = sems
        MAX_RADIUS = 0.5
        radius = MAX_RADIUS*n_trials_prcnt / updates_ntrials_prcnt.max()
        bg_clr = cmap.to_rgba(update)
        # print("Color:", bg_clr)
        CIRCLE = False
        # Not sure why edecolor isn't working, so we are drawing outer circle
        EDGE_WIDTH = 0.003
        if CIRCLE:
            ax_heatmap.add_artist(plt.Circle((i_cur,j_prev),
                                           radius=radius+EDGE_WIDTH, color="k"))
            ax_heatmap.add_artist(plt.Circle((i_cur,j_prev), radius=radius,
                                             color=bg_clr))
        else:
            ax_heatmap.add_artist(plt.Rectangle((i_cur-0.5,j_prev-0.5),
                                                width=1, height=1,
                                                color=bg_clr,))

        # print("radius:", radius)
        draw_outside = CIRCLE and radius < 0.2
        if draw_outside:
            j_offset = radius + 0.05
            j_offset2 = j_offset + 0.1
            txt_clr = "black"
            va = "top"
            font_size = 12
        else:
            j_offset = 0
            j_offset2 = (radius + 0.05) if CIRCLE else 0.15
            # Contrast based on background color:
            # https://stackoverflow.com/a/3943023/11996983
            txt_clr = "black" if (bg_clr[0]*0.299 + bg_clr[1]*0.587 +
                                  bg_clr[2]*0.114) > 0.7265 else "white"
            txt_2nd_clr = txt_clr if CIRCLE else (
                                 "#6b6b6b" if txt_clr == "black" else "#e0e0e0")
            font_size = max(10, radius*15)
            va = "center"
        update_str = f"{update:.2f}%"
        if is_many_subjects:
            update_str += f" ±{update_sem:.2f}%"
        ax_heatmap.text(i_cur, j_prev + j_offset, update_str, color=txt_clr,
                        fontsize=font_size, ha='center', va=va, zorder=10)
        trials_prcnt = n_trials_prcnt#100*n_trials/total_trials
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



    MAX_RADIUS += (0.003*2*np.pi) # for the black outline circles
    ax_heatmap.tick_params(axis="both", which="both", length=0, pad=15)
    ax_heatmap.xaxis.tick_top()
    ax_heatmap.xaxis.set_label_position('top')
    ax_heatmap.set_xlabel("Current Trial")
    # ax_heatmap.xaxis.set_tick_params(rotation=0)
    ax_heatmap.set_xticks(x_cur_rng)
    ax_heatmap.set_xticklabels(x_cur_labels, minor=False)
    ax_heatmap.set_xlim(-MAX_RADIUS, len(x_cur_labels) - MAX_RADIUS)
    ax_heatmap.set_ylabel("Previous Trial")
    ax_heatmap.set_yticks(y_prev_rng, minor=False)
    ax_heatmap.set_yticklabels(y_prev_labels)
    ax_heatmap.set_ylim(len(y_prev_labels) - MAX_RADIUS, -MAX_RADIUS)


    # cbar = ax_heatmap.collections[0].colobar
    # cbar.set_ticks([df.min(),df.max()])
    # cbar.set_ticklabels(["Slow", "Fast"])
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
    # size = fig1.get_size_inches()
    # print("size:", size)
    plt.show()


def _calcGroupUpdate(grp_df, is_many_subjects, prev_trial_val,
                     num_trials_subject=None, num_trials=None):

    if hasattr(grp_df, "name"):
        name = grp_df.name
    else:
        name = "All"
    grp_df = grp_df.copy()
    # stay_update = 100*(grp_df.Stay.sum() -  grp_df.StayBaseline.sum())/grp_df.StayBaseline.sum()
    grp_df = grp_df[grp_df.Stay.notnull()]
    if False: # Don't use it, it doubles the effect of a switch
        grp_df.loc[grp_df.Stay == 0, "Stay"] = -1
        grp_df.loc[grp_df.StayBaseline == 0, "StayBaseline"] = -1
    # print("Prev:", prev_trial_val, "Cur:", name,
    #       "Stay count:", grp_df.Stay.sum(),
    #       "Baseline count:", grp_df.StayBaseline.sum(),
    #       "Total len:", len(grp_df))

    more_cols = ["StayUpdate", "StayMean", "RefStayMean", "Performance",
                 "nTrials", "nTrialsPrnct"]
    if is_many_subjects:
        assert num_trials_subject is not None, (
                        "num_trials_subject must be provided for many subjects")
        more_cols_n_subj = more_cols + ["Name"]
        # for col in more_cols_n_subj:
        #     res_dict[col] = []
        #     res_dict[f"{col}_SEM"] = []

        subject_res_grp_by = grp_df.groupby("Name").apply(_calcUpdate)
        local_dict = {}
        for col in more_cols_n_subj:
            if col == "nTrialsPrnct": # We will calculate it later
                continue
            local_dict[col] = [subject_vals[col]
                               for subject_vals in subject_res_grp_by.values]
        # display(subject_res_dict)
        subject_grps_df = pd.DataFrame(local_dict)
        # display(subject_grps_df)
        subject_grps_df["nTrialsPrnct"] = subject_grps_df.apply(
            lambda row: 100*row["nTrials"]/num_trials_subject[row.Name],
            axis=1)

        res_dict = {}
        for col in more_cols:
            vals = subject_grps_df[col]
            avg = vals.mean()
            sem = vals.sem()
            # print(f"Prev trial: {prev_trial_val} - Cur: {name} - "
            #         f"{col}: {avg:.2f} - SEM: {sem:.2f}")
            res_dict[col] = avg
            res_dict[f"{col}_SEM"] = sem
    else:
        assert num_trials is not None, (
                               "num_trials must be provided for single subject")
        res_dict = _calcUpdate(grp_df)
        res_dict["nTrialsPrnct"] = 100*len(grp_df)/num_trials
    # display(grp_df)
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
    # The following 2 lines the same as the above
    # stay_update = 100*(grp_df.Stay.mean() -  grp_df.StayBaseline.mean())/grp_df.StayBaseline.mean()
    # stay_update = 100*(grp_df.Stay.mean() - grand_actual_stay)/grand_actual_stay
    # A failed attempt to calculate the mean and SEM as function of each session
    # stay_update = grp_df.groupby(["Name", "Date", "SessionNum"]).filter(lambda df:len(df) > 50 and grp_df.StayBaseline.sum() >)
    # stay_update = grp_df.groupby(["Name", "Date", "SessionNum"]).apply(
    #                    lambda grp_df:100*(grp_df.Stay.sum() -  grp_df.StayBaseline.sum())/grp_df.StayBaseline.sum())
    # stay_update = stay_update[stay_update.notnull()].mean()
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
