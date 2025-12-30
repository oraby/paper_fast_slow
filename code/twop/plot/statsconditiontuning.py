from ...common.clr import Choice as ChoiceClr
from ...common.definitions import BrainRegion
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

def plotConditionTuning(df, data_col, epoch, pval, active_traces_df,
                        left_tuning_col, fig_save_prefix,
                        neurouns_total_count_df=None,
                        limit_to_brain_regions=None):
    epoch_active = active_traces_df[active_traces_df.epoch == epoch]
    epoch_active = epoch_active.drop_duplicates(subset=["trace_id"])
    if limit_to_brain_regions is not None:
        df = df[df.BrainRegion == limit_to_brain_regions]
    else:
        assert False, "Not implemented for multiple brain regions"
    df = df[df.data_col == data_col]
    df = df[df.prior_data_col.isna()]
    df = df[df.epoch == epoch]

    USE_ACTIVE_TRACES = False
    # Sub-optimum but easier to argue about
    left_pref_str, right_pref_str = \
                           df.data_val_left.unique(), df.data_val_right.unique()
    assert len(left_pref_str) == len(right_pref_str) == 1
    left_pref_str, right_pref_str = left_pref_str[0], right_pref_str[0]
    if data_col == "ChoiceCorrect":
        left_pref_str, right_pref_str = "Correct", "Incorrect"
    ys = []
    left_preference_ys = []
    for sess, sess_df in df.groupby("ShortName"):
        sess_active_traces = sess_df[
                                   sess_df.trace_id.isin(epoch_active.trace_id)]
        if not len(sess_active_traces):
            print("No active traces for sess:", sess, "in epoch:", epoch,
                  "Sess traces:", len(sess_df))
            if USE_ACTIVE_TRACES: # Avoid division by zero
                continue
        sess_all_traces = sess_df.trace_id
        assert len(sess_all_traces) == len(sess_all_traces.unique())
        if neurouns_total_count_df is not None:
            sess_total_neurons = neurouns_total_count_df[
                                      neurouns_total_count_df.ShortName == sess]
            sess_total_neurons = sess_total_neurons.total_neurons.iloc[0]
            assert sess_total_neurons == len(sess_all_traces), (
                         f"{sess_total_neurons = } != {len(sess_all_traces) =}")
        sess_sgf_traces = sess_df[sess_df.pval <= pval]
        sgf_active_traces = sess_sgf_traces[
                     sess_sgf_traces.trace_id.isin(sess_active_traces.trace_id)]
        if USE_ACTIVE_TRACES:
            prcnt = 100*len(sgf_active_traces)/len(sess_active_traces)
        else:
            prcnt = 100*len(sgf_active_traces)/len(sess_all_traces)
        ys.append(prcnt)
        # display(sgf_active_traces)
        sgf_left_tuned = sgf_active_traces[
                                     sgf_active_traces[left_tuning_col] == True]
        len_sgf = len(sgf_active_traces)
        if not len_sgf: len_sgf = 1
        left_tuned_prcnt = 100*len(sgf_left_tuned)/len_sgf
        left_preference_ys.append(left_tuned_prcnt)

    dist_labels = [f"{epoch} Active\nNeurons", "Trial Outcome\nPreference"]
    y_mean = np.nanmean(ys)
    y_sem = sem(ys)
    fig, ax = plt.subplots(figsize=(10,10))
    # Draw a pie-chart
    _, _, prcnt = ax.pie([100-y_mean, y_mean], labels=dist_labels,
                         colors=[(207/255, 224/255, 227/255), "orange"],
                         autopct='%1.1f%%', startangle=35, explode=(0, 0.1))
    prcnt[0].remove()
    if fig_save_prefix is not None:
        fig.savefig(f"{fig_save_prefix}/pie_{epoch}_{data_col}_dist.jpeg",
                    dpi=300, bbox_inches="tight")
    plt.show()


    fig, ax = plt.subplots(figsize=(10,10))
    pref_means = np.nanmean(left_preference_ys)
    ax.pie([pref_means, 100-pref_means], labels=[left_pref_str, right_pref_str],
           colors=[ChoiceClr.Correct, ChoiceClr.Incorrect],
           autopct='%1.1f%%', startangle=20)
    if fig_save_prefix is not None:
        fig.savefig(f"{fig_save_prefix}/pie_{epoch}_{data_col}_tuning.jpeg",
                    dpi=300, bbox_inches="tight")
    plt.show()

    from matplotlib.patches import ConnectionPatch
    # make figure and assign axis objects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 20))
    fig.subplots_adjust(wspace=0.2)
    if data_col == "ChoiceCorrect":
        data_col = "Trial Outcome"
    br_str = str(BrainRegion(limit_to_brain_regions)).split("_")[0]
    fig.suptitle(f"{data_col} distrubtion and tuning during {epoch} for "
                 f"{br_str}", y=0.7, fontsize='large')
    # pie chart parameters
    wedges, _, percentages = ax1.pie([100-y_mean, y_mean],
                                     labels=dist_labels,
                                     colors=[
                                         (207/255, 224/255, 227/255), "orange"],
                                     autopct='%1.1f%%', startangle=20,
                                     explode=(0, 0.1), radius=1.2,
                                     labeldistance=1.02)
    percentages[0].remove()
    print("done, Labels:")
    # bar chart parameters
    width = .2
    pref_means = np.nanmean(left_preference_ys)
    radius = 0.8
    ax2.pie([pref_means, 100-pref_means], labels=[left_pref_str, right_pref_str],
            colors=[ChoiceClr.Correct, ChoiceClr.Incorrect],
            autopct='%1.1f%%', startangle=55, radius=radius)
    # ax2.set_title('Age of approvers')
    # ax2.legend()
    ax2.axis('off')

    # use ConnectionPatch to draw lines between the two plots
    theta2, theta1 = wedges[0].theta1, wedges[0].theta2
    center, r = wedges[0].center, wedges[0].r
    bar_height = radius

    # draw top connecting line
    x = r * np.cos(np.pi / 180 * theta2) + center[0]
    y = r * np.sin(np.pi / 180 * theta2) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                          xyB=(x, y), coordsB=ax1.transData)
    con.set_color([0, 0, 0])
    con.set_linewidth(2)
    ax2.add_artist(con)

    # draw bottom connecting line
    x = r * np.cos(np.pi / 180 * theta1) + center[0]
    y = r * np.sin(np.pi / 180 * theta1) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, -bar_height), coordsA=ax2.transData,
                          xyB=(x, y), coordsB=ax1.transData)
    con.set_color([0, 0, 0])
    ax2.add_artist(con)
    con.set_linewidth(2)
    if fig_save_prefix is not None:
        fig.savefig(f"{fig_save_prefix}/pie_{epoch}_{data_col}_comb.jpeg",
                    dpi=300, bbox_inches="tight")
    plt.show()
