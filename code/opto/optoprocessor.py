from .bootstrap2regions import bootstrapSignTestApproach2#runPermutationMaxT
from ..figcode.psychometric import plotPsych
from ..common.clr import BrainRegion as BRC
from ..common.definitions import BrainRegion, MatrixState
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
try:
    from IPython.display import display
except ModuleNotFoundError:
    pass
from scipy import stats
from statsmodels.stats import multitest
from typing import Literal, List
import pathlib
from enum import Enum, IntFlag, auto
from functools import partial

class SplitLevelBy(IntFlag):
  AllAnimals = 0
  Session = auto()
  Animal = auto()

def loopConfig(df, applyFn, min_choice_trials,
               which : Literal["partial", "full", "all"] = "all",
               stim_type : Literal["FT", "RT", "all", "feedback"] = "all",
               process_groups_together=False,
               single_subjects=True,
               save_prefix=None, save_figs=False):
    assert which in ["partial", "full", "all" ]
    assert stim_type in ["FT", "RT", "all", "feedback"]
    df = df.copy()
    loop_cols = ["GUI_OptoStartState1", "GUI_OptoStartDelay", "GUI_OptoMaxTime",
                 "GUI_OptoEndState1", "GUI_StimulusTime"]
    df = df.groupby(loop_cols).filter(lambda grp_df: len(grp_df) >= 10)

    ret = _processSubject(df, applyFn, min_choice_trials, "All Mice", loop_cols,
                          which, stim_type, save_prefix=save_prefix,
                          process_groups_together=process_groups_together,
                          save_figs=save_figs)
    if not single_subjects:
        return ret
    for animal_name, animal_df in df.groupby("Name"):
        _processSubject(animal_df, applyFn, min_choice_trials, animal_name,
                        loop_cols, which, stim_type, save_prefix=save_prefix,
                        process_groups_together=process_groups_together,
                        save_figs=save_figs)

    return ret

def _processSubject(df, applyFn, min_choice_trials, subject_name, loop_cols,
                    which : Literal["partial", "full", "all"],
                    stim_type : Literal["FT", "RT", "all", "feedback"],
                    process_groups_together=False,
                    save_prefix=None, save_figs=False):
    if save_figs:
        assert save_prefix is not None, "save_prefix must be specified"
        save_prefix_path = pathlib.Path(save_prefix)
        assert save_prefix_path.parent.exists(), (
                                      f"Directory: {save_prefix} doesn't exist")
        save_prefix_path /= subject_name
        save_prefix_path.mkdir(exist_ok=True, parents=True)

    feedback_start_states = [MatrixState.WaitForReward,
                             MatrixState.WaitForPunish]
    df_rt = df[(df.GUI_StimulusTime >= 5) & ~df.GUI_OptoStartState1.isin(feedback_start_states)]
    df_ft = df[(df.GUI_StimulusTime < 5) & ~df.GUI_OptoStartState1.isin(feedback_start_states)]
    df_feedback = df[df.GUI_OptoStartState1.isin(feedback_start_states)]

    ret_li = []

    stim_type_loops = []
    if stim_type in ["FT", "all"]:
        stim_type_loops.append(("FT", df_ft))
    if stim_type in ["RT", "all"]:
        stim_type_loops.append(("RT", df_rt))
    if stim_type == "feedback":
        stim_type_loops.append(("feedback", df))
    for stim_type, stim_df in stim_type_loops:
        # display(stim_df[loop_cols].value_counts())
        # Get all opto trials during sampling phase
        df_stim_delv = stim_df[stim_df.GUI_OptoStartState1 ==
                               int(MatrixState.stimulus_delivery)]

        df_full_sampling = df_stim_delv[(df_stim_delv.GUI_OptoStartDelay == 0) &
                                        (df_stim_delv.GUI_OptoMaxTime ==
                                         df_stim_delv.GUI_StimulusTime)]
        print("df_full_sampling:", len(df_full_sampling))
        print("df feedback:", len(df_feedback))
        df_partial_sampling = stim_df[~stim_df.index.isin(
                                                        df_full_sampling.index)]
        if stim_type == "feedback":
            start_state = MatrixState.WaitForReward
            start_delay = df_feedback.GUI_OptoStartDelay.mode()[0]
            max_dur = df_feedback.GUI_OptoMaxTime.mode()[0]
            stimulus_time = df_feedback.GUI_StimulusTime.mode()[0]
            end_state = df_feedback.GUI_OptoEndState1.mode()[0]

            ret = applyFn(start_state, start_delay, max_dur, stimulus_time, end_state,
                          stim_type, df_feedback,
                          min_choice_trials=min_choice_trials,
                          subject_name=subject_name, save_prefix=save_prefix,
                          save_figs=save_figs)
            ret_li.append(ret)
            continue

        if which in ["full", "all"]:
            # START_DELAY=0
            MAX_DUR=-1
            for (stimulus_time, end_state), grp_df in df_full_sampling.groupby(
                                    ["GUI_StimulusTime", "GUI_OptoEndState1"]):
                print("End State:", end_state)
                display(grp_df[loop_cols +
                               ["GUI_OptoEndState2"]].value_counts())
                display(grp_df.agg(
                    {'GUI_StimulusTime': ['count', "min", "max", "mean"],
                      #'MinSample': ['min', 'max', 'mean',],
                      "calcStimulusTime": ["min", "max", "mean",],
                      "Date": ["min", "max"],
                     }
                ))
                # assert (grp_df.GUI_StimulusTime < 5).all(), (
                # "All FT trials should have stimulus time less than 5 seconds")
                start_state = MatrixState.stimulus_delivery
                ret = applyFn(start_state, 0, MAX_DUR, stimulus_time, end_state,
                              stim_type, grp_df, min_choice_trials=min_choice_trials,
                              subject_name=subject_name, save_prefix=save_prefix,
                              save_figs=save_figs)
                ret_li.append(ret)

        if which in ["partial", "all"]:
            def _processGroup(start_state, start_delay, max_dur, end_state,
                              stimulus_time, grp_df):
                start_state = MatrixState(start_state)
                return applyFn(start_state, start_delay, max_dur, stimulus_time,
                               end_state, stim_type, grp_df,
                               min_choice_trials=min_choice_trials,
                               subject_name=subject_name,
                               save_prefix=save_prefix,
                               save_figs=save_figs)

            if process_groups_together:
                start_state = df_partial_sampling.GUI_OptoStartState1.unique()
                start_delay = df_partial_sampling.GUI_OptoStartDelay.unique()
                max_dur = df_partial_sampling.GUI_OptoMaxTime.unique()
                end_state = df_partial_sampling.GUI_OptoEndState1.unique()
                stimulus_time = df_partial_sampling.GUI_StimulusTime.unique()
                ret = _processGroup(start_state, start_delay, max_dur, end_state,
                                    stimulus_time, df_partial_sampling)
                ret_li.append(ret)
            else:
                for grp_key, grp_df in df_partial_sampling.groupby(loop_cols):
                    (start_state, start_delay, max_dur, end_state,
                    stimulus_time) = grp_key
                    ret = _processGroup(start_state, start_delay, max_dur, end_state,
                                        stimulus_time, grp_df)
                    ret_li.append(ret)
    return ret_li

def _classifyOptoConfig(start_state, start_delay, max_dur, stimulus_time,
                        end_state, stim_type):
    def stateFriendlyName(state):
        if state == MatrixState.stimulus_delivery:
            state = "Sampling"
            if stimulus_time not in [1, 5]:
                state += f" ({stimulus_time}s)"
        elif state == MatrixState.CenterPortRewardDelivery:
            state = "Decision-Time" if stim_type == "FT" else "Center-Reward"
        elif state == MatrixState.TriggerWaitForStimulus:
            state = "Pre-Stimulus Delay"
        else:
            state = str(MatrixState(state))
        return state

    start_state = stateFriendlyName(start_state)
    end_state = stateFriendlyName(end_state)

    handled = False
    if stim_type == "RT" and stimulus_time >= 5:
        if "Sampling" in start_state and start_delay == 0.3 and \
           max_dur in [0.3, 0.6]:
            duration = "Short RT" if max_dur == 0.3 else "Med RT"
            handled = True
    else:
        print("Start state:", start_state)
        if "Sampling" in start_state and stimulus_time == 1:
            if start_delay == 0 and max_dur == -1:
                duration = "FT Full Sampling "
                handled = True
            elif start_delay == 0 and abs(max_dur - 0.35) < 0.01:
                duration = f"FT Early Sampling"
                handled = True
            elif abs(start_delay - 0.65) < 0.01 and abs(max_dur - 0.35) < 0.01:
                duration = f"FT Late Sampling"
                handled = True


    if not handled:
        duration = (f"{stim_type} - {start_state} - s: {start_delay:.2f}s → "
                    f"e: {start_delay+max_dur:.2f}s")
    duration += f" - End: {end_state}"
    return duration

def plotOptoPsych(start_state, start_delay, max_dur, stimulus_time, end_state,
                  stim_type, df, subject_name, split_level, min_choice_trials,
                  save_prefix=None, save_figs=False):
    title = _classifyOptoConfig(start_state, start_delay, max_dur,
                                stimulus_time, end_state, stim_type)
    title = f"{subject_name} - {title}"

    # df = df[df.ChoiceCorrect.notnull()]
    for brain_region, br_df in df.groupby("GUI_OptoBrainRegion"):
        brain_region = BrainRegion(brain_region)
        cur_title = f"{title} - {brain_region}"
        for grp, grp_df in _loopBrainRegionOptoConfig(br_df, split_level,
                                           min_choice_trials=min_choice_trials):
            fig, ax = plt.subplots(figsize=(10, 6))
            control_trials_df = grp_df[grp_df.OptoEnabled == 0]
            opto_trials_df = grp_df[grp_df.OptoEnabled == 1]
            print("Control Trials:", len(control_trials_df))
            print("Opto Trials:", len(opto_trials_df))
            plotPsych(control_trials_df, title="", subject_name="Control",
                      by_session=False, by_subject=False, ax=ax,
                      is_human_subject=False, combine_sides=False,
                      save_figs=False)
            color = BRC[brain_region]
            plotPsych(opto_trials_df, title="", subject_name="Opto",
                      default_color=color,
                      by_session=False, by_subject=False, ax=ax,
                      is_human_subject=False, combine_sides=False,
                      save_figs=False)
            ax.set_title(f"{cur_title} - {grp}")
            if save_figs:
                save_fp = f"{subject_name}/{brain_region}_psych.svg"
                save_fp = save_fp.replace(" ", "_").replace(":", "_").replace(
                                                                       "=", "_")
                final_save_fp = f"{save_prefix}/{save_fp}"
                # print("Saveing fig to:", final_save_fp)
                pathlib.Path(final_save_fp).parent.mkdir(exist_ok=True)
                fig.savefig(final_save_fp)
            plt.show()

class EffectSizeMetric(Enum):
    Gain = auto()
    DistanceFromRandom = auto()
    Raw = auto()

def plotOptoEffect(start_state, start_delay, max_dur, stimulus_time, end_state,
                   stim_type, df, df_col_name, subject_name,
                   num_iterations : int, split_level, min_choice_trials,
                   # TODO: Handle continous non-binary values
                   mean_or_median : Literal["mean", "median"],
                   effect_size_metric : EffectSizeMetric, #min_sess_perf,
                   only_brain_regions : List[BrainRegion] = [],
                   save_prefix=None, save_figs=False):


    uniq_br = [BrainRegion(br) for br in df.GUI_OptoBrainRegion.unique()]
    br_df_order = [br for br in
                  [BrainRegion.V1_Bi, BrainRegion.RSP_Bi, BrainRegion.PPC_Bi,
                   BrainRegion.M2_Bi, BrainRegion.ALM_Bi]
                   if br in uniq_br]
    br_df_order = br_df_order + [br for br in uniq_br
                                 if br not in br_df_order]
    assert len(br_df_order) <= len(df.GUI_OptoBrainRegion.unique())
    print("Order:", br_df_order)
    effects_size_dict_dict = {}
    max_y = 0

    if len(only_brain_regions):
        br_df_order = [br for br in br_df_order if br in only_brain_regions]

    df = df[df.GUI_OptoBrainRegion.isin(br_df_order)]

    print("Brain Regions:", br_df_order)
    def assignOptoContrlCount(grp_df):
        grp_df = grp_df.copy()
        grp_df["OptoCount"] = (grp_df.OptoEnabled == 1).sum()
        grp_df["ControlCount"] = (grp_df.OptoEnabled == 0).sum()
        grp_df["OptoAccept"] = grp_df.OptoCount >= min_choice_trials
        grp_df["ControlAccept"] = grp_df.ControlCount >= min_choice_trials
        grp_df["ControlPerf"] = grp_df[grp_df.OptoEnabled == 0].ChoiceCorrect.mean()
        grp_df["OptoPerf"] = grp_df[grp_df.OptoEnabled == 1].ChoiceCorrect.mean()
        grp_df["OptoEffect"] = _calcEffectSize(
                                    grp_df[grp_df.OptoEnabled == 0]
                                        [df_col_name].mean(),
                                    grp_df[grp_df.OptoEnabled == 1]
                                        [df_col_name].mean(),
                                    effect_size_metric=effect_size_metric)
        return grp_df
    df_disp = df.groupby(["Name", "GUI_OptoBrainRegion", "GUI_OptoStartDelay",
                          "GUI_OptoMaxTime",], group_keys=False
                         ).apply(assignOptoContrlCount)
    display(df_disp[["Name", "GUI_OptoBrainRegion", "OptoCount",
                        "ControlCount", "OptoAccept", "ControlAccept"]]
                        .value_counts().sort_index())

    num_brain_regions = len(br_df_order)
    num_start_delays = df.GUI_OptoStartDelay.nunique()
    print("Num of Brain Regions:", num_brain_regions)
    if num_brain_regions == 2 and num_start_delays == 2:
        return _2x2BrainRegionOptoConfigLoop(df, df_disp, subject_name, split_level,
                                            min_choice_trials,
                                            effect_size_metric, num_iterations,
                                            br_df_order, save_prefix=save_prefix,
                                            save_figs=save_figs)


    fig, ax = plt.subplots(figsize=(10, 6))
    cur_x_pos = 3  # Start at 3 to leave space for the big legend
    x_ticks = []
    x_labels = []
    y_medians = []
    y_ticks_top = []
    y_ticks_bot = []
    labels = []
    colors = []
    pvals = []
    used_df_li = []

    for brain_region in br_df_order:
        br_df = df[df.GUI_OptoBrainRegion == brain_region]
        assert len(br_df)

        loop = _loopBrainRegionOptoConfig(br_df, split_level,
                                          min_choice_trials=min_choice_trials)
        effect_size_dict, _pval, book_keeping = _processBrainRegionOptoConfig(
                                  loop, df_col_name=df_col_name,
                                  mean_or_median=mean_or_median,
                                  effect_size_metric=effect_size_metric,
                                  #min_sess_perf=min_sess_perf
                                  num_iterations=num_iterations,
                                  )
        pvals.append(_pval)
        if not len(effect_size_dict):
            continue

        effects_size_dict_dict[brain_region] = effect_size_dict
        effect_size_li = list(effect_size_dict.values())
        mean = np.mean(effect_size_li)
        sem = stats.sem(effect_size_li)
        (used_subjects, used_sessions, num_opto_trials,
         num_control_trials, used_df) = book_keeping
        used_df_li.append(used_df)
        num_subjects = len(used_subjects)
        num_sessions = len(used_sessions)

        label = (f"{brain_region} - {mean:.2f} ±{sem:.2f}\n"
                 f"p-val (corrected)=PVAL_HOLDER\n"
                 f"(n={num_subjects} subjects - {num_sessions} sessions - \n"
                 f" {num_opto_trials:,}/{num_control_trials:,} Control trials)")
        labels.append(label)
        color = BRC[brain_region]
        BOX_PLOT = False
        if BOX_PLOT:
            box_plot = \
                ax.boxplot(effect_size_li, positions=[cur_x_pos], widths=0.6,
                           patch_artist=True, boxprops=dict(facecolor=color),
                           medianprops=dict(color="black"),
                           whiskerprops=dict(color="black"),
                           capprops=dict(color="black"),
                           flierprops=dict(markerfacecolor=color, alpha=0.2,
                                           marker='o', markersize=5,
                                           linestyle='none'),
                           labels=[label])
        else:
            ax.bar(cur_x_pos, mean, yerr=sem, label=label, # Just put any label for now
                color=color,)
            # Scatter
            ax.scatter(np.ones(len(effect_size_li))*cur_x_pos, effect_size_li,
                    edgecolor='k', facecolor='white', alpha=0.8)
        top_y = max(effect_size_li)
        bot_y = min(effect_size_li)
        x_ticks.append(cur_x_pos)
        x_labels.append(f"{brain_region}")
        y_medians.append(mean)

        y_ticks_top.append(top_y)
        y_ticks_bot.append(bot_y)

        cur_x_pos += 1

    # Check whether most data is negative or positive
    y_medians_mean = np.mean(y_medians)
    if y_medians_mean > 0:
        y_ticks = y_ticks_top
        pos_y = True
    else:
        y_ticks = y_ticks_bot
        pos_y = False
    max_y = max(y_ticks)

    # print("Effects Size Dict:", effects_size_dict)
    # stats_data = list(effects_size_dict.values())
    # print("Stats Data:", stats_data)
    # stats_pvals_zip = [stats.wilcoxon(data, np.zeros(len(data)))
    #                    for data in stats_data]
    # pvals = [pval for _, pval in stats_pvals_zip]
    pvals = np.array(pvals)
    # print("Wilcoxon:", pvals)
    P_ADJUST = "holm"
    ALPHA = 0.05
    rejected, corrected_pvals, _, _ = multitest.multipletests(pvals, alpha=ALPHA,
                                                              method=P_ADJUST)
    # print("Corrected Pvals:", corrected_pvals)
    # print("Rejected:", rejected)

    for i, corrected_pval in enumerate(corrected_pvals):
        # Update the labels with the corrected pvals
        labels[i] = labels[i].replace("PVAL_HOLDER", f"{corrected_pval:.3f}")
        if len(pvals) == 1:
            labels[i] = labels[i].replace(" (corrected)", "")
        if corrected_pval <= ALPHA:
            assert rejected[i]
            # print("Corrected Pval:", corrected_pval)
            star = "***" if corrected_pval <= 0.001 else "**" if corrected_pval <= 0.01 else "*"
            y = max_y + 5
            x = x_ticks[i]
            print("Y[ticks]:", y_ticks[i])
            y = y_ticks[i] + 0.2 if y_ticks[i] > pos_y else y_ticks[i] - 0.2
            ax.text(x, y, star, fontsize=12, ha="center",
                    va="bottom" if y > pos_y else "top")
        else:
            if rejected[i]:
                print("************** Rejected:", corrected_pval)
                continue
            assert not rejected[i]

        # print("Pval:", pval)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    title = _classifyOptoConfig(start_state, start_delay, max_dur,
                                stimulus_time, end_state, stim_type)
    title = f"{subject_name} - {title}"
    # Set y-axis tp user percent formatter
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title(title)
    # Despine in 3 directions
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlim(0, cur_x_pos)
    if effect_size_metric == EffectSizeMetric.DistanceFromRandom:
        ax.set_ylim(top=120)
        ax.axhline(0, color="black", linestyle="--",
                   label="Control Performance")
        ax.axhline(100, color="black", linestyle="--",
                   label="Chance Performance")
        ax.spines[["bottom"]].set_visible(False)
        ax.set_ylabel("Distance from Random (%)")
    elif effect_size_metric == EffectSizeMetric.Gain:
        # ax.set_ylim(top=45)
        ax.set_ylim(top=65)
        ax.axhline(0, color="black", linestyle="--",)
        ax.set_ylabel("Performance Drop (%)")
    handles_labels = ax.get_legend_handles_labels()
    # Replace with the corrected labels
    # handles_labels = (handles_labels[0], labels)
    # Create rectuangular icons handles for the legend as boxplot doesn't add the labels
    # as we want them
    handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
    ax.legend(handles + handles_labels[0],  labels + handles_labels[1],
              loc="upper left", fontsize="xx-small", bbox_to_anchor=(0, .9))
    if save_figs:
        title = title.replace(" ", "_").replace(":", "_").replace("=", "_")
        metric_str = "Gain" if effect_size_metric == EffectSizeMetric.Gain else\
                     "DistFromRandom"
        if len(only_brain_regions):
            brain_region_str = '_' +  '_'.join(str(br)
                                               for br in only_brain_regions)
        else:
            brain_region_str = ""
        fig.savefig(f"{save_prefix}{metric_str}{brain_region_str}_{title}.svg")
    plt.show()

    return pd.concat(used_df_li, ignore_index=True)

def _2x2BrainRegionOptoConfigLoop(df, df_disp, subject_name, split_level,
                                  min_choice_trials,
                                  effect_size_metric : EffectSizeMetric,
                                  num_iterations : int,
                                  br_df_order: List[BrainRegion],
                                  save_figs: bool = False,
                                  save_prefix: str = None):
    print("********************* Check the next part carefully *********************")
    df_disp = df_disp[df_disp.ControlAccept & df_disp.OptoAccept]
    df = df_disp
    df_disp = df_disp.rename(columns={"GUI_OptoBrainRegion": "OptoBrainRegion"})
    df_disp["OptoBrainRegion"] = df_disp.OptoBrainRegion.apply(lambda br: str(BrainRegion(br))[:-3])
    df_disp["Phase"] = "Early"
    df_disp.loc[df_disp.GUI_OptoStartDelay > 0.5, "Phase"] = "Late"
    display(df_disp[["Name", "OptoBrainRegion", "Phase", "OptoCount", "ControlCount",
                        # "OptoAccept", "ControlAccept",
                        "ControlPerf","OptoPerf", "OptoEffect"]]
                    .value_counts().sort_index())
    # df = df[~((df.Name == "BVAGT4") & (df.GUI_OptoBrainRegion == BrainRegion.ALM_Bi) &
    #           (df.GUI_OptoStartDelay == 0))]
    # Construct a total df
    res_df_li = []
    for brain_region in br_df_order:
        br_df = df[df.GUI_OptoBrainRegion == brain_region]
        early_df = br_df[br_df.GUI_OptoStartDelay <= 0.5]
        late_df = br_df[br_df.GUI_OptoStartDelay > 0.5]
        for (is_early, is_early_df) in [(True, early_df), (False, late_df)]:
            assert len(is_early_df), f"No data for {brain_region} - Early?: {is_early}"
            # print(f"Brain Region: {brain_region} - {is_early}")
            for entry, entry_df in _loopBrainRegionOptoConfig(is_early_df, split_level,
                                        min_choice_trials=min_choice_trials):
                entry_df = entry_df.copy()
                entry_df["IsEarly"] = is_early
                res_df_li.append(entry_df)

    two_regions_df = pd.concat(res_df_li, ignore_index=True)
    used_df = two_regions_df.copy()
    two_regions_df["SessId"] = two_regions_df.apply(
                    lambda row: f"{row.Name}_{row.Date}_{row.SessionNum}",
                    axis=1)
    two_regions_df = two_regions_df[["Name", "SessId", "IsEarly",
                                        "OptoEnabled", "ChoiceCorrect",
                                        "TrialNumber",
                                        "GUI_OptoBrainRegion"]]
    two_regions_df["OptoBrainRegion"] = two_regions_df.apply(
        lambda row: "MFC" if row.GUI_OptoBrainRegion == BrainRegion.M2_Bi.value else
                    ("LFC" if row.GUI_OptoBrainRegion == BrainRegion.ALM_Bi.value else row.GUI_OptoBrainRegion),
        axis=1)
    two_regions_df = two_regions_df.drop("GUI_OptoBrainRegion", axis=1)
    calcEffectSizePartial = partial(_calcEffectSize,
                                    effect_size_metric=effect_size_metric)
    res_tup = bootstrapSignTestApproach2(trials_df=two_regions_df,
                                    iterations=num_iterations,
                                    calcPerfFn=calcEffectSizePartial,
                                    #adjust_scope="per_phase"
                                    )
    res_stats_dict, res_obs_entries = res_tup
    # print("Two Regions Result Dict:", res_stats_dict)
    for k, v in res_stats_dict.items():
        print(f"Phase: {k[0]} - Region: {k[1]} - {k[2]} = {v:.4f}")

    fig, ax = plt.subplots(figsize=(14, 6))
    for is_early, is_early_df in two_regions_df.groupby("IsEarly"):
        phase_name = "Early" if is_early else "Late"
        x_offset = 0 if is_early else 3
        region_max_y = -np.inf
        for brain_region, br_df in is_early_df.groupby("OptoBrainRegion"):
            perf_data = br_df.groupby("Name").apply(
                lambda grp: _calcEffectSize(grp[grp.OptoEnabled == 0].ChoiceCorrect.mean(),
                                            grp[grp.OptoEnabled == 1].ChoiceCorrect.mean(),
                                            effect_size_metric=effect_size_metric))
            p_val_adj = res_stats_dict[(phase_name, brain_region, "p_holm_within_phase")]
            num_subjects = len(perf_data)
            control_count = (br_df.OptoEnabled == 0).sum()
            opto_count = (br_df.OptoEnabled == 1).sum()
            perf_mean, perf_sem = perf_data.mean(), stats.sem(perf_data)
            label = f"{phase_name} {brain_region} - {perf_mean:.2f} ±{perf_sem:.2f}\n" \
                    f"p-val (corrected)={p_val_adj:.4f}\n" \
                    f"(n={num_subjects} subjects - " \
                    f"{opto_count:,} opto\n{control_count:,} Control trials)"
            # Bar
            br_clr = (BRC[BrainRegion.M2_Bi] if brain_region == "MFC" else
                      BRC[BrainRegion.ALM_Bi])
            br_offfset = 0 if brain_region == "MFC" else 1
            ax.bar(x_offset + br_offfset, perf_mean, yerr=perf_sem,
                   label=label, color=br_clr)
            # Scatter
            ax.scatter(np.ones(len(perf_data))*(x_offset + br_offfset),
                       perf_data, edgecolor='k',
                       facecolor='white', alpha=0.8)
            # Annotate each point with subject name
            # for i, subject_name in enumerate(perf_data.index):
            #     ax.annotate(subject_name,
            #                 (x_offset + br_offfset, perf_data.iloc[i]),
            #                 textcoords="offset points", xytext=(0,5),
            #                 ha='center', fontsize=8)
            print(f"{label} - Corrected P-value: {p_val_adj:.4f}")
            y_max = perf_data.max()
            region_max_y = max(region_max_y, y_max)
            if p_val_adj < 0.05:
                # Add significance star
                y, h, col = y_max, y_max*0.05, 'k'
                # ax.plot([x_offset + br_offfset - 0.2, x_offset + br_offfset + 0.2],
                #         [y, y], lw=1.5, c=col)
                sgf_str = "***" if p_val_adj < 0.001 else ("**" if p_val_adj < 0.01 else "*")
                ax.text(x_offset + br_offfset, y + h, sgf_str, ha='center', va='bottom',
                        color=col, fontsize=16)
        cross_region_pval = res_stats_dict[(phase_name, "crossregion",
                                            "p_holm_across_phases")]
        print(f"{phase_name} - MFC vs LFC - Corrected P-value: {cross_region_pval:.4f}")
        if cross_region_pval < 0.05:
            # Add significance star
            y_max = region_max_y
            y, h, col = y_max, y_max*0.05, 'k'
            # Draw arrow between the two bars
            props = {'connectionstyle':'bar','arrowstyle':'-',
                        'shrinkA':.1,'shrinkB':.1,'linewidth':1}
            ax.annotate('',
                        xy=(x_offset, y),
                        xytext=(x_offset + 1, y),
                        arrowprops=props)
            sgf_str = "***" if cross_region_pval < 0.001 else ("**" if cross_region_pval < 0.01 else "*")
            ax.text(0.5 + x_offset, y + h + 5, sgf_str, ha='center', va='bottom',
                    color=col, fontsize=16)

    ax.set_xticks([0, 1, 3, 4])
    ax.set_xticklabels(["Early - MFC", "Early - LFC",
                        "Late - MFC", "Late - LFC"])
    ax.axhline(0, color='black', linestyle='--')
    ax.set_ylabel(f"Effect Size ({effect_size_metric.name})")
    ax.set_title(f"{subject_name} - Early/Late Opto MFC vs LFC Effect Size\n"
                 f"num-iterations={num_iterations:,}")
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.legend(loc="upper center", fontsize="small")
    if save_figs:
        save_fp = f"opto_effect_{subject_name}_MFC_LFC.svg"
        save_fp = save_fp.replace(" ", "_").replace(":", "_").replace(
                                                                "=", "_")
        fig.savefig(f"{save_prefix}{save_fp}")
    plt.show()
    return used_df

def _processBrainRegionOptoConfig(loop, df_col_name,
                                  mean_or_median : Literal["mean", "median"],
                                  effect_size_metric : EffectSizeMetric,
                                  #min_sess_perf
                                  num_iterations : int,
                                  ):
    meanFn = np.mean if mean_or_median == "mean" else np.median
    grps_cntrl_means = []
    grps_opto_means = []
    effect_size_dict = {}
    used_subjects, used_sessions = set(), set()
    opto_trials_count, cntrl_trials_count = 0, 0

    from . import bootstrapping
    df_li = []
    partialAvgEffectSize = partial(_calcAvgEffectSize,
                                   effect_size_metric=effect_size_metric,
                                   meanFn=meanFn)

    for grp_key, grp_df in loop:
        print("Grp Key:", grp_key, "- len:", len(grp_df),
              " - Opto count:", (grp_df.OptoEnabled == 1).sum())
        assert grp_df[df_col_name].nunique() <= 2, (
                                    "Only binary values are implemented",
                                    f"Found: {grp_df[df_col_name].unique()}")
        control_trials = grp_df[grp_df.OptoEnabled == 0]
        opto_trials = grp_df[grp_df.OptoEnabled == 1]

        assert len(control_trials)
        assert len(opto_trials)
        observed_effect_size = partialAvgEffectSize(control_trials[df_col_name],
                                                    opto_trials[df_col_name])
        # print("Observed Effect Size:", observed_effect_size)
        control_trials_mean = meanFn(control_trials[df_col_name])
        opto_trials_mean = meanFn(opto_trials[df_col_name])
        grps_cntrl_means.append(control_trials_mean)
        grps_opto_means.append(opto_trials_mean)
        effect_size_dict[grp_key] = observed_effect_size

        # Bookkeeping
        used_subjects.add(grp_df.Name.iloc[0])
        uniq_sessions = {tuple(row)
                         for row in grp_df[["Name", "Date", "SessionNum"]
                                           ].drop_duplicates().values}
        used_sessions.update(uniq_sessions)
        opto_trials_count += len(opto_trials)
        cntrl_trials_count += len(control_trials)
        df_li.append(grp_df)



    # statistic, pval = stats.ttest_rel(grps_cntrl_means, grps_opto_means)
    df_all = pd.concat(df_li)
    df = df_all[df_all[df_col_name].notnull()].copy()
    # Rename session to unique session ID
    df["SessId"] = df.apply(lambda row: f"{row.Name}_{row.Date}_{row.SessionNum}",
                            axis=1)
    sub_arr = df.Name.unique()
    subj_to_sess_id_arr_dict = {sub: df[df.Name == sub].SessId.unique()
                                for sub in sub_arr}
    sess_id_to_trial_outcome_arr_dict = {
                           sess_id: df[df.SessId == sess_id][df_col_name].values
                            for sess_id in df.SessId.unique()}
    sess_id_to_trial_label_arr_dict = {
                        sess_id: df[df.SessId == sess_id].OptoEnabled.values
                        for sess_id in df.SessId.unique()}
    diff_arr = bootstrapping.bootstrapPerf(
        subj_arr=sub_arr,
        subj_to_sess_id_arr_dict=subj_to_sess_id_arr_dict,
        sess_id_to_trial_outcome_arr_dict=sess_id_to_trial_outcome_arr_dict,
        sess_id_to_trial_label_arr_dict=sess_id_to_trial_label_arr_dict,
        calcPerfFn=partialAvgEffectSize,
        num_iterations=num_iterations)

    diff_arr = diff_arr[~np.isnan(diff_arr)]
    # mean_diff = np.mean(diff_arr)
    # std_err = np.std(diff_arr)
    # confidence_interval = (np.percentile(diff_arr, 2.5),
    #                        np.percentile(diff_arr, 97.5))
    # print(f"\nMean performance difference: {mean_diff:.3f}")
    # print(f"Standard error: {std_err:.3f}")
    # print(f"Confidence interval: {confidence_interval}")
    # print("Effect Size:", observed_effect_size)
    # pval = stats.ttest_1samp(diff_arr, observed_effect_size).pvalue
    # From Svoboda and Brody's papers:
    # The p value was the proportion of iterations in which the sign of the
    # effect was different than the sign calculated using all the data.
    # Because this is a two-sided test
    control_all = df[df.OptoEnabled == 0][df_col_name].values
    opto_all = df[df.OptoEnabled == 1][df_col_name].values
    observed_effect_size = partialAvgEffectSize(control_all, opto_all)
    if observed_effect_size > 0:
        pval = (diff_arr < 0).sum() / len(diff_arr)
    else:
        pval = (diff_arr > 0).sum() / len(diff_arr)
    pval = min(pval*2, 1.0)  # Two-tailed
    print(f"P-value: {pval:.3f}")

    return effect_size_dict, pval, (
        used_subjects, used_sessions, opto_trials_count, cntrl_trials_count,
        df_all)

def _calcAvgEffectSize(control_trials, opto_trials,
                       effect_size_metric : EffectSizeMetric,
                       meanFn : callable):
    control_trials_mean = meanFn(control_trials)
    opto_trials_mean = meanFn(opto_trials)
    return _calcEffectSize(control_trials_mean, opto_trials_mean,
                           effect_size_metric)

def _calcEffectSize(control_trials_mean, opto_trials_mean,
                    effect_size_metric : EffectSizeMetric):
    if effect_size_metric == EffectSizeMetric.Gain:
        effect_size = 100*((opto_trials_mean/control_trials_mean)-1)
        effect_size *= -1 # Convert to drop
        # effect_size = 100*(control_trials_mean - opto_trials_mean)
    elif effect_size_metric == EffectSizeMetric.DistanceFromRandom:
        # e.g if opto is equal to control then effect size is 0, if opto is
        # at chance level (50%) then effect size is 100%
        effect_size = 100*abs(opto_trials_mean-0.5)/(control_trials_mean-0.5)
        effect_size = 100 - effect_size # Convert to distance from random
    else:
        assert False, f"Didn't implement: {effect_size_metric}"
    return effect_size


def _loopBrainRegionOptoConfig(br_df,  split_level : SplitLevelBy,
                               min_choice_trials : int):
    # region_color = BRC[brain_region]
    # print("_+_+_+_+_+_+ Processing: ", brain_region,
    #       "Cur x pos: ", cur_x_pos, "Color:", region_color)
    # print("********* Processing: ", brain_region, "- State:", start_state,
    #       "- Offset:", start_offset, "- Dur:", dur)
    # print("Num animals:", len(br_df.Name.unique()),
    #       "- Num sessions:", len(grpBySess(br_df)),
    #       "- Num trials:", len(br_df),
    #       "- Num opto trials:", (br_df.OptoEnabled == 1).sum())

    # pvals will be used later after the loop finishes
    if split_level == SplitLevelBy.AllAnimals:
        grp_by = [("All Subjects", br_df)]
    elif split_level == SplitLevelBy.Animal:
        grp_by = br_df.groupby("Name")
    elif split_level == SplitLevelBy.Session:
        grp_by = br_df.groupby(["Name", "Date", "SessionNum"])


    for grp_key, grp_df in grp_by:
        if not _shouldProcess(grp_df, min_choice_trials):
            continue
        yield grp_key, grp_df

def _shouldProcess(grp_df, min_choice_trials):
    control_trials = grp_df[grp_df.OptoEnabled == 0]
    opto_trials = grp_df[grp_df.OptoEnabled == 1]
    if len(control_trials) < min_choice_trials or len(opto_trials) < min_choice_trials:
        return False
    return True
