'''Per-trial peak and area for every neuron (feeds Figures 4K, S10A-C, S11).

Backend for the "Correlation between rt and activity on single cell level"
section of ``plottraces3.ipynb``.

``loopNeurons`` reduces each (neuron, trial) to a handful of numbers: when the
neuron peaked, how high, and how much area its rises covered -- the last two
from :mod:`.traceauc`. Every panel about rt-versus-activity is built on the
table it returns.

**Peaks are reported in seconds from sampling onset**, not in frames, and each
trial's own duration travels with the row so the later panels can ask whether
a peak moved with the trial.

``only_active_trials`` decides whether a neuron's quiet trials are dropped
before any of this, and the two published tables differ on it:
``active_traces_max_df`` keeps **only the active trials** (``True``), which is
what the rt/activity correlations are measured on, while ``all_traces_max_df``
keeps every trial (``False``) for the summary panels. Passing the wrong one
changes every correlation downstream without changing the table's shape.

``loopTrialTimeActive`` then folds that per-trial table into one row per
(session, neuron) with the correlations the panels plot.

Extracted from the notebook unchanged, except that the notebook globals it
reached for -- ``KEY_PREFIX``, the sampling window and ``ACQ_RATE`` -- are
parameters or module constants.
'''
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, sem
from tqdm.auto import tqdm

from ..common.definitions import BrainRegion
from ..pipeline.utils import filterNanGaussianConserving
from .traceauc import getPosDeflections, getTraceAUC, trapezoid

#: Frames per second of the imaging.
ACQ_RATE = 30

def analyzeNeuron(neurons_trials_traces_dict, active_traces_row,
                  acq_rate, br, sess_id, trace_id,
                  only_active_trials):
    res = {"TrialNumber":[], "trial_dur_sec":[], "is_active":[],
           "ChoiceCorrect":[], "DVstr":[]}
    for filter_key in ["", "smoothed1_", "smoothed2_"]:
        res[filter_key + "max_pos_sec"] = []
        res[filter_key + "max_amplitude"] = []
        res[filter_key + "auc"] = []
        res[filter_key + "trace"] = []
    for (trial_num, trial_dur, choice_correct, dv_str), raw_trace in \
                                             neurons_trials_traces_dict.items():
        is_active = trial_num in active_traces_row["active_trials_nums"]
        if only_active_trials and not is_active:
            continue
        # trial_dur = len(trace)/acq_rate
        trace_smoothed1 = filterNanGaussianConserving(raw_trace, 1, axis=0)
        trace_smoothed2 = filterNanGaussianConserving(raw_trace, 2, axis=0)
        #
        res["TrialNumber"].append(trial_num)
        res["ChoiceCorrect"].append(choice_correct)
        res["DVstr"].append(dv_str)
        res["trial_dur_sec"].append(trial_dur)
        res["is_active"].append(is_active)
        #
        for prefix, trace in (("", raw_trace),
                               ("smoothed1_", trace_smoothed1),
                               ("smoothed2_", trace_smoothed2)):
            trace_max_pos_idx = np.argmax(trace)
            trace_max = trace[trace_max_pos_idx]
            pos_deflects_idxs = getPosDeflections(trace)
            integrate_val, *_ = getTraceAUC(trace, pos_deflects_idxs)
            res[f"{prefix}trace"].append(trace)
            res[f"{prefix}max_amplitude"].append(trace_max)
            res[f"{prefix}max_pos_sec"].append(trace_max_pos_idx/acq_rate)
            res[f"{prefix}auc"].append(integrate_val)

    res_df = pd.DataFrame(res)
    res_df["BrainRegion"] = br
    res_df["ShortName"] = sess_id
    res_df["trace_id"] = trace_id
    first_cols = ["BrainRegion", "ShortName", "TrialNumber", "trial_dur_sec",
                  "trace_id", "ChoiceCorrect", "DVstr"]
    rest_columns = [col for col in res_df.columns if col not in first_cols]
    res_df = res_df[first_cols + rest_columns]
    return res_df

def loopNeurons(df_unnormalized, active_traces_df, only_active_trials):
    res_df_li = []
    for br, br_df in df_unnormalized.groupby("BrainRegion"):
        for sess, sess_df in br_df.groupby("ShortName"):
            traces_trials_dict_dict = {}
            acq_rate = sess_df.acq_sampling_rate.iloc[0]
            active_traces_sess_df = active_traces_df[
                                             active_traces_df.ShortName == sess]
            done_once = False
            for trial_num, trial_row in sess_df.groupby("TrialNumber"):
                assert len(trial_row) == 1
                trial_row = trial_row.iloc[0]
                trial_dur = trial_row["calcStimulusTime"]
                choice_correct = trial_row["ChoiceCorrect"]
                dv_str = trial_row["DVstr"]
                traces_dict = trial_row.traces_sets["neuronal"]
                if not done_once:
                    for trace_id in traces_dict:
                        traces_trials_dict_dict[trace_id] = {}
                for trace_id, trace_val in traces_dict.items():
                    traces_trials_dict_dict[trace_id][
                     (trial_num, trial_dur, choice_correct, dv_str)] = trace_val
                done_once = True
            # Now loop from neurons prespective
            for trace_id, trial_nums_to_trial_traces_dict in \
                                                traces_trials_dict_dict.items():
                active_traces_row = active_traces_sess_df[
                                     active_traces_sess_df.trace_id == trace_id]
                assert len(active_traces_row) == 1
                active_traces_row = active_traces_row.iloc[0]
                res_df = analyzeNeuron(trial_nums_to_trial_traces_dict,
                                       active_traces_row,
                                       acq_rate=acq_rate, br=br, sess_id=sess,
                                       trace_id=trace_id,
                                       only_active_trials=only_active_trials)
                res_df_li.append(res_df)
    return pd.concat(res_df_li).reset_index(drop=True)


def getTrialTimeVsActiveCountvsMaxVal(sess_df, key_prefix):
    num_neurons = sess_df.trace_id.nunique()
    res_dict = {"TrialNumber":[], "trial_dur_sec":[], "prcnt_active":[],
                "active_max_amplitude_mean":[], "active_max_amplitude_sem":[]}
    for trial_num, trial_df in sess_df.groupby("TrialNumber"):
        assert len(trial_df) == num_neurons, f"{len(trial_df)} != {num_neurons}"
        active_neurons = trial_df[trial_df.is_active]
        active_neurons_prcnt = 100 *len(active_neurons) / num_neurons
        max_amplitude_mean = active_neurons[f"{key_prefix}max_amplitude_rank"].mean()
        # if len(active_neurons) > 1:
        max_amplitude_sem = active_neurons[f"{key_prefix}max_amplitude_rank"].sem()
        # else:
        #     max_amplitude_sem = 0
        res_dict["TrialNumber"].append(trial_num)
        res_dict["trial_dur_sec"].append(trial_df.trial_dur_sec.iloc[0])
        res_dict["prcnt_active"].append(active_neurons_prcnt)
        res_dict["active_max_amplitude_mean"].append(max_amplitude_mean)
        res_dict["active_max_amplitude_sem"].append(max_amplitude_sem)

    res_df = pd.DataFrame(res_dict).sort_values("trial_dur_sec")
    res_df["BrainRegion"] = sess_df.BrainRegion.iloc[0]
    res_df["ShortName"] = sess_df.ShortName.iloc[0]
    return res_df


def loopTrialTimeActive(active_traces_sess_df, key_prefix):
    fig, ax = plt.subplots(figsize=(10, 6))
    twinx_ax = ax.twinx()

    res_dfs = []
    for br, br_df in active_traces_sess_df.groupby("BrainRegion"):
        br_dfs = []
        for sess, sess_df in br_df.groupby("ShortName"):
            res_df = getTrialTimeVsActiveCountvsMaxVal(sess_df, key_prefix)
            br_dfs.append(res_df)
        res_dfs += br_dfs

        br_df = pd.concat(br_dfs)
        cut = pd.cut(br_df.trial_dur_sec, bins=np.arange(0.3, 3.1, .4))

        x, y2_mean, y2_sem, y1_mean, y1_sem = [], [], [], [], []
        for (name, df) in br_df.groupby(cut):
            mean_trial_dur = df.trial_dur_sec.mean()
            x.append(mean_trial_dur)
            sess_active = df.groupby("ShortName").prcnt_active.mean()
            y1_mean.append(sess_active.mean())
            y1_sem.append(sess_active.sem())
            sess_ampl = df.groupby("ShortName").active_max_amplitude_mean.mean()
            y2_mean.append(sess_ampl.mean())
            y2_sem.append(sess_ampl.sem())

        ax.errorbar(x, y1_mean, yerr=y1_sem,
                    label="Active Neurons %",
                    color=BRClr[br], ls="-")
        twinx_ax.errorbar(x, y2_mean, yerr=y2_sem,
                          label="Active Neurons Peak Rank",
                          color=BRClr[br], ls="--")
    ax.spines[["left", "top", "right"]].set_visible(False)
    twinx_ax.spines[["left", "top", "right"]].set_visible(False)
    ax.set_xlabel("Trial Duration (s)")
    h, l = ax.get_legend_handles_labels()
    h2, l2 = twinx_ax.get_legend_handles_labels()
    ax.legend(h + h2, l + l2, fontsize="x-small")
    plt.show()

    return pd.concat(res_dfs).reset_index(drop=True)

# import BRCLR
from ..common.clr import BrainRegion as BRClr
