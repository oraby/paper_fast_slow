'''How many neurons are active around a trial's sampling (Figures S11A, S11B).

Backend for the "Correlatoin between trial duration and number of active
neurons" section of ``plottraces3.ipynb``.

Each trial is reduced to a count: how many of the session's neurons peaked
inside a window around sampling. Two windows are used, and they are specified
differently on purpose:

- **early** (Figure S11A middle/right) -- a fixed window from
  ``TIME_BEFORE_SAMPLING`` before sampling onset to ``cut_after`` seconds
  after it, measured from the **start** of the trial;
- **late** (Figure S11B) -- the last ``cut_before`` **percent** of the trial
  plus ``TIME_AFTER_MOVEMENT`` seconds, measured from its **end**, so it
  scales with how long the animal sampled.

``cut_before_how`` / ``cut_after_how`` choose between seconds (``"FIX"``) and
a share of the trial (``"PRCNT"``), and ``cut_from`` which end the window is
anchored to. Cutting before the start in percent is refused: there is nothing
to take a percentage of yet.

Each row also carries ``total_count``, the session's neuron count, which is
what the percentages downstream are taken out of.

Extracted from the notebook unchanged, except that ``ACQ_RATE`` and
``TIME_BEFORE_SAMPLING`` are module constants and the debug ``display`` calls
are prints.
'''
from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .traceauc import getPosDeflections, getTraceAUC

#: Frames per second of the imaging; the notebook reads it from nowhere either.
ACQ_RATE = 30
#: Seconds of trace kept before sampling onset, as the frames were cut.
TIME_BEFORE_SAMPLING = 0.1
#: An example plot is drawn for every Nth trial, while the budget lasts.
SKIP_EVERY = 100


class ExamplePlotBudget:
    """How many example trials still get drawn, and how far apart.

    The notebook kept this as two module-level counters that the loop reached
    for with ``global``, and that each call had to reset by hand.
    """

    def __init__(self, count=0, skip_every=SKIP_EVERY):
        self.count = count
        self.skip_every = skip_every
        self._skip = 0

    def take(self):
        """True when this trial should be drawn."""
        if not self.count:
            return False
        if self._skip:
            self._skip -= 1
            return False
        self._skip = self.skip_every - 1
        self.count -= 1
        return True

def _commonNumActiveNeurons(trial_df,
                            cut_before, cut_before_how : Literal["FIX", "PRCNT"],
                            cut_after, cut_after_how : Literal["FIX", "PRCNT"],
                            cut_from : Literal["START", "END"],
                            key_prefix, total_neurons_count, budget=None):
# def numActiveNeurons(trial_df, look_back_sec, key_prefix, total_neurons_count):
    assert cut_before_how in ("FIX", "PRCNT")
    assert cut_after_how in ("FIX", "PRCNT")
    assert cut_from in ("START", "END")
    trial_dur_sec = trial_df.trial_dur_sec
    assert trial_dur_sec.nunique() == 1
    trial_dur_sec = trial_dur_sec.iloc[0]
    max_pos_sec = trial_df[key_prefix + "max_pos_sec"] - TIME_BEFORE_SAMPLING
    # cut_off = trial_dur_sec - look_back_sec
    if cut_from == "START":
        assert cut_before_how == "FIX", "Can't cut before start with trial-dur %"
        cut_off_min = -cut_before
        if cut_after_how == "FIX":
            cut_off_max = cut_after
        else:
            cut_off_max = trial_dur_sec*cut_after
    else:
        if cut_before_how == "FIX":
            cut_off_min = trial_dur_sec - cut_before
        else:
            cut_off_min = trial_dur_sec*(1-cut_before)
        assert cut_after_how == "FIX", "Can't cut after end with trial-dur %"
        cut_off_max = trial_dur_sec + cut_after

    max_within_roi_sec = ((max_pos_sec >= cut_off_min) &
                          (max_pos_sec <= cut_off_max))
    num_active_neurons = len(trial_df)
    num_active_within_roi = max_within_roi_sec.sum()
    num_extend_active = 0
    # max_within_first_sec = ((max_pos_sec >= cut_off_min) &
    #                         (max_pos_sec <= cut_off_max))
    # num_active_within_first_sec = max_within_first_sec.sum()

    traces_to_plot = []
    trace_threshs = []
    integrate_masks = []
    is_valids = []
    ex_trace = trial_df.trace.iloc[0]
    mask_start_idx = int(round((TIME_BEFORE_SAMPLING + cut_off_min)*ACQ_RATE))
    mask_end_idx = int(round((TIME_BEFORE_SAMPLING +  cut_off_max)*ACQ_RATE))
    mask_mask = np.full_like(ex_trace, fill_value=False, dtype=bool)
    mask_mask[mask_start_idx:mask_end_idx + 1] = True
    # print(mask_start_idx, mask_end_idx, "Mask mask:", mask_mask)

    for row_idx, row in trial_df.iterrows():
        trace = row[key_prefix + "trace"]
        is_valid = max_within_roi_sec.loc[row_idx]
        traces_to_plot.append(trace)
        pos_deflection_idx = getPosDeflections(trace)
        # integrated_auc, trace_thresh, integrate_mask = _getTraceAUC(trace)
        # integrate_mask &= mask_mask
        (integrated_auc, traces_thresh_li, traces_rng_li,
         traces_min_max_idxs_li, masks_li) = getTraceAUC(trace,
                                                          pos_deflection_idx)
        integrate_mask = mask_mask.copy()
        for rng, sub_mask in zip(traces_rng_li, masks_li):
            integrate_mask[rng[0]:rng[1]] &= sub_mask
        if any(integrate_mask):# and not is_valid:
            num_extend_active += 1
        if len(traces_thresh_li):
            trace_thresh = max(traces_thresh_li) # TODO: Re-plot
        else:
            trace_thresh = np.nan
        trace_threshs.append(trace_thresh)
        integrate_masks.append(integrate_mask)
        is_valids.append(is_valid)
    # print(trial_df)
    # raise

    DEBUG = False
    if DEBUG:
        trial_df = trial_df.copy()
        trial_df["valid"] = max_within_roi_sec
        trial_df["org_time"] = trial_df[key_prefix + "max_pos_sec"]
        trial_df["corrected_time"] = max_pos_sec
        print(trial_df[["org_time", "corrected_time", "trial_dur_sec", "valid"]])
        raise

    if budget is not None and budget.take():
        if True:
            fig, ax = plt.subplots()
            rng = (np.arange(len(ex_trace))/ACQ_RATE) - TIME_BEFORE_SAMPLING
            clr_c = 0
            for trace, trace_thresh, integrate_mask, is_valid in zip(
                traces_to_plot, trace_threshs, integrate_masks, is_valids):
                ax.plot(rng, trace, c=f"C{clr_c}")
                if not is_valid:
                    ax.fill_between(rng[integrate_mask], trace_thresh, trace[integrate_mask],
                                    alpha=0.3, color=f"C{clr_c}")
                clr_c += 1
            [ax.axvline(t, ls="--", c="gray") for t in (0, trial_dur_sec)]
            ax.scatter(max_pos_sec, trial_df[key_prefix + "max_amplitude"])
            ax.axvline(cut_off_min, c="r", ls="--")
            ax.axvline(cut_off_max, c="r", ls="--")
            ax.set_xlabel("Sampling Time (s)")
            ax.set_ylabel("Neurons DF/F")
            ax.spines[["top", "right", "left"]].set_visible(False)

            cut_before_str = f"{cut_before}s" if cut_before_how == "FIX" else \
                             f"{int(100*cut_before)}%"

            cut_after_str = f"{cut_after}s" if cut_after_how == "FIX" else \
                            f"{int(100*cut_after)}%"
            epoch = "from sampling start" if cut_from == "EARLY" else \
                    "from movement start"
            max_ampl_str = f"Max-Amplitude between -{cut_before_str} " \
                            f"to {cut_after_str} from {epoch}"
            ax.set_title(f"{trial_df.ShortName.iloc[0]} - "
                        f"Trial #{trial_df.TrialNumber.iloc[0]} - "
                        f"Trial Dur: {trial_dur_sec:.2f}s - Cut off: {cut_off_min:.2f}s -> {cut_off_max:.2f}\n"
                        f"Session Traces count: {total_neurons_count} - "
                        f"Trial Active count: {num_active_neurons} - "
                        f"{max_ampl_str} Count: {num_active_within_roi}\n"
                        f"Extended Active Count: {num_extend_active}")
            plt.show()

    return trial_dur_sec, num_active_neurons, num_active_within_roi, num_extend_active

def corrTrialDurActivityCommon(df,
                               cut_before, cut_before_how : Literal["FIX", "PRCNT"],
                               cut_after, cut_after_how : Literal["FIX", "PRCNT"],
                               cut_from : Literal["START", "END"],
                               key_prefix, plot_count=0):

    budget = ExamplePlotBudget(plot_count)
    start_end_str = "start" if cut_from == "START" else "end"
    res_dict = {"BrainRegion":[], "ShortName":[], "trial_number":[],
                "ChoiceCorrect":[], "DVstr":[],
                "trial_dur_sec":[], "total_count":[], "trial_active_count":[],
                f"trial_{start_end_str}_max_active_count":[],
                f"trial_{start_end_str}_extend_active_count":[]}
    for br, br_df in df.groupby("BrainRegion"):
        for sess, sess_df in br_df.groupby("ShortName"):
            total_count = sess_df.trace_id.nunique()
            for trial_num, trial_df in sess_df.groupby("TrialNumber"):
                (trial_dur_sec, trial_active_count, trial_end_max_active_count,
                 trial_end_extend_active_count) = \
                           _commonNumActiveNeurons(trial_df,
                                                   cut_before=cut_before,
                                                   cut_before_how=cut_before_how,
                                                   cut_after=cut_after,
                                                   cut_after_how=cut_after_how,
                                                   cut_from=cut_from,
                                                   key_prefix=key_prefix,
                                                   total_neurons_count=total_count,
                                                   budget=budget)
                res_dict["BrainRegion"].append(br)
                res_dict["ShortName"].append(sess)
                res_dict["trial_number"].append(trial_num)
                res_dict["ChoiceCorrect"].append(trial_df.ChoiceCorrect.iloc[0])
                res_dict["DVstr"].append(trial_df.DVstr.iloc[0])
                res_dict["trial_dur_sec"].append(trial_dur_sec)
                res_dict["total_count"].append(total_count)
                res_dict["trial_active_count"].append(trial_active_count)
                res_dict[f"trial_{start_end_str}_max_active_count"].append(trial_end_max_active_count)
                res_dict[f"trial_{start_end_str}_extend_active_count"].append(trial_end_extend_active_count)

    res_df = pd.DataFrame(res_dict)
    return res_df


# cut_off_min = -TIME_BEFORE_SAMPLING, cut_before_how = "FIX",
# cut_off_max = start_sampling_cutoff, cut_after_how = "FIX",
