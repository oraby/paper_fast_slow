'''Which trials a neuron was active in, and where it fired (Figure S9A).

Backend for the "Sequence Extracation" section of ``2pAnalysis.ipynb`` and the
matching cell of ``TwoPLoad.ipynb``, which carried near-identical copies of
these ten functions. Both now import this one; the differences between the
copies are noted below.

**How a trial counts as active.** For one neuron, every trial's trace is
smoothed (Gaussian, ``smooth_sigma``) and reduced to its standard deviation.
The threshold is that neuron's **own** distribution: the
``lowest_percentile``-th percentile of those standard deviations, multiplied by
``above_lowest_percentile_zscores``. A trial is active when its smoothed trace
varies more than that. So the criterion is per neuron and relative — a quiet
neuron is not held to a loud neuron's bar — and `prcnt_valid` is the share of
that neuron's trials which cleared it.

**What comes out.** :class:`StdDistCollector` gathers one row per neuron:
``prcnt_valid``, the peak position and value of each active trial
(``max_idxs``, ``max_vals``), their spread (``max_idxs_std``), the trials
themselves, and the threshold used. Those rows are the ``max_firing_*`` frames
the rest of the analysis is built on, including Figure 4G's active sets.

**Sampling is random but seeded.** ``loopSessionsExampleNeurons`` picks neurons
with ``np.random.random() < random_fraction``, so a fraction below 1 needs the
``seed`` argument to be repeatable; both notebooks pass ``random_fraction=1``
(every neuron) with ``seed=1``.

Differences between the two copies, resolved here:

- **Trace window.** TwoPLoad's copy slices ``trace[trace_start_idx:
  trace_end_idx + 1]``; 2pAnalysis's used the whole array. Its frame is stored
  pre-cut (``sole_owner``, start 0, end ``len - 1``), so the slice is a no-op
  there and the newer form is kept.
- **Smoothing sigma** was a module-level global that had to be set before use
  (with an assert to catch forgetting); it is now a constructor argument,
  defaulting to the 1 both notebooks used.
- **``disable_first_neg_deflect``**, which flattens a trace's initial decay
  before measuring it, exists only in TwoPLoad's copy. Kept, default off.
- **``ChoiceCorrect`` and ``std_threshold``** are collected only by TwoPLoad's
  copy. Kept for both: extra columns are harmless, and Figure 6B needs them.
- **``plotMaxFiring`` could not run in TwoPLoad's copy**: it unpacked seven
  values from a tuple that had grown to nine. The results are a named tuple
  now, so that cannot recur.
'''
from __future__ import annotations

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.auto import tqdm

from ..common.definitions import BrainRegion
from ..pipeline.utils import filterNanGaussianConserving
from .shorth import shorth

#: The threshold is this percentile of a neuron's own per-trial spread ...
DEFAULT_LOWEST_PERCENTILE = 5
#: ... times this.
DEFAULT_ABOVE_LOWEST_PERCENTILE_ZSCORES = 3
#: Share of peaks the shorth interval in ``plotMaxFiring`` must cover.
DEFAULT_DATA_FRACTION = 0.8
#: Both notebooks smooth with sigma 1 before measuring spread.
DEFAULT_SMOOTH_SIGMA = 1


class TraceThresholds(NamedTuple):
    threshold: float
    lower_percentile_std: float
    all_traces_raw: object
    all_traces_smoothed: object
    all_traces_std: np.ndarray
    trial_numbers: np.ndarray


class TraceActivity(NamedTuple):
    all_traces_smoothed: object
    trials_numbers: np.ndarray
    valid_trace_smoothed: object
    valid_traces_idxs: np.ndarray
    max_vals: np.ndarray
    max_idxs: np.ndarray
    prcnt_valid: float
    invalid_trace_smoothed: object
    threshold: float


class DecideNeurons:
    '''Per-neuron activity threshold, from that neuron's own trial spread.'''

    def __init__(self, smooth_sigma=DEFAULT_SMOOTH_SIGMA,
                 lowest_percentile=DEFAULT_LOWEST_PERCENTILE,
                 above_lowest_percentile_zscores=DEFAULT_ABOVE_LOWEST_PERCENTILE_ZSCORES):
        self._smooth_sigma = smooth_sigma
        self._lowest_percentile = lowest_percentile
        self._above_lowest_percentile_zscores = above_lowest_percentile_zscores
        self._threshold = None

    def getThresholds(self, df, trace_id, disable_first_neg_deflect=False):
        all_traces_raw, all_traces_smoothed, all_traces_std, trial_numbers = [], [], [], []
        assert df.TrialNumber.nunique() == len(df)
        for _, row in df.iterrows():
            trace = row["traces_sets"]["neuronal"][trace_id]
            trace = trace[row.trace_start_idx:row.trace_end_idx + 1]
            if disable_first_neg_deflect:
                trace = self._flattenLeadingDecay(trace)
            smoothed_trace = filterNanGaussianConserving(
                trace, sigma=self._smooth_sigma, axis=0)
            all_traces_raw.append(trace)
            all_traces_smoothed.append(smoothed_trace)
            all_traces_std.append(smoothed_trace.std())
            trial_numbers.append(row.TrialNumber)

        all_traces_std = np.array(all_traces_std)
        lower_percentile_std = np.percentile(all_traces_std, self._lowest_percentile)
        self._threshold = lower_percentile_std * self._above_lowest_percentile_zscores
        try:                    # fails when the trials are of unequal length
            all_traces_raw = np.array(all_traces_raw)
        except ValueError:
            pass
        else:
            all_traces_smoothed = np.array(all_traces_smoothed)
        return TraceThresholds(self._threshold, lower_percentile_std,
                               all_traces_raw, all_traces_smoothed,
                               all_traces_std, np.array(trial_numbers))

    @staticmethod
    def _flattenLeadingDecay(trace):
        '''Hold the trace flat until it first turns upwards.'''
        trace_mod = filterNanGaussianConserving(trace.copy(), sigma=2, axis=0)
        first_pos_deflect = np.where(np.diff(trace_mod) > 0)[0]
        first_pos_deflect = first_pos_deflect[0] if len(first_pos_deflect) else 0
        trace = trace.copy()
        trace[:first_pos_deflect] = trace[first_pos_deflect]
        return trace

    def isAccepted(self, trace):
        smoothed_trace = filterNanGaussianConserving(trace, sigma=1, axis=0)
        return smoothed_trace.std() > self._threshold, smoothed_trace


def setupEpochAxes(axs, row):
    '''Mark the epoch boundaries of ``row`` on each axis.'''
    for ax in axs:
        [ax.axvline(rng[0], ls="--", c="gray") for rng in row.epochs_ranges[1:]]
        ax.set_xticks([rng[0] for rng in row.epochs_ranges[1:]])
        ax.set_xticklabels([name for name in row.epochs_names[1:]])
        [ax.spines[_dir].set_visible(False) for _dir in ["left", "right", "top"]]
        ax.set_xlim(left=0)


def traceValidInvalid(trace_id, df, df_query="", decide_neurons=None,
                      disable_first_neg_deflect=False):
    '''Split one neuron's trials into active and inactive.'''
    decide_neurons = decide_neurons or DecideNeurons()
    (threshold, lower_percentile_std, all_traces_raw, all_traces_smoothed,
     all_traces_std, trials_numbers) = decide_neurons.getThresholds(
        df, trace_id, disable_first_neg_deflect=disable_first_neg_deflect)

    valid_traces_idxs = all_traces_std > threshold
    invalid_traces_idxs = ~valid_traces_idxs
    if len(df_query):
        assert all(df.TrialNumber == trials_numbers), "should be the same"
        df = df.query(df_query)
        new_valid_trial_idxs = np.isin(trials_numbers, df.TrialNumber)
        valid_traces_idxs &= new_valid_trial_idxs
        invalid_traces_idxs &= new_valid_trial_idxs

    if isinstance(all_traces_smoothed, np.ndarray):
        valid_trace_smoothed = all_traces_smoothed[valid_traces_idxs]
        invalid_trace_smoothed = all_traces_smoothed[invalid_traces_idxs]
        max_idxs = np.argmax(valid_trace_smoothed, axis=1)
        max_vals = np.amax(valid_trace_smoothed, axis=1)
    else:                       # unequal trial lengths: no rectangular array
        valid_trace_smoothed = [t for t, keep in zip(all_traces_smoothed,
                                                     valid_traces_idxs) if keep]
        invalid_trace_smoothed = [t for t, keep in zip(all_traces_smoothed,
                                                       invalid_traces_idxs) if keep]
        max_idxs = np.array([np.argmax(t) for t in valid_trace_smoothed])
        max_vals = np.array([np.amax(t) for t in valid_trace_smoothed])
    assert len(max_idxs) == len(valid_trace_smoothed)
    prcnt_valid = 100 * len(valid_trace_smoothed) / len(df)
    return TraceActivity(all_traces_smoothed, trials_numbers,
                         valid_trace_smoothed, valid_traces_idxs, max_vals,
                         max_idxs, prcnt_valid, invalid_trace_smoothed, threshold)


def makeResDict():
    return {"trace_id": [], "ShortName": [], "BrainRegion": [],
            "max_idxs_std": [], "prcnt_valid": [], "len_valid": [],
            "len_all": [], "max_idxs": [], "max_vals": [], "ChoiceLeft": [],
            "ChoiceCorrect": [], "active_trial_numbers": [],
            "active_quantile_idx": [], "quantile_idxs_count": [],
            "whole_trace": [], "invalid_whole_trace": [], "std_threshold": []}


class StdDistCollector:
    '''Gathers one row per neuron; pass :meth:`track` as the loop's ``processFn``.

    Replaces the notebooks' module-level ``res_dict`` that ``trackStdDist``
    appended to, which meant every run had to remember to reset it first.
    '''

    def __init__(self, decide_neurons=None, disable_first_neg_deflect=False):
        self.decide_neurons = decide_neurons or DecideNeurons()
        self.disable_first_neg_deflect = disable_first_neg_deflect
        self.res_dict = makeResDict()

    def track(self, trace_id, df, df_query="", pdf=None):
        res = traceValidInvalid(
            trace_id, df, df_query, decide_neurons=self.decide_neurons,
            disable_first_neg_deflect=self.disable_first_neg_deflect)
        assert all(df.TrialNumber == res.trials_numbers)
        valid_df = df[res.valid_traces_idxs]
        if "quantile_idx" in df.columns:
            active_quantile_idxs = np.array(valid_df.quantile_idx)
            quantile_idxs_count = df.groupby("quantile_idx").apply(len).to_dict()
        else:
            active_quantile_idxs = np.nan
            quantile_idxs_count = np.nan
        row = {
            "trace_id": trace_id,
            "ShortName": df.iloc[0].ShortName,
            "BrainRegion": df.iloc[0].BrainRegion,
            "max_idxs_std": res.max_idxs.std() if len(res.max_idxs) > 1 else np.nan,
            "prcnt_valid": res.prcnt_valid,
            "len_valid": len(res.max_idxs),
            "len_all": len(df),
            "max_idxs": res.max_idxs,
            "max_vals": res.max_vals,
            "ChoiceLeft": np.array(valid_df.ChoiceLeft),
            "ChoiceCorrect": np.array(valid_df.ChoiceCorrect),
            "active_trial_numbers": valid_df.TrialNumber.values,
            "active_quantile_idx": active_quantile_idxs,
            "quantile_idxs_count": quantile_idxs_count,
            "whole_trace": res.valid_trace_smoothed,
            "invalid_whole_trace": res.invalid_trace_smoothed,
            "std_threshold": res.threshold,
        }
        for key, value in row.items():
            self.res_dict[key].append(value)

    def frame(self):
        return pd.DataFrame(self.res_dict)


def plotMaxFiring(trace_id, df, df_query="", pdf=None, decide_neurons=None,
                  data_fraction=DEFAULT_DATA_FRACTION):
    '''One neuron's trials, with a histogram of peak positions above them.'''
    res = traceValidInvalid(trace_id, df, df_query, decide_neurons=decide_neurons)
    color = 'gray' if res.prcnt_valid < 10 else "g"

    fig, ax = plt.subplots(figsize=(10, 8))
    x = np.arange(len(res.all_traces_smoothed[0]) + 1)
    valid_traces = res.all_traces_smoothed[res.valid_traces_idxs]
    invalid_traces = res.all_traces_smoothed[~res.valid_traces_idxs]
    for trace in valid_traces:
        ax.plot(x[:-1], trace)
    for trace in invalid_traces:
        ax.plot(x[:-1], trace, c="gray", alpha=0.3)
    if len(res.max_idxs):
        bins_vals, _ = np.histogram(res.max_idxs, bins=x)
        bins_vals = bins_vals.astype(float)
        bins_vals /= bins_vals.max()
        max_y = valid_traces.max()
        bins_vals *= (max_y - valid_traces.min()) * 0.4
        ax.bar(x[:-1], bins_vals, bottom=max_y, width=1, color=color)
        if len(res.max_idxs) > 2:
            distance, short_rng = shorth(res.max_idxs, fraction=data_fraction,
                                         return_range=True)
            distance = 100 * distance / len(trace)   # every trial is the same length
            max_hist = bins_vals.max() + max_y
            if distance == 0:
                short_rng = short_rng[0], short_rng[1] + 1
            ax.fill_between(short_rng, max_y, max_hist, color="yellow", alpha=0.4)
        else:
            distance = np.nan
        distance = f"{distance:.2g}"
        std = f"{valid_traces.std():.2f}"
    else:
        std = distance = "(Not enough data)"
    ax.set_title(f"{trace_id} - {res.prcnt_valid:.2g}% ({len(valid_traces)}/{len(df)})"
                 f" Trials - Std.={std}"
                 f"\nMin-distance convering {100 * data_fraction:.2g}%={distance}%"
                 " of sampling time"
                 + ("" if not len(df_query) else f"\n{df_query}"))
    setupEpochAxes([ax], df.iloc[0])
    _finishFigure(fig, pdf)


def plotNeuron(trace_id, df, df_query="", pdf=None, decide_neurons=None):
    '''One neuron's raw and smoothed trials, split by choice, plus its threshold.'''
    decide_neurons = decide_neurons or DecideNeurons()
    fig, (ax_raw, ax_smooth, ax_hist) = plt.subplots(1, 3, figsize=(20, 5))
    col = "ChoiceLeft"
    df = df[df[col].notnull()]
    cross_thresh_count = 0
    if len(df_query):
        df = df.query(df_query)
    thresholds = decide_neurons.getThresholds(df, trace_id)
    threshold = thresholds.threshold
    all_traces_std = thresholds.all_traces_std
    bins = np.linspace(all_traces_std.min(), all_traces_std.max(), num=10)
    ax_hist.hist([all_traces_std[all_traces_std <= threshold],
                  all_traces_std[all_traces_std > threshold]],
                 histtype="barstacked", bins=bins, color=["gray", "green"])
    ax_hist.axvline(thresholds.lower_percentile_std, ls="--", c="k")
    ax_hist.axvline(threshold, ls="--", c="k")

    traces_left, traces_smoothed_left = [], []
    traces_right, traces_smoothed_right = [], []
    for _, row in df.iterrows():
        trace = row["traces_sets"]["neuronal"][trace_id]
        is_accepted, smoothed_trace = decide_neurons.isAccepted(trace)
        if is_accepted:
            cross_thresh_count += 1
            if row[col] == 1:
                c = "g"
                traces_smoothed_left.append(smoothed_trace)
                traces_left.append(trace)
            else:
                c = "orange"
                traces_smoothed_right.append(smoothed_trace)
                traces_right.append(trace)
        else:
            c = "gray"
        x = np.arange(len(trace))
        ax_raw.plot(x, trace, c=c, lw=1, alpha=0.3)
        ax_smooth.plot(x, smoothed_trace, c=c, lw=1, alpha=0.3)

    for traces, smoothed, c in [(traces_left, traces_smoothed_left, "g"),
                                (traces_right, traces_smoothed_right, "orange")]:
        if len(traces):
            ax_raw.plot(x, np.array(traces).mean(axis=0), c=c, lw=5)
            ax_smooth.plot(x, np.array(smoothed).mean(axis=0), c=c, lw=5)

    prcnt_cross_thresh = 100 * cross_thresh_count / len(df)
    fig.suptitle(f"{trace_id} - %Trials: {prcnt_cross_thresh:.2g}% - "
                 f"({cross_thresh_count}/{len(df)} Trial) - Split criteria={col}")
    ax_raw.set_title("Raw traces")
    ax_smooth.set_title("Smoothed traces")
    ax_hist.set_title(f"Smoothed traces Std. hist\nThreshold="
                      f"{decide_neurons._above_lowest_percentile_zscores}*(lower "
                      f"{decide_neurons._lowest_percentile}% Std.="
                      f"{thresholds.lower_percentile_std:.2f})={threshold:.2f}")
    setupEpochAxes([ax_raw, ax_smooth], row)
    _finishFigure(fig, pdf)


def _finishFigure(fig, pdf):
    if pdf is not None:
        pdf.savefig(fig)
        plt.close(fig)
    else:
        plt.show()


def separateNeuron(trace_id, df, new_trace_id):
    '''The same trials, carrying only this neuron's trace.'''
    new_rows_li = []
    for _, row in df.iterrows():
        row = row.copy()
        row["traces_sets"] = {"neuronal":
                              {new_trace_id: row["traces_sets"]["neuronal"][trace_id]}}
        new_rows_li.append(row)
    return pd.DataFrame(new_rows_li)


def exampleNeuron(sess_df, processFn, df_query, pdf=None,
                  must_process_long_ids=set(), random_fraction=0):
    '''Run ``processFn`` on the session's neurons, named ``<session>_<id>``.'''
    ex_row = sess_df.iloc[0]
    sess_name = ex_row.ShortName
    long_traces_dict = {f"{sess_name}_{trace_id}": trace_id
                        for trace_id in ex_row["traces_sets"]["neuronal"]
                        if (f"{sess_name}_{trace_id}" in must_process_long_ids
                            or np.random.random() < random_fraction)}
    for long_trace_id, trace_id in long_traces_dict.items():
        only_neuron_df = separateNeuron(trace_id, sess_df,
                                        new_trace_id=long_trace_id)
        processFn(long_trace_id, only_neuron_df, pdf=pdf, df_query=df_query)
    return len(long_traces_dict)


def loopSessionsExampleNeurons(df, processFn, random_fraction, pdf_str, save_figs,
                               df_query="", seed=None, must_process_long_ids=set(),
                               fig_save_prefix=None):
    '''Applies processFn to either all or a subset of traces in df.'''
    if seed is not None:
        np.random.seed(seed)
    df = df[df.BrainRegion.isin([BrainRegion.M2_Bi, BrainRegion.ALM_Bi])]
    df = df[df.Layer == "L23"]
    for br, br_df in tqdm(df.groupby("BrainRegion")):
        br_str = str(BrainRegion(br)).split("_")[0]
        quantity = "many" if random_fraction < 1 else "all"
        if save_figs:
            assert fig_save_prefix is not None, "Pass fig_save_prefix to save"
            pdf = PdfPages(f'{fig_save_prefix}/SingleTraces/'
                           f'{pdf_str}_{quantity}_cells_{br_str}.pdf')
        else:
            pdf = None
        for sess, sess_df in tqdm(br_df.groupby("ShortName")):
            exampleNeuron(sess_df, processFn=processFn, pdf=pdf,
                          df_query=df_query, random_fraction=random_fraction,
                          must_process_long_ids=must_process_long_ids)
        if pdf is not None:
            pdf.close()
