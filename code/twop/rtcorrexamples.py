'''The example neurons of Figures S10A and S10B, with their peak/AUC calculation.

Backend for the "Example neurons with their peak-time and AUC correlations --
Figures S10A–B" cell of ``2p_1_population.ipynb``.

These two panels compute the AUC as below. The population panels (4K right,
S10C–D) use ``traceauc.getTraceAUC`` instead, which integrates each positive
deflection above 80% of its rise.

- ``s10TraceAUC`` -- trapezoid sum of the samples at or above
  min + 0.5 * (max - min) of the whole trace, or, given rising segments, of
  those segments.
- ``startsOnDecay`` -- whether a trial's maximum is in its first two frames
  with no later rise of at least 20% of the trace's range; if there is such a
  rise, where the peak moves to.
- ``exampleTrials`` -- one row per active trial: duration, colour, whether it
  starts on a decay, peak time (s from sampling onset) and AUC.
- ``plotS10Example`` -- single-trial traces (sampling- or movement-aligned)
  with a Fast/Slow colour key, then peak time and AUC against sampling duration with their regression
  lines, written to
  ``{fig_save_prefix}/SingleTraces/rt_corr_examples/{MFC|LFC}/{session}_{id}_{panel}.svg``.

With the published neurons this reproduces the r, r², p and slope printed
on both panels.
'''
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from .rtcorrpanels import (TIME_AFTER_MOVEMENT, TIME_BEFORE_SAMPLING,
                           FastSlow_CM, _plotCorr, fastSlowColors,
                           samplingAlignedSavePath)

#: (panel, session, neuron id, alignment of the traces panel). The same two
#: neurons as Figure 4K (left): the rigid one, then the stretching one.
EXAMPLES_S10 = (("S10A", "GP4_28_S1_L41_D250_M2L", 237, "sampling"),
                ("S10B", "GP4_85_s10_L70_D250_ALM", 25, "movement"))

#: The AUC integrates the samples at or above this fraction of the range.
AUC_THRESH = 0.5
#: A maximum at or before this frame index makes the trial a decay candidate.
DECAY_MAX_FRAME = 1
#: A later rise of at least this fraction of the range rescues the trial.
DECAY_RISE_RATIO = 0.2

_NO_SEGMENTS = np.empty((0, 2), dtype=int)


def risingSegments(trace) -> np.ndarray:
    '''``[[start, end], ...]`` of each run of strictly rising samples.

    ``start`` is the sample the rise starts from and ``end`` the last sample
    of the rise, so ``trace[end] - trace[start]`` is the rise's height.
    '''
    trace = np.asarray(trace, dtype=float)
    rising = np.where(np.diff(trace) > 0)[0] + 1
    if not len(rising):
        return _NO_SEGMENTS
    runs = np.split(rising, np.where(np.diff(rising) != 1)[0] + 1)
    return np.array([(run[0] - 1, run[-1]) for run in runs], dtype=int)


def s10TraceAUC(trace, rising_segments=None) -> float:
    '''The S10A–B AUC of one trial, in dF/F × frames.

    Without ``rising_segments``: the trapezoid sum of the samples at or above
    ``min + AUC_THRESH * (max - min)``, taken as one sequence. With them: the
    trapezoid sum of ``trace[start:end]`` of every segment, concatenated (the
    end sample is left out).
    '''
    trace = np.asarray(trace, dtype=float)
    if rising_segments is not None:
        values = np.concatenate([trace[start:end]
                                 for start, end in rising_segments])
    else:
        lo, hi = trace.min(), trace.max()
        values = trace[trace >= lo + AUC_THRESH * (hi - lo)]
    return float(np.trapezoid(values))


def startsOnDecay(trace, max_idx):
    '''``(decaying, new_max_idx, segments)`` of one trial.

    A trial whose maximum is after frame ``DECAY_MAX_FRAME`` does not start on
    a decay: ``(False, None, empty)``. Otherwise each strictly rising segment
    is measured against the range ``trace[max_idx] - min``. If none rises by
    ``DECAY_RISE_RATIO`` of it, the trial starts on a decay:
    ``(True, None, segments)``. If some do, the peak moves to the end of the
    last of those: ``(False, that index, segments)``.
    '''
    trace = np.asarray(trace, dtype=float)
    if max_idx > DECAY_MAX_FRAME:
        return False, None, _NO_SEGMENTS
    segments = risingSegments(trace)
    if not len(segments):
        return True, None, segments
    rises = trace[segments[:, 1]] - trace[segments[:, 0]]
    big = rises / (trace[max_idx] - trace.min()) >= DECAY_RISE_RATIO
    if not big.any():
        return True, None, segments
    return False, int(segments[big, 1].max()), segments


def s10PeakAUC(trace, acq_rate):
    '''``(decaying, peak_sec, auc)`` of one trial.

    ``peak_sec`` is the maximum's time from sampling onset. A trial rescued by
    a later rise takes that rise's end as its peak and integrates its rising
    segments (``startsOnDecay``); any other trial integrates above the
    threshold (``s10TraceAUC``).
    '''
    trace = np.asarray(trace, dtype=float)
    max_idx = int(np.argmax(trace))
    auc = s10TraceAUC(trace)
    decaying, new_max_idx, segments = startsOnDecay(trace, max_idx)
    if not decaying and new_max_idx is not None:
        max_idx = new_max_idx
        auc = s10TraceAUC(trace, segments)
    return decaying, max_idx / acq_rate - TIME_BEFORE_SAMPLING, auc


def sessionAcqRate(sampling_df, sess) -> float:
    '''The session's frame rate, from the sampling-aligned frames.'''
    rates = sampling_df.loc[sampling_df.ShortName == sess, "acq_sampling_rate"]
    if not len(rates):
        raise KeyError(f"Session {sess!r} is not in the sampling frames")
    return float(rates.iloc[0])


def exampleTrials(df, sess, trace_id, key_prefix, acq_rate) -> pd.DataFrame:
    '''One row per trial of the neuron in ``df`` (``loopNeurons``' output).

    Columns: ``TrialNumber``, ``trial_dur_sec``, ``trace``, ``colour``
    (fast red to slow yellow, over all the neuron's trials),
    ``starts_on_decay``, ``peak_sec`` and ``auc``.
    '''
    neuron_df = df[(df.ShortName == sess) & (df.trace_id == trace_id)]
    if not len(neuron_df):
        raise KeyError(f"Neuron {trace_id} of session {sess!r} is not in the table")
    rows = [s10PeakAUC(trace, acq_rate)
            for trace in neuron_df[f"{key_prefix}trace"]]
    decaying, peak_sec, auc = (zip(*rows))
    return pd.DataFrame(dict(
        TrialNumber=neuron_df.TrialNumber.to_numpy(),
        trial_dur_sec=neuron_df.trial_dur_sec.to_numpy(dtype=float),
        trace=list(neuron_df[f"{key_prefix}trace"]),
        colour=fastSlowColors(neuron_df.trial_dur_sec.to_numpy()),
        starts_on_decay=np.array(decaying, dtype=bool),
        peak_sec=np.array(peak_sec, dtype=float),
        auc=np.array(auc, dtype=float)))


def exampleCorrelations(trials) -> dict:
    '''``{"peak": stats, "auc": stats}`` over the trials that do not start
    on a decay; each ``stats`` is ``_plotCorr``'s r, p and slope (degrees).'''
    kept = trials[~trials.starts_on_decay]
    return {key: _plotCorr(None, kept.trial_dur_sec, kept[col])
            for key, col in (("peak", "peak_sec"), ("auc", "auc"))}


def traceTimes(n_frames, acq_rate, aligned):
    '''Each frame's time in s from sampling onset or from movement onset.

    The frames run from ``TIME_BEFORE_SAMPLING`` before sampling onset to
    ``TIME_AFTER_MOVEMENT`` after movement onset.
    '''
    t = np.arange(n_frames) / acq_rate - TIME_BEFORE_SAMPLING
    if aligned == "sampling":
        return t
    if aligned == "movement":
        return t - (t[-1] - TIME_AFTER_MOVEMENT)
    raise ValueError(f"aligned must be 'sampling' or 'movement', not {aligned!r}")


def savePath(fig_save_prefix, br, sess, trace_id, panel) -> Path:
    return (samplingAlignedSavePath(fig_save_prefix, br, sess, trace_id).parent
            / f"{sess}_{trace_id}_{panel}.svg")


def drawColourKey(ax, vertical):
    '''A Fast (red) to Slow (yellow) key inside ``ax``, as in the published
    panels: vertical at the top right with Fast on top, or horizontal at the
    top left with Fast on the left. It has no numbers: ``fastSlowColors``
    spreads the colours over each neuron's own durations.'''
    grad = np.linspace(0, 1, 256)
    if vertical:
        cax = ax.inset_axes([0.86, 0.55, 0.025, 0.38])
        cax.imshow(grad[:, None], aspect="auto", cmap=FastSlow_CM)
        cax.text(1.6, 1, "Fast", transform=cax.transAxes, va="top", fontsize=8)
        cax.text(1.6, 0, "Slow", transform=cax.transAxes, va="bottom",
                 fontsize=8)
    else:
        cax = ax.inset_axes([0.12, 0.90, 0.30, 0.035])
        cax.imshow(grad[None, :], aspect="auto", cmap=FastSlow_CM)
        cax.text(-0.03, 0.5, "Fast", transform=cax.transAxes, ha="right",
                 va="center", fontsize=8)
        cax.text(1.03, 0.5, "Slow", transform=cax.transAxes, ha="left",
                 va="center", fontsize=8)
    cax.set_axis_off()
    return cax


def _regressionLine(ax, x, y):
    fit = linregress(x, y)
    xs = np.array([np.min(x), np.max(x)])
    ax.plot(xs, fit.intercept + fit.slope * xs, c="tab:green", lw=2)


def plotS10Example(df, sampling_df, sess, trace_id, *, panel, aligned,
                   key_prefix, save_figs=False, fig_save_prefix=None,
                   show=True, figsize=(15, 3.8)):
    '''One S10 example row: traces, peak time and AUC against sampling time.

    ``df`` is the active-trials table (``loopNeurons``' output) and
    ``sampling_df`` the sampling-aligned frames it was built from, used only
    for the session's frame rate. The traces panel shows every active trial,
    with a Fast/Slow colour key; the two correlations leave out the trials
    that start on a decay.
    '''
    acq_rate = sessionAcqRate(sampling_df, sess)
    trials = exampleTrials(df, sess, trace_id, key_prefix, acq_rate)
    br = df.loc[(df.ShortName == sess) & (df.trace_id == trace_id),
                "BrainRegion"].iloc[0]
    kept = trials[~trials.starts_on_decay]

    fig, (ax_tr, ax_pk, ax_auc) = plt.subplots(1, 3, figsize=figsize)
    for trace, colour in zip(trials.trace, trials.colour):
        ax_tr.plot(traceTimes(len(trace), acq_rate, aligned), trace,
                   c=colour, lw=1)
    ax_tr.axvline(0, c="gray", ls="--")
    drawColourKey(ax_tr, vertical=aligned == "sampling")
    if aligned == "sampling":
        ax_tr.set_xlim(-TIME_BEFORE_SAMPLING, 1.1)
        ax_tr.set_xlabel("Sampling Time (s)")
    else:
        ax_tr.set_xlim(right=TIME_AFTER_MOVEMENT)
        ax_tr.set_xlabel("Time from movement onset (s)")
    ax_tr.set_ylabel("dF/F")
    ax_tr.set_title(f"{panel}: {sess} — neuron {trace_id}\n"
                    f"{len(trials)} active trials, {len(kept)} in the "
                    "correlations", fontsize=9)

    x = kept.trial_dur_sec
    for ax, col, ylabel in ((ax_pk, "peak_sec", "Max Firing Position (s)"),
                            (ax_auc, "auc", "AUC (dF/F × frames)")):
        ax.scatter(x, kept[col], c=list(kept.colour), s=25, zorder=3)
        _regressionLine(ax, x, kept[col])
        _plotCorr(ax, x, kept[col])
        ax.title.set_fontsize(9)
        ax.set_xlabel("Sampling Time (s)")
        ax.set_ylabel(ylabel)
    lim = max(x.max(), kept.peak_sec.max())
    ax_pk.plot([0, lim], [0, lim], c="gray", ls="--", alpha=.6)
    for ax in (ax_tr, ax_pk, ax_auc):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_figs:
        assert fig_save_prefix is not None, "Pass fig_save_prefix to save"
        save_fp = savePath(fig_save_prefix, br, sess, trace_id, panel)
        save_fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fp, bbox_inches="tight")
    if show:
        plt.show()
    return fig
