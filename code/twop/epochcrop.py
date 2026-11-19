'''Reshape per-trial traces: cut an epoch out, pad, or keep some neurons.

Each row is one trial; ``traces_sets`` holds one trace per neuron and
``epochs_names`` / ``epochs_ranges`` say where each epoch sits (inclusive
sample ranges).

- :func:`onlyEpoch` keeps one epoch of trial-concatenated traces
  (:func:`twop.dataload.loadTrialNormalized`) and rewrites the bookkeeping so
  the result looks like data that only ever had that epoch (Figure S12A).
- :func:`epochStart` does the same for only the first samples of the epoch
  (Figure S12C).
- :func:`padToCommonLength` NaN-pads real-time traces, which differ in length
  from trial to trial, to one length, optionally lining up an epoch's onset
  (Figures S12D, S12J).
- :func:`keepNeurons` drops every trace but the listed neurons'.
'''
from __future__ import annotations

import numpy as np
import pandas as pd


def _cutRow(row, epoch_name, n_samples=None):
    names = list(row.epochs_names)
    if epoch_name not in names:
        raise KeyError(f"{epoch_name!r} is not among the epochs {names}")
    start, end = row.epochs_ranges[names.index(epoch_name)]
    if n_samples is not None:
        end = min(end, start + n_samples - 1)
    traces_sets = {set_name: {trace_id: trace[start:end + 1]
                              for trace_id, trace in traces.items()}
                   for set_name, traces in row.traces_sets.items()}
    return traces_sets, end - start


def onlyEpoch(df, epoch_name, n_samples=None):
    '''A copy of ``df`` whose traces hold only ``epoch_name``'s samples.

    Each row's ``traces_sets`` arrays are cut to that epoch's inclusive range
    (its first ``n_samples`` only, if given); ``epochs_names`` /
    ``epochs_ranges`` become that single epoch starting at 0, and ``epoch`` /
    ``trace_start_idx`` / ``trace_end_idx`` follow.
    '''
    df = df.copy()
    cut = [_cutRow(row, epoch_name, n_samples)
           for row in df.itertuples(index=False)]
    df["traces_sets"] = pd.Series([c[0] for c in cut], index=df.index, dtype=object)
    last = [c[1] for c in cut]
    df["epochs_names"] = pd.Series([[epoch_name]] * len(df), index=df.index,
                                   dtype=object)
    df["epochs_ranges"] = pd.Series([[(0, n)] for n in last], index=df.index,
                                    dtype=object)
    df["epoch"] = epoch_name
    df["trace_start_idx"] = 0
    df["trace_end_idx"] = last
    return df


def epochStart(df, epoch_name, n_samples):
    ''':func:`onlyEpoch` keeping the epoch's first ``n_samples`` samples.

    A trial whose epoch is shorter keeps the whole epoch.
    '''
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, not {n_samples}")
    return onlyEpoch(df, epoch_name, n_samples=n_samples)


def padToCommonLength(df, align_epoch=None):
    '''Real-time traces of different lengths, NaN-padded to one length.

    Each row's traces cover ``trace_start_idx..trace_end_idx`` (inclusive).
    With ``align_epoch=None`` they keep their start (sample 0 lines up) and
    are padded at the end. With an epoch name, each row is padded at the start
    so that epoch's first sample falls on the same index in every row, then
    padded at the end to the common length; ``epochs_ranges`` shift with it.
    ``trace_start_idx`` / ``trace_end_idx`` then span the whole padded trace.
    '''
    df = df.copy()
    rows = list(df.itertuples(index=False))
    if align_epoch is None:
        offsets = [0] * len(rows)
    else:
        onsets = [r.epochs_ranges[list(r.epochs_names).index(align_epoch)][0]
                  - r.trace_start_idx for r in rows]
        offsets = [max(onsets) - onset for onset in onsets]
    lengths = [r.trace_end_idx - r.trace_start_idx + 1 for r in rows]
    total = max(off + n for off, n in zip(offsets, lengths))

    def pad(row, off, n):
        s = row.trace_start_idx
        return {set_name: {tid: np.concatenate([np.full(off, np.nan),
                                                np.asarray(tr[s:s + n], float),
                                                np.full(total - off - n, np.nan)])
                           for tid, tr in traces.items()}
                for set_name, traces in row.traces_sets.items()}

    df["traces_sets"] = pd.Series([pad(r, off, n) for r, off, n in
                                   zip(rows, offsets, lengths)],
                                  index=df.index, dtype=object)
    df["epochs_ranges"] = pd.Series(
        [[(a - r.trace_start_idx + off, b - r.trace_start_idx + off)
          for a, b in r.epochs_ranges] for r, off in zip(rows, offsets)],
        index=df.index, dtype=object)
    df["trace_start_idx"] = 0
    df["trace_end_idx"] = total - 1
    return df


def keepNeurons(df, long_trace_ids):
    '''Drop every trace whose ``<ShortName>_<trace_id>`` is not listed.

    Rows (trials) left with no trace are dropped too.
    '''
    long_trace_ids = set(long_trace_ids)
    df = df.copy()
    kept = [{set_name: {tid: tr for tid, tr in traces.items()
                        if f"{row.ShortName}_{tid}" in long_trace_ids}
             for set_name, traces in row.traces_sets.items()}
            for row in df.itertuples(index=False)]
    df["traces_sets"] = pd.Series(kept, index=df.index, dtype=object)
    return df[[any(len(t) for t in ts.values()) for ts in kept]]
