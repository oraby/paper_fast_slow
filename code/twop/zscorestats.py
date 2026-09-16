"""Per-neuron mean and std of the traces a z-score normalisation consumed.

TwoPLoad's Figure 6B path normalises feedback responses by each neuron's
*sampling-epoch* mean and std — the two numbers
:class:`~...pipeline.tracesnormalize.NormalizeZScore` computes internally and
this repo's pipeline does not keep. (The author's later
``OneDrive/caiman/common/analysis/pipeline`` stores them in a ``<set>_stats``
column; porting that here would also have to change how every normaliser
returns its result, so the statistics are recomputed instead.)

Feed this the same frame the normalisation ran on — for TwoPLoad, the sampling
epochs cut by ``alignSampling`` with ``NoNormalization`` — and it reproduces
those statistics exactly, including the way the normaliser keys traces by trial
number, so a session with two rows for one trial keeps only the last.
"""
import numpy as np
import pandas as pd


def _traceSegment(row, trace):
    """The part of ``trace`` the normaliser sees for ``row``."""
    if row.sole_owner:          # the row owns the whole array already
        return trace
    return trace[row.trace_start_idx:row.trace_end_idx + 1]


def traceStats(segments):
    """``{"mean", "std"}`` over the concatenation, as ``NormalizeZScore`` does."""
    concat = np.hstack(list(segments))
    return {"mean": np.nanmean(concat, axis=0), "std": np.nanstd(concat, axis=0)}


def sessionTraceStats(df, set_name="neuronal"):
    """One row per session: ``ShortName`` and a ``<set_name>_stats`` dict.

    The dict maps trace id to ``{"mean", "std"}``, matching the column the
    ancestor pipeline writes, so a caller can index it the same way.
    """
    rows = []
    for short_name, sess_df in df.groupby("ShortName"):
        by_trace = {}
        for _, row in sess_df.iterrows():
            for trace_id, trace in row.traces_sets[set_name].items():
                # Keyed by trial, like the normaliser's own dict: a repeated
                # trial number replaces the earlier segment rather than adding.
                by_trace.setdefault(trace_id, {})[row.TrialNumber] = \
                    _traceSegment(row, trace)
        rows.append({"ShortName": short_name,
                     f"{set_name}_stats": {trace_id: traceStats(trials.values())
                                           for trace_id, trials in by_trace.items()}})
    return pd.DataFrame(rows)
