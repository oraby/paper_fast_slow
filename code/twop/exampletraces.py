'''Example single-neuron dF/F traces (Figure 4D, right).

Backend for the "Example neurons — Figure 4D (right)" cell of
``2p_1_population.ipynb``.

The panel shows three MFC neurons of session ``GP4_23_s3_L50_D250_M2``, each
a window of about 27 s of its dF/F, stacked top to bottom. The windows come
from different moments of the session: the panel illustrates what single
neurons look like, not one stretch of time. The neurons and windows were found
by matching the published traces against every neuron's whole-session trace.

- ``exampleWindows`` -- the (time, dF/F) of each window.
- ``plotExampleTraces`` -- the figure, written to
  ``{fig_save_prefix}/ExampleTraces/{session}_neurons_{ids}.svg``.
'''
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np

#: The session of the published example neurons.
EXAMPLE_SESSION = "GP4_23_s3_L50_D250_M2"
#: (neuron id, first frame, last frame, colour) of each published trace, top to
#: bottom: brown, green and purple in the paper.
EXAMPLE_NEURONS = ((37, 12753, 13572, "#5b2c0e"),
                   (33, 10389, 11207, "#16891a"),
                   (25, 25217, 26032, "#7311e0"))


def sessionTraces(df, session):
    '''The ``neuronal`` traces of one session and its frame rate.

    Every row of a session shares the same whole-session traces, so the first
    row is enough.
    '''
    sess_df = df[df.ShortName == session]
    if not len(sess_df):
        raise KeyError(f"Session {session!r} is not in the dataframe")
    row = sess_df.iloc[0]
    return row.traces_sets["neuronal"], float(row.acq_sampling_rate)


def exampleWindows(df, session=EXAMPLE_SESSION, neurons=EXAMPLE_NEURONS):
    '''``[(neuron_id, time_s, dff, colour), ...]``, one per neuron, in order.

    ``time_s`` starts at 0 at each window's first frame; both frames are
    included.
    '''
    traces, fs = sessionTraces(df, session)
    windows = []
    for neuron_id, first, last, colour in neurons:
        if neuron_id not in traces:
            raise KeyError(f"Neuron {neuron_id} is not in session {session!r}")
        trace = np.asarray(traces[neuron_id])
        if not 0 <= first <= last < len(trace):
            raise IndexError(f"Frames {first}-{last} are outside neuron "
                             f"{neuron_id}'s {len(trace)} frames")
        dff = trace[first:last + 1]
        windows.append((neuron_id, np.arange(len(dff)) / fs, dff, colour))
    return windows


def savePath(fig_save_prefix, session, neurons) -> pathlib.Path:
    ids = "_".join(str(neuron[0]) for neuron in neurons)
    return (pathlib.Path(fig_save_prefix) / "ExampleTraces" /
            f"{session}_neurons_{ids}.svg")


def plotExampleTraces(df, session=EXAMPLE_SESSION, neurons=EXAMPLE_NEURONS, *,
                      save_figs=False, fig_save_prefix=None, show=True):
    '''Stack the example windows on one axis, the first neuron on top.

    The traces share one dF/F scale; each is offset by the largest window
    range so none overlap. Only the time axis is drawn.
    '''
    windows = exampleWindows(df, session, neurons)
    spacing = max(np.nanmax(dff) - np.nanmin(dff) for _, _, dff, _ in windows)
    fig, ax = plt.subplots(figsize=(3.2, 3.6))
    for i, (neuron_id, t, dff, colour) in enumerate(windows):
        offset = (len(windows) - 1 - i) * spacing
        ax.plot(t, dff - np.nanmin(dff) + offset, color=colour, lw=0.8,
                label=f"Neuron {neuron_id}")
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{session} — neurons "
                 f"{', '.join(str(n[0]) for n in neurons)}", fontsize="small")
    fig.tight_layout()
    if save_figs:
        save_fp = savePath(fig_save_prefix, session, neurons)
        save_fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fp, bbox_inches='tight')
    if show:
        plt.show()
    return fig
