'''Single-neuron traces on fast against slow trials (Figure 4F).

Backend for the "Plot neurons traces in fast vs slow trials" section of
``2p_2_sequences.ipynb``.

Each figure is one neuron, two panels: its fast trials on the left and its
slow trials on the right. A trial where the neuron was **active** is drawn in
green, one where it was not in grey, and the panel title gives the share of
active trials. Both panels share the y-axis.

- ``plotFastSlowNeuron`` -- one neuron's figure.
- ``loopFastSlowNeurons`` -- every neuron of the two tables, region by region,
  or only the ``trace_ids`` asked for (the published example of Figure 4F is
  MFC neuron ``GP4_28_S2_L51_D250_M2L_206``).

A figure is written to ``{fig_save_prefix}/FastSlowTraces/{MFC,LFC}/{trace_id}.svg``.
'''
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt

from ..common.definitions import BrainRegion

#: How many neurons per region a loop that does not save still draws.
MAX_UNSAVED_PER_REGION = 3


def regionName(br) -> str:
    '''"MFC" / "LFC" for a region code (other regions keep their short name).'''
    br_str = str(BrainRegion(br)).split("_")[0]
    return br_str.replace("M2", "MFC").replace("ALM", "LFC")


def _plotNeuronTraces(neuron_active_trials_row, epochs_ranges, epochs_names,
                      dscrp, ax):
    ax.set_xticks([rng[0] for rng in epochs_ranges[1:]])
    for rng in epochs_ranges[1:]:
        ax.axvline(rng[0], color='gray', linestyle='--', alpha=0.5)

    ax.set_xticklabels([name for name in epochs_names[1:]])
    for trace in neuron_active_trials_row.invalid_whole_trace:
        ax.plot(trace, c="gray", alpha=0.3, lw=0.5)
    for trace in neuron_active_trials_row.whole_trace:
        ax.plot(trace, c="g")
    ax.set_xlabel("Normalized Time (s)")
    len_active = len(neuron_active_trials_row.whole_trace)
    len_invalid = len(neuron_active_trials_row.invalid_whole_trace)
    prcnt_active = len_active/(len_active + len_invalid)*100
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.set_title(f"{dscrp} - {prcnt_active:.2g}% Active")
    return ax.get_ylim()


def _sessionEpochs(inf_df, short_name):
    inf_sess_df = inf_df[inf_df.ShortName == short_name]
    epochs_ranges = inf_sess_df.epochs_ranges.apply(tuple)
    epochs_names = inf_sess_df.epochs_names.apply(tuple)
    assert inf_sess_df.acq_sampling_rate.nunique() == 1
    assert epochs_ranges.nunique() == 1
    assert epochs_names.nunique() == 1
    return epochs_ranges.iloc[0], epochs_names.iloc[0]


def savePath(fig_save_prefix, br_str, trace_id) -> pathlib.Path:
    return pathlib.Path(f"{fig_save_prefix}/FastSlowTraces/{br_str}",
                        f"{trace_id}.svg")


def plotFastSlowNeuron(fast_row, slow_row, inf_df, br_str, save_figs,
                       fig_save_prefix=None, show=None):
    '''Draw one neuron's fast and slow trials side by side.

    ``show`` defaults to showing only when not saving (a saving loop closes
    each figure instead, so hundreds of them do not pile up).
    '''
    epochs_ranges, epochs_names = _sessionEpochs(inf_df, fast_row.ShortName)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    y_lim1 = _plotNeuronTraces(fast_row, dscrp="Fast Trials", ax=axs[0],
                               epochs_ranges=epochs_ranges,
                               epochs_names=epochs_names)
    y_lim2 = _plotNeuronTraces(slow_row, dscrp="Slow Trials", ax=axs[1],
                               epochs_ranges=epochs_ranges,
                               epochs_names=epochs_names)
    y_lim = (min(y_lim1[0], y_lim2[0]), max(y_lim1[1], y_lim2[1]))
    axs[0].set_ylim(y_lim)
    axs[1].set_ylim(y_lim)
    fig.suptitle(f"{br_str} - {fast_row.trace_id}")
    if save_figs:
        assert fig_save_prefix is not None, "Pass fig_save_prefix to save"
        save_fp = savePath(fig_save_prefix, br_str, fast_row.trace_id)
        save_fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fp, bbox_inches='tight')
    if show is None:
        show = not save_figs
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def loopFastSlowNeurons(fast_df, slow_df, inf_df, save_figs,
                        fig_save_prefix=None, trace_ids=None, show=None):
    '''Draw every neuron of ``fast_df`` (or only ``trace_ids``), region by region.

    Without saving and without ``trace_ids``, only the first
    ``MAX_UNSAVED_PER_REGION`` neurons of each region are drawn. Asking for
    ``trace_ids`` that are not in ``fast_df`` raises, so an example cannot
    silently go missing. Returns the trace ids drawn.
    '''
    if trace_ids is not None:
        trace_ids = list(trace_ids)
        missing = set(trace_ids) - set(fast_df.trace_id)
        if missing:
            raise KeyError(f"Neurons not in the fast-trials table: {sorted(missing)}")
        fast_df = fast_df[fast_df.trace_id.isin(trace_ids)]
    drawn = []
    for br, fast_df_br in fast_df.groupby("BrainRegion"):
        br_str = regionName(br)
        slow_df_br = slow_df[slow_df.BrainRegion == br]
        count = 0
        for _, fast_row in fast_df_br.iterrows():
            slow_row = slow_df_br[slow_df_br.trace_id == fast_row.trace_id]
            assert len(slow_row) == 1
            count += 1
            if not save_figs and trace_ids is None and \
               count > MAX_UNSAVED_PER_REGION:
                break
            plotFastSlowNeuron(fast_row, slow_row.iloc[0], inf_df, br_str,
                               save_figs=save_figs,
                               fig_save_prefix=fig_save_prefix, show=show)
            drawn.append(fast_row.trace_id)
    return drawn
