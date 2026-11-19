'''Real-time traces of choice-selective subpopulations (Figures S12D, S12J).

Backend for the "Figure S12D (bottom)" and "Figure S12J (right)" sections of
``2p_3_choice.ipynb``. Both draw, per region, the mean activity of a group of
choice-selective neurons on matched (preferred-choice) vs mismatched trials,
fast vs slow, in real time rather than in the time-normalised frame of
Figures 5E / S12E. The choices follow those figures:

- **neurons**: significant for the current choice in the sampling-epoch ROC
  test (``run_data_unnormalized[0].shortlong_df``, :func:`choiceStats`), their
  preferred side from the ROC direction (``IsROCLeftTuend``);
- **trials**: correct and incorrect together, as in 5E / S12E (or, with
  ``correct_only``, correct trials only, saved under a ``_correct_trials``
  name);
- **traces**: the same z-scored real-time sampling traces the ROC test ran on
  (``dataload.loadSamplingUnnormalized``), from 0.1 s before sampling to
  0.1 s after movement onset;
- **statistics**: all trials of all the group's neurons pooled, mean +/- SEM
  over those trials (``plottuning.pooledTrialsMeanTrace``).

The groups:

- S12D: the neurons of the first and of the last sampling bin of the 5E /
  S12E bars (``plottuning.neuronsInBin``), traces aligned to sampling onset;
- S12J: the movement-time choice neurons counted in S12J (left)
  (``movementneurons.movementMask``), traces aligned to movement onset.
'''
from __future__ import annotations

from .epochcrop import keepNeurons, padToCommonLength
from .expanddf import expandDf
from .genrundata import makeLongTraceID
from .movementneurons import (DEFAULT_BINS_COUNT, METHOD_LAST_TWO_BINS,
                              binEdges, movementMask)
from .plottuning import (RealTimeTuningPlot, neuronsInBin,
                         samplingSlowVsFast)

SAMPLING_EPOCH = "Sampling"
MOVEMENT_EPOCH = RealTimeTuningPlot.MOVEMENT_EPOCH
#: Sequence bins of S12D: the first and the last sampling bin (bin 0 is the
#: 0.1 s before sampling, the last bin the movement).
FIRST_SAMPLING_BIN = 1
LAST_SAMPLING_BIN = DEFAULT_BINS_COUNT - 2


def choiceStats(shortlong_df, pval, data_col="ChoiceLeft"):
    '''The ROC rows of the neurons significant for ``data_col``, one per neuron.

    No difficulty split (``DVstr`` null), ``pval <= pval``. The selection of
    Figures 5E / S12E (``nonQuantiledStats`` in ``2p_3_choice.ipynb``).
    '''
    df = shortlong_df[(shortlong_df.data_col == data_col)
                      & shortlong_df.DVstr.isnull()
                      & (shortlong_df.pval <= pval)]
    return df.drop_duplicates(["long_trace_id"])


def sequenceBinNeurons(neuron_firing_iqr_df, epoch_ranges, bin_idx,
                       bins_count=DEFAULT_BINS_COUNT):
    '''``long_trace_id`` set of the neurons in one 5E / S12E sequence bin.'''
    bins = binEdges(epoch_ranges, bins_count)
    return neuronsInBin(neuron_firing_iqr_df, bins, bin_idx)


def movementChoiceNeurons(max_firing_df, epoch_ranges, choice_ids,
                          method=METHOD_LAST_TWO_BINS,
                          bins_count=DEFAULT_BINS_COUNT):
    '''``long_trace_id`` set of the movement-time choice neurons of S12J (left).'''
    df = max_firing_df[max_firing_df.max_idxs.apply(len) > 0]
    is_movement = movementMask(df, epoch_ranges, method=method,
                               bins_count=bins_count)
    return set(df.trace_id[is_movement]) & set(choice_ids)


def realTimeExpanded(sampling_df, long_trace_ids, align_epoch=SAMPLING_EPOCH):
    '''The listed neurons' real-time traces, one row per trial and neuron.

    ``sampling_df`` is ``dataload.loadSamplingUnnormalized()``. Traces are
    NaN-padded to one length with ``align_epoch``'s onset on one sample, then
    expanded (``Raw`` columns) with a ``long_trace_id`` column.
    '''
    df = keepNeurons(sampling_df, long_trace_ids)
    df = padToCommonLength(df, align_epoch=align_epoch)
    return makeLongTraceID(expandDf(df))


def plotSubpopulationTraces(expanded_df, stats_df, long_trace_ids, label,
                            save_name, align_epoch, sampling_rate, pval,
                            save_figs, fig_save_prefix, pool_trials=True,
                            color_by_strategy=False):
    '''Matched vs mismatched, fast vs slow, one axes per region.

    ``stats_df`` is :func:`choiceStats`; only its rows in ``long_trace_ids``
    are drawn. ``pool_trials`` / ``color_by_strategy`` as for
    ``plottuning.RealTimeTuningPlot``.
    Saves ``RT_Stats/traces/<save_name>.svg``.
    '''
    stats_df = stats_df[stats_df.long_trace_id.isin(set(long_trace_ids))]
    plotter = RealTimeTuningPlot(label=label, align_epoch=align_epoch,
                                 sampling_rate=sampling_rate,
                                 save_name=save_name, pool_trials=pool_trials,
                                 color_by_strategy=color_by_strategy)
    samplingSlowVsFast(stats_df=stats_df.copy(),
                       src_activitiy_df=expanded_df,
                       tuning_plotter=plotter, left_tune_col="IsROCLeftTuend",
                       sep_side=False, sep_brain_region=True, pval=pval,
                       skip_qidx2=True, save_fig=save_figs,
                       fig_save_prefix=fig_save_prefix)
    return plotter


#: (sequence bin, label, file name) of the two S12D subpopulations.
S12D_GROUPS = [(FIRST_SAMPLING_BIN, "Sampling-start subpopulation",
                "s12d_sampling_start_subpopulation"),
               (LAST_SAMPLING_BIN, "Decision subpopulation",
                "s12d_decision_subpopulation")]
S12J_LABEL = "Movement-time choice neurons"
S12J_NAME = "s12j_movement_choice_neurons"


#: (file-name suffix, title suffix) of the correct-trials-only versions.
CORRECT_ONLY_SUFFIX = ("_correct_trials", " (correct trials)")


def _samplingRate(sampling_df):
    return float(sampling_df.acq_sampling_rate.median())


def _trials(sampling_df, correct_only):
    '''``sampling_df``, or only its correct trials, with the name suffixes.'''
    if not correct_only:
        return sampling_df, ("", "")
    return sampling_df[sampling_df.ChoiceCorrect == 1], CORRECT_ONLY_SUFFIX


def plotSequenceEndsTraces(sampling_df, stats_df, neuron_firing_iqr_df,
                           epoch_ranges, pval, save_figs, fig_save_prefix,
                           correct_only=False):
    '''Figure S12D (bottom): the first and last sampling bins' neurons.

    One figure per subpopulation, traces aligned to sampling onset.
    ``epoch_ranges`` / ``neuron_firing_iqr_df`` are the ones the 5E / S12E
    bars bin with. Returns ``{bin: long_trace_id set}``.
    '''
    sampling_df, (name_sfx, label_sfx) = _trials(sampling_df, correct_only)
    choice_ids = set(stats_df.long_trace_id)
    groups = {bin_idx: sequenceBinNeurons(neuron_firing_iqr_df, epoch_ranges,
                                          bin_idx) & choice_ids
              for bin_idx, _, _ in S12D_GROUPS}
    expanded_df = realTimeExpanded(sampling_df, set().union(*groups.values()),
                                   align_epoch=SAMPLING_EPOCH)
    for bin_idx, label, name in S12D_GROUPS:
        plotSubpopulationTraces(expanded_df, stats_df, groups[bin_idx],
                                label + label_sfx, name + name_sfx,
                                SAMPLING_EPOCH, _samplingRate(sampling_df),
                                pval, save_figs, fig_save_prefix)
    return groups


def plotMovementNeuronTraces(sampling_df, stats_df, max_firing_df,
                             epoch_ranges, pval, save_figs, fig_save_prefix,
                             correct_only=False, color_by_strategy=True):
    '''Figure S12J (right): S12J (left)'s neurons, aligned to movement onset.

    Fast / slow in their colours (``clr.Strategy``), mismatched dashed, as in
    the paper; ``color_by_strategy=False`` draws S12D's matched / mismatched
    colours instead. Returns the ``long_trace_id`` set drawn.
    '''
    sampling_df, (name_sfx, label_sfx) = _trials(sampling_df, correct_only)
    ids = movementChoiceNeurons(max_firing_df, epoch_ranges,
                                set(stats_df.long_trace_id))
    expanded_df = realTimeExpanded(sampling_df, ids, align_epoch=MOVEMENT_EPOCH)
    plotSubpopulationTraces(expanded_df, stats_df, ids,
                            S12J_LABEL + label_sfx, S12J_NAME + name_sfx,
                            MOVEMENT_EPOCH,
                            _samplingRate(sampling_df), pval, save_figs,
                            fig_save_prefix,
                            color_by_strategy=color_by_strategy)
    return ids
