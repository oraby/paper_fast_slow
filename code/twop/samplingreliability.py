'''Which neurons fire reliably during sampling, from the raw 2P traces.

Shared by ``2p_0_preprocessing``, which keeps only the reliable neurons in the
dF/F frame it writes, and ``2p_4_feedback``, whose Figures 6B and S14A are
measured against the same neurons. Both start from ``df_all_by_epoch`` (the
raw fluorescence) and compute the same frames, so the feedback notebook does
not have to read anything the preprocessing notebook writes besides
``data/2p``.

- ``alignSamplingFrames`` -- the sampling epoch cut out of every trial, in
  real time (``unnormed``) and stretched to a common length (``normed``), both
  z-scored per neuron; plus each neuron's sampling mean and std, which Figure
  6B normalises the feedback responses by.
- ``collectReliability`` -- runs ``tracereliability``'s per-neuron test over a
  frame and returns the collector's dict and table.
- ``makeNormedAll`` -- every epoch of every trial, the feedback epochs cut to
  0.3 s after their start and the others stretched to a common length.
'''
from __future__ import annotations

import pandas as pd

from ..common.definitions import BrainRegion
from ..pipeline import pipeline, traceswidth, tracesnormalize
from ..pipeline.tracesnormalize import NoNormalization
from .alignsampling import alignSampling
from .evd_conditions_common import _alignAroundEpoch
from .tracereliability import StdDistCollector, loopSessionsExampleNeurons
from .zscorestats import sessionTraceStats

TIME_BEFORE_SAMPLING = 0.1
TIME_AFTER_MOVEMENT = 0.1
#: Feedback epochs are cut to this many seconds after their start.
TIME_AFTER_FEEDBACK = 0.3


def alignSamplingFrames(df_all_by_epoch, time_before=TIME_BEFORE_SAMPLING,
                        time_after=TIME_AFTER_MOVEMENT):
    '''``(unnormed_sampling_df, normed_sampling_df, sampling_stats_df)``.

    Built from the raw ``df_all_by_epoch``, not from the dF/F traces.
    '''
    kargs = dict(normalization=tracesnormalize.NormalizeZScore(set_name="neuronal"),
                 time_before_sampling=time_before,
                 time_after_sampling=time_after)
    unnormed_sampling_df = alignSampling(df=df_all_by_epoch.copy(), **kargs,
                                         normalize_epoch_time=False)
    unnormed_sampling_df["epoch"] = "Sampling"
    normed_sampling_df = alignSampling(df_all_by_epoch.copy(), **kargs,
                                       normalize_epoch_time=True)
    normed_sampling_df["epoch"] = "Sampling"
    # Figure 6B normalises feedback responses by each neuron's sampling mean
    # and std -- the two numbers NormalizeZScore uses above, which the pipeline
    # does not keep. The same call without normalising gives exactly the traces
    # it consumed; zscorestats recomputes the statistics from them.
    raw_sampling_df = alignSampling(df=df_all_by_epoch.copy(),
                                    **dict(kargs, normalization=NoNormalization()),
                                    normalize_epoch_time=False)
    sampling_stats_df = sessionTraceStats(raw_sampling_df)
    return unnormed_sampling_df, normed_sampling_df, sampling_stats_df


def collectReliability(sampling_df, decide_neurons, pdf_str, save_figs,
                       fig_save_prefix=None, **collector_kw):
    '''``(res_dict, frame)`` of a ``StdDistCollector`` run over ``sampling_df``.

    ``save_figs`` writes ``loopSessionsExampleNeurons``' per-neuron figures
    under ``pdf_str``.
    '''
    collector = StdDistCollector(decide_neurons, **collector_kw)
    loopSessionsExampleNeurons(sampling_df, random_fraction=1, seed=1,
                               pdf_str=pdf_str, processFn=collector.track,
                               save_figs=save_figs,
                               fig_save_prefix=fig_save_prefix)
    return collector.res_dict, collector.frame()


def makeNormedAll(df):
    '''Every epoch of the L2/3 MFC/LFC trials with a choice, time-normalised.'''
    df = df.reset_index(drop=True)
    df = df[df.Layer == "L23"]
    df = df[df.BrainRegion.isin([BrainRegion.M2_Bi, BrainRegion.ALM_Bi])]
    print("Extracting Reward and Punishment epochs")
    feedback = {}
    for epoch in ("Reward", "Punishment"):
        feedback[epoch] = _alignAroundEpoch(df=df.copy(), align_to_epoch=epoch,
                                            time_before=0,
                                            time_after=TIME_AFTER_FEEDBACK,
                                            limit_trial_end=False,
                                            normalization=None)
    # Drop the feedback epochs, already extracted above, to speed up the time
    # normalization
    df = df[df.ChoiceCorrect.notnull()]
    df_wo_feedback = df[~df.epoch.isin(["Reward", "Punishment"])]
    print(f"Full: {len(df):,} - Len df_wo_feedback: {len(df_wo_feedback):,} ")
    chain = pipeline.Chain(
        pipeline.By("ShortName"), pipeline.By("epoch"),
        traceswidth.NormalizeTime(show_progress=False),
        pipeline.RecombineResults(),
    )
    df_wo_feedback_norm = chain.run(df_wo_feedback)

    df_concat = pd.concat([df_wo_feedback_norm, feedback["Reward"],
                           feedback["Punishment"]], ignore_index=True)
    df_concat = df_concat.sort_values(["BrainRegion", "Date", "ShortName",
                                       "TrialNumber", "state_id"])
    df_concat = df_concat[df_concat.ChoiceCorrect.notnull()]
    return df_concat.reset_index(drop=True)
