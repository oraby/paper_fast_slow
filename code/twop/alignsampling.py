from ..behavior.util.splitdata import splitStimulusTimeByQuantile
from ..common.imaging import getSamplingEpoch
from ..pipeline import pipeline
from ..pipeline import tracesnormalize
import pandas as pd


def _processPart(df, time_before_sampling, time_after_sampling,
                 normalize_epoch_time, concatenate_final_epochs):
  return getSamplingEpoch(df.copy(), epoch="Sampling",
                          normalization=tracesnormalize.NoNormalization(),
                          normalize_after_cutting=False, # doesn't matter here
                          time_epoch_before=time_before_sampling,
                          time_epoch_after=time_after_sampling,
                          # rename_feedback_epoch=False, # doesn't matter here
                          normalize_epoch_time=normalize_epoch_time,
                          concatenate_final_epochs=concatenate_final_epochs)

def alignSampling(df, time_before_sampling, time_after_sampling,
                  normalize_epoch_time : bool, normalization,
                  num_quantiles=None, sep_above_quantile_sec=None,
                  filter_above_sec=None,
                  normalize_sessions_before_splitting=None,
                  concatenate_final_epochs=True,
                  epoch="Sampling"):

    if num_quantiles is not None:
        if not isinstance(normalization, tracesnormalize.NoNormalization):
            assert normalize_sessions_before_splitting is not None, (
                "normalize_sessions_before_splitting must be specified when "
                "num_quantilesis specified")
    if sep_above_quantile_sec is not None:
        assert num_quantiles is not None, ("sep_above_quantiel_sec can only be "
            "used when num_quantiles is specified. sep_above_quantiel_sec "
            "separates very long trials into their own special bin")
        if filter_above_sec is not None:
            assert filter_above_sec > sep_above_quantile_sec, (
                "filter_above_sec trials are removed, while "
                "sep_above_quantile_sec  are kept as a special quantile bin")

    if filter_above_sec:
        # TODO: Create a new quantile for this
        len_before = len(df)
        df = df[df.calcStimulusTime <= filter_above_sec]
        len_now = len(df)
        print(f"Above sec filter: {len_now:,}/{len_before:,} "
            f"({len_before-len_now:,} rows (not trials) removed)")
    # sample_name = df_all_by_epoch.Name.iloc[0]
    # df = df[df.Name == df.Name.iloc[0]]
    # display(df.epoch.unique())
    print("Removing decision time trials and nan-time trials...")
    len_before = len(df)
    df = df.groupby(["Name", "Date", "SessionNum", "TrialNumber"]).filter(
            lambda trial_df:not any(trial_df.epoch == "Decision Time") and any(
                                  trial_df.epoch == "Movement to Lateral Port"))
    df = df[df.calcStimulusTime.notnull()]
    len_now = len(df)
    print(f"Decision time filter: {len_now:,}/{len_before:,} "
            f"({len_before-len_now:,} rows (not trials) removed)")
    print("Done...")
    df = df.reset_index(drop=True)

    _processPart_kwargs = dict(time_before_sampling=time_before_sampling,
                               time_after_sampling=time_after_sampling,
                               normalize_epoch_time=normalize_epoch_time,
                              concatenate_final_epochs=concatenate_final_epochs)

    if normalize_sessions_before_splitting:
        print("Normalizing just sampling across all trials...")
        sess_df_li = []
        for sess, sess_df in df.groupby("ShortName"):
            sess_df = getSamplingEpoch(sess_df.copy(),
                                       epoch=epoch,
                                       normalization=normalization,
                                       time_epoch_before=time_before_sampling,
                                       time_epoch_after=time_after_sampling,
                                       # rename_feedback doesn't matter here
                                       # rename_feedback_epoch=False,
                                       normalize_after_cutting=True,
                                       normalize_epoch_time=False,
                                       concatenate_final_epochs=False)
            sess_df_li.append(sess_df)
        df = pd.concat(sess_df_li)
        df = df.reset_index(drop=True)

    if num_quantiles is not None:
        df_li = []
        df_sampling = df[df.epoch == epoch] # Don't have duplicates
        for q_idx, quantile_df in splitStimulusTimeByQuantile(df_sampling,
                                        quantiles=num_quantiles,
                                        cut_above_sec=sep_above_quantile_sec):
            quantile_df = df.merge(quantile_df[["ShortName", "TrialNumber"]],
                                   on=["ShortName", "TrialNumber"])
            quantile_df = _processPart(quantile_df, **_processPart_kwargs)
            quantile_df["quantile_idx"] = q_idx
            df_li.append(quantile_df)
        df = pd.concat(df_li)
        df = df.sort_index()
    else:
        df = _processPart(df, **_processPart_kwargs)

    if not normalize_sessions_before_splitting and not isinstance(normalization,
                                               tracesnormalize.NoNormalization):
        df = pipeline.Chain(pipeline.BySession(),
                                    normalization,
                            pipeline.RecombineResults(),
        ).run(df.copy())
    return df
