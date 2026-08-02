from ..behavior.util.splitdata import splitStimulusTimeByQuantile
from ..common.imaging import (getSamplingEpoch, _alignAroundEpoch,
                              _assertUniqEpochs, _assignTraceLen,
                              _commonTraceNormalizationInit)
from ..pipeline import pipeline
from ..pipeline import tracesnormalize
from ..pipeline import tracesrestructure
from . import behaviorcommon
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


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


def _matchMovementToSampling(df_sample_n_move, df_move, col, DUP_VALS):
  # I hate this, we have to match trial for trial. A hack for now
  # is to 'rely' on floatig points values will not repeat themselves
  # between trials. We can add more matching colums later
  val_counts = df_move[col].value_counts()
  non_uniq_val_counts = val_counts[val_counts > 1*DUP_VALS]
  if len(non_uniq_val_counts):
    # print("non_uniq_val_counts:", non_uniq_val_counts)
    uniq_val_counts = val_counts[val_counts == 1*DUP_VALS]
    df_move_uniq = df_move[df_move[col].isin(uniq_val_counts.index)]
    non_uniq_df = df_move[df_move[col].isin(non_uniq_val_counts.index)]
    non_uniq_accepted_rows = []
    for _, row in non_uniq_df.iterrows():
      match_df = df_sample_n_move[(df_sample_n_move.ShortName == row.ShortName) &
                                  (df_sample_n_move.TrialNumber == row.TrialNumber)]
      match_len = len(match_df)
      assert match_len <= 1
      if match_len:
        non_uniq_accepted_rows.append(row)

    print("Len before dups:", len(non_uniq_df), "Len after:",
          len(non_uniq_accepted_rows))
    non_uniq_df = pd.DataFrame(non_uniq_accepted_rows)
    df_move = pd.concat([df_move_uniq, non_uniq_df])
    df_sample_n_move_unmatch = df_sample_n_move[
                                      ~df_sample_n_move[col].isin(df_move[col])]
    if len(df_sample_n_move_unmatch):
      print("********* We are going to crash next")
      # print([col for col in df_bin.columns if "max" in col.lower()])
      display(df_sample_n_move_unmatch[["ShortName", "TrialNumber", "MaxTrial",
                                        "ChoiceCorrect", "epoch_time", col]])
      print("Number of unmatches:", len(df_sample_n_move_unmatch),
            "- from ", len(df_sample_n_move))
      if len(df_sample_n_move_unmatch) == 1: # we can handle losing one
        df_sample_n_move = df_sample_n_move[df_sample_n_move.index !=
                                            df_sample_n_move_unmatch.index[0]]
  # df_move = df_move[df_move.calcDecisionTime.isin(
  #                                          df_sample_n_move.calcDecisionTime)]
  return df_sample_n_move, df_move

def _commonAlign(df, epoch, normalization, min_time, max_time,
                time_increment, time_before_or_after, fixed_cut : bool,
                callback):
  # df = df.copy()
  is_sampling = epoch == "Sampling"
  cut_bins = np.arange(min_time, max_time+time_increment, time_increment)
  # Floating-point shit, make sure we didn't include the step afterwards
  cut_bins = cut_bins[cut_bins <= max_time]
  print("Cut bins:", cut_bins)
  df = df[df.ChoiceCorrect.notnull()]
  df = df[df.calcStimulusTime.notnull()]
  # Execute normalization now to execute across all data
  assert (df.sole_owner == False).all()
  df = _commonTraceNormalizationInit(df, normalization=normalization,
                                     rename_feedback_epoch=False)
  # df_align = df[df.epoch == epoch]
  df_align = df[df.epoch.isin(["Sampling", "Movement to Lateral Port"])]
  # We always bisect on sampling
  df_sampling = df[df.epoch == "Sampling"]
  del df # No longer needed, raise an error if used
  for _bin, df_bin in tqdm(df_sampling.groupby(
                                pd.cut(df_sampling.epoch_time, bins=cut_bins))):
    df_bin = df_bin.copy()
    # Not sure why we have repeated indices, but we need to reset it as we might
    # filter later on unique indices
    df_bin = df_bin.reset_index(drop=True)
    # if not (abs(_bin.left - 0.9) < 0.00001):
    #   continue
    print("Bin:", _bin.left,
     f"{df_bin.calcStimulusTime.min() = }, {df_bin.calcStimulusTime.max() = }, "
     f"{df_bin.epoch_time.min() = },  {df_bin.epoch_time.max() = }")

    time_around = _bin.left if not fixed_cut else min_time
    if is_sampling:
      df_align_trial_to_epoch = _alignAroundEpoch(df_bin, align_to_epoch=epoch,
                                              time_before=time_before_or_after,
                                              time_after=time_around,
                                              limit_to_epoch_end="Sampling",
                                              normalization=None)
    else:
      col = "calcStimulusTime"
      df_move = df_align[df_align[col].isin(df_bin[col])]
      DUP_VALS = 2 # One for sampling, and another for movement
      df_bin, df_move = _matchMovementToSampling(df_bin, df_move, col, DUP_VALS)
      assert len(df_move) == len(df_bin)*DUP_VALS, (
                                 f"{len(df_move)= } != {len(df_bin)*DUP_VALS=}")
      df_align_trial_to_epoch = _alignAroundEpoch(df_move, align_to_epoch=epoch,
                                                time_before=time_around,
                                                time_after=time_before_or_after,
                                                limit_to_epoch_start="Sampling",
                                                normalization=None)
    callback(df_align_trial_to_epoch, _bin, is_sampling)

def getSamplingVario(df, epoch, normalization, min_time, max_time,
                     time_increment, time_before_or_after):
  '''sampling_start_df = getSampling(df_all_by_epoch.copy(), epoch="Sampling")
  movement_start_df = getSampling(df_all_by_epoch.copy(), epoch="Movement to Lateral Port")
  '''
  res_df_li = []
  is_sampling_glbl = None
  def _executeForBinCb(df_align_trial_to_epoch, _bin, is_sampling):
    nonlocal res_df_li, is_sampling_glbl
    df_concat_align_trial_to_epoch = behaviorcommon.concatenateEpochs(
                                         df_align_trial_to_epoch, by_trial=True)
    # For some reason, some trials Samplnig ends with +1 in the end, so we need
    # to trim it
    cut_from_beginning = False if is_sampling else True # more verbose
    df_concat_align_trial_to_epoch = _assignTraceLen(df_concat_align_trial_to_epoch)
    print("For bin:", _bin.left, "to:", _bin.right)
    display(df_concat_align_trial_to_epoch[["epoch", "trace_len", "trace_start_idx",
                                            "trace_end_idx", "epochs_names", "epochs_ranges"]])
    cutter = tracesrestructure.CutLongTraces(
                         maxTraceLenFn=lambda traces_lens:traces_lens.min() - 1,
                         cut_from_beginning=cut_from_beginning,)
    df_concat_cut_to_size = cutter.process(df_concat_align_trial_to_epoch)
    df_concat_cut_to_size = _assignTraceLen(df_concat_cut_to_size)
    display(df_concat_cut_to_size[["epoch", "trace_len", "trace_start_idx"]])
    df_concat_cut_to_size = df_concat_cut_to_size.copy() # Avoid defrtag warning
    # df_concat_cut_to_size = df_concat_align_trial_to_sampling.copy()
    df_concat_cut_to_size["dur_bin"] = _bin
    # df_concat_cut_to_size["bin_left"] = _bin.left
    # df_concat_cut_to_size["bin_right"] = _bin.right
    res_df_li.append(df_concat_cut_to_size)
    is_sampling_glbl = is_sampling
  _commonAlign(df, epoch, normalization, min_time, max_time,
                time_increment, time_before_or_after, fixed_cut=False,
                callback=_executeForBinCb)
  assert is_sampling_glbl is not None
  res_df = pd.concat(res_df_li)

  extend_at_beginning = False if is_sampling_glbl else True # Again, more verbose
  offset_epochs_ranges_starting_epoch = None if is_sampling_glbl else \
                                        "Movement to Lateral Port"

  extender = tracesrestructure.ExtendShortTraces(minTraceLenFn=np.max,
        extend_at_beginning=extend_at_beginning,
        offset_epochs_ranges_starting_epoch=offset_epochs_ranges_starting_epoch)
  res_df = extender.process(res_df)
  res_df = _assignTraceLen(res_df)
  display(res_df[["epoch", "trace_len", "trace_start_idx"]])
  return res_df


def getSamplingFixed(df, normalization, epoch, min_time, max_time,
                     time_increment, time_before_or_after):
  '''sampling_start_df = alignAroundSampling(df_all_by_epoch.copy())
  movement_start_df = getSampling(df_all_by_epoch.copy(), epoch="Movement to Lateral Port")
  '''
  # df = df.copy()
  assert time_before_or_after >= 0
  res_df_li = []
  is_sampling_glbl = None
  def _executeForBinCb(df_align_trial_to_epoch, _bin, is_sampling):
    nonlocal res_df_li, is_sampling_glbl
    display(df_align_trial_to_epoch[["epoch", "trace_start_idx", "trace_end_idx"]])
    df_concat_align_trial_to_epoch = behaviorcommon.concatenateEpochs(
                                         df_align_trial_to_epoch, by_trial=True)
    df_concat_cut_to_size = df_concat_align_trial_to_epoch
    df_concat_cut_to_size = _assignTraceLen(df_concat_cut_to_size)
    display(df_concat_cut_to_size[["epoch", "trace_len", "trace_start_idx", "trace_end_idx", "epochs_names", "epochs_ranges"]])
    df_concat_cut_to_size = df_concat_cut_to_size.copy() # Avoid defrtag warning
    df_concat_cut_to_size["dur_bin"] = _bin
    # Avoid defrag warning by copying
    res_df_li.append(df_concat_cut_to_size.copy())
    is_sampling_glbl = is_sampling
  _commonAlign(df, epoch, normalization, min_time, max_time,
               time_increment, time_before_or_after, fixed_cut=True,
               callback=_executeForBinCb)
  assert is_sampling_glbl is not None
  res_df = pd.concat(res_df_li)
  return res_df

def getSamplingQuantiles(df, normalization, epoch, quantiles, time_before,
                         time_after, drop_unequal_len : bool,
                         use_assignDVStr=True, cut_above_sec=None):
  assert time_before >= 0
  assert time_after >= 0
  df = df[df.ChoiceCorrect.notnull()]
  df = df[df.calcStimulusTime.notnull()]
  # df_sampling = df[df.epoch == "Sampling"]
  # df_sampling = df_sampling.copy()
  assert (df.sole_owner == False).all()
  df = _commonTraceNormalizationInit(df, normalization=normalization,
                                     rename_feedback_epoch=False)
  valid_epochs = ["Sampling", "Movement to Lateral Port"]
  assert epoch in valid_epochs
  if epoch == "Sampling":
    df = df[df.epoch == epoch]
    df["SamplingEpochTime"] = df.epoch_time
  else:
    # Slower but hopefully should be more robust
    df = df.copy()
    df = df[df.epoch.isin(valid_epochs)]
    df["SamplingEpochTime"] = np.nan
    for trial_info, trial_df in df.groupby(
                                 ["Name", "Date", "SessionNum", "TrialNumber"]):
      trial_movement = trial_df[trial_df.epoch == "Movement to Lateral Port"]
      if len(trial_movement) < 1:
        continue
      assert len(trial_movement) == 1
      trial_sampling = trial_df[trial_df.epoch == "Sampling"]
      # It can happen in very extreme cases that only movement would exist,
      # which would happen if imaging started during the movement epoch
      assert len(trial_sampling) == 1
      # Set everything as we need to keep, not just movement
      df.loc[trial_df.index, "SamplingEpochTime"] = \
                                               trial_sampling.epoch_time.iloc[0]
    # df = df[df.epoch == epoch] # I need the sampling epoch as well for
    # _alignAroundEpoch()
    # assert df.SamplingEpochTime.notnull().all()
  # TODO: Do the same check for the epoch before/after
  len_before = len(df)
  df = df[df.SamplingEpochTime >= time_after]
  _assertUniqEpochs(df)
  print(f"Dropped {len_before - len(df)}/{len_before} short sampling time")
  df_grps = splitStimulusTimeByQuantile(df, quantiles=quantiles,
                                        use_assignDVStr=use_assignDVStr,
                                        separate_by_session=True,
                                        cut_above_sec=cut_above_sec)
  limit_to_epoch_start = "Sampling" if epoch == "Movement to Lateral Port" else None
  limit_to_epoch_end =   "Sampling" if epoch == "Sampling" else None
  res_df_li = []
  for quantile, df_quantile in df_grps:
    print("Epoch:", epoch, "Quantile:", quantile, "len:", len(df_quantile))
    df_quantile = alignAroundEpoch(df_quantile, epoch=epoch,
                                   time_before=time_before,
                                   time_after=time_after,
                                   normalization=None,
                                   drop_unequal_len=drop_unequal_len)
    df_quantile["quantile_idx"] = quantile
    res_df_li.append(df_quantile)
  res_df = pd.concat(res_df_li)
  display(res_df[["epoch", "quantile", "quantile_idx", "trace_len", "trace_start_idx", "trace_end_idx", "epochs_names", "epochs_ranges"]])
  return res_df

def alignAroundEpoch(df, epoch, time_before, time_after, normalization,
                     drop_unequal_len):
  SAMPLING, MOVEMENT = "Sampling", "Movement to Lateral Port"
  assert epoch in [SAMPLING, MOVEMENT]
  limit_to_epoch_start = SAMPLING if epoch == MOVEMENT else None
  limit_to_epoch_end =   SAMPLING if epoch == SAMPLING else None
  _assertUniqEpochs(df)
  # for sess, sess_df in grpBySess(df): # No need, for easier debugging
  df = _alignAroundEpoch(df, align_to_epoch=epoch, time_before=time_before,
                         time_after=time_after,
                         limit_to_epoch_start=limit_to_epoch_start,
                         limit_to_epoch_end=limit_to_epoch_end,
                         normalization=normalization)
  df = behaviorcommon.concatenateEpochs(df, by_trial=True)
  df = _assignTraceLen(df)
  if drop_unequal_len:
    # Bad hack that we should fix later
    df = df[df.trace_len == df.trace_len.mode()[0]]
    assert df.trace_len.nunique() == 1, display(df.trace_len.value_counts())
  return df
