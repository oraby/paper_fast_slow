from ..behavior.util.splitdata import splitStimulusTimeByQuantile
# from ....common.analysis.alignsampling import getSamplingEpoch
from ..pipeline import pipeline, tracesnormalize, tracesrestructure, traceswidth
try:
     from IPython.display import display
except ImportError:
     pass
import pandas as pd
from itertools import chain as iterchain
from typing import Literal


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
                  decision_epoch_filter : Literal["remove", "keep"]="remove"):
    assert decision_epoch_filter in ("remove", "keep")
    if num_quantiles is not None:
        if not isinstance(normalization, tracesnormalize.NoNormalization):
            assert normalize_sessions_before_splitting is not None, (
                "normalize_sessions_before_splitting must be specified "
                "when num_quantilesis specified")
    if sep_above_quantile_sec is not None:
        assert num_quantiles is not None, ("sep_above_quantiel_sec can only "
            "be used when num_quantiles is specified. sep_above_quantiel_sec "
            "separates very long trials into their own special bin")
        if filter_above_sec is not None:
            assert filter_above_sec > sep_above_quantile_sec, (
                "filter_above_sec trials are removed, while "
                "sep_above_quantile_sec are kept as a special quantile bin")
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

    if decision_epoch_filter == "remove":
        df = df.groupby(["Name", "Date", "SessionNum", "TrialNumber"]) \
               .filter(lambda trial_df:(not any(trial_df.epoch ==
                                                                "Decision Time")
                                        and any(trial_df.epoch ==
                                                   "Movement to Lateral Port")))
    else:
        df = df.groupby(["Name", "Date", "SessionNum", "TrialNumber"]) \
               .filter(lambda trial_df:any(trial_df.epoch == "Decision Time"))
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
                                       epoch="Sampling",
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
        df_sampling = df[df.epoch == "Sampling"] # Don't have duplicates
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

def getSamplingEpoch(df, normalization, epoch, time_epoch_before,
                     time_epoch_after, *, normalize_after_cutting,
                     rename_feedback_epoch=False,
                     normalize_epoch_time=False,
                     concatenate_final_epochs=False):
    assert time_epoch_before >= 0
    assert time_epoch_after >= 0
    df = df[df.ChoiceCorrect.notnull()]
    df = df[df.calcStimulusTime.notnull()]
    # We don't have to run the normalization yet, we will run it later because
    # the df should still have the full trace (i.e sole_owner = False)
    assert (df.sole_owner == False).all()
    if not normalize_after_cutting:
        df = _commonTraceNormalizationInit(df, normalization=normalization,
                                    rename_feedback_epoch=rename_feedback_epoch)
    df_trials_tup_li = []
    # Drop trials where not much enough time before or after
    # assert epoch == "Sampling", (
    #      f"didn't handle it for {epoch} yet to check for epochs after/before")
    for trial_info, trial_df in df.groupby(
                                 ["Name", "Date", "SessionNum", "TrialNumber"]):
        epoch_row_before = None
        epoch_row_cur = None
        epoch_row_after = None
        stop_next = False
        old_row_index = trial_df.index[0] - 1
        for row_index, row in trial_df.iterrows():
            assert row_index - 1 == old_row_index, (
                 f"Unsorted or missing df - old idx={row_index} - "
                 f"new idx: {row_index}"+
                 # A hack to combine display with print
                 ("" if (display(trial_df[trial_df.index == old_row_index][
                                          ["TrialNumber", "DVstr", "epoch"]]) or
                         display(trial_df[trial_df.index == row_index][
                                          ["TrialNumber", "DVstr", "epoch"]]))
                  else ""))
            if stop_next:
                epoch_row_after = row
                break
            elif row.epoch != epoch:
                epoch_row_before = row
            else:
                stop_next = True
                epoch_row_cur = row
            old_row_index = row_index
        assert epoch_row_before is not None
        assert epoch_row_cur is not None, (
            print("Didn't find", epoch, "in:") or display(trial_df.epoch) or
            print(f"where trial number: ", trial_df.TrialNumber.unique()))
        if trial_info[-1] != trial_df.MaxTrial.iloc[0]:
            assert epoch_row_after is not None, display(trial_df) or (
                           f"Trial info: {trial_info} - row index: {row_index}")
        elif epoch_row_after is None:
            continue
        if epoch_row_before.epoch_time < time_epoch_before:
            continue
        elif epoch_row_after.epoch_time < time_epoch_after:
            continue
        df_trials_tup_li.append(
                             (epoch_row_before, epoch_row_cur, epoch_row_after))
    before_cur_df = [(before, cur) for before, cur, after in df_trials_tup_li]
    before_cur_df = list(iterchain(*before_cur_df))
    before_cur_df = pd.DataFrame(before_cur_df)
    before_cur_df =    _alignAroundEpoch(before_cur_df, align_to_epoch=epoch,
                                         time_before=time_epoch_before,
                                         time_after=0,
                                         limit_to_epoch_start=False,
                                         limit_to_epoch_end=False,
                                         normalization=None)
    before_cur_df = before_cur_df[before_cur_df.epoch != epoch]
    before_cur_df = _assignTraceLen(before_cur_df)
    # display(before_cur_df[["epoch", "trace_len"]])
    # display(before_cur_df.trace_len.unique())

    # It is annoying to repeat this, we can optimize it later
    after_df = pd.DataFrame([after for before, cur, after in df_trials_tup_li],
                                                    columns=df.columns)
    epoch_after = after_df.epoch.unique()
    assert len(epoch_after) == 1, (
           f"Didn't implement it yet for muliple epochs - Found: {epoch_after}")
    epoch_after = epoch_after[0]
    del after_df # Don't use by mistake
    after_cur_df = [(cur, after) for before, cur, after in df_trials_tup_li]
    assert len(before_cur_df) == len(after_cur_df), (
        "Unexpected mistmatch (1) - "
        f"{len(before_cur_df) = } != {len(after_cur_df) = }")
    after_cur_df = list(iterchain(*after_cur_df))
    after_cur_df = pd.DataFrame(after_cur_df)
    after_cur_df =    _alignAroundEpoch(after_cur_df,
                                        align_to_epoch=epoch_after,
                                        time_after=time_epoch_after,
                                        # Set very large number, we will stop
                                        # before at the epoch before
                                        time_before=500,
                                        limit_to_epoch_start=epoch,
                                        limit_to_epoch_end=False,
                                        normalization=None)
    after_cur_df = after_cur_df[after_cur_df.epoch == epoch_after]
    after_cur_df = _assignTraceLen(after_cur_df)
    # display(after_cur_df[["epoch", "trace_len"]])
    # display(after_cur_df.trace_len.unique())
    assert len(before_cur_df) == len(after_cur_df), "Unexpected mistmatch (2)"

    cur_df = [cur for before, cur, after in df_trials_tup_li]
    assert len(cur_df) == len(after_cur_df), "Unexpected mistmatch (3)"
    cur_df = pd.DataFrame(cur_df)
    if normalize_epoch_time:
        chain = pipeline.Chain(
            # pipeline.BySession(), pipeline.By("TrialNumber"),
                traceswidth.NormalizeTime(),
            # pipeline.RecombineResults()
        )
        cur_df = chain.run(cur_df)
        assert len(cur_df) == len(after_cur_df), "Unexpected mistmatch (4)"
    # Can't use zip: https://stackoverflow.com/a/54323818/11996983
    # all_rows_df = list(iterchain(zip(before_cur_df, cur_df, after_cur_df)))
    all_rows_df = []
    for idx in range(len(before_cur_df)):
        all_rows_df += [before_cur_df.iloc[idx], cur_df.iloc[idx],
                                        after_cur_df.iloc[idx]]
    assert len(all_rows_df) == len(all_rows_df), "Unexpected mismatch (5)"
    all_rows_df = pd.DataFrame(all_rows_df)
    # display(all_rows_df)
    if concatenate_final_epochs:
        chain = pipeline.Chain(
            pipeline.BySession(), pipeline.By("TrialNumber"),
                tracesrestructure.ConcatEpochs(ignore_existing_concat=True,
                                               assume_continuos=True),
            pipeline.RecombineResults()
        )
        all_rows_df = chain.run(all_rows_df)
    if normalize_after_cutting:
        chain = pipeline.Chain(
            pipeline.BySession(),
                pipeline.ApplyFullTrace(normalization),
            pipeline.RecombineResults(),
        )
        all_rows_df = chain.run(all_rows_df)
        assert len(all_rows_df) == len(all_rows_df), "Unexpected mistmatch (6)"
    all_rows_df = _assignTraceLen(all_rows_df)
    # display(all_rows_df.trace_len.unique())
    return all_rows_df

def _alignAroundEpoch(df, align_to_epoch, time_before, time_after,
                      normalization, limit_to_epoch_start=[],
                      limit_to_epoch_end=[]):
    # df = df[df.epoch == align_to_epoch]
    if normalization is not None:
        normalization_li = [pipeline.BySession(),
                                pipeline.ApplyFullTrace(normalization),
                            pipeline.RecombineResults()
    ]
    else:
        normalization_li = []
    df = df.copy()
    assert all(df.sole_owner == False), "all rows traces should be full trace"
    chain = pipeline.Chain(
        *normalization_li,
        # pipeline.BySession(),    # Is this needed?
        tracesrestructure.AlignTraceAroundEpoch(epoch_name_li=[align_to_epoch],
                                    time_before_sec=time_before,
                                    time_after_sec=time_after,
                                    limit_to_epoch_start=limit_to_epoch_start,
                                    limit_to_epoch_end=limit_to_epoch_end),
        # pipeline.By(),
        #         pipeline.TraceAvg(avg_rows=True, avg_row_traces=False),
        #                           #trial_name_df_col="epoch"),
        pipeline.RecombineResults(),
    )
    return chain.run(df)

def _commonTraceNormalizationInit(df, normalization,
                                  rename_feedback_epoch : bool):
    df = df.copy()
    chain = pipeline.Chain(
        pipeline.BySession(),
            pipeline.ApplyFullTrace(normalization),
        pipeline.RecombineResults(),
    )
    df = chain.run(df)
    if rename_feedback_epoch:
        def addHasFeedbackWait(sess_df):
            sess_df["HasFeedbackWait"] = any(sess_df.epoch == "Feedback Wait")
            return sess_df
        df = df.groupby("anlys_path", group_keys=False).apply(
                                                             addHasFeedbackWait)
        df.loc[df.HasFeedbackWait & (df.epoch == "Feedback Wait"),
               "epoch"] = "Feedback Start"
        df.loc[~df.HasFeedbackWait & df.epoch.isin(["Reward", "Punishment"]),
               "epoch"] = "Feedback Start"
    return df

def _assertUniqEpochs(df):
    uniq_df = df[["Name", "Date", "SessionNum", "TrialNumber", "epoch"]]
    assert len(uniq_df) == len(uniq_df.drop_duplicates())

def _assignTraceLen(df):
    df = df.copy()
    df["trace_len"] = df.trace_end_idx - df.trace_start_idx + 1
    return df
