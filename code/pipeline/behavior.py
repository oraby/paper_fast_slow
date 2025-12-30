from .pipeline import DFProcessor
from ..behavior.util.splitdata import grpBySess
import numpy as np
import pandas as pd


class CountContPrevOutcome(DFProcessor):
    def __init__(self, no_choice_as_incorrect : bool=True, extraCountsFn=None):
        '''Example usage of extraCountsFn:
        _last_sess_df = df[df.ShortName == df.ShortName.iloc[0]]
        def prevReward(row):
            nonlocal _last_sess_df
            if row.ShortName != _last_sess_df.ShortName.iloc[0]:
                _last_sess_df = df[df.ShortName == row.ShortName]
            rel_rows = _last_sess_df[_last_sess_df.TrialNumber ==
                                     row.TrialNumber]
            rel_row = rel_rows[rel_rows.epoch == "Feedback Wait"]
            # assert len(rel_row) == 1,
            if len(rel_row) != 1:
                print(f"No epoch in {rel_rows.epoch.values}")
                return {}
            # print("rel_row:", rel_row)
            rel_row = rel_row.iloc[0]
            activity = {area_id:activity[0]
                        for area_id, activity in
                                        rel_row.traces_sets["neuronal"].items()}
            return activity
        '''
        self._no_choice_as_incorrect = no_choice_as_incorrect
        self._extraCountsFn = extraCountsFn

    def process(self, df):
        df = df.copy()
        # if self._warn:
        #     print("Take care that we only iterate on the whole df, so "
        #           "state_df or df unsplit by sessions wouldn't work")
        sess_dfs = []
        for sess, sess_df in grpBySess(df):
            sess_df = self._processSession(sess_df)
            sess_dfs.append(sess_df)
        return pd.concat(sess_dfs)

    def _processSession(self, df):
        def count(prev_count, current_res, no_choice_as_incorrect):
            if np.isnan(current_res):
                if no_choice_as_incorrect:
                    prev_count = prev_count - 1 if prev_count < 0 else -1
                else:
                    prev_count = 0
            elif not current_res:
                prev_count = prev_count - 1 if prev_count < 0 else -1
            elif current_res:
                prev_count = prev_count + 1 if prev_count > 0 else 1
            return prev_count
        prev_outcome_count = 0
        prev_direction_is_left_count = 0
        prev_left_rewarded_count = 0
        prev_dv = np.nan
        prev_calcStimulusTime1 = np.nan
        prev_calcStimulusTime2 = np.nan
        prev_calcStimulusTime3 = np.nan
        prev_calcStimulusTime4 = np.nan
        prev_extra_counts = {}
        last_trial_num = df.TrialNumber.min() - 1
        df = df.copy()
        df["PrevOutcomeCount"] = np.nan
        df["PrevDirectionIsLeftCount"] = np.nan
        df["PrevLeftRewardedCount"] = np.nan
        df["PrevDV"] = np.nan
        df["PrevCalcStimulusTime1"] = np.nan
        df["PrevCalcStimulusTime2"] = np.nan
        df["PrevCalcStimulusTime4"] = np.nan
        if len(df) == df.TrialNumber.nunique():
            loop_res = df.iterrows()#itertuples(index=False)
            single_trial = True
            trials_rows = []
        else:
            loop_res = df.groupby("TrialNumber")
            single_trial = False
            trials_dfs = []
        for _, trial_df in loop_res:
            # trial_df = loop_item if single_trial else loop_item[1]
            row = trial_df if single_trial else trial_df.iloc[0]
            trial_num = row.TrialNumber
            # assert trial_num == last_trial_num + 1, f"Gaps in the dataframe"
            if trial_num != last_trial_num + 1:
                print("last trial_num:", last_trial_num, "Trial num:",
                      trial_num)
                prev_outcome_count = 0
                prev_direction_is_left_count = 0
                prev_left_rewarded_count = 0
                prev_dv = np.nan
                prev_calcStimulusTime1 = np.nan
                prev_calcStimulusTime2 = np.nan
                prev_calcStimulusTime3 = np.nan
                prev_calcStimulusTime4 = np.nan
                prev_extra_counts = {}
            last_trial_num = trial_num
            trial_df["PrevOutcomeCount"] = prev_outcome_count
            trial_df["PrevDirectionIsLeftCount"] = prev_direction_is_left_count
            trial_df["PrevLeftRewardedCount"] = prev_left_rewarded_count
            trial_df["PrevDV"] = prev_dv
            trial_df["PrevCalcStimulusTime1"] = prev_calcStimulusTime1
            trial_df["PrevCalcStimulusTime2"] = prev_calcStimulusTime2
            trial_df["PrevCalcStimulusTime3"] = prev_calcStimulusTime3
            trial_df["PrevCalcStimulusTime4"] = prev_calcStimulusTime4

            prev_outcome_count = count(prev_outcome_count, row.ChoiceCorrect,
                            no_choice_as_incorrect=self._no_choice_as_incorrect)
            prev_direction_is_left_count = count(prev_direction_is_left_count,
                                                 row.ChoiceLeft,
                                                 no_choice_as_incorrect=False)
            prev_left_rewarded_count = count(prev_left_rewarded_count,
                                             row.LeftRewarded,
                                             no_choice_as_incorrect=False)
            prev_dv = row.DV
            prev_calcStimulusTime4 = prev_calcStimulusTime3
            prev_calcStimulusTime3 = prev_calcStimulusTime2
            prev_calcStimulusTime2 = prev_calcStimulusTime1
            prev_calcStimulusTime1 = row.calcStimulusTime
            if self._extraCountsFn is not None:
                # TODO: Handle the first trial differently by re-populating with
                #    nans using the keys from the other trials
                for key, val in prev_extra_counts.items():
                        trial_df[key] = val
                prev_extra_counts = self._extraCountsFn(row)
            if single_trial:
                trials_rows.append(trial_df)
            else:
                trials_dfs.append(trial_df)
        if single_trial:
            df = pd.DataFrame(trials_rows)
        else:
            df = pd.concat(trials_dfs)
        return df

    def _processSession2(self, df):
        def count(prev_count, current_res, no_choice_as_incorrect):
            if np.isnan(current_res) and not no_choice_as_incorrect:
                prev_count = 0
            elif not current_res or np.isnan(current_res):
                prev_count = prev_count - 1 if prev_count < 0 else -1
            elif current_res:
                prev_count = prev_count + 1 if prev_count > 0 else 1
            return prev_count
        prev_outcome_count = 0
        prev_direction_is_left_count = 0
        prev_left_rewarded_count = 0
        prev_dv = np.nan
        prev_extra_counts = {}
        last_trial_num = df.TrialNumber.min() - 1
        trials_dfs = []
        for trial_num, trial_df in df.groupby("TrialNumber"):
            # assert trial_num == last_trial_num + 1, f"Gaps in the dataframe"
            if trial_num != last_trial_num + 1:
                print("last trial_num:", last_trial_num, "Trial num:",
                      trial_num)
                prev_outcome_count = 0
                prev_direction_is_left_count = 0
                prev_left_rewarded_count = 0
                prev_dv = np.nan
                prev_extra_counts = {}
            last_trial_num = trial_num
            row = trial_df.iloc[0] # A representative to the whole trial
            trial_df = trial_df.copy()
            trial_df["PrevOutcomeCount"] = prev_outcome_count
            trial_df["PrevDirectionIsLeftCount"] = prev_direction_is_left_count
            trial_df["PrevLeftRewardedCount"] = prev_left_rewarded_count
            trial_df["PrevDV"] = prev_dv
            prev_outcome_count = count(prev_outcome_count, row.ChoiceCorrect,
                            no_choice_as_incorrect=self._no_choice_as_incorrect)
            prev_direction_is_left_count = count(prev_direction_is_left_count,
                                                 row.ChoiceLeft,
                                                 no_choice_as_incorrect=False)
            prev_left_rewarded_count = count(prev_left_rewarded_count,
                                             row.LeftRewarded,
                                             # no_choice_as_incorrect makes no
                                             # difference here, LeftRewarded is
                                             # always known
                                             no_choice_as_incorrect=False)
            prev_dv = row.DV
            if self._extraCountsFn is not None:
                # TODO: Handle the first trial differently by re-populating with
                # nans using the keys from the other trials
                for key, val in prev_extra_counts.items():
                    trial_df[key] = val
                prev_extra_counts = self._extraCountsFn(row)
            trials_dfs.append(trial_df)
        return pd.concat(trials_dfs)

    def descr(self) -> str:
        return "Counting the number continuous previous trials outcomes"

class CreateCorrectIncorrectCopies(DFProcessor):
    '''Use just before ConcatEpochs() to have the same copy of your data except
    one is with reward and another is with punishment phase'''
    def process(self, data):
        # TODO: Add different session handler for each group
        correct_df = data.copy()
        incorrect_df = data.copy()
        correct_df = correct_df[~correct_df.epoch.str.contains("Punish")]
        incorrect_df = incorrect_df[~incorrect_df.epoch.str.contains("Reward")]
        # TODO: data["TrialNumber"].astype(str)
        correct_df["TrialNumber"] = correct_df["TrialNumber"] + " Reward"
        incorrect_df["TrialNumber"] = incorrect_df["TrialNumber"] + " Punish"
        correct_df["ChoiceCorrect"] = 1
        incorrect_df["ChoiceCorrect"] = 0
        return [correct_df, incorrect_df]

    def descr(self):
        return ("Duplicating traces, both with same initial epochs but one "
                "contains only reward epoch while the other contains only "
                "punishment epoch")

class Diff(DFProcessor):
    def __init__(self, col_start : str, col_end : str, output_col : str):
        self._col_start = col_start
        self._col_end = col_end
        self._output_col = output_col

    def process(self, df):
        df = df.copy()
        df[self._output_col] = df[self._col_start] - df[self._col_end]
        return df

    def descr(self):
            return (f"Created col {self._output_col} as the difference between "
                    f"{self._col_end} - {self._col_start}")

class OnlyCompleteTrial(DFProcessor):
    def __init__(self, verbose : bool):
        self._verbose = verbose

    def process(self, data):
        def fltrEpochList(epochs_names):
            return (#len(epochs_names) in self._num_epochs_per_trial and
                    ("Timeout Punishment" in epochs_names or
                     "Reward" in epochs_names)
                    and "Pre-Trial Start" in epochs_names)
        if "epochs_names" in data.columns:
            return data[data.epochs_names.apply(fltrEpochList)]
        else:
            def filterStatesTrials(grp_df):
                uniq_epochs = grp_df.epoch.unique()
                assert len(uniq_epochs) == len(grp_df)
                res = fltrEpochList(uniq_epochs)
                if self._verbose and not res:
                    print(f"Trial number: {grp_df.name} - "
                          f"Epochs: {len(uniq_epochs)}: {uniq_epochs}")
                # return len(uniq_epochs) in self._num_epochs_per_trial
                return res
            return data.groupby("TrialNumber").filter(filterStatesTrials)

    def descr(self):
        return ("Removing trials without incomplete epochs (neither Reward nor "
                "Punishment)")
