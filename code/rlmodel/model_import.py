# import pyddm
# import pyddm.plot
# import matplotlib.pyplot as plt
# import numba
import numpy as np
import pandas as pd

#CHOICE_COL_NAME = "ChoiceLeft"
CHOICE_COL_NAME = "ChoiceCorrect"
_CHOICE_NAME_KARGS = dict(choice_names=("Left", "Right")) if CHOICE_COL_NAME == "ChoiceLeft" else {}


LOG_CIEL = .01
_ciel_max = np.log(1/LOG_CIEL)
# Initial position should be between -1 and 1
# initialBiasFn = lambda BIAS_COEF, Q_L, Q_R: BIAS_COEF * np.log(max(Q_L, LOG_CIEL) / max(Q_R, LOG_CIEL)) / ciel_max
# def initialBiasFn(BIAS_COEF, Q_val, ALPHA, SessId):
def initialBiasFn(BIAS_COEF, Q_val, ALPHA):
    # print(f"BIAS_COEF: {BIAS_COEF} - Q_L: {Q_L} - Q_R: {Q_R}")
    # return (BIAS_COEF * np.log(np.clip(np.array(Q_L), LOG_CIEL, 1) / np.clip(np.array(Q_R), LOG_CIEL, 1))) / _ciel_max
    return np.clip(BIAS_COEF * Q_val, -1, 1)

from scipy import stats
class InitBiasDist:
    _q_vals = None

    @staticmethod
    def setDist(q_vals):
        assert q_vals.min() >= -1
        assert q_vals.max() <= 1
        InitBiasDist._q_vals = q_vals
        # self._pdf = stats.gaussian_kde(q_vals)


    @staticmethod
    def initialBiasFn(BIAS_COEF, ALPHA, x):
        # This breaks with self inside pyddm, so we use it as a staticmethod
        '''From the PyDDM documentaiton:
        Instead of starting point coming from a fixed point, it can also come from a distribution. To do this, the
        “starting_point” in gddm() can accept the variable “x”, which is the domain of the distribution, i.e., all of
        the potential starting points over which the distribution is defined. Then it must output a vector of the
        same length as x describing the probability density at each point. (Since the starting point is
        not fixed in general, and may even vary trial to trial, the length of x may be different on each trial.) If
        the output does not sum to 1, it will be renormalised.'''
        # Create a pdf distribution from the Q values
        # return self._pdf(x)
        scaled = np.clip(BIAS_COEF * InitBiasDist._q_vals, -1, 1)
        dist = stats.gaussian_kde(scaled)
        return dist(x)


class DriftGain:
    Name = "DriftGain"
    @staticmethod
    def driftFn(DRIFT_COEF, RewardRate, DV, BETA): # Force having ALPHA and BETA to include as parameters
        #   print(f"DRIFT_COEF: {DRIFT_COEF} - RewardRate: {RewardRate} - DV: {DV}")
        # return DRIFT_COEF * np.outer(RewardRate, DV).squeeze()
        return DRIFT_COEF * RewardRate * DV

    @staticmethod
    def driftAbsFn(DRIFT_COEF, RewardRate, DVabs, BETA):
        return DRIFT_COEF * RewardRate * DVabs

    @staticmethod
    # def noiseFn(NOISE_SIGMA, SessId):
    def noiseFn(NOISE_SIGMA):
        return NOISE_SIGMA

class NoiseGain:
    Name = "NoiseGain"
    @staticmethod
    def driftFn(DRIFT_COEF, DV, BETA):
        return DRIFT_COEF * DV

    @staticmethod
    def driftAbsFn(DRIFT_COEF, DVabs, BETA):
        return DRIFT_COEF * DVabs

    @staticmethod
    # def noiseFn(NOISE_SIGMA, RewardRate, SessId):
    def noiseFn(NOISE_SIGMA, RewardRate):
        return RewardRate*NOISE_SIGMA


class DriftNoiseGain:
    Name = "DriftNoiseGain"
    @staticmethod
    def driftFn(DRIFT_COEF, RewardRate, DV, BETA):
        return DRIFT_COEF * RewardRate * DV

    @staticmethod
    def driftAbsFn(DRIFT_COEF, RewardRate, DVabs, BETA):
        return DRIFT_COEF * RewardRate * DVabs

    @staticmethod
    # def noiseFn(NOISE_SIGMA, RewardRate, SessId):
    def noiseFn(NOISE_SIGMA, RewardRate):
        return max(RewardRate, .001)*NOISE_SIGMA

def driftOnlyCohrence(DRIFT_COEF, DV):
    return DRIFT_COEF * DV

def driftOnlyCohrenceAbs(DRIFT_COEF, DVabs):
    return DRIFT_COEF * DVabs

def makeModel(GainClass, include_Q, include_RewardRate, T_dur=5.0,
              include_choice_outcome=False, intiialBiasDistInst=None):
    print("Remove non-decision time once we include EWD trials")
    gddm_kwargs = {}
    if CHOICE_COL_NAME == "ChoiceLeft":
        conditions = ["DV"]
        assert include_choice_outcome
        conditions += ["ChoiceCorrect"]
    else:
        conditions = ["DVabs"]

    # conditions += ["SessId"]
    # parameters={"NonDecisionTime": 0.15, #(0, .5),
    #             "Bound": 1,#6, # (1, 5),
    #             # "Bound": (1, 10),
    #             "DRIFT_COEF": (0, 10),
    #             "NOISE_SIGMA": (0.01, 10),
    # }
    parameters={"NonDecisionTime": .2,#.15, #(0, .5),
                "Bound": 2,#6, # (1, 5),
                # "Bound": (1, 10),
                "DRIFT_COEF": (0.01, 8),
                "NOISE_SIGMA": (0.01, 4),
    }
    if include_Q:
        if intiialBiasDistInst is None:
            gddm_kwargs["starting_position"] = initialBiasFn
            conditions += ["Q_val"]
            parameters["BIAS_COEF"] = (0, 3)
        else:
            gddm_kwargs["starting_position"] = intiialBiasDistInst.initialBiasFn
            parameters["BIAS_COEF"] = (-2, 2)
        parameters["ALPHA"] = (0.01, 1) # .7

    if include_RewardRate:
        gddm_kwargs["drift"] = GainClass.driftAbsFn if "DVabs" in conditions else GainClass.driftFn
        conditions += ["RewardRate"]
        parameters["BETA"] = (0.01, 1) # .8
    else:
        gddm_kwargs["drift"] = driftOnlyCohrenceAbs if "DVabs" in conditions else driftOnlyCohrence

    m = pyddm.gddm(
        **gddm_kwargs,
        noise=GainClass.noiseFn,
        nondecision="NonDecisionTime",#nondecision_time,
        # bound=2, #"Bound",
        bound="Bound",
        parameters=parameters,
        conditions=conditions,
        T_dur=T_dur,
        name=GainClass.Name,
        **_CHOICE_NAME_KARGS)
    return m


def groupDV(df):
    df = df.copy()
    df["DVabs"] = df.DV.abs()
    # Approximate every cloase DV together
    GROUP_EVERY = .3
    GROUP_EVERY_DEC = int(GROUP_EVERY*10)
    df["DVabs"] = (df.DVabs *10 // GROUP_EVERY_DEC) * GROUP_EVERY_DEC / 10
    return df

def prepareDF(df_behavior_rl, T_dur, include_Q, include_RewardRate, short_sample=False):
    if short_sample:
        # df_behavior_rl = df_behavior_rl[df_behavior_rl.Name == "Rbp4_M2_1"]
        df_behavior_rl = df_behavior_rl[df_behavior_rl.Name == "Avgat1"]
    df_behavior_rl_filtered = df_behavior_rl.query("ChoiceCorrect.notnull() and ChoiceLeft.notnull()")
    df_behavior_rl_filtered = df_behavior_rl_filtered.sort_values(["Name", "Date", "SessionNum", "TrialNumber"])
    # Why do we have null DV?
    assert df_behavior_rl_filtered.DV.notnull().all()
    # df_behavior_rl_filtered = df_behavior_rl_filtered.query("DV.notnull()")
    df_behavior_rl_filtered = groupDV(df_behavior_rl_filtered)

    keep_cols = ["Name", "Date", "SessionNum", "TrialNumber",
                 "ChoiceLeft", "ChoiceCorrect", "calcStimulusTime", "DV", "DVabs"]
    if include_Q:
        keep_cols += ["Q_val", "Q_L", "Q_R"]
    if include_RewardRate:
        keep_cols += ["RewardRate"]
    df_behavior_rl_filtered = df_behavior_rl_filtered[keep_cols]
    df_behavior_rl_filtered = df_behavior_rl_filtered.copy()
    df_behavior_rl_filtered["SessId"] = df_behavior_rl_filtered.apply(
                lambda x: f"{x['Name']}_{x['Date']}_{x['SessionNum']}", axis=1)

    ID_COL = "SessId"
    # df_behavior_rl_filtered = df_behavior_rl_filtered.drop(columns=["Date", "SessionNum", _NOT_ID_COL])
    df_behavior_rl_filtered = df_behavior_rl_filtered.rename(columns={"calcStimulusTime": "RT",
                                                                      })
    df_behavior_rl_filtered = df_behavior_rl_filtered[df_behavior_rl_filtered.RT <= T_dur]
    # display(df_behavior_rl_filtered)
    sample = pyddm.Sample.from_pandas_dataframe(df_behavior_rl_filtered,
                                                rt_column_name="RT",
                                                choice_column_name=CHOICE_COL_NAME,
                                                **_CHOICE_NAME_KARGS,
                                                )
    # TODO: Use DVstr+Direction insteadd of DV, but which coherence to use?
    return sample, df_behavior_rl_filtered, ID_COL

# @numba.jit("f8,   f8,  f8", nopython=True)
def calcQVal(Q_L, Q_R, group_every=0):#round_decimals=0):
    ''' Returns a value between -1 and 1'''
    Q_L = np.clip(np.asarray(Q_L), LOG_CIEL, 1)
    Q_R = np.clip(np.asarray(Q_R), LOG_CIEL, 1)
    Q_val = np.log(Q_L / Q_R) / _ciel_max
    # if round_decimals != 0:
    #     Q_val = np.around(Q_val, round_decimals)
    if group_every != 0:
        Q_val = np.round(Q_val / group_every) * group_every
    # assert np.isnan(Q_val).sum() == 0 #, f"Q_val has NaNs: {Q_val} - Q_L: {Q_L} - Q_R: {Q_R}"
    return Q_val


def calcNewQL_QR(cur_outcome, cur_choice_left, cur_q_L, cur_q_R, ALPHA):
    '''Calculate the new Q_L and Q_R values'''
    no_decision_mask = np.isnan(cur_outcome)
    if no_decision_mask.any():
        cur_outcome = cur_outcome.copy()
        cur_outcome[no_decision_mask] = 0 # Treat no decision as incorrect
    new_q_L = cur_q_L + ALPHA*(cur_outcome - cur_q_L)
    new_q_R = cur_q_R + ALPHA*(cur_outcome - cur_q_R)
    # See which values to propagate from previous trial
    new_q_L[cur_choice_left != 1] = cur_q_L[cur_choice_left != 1]
    new_q_R[cur_choice_left != 0] = cur_q_R[cur_choice_left != 0]
    return new_q_L, new_q_R


def calcNewRewardRate(cur_outcome, cur_reward_rate, BETA, group_every=0):
    '''Calculate the new reward rate'''
    no_decision_mask = np.isnan(cur_outcome)
    if no_decision_mask.any():
        cur_outcome = cur_outcome.copy()
        cur_outcome[no_decision_mask] = 0 # Treat no decision as incorrect
    new_reward_rate = cur_reward_rate + BETA*(cur_outcome - cur_reward_rate)
    if group_every != 0:
        new_reward_rate = np.round(new_reward_rate / group_every) * group_every
    return new_reward_rate


def processMultipleSess(mult_sess_df, alpha, beta, include_Q, include_RewardRate, #round_decimals
                        group_every, callBetweenTrialFn=None):
    max_trial_num = mult_sess_df.TrialNumber.max()
    in_place_modified = "Q_val" in mult_sess_df.columns or "RewardRate" in mult_sess_df.columns
    if not in_place_modified:
        mult_sess_df = mult_sess_df.copy()
        if include_Q:
            mult_sess_df["Q_val"] = np.nan
            mult_sess_df["Q_L"] = np.nan
            mult_sess_df["Q_R"] = np.nan
        if include_RewardRate:
            mult_sess_df["RewardRate"] = np.nan

    # Set the initial values for the first trial of each session
    _temp_first_sess_trials = []
    min_trial_num = mult_sess_df.TrialNumber.min()
    # TODO: Implement this vectorized
    for sess, sess_df in mult_sess_df.groupby("SessId"):
        assert sess_df.TrialNumber.min() == min_trial_num, (
            "Its more convenient to implement if all sessions start with the "
            "same trial number")
        sess_first_trial = sess_df[sess_df.TrialNumber == min_trial_num]
        assert len(sess_first_trial) == 1
        if include_Q:
            mult_sess_df.loc[sess_first_trial.index, "Q_L"] = .5
            mult_sess_df.loc[sess_first_trial.index, "Q_R"] = .5

        if include_RewardRate:
            mult_sess_df.loc[sess_first_trial.index, "RewardRate"] = .5
        _temp_first_sess_trials.append(sess_first_trial)
    _temp_first_sess_trials = pd.concat(_temp_first_sess_trials)

    if include_Q:
        init_val = [.5]*len(_temp_first_sess_trials)
        next_trials_Q_L = pd.Series(init_val,
                                    index=_temp_first_sess_trials.SessId)
        next_trials_Q_R = pd.Series(init_val,
                                    index=_temp_first_sess_trials.SessId)

    if include_RewardRate:
        init_val = [.5]*len(_temp_first_sess_trials)
        next_trials_reward_rate = pd.Series(init_val,
                                                 index=_temp_first_sess_trials.SessId)

    df_trials_li = []
    current_sorting = mult_sess_df.index


    for trial_num in range(min_trial_num,  max_trial_num+1):
        multi_trials = mult_sess_df[mult_sess_df.TrialNumber == trial_num].copy()
        if len(multi_trials) == 0:
            continue # Can this happen?

        cur_trial_sess_idxs = multi_trials.SessId
        if include_Q:
            cur_q_L = next_trials_Q_L[cur_trial_sess_idxs].values
            cur_q_R = next_trials_Q_R[cur_trial_sess_idxs].values
            assert len(cur_q_L) == len(multi_trials)
            assert len(cur_q_R) == len(multi_trials)
            multi_trials["Q_L"] = cur_q_L
            multi_trials["Q_R"] = cur_q_R
            cur_trial_Q = calcQVal(cur_q_L, cur_q_R, group_every=group_every)
            multi_trials["Q_val"] = cur_trial_Q

        if include_RewardRate:
            # reward_rate = multi_trials.RewardRate.values
            cur_reward_rate = next_trials_reward_rate[cur_trial_sess_idxs].values
            assert len(cur_reward_rate) == len(multi_trials)
            multi_trials["RewardRate"] = cur_reward_rate


        if callBetweenTrialFn is None:
            cur_trials_outcome = multi_trials.ChoiceCorrect.values
            cur_trials_choice_left = multi_trials.ChoiceLeft.values
        else:
            # print("I said:")
            # display(multi_trials)
            cur_trials_outcome, cur_trials_choice_left, new_multi_trials = \
                                                callBetweenTrialFn(multi_trials)
            assert new_multi_trials.index.equals(multi_trials.index)
            multi_trials = new_multi_trials
            # print("Outcomes:", cur_trial_outcome.values)
            # print("Choice Left:", cur_trial_choice_left.values)
            # print("DV:", dv)

        df_trials_li.append(multi_trials)

        if include_Q:
            naxt_q_L, next_q_R = calcNewQL_QR(cur_trials_outcome,
                                              cur_trials_choice_left,
                                              cur_q_L, cur_q_R, alpha)
            # Make sure that we have no nans
            assert np.isnan(naxt_q_L).sum() == 0
            assert np.isnan(next_q_R).sum() == 0
            next_trials_Q_L = pd.Series(naxt_q_L, index=cur_trial_sess_idxs)
            next_trials_Q_R = pd.Series(next_q_R, index=cur_trial_sess_idxs)

        if include_RewardRate:
            next_reward_rate = calcNewRewardRate(cur_trials_outcome,
                                                 cur_reward_rate,
                                                 beta, group_every)
            assert np.isnan(next_reward_rate).sum() == 0
            next_trials_reward_rate = pd.Series(next_reward_rate,
                                                index=multi_trials.SessId)

    # if callBetweenTrialFn is not None:
    mult_sess_df = pd.concat(df_trials_li)
    mult_sess_df = mult_sess_df.loc[current_sorting, :]

    return_li = [mult_sess_df, in_place_modified]
    if include_Q:
        assert mult_sess_df.Q_L.notnull().all()
        assert mult_sess_df.Q_R.notnull().all()
        assert mult_sess_df.Q_val.notnull().all()
        return_li += [mult_sess_df.Q_L, mult_sess_df.Q_R]
    if include_RewardRate:
        assert mult_sess_df.RewardRate.notnull().all()
        return_li.append(mult_sess_df.RewardRate)
    return tuple(return_li)



# class LossRLFast(pyddm.LossFunction):
#     name = "rl_loss_fast"
#     # ROUND = 2 # Number of decimal digits to round to, lower number gives better performance but lower accuracy
#     GROUP_EVERY = .05
#     id_col = None
#     intiialBiasDistInst = None

#     def setup(self, **kwargs):
#         assert LossRLFast.id_col is not None, "LossRL.id_col must be specified"
#         self.sessions = self.sample.condition_values(LossRLFast.id_col)
#         self.df =  self.sample.to_pandas_dataframe(choice_column_name=CHOICE_COL_NAME)
#         self.df = self.df.sort_values(["SessId", "TrialNumber"])
#         # self.df["Qval"] = 0
#         # self.df["RewardRate_Val"] = 0
#         # print("I'm called....")

#     def loss(self, model):
#         likelihood = 0
#         if "Q_val" in self.df.columns:
#             alpha = model.get_dependence("initialcondition").ALPHA
#             include_Q = True
#             global_q_L_dev = {}
#             global_q_R_dev = {}
#             global_q_val_dev = {}
#         else:
#             alpha = 0
#             include_Q = False

#         if "RewardRate" in self.df.columns:
#             beta = model.get_dependence("drift").BETA
#             include_RewardRate = True
#             global_reward_rate_dev = {}
#         else:
#             beta = 0
#             include_RewardRate = False

#         # # print(f"Load every {LossRLFast.GROUP_EVERY} Q_val")
#         # for sess, sess_df in self.df.groupby("SessId"):
#         #     # sess_df = sess_df.sort_values("TrialNumber")
#         #     # display(sess_df.TrialNumber)
#         #     sess_df, in_place_modified, *_rest = processSess2(sess_df,
#         #                                                       alpha=alpha,
#         #                                                       beta=beta,
#         #                                                       include_Q=include_Q,
#         #                                                       include_RewardRate=include_RewardRate,
#         #                                                       #round_decimals=LossRLFast.ROUND)
#         #                                                       group_every=LossRLFast.GROUP_EVERY)
#         #     assert in_place_modified
#         #     rest_idx = 0
#         #     if include_Q:
#         #         if LossRLFast.intiialBiasDistInst is not None:
#         #             LossRLFast.intiialBiasDistInst.setDist(sess_df["Q_val"])
#         #         global_q_L_dev[sess] = _rest[rest_idx]
#         #         global_q_R_dev[sess] = _rest[rest_idx+1]
#         #         global_q_val_dev[sess] = sess_df["Q_val"].values
#         #         rest_idx += 2
#         #     if include_RewardRate:
#         #         global_reward_rate_dev[sess] = _rest[rest_idx]

#         # Use processMultipleSess instead
#         self.df, in_place_modified, *_rest = processMultipleSess(self.df,
#                                                                 alpha=alpha,
#                                                                 beta=beta,
#                                                                 include_Q=include_Q,
#                                                                 include_RewardRate=include_RewardRate,
#                                                                 #round_decimals=LossRLFast.ROUND,
#                                                                 group_every=LossRLFast.GROUP_EVERY)
#         assert in_place_modified
#         rest_idx = 0
#         if include_Q:
#             if LossRLFast.intiialBiasDistInst is not None:
#                 LossRLFast.intiialBiasDistInst.setDist(self.df["Q_val"])
#             global_q_L_dev = _rest[rest_idx]
#             global_q_R_dev = _rest[rest_idx+1]
#             global_q_val_dev = self.df["Q_val"].values
#             rest_idx += 2
#         if include_RewardRate:
#             global_reward_rate_dev = _rest[rest_idx]

#         # print(self.df)
#         # print("Creating sample")
#         new_sample = pyddm.Sample.from_pandas_dataframe(self.df,
#                                                         rt_column_name="RT",
#                                                         choice_column_name=CHOICE_COL_NAME,
#                                                         **_CHOICE_NAME_KARGS,
#                                                         )
#         # print("Calculating likelihood loss")
#         # display(new_sample.to_pandas_dataframe().Q_val.unique())
#         likelihood = pyddm.get_model_loss(model=model,
#                                           #sample=pyddm.Sample.from_pandas_dataframe(self.df, 'RT', 'choice'),
#                                           sample=new_sample,
#                                           lossfunction=pyddm.LossLikelihood)
#         # print("Calculated likelihood loss")
#         if include_Q:
#             model.last_q_lefts = global_q_L_dev
#             model.last_q_rights = global_q_R_dev
#             model.last_q_vals = global_q_val_dev
#         if include_RewardRate:
#             model.last_reward_rates = global_reward_rate_dev
#         return likelihood


# class LossRL(pyddm.LossFunction):
#     '''Copied and adapted from pyddm RL example'''
#     name = "rl_loss"
#     id_col = None
#     def setup(self, **kwargs):
#         assert LossRL.id_col is not None, "LossRL.id_col must be specified"
#         self.sessions = self.sample.condition_values(LossRL.id_col)
#         #  for s in self.sessions:
#         #      trials = self.sample.subset(session=s).condition_values('trial')
#         #      assert set(trials) == set(range(min(trials), min(trials)+len(trials))), "Trials must be sequentially numbered"
#         self.df =  self.sample.to_pandas_dataframe(choice_column_name=CHOICE_COL_NAME)
#         # display(self.df)
#         #  self.sess_dfs = [self.df[].sort_values('TrialNumber') for s in self.sessions]

#     def loss(self, model):
#         likelihood = 0
#         global_q_L_dev = {}
#         global_q_R_dev = {}
#         global_reward_rate_dev = {}
#         # alpha = model.get_dependence("drift").alpha
#         alpha = model.get_dependence("drift").ALPHA
#         beta = model.get_dependence("drift").BETA

#         for sess, sess_df in self.df.groupby(LossRL.id_col):
#             q_L_dev = [0]
#             q_R_dev = [0]
#             reward_rate_dev = [1]
#             for i, row in sess_df.iterrows():
#                 choice = row[CHOICE_COL_NAME]
#                 p = model.solve_analytical(conditions={"Q_L": q_L_dev[-1],
#                                                        "Q_R": q_R_dev[-1],
#                                                        "DV": row["DV"],
#                                                        "RewardRate": reward_rate_dev[-1]}
#                                                        ).evaluate(row['RT'],
#                                                                   choice
#                                                                   )
#                 q_L_dev, q_R_dev, reward_rate_dev = updateTrial(row,
#                                                                 q_L_dev,
#                                                                 q_R_dev,
#                                                                 reward_rate_dev,
#                                                                 alpha,
#                                                                 beta)
#                 if p <= 0:
#                     return -np.inf
#                 likelihood += np.log(p)
#             global_q_L_dev[sess] = q_L_dev
#             global_q_R_dev[sess] = q_R_dev
#             global_reward_rate_dev[sess] = reward_rate_dev

#         model.last_q_lefts = global_q_L_dev
#         model.last_q_rights = global_q_R_dev
#         model.last_reward_rates = global_reward_rate_dev
#         return -likelihood


def showModel(model, sample):
    model.show()
    pyddm.plot.plot_fit_diagnostics(model=model, sample=sample)


