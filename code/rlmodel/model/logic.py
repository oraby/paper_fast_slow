# import pyddm
# import pyddm.plot
# import matplotlib.pyplot as plt
# import numba
import numpy as np
import pandas as pd
from functools import partial

LOG_CIEL = .01
_ciel_max = np.log(1/LOG_CIEL)

run_logger = None

TRACK_PREV_CHOICE = False

# @numba.jit("f8,   f8,  f8", nopython=True)
def _calcQVal(Q_L, Q_R, group_every=0):#round_decimals=0):
    ''' Returns a value between -1 and 1'''
    Q_L = np.clip(np.asarray(Q_L), LOG_CIEL, 1)
    Q_R = np.clip(np.asarray(Q_R), LOG_CIEL, 1)
    Q_val = np.log(Q_L / Q_R) / _ciel_max
    if group_every != 0:
        Q_val = np.round(Q_val / group_every) * group_every
    # assert np.isnan(Q_val).sum() == 0 #, f"Q_val has NaNs: {Q_val} - Q_L: {Q_L} - Q_R: {Q_R}"
    return Q_val


def _updateNextQL_QR(cur_outcome, cur_choice_left, cur_q_L, cur_q_R, ALPHA):
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


def _updateNextRewardRate(cur_outcome, cur_reward_rate, BETA, group_every=0):
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
        _copied = False
        if include_Q:
            mult_sess_df = mult_sess_df.copy()
            _copied = True
            mult_sess_df["Q_val"] = np.nan
            mult_sess_df["Q_L"] = np.nan
            mult_sess_df["Q_R"] = np.nan
        if include_RewardRate:
            if not _copied:
                mult_sess_df = mult_sess_df.copy()
                _copied = True # Though no really need here for it
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
    
    if TRACK_PREV_CHOICE:
        mult_sess_df["PrevChoiceLeft"] = np.nan
        mult_sess_df["PrevChoiceCorrect"] = np.nan
        mult_sess_df["PrevDV"] = np.nan
        PrevChoiceLeft = pd.Series(np.nan, index=_temp_first_sess_trials.SessId)
        PrevChoiceCorrect = pd.Series(np.nan, index=_temp_first_sess_trials.SessId)
        PrevDV = pd.Series(np.nan, index=_temp_first_sess_trials.SessId)

    df_trials_li = []
    current_sorting = mult_sess_df.index


    for trial_num in range(min_trial_num,  max_trial_num+1):
        multi_trials = mult_sess_df[mult_sess_df.TrialNumber == trial_num].copy()
        if len(multi_trials) == 0:
            assert False, "This should not happen"

        cur_trial_sess_idxs = multi_trials.SessId
        if include_Q:
            cur_q_L = next_trials_Q_L[cur_trial_sess_idxs].values
            cur_q_R = next_trials_Q_R[cur_trial_sess_idxs].values
            assert len(cur_q_L) == len(multi_trials)
            assert len(cur_q_R) == len(multi_trials)
            multi_trials["Q_L"] = cur_q_L
            multi_trials["Q_R"] = cur_q_R
            cur_trial_Q = _calcQVal(cur_q_L, cur_q_R, group_every=group_every)
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

        if TRACK_PREV_CHOICE:
            multi_trials["PrevChoiceLeft"] = PrevChoiceLeft[cur_trial_sess_idxs].values
            multi_trials["PrevChoiceCorrect"] = PrevChoiceCorrect[cur_trial_sess_idxs].values
            multi_trials["PrevDV"] = PrevDV[cur_trial_sess_idxs].values
            PrevChoiceLeft = pd.Series(multi_trials.ChoiceLeft.values, 
                                       index=multi_trials.SessId)
            PrevChoiceCorrect = pd.Series(multi_trials.ChoiceCorrect.values,
                                          index=multi_trials.SessId)
            PrevDV = pd.Series(multi_trials.DV.values,
                               index=multi_trials.SessId)
        
        df_trials_li.append(multi_trials)

        if include_Q:
            naxt_q_L, next_q_R = _updateNextQL_QR(cur_trials_outcome,
                                              cur_trials_choice_left,
                                              cur_q_L, cur_q_R, alpha)
            # Make sure that we have no nans
            assert np.isnan(naxt_q_L).sum() == 0
            assert np.isnan(next_q_R).sum() == 0
            next_trials_Q_L = pd.Series(naxt_q_L, index=cur_trial_sess_idxs)
            next_trials_Q_R = pd.Series(next_q_R, index=cur_trial_sess_idxs)

        if include_RewardRate:
            next_reward_rate = _updateNextRewardRate(cur_trials_outcome,
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




def simulateDDMTrial(dvs, NON_DECISION_TIME, starting_point, BOUND, driftFn, DRIFT_COEF,
                     noiseFn, NOISE_SIGMA, dt, max_dt, driftFn_kwargs={},
                     noiseFn_kwargs={}):
    global runs_dx, runs_noise, runs_starting_point, runs_dvs, runs_bound
    max_possible_dt = max_dt #- nondectime
    num_steps = int(np.ceil(max_possible_dt/dt))
    num_trials = len(dvs)

    # TODO: I have a bug where we look at whether the threshold is crossed
    # before after non-decision time is added, i.e. trials where the threshold
    # is crossed beyond after adding non-decision time should be considered as
    # undecided

    # Starting point is between -1 and 1, scale the starting point to be
    # between -bound and bound
    starting_point_bounded = starting_point * BOUND
    # Initialize the dx array
    size = num_trials, num_steps

    # Generate random noise for all steps and trials
    noise = noiseFn(size, dt, **noiseFn_kwargs)

    # # Calculate the drift component
    # drift = drift_coef * dvs[:, np.newaxis] * dt
    # # Update dx array using cumulative sum
    # dx[:, start_point_idx+1:] = np.cumsum(drift + noise, axis=1) #+ starting_point_bounded[:, np.newaxis]
    dx = driftFn(starting_point_bounded, nondectime=NON_DECISION_TIME, 
                 noise=noise, drift_coef=DRIFT_COEF,
                 dvs=dvs, dt=dt, noise_sigma=NOISE_SIGMA, **driftFn_kwargs)

    if run_logger is not None:
        run_logger.starting_point_bounded = starting_point_bounded
        run_logger.noise = noise
        run_logger.dx = dx
        run_logger.dvs = dvs
        run_logger.drift_coef = DRIFT_COEF
        run_logger.noise_sigma = NOISE_SIGMA
        run_logger.dt = dt
        run_logger.max_dt = max_dt
        run_logger.bounds = BOUND
        run_logger.nondectime = NON_DECISION_TIME
        for key, val in noiseFn_kwargs.items():
            setattr(run_logger, key, val)
        for key, val in driftFn_kwargs.items():
            setattr(run_logger, key, val)

    # Create masks for when the values cross the bounds
    above_mask = dx >= BOUND
    below_mask = dx <= -BOUND

    # Find the first crossing point for each trial
    first_above_idx = np.argmax(above_mask, axis=1)
    first_below_idx = np.argmax(below_mask, axis=1)

    # Determine if the bounds were crossed
    crossed_above = np.any(above_mask, axis=1)
    crossed_below = np.any(below_mask, axis=1)

    # Initialize reaction times and choices
    rts = np.full(num_trials, np.nan)
    choice_left = np.full(num_trials, np.nan)

    # Set reaction times and choices based on the first crossing point
    use_above = crossed_above & (~crossed_below | (first_above_idx < first_below_idx))
    use_below = crossed_below & (~crossed_above | (first_below_idx < first_above_idx))
    undecided = ~use_above & ~use_below

    rts[use_above] = first_above_idx[use_above] * dt
    choice_left[use_above] = 1

    rts[use_below] = first_below_idx[use_below] * dt
    choice_left[use_below] = 0

    rts[undecided] = max_possible_dt

    # assert (rts <= max_dt - nondectime).all()
    # rts += nondectime
    if run_logger is not None:
        run_logger.rts = rts
        run_logger.choice_left = choice_left

    # rts[rts < NON_DECISION_TIME] = NON_DECISION_TIME

    return rts, choice_left, starting_point_bounded


_last_PrevChoiceLeft = None
_last_PrevChoiceCorrect = None
_last_PrevDV = None
def betweenTrialsCb(many_trials_df,
                    biasFn, driftFn, noiseFn, NON_DECISION_TIME, BOUND,
                    biasFn_df_cols={}, driftFn_df_cols=[], noiseFn_df_cols=[],
                    multi_sess_df=None, **ddm_trial_kwargs):
    #DVabs = many_trials_df.DV.abs().values
    global run_logger, _last_PrevChoiceLeft, _last_PrevChoiceCorrect, _last_PrevDV
    dvs = many_trials_df.DV.values
    biasFn_kwargs = {col: many_trials_df[col].values for col in biasFn_df_cols}
    driftFn_kwargs = {col: many_trials_df[col].values for col in driftFn_df_cols}
    noiseFn_kwargs = {col: many_trials_df[col].values for col in noiseFn_df_cols}

    starting_point = biasFn(size=len(many_trials_df), **biasFn_kwargs)
    # The simulation function will scale the starting point between -bound and 
    # bound, i..e starting_point *= BOUND
    starting_point = np.clip(starting_point, -1, 1)

    rts, choice_left, starting_point_bounded = \
        simulateDDMTrial(dvs=dvs,
                         driftFn=driftFn, driftFn_kwargs=driftFn_kwargs,
                         noiseFn=noiseFn, noiseFn_kwargs=noiseFn_kwargs,
                         NON_DECISION_TIME=NON_DECISION_TIME,
                         starting_point=starting_point,
                         BOUND=BOUND, **ddm_trial_kwargs)
    assert len(choice_left) == len(many_trials_df)
    # if many_trials_df.TrialNumber.min() == 1:
    #     print("First trial - Starting point:", starting_point, " - DVs:", dvs)
    #     print("Choice left:", choice_left, " - RTs:", rts)
    #     display(many_trials_df)

    dV_left_mask = many_trials_df.DV.values > 0
    choice_correct = (choice_left == dV_left_mask).astype(float)
    choice_correct[np.isnan(choice_left)] = np.nan
    if multi_sess_df is not None:
        assert multi_sess_df.loc[many_trials_df.index, "SimStartingPoint"].isnull().all()
    many_trials_df = many_trials_df.copy()
    many_trials_df["SimStartingPoint"] = starting_point_bounded
    many_trials_df["SimRT"] = rts
    many_trials_df["SimChoiceCorrect"] = choice_correct
    many_trials_df["SimChoiceLeft"] = choice_left
    # Assign prev. choices and DVs
    if TRACK_PREV_CHOICE:
        sess_idxs = many_trials_df.SessId
        many_trials_df["SimPrevChoiceLeft"] = _last_PrevChoiceLeft[sess_idxs].values
        many_trials_df["SimPrevChoiceCorrect"] = _last_PrevChoiceCorrect[sess_idxs].values
        many_trials_df["SimPrevDV"] = _last_PrevDV[sess_idxs].values
        # Update the last choices and DVs
        _last_PrevChoiceLeft = pd.Series(choice_left, index=sess_idxs)
        _last_PrevChoiceCorrect = pd.Series(choice_correct, index=sess_idxs)
        _last_PrevDV = pd.Series(dvs, index=sess_idxs)
    # many_trials_df["SimMatchReal"] = ((choice_left == many_trials_df.ChoiceLeft) |
    #                                     (np.isnan(choice_left) & many_trials_df.ChoiceLeft.isnull())).astype(int)
    if run_logger is not None:
        run_logger.TrialNumber = many_trials_df.TrialNumber
        run_logger.Name = many_trials_df.Name
        run_logger.SessId = many_trials_df.SessId
        run_logger.SimChoiceCorrect = choice_correct
    # Return the current choices to be used in the next trial
    return (pd.Series(choice_correct, index=many_trials_df.SessId),
            pd.Series(choice_left, index=many_trials_df.SessId),
            many_trials_df)


def simulateDDMMultipleSess(multi_sess_df, include_Q, include_RewardRate,
                            ALPHA, BETA, biasFn, driftFn, noiseFn,
                            NON_DECISION_TIME,  BOUND,
                            biasFn_df_cols={}, driftFn_df_cols=[], noiseFn_df_cols=[],
                            **ddm_trial_kwargs):

    global _last_PrevChoiceLeft, _last_PrevChoiceCorrect, _last_PrevDV
    if TRACK_PREV_CHOICE:
        multi_sess_df["SimPrevChoiceLeft"] = np.nan
        multi_sess_df["SimPrevChoiceCorrect"] = np.nan
        multi_sess_df["SimPrevDV"] = np.nan
        min_trial_num = multi_sess_df.TrialNumber.min()
        first_trials = multi_sess_df[multi_sess_df.TrialNumber == min_trial_num]
        assert len(first_trials) == len(multi_sess_df.SessId.unique()), (
            "All sessions should start at the same trial number")
        _last_PrevChoiceCorrect = pd.Series(np.nan, index=first_trials.SessId)
        _last_PrevChoiceLeft = pd.Series(np.nan, index=first_trials.SessId)
        _last_PrevDV = pd.Series(np.nan, index=first_trials.SessId)

    BOUND = float(BOUND)
    partialBetweenTrialsCb = partial(betweenTrialsCb, biasFn=biasFn, driftFn=driftFn,
                                     noiseFn=noiseFn, NON_DECISION_TIME=NON_DECISION_TIME,
                                     BOUND=BOUND, biasFn_df_cols=biasFn_df_cols,
                                     driftFn_df_cols=driftFn_df_cols,
                                     noiseFn_df_cols=noiseFn_df_cols,
                                     multi_sess_df=multi_sess_df,
                                     **ddm_trial_kwargs)

    ret = processMultipleSess(multi_sess_df, alpha=ALPHA, beta=BETA,
                              include_Q=include_Q, include_RewardRate=include_RewardRate,
                              group_every=0, callBetweenTrialFn=partialBetweenTrialsCb)
    multi_sess_df, in_place_modified, *_rest = ret

    if TRACK_PREV_CHOICE:
        multi_sess_df["IsStay"] = multi_sess_df.ChoiceLeft == multi_sess_df.PrevChoiceLeft
        multi_sess_df["SimIsStay"] = multi_sess_df.SimChoiceLeft == multi_sess_df.SimPrevChoiceLeft

    assert multi_sess_df.SimStartingPoint.notnull().all()
    if include_Q:
        assert multi_sess_df.Q_L.notnull().all()
        assert multi_sess_df.Q_R.notnull().all()
        assert multi_sess_df.Q_val.notnull().all()
    assert in_place_modified or (not include_Q and not include_RewardRate)
    return multi_sess_df


def _choiceCorrLossHist(df, bins, dscrp, real_choice_col, sim_choice_col,
                        DEBUG=False):
    if DEBUG:
        num_incorrect = (df.ChoiceCorrect == 0).sum()
        print(f"Choice loss for {dscrp}")
        print(f"DV: {dscrp} - Num incorrect: {num_incorrect}/{len(df)}")
    loss = 0
    for choice in (1, 0):
        choice_rt = df[df[real_choice_col] == choice].calcStimulusTime
        sim_choice_rt = df[df[sim_choice_col] == choice].SimRT
        hist_real_choice, _ = np.histogram(choice_rt, bins=bins)
        hist_sim_choice, _ = np.histogram(sim_choice_rt, bins=bins)
        choice_loss = np.abs(hist_real_choice - hist_sim_choice)
        choice_loss_sum = choice_loss.sum().astype(float)
        if DEBUG:
            print(f"Choice: {choice} - Choice loss sum: {choice_loss_sum}")
            # print(f"Choice: {choice} - Choice loss: {choice_loss}")
        loss += choice_loss_sum
    return loss

def _choiceCorrLossCount(df, dscrp):
    DEBUG = False
    loss = 0
    for choice in (1, 0):
        real_choice_sum = (df.ChoiceCorrect == choice).sum()
        sim_choice_sum = (df.SimChoiceCorrect == choice).sum()
        choice_loss = np.abs(real_choice_sum - sim_choice_sum)
        if DEBUG:
            print(f"Choice: {choice} - Choice loss: {choice_loss}")
        loss += choice_loss
        # Add a penality for how far are the sim from the real mode
        # real_mode = df[df.ChoiceCorrect == choice].calcStimulusTime.mode()
        # sim_mode = df[df.SimChoiceCorrect == choice].SimRT.mode()
        # if len(real_mode) and len(sim_mode):
        #     mode_loss = np.abs(real_mode.values[0] - sim_mode.values[0])
        #     # Amplify the loss as a function of the number of trials
        #     mode_loss *= real_choice_sum
        #     loss += mode_loss
    # Null choices
    nan_choice_rt = df.ChoiceCorrect.isnull().sum()
    nan_sim_choice_rt = df.SimChoiceCorrect.isnull().sum()
    nan_choice_loss = np.abs(nan_choice_rt - nan_sim_choice_rt)
    loss += nan_choice_loss
    return loss

def _choiceLeftLoss(res_sorted_df, sim_sorted_df, bot_idx, top_idx):
    real_q_df = res_sorted_df.iloc[bot_idx:top_idx]
    sim_q_df = sim_sorted_df.iloc[bot_idx:top_idx]
    real_lefts_sum = real_q_df.ChoiceLeft.sum()
    sim_lefts_sum = sim_q_df.SimChoiceLeft.sum()
    cur_loss =  np.abs(real_lefts_sum - sim_lefts_sum)
    # print(f"{sess} - {dv_str} - q={q_idx} ({bot_idx}:{top_idx}) (len={len(dv_df)}) real_q_df: {real_lefts_sum} - {sim_lefts_sum} = {cur_loss}")
    return cur_loss


def chi2Loss(real_rt, sim_rt, t_dur):
    # ratcliff_quantiles = [.1, .3, .5, .7, .9]
    # bins = np.nanquantile(real_rt, ratcliff_quantiles)
    # bins = np.insert(bins, 0, 0)
    # bins = np.append(bins, t_dur)
    ratcliff_quantiles = [0, .1, .3, .5, .7, .9, 1]
    real_rt = real_rt[real_rt.notnull()].values
    sim_rt = sim_rt[sim_rt.notnull()].values
    bins = np.quantile(real_rt, ratcliff_quantiles)

    freq_real, _ = np.histogram(real_rt, bins=bins)
    freq_sim, _ = np.histogram(sim_rt, bins=bins)
    # Chi2 loss: sum((observed - expected)^2 / expected)
    bins_loss = (freq_sim - freq_real)**2 / freq_real
    bins_loss_sum = bins_loss.sum().astype(float)
    return bins_loss_sum
    

# def calcLoss9(df, dt, t_dur):
#     df = df[df.valid]
#     loss = 0
#     loop_cols = [("ChoiceCorrect", "SimChoiceCorrect"),
#                  ("PrevChoiceCorrect", "PrevChoiceLeft"),
#                  ("IsStay", "SimIsStay"),
#                  #("DVstr", "DVstr"),
#                  ]
#     # display(df)
#     for real_col, sim_col in loop_cols:
#         unique_vals = df[real_col].unique()
#         unique_vals = unique_vals[~pd.isnull(unique_vals)]
#         # print(f"Unique vals for {real_col}: {unique_vals}")
#         for val in unique_vals:
#             real_choice_rt = df[df[real_col] == val].calcStimulusTime
#             sim_choice_rt = df[df[sim_col] == val].SimRT
#             loss += chi2Loss(real_choice_rt, sim_choice_rt, t_dur)
#     return loss


def calcLoss8(df, dt, t_dur):
    df = df[df.valid]
    loss = 0
    for choice in (1, 0):
        real_choice_rt = df[df["ChoiceCorrect"] == choice].calcStimulusTime
        sim_choice_rt = df[df["SimChoiceCorrect"] == choice].SimRT
        loss += chi2Loss(real_choice_rt, sim_choice_rt, t_dur)
    return loss

# def calcLoss7(df, dt, t_dur):
#     # Use only valid trials from the original filtered df
#     df = df[df.valid]
#     real_rt = df.calcStimulusTime.copy()
#     sim_rt = df.SimRT.copy()
#     real_rt[real_rt.isnull()] = t_dur + 1
#     sim_rt[sim_rt.isnull()] = t_dur + 1
#     diff_squares = (real_rt - sim_rt)**2
#     loss = diff_squares.sum().astype(float) / len(df)
#     if np.isnan(loss):
#         loss = np.inf
#     return loss 


# def calcLoss6(df, dt, t_dur):
#     # Use only valid trials from the original filtered df
#     df = df[df.valid]
#     DEBUG = False
#     STEP = dt
#     bins = np.arange(0, t_dur + STEP, STEP)
#     loss = 0
#     # for dv_str, dv_df in [("All", df)]:
#     for dv_str, dv_df in df.groupby("DVstr"):
#         if DEBUG:
#             print("DV str:", dv_str, " - Num trials:", len(dv_df))
#         # loss += _choiceCorrLossHist(dv_df, bins, dv_str,
#         #                             "ChoiceCorrect", "SimChoiceCorrect",
#         #                             DEBUG=DEBUG)
#         for choice in (1, 0):
#             choice_rt = df[df["ChoiceCorrect"] == choice].calcStimulusTime
#             sim_choice_rt = df[df["SimChoiceCorrect"] == choice].SimRT
#             hist_real_choice, _ = np.histogram(choice_rt, bins=bins)
#             hist_sim_choice, _ = np.histogram(sim_choice_rt, bins=bins)
#             # choice_loss = np.abs(hist_real_choice - hist_sim_choice)
#             choice_loss = (hist_real_choice - hist_sim_choice)**2
#             choice_loss_sum = choice_loss.sum().astype(float) / len(bins)
#             loss += choice_loss_sum

#     # Handle nans as one bin
#     # nan_choice_rt = df.ChoiceCorrect.isnull().sum()
#     # nan_sim_choice_rt = df.SimChoiceCorrect.isnull().sum()
#     # loss += np.abs(nan_choice_rt - nan_sim_choice_rt).astype(float)

#     if np.isnan(loss):
#         loss = np.inf
#     return loss #/ len(df)


# def calcLoss5(df, dt, t_dur):
#     # Use only valid trials from the original filtered df
#     df = df[df.valid]
#     # Penalty for incorrect choices, the longer the RT the higher the penalty
#     # display(df)
#     DEBUG = False
#     STEP = dt
#     bins = np.arange(0, t_dur + STEP, STEP)
#     loss = 0
#     # for dv_str, dv_df in [("All", df)]:
#     for dv_str, dv_df in df.groupby("DVstr"):
#         if DEBUG:
#             print("DV str:", dv_str, " - Num trials:", len(dv_df))
#         # dv_left = dv_df.DV > 0
#         # for side, side_choice_df in dv_df.groupby(dv_left):
#         #     # print(f"Choice: {choice} - Choice loss: {choice_loss}")
#         #     if DEBUG:
#         #         _side_str = "Left" if side else "Right"
#         #         print(f"Side: {_side_str} - Num trials: {len(side_choice_df)}")
#         #     loss += _choiceCorrLossHist(side_choice_df, bins, dv_str,
#         #                                 "ChoiceLeft", "SimChoiceLeft",
#         #                                 DEBUG=DEBUG)
#         #     # loss += _choiceLeftLoss(side_choice_df, side_choice_df, 0,
#         #     #                         len(side_choice_df))
#         loss += _choiceCorrLossHist(dv_df, bins, dv_str,
#                                     "ChoiceCorrect", "SimChoiceCorrect",
#                                     DEBUG=DEBUG)


#     # Handle nans as one bin
#     nan_choice_rt = df.ChoiceCorrect.isnull().sum()
#     nan_sim_choice_rt = df.SimChoiceCorrect.isnull().sum()
#     loss += np.abs(nan_choice_rt - nan_sim_choice_rt).astype(float)

#     if np.isnan(loss):
#         loss = np.inf
#     return loss / len(df)


# def calcLoss4(df, dt, t_dur):
#     # Use only valid trials from the original filtered df
#     df = df[df.valid]
#     # Penalty for incorrect choices, the longer the RT the higher the penalty
#     # display(df)
#     DEBUG = False
#     STEP = dt
#     bins = np.arange(0, t_dur + STEP, STEP)
#     loss = 0
#     for sess, sess_df in df.groupby(["Name", "Date", "SessionNum"]):
#         for dv_str, dv_df in sess_df.groupby("DVstr"):
#         # for dv_str, dv_df in [("All", df)]:
#             dv_left = dv_df.DV > 0
#             for side, side_choice_df in dv_df.groupby(dv_left):
#                 # print(f"Choice: {choice} - Choice loss: {choice_loss}")
#                 loss += _choiceCorrLossHist(side_choice_df, bins, dv_str,
#                                             "ChoiceLeft", "SimChoiceLeft")
#                 # loss += _choiceLeftLoss(side_choice_df, side_choice_df, 0,
#                 #                         len(side_choice_df))

#     # Handle nans as one bin
#     nan_choice_rt = df.ChoiceCorrect.isnull().sum()
#     nan_sim_choice_rt = df.SimChoiceCorrect.isnull().sum()
#     loss += np.abs(nan_choice_rt - nan_sim_choice_rt).astype(float)

#     if np.isnan(loss):
#         loss = np.inf
#     return loss / len(df)

# def calcLoss3(df, dt, t_dur):
#     # Use only valid trials from the original filtered df
#     df = df[df.valid]
#     # Penalty for incorrect choices, the longer the RT the higher the penalty
#     # display(df)
#     DEBUG = False
#     STEP = dt
#     bins = np.arange(0, t_dur + STEP, STEP)
#     loss = 0
#     for dv_str, dv_df in df.groupby("DVstr"):
#     # for dv_str, dv_df in [("All", df)]:
#         dv_left = dv_df.DV > 0
#         for side, side_choice_df in dv_df.groupby(dv_left):
#             # print(f"Choice: {choice} - Choice loss: {choice_loss}")
#             # loss += _choiceCorrLossHist(sim_choice_df, bins, dv_str)
#             loss += _choiceCorrLossCount(side_choice_df, dv_str)


#     quantile_loss = 0
#     NUM_QUANTILES = 3
#     for sess, sess_df in df.groupby(["Name", "Date", "SessionNum"]):
#         # for dv_str, dv_df in sess_df.groupby("DVstr"):
#         for dv, dv_df in sess_df.groupby("DV"):
#             real_sess_df = dv_df.sort_values(by="calcStimulusTime")
#             sim_sess_df = dv_df.sort_values(by="SimRT")
#             # sim_choice_rt = np.sort(sess_df.SimRt)
#             bot_idx = 0
#             for q_idx in np.arange(1, NUM_QUANTILES+1):
#                 top_idx = len(sess_df)*q_idx/NUM_QUANTILES
#                 top_idx = int(np.round(top_idx))          
#                 quantile_loss += _choiceLeftLoss(real_sess_df, sim_sess_df,
#                                                  bot_idx, top_idx)
#                 bot_idx = top_idx  
#     loss += quantile_loss

#     # Handle nans as one bin
#     nan_choice_rt = df.ChoiceCorrect.isnull().sum()
#     nan_sim_choice_rt = df.SimChoiceCorrect.isnull().sum()
#     nan_choies_sum = nan_choice_rt + nan_sim_choice_rt
#     if nan_choies_sum:
#         loss += np.abs(nan_choice_rt - nan_sim_choice_rt).astype(float)#/nan_choies_sum

#     if np.isnan(loss):
#         loss = np.inf
#     return loss

# def calcLoss2(df, dt, t_dur):
#     # Use only valid trials from the original filtered df
#     df = df[df.valid]
#     # Penalty for incorrect choices, the longer the RT the higher the penalty
#     # display(df)
#     DEBUG = False
#     STEP = dt
#     bins = np.arange(0, t_dur + STEP, STEP)
#     loss = 0
#     for dv_str, dv_df in df.groupby("DVstr"):
#     # for dv_str, dv_df in [("All", df)]:
#         num_incorrect = (dv_df.ChoiceCorrect == 0).sum()
#         if DEBUG:
#             print(f"DV: {dv_str} - Num incorrect: {num_incorrect}/{len(dv_df)}")
#         # choice_loss_sum += _choiceCorrLossHist(dv_df, bins, dv_str,
#         #                                        "ChoiceCorrect", "SimChoiceCorrect")
#         choice_loss_sum += _choiceLeftLoss(dv_df, dv_df, 0, len(dv_df))
#         # print(f"Choice: {choice} - Choice loss: {choice_loss}")
#         loss += choice_loss_sum


#     quantile_loss = 0
#     NUM_QUANTILES = 3
#     for sess, sess_df in df.groupby(["Name", "Date", "SessionNum"]):
#         # for dv_str, dv_df in sess_df.groupby("DVstr"):
#         for dv, dv_df in sess_df.groupby("DV"):
#             real_sess_df = dv_df.sort_values(by="calcStimulusTime")
#             sim_sess_df = dv_df.sort_values(by="SimRT")
#             # sim_choice_rt = np.sort(sess_df.SimRt)
#             bot_idx = 0
#             for q_idx in np.arange(1, NUM_QUANTILES+1):
#                 top_idx = len(sess_df)*q_idx/NUM_QUANTILES
#                 top_idx = int(np.round(top_idx))
#                 real_q_df = real_sess_df.iloc[bot_idx:top_idx]
#                 sim_q_df = sim_sess_df.iloc[bot_idx:top_idx]
#                 cur_loss += _choiceLeftLoss(real_q_df, sim_q_df,
#                                             bot_idx=bot_idx, top_idx=top_idx)
#                 quantile_loss += cur_loss
#                 bot_idx = top_idx
#     loss += quantile_loss

#     # Handle nans as one bin
#     nan_choice_rt = df.ChoiceCorrect.isnull().sum()
#     nan_sim_choice_rt = df.SimChoiceCorrect.isnull().sum()
#     nan_choies_sum = nan_choice_rt + nan_sim_choice_rt
#     if nan_choies_sum:
#         loss += np.abs(nan_choice_rt - nan_sim_choice_rt).astype(float)#/nan_choies_sum

#     if np.isnan(loss):
#         loss = np.inf
#     return loss


# import time
def makeOneRun(df, include_Q, include_RewardRate, biasFn, driftFn,
               noiseFn, ALPHA, BETA, NON_DECISION_TIME, BOUND,
               DRIFT_COEF, NOISE_SIGMA, dt, t_dur, 
               biasFn_df_cols=[],  biasFn_kwargs={}, 
               driftFn_df_cols=[], driftFn_kwargs={},
               noiseFn_df_cols=[], noiseFn_kwargs={},
               return_df=False):
    from . import bias
    from . import drift
    from . import noise
    
    # Fix the random seed
    # np.random.seed(0)
    rnd_rng = np.random.default_rng(seed=0)
    bias.rnd_default_rng = rnd_rng
    drift.rnd_default_rng = rnd_rng
    noise.rnd_default_rng = rnd_rng
    
    # print("FIxed params:", fixed_params_names)
    df = df.copy() # Must we have this if are running in parallel?
    if len(driftFn_kwargs):
        driftFn = partial(driftFn, **driftFn_kwargs)
    if len(noiseFn_kwargs):
        noiseFn = partial(noiseFn, **noiseFn_kwargs)
    if len(biasFn_kwargs):
        biasFn = partial(biasFn, **biasFn_kwargs)
    
    # print("bias df cols:", biasFn_df_cols)
    # time_start = time.time()
    processed_df = simulateDDMMultipleSess(df, include_Q=include_Q, include_RewardRate=include_RewardRate,
                                           biasFn=biasFn,   biasFn_df_cols=biasFn_df_cols,
                                           driftFn=driftFn, driftFn_df_cols=driftFn_df_cols,
                                           noiseFn=noiseFn, noiseFn_df_cols=noiseFn_df_cols,
                                           ALPHA=ALPHA, BETA=BETA,
                                           NON_DECISION_TIME=NON_DECISION_TIME,
                                           BOUND=BOUND, DRIFT_COEF=DRIFT_COEF,
                                           NOISE_SIGMA=NOISE_SIGMA, dt=dt,
                                           max_dt=t_dur)
    # print(f"Simulation = {time.time() - time_start:.2f}")
    # time_start = time.time()
    total_loss = calcLoss8(processed_df, dt, t_dur)
    # print(f"Loss calc = {time.time() - time_start:.2f}")
    if return_df:
        return total_loss, processed_df
    else:
        return total_loss
