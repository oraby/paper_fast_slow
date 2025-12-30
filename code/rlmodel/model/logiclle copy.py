from . import bias
from . import drift
from . import noise
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
    # Treat no decision as incorrect
    cur_outcome = np.nan_to_num(cur_outcome, nan=0)
    # Propagate the values from the previous trial. Also set nan values 
    # i.e, cur_choice_left != cur_choice_left, to the current values
    new_q_L = np.where(cur_choice_left != cur_choice_left, cur_q_L, 
                       cur_q_L + ALPHA * (cur_outcome - cur_q_L) * cur_choice_left)
    new_q_R = np.where(cur_choice_left != cur_choice_left, cur_q_R, 
                       cur_q_R + ALPHA * (cur_outcome - cur_q_R) * (1 - cur_choice_left))
    return new_q_L, new_q_R


def _updateNextRewardRate(cur_outcome, cur_reward_rate, BETA, group_every=0):
    '''Calculate the new reward rate'''
    # Treat no decision as incorrect
    cur_outcome = np.nan_to_num(cur_outcome, nan=0)
    new_reward_rate = cur_reward_rate + BETA * (cur_outcome - cur_reward_rate)
    if group_every != 0:
        new_reward_rate = np.round(new_reward_rate / group_every) * group_every
    return new_reward_rate


def processMultipleSess(mult_sess_df, alpha, beta, include_Q, include_RewardRate, #round_decimals
                        group_every, callBetweenTrialFn=None):
    min_trial_num = mult_sess_df.TrialNumber.min()
    max_trial_num = mult_sess_df.TrialNumber.max()
    num_sess = len(mult_sess_df.SessId.unique())
    assert len(mult_sess_df) == num_sess*(max_trial_num - min_trial_num + 1), (
        "The dataframe should contain all trials for all sessions")
    
    mult_sess_df = mult_sess_df.sort_values(["TrialNumber", "SessId"])
    mult_sess_df = mult_sess_df.reset_index(drop=True)
    # display(mult_sess_df)
    in_place_modified = "Q_val" in mult_sess_df.columns or \
                         "RewardRate" in mult_sess_df.columns
    _copied = False
    if not in_place_modified:
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

    assert not _copied, "This should not happen if we want high performance"
    DEBUG = False

    # Extra arrays that we should create
    # "SimRT", "SimStartingPoint",  "SimChoiceCorrect", "SimChoiceLeft"
    # "SimPrevChoiceLeft", "SimPrevChoiceCorrect", "SimPrevDV"
    matrix_shape = num_sess * max_trial_num
    # Add one so we can assign always the next trial
    matrix_shape_extra = num_sess * (max_trial_num + 1)
    sim_starting_point = np.empty(matrix_shape, dtype=np.float32)
    sim_rt = np.empty(matrix_shape, dtype=np.float32)
    sim_choice_correct = np.empty(matrix_shape, dtype=np.float32)
    sim_choice_left = np.empty(matrix_shape, dtype=np.float32)
    # sim_prev_choice_left = np.empty(num_sess, dtype=np.float32)
    # sim_prev_choice_correct = np.empty(num_sess, dtype=np.float32)
    # sim_prev_dv = np.empty(num_sess, dtype=np.float32)
    # sim_match_real = np.empty(num_sess, dtype=np.float32)

    if include_Q:
        # Q_L_arr = np.full(matrix_shape_extra, .5, dtype=np.float32)
        # Q_R_arr = np.full(matrix_shape_extra, .5, dtype=np.float32)
        Q_L_arr = np.full(matrix_shape_extra, np.nan, dtype=np.float32)
        Q_R_arr = np.full(matrix_shape_extra, np.nan, dtype=np.float32)
        Q_L_arr[:num_sess] = .5
        Q_R_arr[:num_sess] = .5
        Q_val_arr = np.empty(matrix_shape, dtype=np.float32)
    else:
        # Create mock current Q values
        cur_Q_val = np.full(num_sess, np.nan, dtype=np.float32)

    if include_RewardRate:
        # reward_rate_arr = np.full(matrix_shape_extra, .5, dtype=np.float32)
        reward_rate_arr = np.full(matrix_shape_extra, np.nan, dtype=np.float32)
        reward_rate_arr[:num_sess] = .5
    else:
        # Create mock current reward rate
        cur_reward_rate = np.full(num_sess, np.nan, dtype=np.float32)

    if DEBUG:
        first_trials_df = mult_sess_df.iloc[:num_sess]
        assert all(first_trials_df.TrialNumber == mult_sess_df.TrialNumber.min())


    for trial_idx in range(max_trial_num):
        cur_start_idx = trial_idx*num_sess
        cur_end_idx = (trial_idx+1)*num_sess
        next_trial_end_idx = cur_end_idx + num_sess
        multi_trials = mult_sess_df.iloc[cur_start_idx:cur_end_idx]

        if DEBUG:
            assert len(multi_trials) == num_sess, (f"Trial: {trial_idx} - "
                f"len(multi_trials)={len(multi_trials)} - num_sess={num_sess}")
            assert all(multi_trials.TrialNumber == trial_idx + min_trial_num)

        if include_Q:
            cur_q_L = Q_L_arr[cur_start_idx:cur_end_idx]
            cur_q_R = Q_R_arr[cur_start_idx:cur_end_idx]
            if DEBUG:
                assert len(cur_q_L) == len(multi_trials)
                assert len(cur_q_R) == len(multi_trials)
            cur_Q_val = _calcQVal(cur_q_L, cur_q_R, group_every=group_every)
            if DEBUG and trial_idx == 0:
                print("First trial - Q_L:", cur_q_L, " - Q_R:", cur_q_R,
                      " - Q_val:", cur_Q_val)
                print("Setting idx:", cur_start_idx, " - ", cur_end_idx)
            Q_val_arr[cur_start_idx:cur_end_idx] = cur_Q_val

        if include_RewardRate:
            # reward_rate = multi_trials.RewardRate.values
            # cur_reward_rate = next_trials_reward_rate[cur_trial_sess_idxs].values
            cur_reward_rate = reward_rate_arr[cur_start_idx:cur_end_idx]
            assert len(cur_reward_rate) == len(multi_trials)
            # multi_trials["RewardRate"] = cur_reward_rate


        if callBetweenTrialFn is None:
            cur_trials_outcome = multi_trials.ChoiceCorrect.values
            cur_trials_choice_left = multi_trials.ChoiceLeft.values
        else:
            # print("I said:")
            # display(multi_trials)
            # Function returns: choice_correct, choice_left, rts, starting_point_bounded
            (cur_trials_sim_choice_correct, cur_trials_sim_choice_left,
             cur_trials_sim_rts, cur_trials_sim_starting_poing) = \
                    callBetweenTrialFn(multi_trials, cur_Q_val, cur_reward_rate)
            if DEBUG and trial_idx == 0:
                print("First trial - SimChoiceLeft:", cur_trials_sim_choice_left,
                      " - SimChoiceCorrect:", cur_trials_sim_choice_correct,)

            sim_choice_correct[cur_start_idx:cur_end_idx] = cur_trials_sim_choice_correct
            sim_choice_left[cur_start_idx:cur_end_idx] = cur_trials_sim_choice_left
            sim_rt[cur_start_idx:cur_end_idx] = cur_trials_sim_rts
            sim_starting_point[cur_start_idx:cur_end_idx] = cur_trials_sim_starting_poing
            # print("I got:")
            cur_trials_outcome = cur_trials_sim_choice_correct
            cur_trials_choice_left = cur_trials_sim_choice_left

        if include_Q:
            next_q_L, next_q_R = _updateNextQL_QR(cur_trials_outcome,
                                                  cur_trials_choice_left,
                                                  cur_q_L, cur_q_R, alpha)
            # Make sure that we have no nans
            if DEBUG:
                assert np.isnan(next_q_L).sum() == 0
                assert np.isnan(next_q_R).sum() == 0
            Q_L_arr[cur_end_idx:next_trial_end_idx] = next_q_L
            Q_R_arr[cur_end_idx:next_trial_end_idx] = next_q_R

        if include_RewardRate:
            next_reward_rate = _updateNextRewardRate(cur_trials_outcome,
                                                     cur_reward_rate,
                                                     beta, group_every)
            if DEBUG:
                assert np.isnan(next_reward_rate).sum() == 0
            reward_rate_arr[cur_end_idx:next_trial_end_idx] = next_reward_rate


    mult_sess_df["SimStartingPoint"] = sim_starting_point
    mult_sess_df["SimRT"] = sim_rt
    mult_sess_df["SimChoiceCorrect"] = sim_choice_correct
    mult_sess_df["SimChoiceLeft"] = sim_choice_left
    # mult_sess_df["SimPrevChoiceLeft"] = sim_prev_choice_left
    # mult_sess_df["SimPrevChoiceCorrect"] = sim_prev_choice_correct
    # mult_sess_df["SimPrevDV"] = sim_prev_dv
    # mult_sess_df["SimMatchReal"] = sim_match_real
    if include_Q:
        mult_sess_df["Q_L"] = Q_L_arr[:matrix_shape]
        mult_sess_df["Q_R"] = Q_R_arr[:matrix_shape]
        mult_sess_df["Q_val"] = Q_val_arr
    if include_RewardRate:
        mult_sess_df["RewardRate"] = reward_rate_arr[:matrix_shape]
    
    # Resort the dataframe
    mult_sess_df = mult_sess_df.sort_values(["SessId", "TrialNumber"])
    mult_sess_df = mult_sess_df.reset_index(drop=True)

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


def _getKwargsFromCols(cols, df, cur_Q_val, cur_reward_rate):
    kwargs = {}
    for col in cols:
        if col == "Q_val":
            kwargs[col] = cur_Q_val
        elif col == "RewardRate":
            kwargs[col] = cur_reward_rate
        else:
            kwargs[col] = df[col].values
    return kwargs

_last_PrevChoiceLeft = None
_last_PrevChoiceCorrect = None
_last_PrevDV = None
def betweenTrialsCb(many_trials_df, cur_Q_val, cur_reward_rate,
                    biasFn, driftFn, noiseFn, NON_DECISION_TIME, BOUND,
                    biasFn_df_cols={}, driftFn_df_cols=[], noiseFn_df_cols=[],
                    multi_sess_df=None, **ddm_trial_kwargs):
    #DVabs = many_trials_df.DV.abs().values
    global run_logger, _last_PrevChoiceLeft, _last_PrevChoiceCorrect, _last_PrevDV
    dvs = many_trials_df.DV.values
    biasFn_kwargs = _getKwargsFromCols(biasFn_df_cols, many_trials_df, cur_Q_val, cur_reward_rate)
    driftFn_kwargs = _getKwargsFromCols(driftFn_df_cols, many_trials_df, cur_Q_val, cur_reward_rate)
    noiseFn_kwargs = _getKwargsFromCols(noiseFn_df_cols, many_trials_df, cur_Q_val, cur_reward_rate)

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
    #     print("Q_val:", cur_Q_val, " - RewardRate:", cur_reward_rate)
    #     print("Choice left:", choice_left, " - RTs:", rts)
    #     # display(many_trials_df)
    dV_left_mask = dvs > 0
    choice_correct = (choice_left == dV_left_mask).astype(np.float32)
    choice_correct[np.isnan(choice_left)] = np.nan
    
    if run_logger is not None:
        run_logger.TrialNumber = many_trials_df.TrialNumber
        run_logger.Name = many_trials_df.Name
        run_logger.SessId = many_trials_df.SessId
        run_logger.SimChoiceCorrect = choice_correct

    return choice_correct, choice_left, rts, starting_point_bounded


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


def chi2Loss(real_rt, sim_rt, t_dur):
    ratcliff_quantiles = [.1, .3, .5, .7, .9]
    bins = np.nanquantile(real_rt, ratcliff_quantiles)
    bins = np.insert(bins, 0, 0)
    bins = np.append(bins, t_dur)

    freq_real, _ = np.histogram(real_rt, bins=bins)
    freq_sim, _ = np.histogram(sim_rt, bins=bins)
    # Chi2 loss: sum((observed - expected)^2 / expected)
    bins_loss = (freq_sim - freq_real)**2 / freq_real
    bins_loss_sum = bins_loss.sum().astype(float)
    return bins_loss_sum


def calcLoss8(df, dt, t_dur):
    df = df[df.valid]
    loss = 0
    for choice in (1, 0):
        real_choice_rt = df[df["ChoiceCorrect"] == choice].calcStimulusTime
        sim_choice_rt = df[df["SimChoiceCorrect"] == choice].SimRT
        loss += chi2Loss(real_choice_rt, sim_choice_rt, t_dur)
    return loss




# import time
def makeOneRun(df, include_Q, include_RewardRate, biasFn, driftFn,
               noiseFn, ALPHA, BETA, NON_DECISION_TIME, BOUND,
               DRIFT_COEF, NOISE_SIGMA, dt, t_dur, 
               biasFn_df_cols=[],  biasFn_kwargs={}, 
               driftFn_df_cols=[], driftFn_kwargs={},
               noiseFn_df_cols=[], noiseFn_kwargs={},
               return_df=False):
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
