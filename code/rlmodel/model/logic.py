import numpy as np
import pandas as pd
from functools import partial

LOG_CIEL = .01
_ciel_max = np.log(1/LOG_CIEL)

run_logger = None

TRACK_PREV_CHOICE = False
FORCE_EWD = True
EWD_TIME = .3

def _calcQVal(Q_L, Q_R, group_every=0):#round_decimals=0):
    ''' Returns a value between -1 and 1'''
    Q_L = np.clip(np.asarray(Q_L), LOG_CIEL, 1)
    Q_R = np.clip(np.asarray(Q_R), LOG_CIEL, 1)
    Q_val = np.log(Q_L / Q_R) / _ciel_max
    if group_every != 0:
        Q_val = np.round(Q_val / group_every) * group_every
    # assert np.isnan(Q_val).sum() == 0 #, f"Q_val has NaNs: {Q_val} - Q_L: {Q_L} - Q_R: {Q_R}"
    return Q_val


def _updateNextQL_Q(cur_outcome, cur_choice_left, cur_q_L, cur_q_R, ALPHA):
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


def processMultipleSess(mult_sess_df, alpha, beta, include_Q,
                        include_RewardRate, #round_decimals
                        group_every, callBetweenTrialFn=None):
    min_trial_num = mult_sess_df.TrialNumber.min()
    max_trial_num = mult_sess_df.TrialNumber.max()
    num_sess = mult_sess_df.SessId.unique().shape[0]
    assert mult_sess_df.shape[0] == num_sess*(max_trial_num - min_trial_num + 1), (
        "The dataframe should contain all trials for all sessions")

    # TODO: sort_values() is not implemented in cudf, so check manually the
    # order of the trials
    mult_sess_df.sort_values(["TrialNumber", "SessId"], inplace=True)
    assert mult_sess_df.TrialNumber.is_monotonic_increasing, (
                                ", ".join(mult_sess_df.TrialNumber.astype(str)))
    mult_sess_df.reset_index(drop=True, inplace=True)
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

    if include_Q:
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
            assert multi_trials.shape[0] == num_sess, (f"Trial: {trial_idx} - "
                f"multi_trials.shape[0]={multi_trials.shape[0]} - num_sess={num_sess}")
            assert all(multi_trials.TrialNumber == trial_idx + min_trial_num)

        if include_Q:
            cur_q_L = Q_L_arr[cur_start_idx:cur_end_idx]
            cur_q_R = Q_R_arr[cur_start_idx:cur_end_idx]
            if DEBUG:
                assert cur_q_L.shape[0] == multi_trials.shape[0]
                assert cur_q_R.shape[0] == multi_trials.shape[0]
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
            if DEBUG:
                assert cur_reward_rate.shape[0] == multi_trials.shape[0]

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

        if FORCE_EWD:
            # Values that are dependent on whether the trial was rewarded should
            # flaged as no reward (rather than just wrong choice)
            cur_trials_outcome[cur_trials_sim_rts < EWD_TIME] = 0

        if include_Q:
            next_q_L, next_q_R = _updateNextQL_Q(cur_trials_outcome,
                                                 cur_trials_choice_left,
                                                 cur_q_L, cur_q_R, alpha)
            # Make sure that we have no nans
            # print("Trial:", trial_idx, " - next_q_L:", next_q_L[0], " - next_q_R:", next_q_R[0])
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

    if include_Q:
        # print("Q_L_arr:", Q_L_arr[::num_sess])
        # print("Q_L_arr:", Q_L_arr.shape, "mult_sess_df:", mult_sess_df.shape)
        mult_sess_df.loc[:,"Q_L"] = Q_L_arr[:matrix_shape]
        mult_sess_df.loc[:,"Q_R"] = Q_R_arr[:matrix_shape]
        mult_sess_df.loc[:,"Q_val"] = Q_val_arr
        # display(mult_sess_df.head(10))
    if include_RewardRate:
        mult_sess_df["RewardRate"] = reward_rate_arr[:matrix_shape]

    # Resort the dataframe
    mult_sess_df.sort_values(["SessId", "TrialNumber"], inplace=True)
    mult_sess_df.reset_index(drop=True, inplace=True)

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
    num_trials = dvs.shape[0]

    # Starting point is between -1 and 1, scale the starting point to be
    # between -bound and bound
    starting_point_bounded = starting_point * BOUND
    # Initialize the dx array
    size = num_trials, num_steps

    # Generate random noise for all steps and trials
    noise = noiseFn(size, dt, **noiseFn_kwargs)

    # Calculate the drift component
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

    starting_point = biasFn(size=many_trials_df.shape[0], **biasFn_kwargs)
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
    assert choice_left.shape[0] == many_trials_df.shape[0]
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
        assert first_trials.shape[0] == len(multi_sess_df.SessId.unique()), (
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
    # Get quantile to include only one item in the first bin, otherwise we
    # don't correctly detect non-decision time.
    # first_item_quantile = 1 / (real_rt.shape[0] + 1) + 1e-6
    ratcliff_quantiles = [.1, .3, .5, .7, .9]
    # ratcliff_quantiles = [0, first_item_quantile] + list(np.linspace(.1, .9, 18)) + [1]
    null_real = real_rt.isnull()
    null_real_count = null_real.sum()
    null_sim = sim_rt.isnull()
    null_sim_count = null_sim.sum()
    real_rt = real_rt[~null_real].values
    sim_rt = sim_rt[~null_sim].values

    bins = np.quantile(real_rt, ratcliff_quantiles)
    # Append zero and t_dur to the bins
    bins = np.insert(bins, 0, 0)
    bins = np.insert(bins, 1, real_rt.min() + 1e-6)  # To avoid zero-width first bin
    bins = np.append(bins, t_dur)

    freq_real, _ = np.histogram(real_rt, bins=bins)
    freq_sim, _ = np.histogram(sim_rt, bins=bins)
    # Chi2 loss: sum((observed - expected)^2 / expected)
    bins_loss = (freq_sim - freq_real)**2 / freq_real
    # print("RuntimeWarning in chi2Loss - freq_real:", freq_real, " - freq_sim:", freq_sim)
    # print("Bins:", bins)
    bins_loss_sum = bins_loss.sum().astype(float)
    # Add outliers, i.e null values
    if null_real_count:
        null_loss = (null_sim_count - null_real_count)**2 / null_real_count
        bins_loss_sum += null_loss
    return bins_loss_sum


def calcLoss(df, dt, t_dur, is_loss_no_dir):
    df = df[df.valid]
    loss = 0
    for choice_correct in (1, 0):
        real_choice_mask = df["ChoiceCorrect"] == choice_correct
        sim_choice_mask = df["SimChoiceCorrect"] == choice_correct
        if is_loss_no_dir:
            real_choice_rt = df[real_choice_mask].calcStimulusTime
            sim_choice_rt = df[sim_choice_mask].SimRT
            loss += chi2Loss(real_choice_rt, sim_choice_rt, t_dur)
        else:
            for choice_left in (1, 0):
                dir_real_mask = df["ChoiceLeft"] == choice_left
                dir_sim_mask = df["SimChoiceLeft"] == choice_left
                real_choice_rt = df[real_choice_mask & dir_real_mask].calcStimulusTime
                sim_choice_rt = df[sim_choice_mask & dir_sim_mask].SimRT
                loss += chi2Loss(real_choice_rt, sim_choice_rt, t_dur)

    return loss

# TODO: IF we have multiple GPUs, then we should receive a parameter telling us
# which GPU to use, or our child-process index and the number of GPUs.
_last_df_name = None
_last_df = None
def makeOneRun(df, include_Q, include_RewardRate, biasFn, driftFn,
               noiseFn, NON_DECISION_TIME, BOUND, DRIFT_COEF, NOISE_SIGMA,
               dt, t_dur, ALPHA=np.nan, BETA=np.nan,
               biasFn_df_cols=[],  biasFn_kwargs={},
               driftFn_df_cols=[], driftFn_kwargs={},
               noiseFn_df_cols=[], noiseFn_kwargs={},
               is_loss_no_dir=False,
               return_df=False):
    global np, pd
    global _last_df_name, _last_df
    from . import bias
    from . import drift
    from . import noise

    if include_Q:
        assert ~np.isnan(ALPHA), "ALPHA must be provided if include_Q is True"
    if include_RewardRate:
        assert ~np.isnan(BETA), (
                          "BETA must be provided if include_RewardRate is True")

    org_sess_id = df.SessId
    org_dv_str = df.DVstr
    org_name = df.Name
    has_date = "Date" in df.columns
    if has_date:
        orig_date = df.Date
        date_col = ["Date"]
    else:
        date_col = []
    REPEAT = 1
    if REPEAT > 1 and _last_df_name != df.Name.iloc[0]:
        _last_df_name = df.Name.iloc[0]
        if REPEAT > 1: # TODO: Remove this. Redundant check
            import cupy as cp
            import cudf
            df = df.copy()
            # Assign each session a unique id
            uniq_sess_id = df.SessId.unique()
            sess_id_map = dict(zip(uniq_sess_id, range(len(uniq_sess_id))))
            df["SessId"] = df.SessId.map(sess_id_map)
            df["OrgSessId"] = df.SessId
            # Now by REPEAT to gurantee a unique id for each session/trial
            df["SessId"] *= REPEAT
            # Drop the strubg columns as they consume a lot of memory
            df.drop(columns=["DVstr", "Name"] + date_col, inplace=True)
            # Convert 64 int or float to 32
            for col in df.columns:
                if df[col].dtype == np.int64:
                    df[col] = df[col].astype(np.int32)
                elif df[col].dtype == np.float32:
                    df[col] = df[col].astype(np.float32)
            # Print columns and their data types
            print(df.dtypes)
            df = cudf.from_pandas(df)
            np = cp
            pd = cudf
            bias.np = cp
            drift.np = cp
            noise.np = cp
            df = df.loc[np.repeat(df.index, REPEAT)]
            df = df.reset_index(drop=True)
            # df["SessId"] = df.SessId + "_" + (df.index % REPEAT).astype(str)
            df["SessId"] = df.SessId + (np.arange(len(df)) % REPEAT)
            # display(df.head())
            # Print memory usage
            print(f"Memory usage: {df.memory_usage(deep=True).sum()/1e6:.2f} MB")
        df = df.copy()
        _last_df = df
    else:
        # print("Using last df")
        # if REPEAT > 1:
        #     df = _last_df
        # else:
            df = df.copy()

    # Fix the random seed
    # np.random.seed(0)
    rnd_rng = np.random.default_rng(seed=0)
    bias.rnd_default_rng = rnd_rng
    drift.rnd_default_rng = rnd_rng
    noise.rnd_default_rng = rnd_rng

    # print("FIxed params:", fixed_params_names)
    # df = df.copy() # Must we have this if are running in parallel?
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
    total_loss = calcLoss(processed_df, dt, t_dur,
                          is_loss_no_dir=is_loss_no_dir)
    if REPEAT > 1:
        # THen it is a cudf dataframe
        total_loss = total_loss.get()
    # print(f"Loss calc = {time.time() - time_start:.2f}")
    if return_df:
        if REPEAT > 1:
            # Drop the repeated trials
            # processed_df = processed_df.drop_duplicates(subset=["SessId", "TrialNumber"])
            processed_df = processed_df[processed_df.TrialNumber % REPEAT == 0]
            processed_df = processed_df.to_pandas()
            processed_df.sort_values(["SessId", "TrialNumber"], inplace=True)
            processed_df["SessId"] = org_sess_id
            processed_df["DVstr"] = org_dv_str
            processed_df["Name"] = org_name
            if has_date:
                processed_df["Date"] = orig_date
        return total_loss, processed_df.copy() # Don't mess with _last_df
    else:
        return total_loss
