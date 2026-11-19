import numpy as np
import pandas as pd
from functools import partial
from . import state_updates

LOG_CIEL = state_updates.LOG_CIEL
_ciel_max = state_updates.LOG_CEIL_MAX

run_logger = None

TRACK_PREV_CHOICE = False
FORCE_EWD = True
EWD_TIME = .3

def _calcQVal(Q_L, Q_R, group_every=0):
    ''' Returns a value between -1 and 1'''
    return state_updates.compute_q_value(Q_L, Q_R, group_every=group_every)


def _updateNextQL_Q(cur_outcome, cur_choice_left, cur_q_L, cur_q_R, ALPHA):
    '''Calculate the new Q_L and Q_R values'''
    return state_updates.update_q_values(
        cur_q_L, cur_q_R, cur_choice_left, cur_outcome, ALPHA)


def _updateNextRewardRate(cur_outcome, cur_reward_rate, BETA, group_every=0):
    '''Calculate the new reward rate'''
    return state_updates.update_reward_rate(
        cur_reward_rate, cur_outcome, BETA, group_every=group_every)


def processMultipleSess(mult_sess_df, alpha, beta, include_Q,
                        include_RewardRate,
                        group_every, callBetweenTrialFn=None,
                        latent_nudge_sd=0.0, latent_nudge_rng=None):
    min_trial_num = mult_sess_df.TrialNumber.min()
    max_trial_num = mult_sess_df.TrialNumber.max()
    num_sess = mult_sess_df.SessId.unique().shape[0]
    assert mult_sess_df.shape[0] == num_sess*(max_trial_num - min_trial_num + 1), (
        "The dataframe should contain all trials for all sessions")

    mult_sess_df.sort_values(["TrialNumber", "SessId"], inplace=True)
    assert mult_sess_df.TrialNumber.is_monotonic_increasing, (
                                ", ".join(mult_sess_df.TrialNumber.astype(str)))
    mult_sess_df.reset_index(drop=True, inplace=True)
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
                _copied = True
            mult_sess_df["RewardRate"] = np.nan

    assert not _copied, "This should not happen if we want high performance"

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
        reward_rate_arr = np.full(matrix_shape_extra, np.nan, dtype=np.float32)
        reward_rate_arr[:num_sess] = .5
    else:
        # Create mock current reward rate
        cur_reward_rate = np.full(num_sess, np.nan, dtype=np.float32)


    for trial_idx in range(max_trial_num):
        cur_start_idx = trial_idx*num_sess
        cur_end_idx = (trial_idx+1)*num_sess
        next_trial_end_idx = cur_end_idx + num_sess
        multi_trials = mult_sess_df.iloc[cur_start_idx:cur_end_idx]

        if include_Q:
            cur_q_L = Q_L_arr[cur_start_idx:cur_end_idx]
            cur_q_R = Q_R_arr[cur_start_idx:cur_end_idx]
            cur_Q_val = _calcQVal(cur_q_L, cur_q_R, group_every=group_every)
            Q_val_arr[cur_start_idx:cur_end_idx] = cur_Q_val

        if include_RewardRate:
            cur_reward_rate = reward_rate_arr[cur_start_idx:cur_end_idx]

        if callBetweenTrialFn is None:
            cur_trials_outcome = multi_trials.ChoiceCorrect.values
            cur_trials_choice_left = multi_trials.ChoiceLeft.values
        else:
            # Function returns: choice_correct, choice_left, rts, starting_point_bounded
            (cur_trials_sim_choice_correct, cur_trials_sim_choice_left,
             cur_trials_sim_rts, cur_trials_sim_starting_poing) = \
                    callBetweenTrialFn(multi_trials, cur_Q_val, cur_reward_rate)

            sim_choice_correct[cur_start_idx:cur_end_idx] = cur_trials_sim_choice_correct
            sim_choice_left[cur_start_idx:cur_end_idx] = cur_trials_sim_choice_left
            sim_rt[cur_start_idx:cur_end_idx] = cur_trials_sim_rts
            sim_starting_point[cur_start_idx:cur_end_idx] = cur_trials_sim_starting_poing
            cur_trials_outcome = cur_trials_sim_choice_correct
            cur_trials_choice_left = cur_trials_sim_choice_left

        if FORCE_EWD:
            # Values that depend on whether the trial was rewarded are
            # flagged as no reward (rather than just wrong choice)
            cur_trials_outcome[cur_trials_sim_rts < EWD_TIME] = 0

        if include_Q:
            next_q_L, next_q_R = _updateNextQL_Q(cur_trials_outcome,
                                                 cur_trials_choice_left,
                                                 cur_q_L, cur_q_R, alpha)
            if latent_nudge_rng is not None and latent_nudge_sd:
                next_q_L, next_q_R = state_updates.nudge_q_values(
                    next_q_L, next_q_R, cur_trials_choice_left,
                    latent_nudge_sd, latent_nudge_rng)
            Q_L_arr[cur_end_idx:next_trial_end_idx] = next_q_L
            Q_R_arr[cur_end_idx:next_trial_end_idx] = next_q_R

        if include_RewardRate:
            next_reward_rate = _updateNextRewardRate(cur_trials_outcome,
                                                     cur_reward_rate,
                                                     beta,
                                                     group_every=group_every)
            if latent_nudge_rng is not None and latent_nudge_sd:
                next_reward_rate = state_updates.nudge_reward_rate(
                    next_reward_rate, latent_nudge_sd, latent_nudge_rng)
            reward_rate_arr[cur_end_idx:next_trial_end_idx] = next_reward_rate


    mult_sess_df["SimStartingPoint"] = sim_starting_point
    mult_sess_df["SimRT"] = sim_rt
    mult_sess_df["SimChoiceCorrect"] = sim_choice_correct
    mult_sess_df["SimChoiceLeft"] = sim_choice_left

    if include_Q:
        mult_sess_df.loc[:,"Q_L"] = Q_L_arr[:matrix_shape]
        mult_sess_df.loc[:,"Q_R"] = Q_R_arr[:matrix_shape]
        mult_sess_df.loc[:,"Q_val"] = Q_val_arr
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
                     noiseFn_kwargs={}, uses_scaled_bound=False):
    global runs_dx, runs_noise, runs_starting_point, runs_dvs, runs_bound
    max_possible_dt = max_dt #- nondectime
    num_steps = int(np.ceil(max_possible_dt/dt))
    num_trials = dvs.shape[0]

    # Default semantic: starting_point is in [-1, 1] (fraction of bound)
    # so we multiply by BOUND to get absolute DDM-state units. Under
    # ``--scale-bound`` BOUND is the fitted axis and bias is already in
    # absolute units (see state_updates.compute_starting_point_z with the
    # ``bound`` kwarg + simulateDDM in fit.py); skip the multiplication
    # so the bias doesn't double-scale.
    if uses_scaled_bound:
        starting_point_bounded = starting_point
    else:
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

    if run_logger is not None:
        run_logger.rts = rts
        run_logger.choice_left = choice_left

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
                            latent_nudge_sd=0.0, latent_nudge_rng=None,
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
                              group_every=0, callBetweenTrialFn=partialBetweenTrialsCb,
                              latent_nudge_sd=latent_nudge_sd,
                              latent_nudge_rng=latent_nudge_rng)
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


def makeOneRun(df, include_Q, include_RewardRate, biasFn, driftFn,
               noiseFn, NON_DECISION_TIME, BOUND, DRIFT_COEF, NOISE_SIGMA,
               dt, t_dur, ALPHA=np.nan, BETA=np.nan,
               biasFn_df_cols=[],  biasFn_kwargs={},
               driftFn_df_cols=[], driftFn_kwargs={},
               noiseFn_df_cols=[], noiseFn_kwargs={},
               is_loss_no_dir=False,
               return_df=False,
               seed=0,
               skip_loss=False,
               latent_nudge_sd=0.0, latent_nudge_seed=0):
    from . import bias
    from . import drift
    from . import noise

    if include_Q:
        assert ~np.isnan(ALPHA), "ALPHA must be provided if include_Q is True"
    if include_RewardRate:
        assert ~np.isnan(BETA), (
                          "BETA must be provided if include_RewardRate is True")

    df = df.copy()

    # Fix the random seed
    rnd_rng = np.random.default_rng(seed=seed)
    bias.rnd_default_rng = rnd_rng
    drift.rnd_default_rng = rnd_rng
    noise.rnd_default_rng = rnd_rng

    if len(driftFn_kwargs):
        driftFn = partial(driftFn, **driftFn_kwargs)
    if len(noiseFn_kwargs):
        noiseFn = partial(noiseFn, **noiseFn_kwargs)
    if len(biasFn_kwargs):
        biasFn = partial(biasFn, **biasFn_kwargs)

    # Off unless asked for: with sd 0 no generator is made and the update path
    # is exactly the one the fits were made with.
    latent_nudge_rng = (np.random.default_rng(latent_nudge_seed)
                        if latent_nudge_sd else None)
    processed_df = simulateDDMMultipleSess(df, latent_nudge_sd=latent_nudge_sd,
                                           latent_nudge_rng=latent_nudge_rng,
                                           include_Q=include_Q, include_RewardRate=include_RewardRate,
                                           biasFn=biasFn,   biasFn_df_cols=biasFn_df_cols,
                                           driftFn=driftFn, driftFn_df_cols=driftFn_df_cols,
                                           noiseFn=noiseFn, noiseFn_df_cols=noiseFn_df_cols,
                                           ALPHA=ALPHA, BETA=BETA,
                                           NON_DECISION_TIME=NON_DECISION_TIME,
                                           BOUND=BOUND, DRIFT_COEF=DRIFT_COEF,
                                           NOISE_SIGMA=NOISE_SIGMA, dt=dt,
                                           max_dt=t_dur)
    if skip_loss:
        total_loss = np.nan
    else:
        total_loss = calcLoss(processed_df, dt, t_dur,
                              is_loss_no_dir=is_loss_no_dir)
    if return_df:
        return total_loss, processed_df.copy()
    else:
        return total_loss
