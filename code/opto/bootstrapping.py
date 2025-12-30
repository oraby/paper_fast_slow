
import numpy as np


def bootstrapPerf(subj_arr, subj_to_sess_id_arr_dict,
                  sess_id_to_trial_outcome_arr_dict,
                  sess_id_to_trial_label_arr_dict,
                  num_iterations,
                  calcPerfFn : callable,):
    """Performs hierarchical bootstrapping to calculate performance

    Args:
        data (dict): Dictionary containing the data.
        num_iterations (int): Number of bootstrap iterations.
        calcPerfFn (callable): Function to calculate performance
    Returns:
        np.array: Array of performance
    """
    performance_differences = []

    for _ in range(num_iterations):
        performance_difference = _runOneIteration(subj_arr,
                                            subj_to_sess_id_arr_dict,
                                            sess_id_to_trial_outcome_arr_dict,
                                            sess_id_to_trial_label_arr_dict,
                                            calcPerfFn)
        performance_differences.append(performance_difference)

    return np.array(performance_differences)

def _runOneIteration(subj_arr, subj_to_sess_id_arr_dict,
                     sess_id_to_trial_outcome_arr_dict,
                     sess_id_to_trial_label_arr_dict,
                     calcPerfFn):
    """Runs one iteration of hierarchical bootstrapping.

    Args:
        subj_arr (np.array): Array of subject IDs
        subj_to_sess_id_arr_dict (dict): Dictionary mapping subject ID to
                                         session ID array
        sess_id_to_trial_outcome_arr_dict (dict): Dictionary mapping session ID
                                                  to trial outcome array
        sess_id_to_trial_label_arr_dict (dict): Dictionary mapping session ID
                                                to trial label array
        calcPerfFn (callable): Function to calculate performance
    Returns:
        float: performance of resampled data
    """
    # 1. Resample mice with replacement
    resampled_subject_ids = np.random.choice(subj_arr,
                                             size=subj_arr.shape[0],
                                             replace=True)
    resampled_trials_cntrol = []
    resampled_trials_opto = []

    for subject_id in resampled_subject_ids:
        # 2. Resample sessions within each mouse
        subject_sess_ids = subj_to_sess_id_arr_dict[subject_id]
        resampled_session_ids = np.random.choice(subject_sess_ids,
                                                 size=subject_sess_ids.shape[0],
                                                 replace=True)

        for session_id in resampled_session_ids:
            # 3.  Resample trials within each session
            trials_outcome = sess_id_to_trial_outcome_arr_dict[session_id]
            trials_label = sess_id_to_trial_label_arr_dict[session_id]
            resampled_trial_indices = np.random.choice(trials_outcome.shape[0],
                                            size=(trials_outcome.shape[0], 2),
                                            replace=True)
            resampled_trials_outcome = trials_outcome[
                                                   resampled_trial_indices[:,0]]

            SHUFFLE_LABELS = False
            if SHUFFLE_LABELS:
                resampled_trials_label = trials_label[
                                                   resampled_trial_indices[:,1]]
            else:
                resampled_trials_label = trials_label[
                                                   resampled_trial_indices[:,0]]
            # 4. Separate control and opto trials
            control_trials = resampled_trials_outcome[
                                                    resampled_trials_label == 0]
            opto_trials = resampled_trials_outcome[resampled_trials_label == 1]
            resampled_trials_cntrol.extend(control_trials)
            resampled_trials_opto.extend(opto_trials)

    resampled_trials_cntrol = np.array(resampled_trials_cntrol)
    resampled_trials_opto = np.array(resampled_trials_opto)
    # Calculate performance from the combined resampled data
    perf = calcPerfFn(resampled_trials_cntrol, resampled_trials_opto)
    return perf

def _generateMockData(num_mice, num_sessions_per_mouse,
                      num_trials_per_session):
    """Generates mock behavioral data for multiple mice, sessions, and trials.

    Returns:
    dict: A dictionary representing the mock data.
          Keys are mouse IDs, values are dicts with session IDs as keys,
          and each session contains trial data (control vs. opto, success vs. failure)
    """
    data = {}
    for mouse_id in range(num_mice):
        data[mouse_id] = {}
        for session_id in range(num_sessions_per_mouse):
            trial_data = []
            for trial_id in range(num_trials_per_session):
                #Mock data for control (0) or opto (1), and success (1) or failure (0).
                trial_type = np.random.choice([1]) # 0 for control, 1 for opto
                success = np.random.choice([1], p=[0.3, 0.7] if trial_type==0 else [0.5,0.5]) #higher success rate for control
                trial_data.append((trial_type, success))
            data[mouse_id][session_id] = np.array(trial_data)
    return data

def _calcPerf(control_trials, opto_trials):
    """Calculates the performance (success rate) for control and opto trials.

    Args:
    trials (np.array): array of tuples (trial_type, success)

    Returns:
    tuple: (control_performance, opto_performance)
    """
    control_performance = np.mean(control_trials[:, 1]) if len(control_trials) > 0 else np.nan
    opto_performance = np.mean(opto_trials[:, 1]) if len(opto_trials) > 0 else np.nan

    return control_performance, opto_performance


if __name__ == "__main__":
    # --- Simulation ---
    num_mice = 5
    num_sessions_per_mouse = 3
    num_trials_per_session = 100
    num_iterations = 1000

    # Generate mock data
    mock_data = _generateMockData(num_mice, num_sessions_per_mouse,
                                  num_trials_per_session)

    # Perform bootstrapping
    performance_diffs = bootstrapPerf(mock_data, num_iterations,
                                      calcPerfFn=_calcPerf)

    # Analyze results
    print("Bootstrapped performance differences:")
    print(performance_diffs)

    # Calculate mean and confidence intervals (optional)
    mean_diff = np.mean(performance_diffs)
    std_err = np.std(performance_diffs)
    confidence_interval = (np.percentile(performance_diffs, 2.5),
                           np.percentile(performance_diffs, 97.5))

    print(f"\nMean performance difference: {mean_diff:.3f}")
    print(f"Standard error: {std_err:.3f}")
    print(f"95% Confidence Interval: {confidence_interval}")