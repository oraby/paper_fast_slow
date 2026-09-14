'''Hierarchical bootstrap over subjects, sessions and trials (Figure 3D).

The published p-values come from the sign rule applied to
:func:`bootstrapPerf`'s output -- see ``opto/optoprocessor.py`` for the caller
and ``opto/tests/test_bootstrapping.py`` for what the resampling guarantees.

:mod:`opto.bootstrap2regions` implements a different estimator on the same
idea: it averages one effect *per subject* rather than pooling every
resampled trial, and adds the cross-region comparison for Figure 4C.

**Randomness.** By default the draws come from the global ``numpy.random``
state, exactly as they always have. Nothing in the repo seeds that state
before Figure 3D is computed, so the published p-value is not reproducible
bit-for-bit. Pass ``rng`` to make a run reproducible: an ``int`` behaves like
``np.random.seed(int)`` followed by the default call (same legacy stream), and
a ``numpy.random.RandomState`` or ``Generator`` is used as given.
'''
import numpy as np


def _resolveRng(rng):
    '''``None`` -> the global numpy state; ``int`` -> a legacy seeded stream.'''
    if rng is None:
        return np.random
    if isinstance(rng, (int, np.integer)):
        return np.random.RandomState(rng)
    return rng


def bootstrapPerf(subj_arr, subj_to_sess_id_arr_dict,
                  sess_id_to_trial_outcome_arr_dict,
                  sess_id_to_trial_label_arr_dict,
                  num_iterations,
                  calcPerfFn : callable,
                  rng=None):
    """Performs hierarchical bootstrapping to calculate performance

    Args:
        data (dict): Dictionary containing the data.
        num_iterations (int): Number of bootstrap iterations.
        calcPerfFn (callable): Function to calculate performance
        rng (None | int | np.random.RandomState | np.random.Generator):
            Source of randomness. ``None`` (default) uses the global numpy
            state, as before. See the module docstring.
    Returns:
        np.array: Array of performance
    """
    rng = _resolveRng(rng)
    performance_differences = []

    for _ in range(num_iterations):
        performance_difference = _runOneIteration(subj_arr,
                                            subj_to_sess_id_arr_dict,
                                            sess_id_to_trial_outcome_arr_dict,
                                            sess_id_to_trial_label_arr_dict,
                                            calcPerfFn, rng)
        performance_differences.append(performance_difference)

    return np.array(performance_differences)

def _runOneIteration(subj_arr, subj_to_sess_id_arr_dict,
                     sess_id_to_trial_outcome_arr_dict,
                     sess_id_to_trial_label_arr_dict,
                     calcPerfFn, rng):
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
        rng: Anything with numpy's ``choice`` -- see :func:`_resolveRng`
    Returns:
        float: performance of resampled data
    """
    # 1. Resample mice with replacement
    resampled_subject_ids = rng.choice(subj_arr,
                                       size=subj_arr.shape[0],
                                       replace=True)
    resampled_trials_cntrol = []
    resampled_trials_opto = []

    for subject_id in resampled_subject_ids:
        # 2. Resample sessions within each mouse
        subject_sess_ids = subj_to_sess_id_arr_dict[subject_id]
        resampled_session_ids = rng.choice(subject_sess_ids,
                                           size=subject_sess_ids.shape[0],
                                           replace=True)

        for session_id in resampled_session_ids:
            # 3.  Resample trials within each session
            trials_outcome = sess_id_to_trial_outcome_arr_dict[session_id]
            trials_label = sess_id_to_trial_label_arr_dict[session_id]
            resampled_trial_indices = rng.choice(trials_outcome.shape[0],
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
