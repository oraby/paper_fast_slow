'''One place the 2-photon notebooks load their frames from.

The four 2P notebooks opened ``data/2p`` twenty-four times between them, in
four different idioms -- ``pd.read_pickle``, ``open`` plus ``pickle.load``, a
name built from f-strings, and ``genrundata.loadRunDataDict`` -- each with the
path spelled out relative to the notebook. That meant a file could only be
found when the working directory happened to be ``code/``, and nothing said
which step produced which file.

Each artifact here has a function that says what it holds and what wrote it.
Paths resolve from **this file**, not the working directory, so the loaders
work from a script, a test, or a notebook run from anywhere.

**Everything here is read-only.** Writing is still done where the frame is
built, behind that notebook's ``SAVE_DATA`` flag, so nothing can be rewritten
by accident just by importing this.

Most of these files are too large for git and arrive in ``2p_data.zip`` via
``code/data_downloader.ipynb``; a missing one raises with that instruction
rather than a bare ``FileNotFoundError``.
'''
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

#: ``<repo>/data/2p``, resolved from this file rather than the caller's cwd.
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "2p"

_MISSING = ("{path} is missing.\n"
            "The large 2P frames are not in git: run code/data_downloader.ipynb "
            "to fetch 2p_data.zip, or rebuild this one from the notebook that "
            "writes it.")


def dataPath(name):
    '''Full path of a file in ``data/2p``, whether or not it exists.'''
    return DATA_DIR / name


def loadFrame(name):
    '''Read one pickle from ``data/2p`` by file name.'''
    path = dataPath(name)
    if not path.exists():
        raise FileNotFoundError(_MISSING.format(path=path))
    return pd.read_pickle(path)


def loadObject(name):
    '''Read one pickle that is not a DataFrame (a dict of results, say).'''
    path = dataPath(name)
    if not path.exists():
        raise FileNotFoundError(_MISSING.format(path=path))
    with open(path, "rb") as fp:
        return pickle.load(fp)


# --- raw and cut traces ------------------------------------------------------

def loadEpochTraces():
    '''Raw fluorescence per trial epoch: TwoPLoad's input (23 sessions).'''
    return loadFrame("df_all_by_epoch.pkl")


def loadFilteredDfF():
    '''dF/F traces of the accepted neurons -- TwoPLoad's output, read by the rest.'''
    return loadFrame("df_all_by_epoch_df_f_filtered.pkl")


def loadTrialNormalized():
    '''Per-trial normalised traces (TwoPTraces' heatmaps).'''
    return loadFrame("df_all_by_trial_normalized.pkl")


def loadTrialNormalizedFeedbackAligned():
    '''As above, with sampling concatenated across sessions and feedback aligned.'''
    return loadFrame("df_all_by_trial_normalized_concat_sampling_across_sessions"
                     "_aligned_feedback_0.5s_norm_limited_end.pkl")


def loadFeedbackTraces():
    '''Feedback-epoch traces of the accepted neurons.'''
    return loadFrame("traces_cut_feedback_filtered_full_df.pkl")


def loadSamplingNormalized(time_before=0.1, time_after=0.1):
    '''Sampling epochs, z-scored and time-normalised, cut at the given margins.'''
    return loadFrame(f"normalized_{time_before}s_before_sampling"
                     f"_{time_after}s_after_movement.pkl")


def loadSamplingUnnormalized(num_quantiles=3, time_before=0.1, time_after=0.1):
    '''The same cut, in real time and split into sampling-duration quantiles.'''
    return loadFrame(f"unnormalized_{num_quantiles}_quantiles_{time_before}s"
                     f"_before_sampling_{time_after}s_after_movement.pkl")


# --- neuron sets -------------------------------------------------------------

def loadActiveNeurons(epoch):
    '''Neurons active in an epoch: ``"sampling"``, ``"movement"`` or ``"unnormed"``.'''
    names = {"sampling": "active_neurons_sampling_df.pkl",
             "movement": "active_neurons_movement_df.pkl",
             "unnormed": "active_neurons_unnormed.pkl"}
    if epoch not in names:
        raise ValueError(f"epoch must be one of {sorted(names)}, not {epoch!r}")
    return loadFrame(names[epoch])


def loadNeuronsIqr():
    '''Per-neuron firing-position spread over all trials.'''
    return loadFrame("neurons_iqr_df_all_trials.pkl")


def loadSgfNeurons(which):
    '''Significantly tuned neurons: ``"all"`` variables, or ``"choice"`` only.'''
    names = {"all": "sgf_all.pkl", "choice": "sgf_choice.pkl"}
    if which not in names:
        raise ValueError(f"which must be one of {sorted(names)}, not {which!r}")
    return loadObject(names[which])


# --- per-neuron activity summaries ------------------------------------------

#: The four cached max-firing dicts, by the strategy they were built from.
MAX_FIRING_FILES = {"fast": "q1_res_dict.pkl", "typical": "q2_res_dict.pkl",
                    "slow": "q3_res_dict.pkl", "all": "qall_res_dict.pkl"}


def loadMaxFiring(strategy):
    '''One neuron per row: peak positions and active trials, per strategy.

    ``strategy`` is ``"fast"`` (quantile 1), ``"typical"`` (2), ``"slow"`` (3)
    or ``"all"``. These are the dicts 2pAnalysis's activity criterion produces
    and caches; rebuilding them takes about 20 minutes.
    '''
    if strategy not in MAX_FIRING_FILES:
        raise ValueError(f"strategy must be one of {sorted(MAX_FIRING_FILES)}, "
                         f"not {strategy!r}")
    return loadObject(MAX_FIRING_FILES[strategy])


def loadMaxFiringFrame(strategy):
    '''The same, as a DataFrame.'''
    return pd.DataFrame(loadMaxFiring(strategy))


# --- model / decoder / shuffle artifacts -------------------------------------

def loadRunData(descrp):
    '''The ROC run data, e.g. ``"Unnormalized"`` or ``"Feedback_fixed"``.'''
    # Imported here rather than at module scope: genrundata pulls in IPython,
    # which the plain frame loaders have no use for.
    from .genrundata import loadRunDataDict

    path = dataPath(f"data_runs_{descrp}.pkl")
    if not path.exists():
        raise FileNotFoundError(_MISSING.format(path=path))
    return loadRunDataDict(str(path))


def loadSvmDf():
    '''Decoder accuracies, one row per session per random split (Figure S12G).'''
    return loadObject("svm_df.pkl")


def loadRtCorrShuffled():
    '''The shuffled control for the rt/activity correlations (1,000 draws).'''
    return loadObject("rt_corr_shuffled.pkl")
