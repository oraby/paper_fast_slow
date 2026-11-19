'''Tests for the shared 2P loader.

The point of the module is that a frame can be found without the working
directory happening to be ``code/``, and that a missing file says what to do
about it rather than raising a bare FileNotFoundError.

These run against a temporary data directory, so they neither need the real
8 GB of frames nor can touch them.
'''
from __future__ import annotations

import pickle

import pandas as pd
import pytest

from .. import dataload


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(dataload, "DATA_DIR", tmp_path)
    return tmp_path


def writeFrame(data_dir, name, df=None):
    df = pd.DataFrame({"x": [1, 2, 3]}) if df is None else df
    df.to_pickle(data_dir / name)
    return df


def writeObject(data_dir, name, obj):
    with open(data_dir / name, "wb") as fp:
        pickle.dump(obj, fp)
    return obj


def test_paths_resolve_from_the_package_not_the_caller():
    """The real DATA_DIR is the repo's, whatever the working directory is."""
    assert dataload.DATA_DIR.name == "2p"
    assert dataload.DATA_DIR.parent.name == "data"
    assert dataload.DATA_DIR.is_absolute()


def test_a_frame_is_read_by_name(data_dir):
    writeFrame(data_dir, "df_all_by_epoch.pkl")
    assert list(dataload.loadEpochTraces().x) == [1, 2, 3]


def test_an_object_that_is_not_a_frame_comes_back_as_it_was(data_dir):
    writeObject(data_dir, "svm_df.pkl", {"accuracy": [0.9]})
    assert dataload.loadSvmDf() == {"accuracy": [0.9]}


def test_a_missing_file_says_how_to_get_it(data_dir):
    with pytest.raises(FileNotFoundError, match="data_downloader"):
        dataload.loadFilteredDfF()


def test_the_message_names_the_file(data_dir):
    with pytest.raises(FileNotFoundError, match="svm_df.pkl"):
        dataload.loadSvmDf()


def test_the_sampling_window_is_part_of_the_name(data_dir):
    writeFrame(data_dir, "normalized_0.1s_before_sampling_0.1s_after_movement.pkl")
    assert len(dataload.loadSamplingNormalized(0.1, 0.1)) == 3
    with pytest.raises(FileNotFoundError, match="0.2s_before"):
        dataload.loadSamplingNormalized(0.2, 0.1)


def test_the_quantile_count_is_part_of_the_unnormalised_name(data_dir):
    writeFrame(data_dir, "unnormalized_3_quantiles_0.1s_before_sampling"
                         "_0.1s_after_movement.pkl")
    assert len(dataload.loadSamplingUnnormalized(3, 0.1, 0.1)) == 3
    with pytest.raises(FileNotFoundError, match="unnormalized_5_quantiles"):
        dataload.loadSamplingUnnormalized(5, 0.1, 0.1)


@pytest.mark.parametrize("epoch, name", [
    ("sampling", "active_neurons_sampling_df.pkl"),
    ("movement", "active_neurons_movement_df.pkl"),
    ("unnormed", "active_neurons_unnormed.pkl")])
def test_each_active_neuron_set_has_its_own_file(data_dir, epoch, name):
    writeFrame(data_dir, name)
    assert len(dataload.loadActiveNeurons(epoch)) == 3


def test_an_unknown_neuron_set_is_refused(data_dir):
    with pytest.raises(ValueError, match="epoch must be one of"):
        dataload.loadActiveNeurons("feedback")


@pytest.mark.parametrize("strategy, name", [
    ("fast", "q1_res_dict.pkl"), ("typical", "q2_res_dict.pkl"),
    ("slow", "q3_res_dict.pkl"), ("all", "qall_res_dict.pkl")])
def test_max_firing_dicts_are_named_by_strategy(data_dir, strategy, name):
    writeObject(data_dir, name, {"trace_id": ["n1"], "prcnt_valid": [50.0]})
    assert dataload.loadMaxFiring(strategy)["trace_id"] == ["n1"]
    assert len(dataload.loadMaxFiringFrame(strategy)) == 1


def test_an_unknown_strategy_is_refused(data_dir):
    with pytest.raises(ValueError, match="strategy must be one of"):
        dataload.loadMaxFiring("quick")


def test_the_two_tuned_neuron_sets_are_kept_apart(data_dir):
    writeObject(data_dir, "sgf_all.pkl", "every variable")
    writeObject(data_dir, "sgf_choice.pkl", "choice only")
    assert dataload.loadSgfNeurons("all") == "every variable"
    assert dataload.loadSgfNeurons("choice") == "choice only"
    with pytest.raises(ValueError, match="which must be one of"):
        dataload.loadSgfNeurons("prior")


def test_data_path_does_not_require_the_file_to_exist(data_dir):
    path = dataload.dataPath("not_written_yet.pkl")
    assert path.parent == data_dir
    assert not path.exists()


def test_the_loader_never_writes(data_dir):
    """Importing or reading must not create anything."""
    writeFrame(data_dir, "df_all_by_epoch.pkl")
    before = sorted(p.name for p in data_dir.iterdir())
    dataload.loadEpochTraces()
    dataload.dataPath("something.pkl")
    assert sorted(p.name for p in data_dir.iterdir()) == before
