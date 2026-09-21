"""The save directory the opto panels write into.

A full run died in `Psychometric/free_sampling/` because the per-subject
directory was never created -- the `mkdir` was commented out, so panels saved
only where an older run had left a folder behind.
"""
import pytest

from ..optoprocessor import prepareSaveDir


def test_the_subject_directory_is_created(tmp_path):
    prefix = tmp_path / "Psychometric" / "free_sampling"
    prefix.mkdir(parents=True)
    made = prepareSaveDir(f"{prefix}/", "All_Mice")
    assert made.is_dir()
    assert made == prefix / "All_Mice"


def test_it_is_fine_for_the_directory_to_exist_already(tmp_path):
    prefix = tmp_path / "Psychometric" / "fixedtime_full_inhib"
    (prefix / "vgat-40").mkdir(parents=True)
    assert prepareSaveDir(f"{prefix}/", "vgat-40").is_dir()


def test_a_mistyped_prefix_is_refused_rather_than_created(tmp_path):
    # parents=True would otherwise build the whole wrong tree in silence.
    with pytest.raises(AssertionError, match="doesn't exist"):
        prepareSaveDir(f"{tmp_path}/Psychomteric/free_sampling/", "All_Mice")
    assert not (tmp_path / "Psychomteric").exists()


def test_no_prefix_is_refused(tmp_path):
    with pytest.raises(AssertionError, match="save_prefix must be specified"):
        prepareSaveDir(None, "All_Mice")
