"""One MAP parameter cannot contradict itself, which the two booleans could."""
import pytest

from ..referencemap import MapValues, atlas, directory, isRedefined, resolve


def test_the_two_maps_are_the_whole_of_it():
    assert MapValues.ALL == (MapValues.REDEFINED, MapValues.STANDARD)
    assert {directory(m) for m in MapValues.ALL} == {"redefined_map", "standard_map"}
    assert {atlas(m) for m in MapValues.ALL} == {"dorsal_redefined", "dorsal_cortex"}


def test_the_default_is_the_papers_map():
    assert isRedefined(MapValues.REDEFINED)
    assert not isRedefined(MapValues.STANDARD)
    assert directory(MapValues.REDEFINED) == "redefined_map"


@pytest.mark.parametrize("given", ["standard", "STANDARD", " Standard "])
def test_case_and_padding_do_not_matter(given):
    assert resolve(given) == MapValues.STANDARD


def test_an_unknown_map_says_what_was_allowed():
    with pytest.raises(ValueError, match="redefined, standard"):
        resolve("allen")


def test_a_non_string_is_refused_by_type():
    # --param MAP=True used to be expressible; now it cannot mean anything.
    with pytest.raises(TypeError, match="must be a string"):
        resolve(True)


def test_every_accessor_validates():
    for fn in (directory, atlas, isRedefined):
        with pytest.raises(ValueError):
            fn("dorsal_cortex")          # the atlas name, not the MAP value
