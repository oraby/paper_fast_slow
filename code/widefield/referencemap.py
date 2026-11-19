"""Which cortical reference map the widefield panels are drawn on.

One parameter, ``MAP``, names the map, so the choice cannot be
self-contradictory. The notebook declares ``MAP = MapValues.REDEFINED``;
``code/run_notebooks.py`` runs it once more with ``MAP = "standard"``, and
``--param MAP=standard`` picks one map for a run.

The value is a plain string on purpose: papermill injects literals, so the
parameters cell cannot hold an enum member. ``MapValues`` gives the names to
compare against, and :func:`resolve` rejects anything else by listing what is
allowed rather than failing somewhere downstream on an undefined variable.
"""
from __future__ import annotations


class MapValues:
    """The reference maps, as the ``MAP`` parameter spells them."""

    #: The MFC/LFC redefinition this paper uses. The default.
    REDEFINED = "redefined"
    #: The standard Allen dorsal-cortex parcellation.
    STANDARD = "standard"

    #: Every accepted value, in the order they are offered.
    ALL = (REDEFINED, STANDARD)


#: ``results/WF/<directory>/`` for each map.
_DIRECTORY = {MapValues.REDEFINED: "redefined_map",
              MapValues.STANDARD: "standard_map"}

#: The atlas name the imaging pipeline knows each map by.
_ATLAS = {MapValues.REDEFINED: "dorsal_redefined",
          MapValues.STANDARD: "dorsal_cortex"}


def resolve(value: str) -> str:
    """Normalise and check one ``MAP`` value, or say what was allowed."""
    if not isinstance(value, str):
        raise TypeError(f"MAP must be a string, got {type(value).__name__}: "
                        f"{value!r}. One of {', '.join(MapValues.ALL)}.")
    got = value.strip().lower()
    if got not in _DIRECTORY:
        raise ValueError(f"MAP={value!r} is not a reference map. "
                         f"One of {', '.join(MapValues.ALL)}.")
    return got


def directory(value: str) -> str:
    """The ``results/WF/`` sub-directory this map's figures are written to."""
    return _DIRECTORY[resolve(value)]


def atlas(value: str) -> str:
    """The atlas name the imaging pipeline expects for this map."""
    return _ATLAS[resolve(value)]


def isRedefined(value: str) -> bool:
    """Whether this is the MFC/LFC map, which most panels are drawn on."""
    return resolve(value) == MapValues.REDEFINED
