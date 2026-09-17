"""Tests for the notebook runner and the parameters contract it relies on.

The contract, checked against the real notebooks on every test run:

- every figure notebook has exactly one cell tagged ``parameters``;
- that cell declares ``SAVE_FIGS``, ``SAVE_DATA`` and ``PAPER_FIGURES_ONLY``,
  all ``False``, so running a notebook as it stands writes nothing;
- no other cell redefines them, and none of their uses comes before that cell;
- no cell passes ``save_figs=True`` literally, which would ignore the flag.

These are what make ``uv run python code/run_notebooks.py`` safe to point at the
whole repository. Nothing here executes a notebook.
"""
from __future__ import annotations

import json
import re

import pytest

from ... import run_notebooks as rn

FLAGS = list(rn.STANDARD_PARAMETERS)


def _load(relpath):
    return json.load(open(rn.CODE_DIR / relpath, encoding="utf-8"))


def _code(cell):
    return "".join(cell["source"]) if cell["cell_type"] == "code" else ""


def _uncommented(text):
    return "\n".join(line for line in text.split("\n")
                     if not line.strip().startswith("#"))


# --- the notebooks keep the contract ------------------------------------------

@pytest.mark.parametrize("relpath", rn.NOTEBOOKS)
def test_the_notebook_exists(relpath):
    assert (rn.CODE_DIR / relpath).is_file()


@pytest.mark.parametrize("relpath", rn.NOTEBOOKS)
def test_one_parameters_cell_declaring_every_flag_off(relpath):
    declared = rn.declaredParameters(_load(relpath))
    for flag in FLAGS:
        assert declared.get(flag) is False, f"{flag} must default to False"


@pytest.mark.parametrize("relpath", rn.NOTEBOOKS)
def test_no_other_cell_redefines_a_flag(relpath):
    nb = _load(relpath)
    params = rn.parametersCell(nb)
    pattern = re.compile(rf"^\s*({'|'.join(FLAGS)})\s*=", re.M)
    offenders = [i for i, cell in enumerate(nb["cells"])
                 if cell is not params and pattern.search(_uncommented(_code(cell)))]
    assert not offenders, f"cells {offenders} redefine a save flag"


@pytest.mark.parametrize("relpath", rn.NOTEBOOKS)
def test_flags_are_not_used_before_they_are_declared(relpath):
    nb = _load(relpath)
    params = rn.parametersCell(nb)
    params_at = next(i for i, c in enumerate(nb["cells"]) if c is params)
    pattern = re.compile(rf"\b({'|'.join(FLAGS)})\b")
    early = [i for i, cell in enumerate(nb["cells"][:params_at])
             if pattern.search(_uncommented(_code(cell)))]
    assert not early, f"cells {early} use a flag before the parameters cell"


@pytest.mark.parametrize("relpath", rn.NOTEBOOKS)
def test_no_literal_save_true(relpath):
    literal = re.compile(r"\bsave_figs?\s*=\s*True\b")
    offenders = [i for i, cell in enumerate(_load(relpath)["cells"])
                 if literal.search(_uncommented(_code(cell)))]
    assert not offenders, f"cells {offenders} save regardless of SAVE_FIGS"


# --- the runner's own logic ----------------------------------------------------

def test_selecting_by_name_keeps_run_order():
    chosen = rn.selectNotebooks(["opto", "behavior"])
    assert chosen == ["behavior.ipynb", "opto.ipynb"]


def test_selecting_a_nested_notebook_by_its_stem():
    assert rn.selectNotebooks(["model_analysis"]) == ["rlmodel/model_analysis.ipynb"]


def test_an_unknown_name_is_refused():
    with pytest.raises(SystemExit, match="Unknown notebook"):
        rn.selectNotebooks(["behaviour"])


def test_no_selection_means_every_notebook():
    assert rn.selectNotebooks(None) == rn.NOTEBOOKS


def _nb(parameters_source, extra_cells=()):
    cells = [{"cell_type": "code", "metadata": {"tags": ["parameters"]},
              "source": [parameters_source]}]
    cells += [{"cell_type": "code", "metadata": {}, "source": [s]} for s in extra_cells]
    return {"cells": cells}


def test_declared_parameters_are_read_with_their_defaults():
    nb = _nb("SAVE_FIGS = False\nSAVE_DATA = False\nPAPER_FIGURES_ONLY = False\n"
             "MFC_LFC_MAP = True  # a notebook's own switch")
    assert rn.declaredParameters(nb) == {
        "SAVE_FIGS": False, "SAVE_DATA": False, "PAPER_FIGURES_ONLY": False,
        "MFC_LFC_MAP": True}


def test_two_parameters_cells_are_refused():
    nb = _nb("SAVE_FIGS = False")
    nb["cells"].append({"cell_type": "code", "metadata": {"tags": ["parameters"]},
                        "source": ["SAVE_DATA = False"]})
    with pytest.raises(ValueError, match="exactly one"):
        rn.parametersCell(nb)


def test_the_standard_flags_are_always_injected():
    declared = dict.fromkeys(FLAGS, False)
    params = rn.buildParameters(declared,
                                {"save_figs": True, "save_data": False,
                                 "paper_figures_only": True}, extra={})
    assert params == {"SAVE_FIGS": True, "SAVE_DATA": False, "PAPER_FIGURES_ONLY": True}


def test_an_extra_parameter_goes_only_where_it_is_declared():
    flags = dict.fromkeys(["save_figs", "save_data", "paper_figures_only"], False)
    widefield = dict.fromkeys(FLAGS, False) | {"MFC_LFC_MAP": True}
    behavior = dict.fromkeys(FLAGS, False)
    extra = {"MFC_LFC_MAP": False}
    assert rn.buildParameters(widefield, flags, extra)["MFC_LFC_MAP"] is False
    assert "MFC_LFC_MAP" not in rn.buildParameters(behavior, flags, extra)


def test_a_notebook_missing_a_standard_flag_is_refused():
    flags = dict.fromkeys(["save_figs", "save_data", "paper_figures_only"], False)
    with pytest.raises(ValueError, match="PAPER_FIGURES_ONLY"):
        rn.buildParameters({"SAVE_FIGS": False, "SAVE_DATA": False}, flags, {})


def test_an_extra_no_notebook_declares_is_flagged_as_a_typo():
    declared = {"widefield.ipynb": {"MFC_LFC_MAP": True},
                "behavior.ipynb": {"SAVE_FIGS": False}}
    assert rn.unusedExtra({"MFC_LFC_MAP": 1, "TYPO": 2}, declared) == ["TYPO"]


@pytest.mark.parametrize("raw, expected", [
    (["MFC_LFC_MAP=False"], {"MFC_LFC_MAP": False}),
    (["N=3"], {"N": 3}),
    (["LABEL=hello"], {"LABEL": "hello"}),        # bare strings are allowed
])
def test_extra_parameters_are_parsed_as_python_literals(raw, expected):
    assert rn.parseExtra(raw) == expected


@pytest.mark.parametrize("bad", ["NOEQUALS", "1BAD=2", "=3"])
def test_a_malformed_extra_parameter_is_refused(bad):
    with pytest.raises(SystemExit, match="NAME=VALUE"):
        rn.parseExtra([bad])


def test_list_mode_runs_nothing(capsys):
    assert rn.main(["--list", "--only", "opto"]) == 0
    assert "opto.ipynb" in capsys.readouterr().out
