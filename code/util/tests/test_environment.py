"""The environment is self-sufficient: everything imports from the lockfile.

The point of a single ``uv`` environment is that nothing needs a second
interpreter and no import arrives by luck. Both are properties that decay
quietly -- a module grows an import nobody declares, and it keeps working
locally because some other package happened to pull it in. These two tests
would have caught the eight dependencies that were undeclared before
``pyproject.toml`` was audited (``matplotlib_venn``, ``cv2``, ``tifffile``,
``requests``, ``ipython``, ``ipywidgets``, ``PIL``, ``dill``).

Notebooks are checked as well as modules, because half of those eight were used
only from notebook cells.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import sys
import warnings

import pytest

#: Vendored third-party trees. Their own example/demo files import the package
#: by its published name rather than through this repo, so they are not part of
#: what this environment has to satisfy.
VENDORED = ("psychofit", "relogit")

#: ``code/figcode/psychofit/Examples.ipynb`` is upstream's demo notebook and
#: imports ``psychofit`` as an installed package. Nothing in the analysis path
#: does -- ``figcode/psychometric.py`` imports it relatively.
NOTEBOOK_IMPORT_EXEMPT = {"psychofit"}


def _repoRoot():
    here = os.path.dirname(os.path.abspath(__file__))       # code/util/tests
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def _moduleNames():
    """Every importable module under ``code/``, as dotted paths."""
    root = _repoRoot()
    package = os.path.basename(root)
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, "code")):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", ".pytest_cache")
                       and d not in VENDORED]
        for filename in sorted(filenames):
            if not filename.endswith(".py") or filename == "__init__.py":
                continue
            rel = os.path.relpath(os.path.join(dirpath, filename), root)
            yield f"{package}.{rel[:-3].replace(os.sep, '.')}"


def _notebookImportRoots():
    """Top-level module of every absolute import in a notebook code cell."""
    root = _repoRoot()
    found = {}
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, "code")):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", ".ipynb_checkpoints")]
        for filename in sorted(filenames):
            if not filename.endswith(".ipynb"):
                continue
            path = os.path.join(dirpath, filename)
            try:
                doc = json.load(open(path, encoding="utf-8"))
            except (ValueError, OSError):
                continue
            for cell in doc.get("cells", []):
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell["source"])
                # IPython magics and shell escapes are not valid Python.
                source = "\n".join(
                    "" if line.lstrip().startswith(("%", "!", "?")) else line
                    for line in source.split("\n"))
                try:
                    with warnings.catch_warnings():
                        # Cell sources carry stray escapes ("\o", "\D") that
                        # ast.parse warns about; irrelevant to the imports.
                        warnings.simplefilter("ignore", SyntaxWarning)
                        tree = ast.parse(source)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            found.setdefault(alias.name.split(".")[0],
                                             set()).add(filename)
                    elif isinstance(node, ast.ImportFrom):
                        # level > 0 is a relative import of this repo's own code
                        if node.level == 0 and node.module:
                            found.setdefault(node.module.split(".")[0],
                                             set()).add(filename)
    return found


@pytest.mark.parametrize("module", sorted(_moduleNames()))
def test_module_imports(module):
    """Every module under code/ imports with only the locked dependencies."""
    sys.path.insert(0, os.path.dirname(_repoRoot()))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            importlib.import_module(module)
    finally:
        sys.path.pop(0)


def test_every_notebook_import_resolves():
    """No notebook depends on a package the environment does not have."""
    stdlib = set(sys.stdlib_module_names)
    missing = {}
    for name, notebooks in _notebookImportRoots().items():
        if name in stdlib or name in NOTEBOOK_IMPORT_EXEMPT:
            continue
        if importlib.util.find_spec(name) is None:
            missing[name] = sorted(notebooks)
    assert not missing, (
        "notebook imports that the environment cannot satisfy -- add them to "
        f"pyproject.toml: {missing}")
