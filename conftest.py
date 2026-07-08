"""Repo-wide pytest setup.

Windows + conda DLL fix
-----------------------
When pytest runs under the conda env's ``python.exe`` *without the env being
"activated"*, the env's native-DLL directories (``Library\\bin`` etc.) are not
on ``PATH``. The first time a test draws with matplotlib (``ax.hist`` → bar →
the Agg/native path) or hits certain numpy paths, the C extension fails to load
with a Windows delay-load fault::

    Windows fatal exception: code 0xc06d007f

which looks like a random "segfault". Adding those directories here — derived
from ``sys.prefix`` so nothing is hardcoded — makes ``pytest`` work whether or
not the env is activated. No-op off Windows or when the directories are absent.

(The user's Jupyter kernel already has these directories on ``PATH``, so
notebooks are unaffected; this only matters for headless pytest / script runs.)
"""
from __future__ import annotations

import os
import sys

# os.add_dll_directory returns handles that must stay alive for the search
# path to remain in effect — keep them for the whole test session.
_DLL_DIR_HANDLES = []


def _add_conda_dll_dirs() -> None:
    if sys.platform != "win32":
        return
    candidates = [
        sys.prefix,
        os.path.join(sys.prefix, "Library", "bin"),
        os.path.join(sys.prefix, "Library", "mingw-w64", "bin"),
        os.path.join(sys.prefix, "Library", "usr", "bin"),
        os.path.join(sys.prefix, "DLLs"),
    ]
    dirs = [d for d in candidates if os.path.isdir(d)]
    # 1) Prepend to PATH (covers delay-load lookups that consult PATH).
    for d in reversed(dirs):
        parts = os.environ.get("PATH", "").split(os.pathsep)
        if d not in parts:
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    # 2) Register as explicit DLL search directories (Python 3.8+ on Windows).
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        for d in dirs:
            try:
                _DLL_DIR_HANDLES.append(add_dll_directory(d))
            except OSError:
                pass


_add_conda_dll_dirs()
