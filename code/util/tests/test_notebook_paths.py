"""No notebook carries a path from the machine it was last run on.

Python prefixes every warning with the absolute path of the file that raised
it, and a notebook saved after a run keeps that text in its outputs. These
tests read every notebook under ``code/`` (sources and saved outputs) and fail
on a user-home path, so such outputs are cleared before they are committed.
"""
from __future__ import annotations

import json
import re

import pytest

from ... import run_notebooks as rn

NOTEBOOKS = sorted(p.relative_to(rn.CODE_DIR).as_posix()
                   for p in rn.CODE_DIR.rglob("*.ipynb")
                   if ".ipynb_checkpoints" not in p.parts)

# Home directories on Windows, macOS and Linux, and the Windows temp folder
# that ipykernel names in warnings raised from cell code.
HOME_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]"
                       r"|AppData[\\/]"
                       r"|(?<![\w.])/(?:home|Users)/[^/\s]+/")


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from _strings(item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)


def _cellText(cell):
    yield from _strings(cell.get("source", ""))
    for output in cell.get("outputs", []):
        yield from _strings({k: v for k, v in output.items()
                             if k not in ("metadata",)})


def test_the_scan_finds_the_notebooks():
    assert len(NOTEBOOKS) >= 15


@pytest.mark.parametrize("text", [
    r"C:\Users\someone\project\code\mod.py:12: FutureWarning: x",
    r"C:\\Users\\someone\\x",
    r"C:\Users\someone\AppData\Local\Temp\ipykernel_1\1.py:3: Warning",
    "/home/someone/project/code/mod.py:12: FutureWarning: x",
    "/Users/someone/project/x.pkl",
])
def test_home_paths_are_recognised(text):
    assert HOME_PATH.search(text)


@pytest.mark.parametrize("text", [
    "data/2p/df.pkl",
    "Loading data/2p/normalized.pkl",
    "results/RLModel/fig.svg",
    "If this checkout sits inside OneDrive (or another cloud-synced folder)",
])
def test_relative_paths_pass(text):
    assert not HOME_PATH.search(text)


@pytest.mark.parametrize("relpath", NOTEBOOKS)
def test_no_home_path_in_sources_or_outputs(relpath):
    nb = json.load(open(rn.CODE_DIR / relpath, encoding="utf-8"))
    hits = []
    for i, cell in enumerate(nb["cells"]):
        for text in _cellText(cell):
            match = HOME_PATH.search(text)
            if match:
                start = max(match.start() - 20, 0)
                hits.append(f"cell {i}: ...{text[start:match.end() + 40]}...")
                break
    assert not hits, ("home-directory paths (clear the cell's outputs, or "
                      "print paths relative to the repository):\n"
                      + "\n".join(hits[:10]))
