# Project agent guide — paper_fast_slow

## Running Python / tests: use `uv` (not bare `python`, not conda)

The dependency environment is managed by **uv** (`pyproject.toml` + `uv.lock`,
virtualenv at `.venv/`). uv's PyPI wheels self-contain their native DLLs, so
there is **no** conda-activation / DLL-search-path problem (the Windows
`0xc06d007f` native fault seen when the conda env is not activated), and runs
are reproducible from the lockfile.

- **Run the test suite** from this repo root (`paper_fast_slow/`):
  ```
  uv run pytest                    # or:  uv run pytest -k <expr> -q
  ```
  (or `powershell -File code\rlmodel\run_tests.ps1` from anywhere.)
- **Run a script / one-off**:
  ```
  uv run python path\to\script.py
  uv run python -c "import numpy as np; ..."
  ```
- **Add a dependency**: put it in `pyproject.toml` `[project].dependencies`, then
  `uv sync`. Do not `pip install` into the venv.

Do **not** invoke the bare `python` / `pytest` on PATH. The conda envs are no
longer used for anything local: every module and every notebook import resolves
from the lockfile, and `code/util/tests/test_environment.py` fails the suite if
that stops being true. (`rlmodel/slurm/` still activates conda on the cluster's
compute nodes — that is the cluster's property, not this repo's.)

**This checkout lives inside OneDrive**, whose cloud-files driver cannot
hardlink. A fresh `uv sync` / `uv run --isolated` fails with
`os error 396 … incompatible hardlinks`. Set `UV_LINK_MODE=copy` (or pass
`--link-mode=copy`) when populating an environment from scratch; an existing
`.venv` is unaffected.

## Import layout (why `uv run pytest` works from the repo root)

Source is imported as `paper_fast_slow.code.rlmodel.*` (deep relative imports;
`paper_fast_slow`, `code`, `rlmodel`, `model`, `tests` each have `__init__.py`).
The `pytest` **console script** (what `uv run pytest` uses) keeps cwd off
`sys.path`, so pytest adds the repo's parent via the `__init__` chain and the
local `code/` package does not shadow the stdlib `code` module. Therefore:

- Run tests from the repo root with `uv run pytest` — no parent-directory dance.
- Avoid `python -m pytest` from here: `-m` prepends cwd to `sys.path`, which puts
  the local `code/` package on the path and re-triggers the stdlib-`code` shadow
  (`module 'code' has no attribute 'InteractiveConsole'`).


# Generally directions

- Don't assume, ask when unsure.

# New notebook creation
The new notebook:
- Should follow the same format of running as a library, setting matplotlib save params, loading data like, as the
other notebooks
- Main code should live in python files, the notebook itself calls the backend code. And the cell code should produce
least side-effects in the global variables, so it should be created as function call if necessary (see other notebooks).
- Reuse existing code when possible. If you want to reuse existing code but it live inside notebook cells, then create
unit tests and move it outside.
- A global flag should control whether to save the figures or not.



# Data files under `data/` — if a pickle fails to load

Every pickle under `data/` was rewritten on **2026-08-31** so that it loads on
any machine, under any pandas version, whatever the checkout is called. The
originals are kept, byte-for-byte, in `data_bak/` (git-ignored, same layout).

**So when a load fails, first check whether the backup loads:**

```
uv run python -c "import pandas as pd; print(pd.read_pickle(r'data_bak/<same/path>.pkl').shape)"
```

- **Backup loads, `data/` copy does not** → the rewrite is at fault. Restore
  that file from `data_bak/` and re-run the conversion for it; see
  `docs/data-portability.md`.
- **Both fail the same way** → not the rewrite. It is the loading code, or the
  environment.
- **Backup fails but `data/` copy works** → expected for many files. The
  backups are the *old* artifacts and most of them cannot be read by a plain
  `pickle.load` at all; that is exactly why they were rewritten.

What the rewrite removed: `pandas.core.indexes.numeric.Int64Index` (deleted in
pandas 2.0), conda-built `string` extension arrays (unreadable by the PyPI
build), the `caiman.…States` column (`caiman` is the author's earlier
`OneDrive/caiman` project, not importable from this repo; nothing here reads the
column, and the notebooks were already discarding it through an inline stub), and
`scipy.io.matlab._mio5_params.mat_struct` (a private path scipy has renamed
once already).

**All 270 open with a plain `pd.read_pickle`**, on any machine, under any
checkout name. The RL-model fits and `data_runs_*.pkl` used to embed
repo-defined objects; they now store the model functions by their registry name
and `MLEModelConfig` / `RunData` as plain dicts.

**Writing a new pickle under `data/`: use the guard, never `pickle.dump`.**

```python
from ..util.portablepickle import savePortable    # any payload
from .fitio import saveFit                        # a {subject: payload} fit
from ..twop.genrundata import saveRunDataDict     # a {run_idx: RunData}
```

`savePortable` walks the object graph first and raises `NotPortableError`
naming the offender and where it sits, rather than writing a file that only
this checkout can read.

To get the typed objects back, read with `fitio.loadFit` or
`genrundata.loadRunDataDict`; a bare `pd.read_pickle` returns the same payload
with those values left as plain dicts. Both work — that is the point.

Do **not** hardcode `paper_fast_slow.` in an import. Scripts needing the deep
package path derive it from the checkout directory (see `_bootstrap()` in
`golden_fig1l.py` / `metrics_runner.py`).

**Checks.** `uv run pytest` exercises the guard on every run
(`code/util/tests/test_saveportable.py`, `code/rlmodel/model/tests/test_fitio.py`).
Before a release, or after unzipping data from the download site, run the
repo-wide audit — it reads ~8 GB, so it is deliberately not in the suite:

```
uv run python code/util/migratepickles.py --repo . --out ./data_portable --audit
```

Tooling lives in [`code/util/`](code/util/); the full account, including the
survey and the procedure, is in [`docs/data-portability.md`](docs/data-portability.md).
