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

Do **not** invoke the bare `python` / `pytest` on PATH and do **not** rely on
activating the conda `py312` env for tooling — both re-introduce the native-DLL
load fault when the env is not activated.

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


