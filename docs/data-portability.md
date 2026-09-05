# Data portability — survey, cause, and the one-off migration

Status: **done and installed.** All 270 pickles under `data/` have been
rebuilt in place; every one now loads with a plain `pd.read_pickle`. The
untouched originals are in `data_bak/` (git-ignored, same layout).

## What was wrong

Every `data/**/*.pkl` was loaded with a plain `pickle.load` — no compat shim,
no stubs, nothing on `sys.path`. That is the test of whether a file will still
open on a bare machine years from now.

| | files | size |
|---|---|---|
| opened plainly | 131 | 1.7 GB |
| **did not** | **139** | **6.6 GB** |

Two independent causes:

| Cause | files | readable in |
|---|---|---|
| `pandas.core.indexes.numeric` — `Int64Index`, removed in pandas 2.0 | 123 | pandas ≥2 **via its compat shim** |
| pandas `string` extension array, conda-built vs PyPI-built | 16 | conda **py312** only |

Plus four foreign classes embedded on top:

| Class | disposition |
|---|---|
| `paper_fast_slow.code.twop.genrundata.RunData` | namedtuple of 5 frames → field-keyed dict |
| `caiman…matreader.states.{States,StartEnd}` | column dropped (below) |
| `scipy.io.matlab._mio5_params.mat_struct` | → plain dict |
| `paper_fast_slow.code.rlmodel.model.*` functions, `MLEModelConfig` | recorded by name, resolved prefix-agnostically |

### `pd.read_pickle` does handle Int64Index

`pandas.compat.pickle_compat._class_locations_map` maps
`("pandas.core.indexes.numeric", "Int64Index")` → `("pandas.core.indexes.base",
"Index")`, and `pd.read_pickle` applies it. A plain `pickle.Unpickler` subclass
does **not** — which is why the first pass here mistakenly concluded that 122
files needed the old conda env.

With the shim applied, only **16** files genuinely resist, and the `wfield`
conda environment is not needed for any of this.

The shim is still worth migrating away from. It is a *private* module whose own
docstring reads "Support pre-0.12 series pickle compatibility" — it exists to
carry pandas-1.x pickles and will not outlive them. The repo already pins
`pandas<3.0.0`, partly for this.

## Risk, and why originals were never touched

| | files | size |
|---|---|---|
| git-tracked (recoverable) | 111 | 0.52 GB |
| **untracked** (irreversible) | **27** | **4.89 GB** |

The untracked set includes all 15 RLModel fit pickles — days of cluster time in
differential-evolution runs — and every large 2P frame.

So: `data/` is read-only throughout. Output goes to `data_portable/`, and the
139 not-already-portable originals were copied to `data_bak/` (6.2 GB, mirroring
the original layout, git-ignored) before anything ran.

## The format

Frames are decomposed into plain numpy arrays plus builtins. Verified across all
270 payloads, the complete set of modules any of them imports is:

```
numpy, builtins, _codecs, datetime, collections, decimal, uuid
```

No pandas. No repo module. No scipy, no caiman. A payload therefore loads on any
machine, under any pandas version, whatever the checkout is called.

Parquet was rejected: the neural frames carry `traces_sets`, a
dict-of-dict-of-ndarray per row (documented in `code/README.md`), which has no
Parquet representation.

### Prefix-agnostic resolution

Two things could not be reduced to plain data — the model's bias/drift/noise
**functions** and the `MLEModelConfig` dataclass, both stored inside the fit
payloads. Rather than let pickle import them (which pins the checkout's name),
they are recorded as `module` + `qualname` and resolved by
`portablepickle._importSuffix`, which walks the recorded path from the right and
imports the longest suffix that resolves inside the running package.

So a pickle written on a machine where the repo was `paper_fast_slow`, or
`code`, or `someones_fork.code`, resolves the same way here. `LegacyUnpickler`
uses the same lookup for reading the old files. Covered by
`test_callable_payload_resolves_under_a_renamed_prefix`.

## The `States` column

Dropped, on three independent checks:

1. one column out of 258 in the frames that carry it;
2. the notebooks' existing inline unpickler already replaced its values with a
   throwaway `Dummy`, so nothing has read real values for as long as that shim
   has existed;
3. the only occurrence of the name `States` anywhere in the repo is inside that
   shim.

`toPortable` records what it dropped in `__pfs_dropped_columns__`, so the
decision is auditable rather than silent.

## Procedure (as run)

```bash
# 0. back up everything not already plainly loadable  ->  data_bak/
#    (classify_and_backup.py, one-off)

# 1a. the 248 readable here
uv run python code/util/migratepickles.py \
    --repo <REPO> --out <REPO>/data_portable --group legacy-index

# 1b. the 22 needing the conda build. Run from the repo's PARENT so
#     `paper_fast_slow.code.util...` imports with package context (which is what
#     lets the loader resolve the embedded function refs) and cwd is not the
#     repo root (so the local `code` package cannot shadow stdlib `code`).
conda run -n py312 --no-capture-output python -c \
  "import pdb,sys; sys.path.insert(0, r'<REPO_PARENT>'); \
   import paper_fast_slow.code.util.migratepickles as m; \
   m.migrate(r'<REPO>', r'<REPO>/data_portable', 'string-dtype')"

# 2. re-encode with the current numpy (stage 1 writes numpy.core.* paths that
#    numpy 2 resolves only through a deprecation shim)
uv run python code/util/migratepickles.py \
    --repo <REPO> --out <REPO>/data_portable --renormalize

# 3. load every payload and compare against the recorded fingerprint
uv run python code/util/migratepickles.py \
    --repo <REPO> --out <REPO>/data_portable --verify
```

Each stage writes `manifest_<group>.json` recording, per file, the foreign
references found, a before/after structural fingerprint, byte sizes and timing.

## A payload is not a drop-in replacement

Renaming `data_portable/x.pkl.portable.pkl` onto `data/x.pkl` does **not**
work, and this is worth stating plainly because it looks like it should.
A payload is a `dict` describing a frame, not the frame:

```python
>>> pd.read_pickle("data/2p/sgf_choice.pkl")     # after a plain rename
{'__pfs_kind__': 'dataframe', 'columns': [...], 'data': [...], ...}
```

There are **73 load sites** in the repo (26 in modules, 47 in notebook cells),
all of them `pd.read_pickle` / `pickle.load`. A rename breaks every one.

So the payloads are not moved. `--install` **rebuilds** each one with
`fromPortable` and writes the reconstructed object back over `data/` as an
ordinary pickle. Callers are untouched, and the rewritten file is written by
the current pandas — so the removed `Int64Index`, the conda-built `string`
arrays, the `caiman` column and the private `mat_struct` path are all gone from
it.

```bash
uv run python code/util/migratepickles.py     --repo <REPO> --out <REPO>/data_portable --install         # dry run
uv run python code/util/migratepickles.py     --repo <REPO> --out <REPO>/data_portable --install --yes   # write
```

`--install` refuses to touch any file that has no counterpart in `data_bak/`.

### The 17 that still name this package

Two groups hold repo-defined objects rather than plain data, so their rebuilt
pickles legitimately reference it — the 15 `data/RLModel/*.pkl` fits (the
model's bias/drift/noise functions and `MLEModelConfig`) and the two
`data/2p/data_runs_*.pkl` (the `RunData` namedtuple):

| file | rebuilt pickle references |
|---|---|
| `2p/sgf_choice.pkl` | builtins, numpy, pandas |
| `behavior/df_human_subjects.pkl` | builtins, datetime, numpy, pandas |
| `2p/data_runs_Unnormalized.pkl` | builtins, datetime, numpy, pandas |
| `RLModel/metrics/metrics_fig1l.pkl` | builtins, numpy, pandas |
| **`RLModel/chisq_Classic_*.pkl`** | builtins, numpy, pandas, **paper_fast_slow** |

That reference is inherent — a fit payload describes a model configuration, so
it needs the model code. It is made prefix-agnostic by reading those files with
`portablepickle.legacyLoad` instead of a bare `pickle.load`; its lookup walks
the recorded module path from the right, so a renamed or relocated checkout
still resolves. Recorded in `CLAUDE.md`.

## Result

- **270/270 rebuilt in place.** `data/` is 8.2 GB; `data_bak/` holds the 7.8 GB
  of originals.
- **270/270 load with a plain `pd.read_pickle`** — the way all 73 load sites in
  the repo actually load them. Zero code changes were needed.
- **253** name only pandas, numpy and the standard library. The other **17** name
  this package, by necessity (above).
- No file anywhere under `data/` still needs the pandas compat shim, the conda
  build, `caiman`, or a private scipy path.
- Test suite: **835 passed, 1 skipped**.

## Portable across checkouts — closed

The 17 files that named this package no longer do. Rather than teach every
reader a special loader, the *objects* were replaced with plain data:

| Was | Now stored as | Rebuilt by |
|---|---|---|
| `biasFn` / `driftFn` / `noiseFn` function objects (12 chisq fits) | their registry key, e.g. `"Classic"` | `fitio.fromStorable` via `DRIFT_FN_DICT` etc. |
| `MLEModelConfig` dataclass (3 mle fits) | `dataclasses.asdict` | `fitio.fromStorable` |
| `RunData` namedtuple (2 `data_runs_*`) | `._asdict()` | `genrundata.loadRunDataDict` |

Nothing outside `fit.py` ever read `fixed_params_vals` — every consumer that
needs the functions already re-derives them from strings (`compare.py:300`
from the parsed filename, `posterior_simulate.py:193` from `model_config`) —
so replacing them was free.

```
uv run python code/util/migratepickles.py --repo . --out ./data_portable --audit
270/270 pickles load with a bare pickle.load and import only
['_codecs', 'builtins', 'collections', 'copyreg', 'datetime', 'decimal',
 'numpy', 'pandas', 'scipy', 'uuid']
```

`LegacyUnpickler` is now used only by the migration tool itself and for reading
`data_bak/`. It is out of the live path.

### Keeping it that way

Three layers, weakest last:

1. **Write-time guard.** `portablepickle.savePortable` walks the payload and
   raises `NotPortableError` — naming the class and its path in the object
   graph — instead of writing. `fit.py`'s three save sites and
   `genrundata.saveRunDataDict` go through it.
2. **Unit tests**, on every `uv run pytest`:
   `util/tests/test_saveportable.py` (15) and
   `rlmodel/model/tests/test_fitio.py` (12).
3. **Repo-wide audit**, `migratepickles --audit`: loads every
   `data/**/*.pkl` with a recording unpickler and checks the import set. It
   reads ~8 GB, so it is a release/post-download check, not a test.

No pre-commit hook, deliberately: 41 of the 270 files are untracked, so a hook
is blind to exactly the large generated ones most likely to go wrong, and
layer 1 catches them at creation whether tracked or not.

The guard earned its place immediately — it caught that `functools.partial`
registry entries (`DriftGain-RewardRate` and friends) do not survive an
identity comparison, because unpickling one builds a *new* partial. Four fits
would have been silently written with an unresolvable name.

## Still open

- **`data_portable/`** (8.4 GB of intermediate payloads) can be deleted once you
  are satisfied — `data_bak/` is the copy that matters. Both are git-ignored.
- **`data_bak/` is 7.8 GB and git-ignored**, so it will not survive a fresh
  clone. Keep it locally until the rewritten `data/` has been committed and
  exercised end-to-end.
- **`data/to_delete/`** (5 files, ~0.5 GB) was rebuilt for completeness but
  should probably just be deleted.
- **The three inline unpicklers** in `TwoPTraces.ipynb`, `2pAnalysis.ipynb` and
  `model_viewer.ipynb` are now dead code — folded into workstream **D**.
- **`conftest.py`'s Windows DLL workaround** exists for the conda envs; once
  nothing reads the originals it can go — workstream **A4**.
