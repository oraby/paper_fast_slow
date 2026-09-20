# Overview

The code is organized into Jupyter notebooks that share common Python modules.

The notebooks are as follows:

- **Behavior**
    - [`behavior.ipynb`](behavior.ipynb)
      Analyses of behavioural data, mice and human.

- **DDM RL-Model** (see [`rlmodel/README.md`](rlmodel/README.md) for the model's
  design and implementation)
    - [`rlmodel/model_analysis.ipynb`](rlmodel/model_analysis.ipynb)
      Runs and saves the model-fitting analysis; the manuscript's model figures
      come from here.
    - [`rlmodel/model_viewer.ipynb`](rlmodel/model_viewer.ipynb)
      Interactively modify model parameters, and quantify data fit across model
      variants.
    - [`rlmodel/model_interactive.ipynb`](rlmodel/model_interactive.ipynb)
      The parameter-tweaking GUI on its own.
    - [`rlmodel/model_to_behavior.ipynb`](rlmodel/model_to_behavior.ipynb)
      Generates the schematic figure from the Q + reward-rate model fits.
    - [`rlmodel/model_neural_correlate.ipynb`](rlmodel/model_neural_correlate.ipynb)
      Correlates single neurons against the model's latent variables.
    - [`rlmodel/model_compare.ipynb`](rlmodel/model_compare.ipynb)
      Per-subject x fitting-criterion comparison grid.
    - [`rlmodel/mle_debug.ipynb`](rlmodel/mle_debug.ipynb),
      [`rlmodel/scale_bound_equivalence.ipynb`](rlmodel/scale_bound_equivalence.ipynb)
      Diagnostics, not figure sources.

- **Optogenetics**
    - [`opto.ipynb`](opto.ipynb)
      Analyses of optogenetic perturbations.

- **Wide-Field Imaging**
    - [`widefield.ipynb`](widefield.ipynb)
      Cortical calcium dynamics and the MFC/LFC segmentation.

- **2-Photon Imaging**
    - [`data_downloader.ipynb`](data_downloader.ipynb)
      Downloads the two-photon dataframes that are too large for the
      repository. **It overwrites what is already in `data/2p/`** -- see
      [`../docs/data-portability.md`](../docs/data-portability.md).
    - [`TwoPLoad.ipynb`](TwoPLoad.ipynb)
      Builds the per-session trace frames the other 2P notebooks consume.
    - [`2pAnalysis.ipynb`](2pAnalysis.ipynb)
      Two-photon analyses derived from statistical significance tests.
    - [`TwoPTraces.ipynb`](TwoPTraces.ipynb)
      Single-cell and population-level heatmaps.
    - [`plottraces3.ipynb`](plottraces3.ipynb)
      Summed population activity by trial duration and quantile, per-session
      choice decoders, and reaction-time/activity correlations.
    - [`2pSeqWithinDeviation.ipynb`](2pSeqWithinDeviation.ipynb)
      Within-strategy trial-to-trial sequence deviation, with an interactive
      replay of the shuffle calibration.

- **Movement Tracking**
    - [`Tracking.ipynb`](Tracking.ipynb)
      SLEAP-based posture and movement analyses.

Which notebook produces which manuscript panel is mapped in
[`../docs/manuscript-figure-map.md`](../docs/manuscript-figure-map.md). Note
that notebook section headings still carry older figure numbering.

# Data Schema

Data are primarily stored in serialized pandas pickle files. The following
sections describe the common behavioral and neural columns found in the
dataframes.

## Behavior Data

| Column Name  | Description | Data-Type & Range |
| -------------: | :-------------: | :-------------: |
|||
| |**__________ Trial Identifier Columns __________**  |
| Name  | Mouse or human subject identifier | `str` |
| Date  | Date on which the session was acquired | `datetime.date` |
| SessionNum  | Session number within a given day. Only one session is typically acquired per day; the number may increase if an incorrect subject was launched or a previous session crashed | `int`, starting at `1` for each day |
| TrialNumber  | Trial number within the session | `int`, starting at `1` for each session |
| TrialStartSysTime  | Unix system time at which the trial started | `float` |
|||
| |**__________ (Previous) Trial Properties Columns __________**  | |
| calcStimulusTime  | Trial sampling time, i.e. duration for which the stimulus was presented | `float`, > 0 |
| (Prev)DV | (Previous) trial **D**ecision **V**ariable, encoding trial direction and difficulty | `float`, ranges from `1` (easiest coherence to the left) to `-1` (easiest coherence to the right) |
| (Prev)DVstr  | Label assigned to the (previous) trial difficulty. See Methods section for assignment details | `str`: `Easy`, `Med`, `Hard` |
| ChoiceCorrect | Whether the subject selected the correct (rewarded) choice | `float`, `1`=correct, `0`=incorrect, `nan`=no choice |
| ChoiceLeft  | Whether the subject chose the left side | `float`, `1`=left, `0`=right, `nan`=no choice |
|||
| |**__________ Other Trial Properties __________**  | |
| quantile_idx  | Sampling-time classification (`Fast`, `Typical`, or `Slow`) for the current trial, given trial difficulty | `int`, `1`=Fast, `2`=Typical, `3`=Slow |
| Stay  | Whether the subject repeated the previous choice (stay vs. switch) | `float`, `1`=stay, `0`=switch, `nan`=not applicable |
| StayBaseline  | Whether current trial's correct decision, as assigned by the task generator, is a stay (i.e. repeated choice) given the subject's previous choice | `float`, `1`=stay, `0`=switch, `nan`=not applicable |
| PrevOutcomeCount  | Number of consecutive correct or incorrect trials preceding the current trial. The counter resets when outcome switches | `int`, positive for correct streaks and negative for incorrect streaks |
| RewardRate  | Ratio of correct trials over the previous five trials. *Computed within the notebooks* | `float`, range `[0, 1]` |

## Optogenetics Data

| Column Name  | Description | Data-Type & Range |
| -------------: | :-------------: | :-------------: |
| (Prev)OptoEnabled  | Whether optogenetic manipulation (laser stimulation) was applied in the previous trial | `float`, `1`=True, `0`=False |
| GUI_OptoBrainRegion  | Brain region targeted by optogenetic manipulation | `int`, value of the `BrainRegion` enum in `common/definitions.py` |
| GUI_OptoStartState(1\|2) | Behavioral state(s) at which the laser onset signal was sent | `int`, value of the `MatrixState` enum |
| GUI_OptoStartDelay | Delay (in seconds) between the start signal and laser emission | `float` |
| GUI_OptoMaxTime | Maximum allowed laser emission duration (seconds) | `float` |
| GUI_OptoEndState(1\|2) | Behavioral state(s) at which laser emission was terminated | `int`, value of the `MatrixState` enum |

## Neural Data

Unlike behavioral dataframes, where each row corresponds to a single trial,
neural dataframes (both 2-photon and wide-field) represent each trial across
multiple rows, with each row corresponding to a specific trial epoch.

Neural traces are stored as a dictionary of dictionaries. In the most common
case, the outer dictionary contains a single key, `neuronal`, whose value is an
inner dictionary.

The inner dictionary keys correspond to trace identifiers. For wide-field data,
these identifiers are brain-region names (e.g., `MFC_left`, `MFC_right`, or
`MFC_Bi`). For 2-photon imaging, they are integer identifiers assigned to each
neuron during preprocessing. Each value is a NumPy array of `float`s, where each
index corresponds to the ΔF/F value at a given frame index (starting at `0`).
All traces within a session therefore share the same length.

Using synchronization signals, frames from the acquisition session are aligned
to behavioral epochs via `trace_start_idx` and `trace_end_idx`. The
dictionary-of-dictionaries (`traces_sets`) is assigned to each epoch row.

Although assigning mutable objects to dataframe rows is generally discouraged:

- The dictionary is a shared mutable object; modifying it in one row affects
  all rows.
- Vectorized pandas and NumPy operations cannot be directly applied.

This approach nevertheless offers several advantages:
- Behavioral epochs are event-driven and vary in duration; we are not aware of
  an alternative representation without substantial drawbacks that supports
  variable-length epochs.
- Each row retains access to the full session data, simplifying operations such
  as trimming traces relative to epoch boundaries.
- Because the dictionary is shared, memory usage is reduced; only object
  references are stored per row. Serialization to disk, however, requires
  explicit handling.
- When a row performs an in-place manipulation, it creates a new dictionary and
  becomes the sole owner. The `sole_owner` flag indicates whether the underlying
  `traces_sets` is shared (`False`) or unique (`True`).

| Column Name  | Description | Data-Type & Range |
| -------------: | :-------------: | :-------------: |
| epoch  | Name of the underlying behavioral epoch | `str` |
| traces_sets  | Dictionary of dictionaries holding neural traces | `dict` of `dict` |
| sole_owner | Whether `traces_sets` is uniquely owned by the row | `bool` |
| trace_start_idx  | Frame index at which the epoch begins | `int` |
| trace_end_idx  | Frame index at which the epoch ends (inclusive; use `trace_end_idx + 1`) | `int` |
| acq_sampling_rate | Frame acquisition rate (Hz) | `float` |
|||
|||
| epochs_names | Names of concatenated epochs, used primarily for x-axis labeling | `list` of `str` |
| epochs_ranges | Index ranges for each entry in `epochs_names` | `list` of `(start_idx, end_idx)` tuples |

# Dependencies

The environment is managed by [`uv`](https://docs.astral.sh/uv/) from
[`pyproject.toml`](../pyproject.toml) and `uv.lock` at the repository root.
There is nothing to install by hand and no conda environment: `uv` creates
`.venv/` from the lockfile, and its PyPI wheels carry their own native
libraries.

```
uv sync                  # create/refresh .venv from the lockfile
uv run pytest            # run the suite (from the repository root)
uv run python <script>   # run anything else
uv run jupyter lab       # notebooks, against the same environment
```

If this checkout sits inside OneDrive (or another cloud-synced folder), a
fresh `uv sync` fails to hardlink out of the cache with `os error 396`. Set
`UV_LINK_MODE=copy` for that first sync; an existing `.venv` is unaffected.

To add a dependency, put it in `pyproject.toml` under `[project].dependencies`
and run `uv sync`. Do not `pip install` into the virtualenv -- the lockfile is
what makes a run reproducible.

`code/util/tests/test_environment.py` keeps this honest: it imports every
module under `code/` and checks that every absolute import in every notebook
cell resolves from the locked environment alone. A dependency that is used but
not declared fails the suite rather than working by accident on one machine.

## Optional: GPU

The MLE likelihood has a CuPy backend
(`rlmodel/model/array_backend.resolve_array_backend`). CuPy is imported lazily
and callers fall back to NumPy when it is absent, so it is **not** declared as a
dependency -- the correct wheel name depends on your CUDA version
(`cupy-cuda12x` and so on). Install it yourself if you want the GPU path;
nothing else changes.

# Running the notebooks

[`run_notebooks.py`](run_notebooks.py) executes the figure notebooks with
[`papermill`](https://papermill.readthedocs.io), from the repository root:

```
uv run python code/run_notebooks.py --list                   # what would run
uv run python code/run_notebooks.py                          # everything, writes nothing
uv run python code/run_notebooks.py --save-figs --paper-figures-only
uv run python code/run_notebooks.py --save-figs --only behavior opto
uv run python code/run_notebooks.py --only widefield --param MFC_LFC_MAP=False
```

Every figure notebook has one cell tagged `parameters` declaring three flags,
all `False` by default, so a plain run -- or opening the notebook and running
all cells -- writes nothing:

| flag | runner option | effect |
|---|---|---|
| `SAVE_FIGS` | `--save-figs` | write figures under `results/` |
| `SAVE_DATA` | `--save-data` | rewrite cached intermediate data under `data/` |
| `PAPER_FIGURES_ONLY` | `--paper-figures-only` | skip the per-subject / per-session figures |

Each cell that saves a figure is tagged `paper-figure` (saves whenever
`SAVE_FIGS` is on) or `per-subject` (saves only under
`SAVE_FIGS and not PAPER_FIGURES_ONLY`). Where a paper panel is one example
picked from a loop over every session or neuron, the whole loop stays
`paper-figure`, so a paper-only run can still write more than the manuscript
shows. Executed copies go to `runs/<timestamp>/` (git-ignored).
`code/util/tests/test_run_notebooks.py` fails the suite if a notebook breaks
this contract -- a missing tag, a literal `save_figs=True`/`False` in a paper
cell, or a side flag overriding `SAVE_FIGS`.

`rlmodel/model_to_behavior.ipynb` is the long one: Figure 7D resamples each of
the 272 fitted sessions 1,000 times and simulates every trial, ~10 minutes over
ten cores. Its parameters cell carries the three knobs -- `RESAMPLE_COUNT`,
`NUDGE_LATENT_SD` and `SURFACE_WORKERS` (each worker needs ~3-4 GiB) -- and
`model/qrsurface.py` explains what they do and why the last two exist.
