# Overview

The code is organized into Jupyter notebooks that share common Python modules.

The notebooks are as follows:

- **Behavior**
    - [`behavior.ipynb`](behavior.ipynb)
      Analyses of behavioural data, mice and human.

- **DDM RL-Model** (see [`rlmodel/README.md`](rlmodel/README.md) for the model's
  design and implementation)
    - [`rlmodel/rlmodel_analysis.ipynb`](rlmodel/rlmodel_analysis.ipynb)
      Runs and saves the model-fitting analysis; the manuscript's model figures
      come from here.
    - [`rlmodel/rlmodel_interactive.ipynb`](rlmodel/rlmodel_interactive.ipynb)
      The parameter-tweaking GUI on its own.
    - [`rlmodel/rlmodel_to_behavior.ipynb`](rlmodel/rlmodel_to_behavior.ipynb)
      Generates the schematic figure from the Q + reward-rate model fits.
    - [`rlmodel/rlmodel_neural_correlate.ipynb`](rlmodel/rlmodel_neural_correlate.ipynb)
      Correlates single neurons against the model's latent variables.
    - [`rlmodel/rlmodel_compare.ipynb`](rlmodel/rlmodel_compare.ipynb)
      Per-subject x fitting-criterion comparison grid.
    - [`rlmodel/diagnostics/mle_debug.ipynb`](rlmodel/diagnostics/mle_debug.ipynb),
      [`rlmodel/diagnostics/scale_bound_equivalence.ipynb`](rlmodel/diagnostics/scale_bound_equivalence.ipynb)
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
      repository. **It overwrites what is already in `data/2p/`.**
    - [`2p_0_preprocessing.ipynb`](2p_0_preprocessing.ipynb)
      **Run first.** Selects the neurons active during sampling, computes their
      dF/F, runs the per-neuron choice-selectivity tests and builds the
      max-firing tables. The only 2P notebook that writes data another one
      reads; it draws no published panel.
    - [`2p_1_population.ipynb`](2p_1_population.ipynb)
      Population activity against sampling time, and the single-neuron
      sampling-time correlations (Figures 4D right, 4E, 4K, S8, S10, S11).
    - [`2p_2_sequences.ipynb`](2p_2_sequences.ipynb)
      Neuronal sequences on fast and slow trials (Figures 4F–4J, S9).
    - [`2p_3_choice.ipynb`](2p_3_choice.ipynb)
      Choice selectivity and the choice decoders (Figures 5B–5F, S12E–J).
    - [`2p_4_feedback.ipynb`](2p_4_feedback.ipynb)
      Feedback responses and their link to the next trial (Figures 6A–6E,
      S14A).

- **Movement Tracking**
    - [`tracking.ipynb`](tracking.ipynb)
      SLEAP-based posture and movement analyses.

## Notebook, figures, and what it depends on

Every notebook runs from what is already in `data/` (in git, or fetched by
`data_downloader.ipynb`). "Depends on" names the notebook that *writes* an
input, for when that input is regenerated: `2p_0_preprocessing` is the only
notebook whose output another notebook reads, so it is the one to re-run first.

| notebook | figures | reads | depends on |
|---|---|---|---|
| `behavior.ipynb` | 1C–1I, 2A–2B, S1C–D, S2, S3A–I | `data/behavior/` | — |
| `tracking.ipynb` | S3J–M | `data/tracking/` | — |
| `opto.ipynb` | 3D, 4C, 5G–H, 6D, S6, S7C–D, S13 | `data/opto/` | — |
| `widefield.ipynb` | 3C, 3E–F, 4B (bottom), S5, S7A–B | `data/wf/` | — |
| `2p_0_preprocessing.ipynb` | none (writes `data/2p/`) | `data/2p/` | — |
| `2p_1_population.ipynb` | 4D (right), 4E, 4K, S8, S10, S11A–B | `data/2p/` | `2p_0_preprocessing` |
| `2p_2_sequences.ipynb` | 4F–4J, S9B–E, S9G | `data/2p/` | `2p_0_preprocessing` |
| `2p_3_choice.ipynb` | 5B–F, S12E–J | `data/2p/` | `2p_0_preprocessing` |
| `2p_4_feedback.ipynb` | 6A (bottom), 6B, 6C, 6E, S14A | `data/2p/` | `2p_0_preprocessing` |
| `rlmodel/rlmodel_analysis.ipynb` | 2E–G, S4A–B, S14B | `data/RLModel/`, `data/behavior/` | — |
| `rlmodel/rlmodel_to_behavior.ipynb` | 7D | `data/RLModel/` | — |
| `rlmodel/rlmodel_neural_correlate.ipynb` | 7A–B, S14C–E, S14G–H | `data/RLModel/`, `data/behavior/`, `data/2p/` | — |
| `rlmodel/rlmodel_compare.ipynb` | none (per-subject diagnostic grids) | `data/RLModel/`, `data/behavior/` | — |

`code/run_notebooks.py` runs them in this order. The interactive viewer
(`rlmodel/rlmodel_interactive.ipynb`) and the notebooks in
`rlmodel/diagnostics/` are not figure sources and are not run.

Which notebook produces which manuscript panel is mapped in
[`../docs/manuscript-figure-map.md`](../docs/manuscript-figure-map.md).
Notebook section headings carry the published figure numbers (Figures 1–7,
S1–S14), so a panel can be found by searching a notebook for, e.g., "Figure 4H".

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
| SessionNum  | Session number within a given day. Only one session is typically acquired per day; the number is higher than 1 if a session was restarted | `int`, starting at `1` for each day |
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

## Loading and saving data

Every pickle under `data/` opens with a plain `pd.read_pickle`, on any machine
and with any recent pandas: the files hold only pandas, numpy and built-in
types, never objects defined in this repository.

Two kinds of file store such objects in a portable form and have typed loaders
that rebuild them:

| File | Typed loader | What it returns |
|---|---|---|
| Model fits, `data/RLModel/{chisq,mle}_*.pkl` | `rlmodel.model.fitio.loadFit` | `{subject: payload}` with the model functions resolved from the registry and the fit configuration as an `MLEModelConfig` |
| Two-photon run data, `data/2p/data_runs_*.pkl` | `twop.genrundata.loadRunDataDict` | `{run_idx: RunData}` |

A plain `pd.read_pickle` returns the same payload with those values left as
plain dicts.

To write a new pickle under `data/`, use the guard instead of `pickle.dump`:
`util.portablepickle.savePortable` for any payload, `fitio.saveFit` for a fit,
`genrundata.saveRunDataDict` for run data. The guard walks the object first and
raises `NotPortableError`, naming the offending object and where it sits,
rather than writing a file that only this checkout could read.

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
uv run python code/run_notebooks.py --only widefield --param MAP=standard
```

`widefield.ipynb` runs twice, once per reference map: the redefined MFC/LFC
map (its default) and the standard Allen map, which alone draws 3C, S5D and
S5E. `--param MAP=...` makes both runs use that one map, so it then runs once.

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
`SAVE_FIGS and not PAPER_FIGURES_ONLY`). Executed copies go to
`runs/<timestamp>/` (git-ignored). `code/util/tests/test_run_notebooks.py`
fails the suite if a notebook breaks this contract -- a missing tag, a literal
`save_figs=True`/`False` in a paper cell, or a side flag overriding
`SAVE_FIGS`.

A published panel that is one example out of a loop over every neuron,
session or subject (2E, 2F, S4A, 4F, 5B right, S9C, S14E, S14F, S14H, ...) has its
own `paper-figure` cell drawing just that example, ahead of the `per-subject`
cell that loops over all of them; a paper-only run writes the example and
skips the loop.

## The long ones

| notebook | cost | why |
|---|---|---|
| `rlmodel/rlmodel_compare.ipynb` | **~6 hours** | one comparison grid per subject x model, two forward passes per column. Entirely `per-subject`: a paper-only run skips it in seconds |
| `rlmodel/rlmodel_to_behavior.ipynb` | ~10 min over ten cores | Figure 7D resamples each of the 272 fitted sessions 1,000 times and simulates every trial. Its parameters cell carries `RESAMPLE_COUNT`, `NUDGE_LATENT_SD` and `SURFACE_WORKERS` (each worker needs ~3-4 GiB); `model/qrsurface.py` explains what they do |
| `rlmodel/rlmodel_neural_correlate.ipynb` | minutes, but 1,218 files | one figure per tuned neuron, per parameter. The per-neuron cells are `per-subject`; Figures 7A, 7B and S14C are not |
