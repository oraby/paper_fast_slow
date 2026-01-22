# Overview

The code is organized into Jupyter notebooks that share common Python modules.

The notebooks are as follows:

- **Behavior**
    - [`behavior.ipynb`](behavior.ipynb)
      Analyses of behavioral data.

- **DDM RL-Model**
    - [`rlmodel/model_viewer.ipynb`](rlmodel/model_viewer.ipynb)
      Provides tools to interactively modify model parameters and performs
      analyses that quantify data fit across different model variants.

    - [`rlmodel/model_to_behavior.ipynb`](rlmodel/model_to_behavior.ipynb)
      A short notebook that generates schematic figures based on model fits.

    - *Note: Please see [`rlmodel/README.md`](rlmodel/README.md) for additional
      design and implementation details.*

- **Optogenetics**
    - [`opto.ipynb`](opto.ipynb)
      Analyses of optogenetic perturbations.

- **Wide-Field Imaging**
    - [`widefield.ipynb`](widefield.ipynb)
      Analyses of optogenetic perturbations using wide-field imaging data.

- **2-Photon Imaging**
    - [`TwoPAnalysis.ipynb`](TwoPAnalysis.ipynb)
      Two-photon imaging analyses derived from statistical significance tests.

    - [`TwoPTraces.ipynb`](TwoPTraces.ipynb)
      Two-photon imaging analyses focusing on single-cell and population-level
      heatmaps.

- **Movement Tracking**
    - [`Tracking.ipynb`](Tracking.ipynb)
      Movement tracking analyses.


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

The following conda packages were used to run the analyses:


```
channels:
  - defaults
dependencies:
  - python=3.11
  - ipykernel
  - matplotlib
  - numpy
  - pandas[version='<2.0.0']
  - scikit-learn
  - scipy
  - tqdm
  - ipywidgets
  - jupyterlab_widgets
  - scikit-image
  - seaborn
  - opencv
  - scikit-posthocs
  - nbconvert
  - ipympl
  - h5py
```

The following pip packages were also required:
```
matplotlib-inline         0.1.6
matplotlib-venn           1.1.1
Pillow                    10.0.1
pyddm                     0.8.0
statsmodels               0.14.1
tifffile                  2023.4.12
```

# Missing

This package is currently missing the following:

- A requirements metadata file specifying library dependencies.
- A command-line execution script (e.g., a [`uv`](https://docs.astral.sh/uv/guides/scripts/)
  script combined with [`papermill`](https://papermill.readthedocs.io)) to enable
  batch execution of all notebooks.
    - Command-line arguments would allow generating either only paper figures or
      all figures (including per-subject figures).