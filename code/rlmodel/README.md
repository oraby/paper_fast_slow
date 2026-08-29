# Overview

This document summarizes the design decisions and implementation rationale
underlying the models used in this work.

# Models Scope

## Objective:
- Identify a model that explains the substantial variability in sampling time.
- Favor implementations that are straightforward and interpretable.
    - Each model variant reflects a distinct behavioral phenotype
        - Q-learning captures choice bias
        - R-learning captures reward rate
    - Use a minimal and consistent parameterization
        - No separate parameters for rewarded vs. non-rewarded outcomes
        - No separate learning rates for left vs. right states
        - Non-visited Q-states, i.e. $Q^n_L$ and $Q^n_R$, are not updated
          (i.e., they do not decay toward a baseline value)
        - Assumes a linear mapping between stimulus coherence and drift rate
          (rather than a psychometric function such as an error function)
    - Use *textbook* formulations.
        - Exception: Q-learning includes an additional parameter, $offset$.
        - $offset$ represents a subject-specific, constant motor or cognitive
          bias. The $offset$ is additive and independent of trial history,
          unlike Q-value–dependent bias.

**What it isn't:**
- An attempt to maximize goodness of fit.
   - Several previous studies (e.g. [Gupta et al.](https://www.nature.com/articles/s41467-024-44880-5),
   [DePasquale et al.](https://elifesciences.org/articles/84955),
   [Shinn et al.](https://elifesciences.org/articles/56938)) have achieved strong model fits.
   - Many prior works use Max-Likelihood–Estimation based fitting, which can
     improve fit quality at the cost of increased computational complexity. We
     are open to adopting this approach if reviewers find it more appropriate.
   - Increasing model complexity (e.g., more parameters) can improve fit quality,
     but at the cost of interpretability.
- A claim of novelty in combining Q-learning or R-learning individually with
  DDM.
    - Previous work on DDM Q-learning:
    - Previous work on DDM R-learning:
    - **Exception**: To the best of our knowledge, this is the first work to
      combine both approaches within a single framework.
- An effort to reduce the number of parameters to the absolute minimum.
    - For example, in the combined Q- and R-learning model, $RR^n$ could be
      inferred from $Q^n_L$ and $Q^n_R$.
        - This would complicate direct model comparison.
        - This approach may not generalize to settings with more than two
          choices beyond $Q_L$ and $Q_R$.
- An exploration of alternative DDM variants.
  - Extensions such as collapsing decision boundaries, or step-wise
    drift-rate modulation, are not considered.
  - Although such variants may yield improved fits, our interpretation does not
    justify favoring one formulation over another. We therefore adopt the
    standard DDM.
  - We are open to including alternative variants if reviewers believe they
    would improve interpretability.
  - Given the overlap in how several parameters influence choice bias and
    reaction times, we do not claim full parameter identifiability.

## What is the value of this work?

- Demonstrates limitations in capturing behavioral phenomena using a single RL
  model alone, or without RL-based modeling.
- Given the strong influence of trial history reported in our paper—specifically
  reward rate and bias—these models provide a principled framework for capturing
  these behavioral phenotypes.
- The adapted formulations are intended as generalizations that future work can
  tailor to specific experimental contexts and findings.

# Implementation Details

## Repository structure

The code is organized as follows:

- [`model_interactive.ipynb`](model_interactive.ipynb)

    A tool that allows for real-time tweaking of the models parameters.
  Inspired by [PyDDM](https://github.com/mwshinn/PyDDM) GUI implementation.

- [`model_analysis.ipynb`](model_analysis.ipynb)

    Runs and saves the analysis for the model fitting.
  Here you will find the paper’s figures

- [`model_to_behavior.ipynb`](model_to_behavior.ipynb)

    Generates schematic-like figure based on Q-value + R-learning model.
    Check [# Schematic-like figure (Fig. 5f, middle)](#schematic-like-figure-fig-5f-middle) section below for more details.

- [`model_runner.py`](model_runner.py)

    Provides a command-line interface to run the model optimization. Run
  `python model_runner.py --help` for more information.

- [`metrics_runner.py`](metrics_runner.py)

    Command-line interface for collecting
  [`model_analysis.ipynb`](model_analysis.ipynb)'s repeat evaluations on a
  cluster instead of in the notebook. See
  [Collecting the aggregate metrics on a cluster](#collecting-the-aggregate-metrics-on-a-cluster).

- [`model/`](model/)

  Contains the implementation of the model logic and functions:

  - [`fit.py`](model/fit.py)

    Runs parameters optimization for a given subject and model and saves the
  results to the disk.

  - [`logic.py`](model/logic.py)

    Makes a single run for a given subject and model parameters. Here
  you will find the loss function.

  - [`bias.py`](model/bias.py), [`drift.py`](model/drift.py) and
    [`noise.py`](model/noise.py)

    Contains the different implementations for the DDM components, including
  taking Q-values and Reward rate into account.

  - [`initvals.py`](model/initvals.py)

    Contains the initial values and the permitted ranges for the different
    models parameters.

  - [`visualize.py`](model/visualize.py)

    Contains the implementation for the GUI components for the interactive
    [`model_interactive.ipynb`](model_interactive.ipynb) notebook.

  - [`plotter.py`](model/plotter.py)

    Contains plot functions for the model results for both the GUI
    interactive notebook and the analysis notebook.

- [`/data/RLModel/`](/data/RLModel/)

    Contains serialized `pkl` files of the optimized parameter fitting for the
  different models for the different subjects.


## Models

A model is defined by the combining one of each of [bias](model/bias.py),
[drift](model/drift.py) and [noise](model/noise.py) functions.


### Execution sequence

Q-values and Reward-Rate updates are implemented in the
[`model/logic.py`](model/logic.py) file (`_calcQVal()`, `_updateNextQL_Q()`, `_updateNextRewardRate()`).

An optimization run is controlled by [`model/logic.py`](model/logic.py) which
calls `simulateDDMMultipleSess()` -> `processMultipleSess()` which calls
`betweenTrialsCb()`, `_calcQVal()`, `_updateNextQL_Q()` and
`_updateNextRewardRate()` if used.

The `betweenTrialsCb()` function calls `simulateDDMTrial()`.

### Initial values

If Q-values are included in the model, the initial trial Q-values are set to 0.5 for
both left and right choices.

If Reward-Rate is included in the model, the initial trial Reward-Rate is set to 0.5.

### Implemented models

The following table of models were implemented:

| Model Name        | Bias Function  | Drift Function           | Noise Function      |
|-------------------|----------------|--------------------------|---------------------|
| Classic DDM       | `_biasNone()`  | `_driftClassic()`        | `_noiseNormal()`    |
| Q-Learning DDM    | `_biasQVal()`  | `_driftClassic()`        | `_noiseNormal()`    |
| R-Learning DDM    | `_biasNone()`  | `_noiseGainRewardRate()` | `_noiseNormal()`    |
| Q+R-Learning DDM  | `_biasQVal()`  | `_noiseGainRewardRate()` | `_noiseNormal()`    |

#### Reward-rate channels

R-learning models differ in *which* quantity the learned reward rate `r`
modulates. Exactly one channel is active per fit; all three are implemented as
drift functions (the noise and bound variants use the equivalent rescaling of
the diffusion, so the solver never needs a per-trial bound):

| Channel | Per-trial effect | Drift function | Selected by |
|---|---|---|---|
| Noise (default) | `σ = S·r` | `_noiseGainRewardRate()` | — |
| Threshold | `b = BOUND·(2 − r)` | `_boundGainRewardRate()` | `--scale-bound` |
| Drift | `μ = V·DV·g(r)` | `_driftGainRewardRate()` | `--use-drift-rr` |

The drift channel's step is `d += DV·V·g(r)·dt + S·√dt·ε` with `σ` and the
bound both flat. `g(r)` comes from `--drift-rr-map`: `2-r` (default,
`d += DV·(2V − r·V)`) or `1+r` (`d += DV·(V + r·V)`). Note the two mappings
run opposite ways — under `2-r` a *high* reward rate weakens the drift, which
is the opposite speed direction from the noise and threshold channels.

`--use-drift-rr` overrides the noise/threshold channel and requires an
R-learning `--drift`; `--scale-bound` then only selects which of
(`BOUND`, `NOISE_SIGMA`) is the fitted scale axis. In `model_interactive` the
same choice is the **RR as Drift** checkbox plus the **RR-Drift Map** dropdown.

The user-facing `--drift RewardRate*` name resolves to the matching internal
`DRIFT_FN_DICT` key (`NoiseGain-` / `Bound-` / `DriftGain-` /
`DriftGain(1+r)-`) via `drift.resolve_drift_alias`, and that key is part of the
saved-fit filename, so the channels never overwrite each other.

## Loss function

We adapted the [Chi-Square Fitting Method](https://pmc.ncbi.nlm.nih.gov/articles/PMC2474747/) to compute the loss between the real and simulated data (see
[`model/logic.py:calcLoss()`](model/logic.py)). The loss is summed across each
condition. Conditions are: choice correctness (correct/incorrect) and choice
direction (left/right). Within each condition, the reaction times are binned
into quantiles (0.1, 0.3, 0.5, 0.7 and 0.9), an additional bin below the 0.1
quantile that contains exactly one observation is preappended. This helps
in detecting non-decision time better. Additionally, one more bin is created for
trials where the subject did not respond within the maximum allowed duration
(3s), either due to long reaction times or no response.
The loss is computed as the sum of the squared differences between the observed
and expected counts in each bin, normalized by the expected counts.

## Data inclusion criteria
- Subjects with fewer than 2,500 trials across all sessions are excluded.
- Trials that are not included in the behavioral analysis (e.g.
  optogenetic trials, trials with no response, etc.) are kept but marked as
  invalid in the dataframe (`df['valid'] = False`); as Q-value and reward rate
  updates occur on every trial, valid and invalid trials are both used for
  updating the Q-values and reward rates, but only valid trials are used for
  computing the loss.

## Data preprocessing

For each subject, the longest session length is determined. All other sessions
are padded with trials to match this length. This ensures that all sessions
have the same number of trials, which is necessary for a faster vectorized
computation. The padding trials are marked as invalid (`df['valid'] = False`) so
that they do not affect the Q-value and reward rate updates or the loss
computation.


# Fit results

## Saved results - pickle file structure

The optimized fitting results for model are saved in the
[`/data/rlmodel/`](/data/rlmodel/) directory as `{model_combination}.pkl`.
Each file is a dictionary of dictionaries. The outer dictionary keys are
subject names. The inner dictionary has the following structure:

- `"subject_df"`: The subject’s behavioral dataframe before running the simulation.
- `"dt"`: DDM time step.
- `"t_dur"`: Maximum allowed trial duration for the DDM.
- `"include_Q"`: Whether the model uses Q-values and therefore performs Q-value updates between trials.
- `"include_RewardRate"`: Whether the model uses a reward rate and therefore performs reward-rate updates between trials.
- `"is_loss_no_dir"`: Flag whether loss function should ignore choice direction. In the current work, choice direction is always considered.

The following fields are primarily used by the optimization function, and some are redundant with fields defined above:

- `"fixed_params_names"`: Names of parameters that were fixed during optimization. Some are used purely for coding abstraction across different models. These include:
  - `"df"`: The subject behavioral dataframe.
  - `"biasFn"`, `"driftFn"`, `"noiseFn"`: The bias, drift, and noise functions used.
  - `"biasFn_df_cols"`, `"driftFn_df_cols"`, `"noiseFn_df_cols"`: Dataframe columns passed as keyword arguments to the corresponding functions.
  - `"biasFn_kwargs"`, `"driftFn_kwargs"`, `"noiseFn_kwargs"`: Keyword arguments for the corresponding functions.
  - `"include_Q"`, `"include_RewardRate"`, `"dt"`, `"t_dur"`, `"is_loss_no_dir"`: As defined above.
  - `"return_df"`: Whether to return the full simulation dataframe, including choices and reaction times.

- `"fixed_params_vals"`: Values corresponding to the parameters listed in `fixed_params_names`.

- `"params_names"`: Names of the parameters that were optimized. For example, a classical DDM would include:
  - `["NON_DECISION_TIME", "BOUND", "DRIFT_COEF", "NOISE_SIGMA"]`

- `"params_init"`: Initial values of the parameters that were optimized.

- `"OptimRes"`: [Optimization result](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html)
  as returned by `scipy`’s
  [`differential_evolution`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html).
  This includes multiple fields, notably:
  - `"x"`: Best-fit parameters found, corresponding to `params_names`.
  - `"fun"`: Value of the loss function at the best-fit parameters.


## Results Visualization

Results from model fitting are stored in the
[`/results/RLModel/`](/results/RLModel/). These include:
- [`figs/{subject_name}/`](/results/RLModel/figs/)

  Figures for each subject and model. Includes:
  - Reaction time distributions for real data vs model based on:
    - correct and incorrect trials (Row 1, Col 1)
    - left and right choices (Row 1, Col 2)
  - Psychometric curves for real data vs model based strategy (Fast/Slow)
    (Row 1, Col 3)
  - Current trial strategy (Fast/Typical/Slow) as a function of previous trial
    outcome (previous correct/incorrect), similar to `StrategyByPrevCorrect`
    below (Row 1, Col 4 upper)
  - Win-stay and lose-switch behavior observed across trials (Row 1,
    Col 4 lower)
  - Reaction time distributions for real data vs model based on correct and
    incorrect trials by difficulty (Easy/Med/Hard) (Row 2, Col 1)
  - Reward rates observed across trials (if R-learning is included in the model)
    (Row 2, Col 2 upper)
  - Motor bias observed across sessions for real data vs model (Row 2,
    Col 2 lower)
  - Starting-point bias observed across model trials transformed as a function
    of correct/incorrect choices (Row 2, Col 3 upper).
  - Q-values ($Q_{left}$, $Q_{right}$ and $Q_{val}$, i.e.
    $log(\frac{Q_{left}}{Q_{right}})$) observed across trials (if Q-learning is
    included in the model) (Row 2, Col 3 lower)
  - Current reaction time as a function of number of previous trial outcomes
    (2 previous incorrect, 1 previous incorrect, 1 previous correct and 2
    previous correct) in real data vs model (Row 2, Col 4)


- [`RewardRate/{subject_name}.svg`](/results/RLModel/RewardRate/)

  Reward rate for the real data and each model for each subject.

- [`StrategyByPrevCorrect/{subject_name}.svg`](/results/RLModel/StrategyByPrevCorrect/)

  Current trial strategy (Fast/Typical/Slow) as a function of previous trial
  outcome (previous correct/incorrect) for the real data vs each model for
  the given subject.

- [`aggregates_R2_bar.svg`](/results/RLModel/aggregates_R2_bar.svg)

  Subjects' Psychometric $R^2$ and Reward-Rate $r$ Pearson correlation for
  each model as bar plots (Fig. 1l).

- [`aggregates_R2_subj_color.svg`](/results/RLModel/aggregates_R2_subj_color.svg)

  Subjects' metrics correlation for each model as scatter plots a distinct
  color assigned for each subject.

- [`Model_{model_name}_Across Subjects_by_fixed_subj_color.svg](/results/RLModel/)

  An expanded correlation plot for each model showing each subject's
  individual metrics. Each subject is assigned a distinct color across the same
  model and different models.


## Collecting the aggregate metrics on a cluster

The bar figures above are averaged over `NUM_EVALUATIONS` repeat simulations per
(model × subject) — the DDM forward pass is stochastic, so iteration *i* is
seed *i*. One evaluation costs ~2.5–12 s (mostly the psychometric fit's 20
Nelder-Mead multi-starts), so at `NUM_EVALUATIONS = 100` collecting all four
figure presets in the notebook takes many hours.

[`metrics_shards.py`](model/metrics_shards.py) runs the same work as thousands
of independent single-CPU Slurm tasks — one per (model, subject, iteration)
combination — each writing **its own** result file, so nothing is shared and
nothing can race. A merge step then writes the very pickle
[`model_analysis.ipynb`](model_analysis.ipynb) already reads, so the notebook
just cache-hits.

```bash
# from the project root, in the conda env (the fit pickles need it)
python code/rlmodel/slurm/launch_metrics.py --num-evaluations 100 \
    --max-concurrent 200
```

That does everything: builds a work dir per figure, submits the array(s), and
chains a dependent merge job. To run one figure, or to see the `sbatch` commands
first:

```bash
python code/rlmodel/slurm/launch_metrics.py --figure fig1l --dry-run
```

The three phases are also usable on their own via
[`metrics_runner.py`](metrics_runner.py) — including a plain local run, no
Slurm involved:

```bash
python code/rlmodel/metrics_runner.py --mode prepare --figure fig1l \
    --num-evaluations 100
python code/rlmodel/metrics_runner.py --mode run  --work-dir <dir> --all --num-cpus 8
python code/rlmodel/metrics_runner.py --mode merge --work-dir <dir>
```

Notes:

- **`prepare` is what makes a task cheap.** It resolves the fits and slices the
  behavior dataframe once, so a task loads two small files instead of
  re-preparing the behavior frame and unpickling a 29–339 MB fit file for a few
  floats.
- **Results are reproducible.** Each task seeds the global RNG from its own
  (model, subject, iteration) coordinates, so two full runs of the pipeline
  produce identical frames — which a serial notebook collection does *not*,
  because the psychometric multi-starts draw from an unseeded global.
- **Resubmitting is cheap.** A finished combination already has its file and is
  skipped, so re-running the whole array only redoes what is missing. `merge`
  prints a ready-to-paste `--array=` spec for the combinations that failed;
  feed it back with `--skip-prepare --array <spec>`.
- **`conda: command not found` in a job log** means the compute node can't see
  conda — `conda` is a shell *function* from an interactive rc file and doesn't
  exist in a batch job. The launcher detects the conda root from your shell and
  exports it, and [`activate_conda.sh`](slurm/activate_conda.sh) falls back to
  `CONDA_EXE` and the usual install dirs. If it still can't find it, pass
  `--conda-base $HOME/miniconda3`.
- **If `sbatch` starts rejecting submissions**, the cluster caps how many jobs
  you may have *queued* (`sacctmgr show assoc user=$USER
  format=user,maxjobs,maxsubmitjobs`). `--max-concurrent` won't help — it only
  throttles what *runs*. Use `--items-per-task K` instead: each array task then
  walks `K` consecutive combinations, so 15,900 of them at `K=20` is 795 jobs
  rather than 15,900. Same total work, `K`× longer per job, so raise the
  `--time` in [`metrics.sbatch`](slurm/metrics.sbatch) to match.
- **After a cluster run**, copy `data/RLModel/metrics/*.pkl` back. Cells 18 /
  26 / 28 / 30 then print `Loaded …metrics_<name>.pkl` and simulate nothing.
  The `simcache_<name>.pkl` sidecar restores the seed-0 per-subject frames, so
  the per-subject fit panels work off a cache hit too.

# Schematic-like figure (Fig. 5f, middle)

![Schematic-like figure](/results/RLModel/Q_R_Heatmap.svg)

This figure is generated using the
[`rlmodel/model_to_behavior.ipynb`](model_to_behavior.ipynb) notebook.

For each difficulty (colored plane), the figure plots the expected sampling
time, i.e. reaction time, on the z-axis as a function of the relative Q-value
(x-axis) and reward rate (y-axis). The plot is generated only for correct choice
trials.

The relative Q-value is starting point Q-value bias transformed as a function of
matching (+ve value) or mismatching (-ve value) the correct choice of the
current trials.

As the existing data is not sufficient to cover the entire 2D space of
relative Q-value and reward rate, we resample each session 10,000 times to
generate synthetic data covering the entire space. Within each session,
valid trials are resampled with replacement and each trial is randomly assigned
a different trial difficulty (DV) from a uniform distribution between -1 and 1.

The sampling time is the result of simulating the DDM with the given model
parameters for the subject with the resampled data. The sampling time is
z-scored within each subject to account for inter-subject variability. A final
Gaussian filter is applied to smooth the transitions across neighboring bins.
