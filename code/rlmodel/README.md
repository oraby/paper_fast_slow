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
   - Many prior works use Maximum-Likelihood-Estimation based fitting, which can
     improve fit quality at the cost of increased computational complexity.
     The model is fitted both ways here: with the χ² loss (Figures 2E–G, S4)
     and with MLE, alone or jointly with χ² (Figures 7A–B, S14B–H). See
     [Fitting criteria](#fitting-criteria).
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

    Runs and saves the analysis for the model fitting: Figures 2E–G, S4 and
  S14B.

- [`model_to_behavior.ipynb`](model_to_behavior.ipynb)

    Generates the landscape figure based on the Q-value + R-learning model
    (Figure 7D). See [Landscape figure (Figure 7D)](#landscape-figure-figure-7d)
    below for more details.

- [`model_neural_correlate.ipynb`](model_neural_correlate.ipynb)

    Correlates single-neuron activity with the fitted model's latents
  (Figures 7A–B, S14C–H).

- [`model_compare.ipynb`](model_compare.ipynb)

    Compares the fitting criteria — χ², MLE and joint (Figure S14B).

- [`model_runner.py`](model_runner.py)

    Provides a command-line interface to run the model optimization. It is a
  module of the package, so run it from the repository root:
  `uv run python -m code.rlmodel.model_runner --help`.

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

    Makes a single simulated run for a given subject and model parameters.
  Here you will find the χ² loss function.

  - [`state_updates.py`](model/state_updates.py)

    The Q-value and reward-rate updates, shared by the χ² and MLE paths.

  - [`mle.py`](model/mle.py), [`mle_batch.py`](model/mle_batch.py),
    [`mle_likelihood.py`](model/mle_likelihood.py),
    [`first_passage.py`](model/first_passage.py) and
    [`diffusion/`](model/diffusion/)

    The maximum-likelihood path: the first-passage density of the diffusion,
  the per-trial likelihood, and the batched population objective.

  - [`fitio.py`](model/fitio.py)

    Reads and writes the saved fits (see
  [Saved results](#saved-results---pickle-file-structure)).

  - [`qrsurface.py`](model/qrsurface.py)

    Figure 7D's surface: resamples each fitted subject into pseudo-sessions
  with fresh stimulus strengths, simulates them, and bins z-scored sampling
  time over reward rate × relative Q × difficulty. One subject per worker,
  returning per-facet sums. Its docstring records why the facets average
  per-subject means and why the latents carry a small per-trial nudge — both
  consequences of the fitted learning rates, and both written up for the
  manuscript in [`methods_model_revision.md`](methods_model_revision.md)
  Block 13.

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


## The model

A model is one [bias](model/bias.py) + one [drift](model/drift.py) + one
[noise](model/noise.py) function. The equations below are what the code runs;
the manuscript's own numbering for each of them is in
[`docs/manuscript-methods-map.md`](../../docs/manuscript-methods-map.md#the-model).

### Baseline DDM

Evidence accumulates from a starting point until it reaches one of two absorbing
bounds at $\pm a/2$:

$$x_{t+\Delta t} = x_t + k \cdot Cohr \cdot \Delta t + s \cdot \epsilon_t \cdot \sqrt{\Delta t}, \qquad x_0 = Z \cdot a/2$$

- $Cohr \in [-1, 1]$ — the trial's stimulus coherence and direction (`DV` in the
  dataframe); $Cohr > 0$ means *left* is the correct choice.
- $\epsilon_t \sim N(0, 1)$, $\Delta t = 0.005\,$s (`initvals.DT`).
- $Z \in [-1, 1]$ — the normalised starting point, 0 without Q-learning.
- The sampling time is $T_{st} = T_0 + T_{decision}$, capped at
  $T_{max} = 4.8\,$s (`initvals.T_dur`); a trial that reaches neither bound by
  $T_{max}$ is a no-choice trial.

The non-decision time $T_0$ is not added after the fact: the drift and the noise
are zeroed for the first $T_0/\Delta t$ steps, so the accumulator sits at its
starting point and the crossing index already includes $T_0$.

Free parameters: $k$ (`DRIFT_COEF`), $s$ (`NOISE_SIGMA`), $T_0$
(`NON_DECISION_TIME`).

Code: `drift.py::_driftClassic`, `noise.py::_noiseNormal`,
`bias.py::_biasNone`, stepping in `logic.py::simulateDDMTrial`.

### Q-learning + DDM — trial-history bias

Action values start at $Q_L^1 = Q_R^1 = 0.5$ and only the chosen side updates;
a no-choice trial leaves both alone:

$$Q_{L|R}^{n} = Q_{L|R}^{n-1} + \alpha \cdot (Reward^{n-1} - Q_{L|R}^{n-1}) \quad \text{if } ChoiceDir^{n-1} = L|R$$

The two values become one normalised log ratio, floored at 0.01 so the ratio
cannot blow up, and rescaled onto $[-1, 1]$:

$$q^n = \log\left(\frac{\mathrm{clip}(Q_L^n,\, 0.01,\, 1)}{\mathrm{clip}(Q_R^n,\, 0.01,\, 1)}\right) \Big/ \log(100)$$

which then sets the starting point, together with the subject's constant motor
bias $offset$:

$$Z^n = \mathrm{clip}(\delta \cdot q^n + offset,\, -1,\, 1)$$

Free parameters: $\alpha$ (`ALPHA`), $\delta$ (`BIAS_COEF`), $offset$
(`Q_VAL_OFFSET`).

Code: `state_updates.py::update_q_values`, `::compute_q_value`,
`::compute_starting_point_z`; `bias.py::_biasQVal` for the simulator.

> **Known divergence.** The two fitting paths compose the starting point in a
> different order: MLE computes $\mathrm{clip}(\delta \cdot q + offset, \pm 1)$
> (`state_updates.py`), χ² computes
> $\mathrm{clip}(\mathrm{clip}(q + offset, \pm 1) \cdot \delta, \pm 1)$
> (`bias.py::_biasQVal`). The paper states the MLE form. Recorded in
> [`methods_model_revision.md`](methods_model_revision.md) under "Known code
> issue"; deliberately not fixed here, because fixing it changes the published
> χ² fits — it lands with the next refit.

### R-learning + DDM — reward-rate modulation

A single reward rate, also starting at 0.5 each session, tracks how well the
animal is doing:

$$RR^{n} = RR^{n-1} + \beta \cdot (Reward^{n-1} - RR^{n-1})$$

and scales the diffusion noise, so a well-performing animal accumulates more
noisily and answers sooner:

$$x_{t+\Delta t}^n = x_t^n + k \cdot Cohr^n \cdot \Delta t + RR^n \cdot s \cdot \epsilon_t \cdot \sqrt{\Delta t}$$

Free parameter: $\beta$ (`BETA`).

Code: `state_updates.py::update_reward_rate`,
`drift.py::_noiseGainRewardRate`.

### Q + R-learning + DDM

Both of the above at once — the model behind Figures 2E–G. Nothing new is
added: $Z^n$ comes from the Q-values and the noise gain from the reward rate.

### What counts as a reward

The learning signal is the *simulated* outcome, not the animal's: under χ²
each trial's Q and reward-rate update reads the choice the model just made
(`logic.py::processMultipleSess`). A simulated response faster than
`logic.EWD_TIME` (0.3 s) is treated as unrewarded whatever bound it hit
(`FORCE_EWD`), mirroring the early-withdrawal trials in the data. Under MLE the
latents are propagated with the **animal's** observed choices and outcomes
instead (teacher forcing) — see [Fitting criteria](#fitting-criteria).

### The scale axis: BOUND or NOISE_SIGMA, never both

The bound $a$ and the noise $s$ are near-degenerate — doubling both leaves
behaviour almost unchanged — so exactly one of the pair is fitted and the other
is frozen. By default `NOISE_SIGMA` is fitted and `BOUND` is frozen at 1.0, so
the bounds sit at $\pm 1$ and $Z$ is a fraction of the bound. `--scale-bound`
swaps them: `BOUND` is fitted in $[0.3, 5.0]$, `NOISE_SIGMA` is frozen at 1.0,
and the bias is then read in absolute DDM-state units. `InitVals`' `BOUND` /
`_BOUND_FIXED` and `NOISE_SIGMA` / `_NOISE_FIXED` pairs encode this, and
`fit.simulateDDM` clamps the inactive axis at fit time.

### Fitted parameters

Every parameter the optimiser can fit, with the (min, max) range and initial
value from [`model/initvals.py`](model/initvals.py). A parameter enters a fit
only if the selected model uses it.

| Parameter | Symbol | Range (init) | Fitted when |
|---|---|---|---|
| `DRIFT_COEF` | $k$ | 0 – 20 (1) | always |
| `NOISE_SIGMA` | $s$ | 0 – 5 (1.5) | default scale axis |
| `BOUND` | $a/2$ | 0.3 – 5 (1) | `--scale-bound` only |
| `NON_DECISION_TIME` | $T_0$ | 0 – 1 s (0.3) | always |
| `ALPHA` | $\alpha$ | 0 – 1 (0.3) | Q-learning models |
| `BIAS_COEF` | $\delta$ | 0 – 1 (0.95) | Q-learning models |
| `Q_VAL_OFFSET` | $offset$ | −1 – 1 (0) | Q-learning models |
| `BETA` | $\beta$ | 0 – 1 (0.3) | R-learning models |
| `LAPSE_RATE` | $\lambda$ | 0 – 0.1 (0.02) | MLE / joint fits only |

`--init-val NAME=MIN,MAX[,DEFAULT]` overrides any row for one run.

### Execution sequence

Q-values and Reward-Rate updates are implemented in
[`model/state_updates.py`](model/state_updates.py) (`compute_q_value()`,
`update_q_values()`, `update_reward_rate()`), which the χ² path reaches through
the thin wrappers in [`model/logic.py`](model/logic.py) (`_calcQVal()`,
`_updateNextQL_Q()`, `_updateNextRewardRate()`).

A χ² optimization run is controlled by [`model/logic.py`](model/logic.py) which
calls `simulateDDMMultipleSess()` -> `processMultipleSess()` which calls
`betweenTrialsCb()`, `_calcQVal()`, `_updateNextQL_Q()` and
`_updateNextRewardRate()` if used.

The `betweenTrialsCb()` function calls `simulateDDMTrial()`.

All sessions of a subject step forward together, one trial index at a time, so
the per-trial simulation is one vectorised call across sessions rather than a
Python loop over trials.

### Implemented models

The four models in the paper (registry keys, as passed to `model_runner`, in
brackets):

| Model Name        | Bias Function                        | Drift Function                                   | Noise Function                   |
|-------------------|--------------------------------------|--------------------------------------------------|----------------------------------|
| Classic DDM       | `_biasNone()` (`None_`)              | `_driftClassic()` (`Classic`)                    | `_noiseNormal()` (`Normal(0, 1)`) |
| Q-Learning DDM    | `_biasQVal()` (`Q-Val (Offset)`)     | `_driftClassic()` (`Classic`)                    | `_noiseNormal()` (`Normal(0, 1)`) |
| R-Learning DDM    | `_biasNone()` (`None_`)              | `_noiseGainRewardRate()` (`NoiseGain-RewardRate`) | `_noiseNormal()` (`Normal(0, 1)`) |
| Q+R-Learning DDM  | `_biasQVal()` (`Q-Val (Offset)`)     | `_noiseGainRewardRate()` (`NoiseGain-RewardRate`) | `_noiseNormal()` (`Normal(0, 1)`) |

These four, plus the reward-rate channels below, are the whole registry: the
variants that never reached a figure (a `Decay Q` drift family, a
`Decaying Q-Val` noise, four non-Q bias functions and an asymmetric
learning-rate experiment) were removed on 2026-09-17 and are recoverable from
commit `67ef1ac`.

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

## Fitting criteria

Parameters are optimised with `scipy`'s
[`differential_evolution`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
under one of three criteria, chosen with `model_runner`'s `--fit-mode` and
weight flags. The equations as the paper states them are in
[`docs/manuscript-methods-map.md`](../../docs/manuscript-methods-map.md#the-model).

| Criterion | Flags | Saved as | Used for |
|---|---|---|---|
| χ² | `--fit-mode chisq` | `chisq_*.pkl` | Figures 2E–G, S4 |
| MLE | `--fit-mode mle` | `mle_*.pkl` | Figure S14B (reference for the joint loss) |
| Joint MLE + χ² | `--fit-mode mle --mle-chi2-weight 0.5` | `mle_*_mleW1_chi2W0.5.pkl` | Figures 7A–B, S14B–H |

Figure S14B also compares a joint fit at `--mle-chi2-weight 0.1`.

### χ² loss

We adapted the [Chi-Square Fitting Method](https://pmc.ncbi.nlm.nih.gov/articles/PMC2474747/) to compute the loss between the real and simulated data (see
[`model/logic.py:calcLoss()`](model/logic.py)). The loss is summed across each
condition. Conditions are: choice correctness (correct/incorrect) and choice
direction (left/right). Within each condition, the reaction times are binned
at the quantiles (0.1, 0.3, 0.5, 0.7 and 0.9) of the observed reaction times.
The first bin is split just above the fastest observed trial, so that it holds
exactly that one observation; this helps in detecting non-decision time better.
The last bin ends at the maximum allowed duration (4.8 s), giving seven bins per
condition. Additionally, one more term compares the number of trials where the
subject did not respond within the maximum allowed duration, either due to long
reaction times or no response.
The loss is computed as the sum of the squared differences between the observed
and simulated counts in each bin, normalized by the observed counts.

During fitting, the latents ($Q_L$, $Q_R$, reward rate) are propagated with the
**model's own** simulated choices and outcomes.

### MLE loss

No simulation: the first-passage density of the discretised diffusion is
propagated forward from the starting point, absorbing mass at each bound, until
the maximum duration ([`model/mle.py`](model/mle.py),
[`model/first_passage.py`](model/first_passage.py)). A trial's likelihood is the
joint density of its observed choice and sampling time, mixed with a uniform
contaminant of free rate `LAPSE_RATE` (≤ 0.1) for responses the diffusion cannot
generate. Observed no-choice trials are excluded, and the likelihood is
renormalised to condition on a choice being made.

Unlike χ², the latents are propagated with the **animal's** observed choices and
outcomes (teacher forcing).

`--mle-backend CPU` runs on NumPy; `--mle-backend GPU` runs the same computation
on CuPy and fails rather than falling back when CUDA is unavailable.

### Joint MLE + χ² loss

Fitted alone, MLE favours parameter sets that hold the Q-value and reward rate
effectively constant, which disables both learning components. The joint loss
adds the χ² term back:

$$L = w_{MLE} \cdot \frac{-\log L}{-\log L^*} + w_{\chi^2} \cdot \frac{\chi^2}{\chi^{2*}}$$

where each reference ($-\log L^*$, $\chi^{2*}$) is the subject's own best loss
from the pure MLE and pure χ² fits, so each term is 1 at its own optimum
(`model/fit.py`, `tests/test_joint_loss.py`). Both reference fits must therefore
exist before a joint fit is run.

> Joint fits carry their weights in the filename (`_mleW1_chi2W0.5`); pure MLE
> and χ² fits keep the plain name, which is how the joint fit finds its
> references. `--mle-conditions`, `--mle-choice-weight`, `--mle-rt-weight` and
> `--mle-choice-norm` are **not** in the filename, so re-running with different
> values overwrites the existing fit.

## Data inclusion criteria
- Every subject is fitted (22 in the saved fits), but subjects with fewer than
  2,500 trials across all sessions are excluded from the analyses and figures,
  leaving nine (`MIN_NUM_TRIALS` in [`model/aggregate.py`](model/aggregate.py)).
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
computation. The padding trials repeat the session's last trial and are marked
as invalid (`df['valid'] = False`), so they do not enter the loss; and since
they come after the session's last real trial, the updates they cause cannot
reach any real trial.


# Fit results

## Saved results - pickle file structure

The optimized fitting results for model are saved in the
[`/data/RLModel/`](/data/RLModel/) directory as
`{fit_mode}_{drift}_bias{bias}_{noise}_{t_dur}s_dt{dt}[suffixes].pkl`, e.g.
`chisq_NoiseGain-RewardRate_biasQ-Val (Offset)_Normal(0, 1)_4.8s_dt0.005.pkl`.
Each file is a dictionary of dictionaries. The outer dictionary keys are
subject names.

Every file opens with a plain `pd.read_pickle`, on any machine: the bias, drift
and noise functions are stored by their registry key (e.g. `"Q-Val (Offset)"`),
`OptimRes` as a plain dict, and an MLE fit's `model_config` as a dict.
[`model/fitio.py`](model/fitio.py)'s `loadFit` puts the functions and the config
back; write fits only through its `saveFit`. The mid-run optimisation trace
(`candidate_losses_df`) was stripped from the saved files by
[`model/stripfits.py`](model/stripfits.py).

The inner dictionary of a **χ² fit** has the following structure:

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

An **MLE or joint fit** shares `subject_df`, `dt`, `t_dur`, `include_Q`,
`include_RewardRate`, `params_names`, `params_init` and `OptimRes`, and adds:

- `"mle_df"`: The per-trial latents the fitted model produced ($Q_L$, $Q_R$,
  reward rate, starting point, …). This is what
  [`model/neural_correlate.py`](model/neural_correlate.py) correlates with
  neuronal activity (Figures 7A–B).
- `"model_config"`: The `MLEModelConfig` the fit ran with, as a dict.
- `"loglik"`, `"neg_loglik"`, `"aic"`, `"bic"`, `"n_trials_loss"`: The
  likelihood at the best fit. For a joint fit these cover only the MLE term.
- Joint fits only: `"mle_mle_weight"`, `"mle_chi2_weight"`, `"ref_mle"`,
  `"ref_chi2"`, `"mle_raw_loss"`, `"chi2_raw_loss"`, `"mle_part_loss"`,
  `"chi2_part_loss"`, `"total_loss"` — the breakdown of the joint loss at the
  best fit.


## Results Visualization

Results from model fitting are stored in the
[`/results/RLModel/`](/results/RLModel/). These include:
- [`figs/{subject_name}/`](/results/RLModel/figs/)

  Figures for each subject and model (the example mouse's are Figures 2F and
  S4A). Includes:
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
  - Q-values ($Q_{left}$, $Q_{right}$ and $Q_{val}$, the normalised log
    ratio defined under [Implemented models](#implemented-models)) observed across trials (if Q-learning is
    included in the model) (Row 2, Col 3 lower)
  - Current reaction time as a function of number of previous trial outcomes
    (2 previous incorrect, 1 previous incorrect, 1 previous correct and 2
    previous correct) in real data vs model (Row 2, Col 4)


- [`RewardRate/{subject_name}.svg`](/results/RLModel/RewardRate/)

  Reward rate for the real data and each model for each subject (the example
  mouse's is Figure 2E).

- [`StrategyByPrevCorrect/{subject_name}.svg`](/results/RLModel/StrategyByPrevCorrect/)

  Current trial strategy (Fast/Typical/Slow) as a function of previous trial
  outcome (previous correct/incorrect) for the real data vs each model for
  the given subject.

- [`aggregates_R2_bar.svg`](/results/RLModel/aggregates_R2_bar.svg)

  Subjects' Psychometric $R^2$ and Reward-Rate $r$ Pearson correlation for
  each model as bar plots (Figure 2G). The same comparison for the
  reward-rate channels and the scale-bound variant is in
  `aggregates_R2_drift_rr.svg` and `aggregates_R2_scale_bound.svg` (Figure
  S4B–C), and for the fitting criteria in `aggregates_R2_mle_weights.svg`.

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
# from the project root
uv run python code/rlmodel/slurm/launch_metrics.py --num-evaluations 100 \
    --max-concurrent 200
```

That does everything: builds a work dir per figure, submits the array(s), and
chains a dependent merge job. To run one figure, or to see the `sbatch` commands
first:

(`fig1l` is the preset for Figure 2G; the name predates the current figure
numbering.)

```bash
uv run python code/rlmodel/slurm/launch_metrics.py --figure fig1l --dry-run
```

The three phases are also usable on their own via
[`metrics_runner.py`](metrics_runner.py) — including a plain local run, no
Slurm involved:

```bash
uv run python code/rlmodel/metrics_runner.py --mode prepare --figure fig1l \
    --num-evaluations 100
uv run python code/rlmodel/metrics_runner.py --mode run  --work-dir <dir> --all --num-cpus 8
uv run python code/rlmodel/metrics_runner.py --mode merge --work-dir <dir>
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

# Landscape figure (Figure 7D)

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
