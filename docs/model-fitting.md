# RL-DDM model fitting: design rationale and validation

This file is kept for reference. It records why the RL-DDM fitting is built the
way it is and how it is checked. It is not a user guide. The code and
[`code/rlmodel/README.md`](../code/rlmodel/README.md) are authoritative. The
README carries the model equations, the parameter table, the CLI flags and the
saved-fit layout, and this file does not repeat them. For the equations as the
published Methods number them, see
[`manuscript-methods-map.md`](manuscript-methods-map.md#the-model).

All paths below are relative to `code/rlmodel/`.

---

## 1. What is fitted, and under which criterion

Every model (one bias, one drift and one noise function, as listed in the
README's *Implemented models*) is fitted to each mouse separately with
`scipy.optimize.differential_evolution`. The fit can use one of three
objectives.

| Criterion | What it compares | Latent history | Used for |
|---|---|---|---|
| χ² | Simulated vs observed sampling-time counts, in quantile bins per outcome × direction cell | the model's own simulated choices and outcomes | Figures 2E–G, S4 |
| MLE | The joint density of each observed (choice, sampling time) pair | the animal's observed choices and outcomes (teacher forcing) | the reference term of the joint fit; Figure S14B |
| Joint | w_MLE · (−log L / −log L\*) + w_χ² · (χ² / χ²\*) | both of the above, each in its own term | Figures 7A–B, S14C–D (w_MLE = 1, w_χ² = 0.5) |

**Why two criteria constrain different things.** χ² is generative: it runs the
whole session forward, letting the model's own choices drive its Q-values and
reward rate. It then scores the shape of the sampling-time distribution in each
cell. It cannot tell when within a session a fast or slow trial happened. MLE is
trial-by-trial: it conditions the Q/RR recursion on what the animal actually did,
and it scores the timing of every individual response. The two objectives
therefore condition the learning recursion on different histories. That is why
they can disagree, and why the joint objective is not redundant.

**Why a joint criterion.** Fitted alone, the likelihood penalises fast errors
toward the non-preferred side so heavily that it favours parameter sets in which
the learning rates hold Q and RR effectively constant, which disables both
reinforcement-learning components. Adding the χ² term back keeps the generative
behaviour (reward-rate dynamics, fast/slow psychometrics) in the objective.

**Why normalise each term by its own optimum.** The raw losses are on unrelated
scales: a negative log-likelihood in the thousands of nats against a χ² sum of
squared count residuals. Dividing each by the subject's standalone-best loss
(the pure-MLE and pure-χ² fits of the same model) makes each term equal 1 at its
own optimum. The weights then express a like-for-like trade-off, so
w_χ² = 0.5 means half as much weight on the χ² term's relative excess as on the
MLE term's. A joint fit therefore needs both reference fits to exist first
(`fit.py::_load_reference_loss`). Figure S14B compares pure χ², pure MLE, and
joint fits at w_χ² = 0.1 and 0.5 on the behavioural metrics of Figure 2G. The
reported fits use w_χ² = 0.5.

**χ² specifics worth knowing.** The simulator runs under a fixed seed
(`logic.py::makeOneRun`, `seed=0`), so the χ² loss is deterministic for a given
parameter vector. Without the seed, differential evolution would be comparing
noisy evaluations. Difficulties are pooled inside each outcome × direction cell,
so difficulty constrains the fit through the drift (k · Cohr) rather than
through the binning. The extra bin edge just above the fastest observed trial
isolates that trial, which sharpens the non-decision-time estimate.

---

## 2. How the likelihood is computed

### First-passage density by probability-mass propagation

The likelihood does not simulate trials or smooth densities. For each trial the
probability distribution of the accumulator is propagated forward on a fixed
grid, and the mass that crosses each bound is absorbed at every step.

- **State grid.** Bin centres at ±dx/2, ±3dx/2, … inside the bounds, with
  dx = 0.02. At bounds ±1 this gives 100 bins. The starting point
  x₀ = Z · bound is snapped to the nearest bin centre.
- **Time grid.** Δt = 0.005 s up to T_max = 4.8 s, the same values the
  simulator uses (`initvals.DT`, `initvals.T_dur`).
- **Transition.** One step maps a bin to N(x + μΔt, σ²Δt). Mass moving to each
  interior bin is an exact difference of Gaussian CDFs over that bin's edges.
  The mass beyond ±bound is the CDF tail. Because the kernel is built from CDF
  differences rather than sampled densities, the absorbed mass plus the
  surviving mass sums to one up to floating-point error.
- **Reading the likelihood.** The decision time is the observed sampling time
  minus T₀. The trial's density is the absorbed mass at the observed bound in
  the bin that contains that decision time, divided by Δt. Choice and sampling
  time are therefore one joint density, not two separate terms.
- **Noise scaling.** The step noise is σ·√Δt, in both the simulator and the
  likelihood. With this scaling the accumulated variance grows with elapsed
  time, and the fitted σ does not depend on the integration step.

Three solvers implement this (`diffusion/`, dispatched by `first_passage.py`).
`single.py` is the slow reference. It keeps the full p(x, t) for plotting
(`mle_visualize.py`, `mle_debug.ipynb`) and is the oracle the equivalence tests
compare against. `vectorized_const_mu.py` and `vectorized_time_mu.py` are fast
single-trial paths. The fitting path is the batched solver in `mle_batch.py`
(section 4). It assumes one drift per trial, which covers every model in the
registry, and it refuses a time-varying drift rather than mis-handling one.

### Trials, padding and the latent recursion

- Q-values and reward rate are updated on every real trial, including invalid
  ones, from the animal's observed choice and outcome (`state_updates.py`,
  shared with the simulator). Only valid trials with a recorded choice enter
  the loss.
- Sessions are padded to equal length so that the recursion can step all
  candidates × sessions together, one trial position at a time. Padding trials
  are invalid and come after a session's last real trial, so they can never
  reach the loss.
- A trial whose inputs make the solver ill-posed (non-finite or non-positive
  σ, or a starting point outside the bounds) is assigned the floor
  likelihood 10⁻³⁰⁰. A candidate is never discarded as NaN, so differential
  evolution always gets a finite, ranked loss.

---

## 3. Key numerical choices, and why

### Lapse (contamination) mixture

Some responses cannot come from the diffusion at all: anticipations faster than
T₀, where the DDM density is exactly zero, and inattentive responses at any
latency. Under a pure diffusion likelihood each such trial sits at the floor
(log 10⁻³⁰⁰ ≈ −691 nats), and a handful of them dominate the fit. The density
is therefore mixed with a uniform contaminant spread over the two choices and
(0, T_max]: L = (1 − λ)·p_DDM + λ / (2·T_max).

- λ (`LAPSE_RATE`) is a fitted parameter in [0, 0.1] with an initial value of
  0.02. The upper limit of 0.1 is the usual psychophysics bound. It is fitted
  only under MLE and joint fits; χ² never sees it.
- Choice trials with sampling time ≤ T₀ go through the mixture with p_DDM = 0
  and score log(λ / 2T_max), not the floor. As a result T₀ is set by the bulk of
  the data. The optimiser is not forced to keep T₀ below the fastest outlier.
- The mixture belongs to the likelihood only. Posterior simulation from a
  fitted parameter set does not reproduce lapses.
- To fit without a lapse term, pass `--init-val LAPSE_RATE=0,0,0`.

### Conditioning on a response (conditional normalisation)

On every trial the diffusion puts some probability on reaching neither bound by
T_max. The retained data contain no such trials: observed no-choice trials and
trials beyond T_max are invalid. The data are therefore conditioned on a
response having occurred, and the model is conditioned on the same event. Each
trial's likelihood is divided by (1 − λ)(P_upper + P_lower) + λ, where P_upper
and P_lower are the probabilities of reaching each bound by T_max
(`mle.py::_apply_choice_rt_weights`, `choice_norm="conditional"`, which is the
CLI default). Without this step the likelihood of the retained trials would not
be a proper density.

The term is not a constant. It depends on θ through coherence, Z and σ, so it
moves the optimum. At a good fit it is worth only a few nats. It matters most
during the search, where small-σ candidates rarely respond within T_max. The
trade-off is that conditioning scores the shape of the choice/RT distribution
among responders and discards the response rate. `--mle-choice-norm marginal`
keeps the response rate instead.

The same helper can split the joint log-density into log P(choice) +
log p(RT | choice) and weight the two parts (`--mle-choice-weight`,
`--mle-rt-weight`). It can also re-balance trials across conditions
(`--mle-conditions`). Every saved MLE and joint fit uses unit weights and no
condition balancing, so its objective is the conditioned joint density.

### Terminal-C redistribution of residual mass

At T_max some mass is still inside the bounds. The batched solver can split that
mass by position with a threshold C · bound. Mass above +C·bound is counted as
an upper choice, mass below −C·bound as a lower choice, and the band in between
as genuine no-decision mass. The no-decision mass is the likelihood of an
observed no-choice trial. With C = 0 nearly all residual mass becomes a choice.
With C = 1 (the default, `initvals.MLE_TERMINAL_C`) the no-decision mass equals
the full survival mass. Every saved fit uses C = 1. Observed no-choice trials
are excluded from the loss anyway, so C does not change the reported fits. The
option remains for fitting data that include no-choice trials.

### Bound vs noise: one scale axis, and the bound channel as a rescaling

The bound a and the noise s are nearly degenerate: scaling both together leaves
behaviour almost unchanged. Fitting both would leave the optimiser on a ridge,
so exactly one of the two is free. By default `NOISE_SIGMA` is free and `BOUND`
is fixed at 1, so Z is a fraction of the bound. `--scale-bound` frees `BOUND`
instead, fixes `NOISE_SIGMA` at 1, and reads the bias in absolute units.

The reward-rate *bound* channel needs a bound that changes per trial,
b_n = BOUND · (2 − RR_n). The solver never handles that directly. It uses the
diffusion rescaling identity: a process with bound b is observationally
identical to one with bound 1 when μ, σ and the starting point are all divided
by the same factor. The factor is applied per trial
(s_n = 2 − RR_n: μ/s_n, σ/s_n, z/s_n) and the solver keeps a scalar bound. This
was checked against a reference path that does vary the bound per trial.
Rescaling only μ or only σ does not reproduce it. Rescaling all three does, to
within discretisation noise (`scale_bound_equivalence.ipynb`;
`tests/test_bound_rewardrate.py`). The drift channel is not a rescaling of
anything, so it is implemented directly as a per-trial drift gain.

### Starting point: the two paths

The likelihood computes Z = clip(δ·q + offset, ±1) (`state_updates.py`). The
χ² simulator computes clip(clip(q + offset, ±1)·δ, ±1) (`bias.py::_biasQVal`).
The saved χ² fits and the χ² term of the joint fits use the simulator form.

---

## 4. Optimisation and batching

| | χ² | MLE and joint |
|---|---|---|
| Population | `popsize=100` per parameter | sized from the memory budget, at least 64 candidates per generation (`mle.estimate_population_settings`, `--mle-min-population`) |
| Mutation | (0.5, 1.5) | SciPy default |
| Updating / polish | SciPy default, final local polish | `updating="deferred"`, `vectorized=True`, no polish |
| Parallelism | candidates across CPU processes | a single process that evaluates the whole generation in one call, on NumPy or CuPy (`--mle-backend CPU|GPU`) |

The MLE path is organised so that one differential-evolution generation costs
one solver call.

- **Whole-population evaluation.** With `vectorized=True` and deferred
  updating, SciPy hands over the full population at once. The Q/RR recursion
  then runs over candidates × sessions × trial positions together
  (`mle.objective_from_population`), and every trial of every candidate goes
  through the batched solver in one pass. Polishing is off because a local
  polish evaluates one candidate at a time.
- **Grouping by (μ, σ).** Trials that share a drift and noise share a
  transition kernel. The kernels and their FFTs are computed once per group, in
  one batched call, and gathered per trial. Each time step is then a single
  batched FFT convolution, whatever the number of groups. FFT convolution
  avoids a dense (trials × bins × bins) transition tensor.
- **Gather only what is read.** Each trial needs its density at one time bin.
  The solver writes that one value inside the time loop and keeps no
  (trials × time) density tensor. The tensor would need about 14 GB on a
  typical population.
- **Data stays on the device.** The per-session observations are moved to the
  array backend once and broadcast, not re-tiled every generation. The GPU
  backend fails instead of silently falling back to the CPU, so a "GPU" fit is
  never a slow CPU fit by mistake.
- **One process for MLE.** Several worker processes would compete for one GPU.
  The parallelism comes from the batch, not from processes.

Both paths end in the same model code: χ² and MLE share `state_updates.py`, and
the joint objective evaluates the same parameter vector through both
(`fit.py::_jointVectorizedObjectiveWrapper`).

---

## 5. How the fits are validated

**Posterior predictive checks.** A fit is judged by what it generates. The
fitted parameters are run through the simulator, with simulated history
(`posterior_simulate.py`), and compared with the animal on the behavioural
metrics of Figure 2G: the psychometric R² of the slow − fast difference, and
the reward-rate Pearson r across sampling-time bins. Each comparison is averaged
over 100 seeds (`aggregate.py`). The same comparison of fitting criteria is
Figure S14B, and of reward-rate channels and scale axes, Figure S4B–C. The MLE
term deliberately uses observed history; the predictive check deliberately uses
simulated history.

**Numerical and structural properties, pinned by tests** (`model/tests/`):

| Property | Test file |
|---|---|
| Noise increment has variance σ²Δt (√Δt scaling) | `test_noise_scaling.py` |
| Q and RR update equations, initial values 0.5, only the chosen side updates | `test_state_updates.py` |
| MLE updates latents from the observed, not the modelled, choice | `test_mle_teacher_forcing.py` |
| Upper + lower absorbed mass + survival = 1 | `test_first_passage_mass_conservation.py` |
| Fast solvers match the reference solver | `test_diffusion_equivalence.py` |
| Batched likelihood matches the trial-by-trial path; (μ, σ) grouping; memory-based population sizing | `test_mle_batched.py` |
| Population objective equals the per-candidate objective, also with padded sessions; invalid candidates give a finite penalty; DE runs end to end | `test_mle_smoke.py` |
| Lapse mixture: λ = 0 reproduces the pure DDM; outliers sit at log(λ/2T_max); negligible effect where the DDM density dominates | `test_mle_lapse_rate.py` |
| Conditional normalisation and the choice/RT split, including the lapse-exact choice masses | `test_mle_choice_rt_weight.py` |
| Condition-balanced weights; batched vs trial-by-trial parity | `test_mle_conditions.py` |
| Terminal-C partitions the survival mass; C = 1 equals survival; choice trials do not depend on C | `test_mle_terminal_c.py` |
| Joint loss: each term equals 1 at its reference; weights; zero-weight terms skipped | `test_joint_loss.py` |
| Bound channel by rescaling equals a per-trial bound | `test_bound_rewardrate.py` |
| Scale-axis swap (`--scale-bound`) and absolute-unit bias | `test_scale_bound.py` |
| Drift channel | `test_drift_rewardrate.py` |
| Posterior simulation produces χ²-compatible output from a saved fit | `test_posterior_simulation.py` |
| Saved fits round-trip without repo-defined objects | `test_fitio.py` |
| Figure 2G metrics: sharded cluster collection equals the serial run | `test_aggregate.py`, `test_metrics_shards.py` |

The suite does not include a parameter-recovery study on synthetic data, and the
fits do not claim full parameter identifiability (see the README's *What it
isn't*).

---

## 6. Significance testing for the neuron ↔ model-latent correlations (Figures 7A, 7B, S14C–D)

Code: `model/neural_correlate.py::plot_factor_bars_fastslow_dv`, driven by
`rlmodel_neural_correlate.ipynb`. Tests: `tests/test_neural_correlate.py`.

**What is correlated.** The latents come from the joint fit (w_MLE = 1,
w_χ² = 0.5) of the Q + R model. They are the per-trial values stored with the
fit, so they are the model's state on entering each trial, propagated with the
animal's own choices and outcomes: Q_L, Q_R, q and RR. The trial's signed
evidence (DV) is taken from the trial record. DV is a per-trial task quantity,
not a decision variable read out at a time point. Each neuron contributes one
value per trial: its maximum z-scored activity over the time-normalised
sampling window. Invalid trials are dropped.

**Tuning criterion.** For each neuron and variable, the Pearson r across trials
is computed within the session. A neuron counts as modulated when |r| ≥ 0.3. Q
is one quantity seen three ways, so a neuron is Q-modulated if it passes for any
of Q_L, Q_R or q, and it is counted once. Neurons with an undefined r (fewer
than three trials, or constant activity or latent) are left out of both the
numerator and the denominator. The statistic is the per-session percentage,
averaged across sessions, with SEM across sessions.

**Against chance (every bar).** The null is a per-neuron activity shuffle. Each
neuron's trial activity is permuted, with the latents held in place, and the
tuning decision is re-made. The shuffle is aggregated exactly like the observed
data, so the null is a null of the same session-averaged percentage. Permuting
the activity keeps its variance, so only the covariance term changes, and all
1,000 shuffles of a neuron come from one permutation matrix. The p-value is
one-sided: p = (#{shuffle ≥ observed} + 1) / (N + 1). It is never exactly zero.
The 2.5th–97.5th percentiles of the null give the chance band drawn on the bars.

**Fast vs slow (Figure 7B).** DV tuning is recomputed within the fast and slow
sampling-time terciles. Each subset bar is tested against its own shuffle,
because a third of the trials raises the chance floor. The fast − slow
difference uses a two-sided, session-paired sign-flip permutation
(1,000 iterations). The test pairs each session's fast and slow percentages,
takes the mean paired difference, and builds the null by flipping the sign of
each session's difference independently. Under the null the labels within a
session are exchangeable, and swapping them negates that session's difference,
so sign flipping is the matching permutation for paired data. It also keeps
every test in the figure distribution-free.

**Multiple comparisons.** The vs-chance p-values are Holm–Bonferroni corrected
across the four bars of a figure (Q, RR, DV-fast, DV-slow). Figure 7A pools
both regions. Figures S14C and S14D draw MFC and LFC as separate figures on a
shared y-axis. The fast-vs-slow test is Holm-corrected across regions. Stars
mean *p < 0.05, **p < 0.01, ***p < 0.001. These figures make no MFC-vs-LFC
comparison. A region-difference test exists in the module, a hierarchical
bootstrap over region → session → neuron (10,000 draws, two-sided sign-change
p, as in `opto/bootstrap2regions.py`). It is used only by the separate
fast/slow summary-bar view (`plot_fast_slow_bars`).
