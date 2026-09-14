# Manuscript Methods → code

Section-by-section map of *Materials and Methods* (specifically
*Quantification and Statistical Analysis*) onto the modules that implement it.
Equations are transcribed from the manuscript's OMML source, so this file is
also the authoritative reference for the model when writing model
documentation.

Companion to [`manuscript-figure-map.md`](manuscript-figure-map.md).

---

## Task and dataset constants

Values the code must agree with:

| Quantity | Value | Where it appears in code |
|---|---|---|
| Integration step Δt | 0.005 s | `rlmodel/model/initvals.py::DT` |
| Max simulated trial time T_max | 4.8 s | `rlmodel/model/initvals.py::T_dur` |
| Max stimulus time in the task | 5 s | free-sampling paradigm |
| Minimum sampling time | 0.3 s | free-sampling paradigm; early withdrawals aborted |
| Fixed-sampling duration | 1 s | fixed-time paradigm |
| Early / late inhibition windows | 0–0.35 s / 0.65–1.0 s | `opto/` |
| Mid-sampling inhibition window | 0.3–0.9 s (0.6 s) | `opto/optoreactiontime.py` |
| Feedback inhibition | 3 s | `opto/optofeedback.py` |
| Widefield acquisition | 60 fps (30 per LED), 640×540 after 4× binning | `widefield/` |
| Two-photon acquisition | 30 Hz, 512×512, L2/3 at 150–300 µm | `twop/` |
| Difficulty bands | easy 80–100%, medium 40–60%, hard 0–30% coherence | `behavior/util/assigndvstr.py` (\|DV\| ≥ 0.66 Easy, ≤ 0.33 Hard, else Med) |
| Psychometric bins | mice `[0, 16, 32, 64, 100]`, humans `[0, 1, 5, 10, 20, 50, 100]` | `figcode/psychometric.py` |

---

## Behavioural strategies and exclusions

**Methods — "Behavioural Strategies".** Trials are grouped by difficulty; within
each difficulty the sampling times are split into tertiles: 1st = Fast, 2nd =
Typical, 3rd = Slow.

- `behavior/util/splitdata.py::splitStimulusTimeByQuantile` — the tertile split
  (`quantile_idx` = 1/2/3), used by essentially every downstream analysis.
- `behavior/util/assigndvstr.py::assignDVStr` — the `DVstr` (Easy/Med/Hard) label.

**Methods — "Exclusion criteria".**

| Rule | Implemented in |
|---|---|
| Sampling < 0.3 s (88 trials) excluded | `behavior.ipynb` load cell + `df.valid` |
| Trials still in centre poke after the 5 s cap (131 trials) excluded | as above |
| Mice need > 2,500 trials for model fitting and the explained-variance analysis (n = 9) | `rlmodel/model/fit.py` (`MIN_NUM_TRIALS`), `behavior.ipynb` `groupby("Name").filter(...)` |
| Two mice excluded for behaviour (inconsistent with Figure S2F) | `behavior.ipynb` load cell |
| Human trials > 5 s excluded (0.2% speed, 7.84% accuracy) | `behavior.ipynb` human load cell |
| Animals with < 1,000 trials excluded from the optimal-sampling analysis | `behavior.ipynb` §"Collected as function of sampling time & accuracy" |

**z-scoring convention.** Sampling-time z-scores pool all of a mouse's sessions.
If a subset is selected (e.g. correct trials only, Figure 1G), the z-score is
computed **before** the subset is taken. For humans, analyses that characterise
one experiment type z-score within that type; analyses that compare across types
z-score across both. Implemented in `figcode/util.py::normalizeSTAcrossSubjects`
and `behavior/stdispersion.py`.

---

## Psychometric curves

Fitted with the vendored [psychofit](https://github.com/cortex-lab/psychofit)
library (`figcode/psychofit/`) using the error-function configuration.
Single-sided fits use one lapse rate; two-sided fits (which also characterise
choice bias) use a separate lapse rate per side.

- `figcode/psychometric.py::plotPsych`, `::slowFastPsych`, `::loopPsych`
- Used by Figures 1E, 1F, 2F, S1B/C, S2G/J, S4A, S6B–E, S7C–F.

---

## Statistical testing

| Analysis | Test | Code |
|---|---|---|
| Figure 1I (both), S3I | Paired *t* across subjects, Fast vs Slow | `figcode/prevoutcomecurquantile.py`, `behavior/bias.py`, notebook inline |
| Figure 5C | Shapiro-Wilk, then Student *t* across regions | `twop/sgfneurons.py`, `twop/statstest.py` |
| Figure S2M | Paired *t* + Holm–Bonferroni | `behavior.ipynb` inline |
| Figure S2C | Shapiro-Wilk → one-way ANOVA → Tukey HSD | `behavior/stdispersion.py` |
| Figure S3F | Kruskal-Wallis → Dunn + Holm | `figcode/stheatmap.py` (via `scikit_posthocs`) |
| Figures 3D, 4C, S6G | Hierarchical bootstrap (subject → session → trial), 10,000 iterations, two-tailed sign test, Holm–Bonferroni | `opto/bootstrapping.py::bootstrapPerf`, `opto/bootstrap2regions.py::bootstrapSignTestApproach2` |
| Figures 3F, 6D, S13 | RM-ANOVA + paired *t* with Holm–Bonferroni | `widefield/mfclfcquantiles.py`, `opto/optofeedback.py`, `opto/optoreactiontime.py` (`statsmodels.stats.anova.AnovaRM`) |
| Figures 4J, S9G | Permutation test, 100,000 iterations/session, Holm–Bonferroni | `twop/seqdeviation.py` |
| Figure 7A, S14C/D | Per-neuron activity shuffle, 1,000 draws, one-sided *p*, Holm–Bonferroni | `rlmodel/model/neural_correlate.py` |
| Figure 7B | Session-paired sign-flip permutation, 1,000 iterations, two-sided | `rlmodel/model/neural_correlate.py` |

Normality/variance dispatch (Shapiro-Wilk + Levene → parametric or rank test)
lives in `twop/statstest.py`.

---

## Reward-rate behaviour analyses (Figures 1H, S3B–D)

Running reward rate = proportion of rewarded trials in the previous **five**
trials of the same session (sliding, current trial excluded), so it takes
discrete values in steps of 0.2. Trials without five predecessors are dropped.
Session × bin cells with ≤ 5 trials are dropped. Bins kept only if present in
≥ 75% of the subjects that reach a reward rate of 1.0 — in practice this removes
the 0.0 bin everywhere and the 0.2 bin in the human accuracy task.

- `behavior/rewardrate.py::calcAvgRewardRate` (the 5-trial window),
  `::loopRewardRateAnalysis` (binning, per-subject averaging, Shapiro-Wilk →
  ANOVA/Tukey or Kruskal-Wallis/Dunn-Holm).

---

## Stay/switch update analysis (Figures 1I-right, S3G–H)

Stay = 1 when the current choice direction matches the previous choice.
Because the rewarded-side sequence itself contains repetition structure, raw
stay probabilities are expressed relative to an **error-free stay baseline** —
whether repeating the previous choice would have matched the rewarded direction
on the current trial.

```
update = 100 × (Σ stay − Σ baseline) / N
```

Computed separately for Win trials (previous correct) and Lose trials (previous
incorrect). Figure 1I-right plots the mean of the two **absolute** values per
mouse.

- `behavior/stayswitchupdate.py::calcWinLoseUpdates` — the update formula
  (13 lines, no tests).
- `figcode/stayswitch.py::staySwitchUpdate` — Figure S3H (per-mouse then
  averaged, restricted to the nine mice with > 2,500 trials).
- Figure S3G (800-trial bins along z-scored sampling time, Gaussian σ = 1 bin,
  two mice excluded) is **inline** in `behavior.ipynb`.

Relevant dataframe columns: `Stay`, `StayBaseline`, `PrevOutcomeCount`.

---

## Sampling-time predictors (Figure 2A)

OLS (`statsmodels`) on `log(sampling time)`; fit quality measured by model
log-likelihood. Per animal: fit a null (intercept-only) and a full model, then
refit dropping one predictor at a time and record the percentage loss in
log-likelihood improvement.

Four deliberately low-collinearity predictors:

1. **stimulus difficulty** = `1 − |DV|` (unsigned, so it does not correlate with
   choice side);
2. **outcome-streak count** — signed running count of consecutive identical
   outcomes, reset per session and at trial-numbering discontinuities;
3. **choice side** — binary, chose-left on the current trial;
4. **stay** — binary, current choice repeats the previous one.

No interaction terms. Condition numbers 14.3–22.2; the four contributions sum to
101.9% of the total log-likelihood improvement. Restricted to the nine mice with
≥ 2,500 trials.

**Implementation: inline in `behavior.ipynb`** (`_glmRTFn`, `_simplifyDF`,
`loopSubjects`, `plotVarExplaind`). The dataframe columns used are `DVabs`,
`PrevOutcomeCount`, `ChoiceLeft`, `Stay`.

---

## Tracking (Figures S3J–M)

Posture tracked with [SLEAP](https://sleap.ai/) (four limbs + tail); a body
centroid is computed per frame. Per session an apparatus midline is defined
manually from the head-post; image coordinates are rigidly transformed so the
head-post sits at the origin and the midline is 0°. Missing key points are
linearly interpolated. Per trial, over the sampling epoch, the rotation-angle
distance travelled is the cumulative absolute frame-to-frame change in centroid
polar angle. Mean angle per trial is binned at 5° for the polar histograms.

**Implementation: inline in `Tracking.ipynb`.** No module, no tests.

---

## Optimal sampling-time analysis (Figure 2B)

Per subject, derive an expected reward-rate function of sampling time.
Non-sampling time (side-port → centre-port return, reward collection, or the
enforced timeout) is averaged across sessions separately for correct (τ_C) and
incorrect (τ_I) trials.

Speed–accuracy relation `p(t)`: trials grouped by difficulty, binned at 0.05 s,
bins with < 5 trials dropped, the three difficulties averaged and Gaussian
smoothed, then fitted by non-linear least squares with a chance-level floor:

```
(1)   p(t) = 50 + (50 − λ) · t^β / (t^β + α^β)
```

with bounds α > 0 (scale), β ≥ 1 (shape), 0 ≤ λ ≤ 20 (lapse).

```
(2)   E[T | t] = (τ_C + t)·p(t)/100 + (τ_I + t)·(1 − p(t)/100)

      TPH(t)   = 3600 / E[T | t]

(3)   R(t)     = TPH(t) · p(t)/100

      t* = argmax_t R(t)
```

`R(t)` is min–max normalised within subject for display; the observed sampling
time's mean ± SD is drawn as a horizontal error bar.

**Implementation: [`behavior/optimalsampling.py`](../code/behavior/optimalsampling.py)**
— `perfModel` is eq. (1), `rewardCurve` eqs. (2)–(3), `optimalSamplingTime` the
argmax. Previously inline in `behavior.ipynb` in three duplicated copies, all
now replaced.

### `t*` is not always an interior optimum

Eq. (3) maximises reward per *hour*, not per trial. Because each trial costs
its sampling time **plus** a fixed overhead, sampling longer only pays if the
accuracy it buys outruns the time it costs. When it does not, the argmax sits
at the shortest sampling time on the grid: the model's advice is to guess
immediately and run more trials.

Worked example, both with β = 3, τ_C = 1 s, τ_I = 2 s:

| | | α = 0.3 (evidence accumulates fast) | | α = 3.0 (slowly) | |
|---|---|---|---|---|---|
| **t** | | **accuracy** | **rewards/h** | **accuracy** | **rewards/h** |
| 0.05 s | | 50.2% | 1168 | 50.0% | 1161 |
| 0.50 s | | 91.0% | **2064** | 50.2% | 907 |
| 1.00 s | | 98.7% | 1763 | 51.8% | 751 |
| 3.00 s | | 99.9% | 900 | 75.0% | 635 |
| 5.00 s | | 100.0% | 600 | 91.1% | 539 |
| | | *t\* = 0.54 s* | | *t\* = 0.05 s (the floor)* | |

At α = 0.3, waiting 0.45 s more lifts accuracy from 50% to 91% while the
expected trial barely lengthens — 1.55 s to 1.59 s. It stays almost free
because a correct trial carries the *shorter* overhead (τ_C = 1 s vs
τ_I = 2 s), so being right more often refunds most of the sampling time. Reward
rate nearly doubles.

At α = 3.0, the same 0.45 s buys 0.2 percentage points. The overhead refund
never arrives, the trial stretches from 1.55 s to 1.99 s, and reward rate
falls. Every subsequent second makes it worse, so the best policy is to stop
sampling altogether.

**Near the switch, `t*` is unstable.** It is a global argmax jumping between a
local interior peak and the boundary, so it moves discontinuously:

| α | 1.04 | 1.06 | **1.08** | 1.10 |
|---|---|---|---|---|
| best interior R | 1211 | 1199 | 1187 | 1175 |
| R at the floor | 1192 | 1192 | 1192 | 1192 |
| `t*` | 1.32 s | 1.33 s | **0.01 s** | 0.01 s |

A 2% change in the fitted α swings `t*` from 1.33 s to the floor while the two
policies differ by under 1% in reward rate. Worth knowing before reading `t*`
as a precise quantity from a noisy fit.

**None of this is observed here** — the 17 fitted animals land between 0.53 s
and 1.88 s, all comfortably interior. It is a property of the equation, not a
claim about the mice. It does bear on the paper's framing of impulsivity: under
the paper's own normative objective there are parameter regimes where sampling
briefly *is* the reward-maximising policy rather than a failure of one.

---

## The model

Source: `code/rlmodel/`. Notebook entry points are `model_analysis.ipynb`
(figures), `model_interactive.ipynb` / `model_viewer.ipynb` (GUI),
`model_to_behavior.ipynb` (Figure 7D), `model_neural_correlate.ipynb`
(Figures 7A–B, S14C–H), `model_compare.ipynb` (Figure S14B).

Four models are fitted per mouse: DDM, DDM+QL, DDM+RL, DDM+QL+RL.

### Baseline DDM

```
(14)  Z = 0                                                 [offset starting point]
(25)  x₀ = Z · a/2                                          [initial evidence state]
(36)  x_{t+Δt} = x_t + k·Cohr·Δt + s·ε_t·√Δt
(47)  T_decision = i · Δt
(58)  T_st = T₀ + T_decision      if T₀ + T_decision ≤ T_max
            T_max                 otherwise
```

- `a` — distance between bounds, at ±a/2, a > 0
- `Z ∈ (−1, 1)` — normalised starting point
- `x_t` — accumulated evidence at step t
- `Δt > 0` — time-step increment (0.005 s)
- `Cohr ∈ [−1, 1]` — trial coherence and direction
- `ε_t ~ N(0, 1)` — standard normal noise
- `T₀` — non-decision time (sensory + motor latency)
- `T_max` — maximum simulation time per trial (4.8 s)

Free parameters: `k` (drift scaling, > 0), `s` (noise scaling, > 0).

Code: `model/drift.py::_driftClassic`, `model/noise.py::_noiseNormal`,
`model/bias.py::_biasNone`, stepping in `model/logic.py::simulateDDMTrial`.

### Q-learning + DDM

```
(69)   Q_L^{n=1} = Q_R^{n=1} = 0.5

(710)  Q_{L|R}^n = Q_{L|R}^{n−1} + α·(Reward^{n−1} − Q_{L|R}^{n−1})   if ChoiceDir^{n−1} = L|R
                 = Q_{L|R}^{n−1}                                      otherwise

(11)   q^n = log( clip(Q_L^n, 0.01, 1) / clip(Q_R^n, 0.01, 1) ) ÷ log(100)

(812)  Z^n = clip( δ·q^n + offset, −1, 1 )
```

- `n` — trial number within a session
- `Reward^n ∈ {0, 1}`, `ChoiceDir^n ∈ {0, 1, nil}` (nil = no choice; both action
  values then stay unchanged)
- `q^n ∈ (−1, 1)` — normalised log ratio, floored at 0.01 before the ratio and
  divided by `log(100)` to rescale onto the starting-point range
- `Z^n` — starting point as a fraction of the decision bound

Free parameters: `offset ∈ (−1, 1)` (motor bias), `α ∈ (0, 1)` (Q update rate),
`δ ∈ (0, 1)` (starting-offset scaling).

Code: `model/bias.py::_biasQVal` (registered as `"Q-Val"` with
`Q_VAL_OFFSET = 0` and `"Q-Val (Offset)"` with the offset free);
updates in `model/logic.py::_calcQVal`, `::_updateNextQL_Q`;
state helpers in `model/state_updates.py`.

> **Known divergence:** the MLE path and the χ² path compute the starting point
> differently. The paper states the MLE form without a footnote. Tracked in
> `rlmodel/methods_model_revision.md`; the fix belongs in `model/bias.py`.

### R-learning + DDM

```
(913)   RR^n = RR^{n−1} + β·(Reward^{n−1} − R^{n−1})
(1014)  x_{t+1}^n = x_t^n + k·Cohr^n·Δt + RR^n·s·ε_t·√Δt
```

`RR^n ∈ (0, 1)`, initialised to 0.5 at the start of every session. Free
parameter: `β ∈ (0, 1)` (R-learning update rate).

Reward rate modulates the **noise gain**, not the decision bound — a choice the
paper motivates by the constant motor threshold observed in LFC.

Code: `model/drift.py::_noiseGainRewardRate`, updates in
`model/logic.py::_updateNextRewardRate`.

### Reward-rate channel variants (Figure S4C)

Alternative couplings, one active per fit:

```
(15)  x_{t+1}^n = x_t^n + k·Cohr^n·(2 − RR^n)·Δt + s·ε_t·√Δt     [drift, g(r) = 2−r]
(16)  x_{t+1}^n = x_t^n + k·Cohr^n·(1 + RR^n)·Δt + s·ε_t·√Δt     [drift, g(r) = 1+r]
(17)  b^n = (a/2)·(2 − RR^n)                                     [bound]
```

| Channel | Per-trial effect | `DRIFT_FN_DICT` key | CLI flag |
|---|---|---|---|
| Noise (default) | σ = S·r | `NoiseGain-RewardRate` | — |
| Threshold | b = BOUND·(2 − r) | `Bound-RewardRate` | `--scale-bound` |
| Drift, high r slower | µ = V·DV·(2 − r) | `DriftGain-RewardRate` | `--use-drift-rr --drift-rr-map 2-r` |
| Drift, high r faster | µ = V·DV·(1 + r) | `DriftGain(1+r)-RewardRate` | `--use-drift-rr --drift-rr-map 1+r` |

The bound channel is implemented as an equivalent rescaling of the diffusion
(drift/s_t and noise/s_t with s_t = 2 − r_t), verified in
`rlmodel/scale_bound_equivalence.ipynb` and `tests/test_scale_bound.py`, so the
solver never needs a per-trial bound.

### Q + R-learning + DDM

Except for the drift/bound-scaling variant (eq. 17), the decision boundary is
fixed at `a = 2` (bounds at ±1) so that bound and noise are not fitted
simultaneously; for eq. 17 the noise `s` is fixed at 1 and the bound is free.
This pairing is what `initvals.py`'s `NOISE_SIGMA`/`_NOISE_FIXED` and
`BOUND`/`_BOUND_FIXED` fields encode.

Optimisation: `scipy.optimize.differential_evolution`
(`model/fit.py::simulateDDM`).

### χ² loss

```
(18)  χ² = Σ_groups Σ_bins (n_sim − n_real)² / n_real
```

Four groups: 2 outcomes (correct/incorrect) × 2 choice directions (left/right).
Within each group, observed sampling times are binned at the 0.1, 0.3, 0.5, 0.7,
0.9 quantiles, plus one bin above the fastest observed trial and one final edge
at T_max — **seven bins per group**. The first bin isolates the fastest response
and sharpens the non-decision-time estimate. During fitting the simulation runs
under a fixed seed, so the loss is deterministic for a given parameter set;
correlation figures use 100 different seeds. Only the nine mice with ≥ 2,500
trials are included.

Code: `model/logic.py::calcLoss`.

### MLE loss

No simulation: the first-passage density of the discretised diffusion is
propagated forward (step 0.02 s) from `x₀ = Z^n` at `t = T₀`, absorbing mass at
each bound, until `T_max`. The likelihood of a trial is the single joint density
`p(t^n, c^n)` — the choice selects which boundary's density is read, the
sampling time selects where along it.

Two trial classes the diffusion cannot generate — responses faster than the
non-decision time, and inattentive very slow responses — are handled by mixing
in a uniform contaminant of free rate λ (capped at 0.1), spread across the two
choices and over `(0, T_max]`:

```
(19)  L^n = (1 − λ)·p(t^n, c^n) + λ / (2·T_max)
```

Observed no-choice trials are excluded from the data, so the model is
conditioned on the same event:

```
(20)  L^n = L^n / [ (1 − λ)·(P_upper + P_lower) + λ ]
(21)  −log L(θ) = − Σ_n log L^n(θ)
```

Under the MLE objective the latents `Q_L`, `Q_R` and `RR` are propagated with
the **animal's** observed choices and outcomes (teacher forcing); under χ² they
are propagated with the **model's own** simulated choices and outcomes.

Free parameter: `λ ∈ (0, 0.1)` (contamination / lapse rate).

Code: `model/mle.py`, `model/mle_likelihood.py`, `model/first_passage.py`,
`model/diffusion/` (`single.py`, `vectorized_const_mu.py`,
`vectorized_time_mu.py`), batched path in `model/mle_batch.py`, backend
selection in `model/array_backend.py`.

### Joint MLE + χ² loss

Fitted alone, MLE heavily penalises fast errors toward the non-preferred side
and slow decisions, favouring parameter sets that hold `q` and `RR` effectively
constant — which disables both RL components. The two losses are therefore
combined, each normalised by its own single-objective optimum so that each term
equals 1 at its own best fit:

```
(22)  L_joint(θ) = w_MLE · (−log L(θ) / −log L*) + w_χ² · (χ²(θ) / χ²*)
```

**Figures 7A–B and S14C–E use w_MLE = 1, w_χ² = 0.5.** Figure S14B compares pure
MLE, pure χ², and joint fits at w_χ² = 0.1 and w_χ² = 0.5.

Code: `model/logic.py` (joint objective), `tests/test_joint_loss.py`,
`extract_model_losses.py` (the per-model penalty table behind Figure S14B).

### Figure 2G aggregate metrics

- **Psychometric metric:** refit directional psychometric functions separately
  for fast and slow; per mouse and direction × coherence, take the slow−fast
  difference in left-choice performance for the observed data and for the model,
  then R² between them.
- **Reward-rate metric:** Pearson r between observed and model reward-rate values
  across sampling-time bins.

Code: `model/aggregate.py`, `model/aggregate_plot.py`. Cluster sharding:
`metrics_runner.py` → `model/metrics_shards.py` → `slurm/launch_metrics.py`.

### Figure 7D landscape

Pseudo-sessions built by resampling trials with replacement within session and
randomising stimulus strengths, so the Q and reward-rate latents evolve over
combinations that never occurred. Restricted to simulated correct choices.
Relative Q value `Q_rel` is aligned to the stimulus direction. Binned on a 2-D
grid of `Q_rel ∈ (−1, 1)` × reward rate `∈ (0, 1)`, separately for easy, medium
and hard; each bin holds the mean within-mouse z-scored sampling time, then
smoothed with a Gaussian of σ = 1 bin.

Code: `model/posterior_simulate.py`; the plotting is **inline** in
`model_to_behavior.ipynb` (`plotQ_R_Heatmap`).

---

## Optogenetics data analysis

```
(1123)  Drop = −100 × (Performance_Opto / Performance_Control − 1)
```

Control trials for a session span the first to the last optogenetics trial of
that session's target region.

**Hierarchical bootstrap (Figures 3D, 4C, S6G).** Pooling across mice, 10,000
iterations resampling subjects → sessions → trials with replacement. The
two-tailed *p* is twice the fraction of iterations in which the sign of the
effect differs from the observed effect. Holm–Bonferroni across regions at
family-wise α = 0.05.

- `opto/bootstrapping.py::bootstrapPerf` — Figure 3D. Tested by
  `opto/tests/test_bootstrapping.py`.
- `opto/bootstrap2regions.py::bootstrapSignTestApproach2` — Figures 4C and
  S6G. Tested by `opto/tests/test_bootstrap2regions.py`.
- `opto/permute2regions.py` — an alternative permutation approach, **not
  referenced by any notebook**.

**The two are not the same estimator**, though the Methods describe both the
same way. Both resample subject → session → trial, but `bootstrapPerf` then
*pools every resampled trial* before applying the statistic, so an animal
contributing more trials counts for more; `bootstrapSignTestApproach2`
computes one effect per subject × region × phase and averages those, so every
animal counts once. They agree on balanced data and diverge on unbalanced
data. `bootstrapSignTestApproach2` also handles the cross-region Δ = MFC − LFC
by a *third* scheme — resampling sessions only, and comparing 20 %-trimmed
means of session-level effects — with Holm applied within phase for the
within-region tests and across phases for the cross-region ones.

Reproducibility differs too: `bootstrapSignTestApproach2` takes `seed`
(default 42); `bootstrapPerf` draws from the global `numpy.random` state by
default, and nothing seeds that state before Figure 3D is computed, so the
published p-value is not reproducible bit-for-bit. It takes an optional `rng`
(an `int` replays the same legacy stream as `np.random.seed`). See
`docs/repo-audit.md`.

**Mid-inhibition summary (Figure S13).** Sampling times are z-scored against
each animal's *control* distribution, then a median per condition. RM-ANOVA on
Condition (Control, MFC, LFC), then paired *t* with Holm–Bonferroni. The
S13D regression bins 50 trials and pins the fit through (−1.5 z, 50%).

- `opto/optoreactiontime.py`

**Feedback-epoch inactivation (Figure 6D).** Trials are labelled by whether the
*preceding* trial carried stimulation and whether it was correct. Within each
previous-trial condition, post-correct vs post-incorrect is compared by paired
*t* (Holm-corrected across conditions); across conditions, RM-ANOVA + post-hoc
paired *t*.

- `opto/optofeedback.py::optoFeedback`

---

## Widefield imaging analysis

Adapted from [wfield](https://github.com/jcouto/wfield) with three changes:
**800** SVD components instead of 200; pixels outside the brain zeroed after
registration and before SVD; separate left/right hemisphere fits to absorb
zoom artifacts from camera tilt.

ΔF/F = (F − F0)/F0 per channel/pixel/frame. F0 is the mean frame over rest
periods (inter-trial gaps ≥ 5 s, excluding the first 2 s and last 1 s); if no
such stretch exists in a session, F0 is the per-pixel 0.2 percentile over all
frames of that channel.

Haemodynamic correction: both channels high-passed at 0.1 Hz, the 405 nm channel
additionally low-passed at 14 Hz; each SVT component mean-centred; a per-pixel
regression of 470 on 405 assembled into T:

```
(1224)  SVT_corr = SVT_470 − T × SVT_405
```

then SVT_corr mean-centred again.

**MFC/LFC boundary.** Per session, an averaged sampling-epoch trial is aligned
to the Allen dorsal-cortex map. Across sessions the medial part of secondary
motor cortex is active at sampling onset and the lateral part at the decision
point; the maximum activity inflection at the decision point sets a vertical
cutoff, mirrored across hemispheres, applied uniformly to all sessions.

**z-scored signal (Figures 3C, 3E, 4B-dashed, S5D–E, S6A–B, S7C-left).** Each
region's session-long trace is z-scored, then peri-sampling windows are cut as
100 ms before onset + the sampling epoch + 100 ms after. Only the sampling epoch
is linearly time-normalised, to the group median sampling duration.

**Decision-point comparison (Figure 3F).** Mean activity in the 100 ms window
immediately before movement onset, per strategy and region; the difference
LFC − MFC per session and strategy; RM-ANOVA with Holm–Bonferroni.

**Typical-trial reference set (Figure 3B-left).** Per session, trials between the
0.45 and 0.55 quantiles of sampling duration; the global median of those session
medians; then every trial within ±0.1 s of it.

Code: `widefield/pipelineprocessors.py` (wfield interface, optional import),
`widefield/mfclfcquantiles.py` (Figure 3F, tested), `common/plottracesavg.py`,
`common/_imaging.py`, `common/imaging.py`, `pipeline/`.

---

## Two-photon analysis

### Preprocessing

Suite2p for registration and ROI extraction, followed by manual inspection.
Traces cut 0.1 s before sampling start to 0.1 s after sampling end. A temporary
z-score is applied across the concatenation of all sampling trials; each trial
is re-extracted and its SD computed after 1-frame Gaussian smoothing. A
neuron-trial is **active** when the SD within the sampling epoch exceeds
`3 × (5th percentile of all trial SDs)`. Neurons active in < 5% of trials are
discarded.

> **837 / 2,340 neurons active in MFC (36.44% ± 6.9 SD per session);
> 609 / 1,536 in LFC (38.88% ± 7.51 SD per session).**

ΔF/F0 uses the whole-session trace after Gaussian smoothing (σ = 1 frame,
`scipy.ndimage.gaussian_filter1d` with the default `truncate=4`, i.e. a 9-frame
kernel). F0 is the mode of all frame baselines; a frame's baseline is the
maximum of the minima in the surrounding 60 s window.

Code: `twop/alignsampling.py`, `twop/expanddf.py`, `pipeline/tracesnormalize.py`,
`pipeline/tracesfilter.py`, `pipeline/utils.py::filterNanGaussianConserving`.

### Tuning selectivity

Each trial is reduced to one scalar per neuron (peak activity within the
window); trials are grouped by the variable's levels and compared with a
two-sided Wilcoxon–Mann–Whitney test. "Choice" means the animal's *reported*
choice, not the rewarded port. For multi-variable analyses (Figures 6C, 6E),
each variable is tested independently and combinations are set intersections;
"priors" and "current" groups are set unions.

- Priors = previous choice, previous outcome, previous difficulty (easy/hard);
  identical across epochs.
- Current, Sampling = choice + current difficulty.
- Current, Feedback = choice + current difficulty + current outcome.

Code: `twop/statstest.py`, `twop/runstattest.py`, `twop/fastslowstats.py`,
`twop/sgfneurons.py`, `twop/plottuning.py`.

### Sequence characterisation

A neuron's rank is the **median peak firing position across its active trials**
in time-normalised trials.

**Trial-by-trial rank variability, TRV (Figures S9D–E).** Reference order is
fixed per session from neurons active in > 5% of trials in *both* strategies.
For each trial, the active neurons are listed in peak order but carry their
reference ranks; Spearman's footrule distance to the reference is summed
position by position and divided by the distance the exact reverse would give —
0 when the trial matches the reference, 1 when fully reversed. The session TRV
is the mean over trials.

TRV is calibrated against progressive shuffling using a repeated-insertion
(Mallows) algorithm, with the dispersion parameter tuned so the expected
fraction of inverted pairs matches the target shuffle percentage. Shuffle levels
0%→50% in fine steps (1% up to 10%, 2% to 20%, 5% to 50%), 100 draws per level
per session; linear interpolation converts an observed TRV to an equivalent
shuffle level.

Code: `twop/seqdeviation.py` (+ `twop/tests/test_seqdeviation.py`);
interactive replay in `twop/shuffle_replay.py`, driven by
`2pSeqWithinDeviation.ipynb`.

**Fast-vs-slow rank deviation (Figures 4I–J, S9F–G).**

```
(1325)  deviation = Σ (NeuronRank_Fast − NeuronRank_Slow) / (number of neurons)
```

Ties get the same rank with subsequent indices skipped (1, 2, 2, 2, 5).
Permutation test, 100,000 iterations, the observed *p* projected onto the
resulting z-score distribution, Holm–Bonferroni corrected.

**Single-neuron stretching (Figure 4K, S10).** Traces Gaussian-smoothed with
σ = 2 to suppress Z-axis motion artifacts. AUC is restricted to positive or
sustained deflections: per event, the sum of trace values above
`0.6 × (peak − value at deflection onset)`. Linear regression + Pearson r of AUC
and of peak position against sampling time, over the neuron's active trials;
trials ≥ 3 SD from the mean sampling duration excluded. **Non-rigid = |r| > 0.3.**
Significance assessed against 1,000 within-neuron sampling-time permutations.

Code: **inline** in `plottraces3.ipynb`, with the per-session region comparison
in `twop/plot/corrthreshregions.py`.

### Direction-selective population bins (Figures 5E, S12D–E, I)

Time-normalised sampling split into **7 bin-pairs**: one pair before and one
after sampling start/end, plus 5 equal-time pairs during sampling. Each pair
holds the same set of significantly direction-selective neurons (pooled across a
region's sessions) whose peak-position IQR falls inside the pair boundary, shown
under matched (green) and mismatched (brown) conditions. Neurons whose median
peak falls in the first or last pair are excluded, to avoid predominantly
motor-tuned cells.

Code: `twop/plottuning.py::TrajectoryTuningPlot`, `twop/seqdeviation.py::extractIQR`.

### Choice decoder (Figure S12G)

Logistic regression on significant choice neurons. Features: each neuron's
maximum firing per trial, either at sampling start (0.1 s before to 0.2 s after
onset) or at sampling end (0.2 s before to 0.1 s after offset), after z-scoring
each neuron's sampling-epoch traces. Labels: the animal's direction choice.
1,000 random train/test splits, 30% test.

Code: **inline** in `plottraces3.ipynb`, with `twop/classifyplayground.py` and
the vendored `twop/relogit/` (rare-event logistic regression).

### Coding efficiency (Figure 5F, S12F/H)

```
(1426)  Eff = (Match − Mismatch) / Match

(1527)  ΔEff = √[ ((100·Match − 100·(Match − Mismatch)) / Match² · σ_Match)²
                + ((−100 / Match) · σ_Mismatch)² ]
```

(propagation-of-uncertainty error bars).

Code: `twop/plottuning.py`.

### Population activity and early/late windows (Figures S11A–B)

Population activity is the sum of z-scored activity across time points, over
either all neurons or only significant ones.

- **Early window:** 0.1 s before to 0.3 s after sampling start (0.3 s is the
  task's minimum enforced sampling duration). Trials pooled across animals,
  binned at 0.25 s of sampling duration; linear regression + Pearson r.
- **Late window:** the last 30% of the sampling duration up to 0.1 s after
  sampling end (movement start).
- **Performance version:** easy trials only, binned by early-activity level 0–30%
  in 3% steps; performance expressed relative to the mean over all easy trials.

Code: `twop/plot/activitysum.py`; the binning and regressions are **inline** in
`plottraces3.ipynb`.

### Feedback carry-over (Figure S14A)

Neurons whose feedback-epoch peak significantly correlates (Pearson, *p* < 0.05)
with the next trial's sampling-epoch peak; least-squares regression of the
absolute correlation against the neuron's normalised firing position within
sampling.

Code: **inline** in `2pAnalysis.ipynb`.

### Correlating single neurons with model latents (Figures 7A–B, S14C–E)

The DDM+QL+RL model is evaluated per two-photon subject with the joint
w_MLE = 1 / w_χ² = 0.5 fit, giving per trial `Q_L^n`, `Q_R^n`, `q^n` and `RR^n`.
Each neuron's per-trial activity is the maximum z-scored fluorescence within the
sampling window (0.1 s before start to 0.1 s after movement onset,
time-normalised). Pearson r across trials per neuron × variable; **modulated
when |r| ≥ 0.3**. Because the three Q parameters describe one quantity, the
Q-modulated set is the union across the three. Reported value = per-session
percentage, averaged over sessions, SEM across sessions (22 combined sessions
for Figure 7A–B; 12 MFC and 10 LFC for Figure S14C–D).

Chance level: per-neuron activity permutation, 1,000 shuffles, giving a null of
the session-averaged percentage; 2.5th/97.5th percentiles for the interval and

```
(28)  p = (#(Percentage_shuffle ≥ Percentage_observed) + 1) / (N + 1)
```

For Figure 7B, DV correlations are computed separately within each session's
fast and slow trials, then compared with a two-sided session-paired sign-flip
permutation (1,000 iterations; an independent random sign per session per
iteration). Holm–Bonferroni within each panel.

Code: `rlmodel/model/neural_correlate.py` (3,510 lines,
`tests/test_neural_correlate.py`), driven by `model_neural_correlate.ipynb`.

> Only 4 of the 6 GP4 imaging animals have model fits (GP4-23 and GP4-28 do
> not), so the 22 sessions in Figure 7A–B come from a subset of the imaging
> cohort.

---

## Software stack named in the Methods

Python with NumPy, SciPy, pandas, scikit-learn, Matplotlib, seaborn and
statsmodels. Additionally: Suite2p (2P extraction), SLEAP (tracking), wfield
(widefield SVD), psychofit (psychometric fits), PsychoPy / Psychtoolbox
(stimulus), Bpod (behavioural control).

`scikit_posthocs` (Dunn's test), `matplotlib_venn` (Figures 4G, 6E), `cv2`,
`tifffile` and `requests` are also used but are not named in the Methods and are
missing from `pyproject.toml` — see
[`repo-audit.md`](repo-audit.md#dependency-declaration-gaps).
