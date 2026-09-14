# Manuscript → code: panel-by-panel figure map

Every panel of *Cortical mechanisms of fast versus slow decision making*
(Neuron revision), mapped to the notebook, backend module and `results/`
artifact that produces it, together with the numbers the manuscript reports for
that panel.

Read the [conventions and numbering-drift note](README.md#numbering-drift)
first — notebook headings use older figure numbers.

**Legend for the "Backend" column**

- `module::function` — code lives in a Python module (importable, testable).
- **inline** — code lives only in notebook cells. These are the refactor targets;
  they are collected in [`repo-audit.md`](repo-audit.md#inline-notebook-code-that-produces-published-panels).
- *not from this repo* — schematic / render / photograph / micrograph.
- *(inferred)* — matched by panel content plus output filename; the code carries
  no explicit figure label.

---

## Datasets behind the figures

| Dataset | File | Driving notebooks |
|---|---|---|
| Mouse behaviour (all paradigms) | `data/behavior/df_behavior.pkl` (82,389 × 22) | `behavior.ipynb`, `opto.ipynb`, `rlmodel/*` |
| Human psychophysics | `data/behavior/df_human_subjects.pkl` (28,109 × 18) | `behavior.ipynb` |
| Chronometry | `data/behavior/df_chrono.pkl` | `behavior.ipynb` |
| Freely-moving vs head-fixed light-chasing | `data/behavior/lc_hf_combined_df.pkl` | `behavior.ipynb` |
| Optogenetics | `data/opto/` | `opto.ipynb` |
| Widefield | `data/wf/` | `widefield.ipynb` |
| Two-photon | `data/2p/` (large files fetched by `data_downloader.ipynb`) | `2pAnalysis.ipynb`, `TwoPTraces.ipynb`, `plottraces3.ipynb`, `2pSeqWithinDeviation.ipynb` |
| Tracking (SLEAP) | `data/tracking/` | `Tracking.ipynb` |
| Model fits | `data/RLModel/{chisq,mle}_*.pkl` | `rlmodel/model_analysis.ipynb`, `model_to_behavior.ipynb`, `model_neural_correlate.ipynb` |

---

## Figure 1 — Behavioural paradigm: mice and humans

> *A novel behavioural paradigm demonstrates similar flexible decision-making
> strategies in mice and humans.*

| Panel | Content & reported numbers | Notebook | Backend | `results/` |
|---|---|---|---|---|
| **1A** | RDK task schematic for head-fixed mice | — | *not from this repo* | — |
| **1B** | 3D render of the floating platform, three nose-pokes | — | *not from this repo* | — |
| **1C** | Sampling-time distributions across four contexts: freely-moving mice (n=20), head-fixed RDK mice (n=20), human accuracy (n=18), human speed (n=18) | `behavior.ipynb` §"Freely moving mice behavior", §"Mouse & human sampling time together", §"All RDK mice vs all human speed-context trials" | `behavior/stkde.py::plotFMHFSubjects`, `::plotMouseHumanSTKde`, `::plotMiceVsHumansST` | `behavior/humans_mice_reaction_time_dist.svg`, `behavior/stimulus_time_mice_vs_humans.svg`, `behavior/fm_hf/` |
| **1D** | Fast/typical/slow histograms × difficulty. n=20, 262 sessions, **63,702 trials** | `behavior.ipynb` "Fig. 1d" | `figcode/stbydifficulty.py::stDistOnly` | `behavior/st_only/` |
| **1E** | Mouse accuracy vs coherence, fast vs slow. n=20; fast 20,864 / slow 20,054 trials. Paired *t*, Holm–Bonferroni: 10% *p*=0.1287, 20% *p*<0.001, 50% *p*=0.003, 100% *p*<0.001 | `behavior.ipynb` "Fig. 1e" | `figcode/psychometric.py::slowFastPsych` (+ vendored `figcode/psychofit/`) | `behavior/psych_mice/`, `behavior/slow_fast_perf.svg` |
| **1F** | Human speed-context accuracy. n=18; legend gives 8,306 / 8,303 trials, Table 1 gives 6,602 / 6,555 for the test. Paired *t*: 3% *p*=1, 8% *p*=1, 14% *p*=1, 33% *p*=0.0107, 75% *p*=0.0108 | `behavior.ipynb` "Fig. 1f" | `figcode/psychometric.py::slowFastPsych` | `behavior/psych_human/` |
| **1G** | Mean z-scored sampling time (correct trials) vs difficulty, fast vs slow. Mice n=20, 15,461 trials: fast slope 0.02 (θ=1.08°), slow slope 0.14 (θ=7.83°). Human speed n=18, 8,185 trials: fast slope 0.06 (θ=3.70°), slow slope 0.74 (θ=36.52°) | `behavior.ipynb` "Fig. 1g-right" plus the mice cell above it | `figcode/stbydifficulty.py::stVsDiffOnly`, `::loopstVsDiffOnly` | `behavior/st_vs_diff_fast_slow/`, `behavior/humans_st/` |
| **1H** | Mean z-scored sampling time vs reward rate (5-trial sliding window). Mice n=20, humans (speed) n=18 | `behavior.ipynb` "Fig. 1h, Ext. Fig. 3c, d" | `behavior/rewardrate.py::calcAvgRewardRate`, `::loopRewardRateAnalysis` | `behavior/rt_by_difficulty/` |
| **1I left** | % previous-correct, fast vs slow. **p = 0.007**, paired *t*, n=20 mice (20,864 / 20,054 trials) | `behavior.ipynb` "Fig 1i-left" | `figcode/prevoutcomecurquantile.py::prevOutcomeCurQuantile` | `behavior/prev_choice_by_quantile.svg` |
| **1I right** | Stay/switch update magnitude (mean of \|win\| and \|lose\| updates), fast vs slow. **p = 0.0046** (Table 1 gives 0.004), paired *t*, n=20 (20,851 / 20,044 trials) | `behavior.ipynb` "Fig. 1i-right" | `behavior/stayswitchupdate.py::calcWinLoseUpdates` + **inline** `plotSubjectsQuantileUpdate` | `behavior/QuantileWinLoseUpdate/` |

---

## Figure 2 — RL-enhanced drift-diffusion model

> *Reinforcement learning-enhanced drift diffusion models capture adaptive
> behavioural strategies.*

| Panel | Content & reported numbers | Notebook | Backend | `results/` |
|---|---|---|---|---|
| **2A** | Variance explained per predictor (OLS log-likelihood, leave-one-out), n=9 mice: coherence **27.15% ± 6.87**, reward history **34.36% ± 7.94**, motor bias **21.02% ± 5.82**, win-stay **19.32% ± 5.46**. Contributions sum to 101.9%; design condition numbers 14.3–22.2 | `behavior.ipynb` "Fig. 2A" | `behavior/varexplained.py` | `behavior/model_OLS_var_explained_w_filter.svg` |
| **2B** | Reward-rate optimum vs observed sampling time (example mouse + population aligned to each animal's optimum). 17 animals; those with <1,000 trials excluded | `behavior.ipynb` §"Reward-optimal sampling time" | `behavior/optimalsampling.py` | `behavior/optimal_sampling/` |
| **2C** | Schematic of the RL-DDM family | — | *not from this repo* | — |
| **2D** | Schematic: Q-learning → starting point (z); R-learning → noise amplitude | — | *not from this repo* | — |
| **2E** | Reward rate vs z-scored sampling time, example mouse vs model variants | `rlmodel/model_analysis.ipynb` "Fig. 1k, Ext. Fig. 5a-3rd column" | `rlmodel/model/plotter.py` | `RLModel/RewardRate/` |
| **2F** | Psychometric fast/slow, example mouse plus DDM+QL+RL fit | `rlmodel/model_analysis.ipynb` §"Save or view models fits for the different subjects" | `rlmodel/model/plotter.py` | `RLModel/figs/{subject}/RewardRate + Q-Val.svg` |
| **2G** | Aggregate model comparison: psychometric R² ("Psy") and reward-rate Pearson r ("Rew"), 100 simulation seeds per subject × model | `rlmodel/model_analysis.ipynb` "Fig. 1l" | `rlmodel/model/aggregate.py`, `rlmodel/model/aggregate_plot.py`; cluster path `metrics_runner.py` → `model/metrics_shards.py` | `RLModel/aggregates_R2_bar.svg` |

---

## Figure 3 — Widefield imaging and cortex-wide optogenetics

> *Spatiotemporal activation of frontal cortex regions causally relates to fast
> and slow decision-making strategies.*

| Panel | Content & reported numbers | Notebook | Backend | `results/` |
|---|---|---|---|---|
| **3A** | Widefield setup schematic | — | *not from this repo* | — |
| **3B** | Pixel-wise ΔF/F maps in three windows of a typical (1.0–1.2 s) trial, Allen atlas outlines | `widefield.ipynb` §"Plot Normalized Sampling" *(inferred)* | `widefield/pipelineprocessors.py`, `common/plottracesavg.py`, `common/_imaging.py::plotHeatMap` | `WF/standard_map/`, `WF/redefined_map/` |
| **3C** | Mean ΔF/F per area (V1, PPC, M1, M2), 21 sessions / 6 mice, 1.0–1.2 s trials; inset = example mouse | `widefield.ipynb` "Fig. 2c-big", "Fig. 2c-inset" | `common/plottracesavg.py::plotNormalized` | `WF/standard_map/RDK/ReactionTime/…_Many_Areas_Clrs2_Comb_ZScore.svg` |
| **3D** | Opto silencing during 1 s fixed sampling. **MFC 28.90% ± 4.5**, **LFC 21.53% ± 6.27** performance drop. Hierarchical bootstrap 10,000 iterations + Holm–Bonferroni: V1 *p*=0.108 (n=7, 1,126 opto / 3,088 control), PPC *p*=0.021 (n=6, 719 / 2,131), MFC *p*<0.001 (n=7, 1,018 / 3,704), LFC *p*=0.01 (n=7, 1,503 / 4,859). Session counts in the Results text: V1 16, PPC 17, MFC 23, LFC 30 | `opto.ipynb` "Fig. 2d" | `opto/optoprocessor.py::plotOptoEffect`, `opto/bootstrap2regions.py::bootstrapSignTestApproach2`, `opto/bootstrapping.py::bootstrapPerf` | `optogenetics/Performance/` |
| **3E** | MFC/LFC segmentation; heatmaps plus z-scored traces for fast / typical / slow (n=6 mice, 30 sessions); δ arrow at end of sampling | `widefield.ipynb` "Fig. 2e" | `widefield/pipelineprocessors.py`, `common/plottracesavg.py` | `WF/RDK/ReactionTime/…/Q1..Q3_midline_Comb_ZScore.svg` |
| **3F** | δ (LFC−MFC) at end of sampling: **fast 0.32 ± 0.06, typical 0.56 ± 0.06, slow 0.63 ± 0.06**. RM-ANOVA + Holm–Bonferroni: fast-vs-typical *p*<0.0001, fast-vs-slow *p*<0.0001, typical-vs-slow *p*=0.1245. n=30 sessions / 6 mice | `widefield.ipynb` §"End of Sampling" | `widefield/mfclfcquantiles.py` (`SamplingAnchor.END`) — **covered by tests** (`widefield/tests/test_mfclfcquantiles.py`) | `WF/MFC_LFC_dist_by_session.svg` |

---

## Figure 4 — Temporal opto dissection and two-photon sequences

> *Optogenetic inhibition and two-photon imaging reveal distinct temporal
> dynamics and neuronal activation sequences in MFC and LFC.*

| Panel | Content & reported numbers | Notebook | Backend | `results/` |
|---|---|---|---|---|
| **4A** | Bilateral MFC/LFC inhibition schematic | — | *not from this repo* | — |
| **4B top** | Early (0–350 ms) / late (650–1000 ms) design over a 1 s fixed sample | — | *not from this repo* | — |
| **4B bottom** | Widefield MFC/LFC z-scored ΔF/F under fixed 1 s sampling, n=4 mice, 12 sessions | `widefield.ipynb` "Fig. 3b-background" | `common/plottracesavg.py` | `WF/RDK/FixedTime/…_QFixedTime_Comb_ZScore.svg` |
| **4C** | Early/late inhibition performance drop. Early: **MFC 14.42% ± 1.43** (n=7), **LFC 7.36% ± 4.25** (n=10). Late: **MFC 22.03% ± 5.92** (n=5), **LFC 15.07% ± 3.37** (n=8). Hierarchical bootstrap + Holm–Bonferroni: MFC-early *p*=0.026, LFC-early *p*=0.156, MFC-late *p*=0.0096, LFC-late *p*=0.0096 | `opto.ipynb` "Fig. 3c" (§"Only two areas") | `opto/optoprocessor.py`, `opto/bootstrap2regions.py` | `optogenetics/Performance/opto_effect_All_Mice_MFC_LFC.svg` |
| **4D** | 2P setup schematic, example micrograph, three example ΔF/F traces. Dataset: n=6 mice, 23 sessions; **837/2,340 active neurons in MFC, 609/1,536 in LFC** | `TwoPTraces.ipynb` (traces only) | schematic and micrograph *not from this repo* | `2P/Sessions/` |
| **4E** | Mean z-scored L2/3 population activity (solid) with widefield overlay (dashed), MFC/LFC × fast/slow, n=6 mice | `TwoPTraces.ipynb` §"Plot average traces for different combinations"; `plottraces3.ipynb` §"Activity Sum" | `twop/plottracesavg.py::plotNormalized`, `twop/plot/activitysum.py::plotActivitySum` | `2P/Sessions/*/summary/` |
| **4F** | Single-trial active (green) / inactive (grey) traces plus peak mini-heatmap | `2pAnalysis.ipynb` §"Plot neurons traces in fast vs slow trials" | **inline** | `2P/Sessions/` |
| **4G** | Venn of ≥10%-active neurons: fast, slow, both — MFC (top), LFC (bottom) | `2pAnalysis.ipynb` §"Overlap between Active Impulsive neurons and Deliberate neurons" | **inline** `_fastSlowOverlap` (uses `matplotlib_venn`) | `2P/FastSlowVenn/valid_10%_M2.svg`, `…_ALM.svg` |
| **4H** | Population heatmaps fast vs slow, ranked on the all-trials peak (example MFC session) | `TwoPTraces.ipynb` "Fig. 3h" | **inline**; `twop/plot/plottracetrialsheatmap.py` implements the same figure but is orphaned (see audit) | `2P/Sequence/` |
| **4I** | Schematic of the rank-deviation method | — | *not from this repo* | — |
| **4J** | Permutation null vs observed rank deviation, 100,000 iterations/session, Holm–Bonferroni. **MFC (n=13):** *p*≤0.001 in 8/13, 0.001<*p*≤0.01 in 1/13, 0.01<*p*≤0.05 in 2/13. **LFC (n=10):** *p*≤0.001 in 8/10, 0.001<*p*≤0.01 in 1/10, 0.01<*p*≤0.05 in 1/10 | `2pAnalysis.ipynb` §"Monte Carlo Simulation / Permutation testing", §"Seq firing deviation" | `twop/seqdeviation.py` (+ `twop/tests/test_seqdeviation.py`) | `2P/SeqWithinDeviation/` |
| **4K left** | Example "rigid" and "stretching" neurons, traces coloured by sampling duration | `plottraces3.ipynb` §"Correlation between rt and activity on single cell level" | **inline** | `2P/RT_Stats/traces/` |
| **4K right** | Pie charts: rigid / stretching (peak-timing only) / AUC only / both, per region. Threshold \|r\| > 0.3 | `plottraces3.ipynb` §"Pie-Chart for rigid/streteching neurons", §"MFC vs LFC bars at the same correlation threshold" | **inline** pies; `twop/plot/corrthreshregions.py::plotRegionBars` for the per-session bars (+ tests) | `2P/RT_Stats/` |

---

## Figure 5 — Choice-selective sequences and mid-sampling inhibition

> *Choice-selective neural sequences in MFC and LFC exhibit distinct modes of
> operation, with differential contributions to voluntary decision-making.*

| Panel | Content & reported numbers | Notebook | Backend | `results/` |
|---|---|---|---|---|
| **5A** | Choice-match / choice-mismatch schematic | — | *not from this repo* | — |
| **5B** | Example-session heatmap plus three example neurons (#9 left-preferring, #38 non-selective, #46 right-preferring) | `TwoPTraces.ipynb` "Fig. 4b" | `twop/plottracesavg.py`, `twop/plottuning.py` | `2P/Sessions/*/sgf_and_not_sgf/…/Direction/` |
| **5C** | % choice-selective neurons per session. **MFC 18.06% ± 3.04 SEM** (837 neurons, 13 sessions), **LFC 33.07% ± 3.68 SEM** (609 neurons, 10 sessions). Per-neuron Mann-Whitney U; across-region Student *t*, **p = 0.00478** | `2pAnalysis.ipynb` §"Sgf. Neurons Pie Charts" | `twop/sgfneurons.py` (+ `twop/tests/test_sgfneurons.py`) | `2P/MFC_correlation_pie_chart.pdf`, `2P/LFC_…`, `2P/Both_MFC_LFC_…` |
| **5D** | Example neuron (#9), matched vs mismatched under slow (top) and fast (bottom) | `2pAnalysis.ipynb` §"Preferred vs anti-preferred - new way" | **inline** | `2P/Sessions/` |
| **5E** | Sequence-ordered population response, matched vs mismatched, slow/fast × MFC/LFC. **MFC 13 sessions / 182 neurons; LFC 10 sessions / 197 neurons.** 7 bin-pairs; first and last excluded when assigning neurons | `2pAnalysis.ipynb` §"Cityscape Figure" | `twop/plottuning.py::TrajectoryTuningPlot` | `2P/Sequence/` |
| **5F** | Coding efficiency `(Match − Mismatch)/Match` at sampling start and decision, MFC vs LFC × slow/fast | `2pAnalysis.ipynb` "Fig. 5F" — `plotTrajectory(as_weight=True, weight_only_first_last=True)` | `twop/plottuning.py` | `2P/Sequence/` |
| **5G** | Mid-sampling (0.3–0.9 s) inhibition design plus example-mouse sampling-duration distributions | `opto.ipynb` "Fig. 4g" | `opto/optoreactiontime.py::optoReactionTime` | `optogenetics/DecisionInitiation/` |
| **5H** | Cumulative sampling-duration distributions and post-offset performance: control / MFC / LFC | `opto.ipynb` "Fig. 4h" | `opto/optoreactiontime.py` | `optogenetics/DecisionInitiation/` |

---

## Figure 6 — Feedback in MFC controls strategy update

> *Medial frontal cortex (MFC) controls feedback-dependent shifts in behavioural
> strategy.*

| Panel | Content & reported numbers | Notebook | Backend | `results/` |
|---|---|---|---|---|
| **6A top** | Self-initiated task structure (feedback between trial *t* and *t+1*) | — | *not from this repo* | — |
| **6A bottom** | Sampling + feedback heatmaps, MFC / LFC, neurons ordered by sampling-phase peak (example mouse) | `TwoPTraces.ipynb` "Fig. 5c bottom / top left / top right" (§"Feedback heatmaps") | `twop/plottracesavg.py` | `2P/Sessions/` |
| **6B** | Feedback-tuned neuron fraction across early (0–0.33) / middle (0.33–0.66) / late (0.66–1) sampling thirds, MFC vs LFC | `2pAnalysis.ipynb` §"Extract sampling and feedback tuned neurons" | **inline** `extractTracesPreferences` | `2P/sgf_tests/` |
| **6C** | Example feedback-tuned neuron multiplexing previous outcome × current choice direction | `2pAnalysis.ipynb` §"Balance Change - New Way - PrevChoiceCorrect tuning" | **inline** | `2P/RewardRate/IncNeuron/BalanceChangeCurChoiceEx_*.svg` |
| **6D** | Feedback-window inhibition → next-trial z-scored sampling time, post-correct vs post-incorrect, n=4 mice. Within-condition paired *t*: control *p*=0.2591, MFC *p*=0.5534, LFC *p*=0.1627. Cross-region post-correct (RM-ANOVA *p*=0.0008): control-vs-MFC *p*=0.0218, control-vs-LFC *p*=0.4495, MFC-vs-LFC *p*=0.0218. Cross-region post-incorrect (RM-ANOVA *p*=0.0373): all pairwise *p* ≥ 0.2176 | `opto.ipynb` "Fig. 5d" | `opto/optofeedback.py::optoFeedback` | `optogenetics/feedback/` |
| **6E** | Venn: MFC tuning to priors vs current-trial variables, sampling vs feedback epochs. **Priors during sampling: 80.1% ± 2.8 SEM** | `2pAnalysis.ipynb` §"Sampling & Feedback Venn Diagarams" | **inline** `_plotBrainRegionTuning`. The superseded module `twop/plot/statspriorcuroverlap.py` targets the same panel but is orphaned | `2P/PriorCurrentTuning/*_prior_current_tuning.svg` |

---

## Figure 7 — Computational and neural framework

| Panel | Content & reported numbers | Notebook | Backend | `results/` |
|---|---|---|---|---|
| **7A** | % neurons per session correlated (\|r\| ≥ 0.3) with model latents. **Q-value (side bias) 10.38% ± 2.68 SEM; R-value (reward rate) 8.89% ± 2.58 SEM.** n=22 sessions. Significance against a per-neuron activity-shuffled null (1,000 shuffles), Holm–Bonferroni. Fit: joint MLE+χ² with w_MLE=1, w_χ²=0.5 | `rlmodel/model_neural_correlate.ipynb` §"Factor modulation bars" | `rlmodel/model/neural_correlate.py` (+ `tests/test_neural_correlate.py`) | `RLModel/neural_correlate/` |
| **7B** | DV-correlated neurons, fast vs slow. **Fast 2.07% ± 0.52; slow 4.56% ± 1.1** per session. Two-sided session-paired sign-flip permutation, 1,000 iterations | `rlmodel/model_neural_correlate.ipynb` §"Fast vs slow: drift-correlated neurons" | `rlmodel/model/neural_correlate.py` | `RLModel/neural_correlate/` |
| **7C** | MFC ("Strategy") → LFC ("Action") schematic | — | *not from this repo* | — |
| **7D** | 3D landscape: z-scored sampling time vs relative Q-value × reward rate, one surface per difficulty. Pseudo-sessions resampled within session with randomised stimulus strengths; Gaussian smoothing σ=1 bin | `rlmodel/model_to_behavior.ipynb` "Fig. 5f, middle" | **inline** `plotQ_R_Heatmap`; simulation through `rlmodel/model/posterior_simulate.py` | `RLModel/Q_R_Heatmap.svg`, `behavior/optimal_sampling/3d_plot.svg` |
| **7E** | Cellular-implementation schematic (slow vs fast) | — | *not from this repo* | — |

---

## Supplementary figures

### Figure S1 — Floating-platform system

| Panel | Content | Notebook | Backend |
|---|---|---|---|
| S1A | Side / top / bottom 3D views of the nose-poke platform | — | *not from this repo* |
| S1B | Free-sampling task structure; example-mouse psychometric across 32 sessions / 8,696 trials; sampling time vs coherence by choice side | `behavior.ipynb` §"Psychometric (overall)" | `figcode/psychometric.py::loopPsych`, `figcode/stbydifficulty.py` |
| S1C | Fixed-variable-sampling task; example-mouse psychometric (9,450 trials); chronometry curve, n=3 mice | `behavior.ipynb` "Ext. Fig. 1c-right" | `behavior/chronometry.py::chronometry` |
| S1D | Freely-moving vs head-fixed KDEs for one mouse: freely-moving light-chasing 963 trials, head-fixed light-chasing 1,089, head-fixed RDK 2,455 | `behavior.ipynb` "Ext. Fig. 1d-right" | `behavior/stkde.py::plotFMHFSubjects` |

`results/behavior/All Animals_chrono.svg`, `behavior/fm_hf/`.

### Figure S2 — Fast/slow strategies in humans and mice

| Panel | Content & numbers | Notebook | Backend |
|---|---|---|---|
| S2A | Human RDK task schematic (Accuracy / Speed contexts) | — | *not from this repo* |
| S2B | z-scored sampling-time distributions across contexts (z-scored across both contexts) | `behavior.ipynb` §"Plot user data" | **inline** `assignZScoredST`, `errorsDistribution`, `processSubject` |
| S2C | Dispersion (IQR). Shapiro-Wilk → one-way ANOVA (*p*<0.001) + Tukey HSD: accuracy-vs-speed *p*=0.001, accuracy-vs-mice *p*=0.001, speed-vs-mice *p*=0.0114. n=18/18/20 | `behavior.ipynb` §"Per-subject z-scored sampling time & subject-sigma dispersion" | `behavior/stdispersion.py::plotSTDispersion` (+ `behavior/tests/test_stdispersion.py`) |
| S2D | Raw-seconds distributions, human speed (n=18) vs mice (n=20) | `behavior.ipynb` §"All RDK mice vs all human speed-context trials" | `behavior/stkde.py::plotMiceVsHumansST` |
| S2E–G | Example mouse M#85 (3,519 trials): histograms by difficulty (E), correct/incorrect sampling time by difficulty (F), fast/slow psychometric (G) | `behavior.ipynb` "Ext Fig. 2j" (E), "Ext. Fig 2e" (F), "Ext. Fig f" (G) | `figcode/stbydifficulty.py`, `figcode/psychometric.py` |
| S2H–I | Human speed (n=18): z-scored sampling time by difficulty (H), correct-vs-incorrect (I) | `behavior.ipynb` "Ext Fig. 2d", "Ext. Fig 2h" | as above |
| S2J | Human accuracy fast/slow psychometric. Paired *t* + Holm: 1% *p*=0.4858, 3% *p*=0.061, 8% *p*=0.0168, 14% *p*=0.0317, 32% *p*=0.9228, 77% *p*=0.4858. n=18 (2,951 / 2,537 trials) | `behavior.ipynb` "Ext. Fig. 3i" | `figcode/psychometric.py::slowFastPsych` |
| S2K–L | Human accuracy (n=18): histograms (K), correct-vs-incorrect (L) | `behavior.ipynb` "Ext Fig. 2g", "Ext. Fig 2k" | `figcode/stbydifficulty.py` |
| S2M | Easy-trial accuracy fast vs slow across the three groups. Paired *t* + Holm: human accuracy *p*=0.165, human speed *p*=0.006, mice *p*<0.001 | `behavior.ipynb` "Ext. Fig. 2l" | **inline** `_plotGroup`, `localSlowFasPsych` |

`results/behavior/humans_mice_sampling_time_dispersion_iqr.svg`, `…_std.svg`.

### Figure S3 — Trial history, sensory evidence, and posture

| Panel | Content & numbers | Notebook | Backend |
|---|---|---|---|
| S3A | Human accuracy: z-scored sampling time of correct trials vs difficulty. Fast slope 0.45 (θ=24.19°), slow slope 0.82 (θ=39.26°) | `behavior.ipynb` "Ext Fig. 2a" | `figcode/stbydifficulty.py::stVsDiffOnly` |
| S3B | Mice sampling time vs reward rate, n=20. Shapiro-Wilk (2/5 bins non-normal) → Kruskal-Wallis **p=0.0001** + Dunn/Holm | `behavior.ipynb` "Fig. 1h, Ext. Fig. 3c, d" | `behavior/rewardrate.py::loopRewardRateAnalysis` |
| S3C | Human speed, n=18. All bins normal → one-way ANOVA **p=0.0006** + Tukey | as above | as above |
| S3D | Human accuracy, n=18. 1/4 bins non-normal → Kruskal-Wallis **p=0.017** + Dunn/Holm | as above | as above |
| S3E | Heatmap of mean sampling time by previous outcome × current difficulty. Left: example mouse M#46. Right: all mice, z-scored | `behavior.ipynb` "Fig 2e left", "Ext. Fig 2e-right, f" | `figcode/stheatmap.py::stHeatmap` |
| S3F | Violin plots of the same. Kruskal-Wallis **p<0.001** + Dunn/Holm, n=20 | as above | as above |
| S3G | Win/lose update vs z-scored sampling time, n=9 mice. Bins of 800 trials, Gaussian σ=1 bin. Two mice excluded (18.62% and 13.26% of trials below z=−1 vs a group mean of 3.28%) | `behavior.ipynb` "Ext. Fig. 3g" | **inline** `_plotOverTime`, `loopDifficulties`, `loopWinStay`, `staySwitchCDF` |
| S3H | Win/lose update by current difficulty, computed per mouse then averaged, n=9 | `behavior.ipynb` "Ext. Fig. 3h" | `figcode/stayswitch.py::staySwitchUpdate` |
| S3I | Choice bias, fast vs slow. Paired *t*, **p = 0.016**, n=20 | `behavior.ipynb` "Ext. Fig. 3i" | `behavior/bias.py::plotBias` |
| S3J | Per-trial centroid rotation-angle distance travelled, by strategy. Computed as the within-trial **range** (intended; Methods wording to be updated, manuscript issue #11) | `Tracking.ipynb` "Extended Fig. 4b-c" | `tracking/centroids.py::plotCentroids` (+ tests) | `tracking/distance_All Tracked Subjects.svg` |
| S3K | Polar histogram of mean rotation angle, 5° bins, **un-normalised** (`choice_normed=False`). Verified against the committed SVG: all 84 bars match a `False` render (r = 1.000000, constant area ratio), not a `True` one (r = 0.94). **Next revision: regenerate with `choice_normed=True`** to match the legend and Methods (manuscript issue #10) | `Tracking.ipynb` "Extended Fig. 4b-c" (same call as S3J) | `tracking/centroids.py::plotCentroids` (+ tests) | `tracking/centroid_rotation_All Tracked Subjects.svg` |
| S3L | Same, relative to each **session's** preferred side (intended, and what the code does; legend and Methods say "animal", manuscript issue #12) | `Tracking.ipynb` "Extended Fig. 4d" | `tracking/centroids.py::plotCentroids(preferred_side_normed=True)` (+ tests) | `tracking/centroid_rotation_preferred_All Tracked Subjects.svg` |
| S3M | Per-animal Kruskal-Wallis on \|trial mean angle − the animal's modal angle\| across strategies, Holm-corrected across the 4 mice. **0/4 significant** (Holm 0.073 / 0.722 / 0.105 / 0.722). This is the intended analysis; the legend and Methods describe a distance-travelled test instead and are to be updated (manuscript issue #9) | `Tracking.ipynb` "Extended Fig. 4e" | `tracking/strategy.py::plotStrategyComparison` (+ tests) | `tracking/strategy_dist_sgf.svg` |

`results/behavior/StaySwitch/`, `behavior/bias/abs_all_bias.svg`, `results/tracking/`.

### Figure S4 — Model variants

| Panel | Content | Notebook | Backend |
|---|---|---|---|
| S4A | Example-mouse fast/slow psychometric vs DDM, DDM+QL and DDM+RL | `rlmodel/model_analysis.ipynb` "Ext. Fig. 5a-1st & 2nd columns" | `rlmodel/model/plotter.py` |
| S4B/C | "Psy"/"Rew" correlations for the four reward-rate channels: noise, bound, drift `g(r)=1+r`, drift `g(r)=2−r`. Modulating the drift reverses the observed speed–accuracy trade-off | `rlmodel/model_analysis.ipynb` §"Model Comparison: reward-rate implementation" and §"…channel" | `rlmodel/model/aggregate.py`; channels in `model/drift.py` (`NoiseGain-`, `Bound-`, `DriftGain-`, `DriftGain(1+r)-`) |

`results/RLModel/aggregates_R2_drift_rr.svg`, `aggregates_R2_scale_bound.svg`, `aggregates_R2_mle_weights.svg`.

### Figure S5 — Widefield, typical trials

| Panel | Content | Notebook | Backend |
|---|---|---|---|
| S5A | Widefield setup plus clear-skull photographs | — | *not from this repo* |
| S5B | Session selection for typical (1000–1200 ms) trials, coloured by mouse | `widefield.ipynb` "Ext. Fig. 6b" | `widefield/pipelineprocessors.py` |
| S5C | Example-mouse pixel-wise maps in successive 200 ms windows | `widefield.ipynb` | `common/plottracesavg.py` |
| S5D | Trial-averaged area traces, example mouse | `widefield.ipynb` "Ext. Fig. 6d" | `common/plottracesavg.py` |
| S5E | Cross-mouse area traces, n=6 mice / 21 sessions | `widefield.ipynb` "Ext. Fig. 6e" | `common/plottracesavg.py` |

`results/WF/typical_trials_rt_distribution.svg` (plus `standard_map/` and `redefined_map/` variants).

### Figure S6 — Fixed-sampling optogenetics per region

| Panel | Content & numbers | Notebook | Backend |
|---|---|---|---|
| S6A | Bilateral inactivation schematic (6 mW/mm², 40 Hz, clear skull) | — | *not from this repo* |
| S6B–E | Control vs opto psychometrics: V1 (B), PPC (C), MFC (D), LFC (E) | `opto.ipynb` "Extended Fig. 7b-e" | `opto/optoprocessor.py::plotOptoPsych` |
| S6F | Effect on motor initiation, MFC / LFC / control. **MFC: no significant delay; LFC: significant delay** | `opto.ipynb` §"Decision Initiation Time" | `opto/optoreactiontime.py` |
| S6G | No-opsin control (Thy1-GP4) MFC inhibition. Hierarchical bootstrap 10,000 iterations, **p = 0.387**, n=3 mice (230 opto / 1,024 control) | `opto.ipynb` "Extended Fig. 7f" | `opto/optoprocessor.py`, `opto/bootstrap2regions.py` |

`results/optogenetics/Psychometric/fixedtime_full_inhib/`, `optogenetics/Performance_Control/`.

### Figure S7 — Frontal dynamics and timed-inhibition psychometrics

| Panel | Content | Notebook | Backend |
|---|---|---|---|
| S7A | MFC/LFC widefield traces under fixed 1 s sampling, example mouse | `widefield.ipynb` "Ext. Fig. 8a" | `common/plottracesavg.py` |
| S7B | Same for free-sampling trials of 1–1.2 s, same mouse | `widefield.ipynb` "Ext. Fig. 8b" | `common/plottracesavg.py` |
| S7C–D | Early-inhibition (0–350 ms) psychometrics, MFC (C) and LFC (D) | `opto.ipynb` "Extended Fig. 8c-f" | `opto/optoprocessor.py::plotOptoPsych` |
| S7E–F | Late-inhibition (650–1000 ms) psychometrics, MFC (E) and LFC (F) | as above | as above |

`results/optogenetics/Psychometric/fixedtime_early_inhib/`, `…/fixedtime_late_inhib/`.

### Figure S8 — L2/3 population dynamics

| Panel | Content | Notebook | Backend |
|---|---|---|---|
| S8A | Schematic: single-neuron traces → normalised population activity | `plottraces3.ipynb` §"Activity Sum" | `twop/plot/activitysum.py::plotActivitySum` |
| S8B | Normalised MFC/LFC population responses, fast vs slow, example mouse | `TwoPTraces.ipynb`, `plottraces3.ipynb` | `twop/plottracesavg.py` |
| S8C | Widefield vs two-photon, aligned to motor initiation | `plottraces3.ipynb` | `twop/plot/activitysum.py` |

### Figure S9 — Sequence organisation

| Panel | Content & numbers | Notebook | Backend |
|---|---|---|---|
| S9A | Single-neuron traces; decision-probability distributions; mean z-scored ΔF/F aligned to sampling start | `2pAnalysis.ipynb` §"Sequence Extracation" | **inline** |
| S9B | CDF of trial-by-trial firing reliability, MFC/LFC × fast/slow | `2pAnalysis.ipynb` §"CDF for active trials per quantiles" | **inline** |
| S9C | LFC example-session heatmap, fast vs slow, sorted on the slow reference | `TwoPTraces.ipynb` | `twop/plottracesavg.py` |
| S9D | Trial-by-trial rank variability (TRV) for LFC with Fast as reference; x-axis is the equivalent shuffle level (0% reproducible, 50% random, 100% reversed) | `2pSeqWithinDeviation.ipynb` §D and §F | `twop/seqdeviation.py` (+ tests), `twop/shuffle_replay.py` (interactive replay) |
| S9E | TRV within vs cross strategy. **MFC Fast-ref: 16.09% ± 1.02 within / 21.18% ± 2.03 cross. MFC Slow-ref: 15.92% ± 1.22 / 20.78% ± 1.75. LFC Fast-ref: 15.32% ± 1.25 / 20.13% ± 1.73. LFC Slow-ref: 17.32% ± 0.92 / 18.99% ± 1.88** | `2pSeqWithinDeviation.ipynb` §E | `twop/seqdeviation.py` |
| S9F | Schematic of the permutation approach against a "typical"-trial reference | `2pAnalysis.ipynb` | *schematic* |
| S9G | Rank mismatch vs permuted typical trials, 100,000 iterations/session. **MFC: 13/13 fast and 13/13 slow at p≤0.001. LFC: 10/10 fast, 9/10 slow at p≤0.001, 1/10 p>0.05** | `2pAnalysis.ipynb` §"Seq firing deviation" | `twop/seqdeviation.py` |

`results/2P/SeqWithinDeviation/`, `2P/SeqWithinDeviation/sessions/`.

### Figure S10 — Single-neuron modulation by sampling duration

| Panel | Content | Notebook | Backend |
|---|---|---|---|
| S10A | Example neuron aligned to sampling onset: traces, peak-time correlation, AUC correlation | `plottraces3.ipynb` §"Correlation between rt and activity on single cell level" | **inline** |
| S10B | Second example, aligned to movement onset | as above | **inline** |
| S10C | Correlation-coefficient distributions against 1,000 shuffled controls: AUC early (<0.2 s), AUC late (>0.2 s), peak timing late | `plottraces3.ipynb` §"Correlation Hists" | **inline** |
| S10D | Proportion of neurons per session with \|r\| > 0.3 for AUC and/or peak timing, MFC vs LFC. Shapiro-Wilk → Mann-Whitney U or Welch *t* | `plottraces3.ipynb` §"MFC vs LFC bars at the same correlation threshold" | `twop/plot/corrthreshregions.py` (+ `twop/tests/test_corrthreshregions.py`) |

### Figure S11 — Population tuning to sampling time and performance

| Panel | Content | Notebook | Backend |
|---|---|---|---|
| S11A left | Mean z-scored population activity aligned to sampling onset, MFC/LFC × fast/slow | `plottraces3.ipynb` | `twop/plot/activitysum.py` |
| S11A middle | % active neurons in the early window (−0.1 to +0.3 s) vs sampling duration, hard trials, 0.25 s bins; linear fit + Pearson r | `plottraces3.ipynb` §"Active Early" | **inline** |
| S11A right | Early activity vs performance on easy trials, binned 0–30% in 3% steps | `plottraces3.ipynb` §"Plot Active Early then Late" | **inline** |
| S11B | Same for the last 30% of sampling up to +0.1 s (movement start). **MFC negatively correlated with duration; LFC flat** | `plottraces3.ipynb` §"Active Late" | **inline** |

### Figure S12 — Choice-selective population dynamics

| Panel | Content & numbers | Notebook | Backend |
|---|---|---|---|
| S12A | MFC example-session heatmaps for rightward vs leftward choices | `TwoPTraces.ipynb` | `twop/plottracesavg.py` |
| S12B | Proportion of choice-selective neurons over normalised sampling time | `2pAnalysis.ipynb` | `twop/plottuning.py` |
| S12C | % choice-encoding neurons at sampling start, fast vs slow | `2pAnalysis.ipynb` | `twop/plottuning.py` |
| S12D | Matched/mismatched subgroup activity at the start and end of sampling | `2pAnalysis.ipynb` "Fig. S8E" — `plotTrajectory(plot_single_pts=True)` | `twop/plottuning.py::TrajectoryTuningPlot` |
| S12E | As Figure 5E but including incorrect trials, with single-neuron grey lines | `2pAnalysis.ipynb` | `twop/plottuning.py` |
| S12F | Coding efficiency for all trials, with single-neuron points | `2pAnalysis.ipynb` "Fig. S8F" | `twop/plottuning.py` |
| S12G | Logistic-regression choice decoder at sampling start/end, fast vs slow. 1,000 random 70/30 splits | `plottraces3.ipynb` §"Decoders" | **inline** plus `twop/classifyplayground.py`, vendored `twop/relogit/` |
| S12H | Coding efficiency across the sequence, slow (solid) vs fast (dashed) | `2pAnalysis.ipynb` "Fig. S8H" | `twop/plottuning.py` |
| S12I | Linear fits of matched vs mismatched activity across bins | `2pAnalysis.ipynb` "Fig. S8J" | `twop/plottuning.py` |
| S12J | Movement-tuned neurons at the end of sampling: **MFC 5.10% ± 1.02** (38/837 neurons, 13 sessions), **LFC 11.85% ± 1.50** (68/609 neurons, 10 sessions), Student *t* **p = 0.000941** | `2pAnalysis.ipynb` §"Movement-time neurons: MFC vs LFC" | `twop/movementneurons.py` (+ `twop/tests/test_movementneurons.py`) |

`results/2P/MovementNeurons/last_two_bins_prcnt_movement.svg`, `…_choice.svg`.

### Figure S13 — Mid-sampling inhibition detail

| Panel | Content | Notebook | Backend |
|---|---|---|---|
| S13A | Performance drop under MFC and LFC inhibition | `opto.ipynb` "Extended Fig. 14a-b" | `opto/optoreactiontime.py` |
| S13B | Decision probability inside the 0.3–0.9 s inhibition window | as above | as above |
| S13C | Cumulative trial distribution over z-scored sampling time | `opto.ipynb` "Extended Fig. 14c-d" | as above |
| S13D | Performance vs z-scored sampling time; linear fits pinned to (−1.5 z, 50%); 50-trial bins | as above | as above |
| S13E | Median z-scored sampling time, all trials | `opto.ipynb` "Extended Fig. 14e-g" | as above |
| S13F | Mean performance, all trials | as above | as above |
| S13G | As S13E, restricted to trials ending after the inhibition window (>0.9 s) | as above | as above |

Statistics: RM-ANOVA on Condition (Control, MFC, LFC), then paired *t* with
Holm–Bonferroni.

### Figure S14 — Feedback history and model–neural correlation

| Panel | Content & numbers | Notebook | Backend |
|---|---|---|---|
| S14A | Feedback ↔ next-trial-sampling correlation vs normalised sampling position. **MFC r = −0.24, R² = 0.06, p = 0.00001; LFC r = −0.16, R² = 0.03, p = 0.03** | `2pAnalysis.ipynb` §"Sgf. between Prev. Correct vs Incorrect" | **inline** |
| S14B | Fitting-criterion comparison. Mean model penalty: **pure χ² 32.055 ± 13.155** (λ=0), **pure MLE 1.066 ± 0.112**, **joint w_MLE=1 / w_χ²=0.1 → 1.141 ± 0.124**, **joint w_MLE=1 / w_χ²=0.5 → 1.235 ± 0.132** | `rlmodel/model_analysis.ipynb` §"Model Comparison: fitting criterion"; `rlmodel/model_compare.ipynb` | `rlmodel/model/compare.py`, `rlmodel/extract_model_losses.py` |
| S14C | Figure 7A/B restricted to MFC (12 sessions) | `rlmodel/model_neural_correlate.ipynb` | `rlmodel/model/neural_correlate.py` |
| S14D | Same for LFC (10 sessions) | as above | as above |
| S14E | Example neuron modulated by the R-value (reward rate) | as above §"Per-neuron scatters" | as above |
| S14F | Example neuron modulated by the Q-value (side bias) | as above | as above |
| S14G | Neurons correlated with the decision variable under fast and/or slow strategies | as above §"Per-neuron fast/slow scatters" | as above |
| S14H | Example neuron: fast r = 0.11 vs slow r = 0.55 | as above | as above |

---

## Panels with no code path in this repository

Beyond the schematics and photographs listed panel-by-panel above:

- **Figures 1A–B, 2C–D, 3A, 4A, 4B-top, 4D (schematic and micrograph), 4I, 5A,
  6A-top, 7C, 7E, S1A, S2A, S5A, S6A, S9F** — illustrations, renders and imaging
  hardware photographs.
- **Supplementary Videos 1–3** — raw behavioural video, not derived from the
  analysis pipeline.
- **Key Resources Table** and **Supplementary Data Table 2** — maintained in the
  manuscript, not generated from code.
- **Supplementary Data Table 1** — assembled by hand from the per-panel test
  output printed by the notebooks; there is no single script that emits it.

## Panels whose driver notebook is missing

`twop/plot/statspriorcuroverlap.py`, `statssamplingfeedback.py`,
`statsregiondist.py`, `statsearlylatesampling.py`, `statsbetweenregions.py` and
`plottracetrialsheatmap.py` all draw figures matching Figures 6B/6E and 4H, but
**no notebook in the repository imports them**. The live implementations for
those panels are inline cells in `2pAnalysis.ipynb` and `TwoPTraces.ipynb`. See
[`repo-audit.md`](repo-audit.md#orphaned-backends).
