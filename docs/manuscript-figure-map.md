# Manuscript → code: panel-by-panel figure map

Every panel of **Nashaat, Oraby et al., "The neural mechanisms of fast versus
slow decision-making", *Neuron* (2026, advance online publication),
[doi:10.1016/j.neuron.2026.10.024](https://doi.org/10.1016/j.neuron.2026.10.024)**,
mapped to the notebook section, backend module and `results/` artifact that
produce it, together with the numbers the manuscript reports for that panel.

Figure numbers are the published ones (Figures 1–7, S1–S14). Notebook section
headings carry the same numbers, so a heading quoted in the **Notebook** column
can be found verbatim in the notebook.

**Legend for the "Backend" column**

- `module::function` — code lives in a Python module (importable, testable).
- **inline** — code lives only in notebook cells.
- *not from this repo* — schematic / render / photograph / micrograph.

**Panels a run does not reproduce** are marked **✗ not drawn**, with the
reason: either no cell draws them, or the cell is a loop over every neuron /
session / subject and the published example's identity was never recorded (so
the loop draws it among all the others, but nothing pins which one it was).

**How this table was checked.** Every notebook was run under a write guard
that records each file it writes (2026-09-21), and every row was then checked
by content against the final legends and figure panels (2026-11-14): the panel
in the PDF against the notebook's embedded output or the `results/` SVG.

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
| Two-photon | `data/2p/` (large files fetched by `data_downloader.ipynb`) | `2p_0_preprocessing.ipynb` (run first), `2p_1_population.ipynb`, `2p_2_sequences.ipynb`, `2p_3_choice.ipynb`, `2p_4_feedback.ipynb` |
| Tracking (SLEAP) | `data/tracking/` | `tracking.ipynb` |
| Model fits | `data/RLModel/{chisq,mle}_*.pkl` | `rlmodel/rlmodel_analysis.ipynb`, `rlmodel_to_behavior.ipynb`, `rlmodel_neural_correlate.ipynb` |

---

## Figure 1 — Behavioural paradigm: mice and humans

> *A novel behavioral paradigm demonstrates similar flexible decision-making
> strategies in mice and humans.*

| Panel | Content & reported numbers | Notebook | Backend | `results/` |
|---|---|---|---|---|
| **1A** | RDK task schematic for head-fixed mice | — | *not from this repo* | — |
| **1B** | 3D render of the floating platform, three nose-pokes | — | *not from this repo* | — |
| **1C** | Sampling-time distributions across four contexts: freely-moving mice (n=20), head-fixed RDK mice (n=20), human accuracy (n=18), human speed (n=18) | `behavior.ipynb` "Mouse & human sampling time together — Figure 1C" | `behavior/stkde.py::plotMouseHumanSTKde` | `behavior/fm_hf/stimulus_time_mouse_a-14_human_Subject44_2025_zscored.svg` |
| **1D** | Fast/typical/slow histograms × difficulty. n=20, 262 sessions, **63,702 trials** | `behavior.ipynb` "Figure 1D" | `figcode/stbydifficulty.py::stDistOnly` | `behavior/st_only/` |
| **1E** | Mouse accuracy vs coherence, fast vs slow. n=20; fast 20,864 / slow 20,054 trials. Paired *t*, Holm–Bonferroni: 10% *p*=0.1287, 20% *p*<0.001, 50% *p*=0.003, 100% *p*<0.001 | `behavior.ipynb` "Figure 1E" | `figcode/psychometric.py::slowFastPsych` (+ vendored `figcode/psychofit/`) | `behavior/psych_mice/`, `behavior/slow_fast_perf.svg` |
| **1F** | Human speed-context accuracy. n=18; legend gives 8,306 / 8,303 trials. Paired *t*: 3% *p*=1, 8% *p*=1, 14% *p*=1, 33% *p*=0.0107, 75% *p*=0.0108 | `behavior.ipynb` "Figure 1F" | `figcode/psychometric.py::slowFastPsych` | `behavior/psych_human/` |
| **1G** | Mean z-scored sampling time (correct trials) vs difficulty, fast vs slow. Mice n=20, 15,461 trials: fast slope 0.02 (θ=1.08°), slow slope 0.14 (θ=7.83°). Human speed n=18, 8,185 trials: fast slope 0.06 (θ=3.70°), slow slope 0.74 (θ=36.52°) | `behavior.ipynb` "Plot Fast/Slow Reaction-Time vs Difficulty — Figure 1G (left)", "Figure 1G (right)" | `figcode/stbydifficulty.py::stVsDiffOnly`, `::loopstVsDiffOnly` | `behavior/st_vs_diff_fast_slow/`, `behavior/humans_st/` |
| **1H** | Mean z-scored sampling time vs reward rate (5-trial sliding window). Mice n=20, humans (speed) n=18 | `behavior.ipynb` "Figures 1H, S3B–S3D" | `behavior/rewardrate.py::calcAvgRewardRate`, `::loopRewardRateAnalysis` | `behavior/reward_rate5/` |
| **1I left** | % previous-correct, fast vs slow. **p = 0.007**, paired *t*, n=20 mice (20,864 / 20,054 trials) | `behavior.ipynb` "Figure 1I (left)" | `figcode/prevoutcomecurquantile.py::prevOutcomeCurQuantile` | `behavior/prev_choice_by_quantile.svg` |
| **1I right** | Win-stay/lose-switch update, fast vs slow. **p = 0.0046**, paired *t*, n=20 (20,851 / 20,044 trials) | `behavior.ipynb` "Figure 1I (right)" | `behavior/stayswitchupdate.py::calcWinLoseUpdates`, `::plotQuantileUpdate` | `behavior/QuantileWinLoseUpdate/` |

---

## Figure 2 — RL-enhanced drift-diffusion model

> *RL-enhanced DDMs capture adaptive behavioral strategies.*

| Panel | Content & reported numbers | Notebook | Backend | `results/` |
|---|---|---|---|---|
| **2A** | Variance explained per predictor, n=9 mice: coherence **27.15% ± 6.87**, reward history **34.36% ± 7.94**, motor bias **21.02% ± 5.82**, win-stay **19.32% ± 5.46** | `behavior.ipynb` "Figure 2A" | `behavior/varexplained.py` | `behavior/model_OLS_var_explained_w_filter.svg` |
| **2B** | Reward-rate optimum vs observed sampling time. Left: example mouse ("No. 5" in the figure = **BVGAT1**). Right: 17 animals aligned to each animal's optimum, mean ± SD | `behavior.ipynb` "Figure 2B" (every mouse: "Figure 2B left, every mouse (full run only)") | `behavior/optimalsampling.py` | `behavior/optimal_sampling/subj__BVGAT1.svg`, `behavior/optimal_sampling/all_subj_aligned.svg` |
| **2C** | Schematic of the RL-DDM family | — | *not from this repo* | — |
| **2D** | Schematic: Q-learning → starting point (z); R-learning → noise amplitude | — | *not from this repo* | — |
| **2E** | Reward rate vs z-scored sampling time, example mouse **Avgat3** vs model variants. Drawn from the seed-0 simulation frames in `data/RLModel/metrics/simcache_fig2g.pkl`, so a normal run draws it | `rlmodel/rlmodel_analysis.ipynb` "Figure 2E — example mouse Avgat3" (every subject: "Reward rate — pooled and every subject (full run only)") | `rlmodel/model/subjectfits.py::plotRewardRates` | `RLModel/RewardRate/subject_Avgat3.svg` |
| **2F** | Psychometric fast/slow, example mouse **Avgat3** plus DDM+QL+RL fit | `rlmodel/rlmodel_analysis.ipynb` "Figures 2F and S4A — example mouse Avgat3" (every subject: "Model fits — every subject (full run only)") | `rlmodel/model/subjectfits.py::plotSubjectFits`, `rlmodel/model/plotter.py` | `RLModel/figs/Avgat3/RewardRate + Q-Val.svg` |
| **2G** | Aggregate model comparison: psychometric R² ("Psy") and reward-rate Pearson r ("Rew"), 100 simulation seeds per subject × model | `rlmodel/rlmodel_analysis.ipynb` "Figure 2G" | `rlmodel/model/aggregate.py`, `rlmodel/model/aggregate_plot.py`; cluster path `metrics_runner.py` → `model/metrics_shards.py` | `RLModel/aggregates_R2_bar.svg` |

---

## Figure 3 — Widefield imaging and cortex-wide optogenetics

> *Spatiotemporal activation of frontal cortex regions causally relates to fast
> and slow decision-making strategies.*

| Panel | Content & reported numbers | Notebook | Backend | `results/` |
|---|---|---|---|---|
| **3A** | Widefield setup schematic | — | *not from this repo* | — |
| **3B** | Pixel-wise ΔF/F maps in three windows of a typical (1.0–1.2 s) trial, Allen atlas outlines. ✗ not drawn: the maps are rebuilt from each session's raw widefield SVD components, which are too large to share in this repository; contact the authors if you are interested | — | — | — |
| **3C** | Mean ΔF/F per area (V1, PPC, M1, M2), 21 sessions / 6 mice, 1.0–1.2 s trials; inset = the average of 6 of those sessions from 4 mice (`inset_sessions` in the notebook; the legend calls it the example mouse in 3B) | `widefield.ipynb` "Figure 3C (main)", "Figure 3C (inset)" | `common/plottracesavg.py::plotNormalized` | `WF/standard_map/RDK/ReactionTime/WF10_Ncon1/All/WF10_Ncon1_all_Typical_Subset_Areas_Clrs2_Comb_ZScore.svg` (main) and `…/WF10_Ncon1/All/WF10_Ncon1_inset_Typical_Subset_Areas_Clrs2_Comb_ZScore.svg` (inset) (standard-map run) |
| **3D** | Opto silencing during 1 s fixed sampling. **MFC 28.90% ± 4.5**, **LFC 21.53% ± 6.27** performance drop. Hierarchical bootstrap 10,000 iterations + Holm–Bonferroni: V1 *p*=0.108 (n=7), PPC *p*=0.021 (n=6), MFC *p*<0.001 (n=7), LFC *p*=0.01 (n=7) | `opto.ipynb` "Figure 3D" | `opto/optoprocessor.py::plotOptoEffect`, `opto/bootstrap2regions.py::bootstrapSignTestApproach2`, `opto/bootstrapping.py::bootstrapPerf` | `optogenetics/Performance/` |
| **3E** | MFC/LFC segmentation; z-scored traces for fast / typical / slow (n=6 mice, 30 sessions); δ arrow at end of sampling. The heatmap insets ✗ are not drawn (as 3B: raw SVD data not shared) | `widefield.ipynb` "Figure 3E" | `widefield/pipelineprocessors.py`, `common/plottracesavg.py` | `WF/redefined_map/RDK/ReactionTime/…/…_Q{1,2,3}_Comb_ZScore.svg` |
| **3F** | δ (LFC−MFC) at end of sampling: **fast 0.32 ± 0.06, typical 0.56 ± 0.06, slow 0.63 ± 0.06**. RM-ANOVA + Holm–Bonferroni: fast-vs-typical *p*<0.0001, fast-vs-slow *p*<0.0001, typical-vs-slow *p*=0.1245. n=30 sessions / 6 mice | `widefield.ipynb` "End of Sampling — Figure 3F" | `widefield/mfclfcquantiles.py` (`SamplingAnchor.END`, + tests) | `WF/redefined_map/MFC_LFC_dist_by_session.svg` |

---

## Figure 4 — Temporal opto dissection and two-photon sequences

> *Optogenetic inhibition and two-photon imaging reveal distinct temporal
> dynamics and neuronal activation sequences in MFC and LFC.*

| Panel | Content & reported numbers | Notebook | Backend | `results/` |
|---|---|---|---|---|
| **4A** | Bilateral MFC/LFC inhibition schematic | — | *not from this repo* | — |
| **4B top** | Early (0–350 ms) / late (650–1000 ms) design over a 1 s fixed sample | — | *not from this repo* | — |
| **4B bottom** | Widefield MFC/LFC z-scored ΔF/F under fixed 1 s sampling, n=4 mice, 12 sessions | `widefield.ipynb` "Figure 4B (background)" | `common/plottracesavg.py` | `WF/redefined_map/RDK/FixedTime/…_QFixedTime_Comb_ZScore.svg` |
| **4C** | Early/late inhibition performance drop. Early: **MFC 14.42% ± 1.43** (n=7), **LFC 7.36% ± 4.25** (n=10). Late: **MFC 22.03% ± 5.92** (n=5), **LFC 15.07% ± 3.37** (n=8) | `opto.ipynb` "Figure 4C" | `opto/optoprocessor.py`, `opto/bootstrap2regions.py` | `optogenetics/Performance/opto_effect_All_Mice_MFC_LFC.svg` |
| **4D** | 2P setup schematic, example field of view, three example ΔF/F traces. Dataset: n=6 mice, 23 sessions; **837/2,340 active neurons in MFC, 609/1,536 in LFC**. ✗ the three example traces: no drawing cell found | — | schematic and micrograph *not from this repo* | — |
| **4E** | Mean z-scored L2/3 population activity (solid) with widefield overlay (dashed), MFC/LFC × fast/slow, n=6 mice | `2p_1_population.ipynb` "Actually plot — Figure 4E", "Population activity sum — Figures 4E and S11A (left)" | **inline** `sumActivity`, `activitySumTake3`, `plotSum`; `twop/plot/activitysum.py::plotActivitySum` | `2P/activity_sum/`, `2P/activity_sum2/` |
| **4F** | Single-trial active (green) / inactive (grey) traces plus peak mini-heatmap. Example MFC neuron **GP4_28_S2_L51_D250_M2L #206** (43% active on fast trials, 31% on slow) | `2p_2_sequences.ipynb` "Figure 4F — example MFC neuron GP4_28_S2_L51_D250_M2L #206" (every neuron: "Every neuron (full run only)") | `twop/fastslowneurontraces.py` | `2P/FastSlowTraces/MFC/GP4_28_S2_L51_D250_M2L_206.svg` (all neurons: `2P/FastSlowTraces/{MFC,LFC}/`) |
| **4G** | Venn of ≥10%-active neurons: fast, slow, both — MFC (top), LFC (bottom) | `2p_2_sequences.ipynb` "Overlap between neurons active on fast and slow trials — Figure 4G" | `twop/fastslowvenn.py::loopBRPlotVennOverlap` | `2P/FastSlowVenn/` |
| **4H** | Population heatmaps fast vs slow, ordered by the all-trials peak (example MFC session `GP4_28_S2_L51_D250_M2L`) | `2p_2_sequences.ipynb` "Figure 4H" | `twop/sessionheatmaps.py::loopHeatmapCombinations`; `twop/plottracesavg.py` | `2P/Sessions/MFC/GP4_28_S2_L51_D250_M2L/summary/` |
| **4I** | Schematic of the rank-deviation method | — | *not from this repo* | — |
| **4J** | Left: observed rank deviation vs a null from permuted strategy labels (example LFC session `GP4_85_s7`, deviation 5.386). Right: z-scored deviation per session. 100,000 iterations/session, Holm–Bonferroni. Unseeded, so the z-scores move in the second decimal between runs | `2p_2_sequences.ipynb` "Seq firing deviation — Figure 4J": "Figure 4J (left) — example LFC session GP4_85_s7_L70_D250_ALM", "Figure 4J (right) — every session" | `twop/seqpermutation.py` (left); **inline** `plotDeviationAgg` (right) | `2P/Sequence/deviation_rank_ALM - GP4_85_s7_L70_D250_ALM_10_prcnt_active.svg`, `2P/Sequence/deviation_summary_zscore_10_prcnt_valid.svg` |
| **4K left** | Example "rigid" and "stretching" neurons, traces coloured by sampling duration. ✗ not drawn: every `loopNeuronsPlot` call passes `plot=False, save_figs=False` (saving would also fail: `2P/SingleTraces/rt_corr/` is never created), and the two examples' identities were not recorded | `2p_1_population.ipynb` "Correlation between rt and activity on single cell level" | `twop/rtcorrpanels.py::loopNeuronsPlot` (+ tests) | — |
| **4K right** | Pie charts: rigid / peak-timing / AUC / both, per region. Threshold \|r\| ≥ 0.3 | `2p_1_population.ipynb` "Pie-Chart for rigid/stretching neurons — Figure 4K (right)" | `twop/rtcorrpies.py::plotCorrThreshPieChart` | `2P/{MFC,LFC,Both_MFC_LFC}_correlation_pie_chart.pdf` |

---

## Figure 5 — Choice-selective sequences and mid-sampling inhibition

> *Choice-selective neural sequences in MFC and LFC exhibit distinct modes of
> operation, with differential contributions to voluntary decision-making.*

| Panel | Content & reported numbers | Notebook | Backend | `results/` |
|---|---|---|---|---|
| **5A** | Choice-match / choice-mismatch schematic | — | *not from this repo* | — |
| **5B left** | Activation sequence across sampling, example MFC session `GP4_28_S2_L51_D250_M2L` | `2p_3_choice.ipynb` "Figure 5B (left)" | `twop/sessionheatmaps.py::loopHeatmapCombinations`; `twop/plottracesavg.py` | `2P/Sessions/MFC/GP4_28_S2_L51_D250_M2L/summary/` |
| **5B right** | Example neurons labelled #9 (left-preferring), #38 (non-selective), #46 (right-preferring) of that session. The labels are rows of the 5B-left heatmap, not trace ids: they are trace ids **2, 95 and 20**, at rows 9, 38 and 50 of the heatmap as drawn today. #46's preferred (right) curve is brown in the notebook, green in the paper | `2p_3_choice.ipynb` "Figure 5B (right) — neurons labelled #9, #38 and #46 = trace ids 2, 95 and 20 of GP4_28_S2_L51_D250_M2L" (every session: "Every session (full run only)") | `twop/sessionheatmaps.py::loopCombinationsTraces`; `twop/plottracesavg.py`, `twop/plottuning.py` | `2P/Sessions/MFC/GP4_28_S2_L51_D250_M2L/sgf_and_not_sgf/no_quantiles/Direction/{2,95,20}_*.svg` |
| **5C** | % choice-selective neurons per session. **MFC 18.06% ± 3.04 SEM** (837 neurons, 13 sessions), **LFC 33.07% ± 3.68 SEM** (609 neurons, 10 sessions). Per-neuron Mann-Whitney U; across-region Student *t*, **p = 0.00478** | `2p_3_choice.ipynb` "Sgf. Neurons Pie Charts — Figure 5C" | `twop/sgfneurons.py` (+ tests) | `2P/sgf_tests/Cur. Direction.svg` |
| **5D** | Example neuron #9 (trace id 2, see 5B right), matched vs mismatched, slow (top) and fast (bottom). ✗ not drawn: it is the per-neuron Direction plot of 5B right split by strategy, and 2p_3_choice runs that loop only with `split_by_quantile=False` | — | `twop/plottracesavg.py` | — |
| **5E** | Sequence-ordered population response, matched vs mismatched, slow/fast × MFC/LFC. **MFC 13 sessions / 182 neurons; LFC 10 sessions / 197 neurons.** Correct and incorrect trials together (`choice_outcome=None`), the same set as S12E | `2p_3_choice.ipynb` "Current-Trial Left-vs-Right — Figure 5E" | `twop/plottuning.py::TrajectoryTuningPlot` | `2P/RT_Stats/traces/hist_choice_zscored_*_sep_quantiles_combined_choices_traces.svg` |
| **5F** | Coding efficiency `(Match − Mismatch)/Match` at sampling start and decision, MFC vs LFC × slow/fast | `2p_3_choice.ipynb` "Figure 5F" | `twop/plottuning.py::TrajectoryTuningPlot` | `2P/RT_Stats/traces/eff_hist_choice_zscored_*.svg` |
| **5G** | Mid-sampling (0.3–0.9 s) inhibition design plus example-mouse sampling-duration distributions (mouse **vgat-94**) | `opto.ipynb` "Figure 5G" | `opto/optoreactiontime.py::optoReactionTime` | `optogenetics/RT/BySubject/vgat-94_total_hist.svg` |
| **5H** | Cumulative sampling-duration distributions and post-offset performance: control / MFC / LFC, n=7 mice | `opto.ipynb` "Figure 5H" | `opto/optoreactiontime.py` | `optogenetics/RT/Late decision_probability_SEM.svg`, `optogenetics/RT/Late decision_performance_SEM.svg` |

---

## Figure 6 — Feedback in MFC controls strategy update

> *MFC controls feedback-dependent shifts in behavioral strategy.*

| Panel | Content & reported numbers | Notebook | Backend | `results/` |
|---|---|---|---|---|
| **6A top** | Self-initiated task structure (feedback between trial *t* and *t+1*) | — | *not from this repo* | — |
| **6A bottom** | Sampling + feedback heatmaps, MFC / LFC, neurons ordered by sampling-phase peak. The cell draws MFC session `GP4_28_S2_L51_D250_M2L` only; ✗ the LFC heatmap is not drawn by it | `2p_4_feedback.ipynb` "Feedback heatmaps — Figure 6A (bottom)" | `twop/sessionheatmaps.py::avg`; `twop/plottracesavg.py` | not in `results/` |
| **6B** | Feedback-tuned neuron fraction across early (0–0.33) / middle (0.33–0.66) / late (0.66–1) sampling thirds, MFC vs LFC (totals 50.73% / 24.61%) | `2p_4_feedback.ipynb` "Feedback percentage and distribution — Figure 6B" | **inline** `plotFeedbackDist` | `2P/feedback_neurons_dist_andfiring_pos.svg` |
| **6C** | Example feedback-tuned neuron (`45_GP4_28_S1_L41_D250_M2L`) multiplexing previous outcome × current choice: trace over sampling and feedback (bottom), "Prior" inset (top left), "Current" inset (top right) | `2p_4_feedback.ipynb` "Figure 6C (bottom)", "Figure 6C (top left)", "Figure 6C (top right)" | `twop/sessionheatmaps.py::avg`; `twop/plottracesavg.py` | `2P/Sessions/MFC/GP4_28_S1_L41_D250_M2L/sgf_and_not_sgf/no_quantiles/{PrevChoiceCurDir,PrevChoiceCorrect,Direction}/45_*.svg` |
| **6D** | Feedback-window inhibition → next-trial z-scored sampling time, post-correct vs post-incorrect, n=4 mice. Within-condition paired *t*: control *p*=0.2591, MFC *p*=0.5534, LFC *p*=0.1627. Cross-region post-correct (RM-ANOVA *p*=0.0008): control-vs-MFC *p*=0.0218, control-vs-LFC *p*=0.4495, MFC-vs-LFC *p*=0.0218. Cross-region post-incorrect (RM-ANOVA *p*=0.0373): all pairwise *p* ≥ 0.2176 | `opto.ipynb` "Figure 6D" | `opto/optofeedback.py::optoFeedback` | `optogenetics/feedback/` |
| **6E** | Venn: MFC tuning to priors vs current-trial variables, sampling vs feedback epochs. **Priors during sampling: 80.1% ± 2.8 SEM** | `2p_4_feedback.ipynb` "Sampling & Feedback Venn Diagrams — Figure 6E" | `twop/priorcurtuning.py::plotPriorCurrentTuning` | `2P/PriorCurrentTuning/` |

---

## Figure 7 — Computational and neural framework

| Panel | Content & reported numbers | Notebook | Backend | `results/` |
|---|---|---|---|---|
| **7A** | % neurons per session correlated (\|r\| ≥ 0.3) with model latents. **Q-value (side bias) 10.38% ± 2.68 SEM; R-value (reward rate) 8.89% ± 2.58 SEM.** n=22 sessions. Per-neuron activity-shuffled null (1,000 shuffles), Holm–Bonferroni. Fit: joint MLE+χ² with w_MLE=1, w_χ²=0.5 | `rlmodel/rlmodel_neural_correlate.ipynb` "Factor modulation bars (Q / Reward-Rate / DV) — Figures 7A, 7B, S14C and S14D" | `rlmodel/model/neural_correlate.py::plot_factor_bars_fastslow_dv` (+ tests) | `RLModel/neural_correlate/<model>/factor_bars/bars_fastslowDV_combined.svg` |
| **7B** | DV-correlated neurons, fast vs slow. **Fast 2.07% ± 0.52; slow 4.56% ± 1.1** per session. Two-sided session-paired sign-flip permutation, 1,000 iterations | as 7A (same figure); also "Fast-vs-slow summary bars — Figure 7B" | `rlmodel/model/neural_correlate.py` | as 7A; `…/fast_slow_bars/DV/bars_combined.svg` |
| **7C** | MFC ("strategy") → LFC ("action") schematic | — | *not from this repo* | — |
| **7D** | 3D landscape: z-scored sampling time vs relative Q-value × reward rate, one surface per difficulty | `rlmodel/rlmodel_to_behavior.ipynb` "Figure 7D" | **inline** `plotQ_R_Heatmap`; simulation through `rlmodel/model/qrsurface.py` | `RLModel/Q_R_Heatmap.svg` |
| **7E** | Cellular-implementation schematic (slow vs fast) | — | *not from this repo* | — |

---

## Supplementary figures

### Figure S1 — Floating Reality platform

| Panel | Content | Notebook | Backend |
|---|---|---|---|
| S1A | Side / top / bottom 3D views of the nose-poke platform | — | *not from this repo* |
| S1B | Free-sampling task structure; example-mouse (M#22) psychometric across 32 sessions / 8,696 trials; sampling time vs coherence by choice side. ✗ not drawn: the only psychometric call (`behavior.ipynb` "Psychometric (overall) - Humans & Mice") draws group curves, not one mouse by session, and nothing draws sampling time vs coherence by side | — | — |
| S1C | Fixed-variable-sampling task; example-mouse (M#4) psychometric (9,450 trials) ✗ not drawn; chronometry curve, n=3 mice | `behavior.ipynb` "Figure S1C (right)" | `behavior/chronometry.py::chronometry` |
| S1D | Freely-moving vs head-fixed KDEs for one mouse: freely-moving light-chasing 963 trials, head-fixed light-chasing 1,089, head-fixed RDK 2,455 | `behavior.ipynb` "Figure S1D (right)" | `behavior/stkde.py::plotFMHFSubjects` |

`results/behavior/All Animals_chrono.svg`, `behavior/fm_hf/`.

### Figure S2 — Fast/slow strategies in humans and mice

| Panel | Content & numbers | Notebook | Backend |
|---|---|---|---|
| S2A | Human RDK task schematic (Accuracy / Speed contexts) | — | *not from this repo* |
| S2B | z-scored sampling-time distributions across contexts (z-scored across both contexts). The notebook figure also draws a mouse curve and a violin panel | `behavior.ipynb` "Plot user data — Figure S2B" | `behavior/stdistribution.py::plotSTDistribution` → `results/behavior/humans_mice_reaction_time_dist.svg` |
| S2C | Dispersion (IQR). Shapiro-Wilk → one-way ANOVA (*p*<0.001) + Tukey HSD: accuracy-vs-speed *p*=0.001, accuracy-vs-mice *p*=0.001, speed-vs-mice *p*=0.0114. n=18/18/20 | `behavior.ipynb` "Per-subject z-scored sampling time & subject-sigma dispersion — Figure S2C" | `behavior/stdispersion.py::plotSTDispersion` (+ tests) |
| S2D | Raw-seconds distributions, human speed (n=18) vs mice (n=20) | `behavior.ipynb` "All RDK mice vs all human speed-context trials — Figure S2D" | `behavior/stkde.py::plotMiceVsHumansST` → `results/behavior/stimulus_time_mice_vs_humans.svg` |
| S2E–G | Example mouse M#85 (3,519 trials): histograms by difficulty (E), correct/incorrect sampling time by difficulty (F), fast/slow psychometric (G) | `behavior.ipynb` "Figure S2E", "Figure S2F", "Figure S2G" | `figcode/stbydifficulty.py`, `figcode/psychometric.py` |
| S2H–I | Human speed (n=18): z-scored sampling time by difficulty (H), correct-vs-incorrect (I) | `behavior.ipynb` "Figure S2H", "Figure S2I" | as above |
| S2J | Human accuracy fast/slow psychometric. Paired *t* + Holm: 1% *p*=0.4858, 3% *p*=0.061, 8% *p*=0.0168, 14% *p*=0.0317, 32% *p*=0.9228, 77% *p*=0.4858. n=18 (2,951 / 2,537 trials) | `behavior.ipynb` "Figure S2J" | `figcode/psychometric.py::slowFastPsych` |
| S2K–L | Human accuracy (n=18): histograms (K), correct-vs-incorrect (L) | `behavior.ipynb` "Figure S2K", "Figure S2L" | `figcode/stbydifficulty.py` |
| S2M | Easy-trial accuracy fast vs slow across the three groups. Paired *t* + Holm: human accuracy *p*=0.165, human speed *p*=0.006, mice *p*<0.001 | `behavior.ipynb` "Figure S2M" | `behavior/fastslowperf.py::collectAllGroups`, `::plotFastSlowAccuracy` |

`results/behavior/humans_mice_sampling_time_dispersion_iqr.svg`, `…_std.svg`.

### Figure S3 — Trial history, sensory evidence, and posture

| Panel | Content & numbers | Notebook | Backend |
|---|---|---|---|
| S3A | Human accuracy: z-scored sampling time of correct trials vs difficulty. Fast slope 0.45 (θ=24.19°), slow slope 0.82 (θ=39.26°) | `behavior.ipynb` "Figure S3A" | `figcode/stbydifficulty.py::stVsDiffOnly` |
| S3B | Mice sampling time vs reward rate, n=20. Shapiro-Wilk (2/5 bins non-normal) → Kruskal-Wallis **p=0.0001** + Dunn/Holm | `behavior.ipynb` "Figures 1H, S3B–S3D" | `behavior/rewardrate.py::loopRewardRateAnalysis` |
| S3C | Human speed, n=18. All bins normal → one-way ANOVA **p=0.0006** + Tukey | as above | as above |
| S3D | Human accuracy, n=18. 1/4 bins non-normal → Kruskal-Wallis **p=0.017** + Dunn/Holm | as above | as above |
| S3E | Heatmap of mean sampling time by previous outcome × current difficulty. Left: example mouse M#46. Right: all mice, z-scored; the notebook draws it from n=20 subjects / 262 sessions / 63,668 trials | `behavior.ipynb` "Figure S3E (left)", "Figures S3E (right) and S3F" | `figcode/stheatmap.py::stHeatmap` |
| S3F | Violin plots of the same. Kruskal-Wallis **p<0.001** + Dunn/Holm. Same trials as S3E right | `behavior.ipynb` "Figures S3E (right) and S3F" | as above |
| S3G | Win/lose update vs z-scored sampling time, **n=18 mice** (20 minus 2 excluded: RDK_WT1, RDK_WT6, with 18.62% and 13.26% of trials below z=−1 vs a group mean of 3.28%). Bins of 800 trials, Gaussian σ=1 bin | `behavior.ipynb` "Figure S3G" | `behavior/stayswitchupdate.py::outlierSubjects`, `::plotUpdateOverSamplingTime` |
| S3H | Win/lose update by current difficulty, computed per mouse then averaged | `behavior.ipynb` "Figure S3H" | `figcode/stayswitch.py::staySwitchUpdate` |
| S3I | Choice bias, fast vs slow. Paired *t*, **p = 0.016**, n=20 | `behavior.ipynb` "Figure S3I" | `behavior/bias.py::plotBias` |
| S3J | Per-trial centroid rotation-angle distance travelled, by strategy. Computed as the within-trial **range** (`max − min`) | `tracking.ipynb` "Figures S3J and S3K" | `tracking/centroids.py::plotCentroids` (+ tests) → `results/tracking/distance_All Tracked Subjects.svg` |
| S3K | Polar histogram of mean rotation angle, 5° bins, **un-normalised** (`choice_normed=False`) | `tracking.ipynb` "Figures S3J and S3K" | `tracking/centroids.py::plotCentroids` (+ tests) → `results/tracking/centroid_rotation_All Tracked Subjects.svg` |
| S3L | Same, relative to each **session's** preferred side | `tracking.ipynb` "Figure S3L" | `tracking/centroids.py::plotCentroids(preferred_side_normed=True)` (+ tests) → `results/tracking/centroid_rotation_preferred_All Tracked Subjects.svg` |
| S3M | Per-animal Kruskal-Wallis on \|trial mean angle − the animal's modal angle\| across strategies, Holm-corrected across the 4 mice. **0/4 significant** | `tracking.ipynb` "Figure S3M" | `tracking/strategy.py::plotStrategyComparison` (+ tests) → `results/tracking/strategy_dist_sgf.svg` |

`results/behavior/StaySwitch/`, `behavior/bias/abs_all_bias.svg`, `results/tracking/`.

### Figure S4 — Model variants

| Panel | Content | Notebook | Backend |
|---|---|---|---|
| S4A | Example-mouse (**Avgat3**) fast/slow psychometric vs DDM, DDM+QL and DDM+RL | `rlmodel/rlmodel_analysis.ipynb` "Figures 2F and S4A — example mouse Avgat3" | `rlmodel/model/subjectfits.py::plotSubjectFits` → `results/RLModel/figs/Avgat3/{Classic,Q-Val,RewardRate}.svg` |
| S4B | "Psy"/"Rew" correlations for the four reward-rate channels: noise, bound, drift (+), drift (−). The "Drift Rate (+)" column is the code's `RR (Drift 2-r)`, μ = V·DV·(2−r); "Drift Rate (−)" is `RR (Drift 1+r)`, μ = V·DV·(1+r) | `rlmodel/rlmodel_analysis.ipynb` "Model Comparison: reward-rate channel — Figure S4B" | `rlmodel/model/aggregate.py`; channels in `model/drift.py` → `results/RLModel/aggregates_R2_drift_rr.svg` |

### Figure S5 — Widefield, typical trials

| Panel | Content | Notebook | Backend |
|---|---|---|---|
| S5A | Widefield setup plus clear-skull photographs | — | *not from this repo* |
| S5B | Session selection for typical (1000–1200 ms) trials, coloured by mouse | `widefield.ipynb` "Figure S5B" | `widefield/pipelineprocessors.py` → `results/WF/<map>_map/typical_trials_rt_distribution.svg` (the same figure under either map) |
| S5C | Example-mouse pixel-wise maps in successive 200 ms windows. ✗ not drawn (as 3B: rebuilt from the raw SVD components, too large to share; contact the authors) | — | — |
| S5D | Trial-averaged area traces, example mouse M_WF4 (`widefield_4` in the data), session WF4_M13, z-scored over its typical trials only (standard-map run) | `widefield.ipynb` "Figure S5D" | `common/plottracesavg.py` → `results/WF/standard_map/RDK/ReactionTime/WF4_M13/All/WF4_M13_all_Typical_Many_Areas_Clrs2_ZScore.svg` |
| S5E | Cross-mouse area traces, n=6 mice / 21 sessions (standard-map run) | `widefield.ipynb` "Figure S5E" | `common/plottracesavg.py` → `results/WF/standard_map/RDK/ReactionTime/WF10_Ncon1/All/WF10_Ncon1_all_Typical_Many_Areas_Clrs2_Comb_ZScore.svg` |

### Figure S6 — Fixed-sampling optogenetics per region

| Panel | Content & numbers | Notebook | Backend |
|---|---|---|---|
| S6A | Bilateral inactivation schematic (6 mW/mm², 40 Hz, clear skull) | — | *not from this repo* |
| S6B–E | Control vs opto psychometrics: V1 (B), PPC (C), MFC (D), LFC (E) | `opto.ipynb` "Figures S6B–S6E" | `opto/optoprocessor.py::plotOptoPsych` → `results/optogenetics/Psychometric/fixedtime_full_inhib/` |
| S6F | Effect on motor initiation, MFC / LFC / control (z-scored decision time). **MFC: no significant delay; LFC: significant delay.** | `opto.ipynb` "Decision time, subject-paired bars — Figure S6F" | `opto/optoreactiontime.py` → `results/optogenetics/DecisionInitiation/_mt_bars_z_scored_only_after_opto.svg` |
| S6G | No-opsin control (Thy1-GP4) MFC inhibition. Hierarchical bootstrap 10,000 iterations, **p = 0.387**, n=3 mice (230 opto / 1,024 control) | `opto.ipynb` "Figure S6G" | `opto/optoprocessor.py`, `opto/bootstrap2regions.py` → `results/optogenetics/Performance_Control/` |

### Figure S7 — Frontal dynamics and timed-inhibition psychometrics

| Panel | Content | Notebook | Backend |
|---|---|---|---|
| S7A | MFC/LFC widefield traces under fixed 1 s sampling, example mouse | `widefield.ipynb` "Figure S7A" | `common/plottracesavg.py` |
| S7B | Same for free-sampling trials of 1–1.2 s, same mouse | `widefield.ipynb` "Figure S7B" | `common/plottracesavg.py` |
| S7C–D | Early-inhibition (0–350 ms) psychometrics, MFC (C) and LFC (D) | `opto.ipynb` "Figures S7C–S7F" | `opto/optoprocessor.py::plotOptoPsych` → `results/optogenetics/Psychometric/fixedtime_early_inhib/` |
| S7E–F | Late-inhibition (650–1000 ms) psychometrics, MFC (E) and LFC (F) | as above | as above → `…/fixedtime_late_inhib/` |

### Figure S8 — L2/3 population dynamics

These rows attribute the panels to the notebook and modules that draw this
kind of figure; they were not confirmed cell by cell.

| Panel | Content | Notebook | Backend |
|---|---|---|---|
| S8A | Schematic: single-neuron traces → normalised population activity | `2p_1_population.ipynb` | `twop/plot/activitysum.py::plotActivitySum` |
| S8B | Normalised MFC/LFC population responses, fast / typical / slow, example mouse | `2p_1_population.ipynb` | `twop/plottracesavg.py` |
| S8C | Widefield vs two-photon, aligned to motor initiation | `2p_1_population.ipynb` | `twop/plot/activitysum.py` |

### Figure S9 — Sequence organisation

| Panel | Content & numbers | Notebook | Backend |
|---|---|---|---|
| S9A | Single-neuron traces; decision-probability distributions; mean z-scored ΔF/F aligned to sampling start | `2p_0_preprocessing.ipynb` "Max-firing tables" | `twop/tracereliability.py` |
| S9B | CDF of trial-by-trial firing reliability, MFC/LFC × fast/slow | `2p_2_sequences.ipynb` "CDF for active trials per quantiles — Figure S9B" | `twop/activetrialscdf.py::plotActiveDistCDF` → `results/2P/active_cdf/` |
| S9C | LFC example-session heatmap, fast vs slow, aligned by the average peak across trials. Example session `GP4_23_s8_L50_D250_ALM` (94 neurons); the published row order differs from the current sorting | `2p_2_sequences.ipynb` "Figure S9C — LFC example session GP4_23_s8_L50_D250_ALM" (every session: "All sessions — Figures 4H and S9C (full run only)") | `twop/sessionheatmaps.py::loopHeatmapCombinations` → `results/2P/Sessions/LFC/GP4_23_s8_L50_D250_ALM/summary/hm_All__q{1,3}_GP4_23_s8_L50_D250_ALM_ALM_sorted_to_all_shared_rng.svg` (q1 fast, q3 slow) |
| S9D | Trial-by-trial rank variability for LFC with **slow** trials as reference, against incremental reference-rank shuffling (0% reproducible, 50% random, 100% reversed) | `2p_2_sequences.ipynb` "F3. The F calibration, recentered so random = 50% (matches F2) — Figure S9D" | `twop/seqdeviation.py` (+ tests), `twop/shuffle_replay.py` (interactive replay) |
| S9E | Within vs cross strategy. **MFC Fast-ref: 16.09% ± 1.02 within / 21.18% ± 2.03 cross. MFC Slow-ref: 15.92% ± 1.22 / 20.78% ± 1.75. LFC Fast-ref: 15.32% ± 1.25 / 20.13% ± 1.73. LFC Slow-ref: 17.32% ± 0.92 / 18.99% ± 1.88** | `2p_2_sequences.ipynb` "F2. "Same-sequence?" panels on the shuffle-level axis (random = 50%) — Figure S9E" | `twop/seqdeviation.py` |
| S9F | Schematic of the permutation approach against a "typical"-trial reference | — | *schematic* |
| S9G | Rank mismatch vs permuted typical trials, 100,000 iterations/session. Left: example MFC session **GP4_28_S6_L40_D250_M2L**. **MFC: 13/13 fast and 13/13 slow at p≤0.001. LFC: 10/10 fast, 9/10 slow at p≤0.001, 1/10 p>0.05**. Unseeded | `2p_2_sequences.ipynb` "Permutation test of the sequence against typical trials — Figure S9G": "Figure S9G (left) — example MFC session GP4_28_S6_L40_D250_M2L", "Figure S9G (right) — every session" | `twop/seqpermutation.py` (left); **inline** `plotPenalityCost` (right) | `results/2P/Sequence/seq_rank_M2 - GP4_28_S6_L40_D250_M2L_Permutation_ Z-Scored - _10_prcnt_active.svg`, `results/2P/Sequence/penality_sgf_summary_Permutation_normalized_10_prcnt_valid.svg` |

### Figure S10 — Single-neuron modulation by sampling duration

| Panel | Content | Notebook | Backend |
|---|---|---|---|
| S10A | Example neuron aligned to sampling onset: traces, peak-time correlation, AUC correlation. ✗ not drawn (as 4K left) | `2p_1_population.ipynb` "Correlation between rt and activity on single cell level" | `twop/rtcorrpanels.py::loopNeuronsPlot` |
| S10B | Second example, aligned to movement onset. ✗ not drawn (as 4K left) | as above | as above |
| S10C | Correlation-coefficient distributions against 1,000 shuffled controls: AUC early (<0.2 s), AUC late (>0.2 s), peak timing late | `2p_1_population.ipynb` "Correlation Hists — Figure S10C" | `twop/rtcorrhists.py::plotCorrDistributions` |
| S10D | Proportion of neurons per session with \|r\| ≥ 0.3 for AUC and/or peak timing, MFC vs LFC | `2p_1_population.ipynb` "MFC vs LFC bars at the same correlation threshold — Figure S10D" | `twop/plot/corrthreshregions.py` (+ tests) → `results/2P/rt_corr_regions_bars_above_0.3.svg` |

### Figure S11 — Population tuning to sampling time and performance

| Panel | Content | Notebook | Backend |
|---|---|---|---|
| S11A left | Mean z-scored population activity aligned to sampling onset, MFC/LFC × fast/slow | `2p_1_population.ipynb` "Population activity sum — Figures 4E and S11A (left)" | **inline** `plotSum` |
| S11A middle, right | % active neurons in the early window vs sampling duration (hard trials); early activity vs performance (easy trials) | `2p_1_population.ipynb` "Plot Active Early then Late — Figures S11A (middle, right) and S11B (right)" | `twop/activeneuronplots.py` → `results/2P/rt_vs_early_active_neurons_corr.svg`, `…_perf_rel_corr.svg` |
| S11B left | Decision-aligned population activity. ✗ no drawing cell found | — | — |
| S11B right | Same for the last 30% of sampling up to +0.1 s. **MFC negatively correlated with duration; LFC flat** | as S11A middle | `twop/activeneuronplots.py` → `results/2P/rt_vs_last_active_neurons_corr.svg` |

### Figure S12 — Choice-selective population dynamics

| Panel | Content & numbers | Notebook | Backend |
|---|---|---|---|
| S12A | MFC example-session heatmaps for rightward vs leftward choices. ✗ no drawing cell found | — | — |
| S12B | Proportion of choice-selective neurons over normalised sampling time. ✗ no drawing cell found | — | — |
| S12C | % choice-encoding neurons at sampling start, fast vs slow. ✗ no drawing cell found | — | — |
| S12D | Matched/mismatched subgroup activity at the start and end of sampling. ✗ no drawing cell found | — | — |
| S12E | As Figure 5E with single-neuron grey lines, all trials | `2p_3_choice.ipynb` "Figure S12E" | `twop/plottuning.py::TrajectoryTuningPlot` → `results/2P/RT_Stats/traces/` |
| S12F | Coding efficiency for all trials, with single-neuron points | `2p_3_choice.ipynb` "Figure S12F" | as above |
| S12G | Logistic-regression choice decoder at sampling start/end, fast vs slow. 1,000 random 70/30 splits | `2p_3_choice.ipynb` "Decoder accuracy across epochs — Figure S12G" | `twop/decoderplots.py::runAccuracyAcrossEpochs`, vendored `twop/relogit/` → `results/2P/decoders/` |
| S12H | Coding efficiency across the sequence, slow (solid) vs fast (dashed) | `2p_3_choice.ipynb` "Figure S12H" | as S12E |
| S12I | Linear fits of matched vs mismatched activity across bins | `2p_3_choice.ipynb` "Figure S12I" | as S12E |
| S12J left | Movement-tuned neurons at the end of sampling: **MFC 5.10% ± 1.02** (38/837 neurons, 13 sessions), **LFC 11.85% ± 1.50** (68/609 neurons, 10 sessions), Student *t* **p = 0.000941** | `2p_3_choice.ipynb` "Movement-time neurons: MFC vs LFC — Figure S12J (left)" | `twop/movementneurons.py` (+ tests) → `results/2P/MovementNeurons/last_two_bins_prcnt_movement_choice.svg` |
| S12J right | Matched and mismatched responses among movement-tuned neurons. ✗ no drawing cell found | — | — |

### Figure S13 — Mid-sampling inhibition detail

| Panel | Content | Notebook | Backend |
|---|---|---|---|
| S13A | Performance drop under MFC and LFC inhibition | `opto.ipynb` "Figures S13A and S13B" | `opto/optoreactiontime.py` |
| S13B | Decision probability inside the 0.3–0.9 s inhibition window | as above | as above |
| S13C | Cumulative trial distribution over z-scored sampling time | `opto.ipynb` "Figures S13C and S13D" | as above |
| S13D | Performance vs z-scored sampling time; linear fits | as above | as above |
| S13E | Median z-scored sampling time, all trials | `opto.ipynb` "Figures S13E–S13G" | as above |
| S13F | Mean performance after inhibition offset (`AFTER_OPTO_END_DECISION_PERF`, `_perf_bars_perf_only_after_opto.svg`) | as above | as above |
| S13G | As S13E, restricted to trials ending after the inhibition window (>0.9 s) | as above | as above |
| S13H | Mean choice accuracy on trials following a feedback-inhibited trial: control / MFC / LFC, n=4 mice | `opto.ipynb` "Next-Trial Performance — Figure S13H" | `opto/optofeedback.py::optoFeedback` → `results/optogenetics/feedback/_perf_bars_raw_raw_comm_cntrl.svg` |

Statistics (A–G): RM-ANOVA on Condition (Control, MFC, LFC), then paired *t*
with Holm–Bonferroni.

### Figure S14 — Feedback history and model–neural correlation

| Panel | Content & numbers | Notebook | Backend |
|---|---|---|---|
| S14A | Feedback ↔ next-trial-sampling correlation vs normalised sampling position. **MFC r = −0.24, R² = 0.06, p = 0.00001; LFC r = −0.16, R² = 0.03, p = 0.03** | `2p_4_feedback.ipynb` "Feedback correlation with next trial — Figure S14A" | **inline** `corrNextTrial` → `results/2P/corr_feedback_next_sampling.svg` |
| S14B | Fitting-criterion comparison: pure χ², pure MLE, joint w_MLE=1 / w_χ²=0.1, joint w_MLE=1 / w_χ²=0.5 | `rlmodel/rlmodel_analysis.ipynb` "Model Comparison: fitting criterion — Figure S14B" | `rlmodel/model/aggregate.py` → `results/RLModel/aggregates_R2_mle_weights.svg` |
| S14C | Figures 7A/7B restricted to MFC (12 sessions) | as 7A | `rlmodel/model/neural_correlate.py` → `…/factor_bars/bars_fastslowDV_MFC.svg` |
| S14D | Same for LFC (10 sessions) | as 7A | as above → `…/factor_bars/bars_fastslowDV_LFC.svg` |
| S14E | Example neuron modulated by the R-value: neuron 13 of MFC session `GP4_80_S4_L50_D250_M2m` (figure: r = +0.323, bins n = 6/35/87/105; the 4.8 s refit gives r = +0.320, n = 18/42/48/125 — the same 233 trials, a count no other session has) | `rlmodel/rlmodel_neural_correlate.ipynb` "Figure S14E — example neuron 13 of MFC session GP4_80_S4_L50_D250_M2m, by R-value" | `rlmodel/model/neural_correlate.py::plot_neuron_param_traces` (`only=`) → `…/param_traces/RewardRate/+0.320_GP4_80_S4_L50_D250_M2m_13_MFC.svg` |
| S14F | Example neuron modulated by the Q-value (figure: r = +0.372, bins n = 23/22/65/43/80). Same session as S14E (`GP4_80_S4_L50_D250_M2m`, by its 233 trials). ✗ neuron not identifiable under the current fit, whose Q-value bins hold different trials; drawn only by the per-neuron loop when its \|r\| clears 0.3 | `rlmodel/rlmodel_neural_correlate.ipynb` "Every tuned neuron, every parameter (full run only)" | `rlmodel/model/neural_correlate.py::plot_neuron_param_traces` → `…/param_traces/Q_val/` |
| S14G | Neurons correlated with the decision variable under fast and/or slow strategies (n=70, mean \|r\| 0.190 vs 0.302) | `rlmodel/rlmodel_neural_correlate.ipynb` "Paired per-neuron correlation strength — Figure S14G" | `rlmodel/model/neural_correlate.py::paired_neuron_r` → `…/paired_neuron_r/DV_fast_vs_slow.svg` |
| S14H | Example neuron: neuron 18 of MFC session `GP4_85_S1_L70_D250_mm2`, fast r = 0.11 vs slow r = 0.55 | `rlmodel/rlmodel_neural_correlate.ipynb` "Figure S14H — example neuron 18 of MFC session GP4_85_S1_L70_D250_mm2, by DV" | `rlmodel/model/neural_correlate.py::plot_neuron_fastslow_traces` (`only=`) → `…/drift_traces/DV/f+0.110_s+0.548_GP4_85_S1_L70_D250_mm2_18_MFC.svg` |

`<model>` = `RewardRate - Q-Val (Offset) - Normal(0, 1) - 4.8s - weighted Chi² (MLE=1, Chi²=0.5)`.

---

## Panels with no code path in this repository

Beyond the schematics and photographs listed panel by panel above:

- **Figures 1A–B, 2C–D, 3A, 4A, 4B top, 4D (schematic and micrograph), 4I, 5A,
  6A top, 7C, 7E, S1A, S2A, S5A, S6A, S9F** — illustrations, renders and imaging
  hardware photographs.
- **Supplementary Videos 1–3** — raw behavioural video, not derived from the
  analysis pipeline.
- **Key Resources Table** and **Table S2** — maintained in the manuscript, not
  generated from code.
- **Table S1** (summary of statistical analyses) — assembled by hand from the
  per-panel test output the notebooks print; no single script emits it.
