import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[3]))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.twop.movementneurons import (  # noqa: E402
    DEFAULT_BINS_COUNT,
    FIG_SIZE,
    LFC,
    METHOD_INITIATION_ONSET,
    METHOD_IQR_Q1,
    METHOD_IQR_Q1_TWO_BINS,
    METHOD_LABELS,
    METHOD_LAST_BIN,
    METHOD_LAST_TWO_BINS,
    METHOD_MOVEMENT_EPOCH,
    METHOD_SPECS,
    METRIC_CHOICE,
    METRIC_CHOICE_OF_MOVEMENT,
    METRIC_MOVEMENT,
    METRIC_MOVEMENT_CHOICE,
    MFC,
    MOVEMENT_METHODS,
    REGIONS,
    SAVE_DPI,
    SESSION_COLUMNS,
    TEST_MWU,
    TEST_TTEST_STUDENT,
    TEST_TTEST_WELCH,
    WINDOW_INITIATION_MOVEMENT,
    WINDOW_MOVEMENT,
    _metricFromCells,
    _regionTest,
    _sessionCells,
    animalOf,
    binEdges,
    bootstrapProbability,
    bootstrapSummary,
    formatBootstrap,
    formatSummary,
    hierarchicalBootstrap,
    getSgfNeurons,
    maxFiringHist,
    methodSpec,
    movementMask,
    movementOnset,
    perSessionCounts,
    perSessionCountsAllMethods,
    plotRegionBars,
    regionSummary,
    windowOnset,
)

# The real sampling-aligned epochs (normed_sampling_df.iloc[0].epochs_ranges):
# 0.1s before sampling, sampling, 0.1s of movement to the lateral port.
EPOCH_RANGES = [(0, 2), (3, 29), (30, 32)]
M2_BI, ALM_BI = 15, 6  # BrainRegion enum values -> MFC / LFC


# ---------------------------------------------------------------------------
# Bins
# ---------------------------------------------------------------------------
def test_bin_edges_are_pre_sampling_then_sampling_then_movement():
    bins = binEdges(EPOCH_RANGES, DEFAULT_BINS_COUNT)
    # 7 bins -> 8 edges: pre-sampling, 5 equal sampling bins, movement.
    assert len(bins) == DEFAULT_BINS_COUNT + 1
    assert bins[0] == 0 and bins[1] == 3      # the -0.1s pre-sampling bin
    assert bins[-2] == 30 and bins[-1] == 33  # the movement bin
    sampling_widths = np.diff(bins[1:-1])
    assert np.allclose(sampling_widths, sampling_widths[0])


def test_max_firing_hist_bins_median_peak():
    df = pd.DataFrame({
        "BrainRegion": [M2_BI] * 3,
        "trace_id": ["a", "b", "c"],
        "max_idxs": [np.array([0, 1, 2]),      # med 1  -> first bin
                     np.array([14, 15, 16]),   # med 15 -> a sampling bin
                     np.array([31, 32, 32])],  # med 32 -> last (movement) bin
    })
    hist_df, bins = maxFiringHist(df, num_bins=DEFAULT_BINS_COUNT,
                                  epoch_ranges=EPOCH_RANGES)
    assert list(hist_df.fire_pos) == [1.0, 15.0, 32.0]
    assert np.allclose(bins, binEdges(EPOCH_RANGES, DEFAULT_BINS_COUNT))
    assert list(hist_df.bin_val_left) == [0.0, 13.8, 30.0]


def test_max_firing_hist_bin_idx_counts_only_populated_bins():
    """Why movementMask keys on bin_val_left: bin_idx is not the bin number.

    Only 3 of the 7 bins are populated here, so the movement neuron gets
    bin_idx 2 rather than 6 -- the quirk is inherited from the notebook.
    """
    df = pd.DataFrame({
        "BrainRegion": [M2_BI] * 3,
        "trace_id": ["a", "b", "c"],
        "max_idxs": [np.array([1]), np.array([15]), np.array([32])],
    })
    hist_df, _bins = maxFiringHist(df, num_bins=DEFAULT_BINS_COUNT,
                                   epoch_ranges=EPOCH_RANGES)
    assert list(hist_df.bin_idx) == [0, 1, 2]
    mask = movementMask(df, EPOCH_RANGES, method=METHOD_LAST_BIN)
    assert list(mask) == [False, False, True]


# ---------------------------------------------------------------------------
# Movement criteria
# ---------------------------------------------------------------------------
def _neuron(trace_id, peaks, br=M2_BI, sess="sessA", prcnt_valid=50.0):
    return {"trace_id": trace_id, "ShortName": sess, "BrainRegion": br,
            "max_idxs": np.array(peaks), "prcnt_valid": prcnt_valid}


def _criteria_df():
    return pd.DataFrame([
        # median 5 -> mid-sampling, never a movement neuron.
        _neuron("sampling", [4, 5, 6]),
        # median 31 and q1 31 -> movement neuron under every criterion.
        _neuron("always", [30, 31, 32, 31]),
        # median 31 but q1 = 23.75 -> it peaks before movement onset in a
        # quarter of its trials, so the q1 criterion rejects it.
        _neuron("median_only", [2, 31, 31, 32]),
        # median exactly on the onset sample: inside the movement epoch, but
        # pd.cut's left-open last bin (30, 33] excludes it.
        _neuron("on_onset", [29, 30, 31]),
    ])


def test_movement_epoch_and_last_bin_differ_only_on_the_onset_sample():
    df = _criteria_df()
    last_bin = movementMask(df, EPOCH_RANGES, method=METHOD_LAST_BIN)
    epoch = movementMask(df, EPOCH_RANGES, method=METHOD_MOVEMENT_EPOCH)
    assert list(last_bin) == [False, True, True, False]
    assert list(epoch) == [False, True, True, True]
    assert movementOnset(EPOCH_RANGES) == 30


# ---------------------------------------------------------------------------
# The initiation + movement window (last two bins)
# ---------------------------------------------------------------------------
def test_window_onset_is_the_movement_or_the_initiation_bin_edge():
    # 7 bins over [(0,2), (3,29), (30,32)]: ... (19.2, 24.6], (24.6, 30], (30, 33]
    assert windowOnset(EPOCH_RANGES, DEFAULT_BINS_COUNT,
                       WINDOW_MOVEMENT) == 30
    assert windowOnset(EPOCH_RANGES, DEFAULT_BINS_COUNT,
                       WINDOW_INITIATION_MOVEMENT) == pytest.approx(24.6)
    # The 1-bin window is the movement epoch onset by construction.
    assert windowOnset(EPOCH_RANGES) == movementOnset(EPOCH_RANGES)


def test_last_two_bins_adds_the_initiation_bin_and_nothing_earlier():
    df = pd.DataFrame([
        _neuron("early_sampling", [10, 11, 12]),   # med 11  -> bin 2
        _neuron("mid_sampling", [20, 21, 22]),     # med 21  -> bin 4, still out
        _neuron("initiation", [26, 27, 28]),       # med 27  -> bin 5, NEW
        _neuron("movement", [31, 31, 32]),         # med 31  -> bin 6
    ])
    last_bin = movementMask(df, EPOCH_RANGES, method=METHOD_LAST_BIN)
    last_two = movementMask(df, EPOCH_RANGES, method=METHOD_LAST_TWO_BINS)
    assert list(last_bin) == [False, False, False, True]
    assert list(last_two) == [False, False, True, True]


def test_last_two_bins_is_a_superset_of_last_bin():
    df = _criteria_df()
    last_bin = movementMask(df, EPOCH_RANGES, method=METHOD_LAST_BIN)
    last_two = movementMask(df, EPOCH_RANGES, method=METHOD_LAST_TWO_BINS)
    assert (last_two | last_bin).equals(last_two)


def test_two_bin_window_applies_to_the_threshold_criteria_too():
    # med 25 / q1 25: inside the initiation bin, before the movement bin.
    df = pd.DataFrame([_neuron("initiation", [25, 25, 25])])
    for method, expected in [(METHOD_MOVEMENT_EPOCH, False),
                             (METHOD_INITIATION_ONSET, True),
                             (METHOD_IQR_Q1, False),
                             (METHOD_IQR_Q1_TWO_BINS, True)]:
        got = movementMask(df, EPOCH_RANGES, method=method).iloc[0]
        assert bool(got) is expected, method


def test_method_specs_cover_the_default_run_list():
    assert set(MOVEMENT_METHODS) <= set(METHOD_SPECS)
    assert set(METHOD_LABELS) == set(METHOD_SPECS)
    assert METHOD_LAST_TWO_BINS in MOVEMENT_METHODS
    for method, (_criterion, n_last_bins) in METHOD_SPECS.items():
        assert n_last_bins in (WINDOW_MOVEMENT, WINDOW_INITIATION_MOVEMENT), \
            method


def test_method_spec_rejects_unknown_method():
    with pytest.raises(AssertionError):
        methodSpec("whatever")


def test_iqr_q1_criterion_is_stricter_and_keeps_input_order():
    df = _criteria_df()
    q1 = movementMask(df, EPOCH_RANGES, method=METHOD_IQR_Q1)
    # extractIQR sorts by median; the mask must come back in input order.
    assert list(q1.index) == list(df.index)
    assert list(q1) == [False, True, False, False]


def test_movement_mask_rejects_unknown_method():
    with pytest.raises(AssertionError):
        movementMask(_criteria_df(), EPOCH_RANGES, method="whatever")


# ---------------------------------------------------------------------------
# getSgfNeurons
# ---------------------------------------------------------------------------
def _stats_df():
    return pd.DataFrame({
        "long_trace_id": ["s_1", "s_2", "s_3", "s_4", "s_5"],
        "data_col": ["ChoiceLeft", "ChoiceLeft", "ChoiceLeft",
                     "PrevChoiceLeft", "ChoiceLeft"],
        "prior_data_col": [None, None, None, None, "PrevChoiceCorrect"],
        "DVstr": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "pval": [0.01, 0.5, 0.05, 0.001, 0.001],
    })


def test_get_sgf_neurons_keeps_significant_unconditioned_data_col():
    # s_2 not significant, s_4 a different data_col, s_5 has a prior.
    assert getSgfNeurons(_stats_df(), pval=0.05) == {"s_1", "s_3"}


def test_get_sgf_neurons_drops_dv_split_rows():
    stats_df = _stats_df()
    stats_df["DVstr"] = stats_df.DVstr.astype(object)
    stats_df.loc[0, "DVstr"] = "DV>0"
    assert getSgfNeurons(stats_df, pval=0.05) == {"s_3"}


def test_get_sgf_neurons_can_select_a_prior():
    got = getSgfNeurons(_stats_df(), pval=0.05,
                        prior_data_col="PrevChoiceCorrect")
    assert got == {"s_5"}


# ---------------------------------------------------------------------------
# Per-session counts
# ---------------------------------------------------------------------------
def _counts_df():
    """2 MFC sessions + 1 LFC session with hand-countable movement neurons."""
    movement, sampling = [31, 31, 32], [5, 6, 7]
    rows = []
    # MFC session 1: 4 neurons, 2 movement (m1, m2), 2 choice-selective
    # (m1 -> movement AND choice, s1 -> choice only).
    rows += [_neuron("A_m1", movement, M2_BI, "A"),
             _neuron("A_m2", movement, M2_BI, "A"),
             _neuron("A_s1", sampling, M2_BI, "A"),
             _neuron("A_s2", sampling, M2_BI, "A")]
    # MFC session 2: 2 neurons, 1 movement, no choice-selective neuron.
    rows += [_neuron("B_m1", movement, M2_BI, "B"),
             _neuron("B_s1", sampling, M2_BI, "B")]
    # LFC session: 4 neurons, 1 movement (also choice-selective).
    rows += [_neuron("C_m1", movement, ALM_BI, "C"),
             _neuron("C_s1", sampling, ALM_BI, "C"),
             _neuron("C_s2", sampling, ALM_BI, "C"),
             _neuron("C_s3", sampling, ALM_BI, "C")]
    return pd.DataFrame(rows)


CHOICE_IDS = {"A_m1", "A_s1", "C_m1"}


def test_per_session_counts_percentages_and_labels():
    sess_df = perSessionCounts(_counts_df(), EPOCH_RANGES,
                               choice_sgf_ids=CHOICE_IDS)
    assert list(sess_df.columns) == SESSION_COLUMNS
    sess_df = sess_df.set_index("ShortName")

    assert list(sess_df.loc[["A", "B"], "BrainRegion"]) == [MFC, MFC]
    assert sess_df.loc["C", "BrainRegion"] == LFC

    assert sess_df.loc["A", "n_neurons"] == 4
    assert sess_df.loc["A", "n_movement"] == 2
    assert sess_df.loc["A", "n_choice"] == 2
    assert sess_df.loc["A", "n_movement_choice"] == 1
    assert sess_df.loc["A", METRIC_MOVEMENT] == pytest.approx(50.0)
    assert sess_df.loc["A", METRIC_MOVEMENT_CHOICE] == pytest.approx(25.0)
    assert sess_df.loc["A", METRIC_CHOICE] == pytest.approx(50.0)
    assert sess_df.loc["A", METRIC_CHOICE_OF_MOVEMENT] == pytest.approx(50.0)

    assert sess_df.loc["B", METRIC_MOVEMENT] == pytest.approx(50.0)
    assert sess_df.loc["B", METRIC_MOVEMENT_CHOICE] == pytest.approx(0.0)
    assert sess_df.loc["C", METRIC_MOVEMENT] == pytest.approx(25.0)
    assert sess_df.loc["C", METRIC_MOVEMENT_CHOICE] == pytest.approx(25.0)


def test_per_session_counts_without_choice_ids_is_all_zero():
    sess_df = perSessionCounts(_counts_df(), EPOCH_RANGES)
    assert (sess_df.n_movement_choice == 0).all()
    assert (sess_df[METRIC_MOVEMENT_CHOICE] == 0).all()
    assert sess_df[METRIC_MOVEMENT].notnull().all()


def test_per_session_counts_min_prcnt_valid_shrinks_the_denominator():
    df = _counts_df()
    df.loc[df.trace_id == "A_s2", "prcnt_valid"] = 1.0
    sess_df = perSessionCounts(df, EPOCH_RANGES, min_prcnt_valid=5.0)
    row = sess_df.set_index("ShortName").loc["A"]
    assert row.n_neurons == 3
    assert row[METRIC_MOVEMENT] == pytest.approx(100 * 2 / 3)


def test_per_session_counts_drops_neurons_without_active_trials():
    df = pd.concat([_counts_df(),
                    pd.DataFrame([_neuron("A_dead", [], M2_BI, "A")])],
                   ignore_index=True)
    sess_df = perSessionCounts(df, EPOCH_RANGES)
    assert sess_df.set_index("ShortName").loc["A", "n_neurons"] == 4


def test_choice_of_movement_is_nan_without_movement_neurons():
    df = pd.DataFrame([_neuron("A_s1", [5, 6, 7], M2_BI, "A")])
    row = perSessionCounts(df, EPOCH_RANGES, choice_sgf_ids=set()).iloc[0]
    assert row.n_movement == 0
    assert np.isnan(row[METRIC_CHOICE_OF_MOVEMENT])


def test_per_session_counts_all_methods_covers_every_method():
    sess_df = perSessionCountsAllMethods(_counts_df(), EPOCH_RANGES,
                                         choice_sgf_ids=CHOICE_IDS)
    assert set(sess_df.method) == set(MOVEMENT_METHODS)
    assert len(sess_df) == 3 * len(MOVEMENT_METHODS)


# ---------------------------------------------------------------------------
# Region summary
# ---------------------------------------------------------------------------
def test_region_test_runs_normality_first_and_uses_the_t_test():
    # Two clean normal-looking samples -> Shapiro passes -> t-test.
    rng = np.random.RandomState(1)
    left = rng.normal(10, 2, 12)
    right = rng.normal(20, 2, 11)
    row = _regionTest(left, right)

    assert row[f"{MFC}_shapiro_pval"] > 0.05
    assert row[f"{LFC}_shapiro_pval"] > 0.05
    assert row["is_normal"] is True
    assert row["test"] in (TEST_TTEST_STUDENT, TEST_TTEST_WELCH)
    assert row["pval"] == row["ttest_pval"]
    assert row["pval"] < 0.05
    assert not np.isnan(row["mwu_pval"])  # reported for reference either way


def test_region_test_falls_back_to_mannwhitney_when_not_normal():
    # A heavy outlier breaks normality -> the t-test must not be used.
    left = np.array([1.0, 1.1, 1.2, 1.0, 1.1, 1.3, 1.2, 1.0, 200.0])
    right = np.array([2.0, 2.1, 2.2, 2.0, 2.1, 2.3, 2.2, 2.0])
    row = _regionTest(left, right)

    assert row[f"{MFC}_shapiro_pval"] < 0.05
    assert row["is_normal"] is False
    assert row["test"] == TEST_MWU
    assert row["pval"] == row["mwu_pval"]
    assert not np.isnan(row["ttest_pval"])  # computed, just not selected


def test_region_test_levene_picks_welch_over_student():
    rng = np.random.RandomState(2)
    left = rng.normal(10, 1, 15)
    right = rng.normal(10.5, 12, 15)  # much wider -> unequal variances
    row = _regionTest(left, right)

    assert row["is_normal"] is True
    assert row["levene_pval"] < 0.05
    assert row["test"] == TEST_TTEST_WELCH


def test_region_test_too_few_sessions_for_normality_uses_mannwhitney():
    row = _regionTest([1.0, 2.0], [3.0, 4.0])
    assert np.isnan(row[f"{MFC}_shapiro_pval"])  # Shapiro needs n >= 3
    assert row["is_normal"] is False
    assert row["test"] == TEST_MWU


def test_region_test_constant_input_does_not_crash():
    row = _regionTest([5.0] * 6, [5.0] * 6)
    assert row["is_normal"] is False
    assert row["test"] == TEST_MWU


def test_region_summary_means_sessions_not_neurons():
    sess_df = perSessionCounts(_counts_df(), EPOCH_RANGES,
                               choice_sgf_ids=CHOICE_IDS)
    summary = regionSummary(sess_df, metrics=(METRIC_MOVEMENT,)).iloc[0]
    # MFC sessions are 50% and 50% -> 50%, NOT the pooled 3/6 = 50% by accident:
    # the pooled counts are reported separately.
    assert summary.MFC_mean == pytest.approx(50.0)
    assert summary.MFC_sem == pytest.approx(0.0)
    assert summary.MFC_n_sessions == 2
    assert (summary.MFC_n, summary.MFC_total) == (3, 6)
    assert summary.LFC_mean == pytest.approx(25.0)
    assert summary.LFC_n_sessions == 1
    assert (summary.LFC_n, summary.LFC_total) == (1, 4)
    assert np.isnan(summary.LFC_sem)  # a single session has no SEM


def test_region_summary_reports_a_pvalue_per_method_and_metric():
    rng = np.random.RandomState(0)
    rows = []
    for sess in range(6):  # enough sessions for a meaningful MWU
        for br, sess_prefix, hi in [(M2_BI, "M", True), (ALM_BI, "L", False)]:
            n_movement = 8 if hi else 2
            for idx in range(10):
                peaks = ([31, 31, 32] if idx < n_movement else
                         [5, 6, 7] + [rng.randint(0, 3)])
                rows.append(_neuron(f"{sess_prefix}{sess}_{idx}", peaks, br,
                                    f"{sess_prefix}{sess}"))
    sess_df = perSessionCountsAllMethods(pd.DataFrame(rows), EPOCH_RANGES)
    summary = regionSummary(sess_df, metrics=(METRIC_MOVEMENT,))

    assert list(summary.method) == list(MOVEMENT_METHODS)
    for _, row in summary.iterrows():
        assert row.MFC_mean == pytest.approx(80.0)
        assert row.LFC_mean == pytest.approx(20.0)
        assert row.pval < 0.05


# ---------------------------------------------------------------------------
# Hierarchical bootstrap
# ---------------------------------------------------------------------------
def test_animal_of_takes_the_first_two_tokens_case_insensitively():
    assert animalOf("GP4_85_s10_L70_D250_ALM") == "GP4_85"
    assert animalOf("gp4_81_S5_L50_D250_ALM") == "GP4_81"
    assert animalOf("GP4_23_S1_L50_D250_mm2") == "GP4_23"


def test_per_session_counts_carries_the_animal():
    sess_df = perSessionCounts(_counts_df(), EPOCH_RANGES)
    assert "animal" in sess_df.columns
    assert list(sess_df.columns) == SESSION_COLUMNS


def test_session_cells_is_the_2x2_neuron_table():
    sess_df = perSessionCounts(_counts_df(), EPOCH_RANGES,
                               choice_sgf_ids=CHOICE_IDS)
    row = sess_df[sess_df.ShortName == "A"]
    # Session A: 4 neurons, 2 movement (m1 also choice), 2 choice (m1, s1).
    both, mov_only, choice_only, neither = _sessionCells(row)[0]
    assert (both, mov_only, choice_only, neither) == (1, 1, 1, 1)
    for metric, expected in [(METRIC_MOVEMENT, (2, 4)),
                             (METRIC_MOVEMENT_CHOICE, (1, 4)),
                             (METRIC_CHOICE, (2, 4)),
                             (METRIC_CHOICE_OF_MOVEMENT, (1, 2))]:
        num, den = _metricFromCells(_sessionCells(row), metric)
        assert (num[0], den[0]) == expected, metric


def _clustered_sess_df(n_boot_ready=True):
    """2 animals per region, 3 sessions each. The two animals of a region differ
    a lot, so the session-level SEM understates the animal-level uncertainty."""
    rows = []
    for region, animal_levels in [(MFC, {"M1": 10, "M2": 30}),
                                  (LFC, {"L1": 12, "L2": 34})]:
        for animal, level in animal_levels.items():
            for sess_idx in range(3):
                n_neurons = 100
                n_movement = level
                rows.append({
                    "method": METHOD_LAST_BIN, "BrainRegion": region,
                    "ShortName": f"{animal}_s{sess_idx}", "animal": animal,
                    "n_neurons": n_neurons, "n_movement": n_movement,
                    "n_choice": 0, "n_movement_choice": 0,
                    METRIC_MOVEMENT: 100 * n_movement / n_neurons,
                    METRIC_MOVEMENT_CHOICE: 0.0, METRIC_CHOICE: 0.0,
                    METRIC_CHOICE_OF_MOVEMENT: np.nan})
    return pd.DataFrame(rows, columns=SESSION_COLUMNS)


def test_hierarchical_bootstrap_observed_matches_the_session_mean():
    sess_df = _clustered_sess_df()
    _draws, observed = hierarchicalBootstrap(sess_df, metric=METRIC_MOVEMENT,
                                             n_boot=50)
    for region in (MFC, LFC):
        expected = sess_df[sess_df.BrainRegion == region][METRIC_MOVEMENT].mean()
        assert observed[region] == pytest.approx(expected)


def test_hierarchical_bootstrap_se_exceeds_the_session_sem_when_clustered():
    sess_df = _clustered_sess_df()
    draws, _observed = hierarchicalBootstrap(sess_df, metric=METRIC_MOVEMENT,
                                             n_boot=500, seed=0)
    for region in (MFC, LFC):
        vals = sess_df[sess_df.BrainRegion == region][METRIC_MOVEMENT]
        # Sessions repeat the animal's value exactly -> a tiny session SEM,
        # while resampling animals swings between the two levels.
        assert np.nanstd(draws[region]) > vals.sem()


def test_bootstrap_probability_counts_ties_as_half():
    # Identical distributions -> no evidence of a difference -> p = 1, which
    # only holds if exact ties are split rather than counted as "greater".
    same = np.linspace(0, 1, 100)
    assert bootstrapProbability(same, same.copy()) == pytest.approx(1.0)
    # A saturated region (every draw at 100%) vs one that sometimes ties it.
    saturated = np.full(100, 100.0)
    other = np.concatenate([np.full(50, 100.0), np.full(50, 40.0)])
    assert bootstrapProbability(saturated, other) == pytest.approx(0.5)
    # Fully separated -> p = 0.
    assert bootstrapProbability(np.zeros(100), np.ones(100)) == 0.0


def test_bootstrap_summary_structure_and_ci_covers_the_mean():
    sess_df = _clustered_sess_df()
    boot_df = bootstrapSummary(sess_df, metrics=(METRIC_MOVEMENT,),
                               methods=(METHOD_LAST_BIN,), n_boot=300)
    assert len(boot_df) == 1
    row = boot_df.iloc[0]
    assert row[f"{MFC}_n_animals"] == 2 and row[f"{LFC}_n_animals"] == 2
    assert row[f"{MFC}_ci_lo"] <= row[f"{MFC}_mean"] <= row[f"{MFC}_ci_hi"]
    assert 0 <= row["pval_boot"] <= 1
    assert row["diff_ci_lo"] <= row["diff_mean"] <= row["diff_ci_hi"]


def test_format_bootstrap_mentions_animals_and_the_ci():
    sess_df = _clustered_sess_df()
    text = formatBootstrap(bootstrapSummary(sess_df,
                                            metrics=(METRIC_MOVEMENT,),
                                            methods=(METHOD_LAST_BIN,),
                                            n_boot=200))
    assert "boot-SE" in text
    assert "95% CI" in text
    assert "2 animals" in text
    assert "p_boot=" in text


# ---------------------------------------------------------------------------
# Figures (headless)
# ---------------------------------------------------------------------------
def test_plot_region_bars_saves_at_the_shared_size_and_dpi(tmp_path,
                                                           monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved = {}
    real_savefig = plt.Figure.savefig

    def spy(self, fname, *args, **kwargs):
        saved["size"] = tuple(self.get_size_inches())
        saved["dpi"] = kwargs.get("dpi")
        saved["bbox_inches"] = kwargs.get("bbox_inches")
        return real_savefig(self, fname, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", spy)

    sess_df = perSessionCountsAllMethods(_counts_df(), EPOCH_RANGES,
                                         choice_sgf_ids=CHOICE_IDS)
    plotRegionBars(sess_df, metric=METRIC_MOVEMENT,
                   method=METHOD_LAST_TWO_BINS, save_figs=True,
                   fig_save_prefix=tmp_path)

    out = tmp_path / "MovementNeurons" / f"{METHOD_LAST_TWO_BINS}_{METRIC_MOVEMENT}.svg"
    assert out.exists(), sorted(tmp_path.rglob("*"))
    assert saved["size"] == FIG_SIZE
    assert saved["dpi"] == SAVE_DPI
    assert saved["bbox_inches"] == "tight"
    plt.close("all")


def test_plot_region_bars_into_a_given_ax_does_not_save(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sess_df = perSessionCountsAllMethods(_counts_df(), EPOCH_RANGES,
                                         choice_sgf_ids=CHOICE_IDS)
    _fig, ax = plt.subplots()
    plotRegionBars(sess_df, metric=METRIC_MOVEMENT_CHOICE,
                   method=METHOD_LAST_TWO_BINS, save_figs=True,
                   fig_save_prefix=tmp_path, ax=ax)
    assert not list(tmp_path.rglob("*.svg"))
    # The bars carry the region means, so the panel is really drawn.
    assert len(ax.patches) == len(REGIONS)
    plt.close("all")


def test_format_summary_mentions_both_regions_and_the_pvalue():
    sess_df = perSessionCounts(_counts_df(), EPOCH_RANGES,
                               choice_sgf_ids=CHOICE_IDS)
    text = formatSummary(regionSummary(sess_df))
    assert "MFC: 50.00% +/- 0.00% SEM" in text
    assert "LFC: 25.00%" in text
    assert "Normality (Shapiro-Wilk)" in text
    assert "MannWhitneyU" in text
    assert "% of all neurons" in text
