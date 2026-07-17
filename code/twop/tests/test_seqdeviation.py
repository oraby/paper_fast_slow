import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[3]))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.twop.seqdeviation import (  # noqa: E402
    CONDITION_CROSS,
    CONDITION_MATCHED,
    CROSS_COLUMNS,
    PENALTY_COLUMNS,
    SHUFFLE_LEVELS,
    STRATEGY_FAST,
    STRATEGY_SLOW,
    _invert_curve,
    _mallows_expected_d,
    _mallows_phi,
    _perm_scores,
    _rim_perms,
    _session_shuffle_means,
    _shuffled_trial_scores,
    _trial_penalty,
    build_cross_penalty_df,
    build_penalty_df,
    common_reference_ranks,
    cross_shuffle_test,
    cross_significance,
    extractIQR,
    reference_ranks,
    shuffle_calibration,
    shuffle_level_summary,
    trial_penalties,
)


def test_extract_iqr_median_and_sorting():
    df = pd.DataFrame({
        "trace_id": ["r1", "r2", "r3"],
        "max_idxs": [np.array([1, 2, 3]),      # med 2
                     np.array([10, 20]),        # med 15
                     np.array([5, 5, 5, 5])],   # med 5
    })
    out = extractIQR(df)
    # Sorted ascending by med -> r1 (2), r3 (5), r2 (15).
    assert list(out.trace_id) == ["r1", "r3", "r2"]
    assert list(out.med) == [2.0, 5.0, 15.0]


def test_trial_penalty_matches_worked_example():
    # Task example: reference ranks 1, 5, 10; rank-5 fires first, rank-1 second,
    # rank-10 third -> penalties |1-5|/3, |5-1|/3, 0.
    trial_df = pd.DataFrame({
        "trace_id": ["A", "B", "C"],
        "ref_rank": [1, 5, 10],
        "peak_time": [20.0, 10.0, 30.0],  # B first, A second, C third
    })
    out = _trial_penalty(trial_df).set_index("trace_id")

    assert out.loc["A", "observed_rank"] == 5
    assert out.loc["B", "observed_rank"] == 1
    assert out.loc["C", "observed_rank"] == 10
    assert out.loc["A", "penalty"] == pytest.approx(4 / 3)
    assert out.loc["B", "penalty"] == pytest.approx(4 / 3)
    assert out.loc["C", "penalty"] == 0.0
    assert (out["n_active_in_trial"] == 3).all()


def test_trial_penalty_perfect_order_is_zero():
    trial_df = pd.DataFrame({
        "trace_id": ["A", "B", "C"],
        "ref_rank": [1, 4, 9],
        "peak_time": [1.0, 2.0, 3.0],  # already in reference order
    })
    out = _trial_penalty(trial_df)
    assert (out["penalty"] == 0).all()
    assert (out["norm_penalty"] == 0).all()
    assert (out["gap_norm_penalty"] == 0).all()


def test_norm_penalty_worked_example_mean_is_half():
    # Same {1,5,10} example. Locally the active neurons re-rank to 1,2,3; one
    # adjacent swap -> footrule 2, max (full reversal, n=3) 4 -> trial mean 0.5.
    trial_df = pd.DataFrame({
        "trace_id": ["A", "B", "C"],
        "ref_rank": [1, 5, 10],
        "peak_time": [20.0, 10.0, 30.0],
    })
    out = _trial_penalty(trial_df).set_index("trace_id")
    assert out.loc["A", "local_ref_rank"] == 1
    assert out.loc["A", "local_observed_rank"] == 2
    assert out.loc["C", "norm_penalty"] == 0.0
    # per-neuron norm scaled so the trial mean equals the normalized footrule.
    assert out["norm_penalty"].mean() == pytest.approx(0.5)
    assert out.loc["A", "norm_penalty"] == pytest.approx(0.75)
    # gap-aware: global footrule 8 / this set's max reversal 18.
    assert out["gap_norm_penalty"].mean() == pytest.approx(8 / 18)


def test_gap_norm_penalty_is_gap_sensitive():
    # The caveat: same ordinal swap (first two neurons), different reference gaps.
    # Local norm_penalty is 0.5 for both; gap_norm distinguishes them.
    a = _trial_penalty(pd.DataFrame({"trace_id": ["A", "B", "C"],
                                     "ref_rank": [1, 5, 10],
                                     "peak_time": [20.0, 10.0, 30.0]}))
    b = _trial_penalty(pd.DataFrame({"trace_id": ["A", "B", "C"],
                                     "ref_rank": [1, 10, 15],
                                     "peak_time": [20.0, 10.0, 30.0]}))
    assert a["norm_penalty"].mean() == pytest.approx(0.5)
    assert b["norm_penalty"].mean() == pytest.approx(0.5)      # local: identical
    assert a["gap_norm_penalty"].mean() == pytest.approx(8 / 18)   # 0.444
    assert b["gap_norm_penalty"].mean() == pytest.approx(18 / 28)  # 0.643
    assert b["gap_norm_penalty"].mean() > a["gap_norm_penalty"].mean()


def test_norm_penalty_full_reversal_is_one():
    # Two neurons firing in reverse of reference order = maximal disorder.
    trial_df = pd.DataFrame({
        "trace_id": ["A", "B"],
        "ref_rank": [1, 8],
        "peak_time": [2.0, 1.0],  # rank-8 peaks first -> reversed
    })
    out = _trial_penalty(trial_df)
    assert out["norm_penalty"].mean() == pytest.approx(1.0)
    assert (out["norm_penalty"] == 1.0).all()
    # A full reversal is the max for the set too -> gap_norm mean 1.
    assert out["gap_norm_penalty"].mean() == pytest.approx(1.0)


def test_trial_penalty_tie_in_peak_time_adds_no_deviation():
    # Two neurons peak at the same sample -> tie broken by ref rank -> no
    # artificial deviation for the tied pair.
    trial_df = pd.DataFrame({
        "trace_id": ["A", "B"],
        "ref_rank": [2, 6],
        "peak_time": [5.0, 5.0],
    })
    out = _trial_penalty(trial_df)
    assert (out["penalty"] == 0).all()


def test_reference_ranks_filters_strictly_above_5_and_dense_ranks():
    df = pd.DataFrame({
        "ShortName": ["S1"] * 5,
        "trace_id": ["S1_0", "S1_1", "S1_2", "S1_3", "S1_4"],
        "BrainRegion": [15] * 5,
        "prcnt_valid": [10.0, 20.0, 3.0, 50.0, 5.0],  # 3.0 and exactly 5.0 drop
        "max_idxs": [np.array([5, 5, 5]),    # med 5
                     np.array([10, 10]),     # med 10
                     np.array([1]),          # filtered (3%)
                     np.array([2, 2, 2, 2]), # med 2
                     np.array([0])],         # filtered (exactly 5%)
    })
    ref = reference_ranks(df).set_index("trace_id")
    assert set(ref.index) == {"S1_0", "S1_1", "S1_3"}
    # med order: S1_3 (2) -> S1_0 (5) -> S1_1 (10)
    assert ref.loc["S1_3", "ref_rank"] == 1
    assert ref.loc["S1_0", "ref_rank"] == 2
    assert ref.loc["S1_1", "ref_rank"] == 3
    assert (ref["n_ref_neurons"] == 3).all()


def _toy_strategy_df():
    # One MFC session, two reference neurons; n0 also active in a solo trial 3.
    return pd.DataFrame({
        "ShortName": ["S1", "S1"],
        "trace_id": ["S1_0", "S1_1"],
        "BrainRegion": [15, 15],  # M2_Bi -> MFC
        "prcnt_valid": [50.0, 50.0],
        "active_trial_numbers": [np.array([1, 2, 3]), np.array([1, 2])],
        "max_idxs": [np.array([3, 9, 5]), np.array([7, 1])],
    })


def test_trial_penalties_end_to_end():
    out = trial_penalties(_toy_strategy_df(), STRATEGY_FAST)
    # med: n0=median([3,9,5])=5 -> rank 2 ; n1=median([7,1])=4 -> rank 1.
    # Trial 3 has only n0 active -> dropped (min_active_in_trial=2).
    assert list(out.columns) == PENALTY_COLUMNS
    assert set(out.TrialNumber) == {1, 2}
    assert (out.BrainRegion == "MFC").all()
    assert (out.n_ref_neurons == 2).all()
    assert (out.n_active_in_trial == 2).all()
    assert (out.trial_strategy == STRATEGY_FAST).all()

    t1 = out[out.TrialNumber == 1].set_index("trace_id")
    # Trial 1: peaks n0=3 (rank2) before n1=7 (rank1) -> both off by one slot,
    # i.e. a full reversal of the 2 active neurons -> norm/gap 1.
    assert t1.loc["S1_0", "penalty"] == pytest.approx(0.5)
    assert t1.loc["S1_1", "penalty"] == pytest.approx(0.5)
    assert t1["norm_penalty"].mean() == pytest.approx(1.0)
    assert t1["gap_norm_penalty"].mean() == pytest.approx(1.0)
    # abs_gap = |ref - observed| / (n_ref_neurons - 1) = 1 / (2 - 1) = 1.
    assert t1["abs_gap_penalty"].tolist() == pytest.approx([1.0, 1.0])

    t2 = out[out.TrialNumber == 2]
    # Trial 2: peaks n1=1 (rank1) before n0=9 (rank2) -> reference order -> 0.
    assert (t2.penalty == 0).all()
    assert (t2.norm_penalty == 0).all()
    assert (t2.gap_norm_penalty == 0).all()
    assert (t2.abs_gap_penalty == 0).all()

    assert set(out.trace_num) == {0, 1}


def test_build_penalty_df_concats_fast_and_slow():
    fast = _toy_strategy_df()
    slow = _toy_strategy_df()
    out = build_penalty_df(fast, slow)
    assert set(out.trial_strategy) == {"Fast", "Slow"}
    assert list(out.columns) == PENALTY_COLUMNS


# --- "Same sequence?" (cross) analysis -------------------------------------
def _toy_cross_fast():
    # S1_2 is fast-inactive (3%) -> not in the common set.
    return pd.DataFrame({
        "ShortName": ["S1", "S1", "S1"],
        "trace_id": ["S1_0", "S1_1", "S1_2"],
        "BrainRegion": [15, 15, 15],
        "prcnt_valid": [50.0, 50.0, 3.0],
        "active_trial_numbers": [np.array([1, 2]), np.array([1, 2]), np.array([1])],
        "max_idxs": [np.array([3, 9]), np.array([7, 1]), np.array([5])],
    })


def _toy_cross_slow():
    return pd.DataFrame({
        "ShortName": ["S1", "S1", "S1"],
        "trace_id": ["S1_0", "S1_1", "S1_2"],
        "BrainRegion": [15, 15, 15],
        "prcnt_valid": [50.0, 50.0, 50.0],
        "active_trial_numbers": [np.array([10, 11]), np.array([10, 11]), np.array([10, 11])],
        "max_idxs": [np.array([2, 8]), np.array([6, 1]), np.array([4, 4])],
    })


def test_common_reference_ranks_intersection_and_reref():
    ref = common_reference_ranks(_toy_cross_fast(), _toy_cross_slow())
    # S1_2 is fast-inactive -> excluded from the common set.
    assert set(ref.trace_id) == {"S1_0", "S1_1"}
    ref = ref.set_index("trace_id")
    # Fast order: med([7,1])=4 (S1_1) < med([3,9])=6 (S1_0).
    assert ref.loc["S1_1", "ref_rank"] == 1
    assert ref.loc["S1_0", "ref_rank"] == 2
    assert (ref["n_ref_neurons"] == 2).all()


def test_build_cross_penalty_df_fast_reference():
    out = build_cross_penalty_df(_toy_cross_fast(), _toy_cross_slow(),
                                 reference=STRATEGY_FAST)
    assert list(out.columns) == CROSS_COLUMNS
    assert (out.reference == STRATEGY_FAST).all()
    assert "S1_2" not in set(out.trace_id)            # non-common neuron dropped
    assert (out.n_ref_neurons == 2).all()             # common set size

    matched = out[out.condition == CONDITION_MATCHED]
    cross = out[out.condition == CONDITION_CROSS]
    assert (matched.trial_strategy == STRATEGY_FAST).all()   # ref's own trials
    assert (cross.trial_strategy == STRATEGY_SLOW).all()     # other strategy
    assert set(matched.TrialNumber) == {1, 2}
    assert set(cross.TrialNumber) == {10, 11}

    # Trial 10 (slow) fires the common pair in reverse of the fast order -> 1;
    # trial 11 matches the fast order -> 0.
    c10 = cross[cross.TrialNumber == 10]
    c11 = cross[cross.TrialNumber == 11]
    assert c10.norm_penalty.mean() == pytest.approx(1.0)
    assert (c11.norm_penalty == 0).all()


def test_build_cross_penalty_df_slow_reference_is_symmetric():
    out = build_cross_penalty_df(_toy_cross_fast(), _toy_cross_slow(),
                                 reference=STRATEGY_SLOW)
    assert (out.reference == STRATEGY_SLOW).all()
    matched = out[out.condition == CONDITION_MATCHED]
    cross = out[out.condition == CONDITION_CROSS]
    assert (matched.trial_strategy == STRATEGY_SLOW).all()
    assert (cross.trial_strategy == STRATEGY_FAST).all()


def _synth_cross(n_sessions=8, effect=0.1, seed=0):
    # Sessions where `cross` sits `effect` above `matched` (paired), for testing
    # the significance plumbing.
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        base = rng.uniform(0.2, 0.3)
        for cond, extra in [(CONDITION_MATCHED, 0.0), (CONDITION_CROSS, effect)]:
            for t in range(5):
                v = base + extra + rng.normal(0, 0.01)
                rows.append(dict(BrainRegion="MFC", ShortName=f"S{s}",
                                 condition=cond, TrialNumber=t, reference="Fast",
                                 norm_penalty=v, gap_norm_penalty=v))
    return pd.DataFrame(rows)


def test_cross_significance_structure_and_detects_effect():
    sig = cross_significance(_synth_cross(n_sessions=8, effect=0.1))
    # one row per (region x score); here 1 region x 2 scores.
    assert len(sig) == 2
    assert set(sig.columns) >= {"reference", "BrainRegion", "score", "n_sessions",
                                "matched_mean", "cross_mean", "shapiro_W",
                                "shapiro_p", "test", "statistic", "p_value"}
    row = sig.iloc[0]
    assert row.n_sessions == 8
    assert row.test in {"paired t-test", "Wilcoxon signed-rank"}
    assert 0.0 <= row.p_value <= 1.0
    assert row.cross_mean > row.matched_mean       # effect direction
    assert (sig.p_value < 0.05).all()              # clear paired effect is significant


def _synth_cross_with_ranks(n_sessions=6, n_trials=4, matched=0.1, cross=0.6):
    # Every trial has 4 neurons ranked 1..4; observed scores are set directly so
    # the observed effect is (cross - matched); the shuffle null is rebuilt from
    # the ref ranks. matched << cross -> observed effect should beat the null.
    rows = []
    for s in range(n_sessions):
        for cond, val in [(CONDITION_MATCHED, matched), (CONDITION_CROSS, cross)]:
            for t in range(n_trials):
                for rank in [1, 2, 3, 4]:
                    rows.append(dict(BrainRegion="MFC", ShortName=f"S{s}",
                                     condition=cond, TrialNumber=t, reference="Fast",
                                     ref_rank=rank, n_ref_neurons=4,
                                     norm_penalty=val, gap_norm_penalty=val))
    return pd.DataFrame(rows)


def test_cross_shuffle_test_beats_random():
    out = cross_shuffle_test(_synth_cross_with_ranks(), score_cols=("norm_penalty",),
                             n_perm=200, seed=1)
    assert len(out) == 1  # 1 region x 1 score
    row = out.iloc[0]
    assert set(out.columns) >= {"observed_effect", "null_effect_mean",
                                "null_effect_std", "z_score", "p_shuffle_1sided",
                                "p_shuffle_2sided"}
    # observed effect reads the score columns (0.6 - 0.1); null from shuffled ranks
    assert row.observed_effect == pytest.approx(0.5)
    assert abs(row.null_effect_mean) < 0.1        # shuffle null centered near 0
    assert row.observed_effect > row.null_effect_mean
    assert row.p_shuffle_1sided < 0.05
    assert 0.0 <= row.p_shuffle_2sided <= 1.0


def test_session_shuffle_means_chance_level():
    # 4 neurons ranked 1..4: expected footrule of a random order is (n^2-1)/3 = 5,
    # max footrule = n^2//2 = 8, so norm_penalty chance = 5/8 = 0.625, per session.
    df = _synth_cross_with_ranks(n_sessions=5, n_trials=4)
    out = _session_shuffle_means(df, "norm_penalty", n_perm=400, seed=0,
                                 group_col=None)
    assert list(out.columns) == ["ShortName", "score"]
    assert len(out) == 5                       # one pooled chance value per session
    assert out.score.mean() == pytest.approx(0.625, abs=0.03)
    # split by condition -> one row per (session, condition)
    split = _session_shuffle_means(df, "norm_penalty", n_perm=200, seed=0)
    assert set(split.columns) == {"ShortName", "condition", "score"}
    assert len(split) == 5 * 2


# --- Part F: controlled-shuffle calibration ---------------------------------
def _kendall_inversions(perm):
    """Number of inversions (Kendall distance from identity) of each row."""
    perm = np.atleast_2d(perm)
    n = perm.shape[1]
    return np.array([sum(perm[r, i] > perm[r, j]
                         for i in range(n) for j in range(i + 1, n))
                     for r in range(len(perm))])


def test_mallows_expected_d_endpoints_and_monotone():
    n = 8
    assert _mallows_expected_d(0.0, n) == pytest.approx(0.0)
    # phi = 1 (uniform) -> exactly half the max distance n(n-1)/2.
    assert _mallows_expected_d(1.0, n) == pytest.approx(n * (n - 1) / 4)
    ds = [_mallows_expected_d(p, n) for p in np.linspace(0, 1, 11)]
    assert all(b >= a - 1e-9 for a, b in zip(ds, ds[1:]))   # monotone up in phi


def test_mallows_phi_inverts_the_level():
    n = 12
    max_d = n * (n - 1) / 2
    assert _mallows_phi(n, 0.0) == pytest.approx(0.0)
    assert _mallows_phi(n, 0.5) == pytest.approx(1.0, abs=1e-3)
    for s in (0.1, 0.25, 0.4):
        phi = _mallows_phi(n, s)
        assert _mallows_expected_d(phi, n) / max_d == pytest.approx(s, abs=1e-3)


def test_rim_perms_identity_and_distance():
    rng = np.random.default_rng(0)
    n = 10
    # phi = 0 -> identity for every row.
    assert (_rim_perms(n, 0.0, 20, rng) == np.arange(n)).all()
    # every row is a valid permutation.
    perms = _rim_perms(n, 0.6, 300, rng)
    assert all(sorted(row) == list(range(n)) for row in perms)
    # empirical mean Kendall distance matches the model target for the fitted phi.
    s = 0.3
    phi = _mallows_phi(n, s)
    perms = _rim_perms(n, phi, 4000, rng)
    target = s * n * (n - 1) / 2
    assert _kendall_inversions(perms).mean() == pytest.approx(target, rel=0.06)


def test_shuffle_calibration_anchors_and_monotone():
    from scipy import stats
    df = _synth_cross_with_ranks(n_sessions=3, n_trials=6)  # distinct ranks 1..4
    cal = shuffle_calibration(df, score_col="gap_norm_penalty", n_rep=60,
                              seed=1, show_progress=False)
    assert set(cal.columns) == {"reference", "BrainRegion", "ShortName",
                                "level", "rep", "score"}
    curve = cal.groupby("level")["score"].mean()
    # Endpoints are deterministic: level 0 -> identity -> 0; level 1 -> reversal -> 1.
    assert curve.loc[0.0] == pytest.approx(0.0, abs=1e-9)
    assert curve.loc[1.0] == pytest.approx(1.0, abs=1e-9)
    # 50% is the uniform-random case; n=4 distinct ranks -> 5/8 = 0.625.
    mid = float(np.interp(0.5, curve.index.to_numpy(), curve.to_numpy()))
    assert mid == pytest.approx(0.625, abs=0.04)
    # Monotone increasing (robust to MC noise via a rank correlation).
    rho, _ = stats.spearmanr(curve.index.to_numpy(), curve.to_numpy())
    assert rho > 0.99


def test_invert_curve_roundtrip():
    levels = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    curve_y = np.array([0.0, 0.30, 0.65, 0.85, 1.0])
    # a y sitting exactly on a knot returns that knot's level (in %).
    assert _invert_curve(levels, curve_y, 0.65) == pytest.approx(50.0)
    # midway between two knots interpolates; out-of-range clamps.
    assert 25.0 < _invert_curve(levels, curve_y, 0.475) < 50.0
    assert _invert_curve(levels, curve_y, 5.0) == pytest.approx(100.0)


def test_perm_scores_matches_shuffled_trial_scores():
    # The refactor must be faithful: uniform sigma reproduces the old path exactly.
    e = np.array([1.0, 3.0, 7.0, 8.0])
    cols = ["norm_penalty", "gap_norm_penalty", "abs_gap_penalty"]
    out_old = _shuffled_trial_scores(e, 200, np.random.default_rng(5), cols)
    sigma = np.argsort(np.random.default_rng(5).random((200, len(e))), axis=1)
    out_new = _perm_scores(e, sigma, cols)
    for c in cols:
        assert np.allclose(out_old[c], out_new[c])


def _synth_penalty_for_levels(score_by_group, group_col="trial_strategy",
                              n_sessions=3, n_trials=3, region="MFC"):
    rows = []
    for s in range(n_sessions):
        for grp, val in score_by_group.items():
            for t in range(n_trials):
                for rk in [1, 2, 3, 4]:
                    rows.append({"BrainRegion": region, "ShortName": f"S{s}",
                                 group_col: grp, "TrialNumber": t, "ref_rank": rk,
                                 "n_ref_neurons": 4, "gap_norm_penalty": val})
    return pd.DataFrame(rows)


def test_shuffle_calibration_accepts_trial_strategy_keys():
    # penalty_df has no 'reference'/'condition'; trials are keyed by strategy.
    df = _synth_penalty_for_levels({"Fast": 0.3, "Slow": 0.3}, n_sessions=2, n_trials=4)
    cal = shuffle_calibration(df, n_rep=40, seed=0, show_progress=False,
                              trial_keys=("trial_strategy", "TrialNumber"))
    assert set(cal.columns) == {"reference", "BrainRegion", "ShortName",
                                "level", "rep", "score"}
    assert cal["reference"].isna().all()          # carried through as None
    curve = cal.groupby("level")["score"].mean()
    assert curve.loc[0.0] == pytest.approx(0.0, abs=1e-9)
    assert curve.loc[1.0] == pytest.approx(1.0, abs=1e-9)


def test_shuffle_level_summary_reads_off_the_curve():
    # Identity calibration (score == level) -> shuffle_pct == observed score * 100.
    df = _synth_penalty_for_levels({"Fast": 0.2, "Slow": 0.5})
    lv = np.linspace(0, 1, 11)
    calib = pd.DataFrame({"BrainRegion": "MFC", "level": lv, "score": lv})
    out = shuffle_level_summary(df, calib, group_col="trial_strategy")
    assert set(out.columns) >= {"ShortName", "BrainRegion", "trial_strategy",
                                "score", "shuffle_pct"}
    m = out.groupby("trial_strategy")["shuffle_pct"].mean()
    assert m["Fast"] == pytest.approx(20.0, abs=0.5)     # ordered side (< 50)
    assert m["Slow"] == pytest.approx(50.0, abs=0.5)     # exactly chance


def test_shuffle_levels_spec():
    lv = np.round(SHUFFLE_LEVELS * 100).astype(int)
    assert lv[0] == 0 and lv[-1] == 100
    assert set(range(0, 11, 1)) <= set(lv.tolist())      # 1% steps to 10
    assert {12, 14, 16, 18} <= set(lv.tolist())          # 2% steps 10-20
    assert {25, 30, 55, 95} <= set(lv.tolist())          # 5% steps 20-100
    assert (np.diff(SHUFFLE_LEVELS) > 0).all()           # strictly increasing


