'''Tests for the within-subject z-scoring / subject-dispersion figure.

These pin the properties the figure depends on: the z-scoring is done *within*
each subject x experiment-type group (so between-group offsets are removed,
unlike the pooled version), the dispersion panel reads its statistic off the
raw seconds column (the z-scored sigma is 1.0 by construction), and both
dispersion measures -- standard deviation and inter-quartile range -- give one
point per subject.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ..stdispersion import (collectGroups, compareGroups, dispersionLabel,
                            filterSmallGroups, pValLabel, plotSTDispersion,
                            subjectDispersion, zScoreWithinSubject)

N_TRIALS = 200
COLORS = {"Accuracy": "C0", "Max Outcome": "C1", "Mice": "C2"}
SESSION_TYPES = [("Accuracy", "ReactionTime"), ("Max Outcome", "Competition")]


def _humansDf(seed=0):
    '''Two subjects x two session types, each with its own mean and sigma.'''
    rng = np.random.default_rng(seed)
    rows = []
    # (subject, session_type) -> (mean, sigma) in seconds
    specs = {("S1", "ReactionTime"): (1.5, 0.5), ("S1", "Competition"): (0.6, 0.1),
             ("S2", "ReactionTime"): (2.5, 0.2), ("S2", "Competition"): (0.9, 0.3)}
    for (name, sess), (mean, sigma) in specs.items():
        st = rng.normal(mean, sigma, N_TRIALS)
        rows.append(pd.DataFrame({"Name": name, "session_type": sess,
                                  "calcStimulusTime": st,
                                  "ChoiceCorrect": rng.integers(0, 2, N_TRIALS
                                                                ).astype(float)}))
    return pd.concat(rows, ignore_index=True)


def _miceDf(seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for name, (mean, sigma) in {"M1": (1.0, 0.4), "M2": (1.8, 0.7)}.items():
        st = rng.normal(mean, sigma, N_TRIALS)
        rows.append(pd.DataFrame({"Name": name, "calcStimulusTime": st,
                                  "ChoiceCorrect": rng.integers(0, 2, N_TRIALS
                                                                ).astype(float)}))
    return pd.concat(rows, ignore_index=True)


def test_zscore_is_within_subject_and_exp_type():
    df = _humansDf()
    zscored = zScoreWithinSubject(df, ["Name", "session_type"])

    grp = zscored.groupby(["Name", "session_type"]).calcStimulusTime
    assert np.allclose(grp.mean(), 0, atol=1e-12)
    assert np.allclose(grp.std(ddof=0), 1, atol=1e-12)
    # Pooling the same subject's two session types (the old behaviour) leaves a
    # per-session-type offset behind; z-scoring each one separately does not.
    pooled = zScoreWithinSubject(df, ["Name"])
    pooled_offsets = pooled.groupby(["Name", "session_type"]).calcStimulusTime.mean()
    assert pooled_offsets.abs().max() > 0.5


def test_zscore_of_mice_is_per_animal():
    df = _miceDf()
    zscored = zScoreWithinSubject(df, ["Name"])
    per_animal = zscored.groupby("Name").calcStimulusTime
    assert np.allclose(per_animal.mean(), 0, atol=1e-12)
    # Pooling across animals keeps the between-animal mean difference (1.0 vs
    # 1.8 s), so animal means would not sit at 0.
    pooled = (df.calcStimulusTime - df.calcStimulusTime.mean()) / df.calcStimulusTime.std(ddof=0)
    assert pooled.groupby(df.Name).mean().abs().min() > 0.5


def test_small_groups_are_dropped():
    df = _humansDf()
    short = df[(df.Name == "S1") & (df.session_type == "Competition")].head(10)
    df = pd.concat([df[(df.Name != "S1") | (df.session_type != "Competition")],
                    short], ignore_index=True)

    kept = filterSmallGroups(df, ["Name", "session_type"], min_trials=50)
    assert ("S1", "Competition") not in set(map(tuple,
                                    kept[["Name", "session_type"]].to_numpy()))
    assert len(zScoreWithinSubject(df, ["Name", "session_type"])) == 3*N_TRIALS
    assert len(subjectDispersion(df, ["Name", "session_type"])) == 3


def test_subject_dispersion_is_raw_seconds_sigma():
    df = _humansDf()
    sigmas = subjectDispersion(df, ["Name", "session_type"], measure="std")

    expected = df.groupby(["Name", "session_type"]).calcStimulusTime.std()
    pd.testing.assert_series_equal(sigmas, expected, check_names=False)
    # Sanity: recovers the generating sigmas, and the two session types differ.
    assert sigmas[("S1", "ReactionTime")] == pytest.approx(0.5, abs=0.1)
    assert sigmas[("S1", "Competition")] == pytest.approx(0.1, abs=0.05)
    # "std" is the default measure.
    pd.testing.assert_series_equal(
        sigmas, subjectDispersion(df, ["Name", "session_type"]))


def test_subject_dispersion_iqr_is_the_quartile_spread():
    df = _humansDf()
    iqrs = subjectDispersion(df, ["Name", "session_type"], measure="iqr")

    grp = df.groupby(["Name", "session_type"]).calcStimulusTime
    expected = grp.quantile(0.75) - grp.quantile(0.25)
    pd.testing.assert_series_equal(iqrs, expected, check_names=False)
    # For a normal sample the IQR is ~1.349 sigma, so it tracks the generating
    # sigma but is not the same number as the std measure.
    sigmas = subjectDispersion(df, ["Name", "session_type"], measure="std")
    assert np.allclose(iqrs / sigmas, 1.349, atol=0.15)


def test_subject_dispersion_iqr_ignores_a_heavy_tail_the_std_follows():
    '''The point of offering the IQR: it is robust to the long right tail.'''
    df = _humansDf()
    tail = df[df.Name == "S1"].head(10).copy()
    tail["calcStimulusTime"] = 50.0     # a handful of very slow trials
    df = pd.concat([df, tail], ignore_index=True)
    keys = ["Name", "session_type"]

    clean = _humansDf()
    def _ratio(measure):
        subject = ("S1", "ReactionTime")
        return (subjectDispersion(df, keys, measure=measure)[subject] /
                subjectDispersion(clean, keys, measure=measure)[subject])

    assert _ratio("iqr") == pytest.approx(1, abs=0.15)  # barely moves
    assert _ratio("std") > 5                            # blows up


def test_subject_dispersion_rejects_an_unknown_measure():
    with pytest.raises(AssertionError):
        subjectDispersion(_humansDf(), ["Name"], measure="mad")


def test_dispersion_label_names_the_measure():
    assert dispersionLabel("std") == "Subject σ"
    assert dispersionLabel("iqr") == "Subject IQR"
    assert dispersionLabel("iqr", (0.1, 0.9)) == "Subject IQR (10-90%)"


def test_dispersion_of_zscored_column_is_degenerate():
    '''Why the right panel must not be fed the z-scored column.'''
    df = _humansDf()
    zscored = zScoreWithinSubject(df, ["Name", "session_type"])
    sigmas = subjectDispersion(zscored, ["Name", "session_type"])
    assert np.allclose(sigmas, 1, atol=1e-2)


def test_compare_groups_uses_anova_tukey_when_normal():
    rng = np.random.default_rng(3)
    samples = {"a": rng.normal(0, 1, 40), "b": rng.normal(0.2, 1, 40),
               "c": rng.normal(-0.1, 1, 40)}
    res = compareGroups(samples, verbose=False)

    assert res["is_normal"]
    assert res["test_name"] == "f_oneway"
    assert res["post_hoc_str"].startswith("Tukey")
    assert res["posthoc"].shape == (3, 3)
    assert np.allclose(np.diag(res["posthoc"].to_numpy()), 1)


def test_compare_groups_falls_back_to_kruskal_dunn_when_not_normal():
    rng = np.random.default_rng(4)
    samples = {"a": rng.exponential(1, 60)**3, "b": rng.exponential(2, 60)**3}
    res = compareGroups(samples, verbose=False)

    assert not res["is_normal"]
    assert res["test_name"] == "kruskal"
    assert "Dunn" in res["post_hoc_str"]


@pytest.mark.parametrize("measure", ["std", "iqr"])
def test_collect_groups_one_point_per_subject_and_exp_type(measure):
    groups = collectGroups(_humansDf(), _miceDf(), colors=COLORS,
                           session_types=SESSION_TYPES, measure=measure)
    assert [g["label"] for g in groups] == ["Accuracy", "Max Outcome", "Mice"]
    for group in groups:
        assert len(group["dispersion"]) == 2   # two subjects / animals each
        assert group["n_trials"] == 2*N_TRIALS
        assert (group["dispersion"] > 0).all()
        # Trial-level z-scores, centred within each subject.
        assert group["zscores"].mean() == pytest.approx(0, abs=1e-9)


def test_collect_groups_applies_query_after_the_min_trials_cut_off():
    groups = collectGroups(_humansDf(), None, colors=COLORS,
                           session_types=SESSION_TYPES,
                           df_query="ChoiceCorrect == 1")
    for group in groups:
        assert len(group["dispersion"]) == 2
        assert group["n_trials"] < 2*N_TRIALS


@pytest.mark.parametrize("measure", ["std", "iqr"])
def test_plot_runs_and_returns_stats(measure):
    res = plotSTDispersion(_humansDf(), _miceDf(), colors=COLORS,
                           session_types=SESSION_TYPES, measure=measure)
    assert set(res) >= {"test_name", "pvalue", "posthoc", "normality"}
    assert res["posthoc"].shape == (3, 3)
    ax_disp = plt.gcf().axes[1]
    assert ax_disp.get_ylabel().endswith(dispersionLabel(measure))
    plt.close("all")


@pytest.mark.parametrize("measure", ["std", "iqr"])
def test_plot_saves_one_file_per_measure(tmp_path, measure):
    plotSTDispersion(_humansDf(), _miceDf(), colors=COLORS,
                     session_types=SESSION_TYPES, measure=measure,
                     save_fp=str(tmp_path), save_figs=True)
    saved = [fp.name for fp in tmp_path.iterdir()]
    assert saved == [f"humans_mice_sampling_time_dispersion_{measure}.svg"]
    plt.close("all")


def test_pval_label_always_spells_out_the_number():
    # Significant pairs keep their stars ...
    assert pValLabel(0.0004) == "*** p=0.0004"
    assert pValLabel(0.0123) == "* p=0.0123"
    # ... and a non-significant one still reports its p-value rather than a
    # bare "ns".
    assert pValLabel(0.1227551) == "ns p=0.1228"


def test_plot_annotates_every_pair_with_its_pvalue():
    res = plotSTDispersion(_humansDf(), _miceDf(), colors=COLORS,
                           session_types=SESSION_TYPES)
    texts = [t.get_text() for t in plt.gcf().axes[1].texts]
    posthoc = res["posthoc"].to_numpy()
    expected = [pValLabel(posthoc[i][j])
                for i in range(len(posthoc)) for j in range(i + 1, len(posthoc))]

    assert texts == expected
    assert all("p=" in t for t in texts)
    plt.close("all")
