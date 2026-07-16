"""Tests for the aggregate evaluation controller (``rlmodel/model/aggregate.py``)
and its bar figure (``rlmodel/model/aggregate_plot.py``).

Covers the pure logic — spec resolution, seed plumbing, the sim cache's
iteration-0-only contract, and the bar/dot statistics + x layout — with the
forward simulation and the metric computation monkeypatched, so no behavior
data or real simulation is needed. (The real fit pickles embed a ``subject_df``
and do not unpickle under this environment's pandas, so tests must not read
them; ``test_model_compare.py`` takes the same synthetic approach.)
"""
from __future__ import annotations

import os
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
import numpy as np
import pandas as pd
import pytest

from .. import aggregate
from ..aggregate import (EvalSpec, collect_metrics, load_or_collect_metrics,
                         resolve_spec, rows_for_spec_from, model_key,
                         safe_filename, N_PSYCH_FITS,
                         FIG1L_SPECS, SCALE_BOUND_SPECS, MLE_WEIGHT_SPECS,
                         CHI2_ONLY_RRQ_SPEC)
from ..aggregate_plot import (plot_aggregates, subsample_evaluations,
                              BAR_WIDTH)
from ..compare import ColumnFit, ModelEntry
from ..mle_reeval import parse_fit_filename


_SPEC = EvalSpec("RR+Q", model_key("RewardRate", "Q-Val (Offset)"),
                 "Chi²-Noise", "green")


def _payload(include_Q=True, include_RewardRate=True, fun=100.0):
    return {"params_names": ["DRIFT_COEF"],
            "OptimRes": types.SimpleNamespace(x=np.array([1.0]), fun=fun),
            "include_Q": include_Q, "include_RewardRate": include_RewardRate}


def _fits(subjects=("S1", "S2"), filename=None, column_label="Chi²-Noise"):
    """A minimal ``{model_key: ModelEntry}`` index shaped like discover_fits'."""
    filename = filename or ("chisq_NoiseGain-RewardRate_biasQ-Val (Offset)_"
                            "Normal(0, 1)_4.8s_dt0.005.pkl")
    fid = parse_fit_filename(filename)
    entry = ModelEntry(model_key=fid.model_key, label=fid.model_label,
                       fid_example=fid)
    for s in subjects:
        entry.subjects[s] = [ColumnFit(fid=fid, payload=_payload(),
                                       filename=filename,
                                       column_label=column_label,
                                       order_rank=100.0)]
    return {fid.model_key: entry}


@pytest.fixture
def stub_sim(monkeypatch):
    """Replace the forward pass + metric computation; record the seeds used.

    The fake metric is ``seed``-dependent so the tests can assert which
    trajectory produced which row.
    """
    calls = []

    def fake_compute_sim(subject, fid, payload, df_behavior, include_Q,
                         include_RewardRate, seed=0):
        calls.append({"subject": subject, "seed": seed})
        sim_df = pd.DataFrame({"Name": [subject] * 3_000, "seed": seed})
        return sim_df, 1.0, {}, None

    def fake_subject_metrics(subject, fitted_df, *, n_psych_fits=None):
        seed = int(fitted_df["seed"].iloc[0])
        return {"Name": subject, "NumTrials": len(fitted_df),
                # Distinct per subject AND per seed, so mean/std are non-trivial.
                "R2_Psych": 0.5 + seed + (10 if subject == "S2" else 0),
                "RewardRateCorr": 0.25}

    monkeypatch.setattr(aggregate.compare, "_compute_sim", fake_compute_sim)
    monkeypatch.setattr(aggregate, "subject_metrics", fake_subject_metrics)
    return calls


# --------------------------------------------------------------------------
# Spec resolution
# --------------------------------------------------------------------------
def test_resolve_spec_finds_column_per_subject():
    resolved = resolve_spec(_fits(), _SPEC)
    assert sorted(resolved) == ["S1", "S2"]
    assert all(c.column_label == "Chi²-Noise" for c in resolved.values())


def test_resolve_spec_unknown_model_lists_available():
    spec = EvalSpec("nope", "NoSuchModel|x|y|1|2|sym", "Chi²-Noise", "gray")
    with pytest.raises(KeyError, match="Available"):
        resolve_spec(_fits(), spec)


def test_resolve_spec_unknown_column_lists_available_columns():
    spec = EvalSpec("nope", _SPEC.model_key, "Chi²-Bound", "gray")
    with pytest.raises(KeyError, match="Chi²-Noise"):
        resolve_spec(_fits(), spec)


def test_presets_address_real_filenames():
    """Each preset's (model_key, column_label) must match what the actual
    on-disk filenames parse+classify to — the presets are the only place the
    figures name a model, so a typo here silently loses a bar."""
    from ..compare import classify_column
    cases = {
        "chisq_Classic_biasNone__Normal(0, 1)_4.8s_dt0.005.pkl": FIG1L_SPECS[0],
        "chisq_Classic_biasQ-Val (Offset)_Normal(0, 1)_4.8s_dt0.005.pkl": FIG1L_SPECS[1],
        "chisq_NoiseGain-RewardRate_biasNone__Normal(0, 1)_4.8s_dt0.005.pkl": FIG1L_SPECS[2],
        "chisq_NoiseGain-RewardRate_biasQ-Val (Offset)_Normal(0, 1)_4.8s_dt0.005.pkl": FIG1L_SPECS[3],
        "chisq_Bound-RewardRate_biasNone__Normal(0, 1)_4.8s_dt0.005_scaledB.pkl": SCALE_BOUND_SPECS[1],
        "chisq_Bound-RewardRate_biasQ-Val (Offset)_Normal(0, 1)_4.8s_dt0.005_scaledB.pkl": SCALE_BOUND_SPECS[3],
        "mle_NoiseGain-RewardRate_biasQ-Val (Offset)_Normal(0, 1)_4.8s_dt0.005.pkl": MLE_WEIGHT_SPECS[0],
        "mle_NoiseGain-RewardRate_biasQ-Val (Offset)_Normal(0, 1)_4.8s_dt0.005_mleW1_chi2W0.1.pkl": MLE_WEIGHT_SPECS[1],
        "mle_NoiseGain-RewardRate_biasQ-Val (Offset)_Normal(0, 1)_4.8s_dt0.005_mleW1_chi2W0.5.pkl": MLE_WEIGHT_SPECS[2],
    }
    for filename, spec in cases.items():
        fid = parse_fit_filename(filename)
        _rank, column_label = classify_column(fid)
        assert (fid.model_key, column_label) == (spec.model_key,
                                                 spec.column_label), filename


def test_scale_bound_pairs_share_model_key_across_columns():
    """A·1/A·2 differ only by fitting criterion, likewise A·3/A·4 — that shared
    identity is what makes the comparison meaningful."""
    assert SCALE_BOUND_SPECS[0].model_key == SCALE_BOUND_SPECS[1].model_key
    assert SCALE_BOUND_SPECS[2].model_key == SCALE_BOUND_SPECS[3].model_key
    assert SCALE_BOUND_SPECS[0].column_label != SCALE_BOUND_SPECS[1].column_label


def test_mle_weight_specs_share_one_model():
    assert len({s.model_key for s in MLE_WEIGHT_SPECS}) == 1


def test_chi2_only_reference_matches_the_fig1l_rrq_column():
    """The Chi²-only reference column reuses Fig. 1l's RR+Q Chi²-Noise fit, so
    it must address the exact same (model_key, column_label)."""
    rrq = FIG1L_SPECS[3]                       # "RewardRate\n+ Q-Val"
    assert (CHI2_ONLY_RRQ_SPEC.model_key == rrq.model_key
            == MLE_WEIGHT_SPECS[0].model_key)
    assert CHI2_ONLY_RRQ_SPEC.column_label == rrq.column_label == "Chi²-Noise"


# --------------------------------------------------------------------------
# rows_for_spec_from — reuse a column another figure already collected
# --------------------------------------------------------------------------
def _tidy(spec_label, model_key_, column_label, names=("S1", "S2"), n_iter=2):
    rows = []
    for name in names:
        for i in range(n_iter):
            rows.append({"SpecLabel": spec_label, "ModelKey": model_key_,
                         "ColumnLabel": column_label, "Name": name,
                         "Iteration": i, "R2_Psych": 0.4, "RewardRateCorr": 0.3})
    return pd.DataFrame(rows)


def test_rows_for_spec_from_matches_on_model_and_column_then_relabels():
    frame = _tidy("RewardRate\n+ Q-Val", CHI2_ONLY_RRQ_SPEC.model_key,
                  "Chi²-Noise")
    got = rows_for_spec_from(frame, CHI2_ONLY_RRQ_SPEC)
    assert set(got.SpecLabel) == {"Chi²-only"}          # relabeled
    assert len(got) == len(frame)                       # all rows carried over
    assert set(got.ModelKey) == {CHI2_ONLY_RRQ_SPEC.model_key}


def test_rows_for_spec_from_ignores_other_columns_of_the_same_model():
    key = CHI2_ONLY_RRQ_SPEC.model_key
    frame = pd.concat([_tidy("chi2", key, "Chi²-Noise"),
                       _tidy("mle", key, "MLE")], ignore_index=True)
    got = rows_for_spec_from(frame, CHI2_ONLY_RRQ_SPEC)
    assert set(got.ColumnLabel) == {"Chi²-Noise"}
    assert len(got) == 4


def test_rows_for_spec_from_missing_raises_listing_available():
    frame = _tidy("mle", CHI2_ONLY_RRQ_SPEC.model_key, "MLE")
    with pytest.raises(KeyError, match="Chi²-Noise"):
        rows_for_spec_from(frame, CHI2_ONLY_RRQ_SPEC)


def test_reused_chi2_column_plots_as_first_group_of_four():
    """The notebook flow: borrow the Chi² rows, concat before the MLE rows,
    plot with Chi² as the leading spec."""
    key = CHI2_ONLY_RRQ_SPEC.model_key
    fig1l_like = _tidy("RewardRate\n+ Q-Val", key, "Chi²-Noise")
    chi2_rows = rows_for_spec_from(fig1l_like, CHI2_ONLY_RRQ_SPEC)
    mle = pd.concat([_tidy(s.label, key, s.column_label)
                     for s in MLE_WEIGHT_SPECS], ignore_index=True)
    combined = pd.concat([chi2_rows, mle], ignore_index=True)

    from ..aggregate_plot import plot_aggregates
    ax = plot_aggregates(combined, [CHI2_ONLY_RRQ_SPEC, *MLE_WEIGHT_SPECS],
                         metric_keys=("R2_Psych",))
    # Four bar groups, Chi²-only leftmost.
    xs = sorted(p.get_x() for p in ax.patches)
    assert len(ax.patches) == 4
    assert xs[0] == pytest.approx(min(p.get_x() for p in ax.patches))
    plt.close("all")


# --------------------------------------------------------------------------
# Filename sanitization — spec labels carry newlines for their two-line legends
# --------------------------------------------------------------------------
def test_safe_filename_strips_the_newline_that_broke_savefig():
    # The exact label from the traceback: "RewardRate\n+ Q-Val".
    out = safe_filename(FIG1L_SPECS[3].label)
    assert "\n" not in out and out == "RewardRate + Q-Val"


def test_safe_filename_removes_all_illegal_characters():
    assert not set(safe_filename('a\r\nb\tc/d\\e:f*g?h"i<j>k|l·m')) & set(
        '\r\n\t/\\:*?"<>|·')


def test_safe_filename_is_a_noop_for_a_plain_label():
    assert safe_filename("Classic") == "Classic"


def test_every_preset_label_is_filesystem_safe():
    for spec in (*FIG1L_SPECS, *SCALE_BOUND_SPECS, *MLE_WEIGHT_SPECS):
        out = safe_filename(spec.label)
        assert out and not set(out) & set('\r\n\t/\\:*?"<>|')


# --------------------------------------------------------------------------
# Seed plumbing + collection
# --------------------------------------------------------------------------
def test_single_evaluation_uses_default_seed(stub_sim):
    collect_metrics(_fits(), [_SPEC], df_behavior=None, num_evaluations=1,
                    verbose=False)
    assert [c["seed"] for c in stub_sim] == [0, 0]  # one per subject


def test_multi_evaluation_uses_default_seed_plus_iteration(stub_sim):
    collect_metrics(_fits(subjects=("S1",)), [_SPEC], df_behavior=None,
                    num_evaluations=3, verbose=False)
    assert [c["seed"] for c in stub_sim] == [0, 1, 2]


def test_iteration_zero_matches_single_evaluation(stub_sim):
    """The seeding convention's contract: N>1's iteration 0 reproduces N==1."""
    one = collect_metrics(_fits(), [_SPEC], df_behavior=None,
                          num_evaluations=1, verbose=False)
    many = collect_metrics(_fits(), [_SPEC], df_behavior=None,
                           num_evaluations=3, verbose=False)
    pd.testing.assert_frame_equal(
        one.reset_index(drop=True),
        many[many.Iteration == 0].reset_index(drop=True))


def test_tidy_frame_shape_and_keys(stub_sim):
    df = collect_metrics(_fits(), [_SPEC], df_behavior=None,
                         num_evaluations=3, verbose=False)
    assert len(df) == 2 * 3  # subjects x iterations
    assert list(df.columns[:7]) == ["SpecLabel", "ModelKey", "ColumnLabel",
                                    "Name", "Iteration", "Seed", "NumTrials"]
    assert set(df.SpecLabel) == {"RR+Q"}


def test_sim_cache_retains_only_iteration_zero(stub_sim):
    """Holding every iteration's trial-level frame is what blows memory up."""
    cache = {}
    collect_metrics(_fits(), [_SPEC], df_behavior=None, num_evaluations=4,
                    sim_cache=cache, verbose=False)
    assert sorted(cache["RR+Q"]) == ["S1", "S2"]
    for _loss, sim_df, _bound, _iq, _irr in cache["RR+Q"].values():
        assert (sim_df["seed"] == 0).all()


def test_sim_cache_tuple_shape_matches_notebook(stub_sim):
    cache = {}
    collect_metrics(_fits(subjects=("S1",)), [_SPEC], df_behavior=None,
                    sim_cache=cache, verbose=False)
    loss, sim_df, bound, include_Q, include_RewardRate = cache["RR+Q"]["S1"]
    assert loss == 100.0 and bound == 1.0
    assert include_Q is True and include_RewardRate is True


def test_below_min_trials_subject_dropped_without_extra_sims(stub_sim):
    with pytest.raises(ValueError, match="min_num_trials"):
        collect_metrics(_fits(subjects=("S1",)), [_SPEC], df_behavior=None,
                        num_evaluations=5, min_num_trials=10_000, verbose=False)
    # Bailed after the first simulation rather than running all 5.
    assert len(stub_sim) == 1


def test_num_evaluations_must_be_positive(stub_sim):
    with pytest.raises(ValueError, match="num_evaluations"):
        collect_metrics(_fits(), [_SPEC], df_behavior=None, num_evaluations=0,
                        verbose=False)


def test_missing_include_flags_raises(monkeypatch, stub_sim):
    fits = _fits(subjects=("S1",))
    entry = next(iter(fits.values()))
    entry.subjects["S1"][0].payload.pop("include_Q")
    with pytest.raises(KeyError, match="include_Q"):
        collect_metrics(fits, [_SPEC], df_behavior=None, verbose=False)


# --------------------------------------------------------------------------
# The real subject_metrics on a synthetic-but-realistic simulation frame
# --------------------------------------------------------------------------
def _sim_df(n_sess=6, n_trials=500, seed=0):
    """A simulation frame shaped like the one runAndPlot returns.

    Choice accuracy scales with |DV| so the psychometric fits have real
    structure to bite on; the derived prev-trial / reward-rate columns are
    produced by the same functions runAndPlot uses, rather than hand-faked.
    """
    from ..plotter import _assignPrevTrial
    from ....behavior.rewardrate import calcAvgRewardRate

    rng = np.random.default_rng(seed)
    dv_str_of = {0.9: "Easy", 0.5: "Med", 0.2: "Hard"}
    rows = []
    for s in range(n_sess):
        for t in range(n_trials):
            dv = rng.choice([-0.9, -0.5, -0.2, 0.2, 0.5, 0.9])
            p = 0.5 + 0.45 * abs(dv)
            correct, sim_correct = rng.random() < p, rng.random() < p
            rows.append(dict(
                Name="S1", SessId=f"S1_2024010{s+1}_1", Date=f"2024010{s+1}",
                SessionNum=1, TrialNumber=t, valid=True,
                DVstr=dv_str_of[abs(dv)], DV=dv, DVabs=abs(dv),
                ChoiceCorrect=float(correct),
                ChoiceLeft=float((dv > 0) == correct),
                SimChoiceCorrect=float(sim_correct),
                SimChoiceLeft=float((dv > 0) == sim_correct),
                calcStimulusTime=float(rng.gamma(2, 0.15) + 0.1),
                SimRT=float(rng.gamma(2, 0.15) + 0.1),
                SimStartingPoint=0.0, GUI_TimeOutIncorrectChoice=0.0))
    df = pd.DataFrame(rows)
    df = calcAvgRewardRate(df, choice_cols=["ChoiceCorrect", "SimChoiceCorrect"],
                           rr_postfixs=["", "Sim"], groupby_cols=["SessId"])
    df = _assignPrevTrial(df)
    return df[df.valid]


def test_subject_metrics_produces_every_metric_without_nans():
    """Exercises the ported collectMetric math end-to-end — the psychometric
    fits, quantile splits, reward-rate curves and motor bias all really run."""
    row = aggregate.subject_metrics("S1", _sim_df())
    expected = {"Name", "NumTrials", "R2_WinLose", "R2_PrevOutcomeCount",
                "PrevOutcomeCurQuantileReal", "PrevOutcomeCurQuantileModel",
                "R2_PrevOutcomeQuantile", "R2_RewardRate", "RewardRate5Real",
                "RewardRate5Model", "RewardRateCorr", "MotorBiasAllReal",
                "MotorBiasAllModel", "MotorBiasFastReal", "MotorBiasFastModel",
                "MotorBiasSlowReal", "MotorBiasSlowModel", "PsychAbsReal",
                "PsychAbsModel", "PsychReal", "PsychModel", "R2_Psych",
                "R2_Slow", "R2_Fast", "R2_Total", "WinStayReal",
                "WinStayModel", "LoseSwitchReal", "LoseSwitchModel"}
    assert set(row) == expected
    scalars = {k: v for k, v in row.items()
               if not isinstance(v, (pd.Series, str))}
    assert not [k for k, v in scalars.items() if v is None or np.isnan(v)]


def test_subject_metrics_r2_total_is_fast_plus_slow():
    row = aggregate.subject_metrics("S1", _sim_df())
    assert row["R2_Total"] == pytest.approx(row["R2_Fast"] + row["R2_Slow"])


def test_subject_metrics_reward_rate_series_share_bins():
    row = aggregate.subject_metrics("S1", _sim_df())
    assert len(row["RewardRate5Real"]) and len(row["RewardRate5Model"])
    assert not np.isnan(row["RewardRateCorr"])


# --------------------------------------------------------------------------
# Seed plumbing through the real runAndPlot -> makeOneRun path
# --------------------------------------------------------------------------
def test_run_and_plot_forwards_seed_to_make_one_run(monkeypatch):
    """``runAndPlot`` historically dropped the seed on the floor, pinning every
    simulation to trajectory 0; repeat evaluation depends on it forwarding."""
    from .. import plotter
    seen = {}

    def fake_make_one_run(df, **kwargs):
        seen.update(kwargs)
        return 1.0, df

    monkeypatch.setattr(plotter, "makeOneRun", fake_make_one_run)
    monkeypatch.setattr(plotter, "calcAvgRewardRate", lambda df, **kw: df)
    monkeypatch.setattr(plotter, "_assignPrevTrial", lambda df: df)

    # runAndPlot narrows to a fixed keep_cols list before returning.
    cols = ["Name", "SessId", "TrialNumber", "valid", "DVstr", "DV", "DVabs",
            "ChoiceCorrect", "ChoiceLeft", "GUI_TimeOutIncorrectChoice",
            "SimChoiceCorrect", "SimChoiceLeft", "calcStimulusTime", "SimRT",
            "SimStartingPoint", "Date", "SessionNum"]
    df = pd.DataFrame({c: [1] for c in cols})
    df["Name"], df["valid"] = "S1", True
    common = dict(fig=None, axs=None, include_Q=False, include_RewardRate=False,
                  biasFn=None, driftFn=None, noiseFn=None, plot_bias_dir=False,
                  psych_plot=None, DRIFT_COEF=1.0, NOISE_SIGMA=1.0, BOUND=1.0,
                  ALPHA=np.nan, BETA=np.nan, NON_DECISION_TIME=0.1, t_dur=4.8,
                  dt=0.005, is_small_fig_mode=False, verbose=False)

    plotter.runAndPlot(df.copy(), seed=7, **common)
    assert seen["seed"] == 7

    seen.clear()
    plotter.runAndPlot(df.copy(), **common)   # default keeps the old behavior
    assert seen["seed"] == 0


# --------------------------------------------------------------------------
# Plot: statistics + layout
# --------------------------------------------------------------------------
def _metrics_frame(num_evals, specs=(_SPEC,)):
    rows = []
    for spec in specs:
        for subject, base in (("S1", 0.4), ("S2", 0.8)):
            for i in range(num_evals):
                rows.append({"SpecLabel": spec.label, "Name": subject,
                             "Iteration": i, "R2_Psych": base + 0.1 * i,
                             "RewardRateCorr": base})
    return pd.DataFrame(rows)


def _bar_containers(ax):
    return [c for c in ax.containers if isinstance(c, BarContainer)]


def _dots_container(ax):
    """The last non-bar ErrorbarContainer == the subject dots.

    ``ax.bar(yerr=…)`` registers its own ErrorbarContainer *before* the
    BarContainer, so index-based lookup would grab the wrong one.
    """
    return [c for c in ax.containers if not isinstance(c, BarContainer)][-1]


def _yerr_halfwidth(bar_container):
    lo, hi = bar_container.errorbar.lines[2][0].get_segments()[0][:, 1]
    return (hi - lo) / 2


def test_bar_is_mean_over_subject_means_with_sem_over_subjects():
    """n is the number of subjects, not subjects x evaluations."""
    df = _metrics_frame(num_evals=3)
    ax = plot_aggregates(df, [_SPEC], metric_keys=("R2_Psych",))
    # subject means: S1 = (0.4+0.5+0.6)/3 = 0.5 ; S2 = 0.9 -> bar = 0.7
    assert ax.patches[0].get_height() == pytest.approx(0.7)
    from scipy import stats as _stats
    # SEM over the two subject means, NOT over the six subject x eval values.
    expected_sem = _stats.sem(np.array([0.5, 0.9]))
    assert _yerr_halfwidth(_bar_containers(ax)[0]) == pytest.approx(expected_sem)
    assert expected_sem != pytest.approx(_stats.sem(df.R2_Psych.values))
    plt.close("all")


def test_multi_eval_dots_are_subject_mean_with_std_whisker():
    df = _metrics_frame(num_evals=3)
    ax = plot_aggregates(df, [_SPEC], metric_keys=("R2_Psych",))
    dots = _dots_container(ax)
    assert list(ax.lines[0].get_ydata()) == pytest.approx([0.5, 0.9])
    assert dots.has_yerr
    # STD across each subject's 3 evaluations (0.4/0.5/0.6 and 0.8/0.9/1.0).
    expected_std = pd.Series([0.4, 0.5, 0.6]).std()
    lo, hi = dots.lines[2][0].get_segments()[0][:, 1]
    assert (hi - lo) / 2 == pytest.approx(expected_std)
    plt.close("all")


def test_single_eval_dots_share_x_and_have_no_whisker():
    df = _metrics_frame(num_evals=1)
    ax = plot_aggregates(df, [_SPEC], metric_keys=("R2_Psych",))
    assert list(ax.lines[0].get_xdata()) == pytest.approx([0.0, 0.0])
    assert not _dots_container(ax).has_yerr
    plt.close("all")


def test_multi_eval_dots_span_the_bar_width():
    df = _metrics_frame(num_evals=3)
    ax = plot_aggregates(df, [_SPEC], metric_keys=("R2_Psych",))
    xs = ax.lines[0].get_xdata()
    assert min(xs) == pytest.approx(-BAR_WIDTH / 2)
    assert max(xs) == pytest.approx(BAR_WIDTH / 2)
    plt.close("all")


def test_gap_after_shifts_following_groups():
    a = EvalSpec("A", _SPEC.model_key, "Chi²-Noise", "blue", gap_after=0.5)
    b = EvalSpec("B", _SPEC.model_key, "Chi²-Noise", "green")
    df = pd.concat([_metrics_frame(1, specs=(a,)), _metrics_frame(1, specs=(b,))])
    ax = plot_aggregates(df, [a, b], metric_keys=("R2_Psych", "RewardRateCorr"))
    xs = [p.get_x() + p.get_width() / 2 for p in ax.patches]
    # Group A at 0,1; +2 metrics +1 inter-group gap +0.5 requested -> B at 3.5,4.5
    assert xs == pytest.approx([0, 1, 3.5, 4.5])
    plt.close("all")


def test_no_gap_reproduces_original_group_spacing():
    """Without gap_after the layout must match cell 24's `global_offset_x += 3`."""
    specs = [EvalSpec(n, _SPEC.model_key, "Chi²-Noise", "blue")
             for n in ("A", "B", "C", "D")]
    df = pd.concat([_metrics_frame(1, specs=(s,)) for s in specs])
    ax = plot_aggregates(df, specs, metric_keys=("R2_Psych", "RewardRateCorr"))
    xs = [p.get_x() + p.get_width() / 2 for p in ax.patches]
    assert xs == pytest.approx([0, 1, 3, 4, 6, 7, 9, 10])
    plt.close("all")


def test_title_states_std_and_sem_conventions():
    df = _metrics_frame(num_evals=20)
    ax = plot_aggregates(df, [_SPEC], metric_keys=("R2_Psych",))
    title = ax.get_title()
    assert "SEM" in title and "STD" in title and "20 evaluations" in title
    plt.close("all")


def test_missing_spec_in_metrics_raises():
    df = _metrics_frame(num_evals=1)
    other = EvalSpec("absent", _SPEC.model_key, "Chi²-Noise", "gray")
    with pytest.raises(ValueError, match="No rows for spec"):
        plot_aggregates(df, [other], metric_keys=("R2_Psych",))
    plt.close("all")


# --------------------------------------------------------------------------
# Plot-time subsampling: collect a big N once, choose n <= N when plotting
# --------------------------------------------------------------------------
def test_subsample_keeps_the_first_n_iterations():
    df = subsample_evaluations(_metrics_frame(num_evals=10), 3)
    assert sorted(df.Iteration.unique()) == [0, 1, 2]


def test_subsample_none_keeps_all():
    df = _metrics_frame(num_evals=4)
    assert len(subsample_evaluations(df, None)) == len(df)


def test_subsample_of_ten_equals_collecting_three():
    """The contract that makes plot-time subsampling sound: iteration i is
    seed i, so the first n rows are exactly what collecting n would produce."""
    ax_sub = plot_aggregates(_metrics_frame(num_evals=10), [_SPEC],
                             metric_keys=("R2_Psych",), num_evaluations=3)
    ax_col = plot_aggregates(_metrics_frame(num_evals=3), [_SPEC],
                             metric_keys=("R2_Psych",))
    assert (ax_sub.patches[0].get_height()
            == pytest.approx(ax_col.patches[0].get_height()))
    assert (list(ax_sub.lines[0].get_ydata())
            == pytest.approx(list(ax_col.lines[0].get_ydata())))
    plt.close("all")


def test_subsample_to_one_renders_as_single_evaluation():
    ax = plot_aggregates(_metrics_frame(num_evals=10), [_SPEC],
                         metric_keys=("R2_Psych",), num_evaluations=1)
    assert list(ax.lines[0].get_xdata()) == pytest.approx([0.0, 0.0])
    assert not _dots_container(ax).has_yerr
    plt.close("all")


def test_subsample_title_reports_used_not_collected_count():
    ax = plot_aggregates(_metrics_frame(num_evals=20), [_SPEC],
                         metric_keys=("R2_Psych",), num_evaluations=4)
    assert "4 evaluations" in ax.get_title()
    assert "20 evaluations" not in ax.get_title()
    plt.close("all")


def test_subsample_more_than_collected_raises_naming_both():
    with pytest.raises(ValueError, match="only holds 3"):
        subsample_evaluations(_metrics_frame(num_evals=3), 5)


# --------------------------------------------------------------------------
# Disk cache
# --------------------------------------------------------------------------
@pytest.fixture
def cache_env(tmp_path, stub_sim):
    """A cache dir plus a real (empty) fit file the freshness check can stat."""
    result_dir = tmp_path / "fits"
    result_dir.mkdir()
    fname = ("chisq_NoiseGain-RewardRate_biasQ-Val (Offset)_"
             "Normal(0, 1)_4.8s_dt0.005.pkl")
    (result_dir / fname).write_bytes(b"x")
    return types.SimpleNamespace(
        calls=stub_sim, cache_dir=tmp_path / "cache", result_dir=result_dir,
        fit_fp=result_dir / fname,
        kwargs=dict(cache_name="t", cache_dir=tmp_path / "cache",
                    result_dir=result_dir, df_behavior=None, verbose=False))


def _collect(env, **over):
    kw = {**env.kwargs, **over}
    return load_or_collect_metrics(_fits(), [_SPEC], **kw)


def test_cache_cold_collects_then_warm_does_not(cache_env):
    first = _collect(cache_env, num_evaluations=2)
    assert len(cache_env.calls) == 4          # 2 subjects x 2 evals
    cache_env.calls.clear()

    second = _collect(cache_env, num_evaluations=2)
    assert cache_env.calls == []              # served entirely from disk
    pd.testing.assert_frame_equal(first, second)


def test_cache_force_recompute_recollects(cache_env):
    _collect(cache_env, num_evaluations=1)
    cache_env.calls.clear()
    _collect(cache_env, num_evaluations=1, force_recompute=True)
    assert len(cache_env.calls) == 2


def test_cache_serves_a_smaller_request_without_recomputing(cache_env):
    _collect(cache_env, num_evaluations=5)
    cache_env.calls.clear()
    df = _collect(cache_env, num_evaluations=3)
    assert cache_env.calls == []
    # The full 5 come back; the plot subsamples to 3.
    assert df.Iteration.nunique() == 5


def test_cache_recomputes_for_a_larger_request(cache_env):
    _collect(cache_env, num_evaluations=2)
    cache_env.calls.clear()
    _collect(cache_env, num_evaluations=4)
    assert len(cache_env.calls) == 8          # 2 subjects x 4 evals


def test_cache_recomputes_when_n_psych_fits_changes(cache_env):
    _collect(cache_env, num_evaluations=1, n_psych_fits=N_PSYCH_FITS)
    cache_env.calls.clear()
    _collect(cache_env, num_evaluations=1, n_psych_fits=3)
    assert len(cache_env.calls) == 2


def test_cache_recomputes_when_specs_change_under_same_name(cache_env):
    """A stale cache_name must not silently serve a different model.

    Both columns exist here, so resolve_spec succeeds for either spec — the
    only thing that can force the recollect is the spec_key guard itself.
    """
    fits = _fits()                      # has the Chi²-Noise column
    entry = next(iter(fits.values()))
    noise_col = entry.subjects["S1"][0]
    bound_col = ColumnFit(fid=noise_col.fid, payload=_payload(),
                          filename=noise_col.filename,
                          column_label="Chi²-Bound", order_rank=101.0)
    for subject in entry.subjects:
        entry.subjects[subject] = [entry.subjects[subject][0], bound_col]

    kw = {**cache_env.kwargs, "num_evaluations": 1}
    load_or_collect_metrics(fits, [_SPEC], **kw)
    assert len(cache_env.calls) == 2
    cache_env.calls.clear()

    other = EvalSpec("RR+Q", _SPEC.model_key, "Chi²-Bound", "green")
    df = load_or_collect_metrics(fits, [other], **kw)
    assert len(cache_env.calls) == 2, "spec_key guard did not reject the cache"
    assert set(df.ColumnLabel) == {"Chi²-Bound"}


def test_cache_recomputes_when_a_fit_file_is_newer(cache_env):
    _collect(cache_env, num_evaluations=1)
    cache_env.calls.clear()
    cache_fp = cache_env.cache_dir / "metrics_t.pkl"
    future = cache_fp.stat().st_mtime + 1_000
    os.utime(cache_env.fit_fp, (future, future))
    _collect(cache_env, num_evaluations=1)
    assert len(cache_env.calls) == 2


def test_cache_kept_when_fit_file_is_older(cache_env):
    _collect(cache_env, num_evaluations=1)
    cache_env.calls.clear()
    cache_fp = cache_env.cache_dir / "metrics_t.pkl"
    past = cache_fp.stat().st_mtime - 1_000
    os.utime(cache_env.fit_fp, (past, past))
    _collect(cache_env, num_evaluations=1)
    assert cache_env.calls == []


def test_cache_unreadable_file_recollects_rather_than_raising(cache_env):
    _collect(cache_env, num_evaluations=1)
    (cache_env.cache_dir / "metrics_t.pkl").write_bytes(b"not a pickle")
    cache_env.calls.clear()
    _collect(cache_env, num_evaluations=1)
    assert len(cache_env.calls) == 2


def test_sim_cache_filled_on_collect_and_empty_on_cache_hit(cache_env):
    """A cache hit runs no simulation, so the per-subject figure cells have
    nothing to draw — they must check for this."""
    fresh = {}
    _collect(cache_env, num_evaluations=1, sim_cache=fresh)
    assert sorted(fresh["RR+Q"]) == ["S1", "S2"]

    on_hit = {}
    _collect(cache_env, num_evaluations=1, sim_cache=on_hit)
    assert on_hit == {}
