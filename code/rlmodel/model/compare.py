"""model_compare engine — per-subject × model fitting-criteria comparison grid.

For one subject × model, render a grid whose **columns are the fitting
criteria** found on disk (pure MLE, joint MLE+Chi², Chi²-Noise, Chi²-Bound)
and whose **rows are diagnostic plots** (loss distributions + the existing
behavioral panels from ``plotter.py``). ``model_compare.ipynb`` stays thin:
it loads data, sets config globals, and calls :func:`interactive_viewer`
(one subject × model) or :func:`run_batch` (every combination on disk).

Model identity vs. columns
--------------------------
A "model" (one dropdown entry) is the abstract model — user-facing drift
*alias* + bias + noise + timing + asym variant (see
``mle_reeval.FitFileId.model_key``). ``_scaledB`` and the joint weight suffix
are *column* axes, not identity, which is what lets Chi²-Noise
(``NoiseGain-RewardRate``) and Chi²-Bound (``Bound-RewardRate`` + ``_scaledB``)
— different drift strings — land in the same figure.

Each column runs two forward passes from its fitted params: a pure-MLE
re-evaluation (``mle_reeval``) for the header MLE-Score + the loss-distribution
rows, and a Chi²-style simulation (``plotter.runAndPlot(axs=None)``) for the
behavioral rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from inspect import signature
import pathlib
import pickle
import re

import numpy as np
import matplotlib.pyplot as plt

from . import mle_reeval
from .mle_reeval import (parse_fit_filename, fitted_params_from_result,
                         build_mle_config, evaluate_params_under_mle, FitFileId,
                         prepare_behavior_df)  # re-exported for the notebook
from .plotter import (runAndPlot, _plotHist, _plotPsychs, _plotRewardRateVsRt,
                      _plotQDist, _plotRewardRateDist, _plotBiasDist)
from .util import (PsychometricPlot, biasFnColsAndKwargs, driftFnColsAndKwargs,
                   noiseFnColsAndKwargs)
from .drift import DRIFT_FN_DICT
from .bias import BIAS_FN_DICT
from .noise import NOISE_FN_DICT
from .initvals import InitVals
from ...figcode.psychometric import _psychAxes


DEFAULT_RESULT_DIR = "../../data/RLModel"

# Row identity + default on/off. Header is always on; the two learning-rate
# distribution rows are additionally gated by whether the model learns the
# quantity (Q / reward-rate). Order here is the top-to-bottom grid order.
DEFAULT_ROW_FLAGS = {
    "losses_dist": True,
    "hist_by_loss": True,
    "rt_corr_incorr": True,
    "rt_direction": True,
    "psychometric": True,
    "reward_rate": True,
    "beta_dist": True,     # R-learning only (include_RewardRate)
    "alpha_dist": True,    # Q-learning only (include_Q)
    "bias_dist": True,
}


# --------------------------------------------------------------------------
# Column classification
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnFit:
    fid: FitFileId
    payload: dict
    filename: str
    column_label: str
    order_rank: float


def classify_column(fid: FitFileId):
    """Map a parsed fit to ``(order_rank, column_label)`` or ``None`` to skip.

    The four v1 criteria families, in render order:
      MLE (0) < joint MLE+Chi² by ascending Chi² weight (1+c) <
      Chi²-Noise (100) < Chi²-Bound (101).
    Combos outside these (mle+scaledB, joint+scaledB, …) return ``None``.
    """
    if fid.fit_mode == "mle" and not fid.scaled_bound:
        if fid.chi2_weight == 0.0:
            return 0.0, "MLE"
        return (1.0 + fid.chi2_weight,
                f"MLE={fid.mle_weight:g}, Chi²={fid.chi2_weight:g}")
    if fid.fit_mode == "chisq":
        if not fid.scaled_bound:
            return 100.0, "Chi²-Noise"
        return 101.0, "Chi²-Bound"
    return None


# --------------------------------------------------------------------------
# Disk discovery
# --------------------------------------------------------------------------
@dataclass
class ModelEntry:
    model_key: str
    label: str
    fid_example: FitFileId
    # subject -> ordered list[ColumnFit]
    subjects: dict = field(default_factory=dict)


def discover_fits(result_dir=DEFAULT_RESULT_DIR, *, verbose=True):
    """Glob ``result_dir`` for ``mle_*``/``chisq_*`` pickles and build the
    ``{model_key: ModelEntry}`` index the viewer / batch consume.

    Each top-level pickle is a ``{subject: payload}`` dict. Files whose name
    classifies to a v1 column are indexed; others are skipped (reported when
    ``verbose``). Unreadable pickles are skipped with a note.
    """
    result_dir = pathlib.Path(result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"Result dir not found: {result_dir}")
    models: dict[str, ModelEntry] = {}
    skipped = []
    for fp in sorted(result_dir.glob("*.pkl")):
        if not (fp.name.startswith("mle_") or fp.name.startswith("chisq_")):
            continue
        try:
            fid = parse_fit_filename(fp.name)
        except (NotImplementedError, ValueError, IndexError) as exc:
            skipped.append((fp.name, f"unparsed ({exc})"))
            continue
        col = classify_column(fid)
        if col is None:
            skipped.append((fp.name, "not a v1 column family"))
            continue
        order_rank, column_label = col
        try:
            with fp.open("rb") as f:
                data = pickle.load(f)
        except Exception as exc:  # noqa: BLE001 — tolerate unreadable pickles
            skipped.append((fp.name, f"unreadable ({exc!r})"))
            continue
        if not isinstance(data, dict):
            skipped.append((fp.name, "not a subject->payload dict"))
            continue
        entry = models.get(fid.model_key)
        if entry is None:
            entry = ModelEntry(model_key=fid.model_key, label=fid.model_label,
                               fid_example=fid)
            models[fid.model_key] = entry
        for subject, payload in data.items():
            col_fit = ColumnFit(fid=fid, payload=payload, filename=fp.name,
                                column_label=column_label, order_rank=order_rank)
            cols = entry.subjects.setdefault(str(subject), [])
            # De-dup: if a column with the same label already exists for this
            # subject (two files map to one label), keep the first, note it.
            if any(c.column_label == column_label for c in cols):
                skipped.append((fp.name,
                                f"duplicate column {column_label!r} for "
                                f"{subject!r} (kept earlier)"))
                continue
            cols.append(col_fit)
    # Sort every subject's columns into render order.
    for entry in models.values():
        for cols in entry.subjects.values():
            cols.sort(key=lambda c: c.order_rank)
    if verbose:
        print(f"Discovered {len(models)} model(s), "
              f"{sum(len(e.subjects) for e in models.values())} "
              f"subject×model combos.")
        for name, why in skipped:
            print(f"  skipped {name}: {why}")
    return models


def list_models(fits):
    """``[(model_key, label), …]`` sorted by label for a dropdown."""
    return sorted(((e.model_key, e.label) for e in fits.values()),
                  key=lambda kv: kv[1])


def subjects_for(fits, model_key):
    """Sorted subjects that have at least one column for ``model_key``."""
    entry = fits.get(model_key)
    return sorted(entry.subjects) if entry else []


# --------------------------------------------------------------------------
# Include flags + param routing for the Chi²-style simulation
# --------------------------------------------------------------------------
def _include_flags(payload):
    """(include_Q, include_RewardRate) — from the payload's stored flags,
    falling back to column detection off the drift/bias/noise fns."""
    iq = payload.get("include_Q") if isinstance(payload, dict) else None
    irr = payload.get("include_RewardRate") if isinstance(payload, dict) else None
    if iq is not None and irr is not None:
        return bool(iq), bool(irr)
    return None, None


def _routed_kwargs(params, biasFn, driftFn, noiseFn):
    """Route an UPPERCASE params dict into the bias/drift/noise-fn kwargs and
    the top-level ``runAndPlot`` kwargs, mirroring ``visualize.updateGUI``.

    The fittable-scalar fn params are uppercase (``Q_VAL_DECAY_RATE`` …), so
    membership against the fns' float-kwarg lists routes them directly.
    ``DRIFT_COEF`` / ``NOISE_SIGMA`` are COMMON_ARGS (excluded from those
    lists) → they land in the top-level kwargs. The frozen scale axis absent
    from the fit (``BOUND`` for a noise-scaled fit, ``NOISE_SIGMA`` for a
    scaled-bound fit) defaults to 1.0 (``_BOUND_FIXED`` / ``_NOISE_FIXED``);
    the remaining core params default from ``InitVals``.
    """
    biasFn_df_cols, biasFn_kwargs_li = biasFnColsAndKwargs(biasFn)
    driftFn_df_cols, driftFn_kwargs_li = driftFnColsAndKwargs(driftFn)
    noiseFn_df_cols, noiseFn_kwargs_li = noiseFnColsAndKwargs(noiseFn)
    top_names = set(signature(runAndPlot).parameters)

    biasFn_kwargs, driftFn_kwargs, noiseFn_kwargs, top_kwargs = {}, {}, {}, {}
    for name, value in params.items():
        if name in biasFn_kwargs_li:
            biasFn_kwargs[name] = value
        if name in driftFn_kwargs_li:
            driftFn_kwargs[name] = value
        if name in noiseFn_kwargs_li:
            noiseFn_kwargs[name] = value
        if name in top_names:
            top_kwargs[name] = value

    top_kwargs.setdefault("BOUND", 1.0)         # frozen counterparts
    top_kwargs.setdefault("NOISE_SIGMA", 1.0)
    iv = InitVals()
    for pname in ("DRIFT_COEF", "ALPHA", "BETA", "NON_DECISION_TIME"):
        top_kwargs.setdefault(pname, iv.get(pname).Default)
    return dict(biasFn_kwargs=biasFn_kwargs, biasFn_df_cols=biasFn_df_cols,
                driftFn_kwargs=driftFn_kwargs, driftFn_df_cols=driftFn_df_cols,
                noiseFn_kwargs=noiseFn_kwargs, noiseFn_df_cols=noiseFn_df_cols,
                top_kwargs=top_kwargs)


# --------------------------------------------------------------------------
# The two forward passes (module-level so tests can monkeypatch them)
# --------------------------------------------------------------------------
def _compute_mle(subject, fid, payload, df_behavior, include_Q,
                 include_RewardRate, terminal_c, lapse_override):
    """Re-evaluate the column's params under pure MLE. Returns
    ``(mle_df|None, neg_loglik|None, lapse|None, error|None)``."""
    try:
        params = fitted_params_from_result(payload)
        config = build_mle_config(
            fid, include_Q=include_Q, include_RewardRate=include_RewardRate,
            mle_terminal_c=terminal_c)
        subject_df = df_behavior[df_behavior.Name == subject].copy()
        res = evaluate_params_under_mle(
            params, subject_df, config, lapse_override=lapse_override)
        lapse = params.get("LAPSE_RATE") if fid.fit_mode == "mle" else None
        if lapse_override is not None and fid.fit_mode == "mle":
            lapse = float(lapse_override)
        return res.mle_df, float(res.neg_loglik), lapse, None
    except Exception as exc:  # noqa: BLE001 — degrade one column, not the grid
        return None, None, None, f"{type(exc).__name__}: {exc}"


def _compute_sim(subject, fid, payload, df_behavior, include_Q,
                 include_RewardRate, seed=0):
    """Run the Chi²-style forward simulation for the column's params (no
    plotting). Returns ``(sim_df|None, bound, biasFn_kwargs, error|None)``.

    ``seed`` selects the simulation's RNG trajectory (see
    ``logic.makeOneRun``). It defaults to 0 — the historical single
    trajectory — and is varied per iteration by ``aggregate.collect_metrics``
    when repeat-evaluating a subject.
    """
    try:
        biasFn = BIAS_FN_DICT[fid.bias]
        driftFn = DRIFT_FN_DICT[fid.drift]
        noiseFn = NOISE_FN_DICT[fid.noise]
        params = fitted_params_from_result(payload)
        routed = _routed_kwargs(params, biasFn, driftFn, noiseFn)
        subject_df = df_behavior[df_behavior.Name == subject].copy()
        _, sim_df = runAndPlot(
            subject_df, fig=None, axs=None,
            include_Q=include_Q, include_RewardRate=include_RewardRate,
            biasFn=biasFn, driftFn=driftFn, noiseFn=noiseFn,
            plot_bias_dir=False, psych_plot=PsychometricPlot._None,
            t_dur=fid.t_dur, dt=fid.dt, is_small_fig_mode=False,
            biasFn_kwargs=routed["biasFn_kwargs"],
            biasFn_df_cols=routed["biasFn_df_cols"],
            driftFn_kwargs=routed["driftFn_kwargs"],
            driftFn_df_cols=routed["driftFn_df_cols"],
            noiseFn_kwargs=routed["noiseFn_kwargs"],
            noiseFn_df_cols=routed["noiseFn_df_cols"],
            seed=seed,
            verbose=False, **routed["top_kwargs"])
        return sim_df, routed["top_kwargs"]["BOUND"], routed["biasFn_kwargs"], None
    except Exception as exc:  # noqa: BLE001 — degrade one column, not the grid
        return None, 1.0, {}, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# Per-column context + row rendering
# --------------------------------------------------------------------------
@dataclass
class _ColumnCtx:
    subject: str
    fid: FitFileId
    column_label: str
    include_Q: bool
    include_RewardRate: bool
    t_dur: float
    dt: float
    mle_df: object = None
    mle_score: float | None = None
    lapse: float | None = None
    mle_error: str | None = None
    sim_df: object = None
    sim_error: str | None = None
    bound: float = 1.0
    biasFn_kwargs: dict = field(default_factory=dict)


def _blank(ax, msg=""):
    ax.set_xticks([])
    ax.set_yticks([])
    if msg:
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize="x-small",
                color="0.5", transform=ax.transAxes, wrap=True)


def _fmt(value, spec=",.1f"):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return format(value, spec) if np.isfinite(value) else "n/a"


def _spines(ax, *, left=True, right=True, top=True, bottom=True):
    """Toggle spine visibility on ``ax``. Hidden left/bottom sides also drop
    their ticks, tick labels, and axis label so nothing is left floating with
    no spine to anchor it."""
    for side, on in (("left", left), ("right", right),
                     ("top", top), ("bottom", bottom)):
        ax.spines[side].set_visible(on)
    ax.tick_params(left=left, labelleft=left, bottom=bottom, labelbottom=bottom)
    if not left:
        ax.set_ylabel("")
    if not bottom:
        ax.set_xlabel("")


# --- individual row renderers (each takes a list of axes + the column ctx) --
def _row_header(axes, ctx):
    ax = axes[0]
    ax.axis("off")
    lines = [ctx.column_label, f"MLE-Score: {_fmt(ctx.mle_score)}"]
    if ctx.fid.fit_mode == "mle":
        lines.append(f"λ: {_fmt(ctx.lapse, '.3f')}")
    ax.text(0.5, 0.5, "\n".join(lines), ha="center", va="center",
            fontsize="medium", fontweight="bold", transform=ax.transAxes)


def _row_losses_dist(axes, ctx):
    if ctx.mle_df is None:
        _blank(axes[0], ctx.mle_error or "no MLE df")
        return
    mle_reeval.plot_loss_distribution(axes[0], ctx.mle_df)
    _spines(axes[0], left=False, top=False, right=False)  # keep bottom only


def _row_hist_by_loss(axes, ctx):
    if ctx.mle_df is None:
        _blank(axes[0], ctx.mle_error or "no MLE df")
        return
    mle_reeval.plot_rt_hist_colored_by_loss(axes[0], ctx.mle_df, t_dur=ctx.t_dur)
    _spines(axes[0], left=False, top=False, right=False)  # keep bottom only


def _row_rt_corr_incorr(axes, ctx):
    if ctx.sim_df is None:
        _blank(axes[0], ctx.sim_error or "no sim"); _blank(axes[1])
        return
    _plotHist(ctx.sim_df, axes[0], axes[1], "ChoiceCorrect", "SimChoiceCorrect",
              ctx.t_dur, ctx.dt, legend=True)
    axes[0].set_ylabel("Correct")
    axes[1].set_ylabel("Incorrect")
    for ax in axes:  # back-to-back hist: keep left only
        _spines(ax, top=False, right=False, bottom=False)


def _row_rt_direction(axes, ctx):
    if ctx.sim_df is None:
        _blank(axes[0], ctx.sim_error or "no sim"); _blank(axes[1])
        return
    _plotHist(ctx.sim_df, axes[0], axes[1], "ChoiceLeft", "SimChoiceLeft",
              ctx.t_dur, ctx.dt)
    axes[0].set_ylabel("Left")
    axes[1].set_ylabel("Right")
    for ax in axes:  # back-to-back hist: keep left only
        _spines(ax, top=False, right=False, bottom=False)


def _row_psychometric(axes, ctx):
    ax = axes[0]
    if ctx.sim_df is None:
        _blank(ax, ctx.sim_error or "no sim")
        return
    _psychAxes(ax=ax)
    _plotPsychs(ctx.sim_df, ax, PsychometricPlot.SlowFast)
    ax.set_title("Psychometric (fast/slow)", y=0.9)


def _row_reward_rate(axes, ctx):
    if ctx.sim_df is None:
        _blank(axes[0], ctx.sim_error or "no sim")
        return
    _plotRewardRateVsRt(ctx.sim_df, axes[0], ctx.subject)
    _spines(axes[0], top=False, right=False)


def _row_beta_dist(axes, ctx):
    if ctx.sim_df is None:
        _blank(axes[0], ctx.sim_error or "no sim")
        return
    _plotRewardRateDist(ctx.sim_df, axes[0], include_RewardRate=True)
    _spines(axes[0], top=False, right=False)


def _row_alpha_dist(axes, ctx):
    if ctx.sim_df is None:
        _blank(axes[0], ctx.sim_error or "no sim")
        return
    _plotQDist(ctx.sim_df, axes[0], plot_bias_dir=False, include_Q=True,
               bound=ctx.bound, biasFn_kwargs=ctx.biasFn_kwargs,
               is_small_fig_mode=False)
    _spines(axes[0], top=False, right=False)


def _row_bias_dist(axes, ctx):
    if ctx.sim_df is None:
        _blank(axes[0], ctx.sim_error or "no sim")
        return
    _plotBiasDist(ctx.sim_df, axes[0], ctx.bound, ctx.biasFn_kwargs)
    _spines(axes[0], top=False, right=False)


@dataclass(frozen=True)
class _RowSpec:
    name: str
    label: str
    n_sub: int          # vertical sub-axes per cell (1 or 2)
    needs_sim: bool      # True → the Chi²-style sim df is required
    require: str = ""    # "" | "Q" | "RR" — extra model gate


# Fixed decorations, in inches, so each stays a small constant strip regardless
# of the grid's row/column count (rather than a figure-fraction that balloons on
# a large grid and swallows space):
#   _HEADER_ROW_IN : height of the text-only per-column header strip (row 0).
#   _ROW_LABEL_IN  : width of the rotated per-row label strip down the left.
#   _TITLE_IN      : height reserved at the top for the suptitle.
#   _XLABEL_IN     : height reserved at the bottom for the last row's x-axis.
#   _RIGHT_PAD_IN  : small right-edge gutter.
_HEADER_ROW_IN = 0.6
_ROW_LABEL_IN = 0.85
_TITLE_IN = 0.5
_XLABEL_IN = 0.5
_RIGHT_PAD_IN = 0.2

# Grid row order (top→bottom). Header first, always on.
_ROWS = [
    _RowSpec("header", "", 1, False),
    _RowSpec("losses_dist", "Loss dist", 1, False),
    _RowSpec("hist_by_loss", "RT by loss", 1, False),
    _RowSpec("rt_corr_incorr", "RT corr/incorr", 2, True),
    _RowSpec("rt_direction", "RT direction", 2, True),
    _RowSpec("psychometric", "Psychometric", 1, True),
    _RowSpec("reward_rate", "Reward rate", 1, True),
    _RowSpec("beta_dist", "β / R-value dist", 1, True, require="RR"),
    _RowSpec("alpha_dist", "α / Q-value dist", 1, True, require="Q"),
    _RowSpec("bias_dist", "Bias dist", 1, True),
]

# Renderer dispatch kept separate from ``_ROWS`` (looked up by name at render
# time) so tests can monkeypatch the whole map to no-ops and validate the grid
# scaffolding without invoking matplotlib drawing.
_ROW_RENDERERS = {
    "header": _row_header,
    "losses_dist": _row_losses_dist,
    "hist_by_loss": _row_hist_by_loss,
    "rt_corr_incorr": _row_rt_corr_incorr,
    "rt_direction": _row_rt_direction,
    "psychometric": _row_psychometric,
    "reward_rate": _row_reward_rate,
    "beta_dist": _row_beta_dist,
    "alpha_dist": _row_alpha_dist,
    "bias_dist": _row_bias_dist,
}


def _active_rows(row_flags, include_Q, include_RewardRate):
    flags = {**DEFAULT_ROW_FLAGS, **(row_flags or {})}
    out = []
    for spec in _ROWS:
        if spec.name == "header":
            out.append(spec)
            continue
        if not flags.get(spec.name, True):
            continue
        if spec.require == "Q" and not include_Q:
            continue
        if spec.require == "RR" and not include_RewardRate:
            continue
        out.append(spec)
    return out


# --------------------------------------------------------------------------
# Grid orchestrator
# --------------------------------------------------------------------------
def plot_subject_model(model_key, subject, fits, df_behavior, *, row_flags=None,
                       fig_col_width=4, fig_row_height=3, dpi=100,
                       mle_score_terminal_c=0.0, mle_score_lapse_override=None):
    """Render the subject × model comparison grid and return the Figure.

    Columns = the criteria discovered for this (model, subject); rows = the
    active diagnostic rows (see :data:`DEFAULT_ROW_FLAGS`). Header always
    shows the criterion label + MLE-Score (+ λ for MLE/joint columns).
    """
    entry = fits.get(model_key)
    if entry is None:
        raise KeyError(f"No such model {model_key!r}. "
                       f"Available: {[k for k in fits]}")
    columns = entry.subjects.get(str(subject))
    if not columns:
        raise KeyError(f"Subject {subject!r} has no fits for model {model_key!r}.")

    include_Q, include_RewardRate = _include_flags(columns[0].payload)
    if include_Q is None:
        include_Q, include_RewardRate = True, True  # permissive fallback
    active = _active_rows(row_flags, include_Q, include_RewardRate)
    n_rows, n_cols = len(active), len(columns)

    # The header is a fixed-height text strip (see _HEADER_ROW_IN); every other
    # row is a full plot row. Figure height = drawn rows + the fixed title and
    # x-label strips, so no proportional dead space is reserved.
    header_ratio = _HEADER_ROW_IN / fig_row_height
    height_ratios = [header_ratio if spec.name == "header" else 1.0
                     for spec in active]
    fig_w = n_cols * fig_col_width
    fig_h = sum(height_ratios) * fig_row_height + _TITLE_IN + _XLABEL_IN
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    # Fixed inch margins → the row-label / title / x-label strips stay a small
    # constant size instead of ballooning with the grid. The suptitle sits
    # centred in the top strip (not floating high at y≈1).
    left, right = _ROW_LABEL_IN / fig_w, 1.0 - _RIGHT_PAD_IN / fig_w
    top, bottom = 1.0 - _TITLE_IN / fig_h, _XLABEL_IN / fig_h
    fig.suptitle(f"{entry.label}   —   {subject}", fontsize="large",
                 y=1.0 - 0.5 * _TITLE_IN / fig_h)
    # hspace gives each row's title/xlabel room so it doesn't collide with the
    # neighbouring row; wspace keeps a small gutter between columns.
    gs = fig.add_gridspec(n_rows, n_cols, height_ratios=height_ratios,
                          left=left, right=right, top=top, bottom=bottom,
                          hspace=0.45, wspace=0.25)

    need_sim = any(spec.needs_sim for spec in active)
    for j, col in enumerate(columns):
        ctx = _ColumnCtx(
            subject=str(subject), fid=col.fid, column_label=col.column_label,
            include_Q=include_Q, include_RewardRate=include_RewardRate,
            t_dur=col.fid.t_dur, dt=col.fid.dt)
        # MLE pass (always — the header MLE-Score needs it).
        (ctx.mle_df, ctx.mle_score, ctx.lapse, ctx.mle_error) = _compute_mle(
            ctx.subject, col.fid, col.payload, df_behavior, include_Q,
            include_RewardRate, mle_score_terminal_c, mle_score_lapse_override)
        # Sim pass (only if a behavioral row is active).
        if need_sim:
            (ctx.sim_df, ctx.bound, ctx.biasFn_kwargs,
             ctx.sim_error) = _compute_sim(
                ctx.subject, col.fid, col.payload, df_behavior, include_Q,
                include_RewardRate)
        for i, spec in enumerate(active):
            if spec.n_sub == 1:
                axes = [fig.add_subplot(gs[i, j])]
            else:
                sub = gs[i, j].subgridspec(spec.n_sub, 1, hspace=0.0)
                first = fig.add_subplot(sub[0])
                axes = [first] + [fig.add_subplot(sub[k], sharex=first)
                                  for k in range(1, spec.n_sub)]
            try:
                _ROW_RENDERERS[spec.name](axes, ctx)
            except Exception as exc:  # noqa: BLE001 — never let one cell kill the grid
                _blank(axes[0], f"{spec.name} failed:\n{type(exc).__name__}")

    _add_row_labels(fig, gs, active)
    return fig


def _add_row_labels(fig, gs, active):
    """Rotated row name near the left edge of each row, inside the fixed
    left strip (some panels set their own y-labels/titles, so this is the
    reliable row identifier). Placed left of the first column's y-ticks."""
    for i, spec in enumerate(active):
        if not spec.label:
            continue
        pos = gs[i, 0].get_position(fig)
        fig.text(pos.x0 * 0.22, 0.5 * (pos.y0 + pos.y1), spec.label, rotation=90,
                 va="center", ha="center", fontsize="small", fontweight="bold")


# --------------------------------------------------------------------------
# Batch + interactive entry points
# --------------------------------------------------------------------------
_ILLEGAL_FS = re.compile(r'[<>:"/\\|?*]')


def _safe_filename(name):
    return _ILLEGAL_FS.sub("_", name.replace("·", "-")).strip()


def run_batch(fits, df_behavior, *, row_flags=None,
              out_dir="../../results/RLModel/fig_model_cmp", img_ext="png",
              fig_col_width=4, fig_row_height=3, dpi=100,
              mle_score_terminal_c=0.0, mle_score_lapse_override=None,
              overwrite=True):
    """Render + save every subject × model grid found on disk.

    Writes ``{out_dir}/{model_label}_{subject}.{img_ext}``. Figures are
    closed after saving to bound memory. Returns the list of written paths.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    combos = [(mk, s) for mk, e in fits.items() for s in e.subjects]
    print(f"Rendering {len(combos)} subject×model grids → {out_dir}")
    for idx, (model_key, subject) in enumerate(combos, 1):
        label = fits[model_key].label
        fname = f"{_safe_filename(label)}_{_safe_filename(str(subject))}.{img_ext}"
        out_path = out_dir / fname
        if out_path.exists() and not overwrite:
            print(f"  [{idx}/{len(combos)}] skip (exists) {fname}")
            continue
        try:
            fig = plot_subject_model(
                model_key, subject, fits, df_behavior, row_flags=row_flags,
                fig_col_width=fig_col_width, fig_row_height=fig_row_height,
                dpi=dpi, mle_score_terminal_c=mle_score_terminal_c,
                mle_score_lapse_override=mle_score_lapse_override)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            written.append(out_path)
            print(f"  [{idx}/{len(combos)}] wrote {fname}")
        except Exception as exc:  # noqa: BLE001 — keep going through the batch
            print(f"  [{idx}/{len(combos)}] FAILED {fname}: "
                  f"{type(exc).__name__}: {exc}")
        finally:
            plt.close("all")
    print(f"Done. Wrote {len(written)} figures.")
    return written


def interactive_viewer(fits, df_behavior, *, row_flags=None, **config):
    """Two dropdowns (Model, Subject) → the comparison grid. Intended for a
    ``%matplotlib inline`` notebook cell (the grid is a static image)."""
    import ipywidgets as widgets
    from IPython.display import display

    models = list_models(fits)
    if not models:
        print("No models discovered — check the result directory.")
        return None
    model_dd = widgets.Dropdown(options=[(lbl, key) for key, lbl in models],
                                description="Model",
                                layout=widgets.Layout(width="60%"))
    subj_dd = widgets.Dropdown(description="Subject",
                               layout=widgets.Layout(width="40%"))
    out = widgets.Output()

    def _refresh_subjects():
        subj_dd.options = subjects_for(fits, model_dd.value)

    def _redraw(*_):
        with out:
            out.clear_output(wait=True)
            if not model_dd.value or not subj_dd.value:
                return
            fig = plot_subject_model(model_dd.value, subj_dd.value, fits,
                                     df_behavior, row_flags=row_flags, **config)
            display(fig)
            plt.close(fig)

    def _on_model(_change):
        _refresh_subjects()
        _redraw()

    model_dd.observe(_on_model, names="value")
    subj_dd.observe(_redraw, names="value")
    _refresh_subjects()
    _redraw()
    display(widgets.HBox([model_dd, subj_dd]), out)
    return {"model_dd": model_dd, "subject_dd": subj_dd, "out": out}
