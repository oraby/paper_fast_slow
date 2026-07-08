"""neural_correlate engine — correlate 2-photon neuron activity with MLE latents.

Cross-analysis backend for ``model_neural_correlate.ipynb``. For a chosen
saved **MLE fit**, we re-evaluate its trial-by-trial latents (Q-left, Q-right,
relative Q "Q-val", RewardRate) and ask, **per neuron**, whether that neuron's
per-trial sampling-window activity correlates with each latent.

Design mirrors ``model/compare.py``: the notebook stays a thin config layer and
calls the functions here. Nothing here is duplicated from a notebook —

- the per-(neuron, trial) windowing/reduction mirrors
  ``twop/activity_correlation.py::getTrialsMeanActivity`` but reduces with
  ``np.nanmax`` (max activity within the sampling window) and additionally keeps
  the full time-normalized trace;
- the MLE re-evaluation reuses ``model/mle_reeval.py`` primitives verbatim;
- the pie chart mirrors ``2pAnalysis.ipynb::loopPlotPieChart`` (session
  mean ± SEM).

Confirmed conventions (see the notebook / plan):
- Activity source is the **z-scored + time(width)-normalized** sampling-window
  df (``normalized_0.1s_before_sampling_0.1s_after_movement.pkl``); the stored
  trace and the ``np.nanmax`` value both come from it.
- Correlation is **Pearson**, within session.
- Latents are the **before-trial** values (state entering the trial).
- Brain region is always ``MFC`` / ``LFC`` (never the enum value or ``*_Bi``).
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass, replace
import pathlib
import pickle
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from scipy.stats import linregress, spearmanr
from statsmodels.stats.multitest import multipletests

from .mle_reeval import (parse_fit_filename, fitted_params_from_result,
                         build_mle_config, evaluate_params_under_mle,
                         prepare_behavior_df)  # re-exported for the notebook
from ...common.definitions import BrainRegion


# --------------------------------------------------------------------------
# Parameter registry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ParamSpec:
    """One model latent the notebook can correlate against neural activity.

    ``key``    — short column/folder name (e.g. ``"Q_L"``; matches the plan).
    ``mle_col``— source column on the re-evaluated ``mle_df`` (for ``"drift"``
                 params this is the behavioral column on the 2-photon row).
    ``family`` — ``"Q"`` (needs ``include_Q``) / ``"RR"`` (needs
                 ``include_RewardRate``) — MLE latents gated by include flags;
                 or ``"drift"`` — behavioral (DV / DVabs), always available.
    ``label``  — friendly axis label (e.g. ``"Q-Left"``).
    ``color``  — plot color, kept consistent between scatter and pie.
    ``signed`` — the value can be negative (DV, Q-val). In the trace plots the
                 gradient then encodes only the **magnitude** (0 → max), so the
                 symmetric extremes (e.g. -1 and +1) share a shade, and the
                 negative-side ranges are drawn dashed.
    """
    key: str
    mle_col: str
    family: str
    label: str
    color: str
    signed: bool = False


PARAM_SPECS = (
    ParamSpec("Q_L", "mle_Q_left_before", "Q", "Q-Left", "g"),
    ParamSpec("Q_R", "mle_Q_right_before", "Q", "Q-Right", "orange"),
    ParamSpec("Q_val", "mle_Q_rel_before", "Q", "Q-val", "purple", signed=True),
    ParamSpec("RewardRate", "mle_reward_rate_before", "RR",
              "R-Learning RewardRate", "teal"),
    # Drift: the neuron's evidence encoding — activity vs signed DV and vs |DV|.
    # These live on the 2-photon trial row, not the MLE latents.
    ParamSpec("DV", "DV", "drift", "DV", "steelblue", signed=True),
    ParamSpec("DVabs", "DVabs", "drift", "|DV|", "sienna"),
)
PARAM_BY_KEY = {spec.key: spec for spec in PARAM_SPECS}

# Behavioral (non-MLE) columns carried onto the neuron-trial table from the 2P
# trial row, for the drift / fast-vs-slow analysis.
DRIFT_KEYS = [s.key for s in PARAM_SPECS if s.family == "drift"]  # ["DV","DVabs"]
_FAST_Q, _SLOW_Q = 1, 3  # quantile_idx: fastest / slowest RT tercile

# The trial-identity key shared by the behavior/MLE and the 2-photon dataframes.
IDENTITY_COLS = ["Name", "Date", "SessionNum", "TrialNumber"]

# The two 2-photon brain regions carried in the traces df (M2_Bi -> MFC,
# ALM_Bi -> LFC); everything else is filtered out, matching twop/fastslowstats.
_ANALYSIS_REGIONS = (BrainRegion.M2_Bi, BrainRegion.ALM_Bi)


# --------------------------------------------------------------------------
# Filesystem-safe names (mirrors compare.py::_safe_filename)
# --------------------------------------------------------------------------
_ILLEGAL_FS = re.compile(r'[<>:"/\\|?*]')


def _safe_filename(name):
    return _ILLEGAL_FS.sub("_", str(name).replace("·", "-")).strip()


def _friendly_model_name(fid):
    """Minimal friendly model name for titles / save paths.

    ``FitFileId.model_label`` is the abstract model (drift·bias·noise·timing,
    which deliberately omits the fitting-criterion weights). For this notebook
    we also surface the **fitting criterion** so a weighted-Chi² fit is not
    silently shown as if it were pure MLE: a joint fit gets a
    ``weighted Chi²`` tag, and a Chi²-fit its Noise/Bound variant."""
    base = fid.model_label
    if fid.fit_mode == "mle" and fid.chi2_weight != 0.0:
        base += (f" · weighted Chi² (MLE={fid.mle_weight:g}, "
                 f"Chi²={fid.chi2_weight:g})")
    elif fid.fit_mode == "chisq":
        base += " · Chi²-Bound" if fid.scaled_bound else " · Chi²-Noise"
    return base


# --------------------------------------------------------------------------
# 1. MLE re-evaluation → per-trial latents
# --------------------------------------------------------------------------
def available_params(include_Q, include_RewardRate):
    """The :class:`ParamSpec`s a fit supports given its include flags."""
    return [s for s in PARAM_SPECS
            if (s.family == "Q" and include_Q)
            or (s.family == "RR" and include_RewardRate)]


def load_mle_per_trial(fit_pkl_path, df_behavior=None, *, subjects=None,
                       reuse_fit_df=True, mle_terminal_c=0.0,
                       lapse_override=None, verbose=True):
    """Load a saved MLE fit's per-trial latent dataframe.

    ``fit_pkl_path`` is a single ``{subject: payload}`` pickle (declared at the
    top of the notebook). Only the latent families the fit actually learns
    (``include_Q`` / ``include_RewardRate``) are surfaced — the availability is
    **reported** so a reward-rate-only or Q-only model is handled gracefully.

    ``subjects`` (e.g. from :func:`subjects_in_2p`) restricts loading to those
    subject names; the fit file usually holds many subjects without imaging, and
    the caller should pass only the ones present in the 2-photon data.

    ``reuse_fit_df`` (default **True**) reads the per-trial ``mle_df`` already
    stored in each payload — fast, and it does not need ``df_behavior``. Set it
    ``False`` to **recompute** the latents via ``mle_reeval`` (expensive; needs
    ``df_behavior``, and honours ``mle_terminal_c`` / ``lapse_override``).

    Returns ``(mle_pt, param_keys)`` where ``mle_pt`` has the identity columns,
    ``mle_loglik``, ``mle_valid`` (bool), ``model_label``, ``model_name`` (safe),
    and one renamed column per available latent (``Q_L``/``Q_R``/``Q_val``/
    ``RewardRate``); ``param_keys`` is the list of those available keys.
    """
    fit_pkl_path = pathlib.Path(fit_pkl_path)
    fid = parse_fit_filename(fit_pkl_path.name)
    with fit_pkl_path.open("rb") as f:
        subject_payloads = pickle.load(f)
    if not isinstance(subject_payloads, dict) or not subject_payloads:
        raise ValueError(f"{fit_pkl_path.name} is not a non-empty "
                         "{subject: payload} dict")

    # Restrict to the requested subjects (those with 2-photon data).
    if subjects is not None:
        want = {str(s) for s in subjects}
        selected = {s: p for s, p in subject_payloads.items() if str(s) in want}
        missing = sorted(want - {str(s) for s in subject_payloads})
        if verbose and missing:
            print(f"  {len(missing)} requested subject(s) absent from the fit "
                  f"file: {missing}")
        if not selected:
            raise ValueError(
                f"None of the requested subjects {sorted(want)} are in "
                f"{fit_pkl_path.name} (has: {sorted(map(str, subject_payloads))}).")
        subject_payloads = selected

    # Include flags: take the first (selected) payload, then assert the rest
    # agree so we don't silently mix Q-only and RR-only subjects in one file.
    first_payload = next(iter(subject_payloads.values()))
    include_Q = bool(first_payload["include_Q"])
    include_RewardRate = bool(first_payload["include_RewardRate"])
    for subj, payload in subject_payloads.items():
        if (bool(payload["include_Q"]) != include_Q
                or bool(payload["include_RewardRate"]) != include_RewardRate):
            raise ValueError(
                f"Subject {subj!r} has different include flags than the rest of "
                f"{fit_pkl_path.name}; cannot combine.")

    friendly = _friendly_model_name(fid)
    specs = available_params(include_Q, include_RewardRate)
    if verbose:
        have = ", ".join(s.key for s in specs) or "(none!)"
        _report = [f"Model: {friendly}",
                   f"  subjects loaded: {sorted(map(str, subject_payloads))}",
                   f"  mode: {'reuse stored mle_df' if reuse_fit_df else 'RECOMPUTE'}",
                   f"  include_Q={include_Q}  include_RewardRate={include_RewardRate}"]
        if not include_Q:
            _report.append("  NOTE: Q-values (Q_L, Q_R, Q_val) NOT available — "
                           "this fit does not learn Q.")
        if not include_RewardRate:
            _report.append("  NOTE: RewardRate NOT available — this fit does "
                           "not learn a reward rate.")
        _report.append(f"  Correlatable parameters: {have}")
        print("\n".join(_report))
    if not specs:
        raise ValueError(f"{fit_pkl_path.name} learns neither Q nor RewardRate; "
                         "nothing to correlate.")

    if not reuse_fit_df and df_behavior is None:
        raise ValueError("reuse_fit_df=False recomputes the latents and needs "
                         "df_behavior (from prepare_behavior_df()).")

    config = (None if reuse_fit_df else
              build_mle_config(fid, include_Q=include_Q,
                               include_RewardRate=include_RewardRate,
                               mle_terminal_c=mle_terminal_c))
    frames = []
    for subject, payload in subject_payloads.items():
        if reuse_fit_df:
            stored = payload.get("mle_df")
            if not isinstance(stored, pd.DataFrame):
                raise KeyError(
                    f"Subject {subject!r} has no stored 'mle_df' in "
                    f"{fit_pkl_path.name}; re-run with reuse_fit_df=False.")
            frames.append(stored)
        else:
            subject_df = df_behavior[df_behavior.Name == subject].copy()
            if subject_df.empty:
                if verbose:
                    print(f"  skip {subject!r}: no behavior rows")
                continue
            params = fitted_params_from_result(payload)
            res = evaluate_params_under_mle(params, subject_df, config,
                                            lapse_override=lapse_override)
            frames.append(res.mle_df)
    if not frames:
        raise ValueError("No subject produced an mle_df.")
    mle_df = pd.concat(frames, ignore_index=True)

    rename = {s.mle_col: s.key for s in specs}
    keep = IDENTITY_COLS + [s.mle_col for s in specs] + ["mle_loglik"]
    if "mle_valid_for_loss" in mle_df.columns:
        keep.append("mle_valid_for_loss")
    mle_pt = mle_df[keep].rename(columns=rename).copy()
    mle_pt["mle_valid"] = (mle_df["mle_valid_for_loss"].to_numpy(dtype=bool)
                           if "mle_valid_for_loss" in mle_df.columns
                           else True)
    mle_pt.drop(columns=[c for c in ["mle_valid_for_loss"]
                         if c in mle_pt.columns], inplace=True)
    mle_pt["model_label"] = friendly
    mle_pt["model_name"] = _safe_filename(friendly)
    return mle_pt, [s.key for s in specs]


# --------------------------------------------------------------------------
# 2. 2-photon activity source
# --------------------------------------------------------------------------
def load_2p_activity(path):
    """Load the sampling-window per-(trial × neuron) traces df.

    Keeps one row per ``(Name, Date, SessionNum, TrialNumber)`` (drops any
    non-sampling epochs / duplicates) and restricts to the analysis regions
    (M2_Bi / ALM_Bi). The neuronal traces live in
    ``row["traces_sets"]["neuronal"]`` keyed by ``trace_id``.
    """
    path = pathlib.Path(path)
    with path.open("rb") as f:
        df = pickle.load(f)
    if "epoch" in df.columns:
        sampling = df["epoch"].astype(str).str.lower().eq("sampling")
        if sampling.any():
            df = df[sampling]
    df = df[df.BrainRegion.isin(_ANALYSIS_REGIONS)]
    df = df.drop_duplicates(subset=IDENTITY_COLS, keep="first").copy()
    if "DV" in df.columns and "DVabs" not in df.columns:
        df["DVabs"] = df["DV"].abs()   # evidence magnitude (mirrors prepare_behavior_df)
    return df.reset_index(drop=True)


def subjects_in_2p(df_2p):
    """Sorted unique subject names (``Name``) present in the 2-photon df — pass
    to :func:`load_mle_per_trial` so only imaged subjects are loaded."""
    return sorted(df_2p["Name"].astype(str).unique())


# --------------------------------------------------------------------------
# 3. Per-(neuron, trial) table
# --------------------------------------------------------------------------
def _identity_key_frame(df):
    """A view of ``df`` with the identity columns coerced to hashable, join-
    safe scalars: Date -> normalized ``Timestamp``, Session/Trial -> ``int``."""
    out = pd.DataFrame(index=df.index)
    out["Name"] = df["Name"].astype(str)
    out["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    out["SessionNum"] = df["SessionNum"].round().astype("int64")
    out["TrialNumber"] = df["TrialNumber"].round().astype("int64")
    return out


def _max_loss_excluded_keys(mle_pt, *, tol=1e-9):
    """Set of identity-key tuples to drop when ``exclude_max_loss`` is on.

    Per **subject** (``Name``): among valid trials, find the maximum per-trial
    loss (``-mle_loglik``) and drop trials at (within ``tol`` of) that maximum.
    The lapse mixture means this worst value is not a fixed floor, so it must be
    computed from the data rather than assumed.
    """
    keys = _identity_key_frame(mle_pt).reset_index(drop=True)
    loss = -mle_pt["mle_loglik"].to_numpy(dtype=float)
    valid = mle_pt["mle_valid"].to_numpy(dtype=bool) & np.isfinite(loss)
    names = keys["Name"].to_numpy()
    excluded = set()
    for name in np.unique(names):
        sel = (names == name) & valid
        if not sel.any():
            continue
        max_loss = loss[sel].max()
        drop = sel & (loss >= max_loss - tol)
        for i in np.flatnonzero(drop):
            excluded.add((keys["Name"].iloc[i], keys["Date"].iloc[i],
                          keys["SessionNum"].iloc[i], keys["TrialNumber"].iloc[i]))
    return excluded


def build_neuron_trial_table(df_2p, mle_pt, param_keys, *,
                             exclude_max_loss=False, tol=1e-9,
                             require_valid=True):
    """Build the intermediate per-(neuron, trial) dataframe (principle 7).

    One row per (neuron, matched trial) with: ``trace_id`` (neuron name),
    ``long_trace_id`` (long name ``{ShortName}_{trace_id}``), ``BrainRegion``
    (``MFC``/``LFC`` string), ``ShortName`` (session), ``TrialNumber``,
    ``max_activity`` (``np.nanmax`` of the windowed time-normalized neuronal
    trace), ``trace`` (the full time-normalized trace, kept as an object), and
    one column per available latent in ``param_keys``.

    Model latents are matched to 2-photon trials on the identity key; the inner
    match naturally drops MLE padding trials (their trial numbers never occur in
    2-photon data). ``require_valid`` also drops MLE-invalid (no-choice) trials.
    ``exclude_max_loss`` additionally drops the worst-loss trials per subject.
    """
    param_cols = [PARAM_BY_KEY[k].key for k in param_keys]

    # Index the per-trial latents by identity key for O(1) lookup.
    mle_keys = _identity_key_frame(mle_pt).reset_index(drop=True)
    valid = mle_pt["mle_valid"].to_numpy(dtype=bool)
    param_vals = {c: mle_pt[c].to_numpy(dtype=float) for c in param_cols}
    excluded = (_max_loss_excluded_keys(mle_pt, tol=tol)
                if exclude_max_loss else set())
    lookup = {}
    for i in range(len(mle_keys)):
        if require_valid and not valid[i]:
            continue
        key = (mle_keys["Name"].iloc[i], mle_keys["Date"].iloc[i],
               mle_keys["SessionNum"].iloc[i], mle_keys["TrialNumber"].iloc[i])
        if key in excluded:
            continue
        lookup[key] = {c: param_vals[c][i] for c in param_cols}

    # Epoch layout (sampling-window boundaries) is constant across the
    # width-normalized df; capture it once so the trace plot can draw the
    # sampling-start / movement-start dashed lines (see plot_neuron_param_traces).
    has_epochs = ("epochs_ranges" in df_2p.columns
                  and "epochs_names" in df_2p.columns and len(df_2p))
    epochs_ranges = (tuple(tuple(int(a) for a in rng)
                           for rng in df_2p["epochs_ranges"].iloc[0])
                     if has_epochs else None)
    epochs_names = (list(df_2p["epochs_names"].iloc[0]) if has_epochs else None)

    twop_keys = _identity_key_frame(df_2p).reset_index(drop=True)
    rows = []
    for pos in range(len(df_2p)):
        trial = df_2p.iloc[pos]
        k = twop_keys.iloc[pos]
        key = (k["Name"], k["Date"], k["SessionNum"], k["TrialNumber"])
        params = lookup.get(key)
        if params is None:
            continue  # no matching (valid, kept) model trial
        region = f"{BrainRegion(int(trial.BrainRegion))}"  # -> MFC / LFC
        short = trial.ShortName
        s = int(trial.trace_start_idx)
        e = int(trial.trace_end_idx) + 1
        # Behavioral drift + speed columns, constant across this trial's neurons.
        dv = float(trial["DV"]) if "DV" in df_2p.columns else np.nan
        dvabs = (float(trial["DVabs"]) if "DVabs" in df_2p.columns
                 else abs(dv))
        quantile_idx = (int(trial["quantile_idx"])
                        if "quantile_idx" in df_2p.columns
                        and pd.notnull(trial["quantile_idx"]) else -1)
        neuronal = trial["traces_sets"]["neuronal"]
        for trace_id, full_trace in neuronal.items():
            window = np.asarray(full_trace)[s:e]
            if window.size == 0 or np.all(np.isnan(window)):
                continue
            row = {
                "trace_id": trace_id,
                "long_trace_id": f"{short}_{trace_id}",
                "BrainRegion": region,
                "ShortName": short,
                "TrialNumber": int(k["TrialNumber"]),
                "max_activity": float(np.nanmax(window)),
                "trace": np.asarray(full_trace),
                "epochs_ranges": epochs_ranges,
                "epochs_names": epochs_names,
                "DV": dv,
                "DVabs": dvabs,
                "quantile_idx": quantile_idx,
            }
            row.update(params)
            rows.append(row)
    cols = (["trace_id", "long_trace_id", "BrainRegion", "ShortName",
             "TrialNumber", "max_activity", "trace", "epochs_ranges",
             "epochs_names", "DV", "DVabs", "quantile_idx"] + param_cols)
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------
# 4. Per-neuron correlations
# --------------------------------------------------------------------------
def _clean_xy(x, y):
    """Finite, paired ``(x, y)``; ``None`` when degenerate (constant / <3 pts)
    so the caller can emit NaNs instead of a spurious fit."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return x, y


def _linfit(x, y):
    """Least-squares line through ``(x, y)`` plus its Pearson r / p-value.

    ``scipy.stats.linregress`` gives the fit line **and** ``rvalue`` (== the
    Pearson correlation) and ``pvalue`` (two-sided, H0: slope=0) in one call —
    so the scatter's fit line and the correlation table share one computation.
    Returns ``(slope, intercept, r, p)``; all NaN on degenerate input.
    """
    cleaned = _clean_xy(x, y)
    if cleaned is None:
        return np.nan, np.nan, np.nan, np.nan
    res = linregress(*cleaned)
    return (float(res.slope), float(res.intercept),
            float(res.rvalue), float(res.pvalue))


def _corr(x, y, method):
    """``(r, p)`` for one neuron × parameter, guarded. ``pearson`` reuses the
    least-squares fit (``_linfit``); ``spearman`` is the rank correlation."""
    if method == "spearman":
        cleaned = _clean_xy(x, y)
        if cleaned is None:
            return np.nan, np.nan
        r, p = spearmanr(*cleaned)
        return float(r), float(p)
    _slope, _intercept, r, p = _linfit(x, y)
    return r, p


def compute_neuron_correlations(table, param_keys, *, method="pearson"):
    """Per-neuron correlation dataframe (principle 8).

    Groups ``table`` by ``long_trace_id`` (which encodes the session, so this is
    within-session per neuron) and correlates each neuron's per-trial
    ``max_activity`` against each available latent. Returns one row per neuron
    with ``trace_id``, ``long_trace_id``, ``BrainRegion``, ``ShortName``,
    ``n_trials``, and ``{key}_r`` / ``{key}_p`` per parameter.
    """
    param_cols = [PARAM_BY_KEY[k].key for k in param_keys]
    out = []
    for long_id, neuron_df in table.groupby("long_trace_id"):
        first = neuron_df.iloc[0]
        rec = {
            "trace_id": first.trace_id,
            "long_trace_id": long_id,
            "BrainRegion": first.BrainRegion,
            "ShortName": first.ShortName,
            "n_trials": len(neuron_df),
        }
        for col in param_cols:
            r, p = _corr(neuron_df[col].to_numpy(),
                         neuron_df["max_activity"].to_numpy(), method)
            rec[f"{col}_r"] = r
            rec[f"{col}_p"] = p
        out.append(rec)
    return pd.DataFrame(out)


def split_fast_slow(table):
    """``(fast, slow)`` subsets of the neuron-trial table — fast = fastest RT
    tercile (``quantile_idx == 1``), slow = slowest (``quantile_idx == 3``); the
    middle tercile is excluded, matching the notebooks' fast/slow convention."""
    return (table[table["quantile_idx"] == _FAST_Q],
            table[table["quantile_idx"] == _SLOW_Q])


def drift_correlations(table, *, method="pearson"):
    """``(corr_fast, corr_slow)`` — per-neuron correlations of max activity vs
    the drift columns (``DV``, ``DVabs``), computed **within** the fast and slow
    trial subsets separately (a neuron may be drift-correlated in one but not the
    other)."""
    fast, slow = split_fast_slow(table)
    return (compute_neuron_correlations(fast, DRIFT_KEYS, method=method),
            compute_neuron_correlations(slow, DRIFT_KEYS, method=method))


# --------------------------------------------------------------------------
# 5. Per-neuron scatter plots (principle 9)
# --------------------------------------------------------------------------
def _region_in_name(name, region):
    return region.lower() in str(name).lower()


def _fmt_p(p, *, decimals=3):
    """Format a p-value in fixed-point (never scientific): ``0.032``. Values too
    small to show at ``decimals`` places become ``< 0.001`` rather than an
    exponent or a misleading ``0.000``."""
    if p is None or not np.isfinite(p):
        return "n/a"
    floor = 10.0 ** (-decimals)
    if 0 < p < floor:
        return f"< {floor:.{decimals}f}"
    return f"{p:.{decimals}f}"


def _plot_one_neuron(ax, neuron_df, spec, *, zscore):
    x = neuron_df[spec.key].to_numpy(dtype=float)
    y = neuron_df["max_activity"].to_numpy(dtype=float)
    if zscore:
        sd = np.nanstd(y)
        y = (y - np.nanmean(y)) / sd if sd > 0 else y - np.nanmean(y)
    ax.scatter(x, y, s=14, color=spec.color, alpha=0.6, edgecolors="none")

    # Least-squares fit line; r / p from the same fit go in the legend.
    slope, intercept, r, p = _linfit(x, y)
    _pf = _fmt_p(p)
    p_str = f"p {_pf}" if _pf.startswith("<") else f"p = {_pf}"
    if np.isfinite(slope):
        xs = np.array([np.nanmin(x), np.nanmax(x)])
        ax.plot(xs, slope * xs + intercept, color="k", lw=1.5,
                label=f"r = {r:+.3f}\n{p_str}")
        ax.legend(loc="best", fontsize="small", frameon=False, handlelength=1.0)

    ax.set_xlabel(spec.label)
    ax.set_ylabel("Neuron Activity")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{neuron_df.iloc[0].long_trace_id}\n"
                 f"{neuron_df.iloc[0].BrainRegion} · {p_str}", fontsize="small")


def plot_neuron_results(table, corr_df, param, *, mode="display", top_x=20,
                        zscore=False, min_abs_corr=None, save_root=None,
                        model_name=None, ext="svg"):
    """Scatter neuron max-activity (y, "Neuron Activity") vs a latent (x).

    Neurons are sorted by ``|r|`` for ``param`` (descending).
    ``mode="display"`` shows the top-``top_x`` neurons inline. ``mode="save"``
    writes each neuron to
    ``{save_root}/{model_name}/traces/{param}/{r:+.3f}_{long_name}[_region].{ext}``
    (region appended only when the long name lacks it) and closes each figure
    without showing it. When ``min_abs_corr`` is set, save mode writes **only**
    neurons whose ``|r| >= min_abs_corr`` (the tuned subset).
    """
    spec = PARAM_BY_KEY[param]
    rcol = f"{spec.key}_r"
    if rcol not in corr_df.columns:
        raise KeyError(f"{param!r} not available in this fit "
                       f"(columns: {list(corr_df.columns)})")

    ranked = corr_df.dropna(subset=[rcol]).copy()
    ranked["_abs_r"] = ranked[rcol].abs()
    ranked = ranked.sort_values("_abs_r", ascending=False)

    by_neuron = dict(tuple(table.groupby("long_trace_id")))

    if mode == "save":
        if save_root is None or model_name is None:
            raise ValueError("save mode needs save_root and model_name")
        if min_abs_corr is not None:
            ranked = ranked[ranked["_abs_r"] >= min_abs_corr]
        out_dir = (pathlib.Path(save_root) / _safe_filename(model_name)
                   / "traces" / spec.key)
        out_dir.mkdir(parents=True, exist_ok=True)
        for _, crow in ranked.iterrows():
            long_id = crow.long_trace_id
            neuron_df = by_neuron[long_id]
            r = crow[rcol]  # for the signed-correlation filename prefix
            fig, ax = plt.subplots(figsize=(4, 4))
            _plot_one_neuron(ax, neuron_df, spec, zscore=zscore)
            fname = f"{r:+.3f}_{_safe_filename(long_id)}"
            if not _region_in_name(long_id, crow.BrainRegion):
                fname += f"_{crow.BrainRegion}"
            fig.savefig(out_dir / f"{fname}.{ext}", bbox_inches="tight")
            plt.close(fig)
        print(f"Saved {len(ranked)} neuron plots -> {out_dir}")
        return out_dir

    # display mode
    head = ranked.head(top_x)
    for _, crow in head.iterrows():
        neuron_df = by_neuron[crow.long_trace_id]
        fig, ax = plt.subplots(figsize=(4, 4))
        _plot_one_neuron(ax, neuron_df, spec, zscore=zscore)
        plt.show()
    return head


# --------------------------------------------------------------------------
# 5b. Per-neuron trace averages, grouped by parameter value range
# --------------------------------------------------------------------------
def _param_shade(base_color, t):
    """A single shade of ``base_color``'s hue for ``t`` in ``[0, 1]`` — ``t=0``
    lightest (pale, desaturated, bright), ``t=1`` darkest (saturated, dim).
    Built in HSV so the light→dark span is visible for any hue (a straight
    light-tint→base blend was invisible for already-dark hues like purple)."""
    h, s, _v = colorsys.rgb_to_hsv(*to_rgb(base_color))
    s = max(s, 0.55)                            # ensure the hue can be shaded
    return colorsys.hsv_to_rgb(h, s * (0.30 + 0.70 * t), 0.98 - 0.55 * t)


def _param_gradient(base_color, n):
    """``n`` shades of ``base_color``'s hue, **light → dark** (low value range =
    pale, high value range = dark)."""
    if n == 1:
        return [_param_shade(base_color, 0.6)]
    return [_param_shade(base_color, t) for t in np.linspace(0.0, 1.0, n)]


def _range_colors_styles(spec, value_ranges):
    """Per-range ``(colors, linestyles)`` for the trace overlay.

    Unsigned params keep the plain light→dark gradient over the ordered ranges
    (all solid). For a **signed** param (DV, Q-val) the gradient encodes only the
    range's **magnitude** — each range's colour comes from ``|midpoint| / maxmag``
    — so the symmetric extremes (e.g. ``[-1,-0.35]`` and ``[0.35,1]``) get the
    same shade, and ranges on the negative side (midpoint < 0) are drawn dashed.
    """
    if not spec.signed:
        return _param_gradient(spec.color, len(value_ranges)), \
            ["-"] * len(value_ranges)
    mids = [0.5 * (lo + hi) for lo, hi in value_ranges]
    maxmag = max((abs(m) for m in mids), default=0.0) or 1.0
    colors = [_param_shade(spec.color, abs(m) / maxmag) for m in mids]
    styles = ["--" if m < 0 else "-" for m in mids]
    return colors, styles


def _draw_epoch_lines(ax, epochs_ranges, epochs_names):
    """Dashed vertical lines at each epoch start after the first (sampling
    start, movement start), labelled — matching the other 2p trace notebooks."""
    if not epochs_ranges:
        return
    starts = [rng[0] for rng in epochs_ranges[1:]]
    labels = (list(epochs_names[1:]) if epochs_names is not None
              else [""] * len(starts))
    for xs in starts:
        ax.axvline(xs, ls="--", color="gray", alpha=0.7)
    ax.set_xticks(starts)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize="small")


def _plot_one_neuron_traces(ax, neuron_df, spec, value_ranges, r, *,
                            panel_label=None):
    """Overlay, in one axes, the mean ± SEM time-normalized trace for each
    value range of ``spec`` (gradient-coloured), with the sampling/movement
    epoch lines. ``panel_label`` (e.g. "Fast") gives a short title instead of
    the full neuron id — for multi-panel figures that carry the id in a
    suptitle."""
    vals = neuron_df[spec.key].to_numpy(dtype=float)
    traces = np.vstack([np.asarray(t, dtype=float)
                        for t in neuron_df["trace"].to_numpy()])
    x = np.arange(traces.shape[1])
    colors, styles = _range_colors_styles(spec, value_ranges)
    for i, (lo, hi) in enumerate(value_ranges):
        # Last range is closed on the right so the max value is included.
        in_range = (vals >= lo) & (vals <= hi if i == len(value_ranges) - 1
                                   else vals < hi)
        m = int(in_range.sum())
        if m == 0:
            continue
        grp = traces[in_range]
        mean = np.nanmean(grp, axis=0)
        ax.plot(x, mean, color=colors[i], lw=1.8, ls=styles[i],
                label=f"[{lo:g}, {hi:g}]  n={m}")
        if m >= 2:  # SEM band needs >= 2 trials
            cnt = np.sum(~np.isnan(grp), axis=0)
            with np.errstate(invalid="ignore"):
                se = np.nanstd(grp, axis=0, ddof=1) / np.sqrt(np.maximum(cnt, 1))
            ax.fill_between(x, mean - se, mean + se, color=colors[i], alpha=0.18,
                            linewidth=0)
    _draw_epoch_lines(ax, neuron_df.iloc[0].get("epochs_ranges"),
                      neuron_df.iloc[0].get("epochs_names"))
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Neuron Activity")
    # Remove the left axis (spine + ticks), matching the other 2p trace plots.
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False, labelleft=False)
    r_str = "n/a" if r is None or not np.isfinite(r) else f"{r:+.3f}"
    if panel_label is not None:
        ax.set_title(f"{panel_label} · r={r_str}", fontsize="small")
    else:
        ax.set_title(f"{neuron_df.iloc[0].long_trace_id}\n"
                     f"{neuron_df.iloc[0].BrainRegion} · {spec.label} · r={r_str}",
                     fontsize="small")
    ax.legend(fontsize="x-small", frameon=False, title=spec.label)


def _edges_to_ranges(bin_edges):
    """``[e0, e1, e2, …]`` bin edges → ``[(e0,e1), (e1,e2), …]`` intervals."""
    edges = list(bin_edges)
    if len(edges) < 2:
        raise ValueError("bin_edges needs at least 2 edges (i.e. >= 1 bin).")
    return list(zip(edges[:-1], edges[1:]))


def plot_neuron_param_traces(table, corr_df, param, bin_edges, *,
                             mode="display", top_x=20, min_abs_corr=None,
                             display_figsize=(6, 4), display_dpi=110,
                             save_figsize=(10, 7), save_dpi=300,
                             save_root=None, model_name=None, ext="svg"):
    """Per-neuron mean ± SEM time-normalized traces, grouped by value range.

    For each neuron, the trials are split by the value of ``param`` into the
    bins defined by ``bin_edges`` — a monotonically increasing list of edges
    (``N`` edges → ``N-1`` bins), e.g. the notebook's
    ``PARAM_VALUE_RANGES[param]``; a trial joins bin ``i`` when
    ``edge[i] <= value < edge[i+1]`` (the last bin includes its right edge).
    The mean ± SEM trace of each bin is overlaid in **one axes**,
    gradient-coloured (pale → the parameter's bar colour), with dashed
    sampling-start / movement-start epoch lines.

    Display mode uses the smaller ``display_figsize`` / ``display_dpi``; save
    mode uses the larger ``save_figsize`` / ``save_dpi`` (sized like the other 2p
    single-neuron plots). Neurons are ordered by ``|r|`` for ``param``.
    ``mode="display"`` shows the top-``top_x``; ``mode="save"`` writes each
    neuron under ``{save_root}/{model_name}/param_traces/{param}/`` and, when
    ``min_abs_corr`` is set, only the tuned subset (``|r| >= min_abs_corr``).
    """
    spec = PARAM_BY_KEY[param]
    rcol = f"{spec.key}_r"
    if rcol not in corr_df.columns:
        raise KeyError(f"{param!r} not available in this fit "
                       f"(columns: {list(corr_df.columns)})")
    value_ranges = _edges_to_ranges(bin_edges)

    ranked = corr_df.dropna(subset=[rcol]).copy()
    ranked["_abs_r"] = ranked[rcol].abs()
    ranked = ranked.sort_values("_abs_r", ascending=False)
    by_neuron = dict(tuple(table.groupby("long_trace_id")))

    if mode == "save":
        if save_root is None or model_name is None:
            raise ValueError("save mode needs save_root and model_name")
        if min_abs_corr is not None:
            ranked = ranked[ranked["_abs_r"] >= min_abs_corr]
        out_dir = (pathlib.Path(save_root) / _safe_filename(model_name)
                   / "param_traces" / spec.key)
        out_dir.mkdir(parents=True, exist_ok=True)
        for _, crow in ranked.iterrows():
            long_id = crow.long_trace_id
            r = crow[rcol]
            fig, ax = plt.subplots(figsize=save_figsize, dpi=save_dpi)
            _plot_one_neuron_traces(ax, by_neuron[long_id], spec, value_ranges, r)
            fname = f"{r:+.3f}_{_safe_filename(long_id)}"
            if not _region_in_name(long_id, crow.BrainRegion):
                fname += f"_{crow.BrainRegion}"
            fig.savefig(out_dir / f"{fname}.{ext}", bbox_inches="tight",
                        dpi=save_dpi)
            plt.close(fig)
        print(f"Saved {len(ranked)} neuron trace plots -> {out_dir}")
        return out_dir

    head = ranked.head(top_x)
    for _, crow in head.iterrows():
        fig, ax = plt.subplots(figsize=display_figsize, dpi=display_dpi)
        _plot_one_neuron_traces(ax, by_neuron[crow.long_trace_id], spec,
                                value_ranges, crow[rcol])
        plt.show()
    return head


# --------------------------------------------------------------------------
# 6. Per-brain-region "fraction tuned" bar chart, with a shuffle baseline
# --------------------------------------------------------------------------
def _available_param_keys(corr_df):
    """The parameter keys present in a correlation dataframe (have a ``_r``)."""
    return [s.key for s in PARAM_SPECS if f"{s.key}_r" in corr_df.columns]


def _session_tuned_percent(corr_df, rcol, min_abs_corr):
    """Series: for each session, % of assessable neurons that are tuned
    (``|r| >= min_abs_corr``). Neurons with undefined r are not assessable and
    drop out of both numerator and denominator."""
    assessable = corr_df.dropna(subset=[rcol]).copy()
    assessable["_tuned"] = assessable[rcol].abs() >= min_abs_corr
    # mean of a boolean per session == fraction tuned; ×100 == percent.
    return (100.0 * assessable.groupby("ShortName")["_tuned"].mean()).dropna()


def _shuffle_null_tuned(table, param_cols, min_abs_corr, n_shuffles, rng):
    """Per-neuron shuffle null for the "tuned?" decision (chance baseline).

    The shuffle is at the **neuron level**: each neuron's per-trial
    ``max_activity`` is permuted (its parameter vector stays put), the
    correlation is recomputed, and ``|r| >= min_abs_corr`` re-decides tuning.
    Because permuting the activity leaves its variance unchanged, only the
    fit's numerator moves, so all ``n_shuffles`` r's for a neuron come from one
    ``(n_shuffles, n_trials)`` permutation matrix.

    Returns ``(meta_df, null_tuned, assessable)``:
      - ``meta_df``: one row per neuron with ``long_trace_id`` / ``ShortName`` /
        ``BrainRegion`` (row order defines the neuron index used below);
      - ``null_tuned[param]``: ``(n_neurons, n_shuffles)`` bool — tuned under
        each shuffle;
      - ``assessable[param]``: ``(n_neurons,)`` bool — False for neurons whose
        correlation is undefined (constant param / activity, <3 trials); those
        drop out of both numerator and denominator, matching the observed side.
    """
    metas, null = [], {p: [] for p in param_cols}
    assess = {p: [] for p in param_cols}
    for long_id, ndf in table.groupby("long_trace_id"):
        metas.append((long_id, ndf.iloc[0].ShortName, ndf.iloc[0].BrainRegion))
        y = ndf["max_activity"].to_numpy(dtype=float)
        n = y.size
        yc = y - y.mean()
        sy = np.sqrt((yc ** 2).sum())
        perm = np.argsort(rng.random((n_shuffles, n)), axis=1)  # (K, n)
        yc_perm = yc[perm]                                      # (K, n)
        for col in param_cols:
            x = ndf[col].to_numpy(dtype=float)
            xc = x - x.mean()
            sx = np.sqrt((xc ** 2).sum())
            if n < 3 or sx == 0 or sy == 0:
                assess[col].append(False)
                null[col].append(np.zeros(n_shuffles, dtype=bool))
                continue
            r = (yc_perm @ xc) / (sy * sx)                     # (K,)
            assess[col].append(True)
            null[col].append(np.abs(r) >= min_abs_corr)
    meta_df = pd.DataFrame(metas,
                           columns=["long_trace_id", "ShortName", "BrainRegion"])
    null = {p: np.asarray(v) for p, v in null.items()}
    assess = {p: np.asarray(v) for p, v in assess.items()}
    return meta_df, null, assess


def _null_region_distribution(meta_df, null_p, assess_p, region_mask):
    """Session-mean % tuned under the shuffle for one region × parameter, as an
    ``(n_shuffles,)`` array (per session % tuned, then mean across sessions —
    the same aggregation as the observed side)."""
    idx = np.flatnonzero(region_mask & assess_p)
    if idx.size == 0:
        return None
    sess = meta_df["ShortName"].to_numpy()
    sess_pcts = [100.0 * null_p[np.flatnonzero((sess == s) & region_mask
                                               & assess_p)].mean(axis=0)
                 for s in pd.unique(sess[idx])]
    return np.vstack(sess_pcts).mean(axis=0)


# --------------------------------------------------------------------------
# Significance annotation + permutation / bootstrap tests
#
# The paper reports significance as permutation / hierarchical-bootstrap
# p-values with ``*/**/***`` stars and Holm correction (see the ``opto`` package,
# e.g. ``opto/bootstrap2regions.py`` and ``opto/optofeedback.py``). These helpers
# bring the neural-correlate bar charts in line with that convention:
#   - per-bar "vs chance" uses the per-neuron activity-shuffle null already built
#     by :func:`_shuffle_null_tuned` (``+1/(n+1)`` permutation p-value);
#   - fast-vs-slow uses a session-paired sign-flip permutation;
#   - MFC-vs-LFC (cross-region) uses a region→session→neuron hierarchical
#     bootstrap with a two-sided sign-change p-value.
# All of this is skipped when ``run_stats=False`` for fast plot-only runs.
# --------------------------------------------------------------------------
def _sigstar(p):
    """``*/**/***`` for p < 0.05 / 0.01 / 0.001, ``"n.s."`` above 0.05, ``""``
    when undefined — the mapping used across the paper's figures."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _holm_adjust(pvals):
    """Holm–Bonferroni step-down over a family of p-values (NaN-safe: NaNs are
    passed through and excluded from the correction). Mirrors the
    ``multipletests(method="holm")`` calls used in ``opto``."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    if finite.any():
        out[finite] = multipletests(p[finite], method="holm")[1]
    return out


def _sign_change_p(observed, draws):
    """Two-sided sign-change p-value (Svoboda/Brody convention, as in
    ``opto/bootstrap2regions.py::_sign_change_p_two_sided``): the fraction of
    bootstrap ``draws`` whose sign is opposite the ``observed`` effect, doubled
    and capped at 1."""
    if observed is None or not np.isfinite(observed):
        return np.nan
    valid = np.asarray(draws, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return np.nan
    if observed == 0:
        return 1.0
    opposite = -1.0 if observed > 0 else 1.0
    p_one = float(np.mean(np.sign(valid) == opposite))
    return float(min(1.0, 2.0 * p_one))


def _add_sig_bracket(ax, x1, x2, y, text, *, tick=None, lw=1.0, fontsize=13):
    """Draw a ``[x1, x2]`` significance bracket at height ``y`` with ``text``
    (a star string) centred above it. Adapted from
    ``opto/optofeedback.py::_add_sig_bracket``. ``tick`` is the drop-down length
    of the bracket ends; returns the top y so callers can stack brackets."""
    if not text:
        return y
    if tick is None:
        span = ax.get_ylim()
        tick = 0.02 * (span[1] - span[0])
    ax.plot([x1, x1, x2, x2], [y, y + tick, y + tick, y],
            color="k", lw=lw, zorder=6, clip_on=False)
    ax.text((x1 + x2) / 2.0, y + tick, text, ha="center", va="bottom",
            fontsize=fontsize, color="k", zorder=6)
    return y + tick


def _paired_perm_fastslow(corr_fast, corr_slow, rcol, min_abs_corr, n_perm, rng,
                          *, region_values=None):
    """Session-paired sign-flip permutation of the fast−slow drift-tuning %.

    Pairs the per-session % of drift-tuned neurons in the fast and slow subsets
    (:func:`_session_tuned_percent`) on ``ShortName``; the observed statistic is
    the mean paired difference (fast − slow). The null flips the sign of each
    session's paired difference independently. Returns ``(obs_diff, p)`` with the
    ``+1/(n+1)`` correction; both NaN when fewer than two paired sessions.
    """
    cf = (corr_fast if region_values is None
          else corr_fast[corr_fast.BrainRegion.isin(region_values)])
    cs = (corr_slow if region_values is None
          else corr_slow[corr_slow.BrainRegion.isin(region_values)])
    fast_pct = _session_tuned_percent(cf, rcol, min_abs_corr)
    slow_pct = _session_tuned_percent(cs, rcol, min_abs_corr)
    common = fast_pct.index.intersection(slow_pct.index)
    if len(common) < 2:
        return np.nan, np.nan
    d = (fast_pct.loc[common] - slow_pct.loc[common]).to_numpy(dtype=float)
    obs = float(d.mean())
    signs = rng.choice((-1.0, 1.0), size=(n_perm, d.size))
    null = (signs * d).mean(axis=1)
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    return obs, p


def _pct_from_absr(abs_r, min_abs_corr):
    """% of assessable neurons tuned, from an ``(n_neurons, n_params)`` matrix of
    ``|r|`` (NaN where undefined). Assessable = any param defined; tuned = any
    defined ``|r| >= min_abs_corr`` — the factor OR, matching
    :func:`_session_factor_percent`. NaN when no assessable neuron."""
    if abs_r.size == 0:
        return np.nan
    assessable = np.isfinite(abs_r).any(axis=1)
    if not assessable.any():
        return np.nan
    tuned = (abs_r >= min_abs_corr).any(axis=1)  # NaN >= x is False
    return 100.0 * float(tuned[assessable].mean())


def _session_absr_mats(corr_df, param_keys):
    """List (one per session) of ``(n_neurons, n_params)`` ``|r|`` matrices for a
    per-neuron correlation frame — the fast unit for the region bootstrap."""
    cols = [f"{k}_r" for k in param_keys]
    return [np.abs(g[cols].to_numpy(dtype=float))
            for _, g in corr_df.groupby("ShortName")]


def _region_stat_from_mats(mats, min_abs_corr):
    """Session-mean % tuned (across-session mean of per-session %) for a region."""
    vals = [v for v in (_pct_from_absr(m, min_abs_corr) for m in mats)
            if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def _boot_region_stat(mats, min_abs_corr, rng):
    """One hierarchical-bootstrap replicate of a region's session-mean %:
    resample sessions with replacement, then neurons within each drawn session
    with replacement."""
    n = len(mats)
    if n == 0:
        return np.nan
    vals = []
    for j in rng.integers(0, n, size=n):
        m = mats[j]
        if m.shape[0] == 0:
            continue
        v = _pct_from_absr(m[rng.integers(0, m.shape[0], size=m.shape[0])],
                           min_abs_corr)
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def _hier_bootstrap_region_diff(mats_a, mats_b, min_abs_corr, n_boot, rng):
    """MFC-vs-LFC (region A vs B) hierarchical bootstrap on the "% tuned" stat.

    ``mats_a`` / ``mats_b`` are per-session ``|r|`` matrices (see
    :func:`_session_absr_mats`) for the two regions. Both regions are resampled
    independently at the session then neuron level (:func:`_boot_region_stat`);
    the statistic is ``stat_a - stat_b``. Returns a dict with the observed diff,
    each region's observed stat, and the two-sided sign-change p over ``n_boot``.
    """
    a_obs = _region_stat_from_mats(mats_a, min_abs_corr)
    b_obs = _region_stat_from_mats(mats_b, min_abs_corr)
    if not (np.isfinite(a_obs) and np.isfinite(b_obs)):
        return dict(diff=np.nan, p=np.nan, stat_a=a_obs, stat_b=b_obs)
    obs = float(a_obs - b_obs)
    draws = np.array([_boot_region_stat(mats_a, min_abs_corr, rng)
                      - _boot_region_stat(mats_b, min_abs_corr, rng)
                      for _ in range(n_boot)], dtype=float)
    return dict(diff=obs, p=_sign_change_p(obs, draws),
                stat_a=float(a_obs), stat_b=float(b_obs))


def plot_region_bars(corr_df, table, param_keys=None, *, min_abs_corr=0.3,
                     n_shuffles=1000, by_region=True, seed=0, ci=(2.5, 97.5),
                     run_stats=True, save=False, save_root=None,
                     model_name=None, ext="svg"):
    """Grouped "fraction tuned" bars per brain region.

    For each brain region (``by_region=True`` → one panel per MFC/LFC;
    ``by_region=False`` → a single pooled panel) the x-axis is the model
    parameters. Each **coloured bar** is the observed % of tuned neurons
    (``|r| >= min_abs_corr``), computed per session then averaged across
    sessions; the error bar is the **SEM across sessions** (the distribution of
    per-session percentages within the region).

    When ``run_stats`` (default), each bar is tested against a per-neuron
    activity-shuffle chance level (``n_shuffles`` permutations, ``+1/(n+1)``
    p-value; see :func:`_shuffle_null_tuned`), the p-values are Holm-corrected
    across the bars of a panel, and significance is annotated as ``*/**/***``
    stars (``n.s.`` otherwise). Set ``run_stats=False`` to skip the permutation
    entirely for a fast plot-only run.

    Returns a tidy summary dataframe (one row per region × parameter). Always
    shown; ``save`` also writes the figure + summary under
    ``{save_root}/{model_name}/tuning_bars/``.

    "Tuned" is a per-**neuron** property (its activity vs the parameter across
    the session's trials); the bar is the % of such neurons, so the y-axis reads
    "Tuned neurons (%)".
    """
    if param_keys is None:
        param_keys = _available_param_keys(corr_df)
    param_cols = [PARAM_BY_KEY[k].key for k in param_keys]
    if not param_cols:
        raise ValueError("No available parameters to plot.")
    lo, hi = ci

    rng = np.random.default_rng(seed)
    if run_stats:
        meta_df, null, assess = _shuffle_null_tuned(
            table, param_cols, min_abs_corr, n_shuffles, rng)
    else:
        meta_df = null = assess = None

    if by_region:
        regions = [(r, [r]) for r in sorted(pd.unique(corr_df.BrainRegion))]
    else:
        regions = [("MFC & LFC", None)]

    x = np.arange(len(param_cols))
    fig, axs = plt.subplots(1, len(regions), figsize=(4.2 * len(regions), 4.4),
                            squeeze=False, sharey=True)
    rows = []
    for ax, (region, rvals) in zip(axs[0], regions):
        obs_means, obs_sems, pvals = [], [], []
        panel_rows = []
        for col in param_cols:
            rcol = f"{col}_r"
            grp = (corr_df if rvals is None
                   else corr_df[corr_df.BrainRegion.isin(rvals)])
            pcts = _session_tuned_percent(grp, rcol, min_abs_corr)
            obs_mean = float(pcts.mean()) if len(pcts) else np.nan
            obs_sem = float(pcts.sem()) if len(pcts) > 1 else 0.0
            null_mean = null_lo = null_hi = pval = np.nan
            if run_stats:
                mask = (np.ones(len(meta_df), dtype=bool) if rvals is None
                        else np.isin(meta_df["BrainRegion"].to_numpy(), rvals))
                null_dist = _null_region_distribution(meta_df, null[col],
                                                      assess[col], mask)
                if null_dist is not None and np.isfinite(obs_mean):
                    null_mean = float(null_dist.mean())
                    null_lo, null_hi = np.percentile(null_dist, [lo, hi])
                    # +1/+1 correction so p is never exactly 0.
                    pval = float((np.sum(null_dist >= obs_mean) + 1)
                                 / (n_shuffles + 1))
            obs_means.append(obs_mean); obs_sems.append(obs_sem); pvals.append(pval)
            panel_rows.append({"BrainRegion": region, "param": col,
                               "observed_pct": obs_mean, "observed_sem": obs_sem,
                               "shuffle_pct": null_mean, f"ci{lo:g}": null_lo,
                               f"ci{hi:g}": null_hi, "n_sessions": len(pcts),
                               "p_vs_shuffle": pval})

        # Holm across the bars of this panel, then annotate stars.
        p_holm = _holm_adjust(pvals) if run_stats else np.full(len(pvals), np.nan)
        for r, ph in zip(panel_rows, p_holm):
            r["p_vs_shuffle_holm"] = float(ph) if np.isfinite(ph) else np.nan
        rows.extend(panel_rows)

        colors = [PARAM_BY_KEY[k].color for k in param_keys]
        obs_means = np.array(obs_means, dtype=float)
        obs_sems = np.array(obs_sems, dtype=float)
        ax.bar(x, obs_means, width=0.62, color=colors, alpha=0.85,
               edgecolor="k", linewidth=0.5, yerr=obs_sems, capsize=3, zorder=2)
        if run_stats:
            for xi, om, os_, ph in zip(x, obs_means, obs_sems, p_holm):
                if np.isfinite(om):
                    ax.text(xi, om + os_ + 0.6, _sigstar(ph), ha="center",
                            va="bottom", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([PARAM_BY_KEY[k].label for k in param_keys],
                           rotation=20, ha="right", fontsize="small")
        ax.set_title(f"{region}  (|r| ≥ {min_abs_corr:g})")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0][0].set_ylabel("Tuned neurons (%)")
    fig.suptitle(f"Parameter tuning — {model_name or ''}", y=1.02)
    fig.tight_layout()

    summary = pd.DataFrame(rows)
    if save:
        if save_root is None or model_name is None:
            raise ValueError("save needs save_root and model_name")
        out_dir = (pathlib.Path(save_root) / _safe_filename(model_name)
                   / "tuning_bars")
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "by_region" if by_region else "combined"
        fig.savefig(out_dir / f"bars_{tag}.{ext}", bbox_inches="tight")
        summary.to_csv(out_dir / f"summary_{tag}.csv", index=False)
        print(f"Saved tuning bars -> {out_dir}")
    plt.show()
    return summary


# --------------------------------------------------------------------------
# 7. Fast vs slow: drift-correlated neurons (DV / DVabs)
# --------------------------------------------------------------------------
_FAST_COLOR = "tomato"     # lab convention (behavior.ipynb): fast = red-ish,
_SLOW_COLOR = "goldenrod"  # slow = yellow/gold.


def _rfmt(v):
    return f"{v:+.3f}" if v is not None and np.isfinite(v) else "nan"


def _pct_stats(corr, rcol, min_abs_corr):
    """(session-mean %, SEM, n_sessions) of drift-correlated neurons."""
    pcts = _session_tuned_percent(corr, rcol, min_abs_corr)
    mean = float(pcts.mean()) if len(pcts) else np.nan
    sem = float(pcts.sem()) if len(pcts) > 1 else 0.0
    return mean, sem, len(pcts)


def _bar_vs_chance(bundle, key, region_values, obs_mean, n_perm):
    """``+1/(n+1)`` permutation p of one bar vs its per-neuron activity-shuffle
    chance level, off a precomputed :func:`_shuffle_null_tuned` bundle."""
    meta_df, null, assess = bundle
    mask = (np.ones(len(meta_df), dtype=bool) if region_values is None
            else np.isin(meta_df["BrainRegion"].to_numpy(), region_values))
    null_dist = _null_region_distribution(meta_df, null[key], assess[key], mask)
    if null_dist is None or not np.isfinite(obs_mean):
        return np.nan
    return float((np.sum(null_dist >= obs_mean) + 1) / (n_perm + 1))


def plot_fast_slow_bars(corr_fast, corr_slow, param, *, table=None,
                        min_abs_corr=0.3, n_perm=1000, n_boot=10000,
                        by_region=True, seed=0, run_stats=True, save=False,
                        save_root=None, model_name=None, ext="svg"):
    """Fast-vs-slow % of drift-correlated neurons, with permutation significance.

    For ``param`` (``"DV"`` / ``"DVabs"``), a neuron is drift-correlated when
    ``|r| >= min_abs_corr`` (Pearson, within its fast/slow subset). Each bar is
    the session-mean ± SEM of that %; fast = tomato, slow = goldenrod.
    ``by_region=True`` → a fast|slow pair per MFC/LFC (**4 bars**);
    ``by_region=False`` → all regions pooled (**2 bars**).

    When ``run_stats`` three permutation/bootstrap layers are annotated:
      1. **vs chance** — a ``*/**/***`` star above each bar (per-neuron activity
         shuffle, ``n_perm`` permutations; needs the trial-level ``table`` so the
         fast/slow subsets can be shuffled — skipped with a warning if omitted);
      2. **fast vs slow** — a session-paired sign-flip permutation
         (:func:`_paired_perm_fastslow`) drawn as a bracket over each region's
         pair;
      3. **MFC vs LFC** within each speed — a region→session→neuron hierarchical
         bootstrap (:func:`_hier_bootstrap_region_diff`) drawn as spanning
         brackets (only when ``by_region`` with two regions).
    Each family is Holm-corrected across its comparisons. Returns a tidy summary
    (region × speed rows, plus cross-region rows); ``save`` also writes it.
    """
    spec = PARAM_BY_KEY[param]
    rcol = f"{spec.key}_r"
    for c in (corr_fast, corr_slow):
        if rcol not in c.columns:
            raise KeyError(f"{param!r} not in the correlation dataframe.")

    if by_region:
        regions = sorted(set(corr_fast.BrainRegion) | set(corr_slow.BrainRegion))
    else:
        regions = ["MFC & LFC"]

    rng = np.random.default_rng(seed)
    fast_bundle = slow_bundle = None
    if run_stats:
        if table is not None:
            fast_table, slow_table = split_fast_slow(table)
            fast_bundle = _shuffle_null_tuned(fast_table, [spec.key],
                                              min_abs_corr, n_perm, rng)
            slow_bundle = _shuffle_null_tuned(slow_table, [spec.key],
                                              min_abs_corr, n_perm, rng)
        else:
            warnings.warn("plot_fast_slow_bars: `table` not supplied — skipping "
                          "the per-bar vs-chance test (paired + cross-region "
                          "tests still run).")

    x = np.arange(len(regions))
    w = 0.36
    fig, ax = plt.subplots(figsize=(1.9 * len(regions) + 2.5, 4.4))
    rows, per_region, bar_tops = [], [], []
    for i, region in enumerate(regions):
        rvals = None if not by_region else [region]
        cf = corr_fast if rvals is None else corr_fast[corr_fast.BrainRegion == region]
        cs = corr_slow if rvals is None else corr_slow[corr_slow.BrainRegion == region]
        fmean, fsem, fn = _pct_stats(cf, rcol, min_abs_corr)
        smean, ssem, sn = _pct_stats(cs, rcol, min_abs_corr)
        ax.bar(x[i] - w / 2, fmean, width=w, yerr=fsem, capsize=3,
               color=_FAST_COLOR, edgecolor="k", linewidth=0.5)
        ax.bar(x[i] + w / 2, smean, width=w, yerr=ssem, capsize=3,
               color=_SLOW_COLOR, edgecolor="k", linewidth=0.5)

        fp = sp = pair_diff = pair_p = np.nan
        if run_stats:
            if fast_bundle is not None:
                fp = _bar_vs_chance(fast_bundle, spec.key, rvals, fmean, n_perm)
                sp = _bar_vs_chance(slow_bundle, spec.key, rvals, smean, n_perm)
            pair_diff, pair_p = _paired_perm_fastslow(
                corr_fast, corr_slow, rcol, min_abs_corr, n_perm, rng,
                region_values=rvals)
        per_region.append(dict(region=region, i=i, fmean=fmean, fsem=fsem,
                               smean=smean, ssem=ssem, fp=fp, sp=sp,
                               pair_p=pair_p))
        rows += [{"BrainRegion": region, "speed": "fast", "pct": fmean,
                  "sem": fsem, "n_sessions": fn, "p_vs_chance": fp,
                  "fast_minus_slow": pair_diff, "p_fast_vs_slow": pair_p},
                 {"BrainRegion": region, "speed": "slow", "pct": smean,
                  "sem": ssem, "n_sessions": sn, "p_vs_chance": sp,
                  "fast_minus_slow": pair_diff, "p_fast_vs_slow": pair_p}]
        bar_tops += [fmean + fsem, smean + ssem]
        print(f"\t{region}: fast {fmean:.2f}±{fsem:.2f}% ({fn} sess) | "
              f"slow {smean:.2f}±{ssem:.2f}% ({sn} sess)")

    ymax = max([t for t in bar_tops if np.isfinite(t)] + [1.0])
    step = 0.10 * ymax + 1.0

    if run_stats:
        n = len(per_region)
        # 1. vs-chance stars above each bar (Holm across all bars).
        vc_holm = _holm_adjust([pr["fp"] for pr in per_region]
                               + [pr["sp"] for pr in per_region])
        for k, pr in enumerate(per_region):
            if np.isfinite(pr["fmean"]):
                ax.text(pr["i"] - w / 2, pr["fmean"] + pr["fsem"] + 0.2 * step,
                        _sigstar(vc_holm[k]), ha="center", va="bottom",
                        fontsize=11)
            if np.isfinite(pr["smean"]):
                ax.text(pr["i"] + w / 2, pr["smean"] + pr["ssem"] + 0.2 * step,
                        _sigstar(vc_holm[n + k]), ha="center", va="bottom",
                        fontsize=11)
        # 2. fast-vs-slow paired bracket per region (Holm across regions). Drawn
        # even when n.s. so the comparison is always visible.
        for pr, ph in zip(per_region,
                          _holm_adjust([pr["pair_p"] for pr in per_region])):
            s = _sigstar(ph)
            if s:
                y = max(pr["fmean"] + pr["fsem"], pr["smean"] + pr["ssem"]) \
                    + 0.5 * step
                _add_sig_bracket(ax, pr["i"] - w / 2, pr["i"] + w / 2, y, s)
        # 3. MFC-vs-LFC per speed (hierarchical bootstrap, Holm across speeds).
        if by_region and len(regions) == 2:
            ra, rb = regions[0], regions[1]
            speeds = [("fast", corr_fast, -w / 2), ("slow", corr_slow, w / 2)]
            res_list = []
            for _, cc, _off in speeds:
                ma = _session_absr_mats(cc[cc.BrainRegion == ra], [spec.key])
                mb = _session_absr_mats(cc[cc.BrainRegion == rb], [spec.key])
                res_list.append(_hier_bootstrap_region_diff(
                    ma, mb, min_abs_corr, n_boot, rng))
            cross_holm = _holm_adjust([r["p"] for r in res_list])
            for (speed, _cc, off), res, ph in zip(speeds, res_list, cross_holm):
                rows.append({"BrainRegion": f"{ra} vs {rb}", "speed": speed,
                             "diff_region": res["diff"], "p_cross_region": res["p"],
                             "p_cross_region_holm": (float(ph)
                                                     if np.isfinite(ph) else np.nan)})
                s = _sigstar(ph)
                if s:
                    y = ymax + (1.4 if speed == "fast" else 2.6) * step
                    _add_sig_bracket(ax, 0 + off, 1 + off, y, s)
        ax.set_ylim(top=ymax + 4.0 * step)

    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylabel(f"{spec.label}-correlated neurons (%)")
    ax.set_title(f"{spec.label} drift tuning, fast vs slow (|r| ≥ {min_abs_corr:g})"
                 f"\n{model_name or ''}", fontsize="medium")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Patch(facecolor=_FAST_COLOR, edgecolor="k", label="fast"),
                       Patch(facecolor=_SLOW_COLOR, edgecolor="k", label="slow")],
              frameon=False, fontsize="small")
    fig.tight_layout()

    summary = pd.DataFrame(rows)
    if save:
        if save_root is None or model_name is None:
            raise ValueError("save needs save_root and model_name")
        out_dir = (pathlib.Path(save_root) / _safe_filename(model_name)
                   / "fast_slow_bars" / spec.key)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "by_region" if by_region else "combined"
        fig.savefig(out_dir / f"bars_{tag}.{ext}", bbox_inches="tight")
        summary.to_csv(out_dir / f"summary_{tag}.csv", index=False)
        print(f"Saved fast/slow bars -> {out_dir}")
    plt.show()
    return summary


def _fastslow_ranking(corr_fast, corr_slow, rcol):
    """Neuron table (long_trace_id, BrainRegion, r_fast, r_slow, max_abs) ranked
    by ``max(|r_fast|, |r_slow|)`` descending; neurons undefined in both drop."""
    meta = pd.concat([corr_fast[["long_trace_id", "BrainRegion"]],
                      corr_slow[["long_trace_id", "BrainRegion"]]]) \
             .drop_duplicates("long_trace_id").set_index("long_trace_id")
    rank = meta.copy()
    rank["r_fast"] = corr_fast.set_index("long_trace_id")[rcol]
    rank["r_slow"] = corr_slow.set_index("long_trace_id")[rcol]
    rank["max_abs"] = rank[["r_fast", "r_slow"]].abs().max(axis=1)
    return rank.dropna(subset=["max_abs"]).sort_values("max_abs", ascending=False)


def _plot_one_neuron_overlay(ax, neuron_fast, neuron_slow, spec):
    """Scatter a neuron's fast (tomato) & slow (goldenrod) trials with per-speed
    least-squares fit lines; legend carries each speed's r / p."""
    for ndf, color, name in [(neuron_fast, _FAST_COLOR, "fast"),
                             (neuron_slow, _SLOW_COLOR, "slow")]:
        if ndf is None or len(ndf) == 0:
            continue
        x = ndf[spec.key].to_numpy(dtype=float)
        y = ndf["max_activity"].to_numpy(dtype=float)
        ax.scatter(x, y, s=14, color=color, alpha=0.5, edgecolors="none")
        slope, intercept, r, p = _linfit(x, y)
        if np.isfinite(slope):
            xs = np.array([np.nanmin(x), np.nanmax(x)])
            _pf = _fmt_p(p)
            ptxt = f"p {_pf}" if _pf.startswith("<") else f"p = {_pf}"
            ax.plot(xs, slope * xs + intercept, color=color, lw=2,
                    label=f"{name}: r={r:+.3f}, {ptxt}")
        else:
            ax.plot([], [], color=color, label=f"{name}: r=n/a")
    ax.set_xlabel(spec.label)
    ax.set_ylabel("Neuron Activity")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize="small", frameon=False)


def plot_neuron_fastslow_scatter(table, corr_fast, corr_slow, param, *,
                                 mode="display", top_x=20, min_abs_corr=None,
                                 display_figsize=(6, 5), display_dpi=110,
                                 save_figsize=(9, 7), save_dpi=300,
                                 save_root=None, model_name=None, ext="svg"):
    """Per-neuron activity-vs-drift scatter, fast & slow overlaid.

    Neurons ranked by ``max(|r_fast|, |r_slow|)``. ``mode="display"`` shows the
    top-``top_x``; ``mode="save"`` writes each neuron under
    ``{save_root}/{model_name}/drift_scatter/{param}/`` and, when ``min_abs_corr``
    is set, only the subset with ``max(|r_fast|,|r_slow|) >= min_abs_corr``.
    """
    spec = PARAM_BY_KEY[param]
    rcol = f"{spec.key}_r"
    rank = _fastslow_ranking(corr_fast, corr_slow, rcol)
    tf, ts = split_fast_slow(table)
    by_fast = dict(tuple(tf.groupby("long_trace_id")))
    by_slow = dict(tuple(ts.groupby("long_trace_id")))

    def _draw(fig, ax, long_id, row):
        _plot_one_neuron_overlay(ax, by_fast.get(long_id), by_slow.get(long_id),
                                 spec)
        ax.set_title(f"{long_id}\n{row.BrainRegion} · {spec.label}",
                     fontsize="small")

    if mode == "save":
        if save_root is None or model_name is None:
            raise ValueError("save mode needs save_root and model_name")
        if min_abs_corr is not None:
            rank = rank[rank["max_abs"] >= min_abs_corr]
        out_dir = (pathlib.Path(save_root) / _safe_filename(model_name)
                   / "drift_scatter" / spec.key)
        out_dir.mkdir(parents=True, exist_ok=True)
        for long_id, row in rank.iterrows():
            fig, ax = plt.subplots(figsize=save_figsize, dpi=save_dpi)
            _draw(fig, ax, long_id, row)
            fname = f"f{_rfmt(row.r_fast)}_s{_rfmt(row.r_slow)}_{_safe_filename(long_id)}"
            if not _region_in_name(long_id, row.BrainRegion):
                fname += f"_{row.BrainRegion}"
            fig.savefig(out_dir / f"{fname}.{ext}", bbox_inches="tight",
                        dpi=save_dpi)
            plt.close(fig)
        print(f"Saved {len(rank)} fast/slow scatters -> {out_dir}")
        return out_dir

    for long_id, row in rank.head(top_x).iterrows():
        fig, ax = plt.subplots(figsize=display_figsize, dpi=display_dpi)
        _draw(fig, ax, long_id, row)
        plt.show()
    return rank.head(top_x)


def plot_neuron_fastslow_traces(table, corr_fast, corr_slow, param, bin_edges, *,
                                mode="display", top_x=20, min_abs_corr=None,
                                display_figsize=(11, 4), display_dpi=110,
                                save_figsize=(16, 6), save_dpi=300,
                                save_root=None, model_name=None, ext="svg"):
    """Per-neuron normalized trace averages by value range, in two panels
    (Fast | Slow). Each panel is the gradient-by-range mean ± SEM style with
    sampling/movement epoch lines. Ranking / display / save behave like
    :func:`plot_neuron_fastslow_scatter`; saved under
    ``{save_root}/{model_name}/drift_traces/{param}/``."""
    spec = PARAM_BY_KEY[param]
    rcol = f"{spec.key}_r"
    value_ranges = _edges_to_ranges(bin_edges)
    rank = _fastslow_ranking(corr_fast, corr_slow, rcol)
    tf, ts = split_fast_slow(table)
    by_fast = dict(tuple(tf.groupby("long_trace_id")))
    by_slow = dict(tuple(ts.groupby("long_trace_id")))

    def _panel(ax, ndf, r, label):
        if ndf is None or len(ndf) == 0:
            ax.text(0.5, 0.5, f"no {label.lower()} trials", ha="center",
                    va="center", transform=ax.transAxes, color="0.5")
            ax.axis("off")
            return
        _plot_one_neuron_traces(ax, ndf, spec, value_ranges, r,
                                panel_label=label)

    def _draw(fig, axs, long_id, row):
        _panel(axs[0], by_fast.get(long_id), row.r_fast, "Fast")
        _panel(axs[1], by_slow.get(long_id), row.r_slow, "Slow")
        fig.suptitle(f"{long_id}  ·  {row.BrainRegion}  ·  {spec.label}",
                     fontsize="medium")

    if mode == "save":
        if save_root is None or model_name is None:
            raise ValueError("save mode needs save_root and model_name")
        if min_abs_corr is not None:
            rank = rank[rank["max_abs"] >= min_abs_corr]
        out_dir = (pathlib.Path(save_root) / _safe_filename(model_name)
                   / "drift_traces" / spec.key)
        out_dir.mkdir(parents=True, exist_ok=True)
        for long_id, row in rank.iterrows():
            fig, axs = plt.subplots(1, 2, figsize=save_figsize, dpi=save_dpi,
                                    sharey=True)
            _draw(fig, axs, long_id, row)
            fname = f"f{_rfmt(row.r_fast)}_s{_rfmt(row.r_slow)}_{_safe_filename(long_id)}"
            if not _region_in_name(long_id, row.BrainRegion):
                fname += f"_{row.BrainRegion}"
            fig.savefig(out_dir / f"{fname}.{ext}", bbox_inches="tight",
                        dpi=save_dpi)
            plt.close(fig)
        print(f"Saved {len(rank)} fast/slow trace figures -> {out_dir}")
        return out_dir

    for long_id, row in rank.head(top_x).iterrows():
        fig, axs = plt.subplots(1, 2, figsize=display_figsize, dpi=display_dpi,
                                sharey=True)
        _draw(fig, axs, long_id, row)
        plt.show()
    return rank.head(top_x)
