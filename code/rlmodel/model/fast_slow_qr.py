"""Fast/slow trials in the model's (Q-relative, reward-rate) latent space.

For the modelling mice (>= ``MIN_NUM_TRIALS`` valid trials), take the
trial-by-trial MLE fit (weighted MLE, Chi² weight 0.5) and, for every real
trial, read the model's latent state that drove the decision — the reward-rate
(R-learning) and the Q-value — then colour the trial by whether the animal was
fast or slow. Two views:

- :func:`plot_subject_qr_scatter` — one scatter per animal, x = reward-rate,
  y = Q-relative, red = fast / yellow = slow. Points are shuffled into a single
  scatter so neither colour paints over the other, and kept small so the SVG
  stays manageable.
- :func:`plot_pooled_fastness_heatmap` — all animals pooled into a binned
  heatmap of the *slow fraction* (``n_slow / (n_fast + n_slow)``), autumn-mapped
  so red = mostly-fast, yellow = mostly-slow. This mirrors
  ``model_to_behavior``'s Q/R heatmap with the axes swapped (x = reward-rate).

Latents (``mle_reward_rate_before`` / ``mle_Q_rel_before``, the values *before*
each trial, i.e. what the model used to decide it) are, by default, read from
the per-trial ``mle_df`` the fit already stored at fit time — recomputing is
unnecessary. ``build_fast_slow_qr(reuse_fit_df=False)`` re-evaluates the fitted
params under MLE instead (``compare._compute_mle`` -> ``mle_reeval`` ->
``mle._build_mle_df``), which needs the behavior df. Because MLE is
teacher-forced on the animal's real choices and RTs, fast/slow is the real RT
split (``calcStimulusTime``), computed within each difficulty via
``plotter._dvQuantileFn`` — the same definition used across the model analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import compare
from .aggregate import resolve_spec, safe_filename, MIN_NUM_TRIALS, MLE_WEIGHT_SPECS
from .plotter import _dvQuantileFn
from .initvals import MLE_TERMINAL_C


# The trial-by-trial fit this figure reads: reward-rate (noise gain) + Q-Val,
# fitted by weighted MLE with Chi² weight 0.5. Selected by column label so it
# survives a reordering of MLE_WEIGHT_SPECS.
CHI2_W05_SPEC = next(s for s in MLE_WEIGHT_SPECS
                     if s.column_label == "MLE=1, Chi²=0.5")

# Fast = red, slow = yellow — matching the autumn colormap the pooled heatmap
# uses (autumn(0) = red, autumn(1) = yellow). ``gold`` reads better than pure
# ``yellow`` on white; override via the plot kwargs if you want literal yellow.
FAST_COLOR = "red"
SLOW_COLOR = "gold"

# Latent columns produced by mle._build_mle_df (values *before* each trial).
_Q_COL = "mle_Q_rel_before"
_R_COL = "mle_reward_rate_before"
# The fitted starting point z = clip(BIAS_COEF*q_rel + Q_VAL_OFFSET, -1, 1),
# left/right-signed like q_rel. It is the Q bias already expressed as a
# fraction of the decision threshold (BOUND is fixed at 1), so it is
# comparable across subjects — see build_fast_slow_qr(bias_scaled=True).
_Z_COL = "mle_z"


# --------------------------------------------------------------------------
# Per-trial derived quantities
# --------------------------------------------------------------------------
def add_q_relative(df, *, q_col=_Q_COL, out_col="Q_relative"):
    """Add a DV-relative Q column.

    ``mle_Q_rel_before`` is left/right-signed (``+`` favours left). Rotate it
    onto the stimulus so it is ``+`` when the model's Q favours the *rewarded*
    side and ``-`` when it favours the wrong side — the ``|q|·sign(DV)·sign(q)``
    convention ``model_to_behavior``'s heatmap uses (identical to
    ``sign(DV)·q``). DV == 0 trials keep the raw signed value.
    """
    df = df.copy()
    q = df[q_col].to_numpy(dtype=float)
    dv = df["DV"].to_numpy(dtype=float)
    q_rel = np.abs(q) * (np.sign(dv) * np.sign(q))
    dv0 = dv == 0
    q_rel[dv0] = q[dv0]
    df[out_col] = q_rel
    return df


def label_fast_slow(df, *, rt_col="calcStimulusTime", speed_col="Speed"):
    """Return only the fast and slow trials, tagged in ``speed_col``.

    Fast = shortest-RT third, slow = longest third, split *within* each
    difficulty (``plotter._dvQuantileFn``); the typical (middle) third is
    dropped. Trials with a null RT are excluded by ``_dvQuantileFn``.
    """
    fast, _typical, slow = _dvQuantileFn(df, rt_col, as_df=True)
    fast = fast.copy()
    fast[speed_col] = "Fast"
    slow = slow.copy()
    slow[speed_col] = "Slow"
    return pd.concat([fast, slow])


def label_speeds(df, *, rt_col="calcStimulusTime", speed_col="Speed"):
    """Return *all* trials tagged ``Fast``/``Typical``/``Slow`` in ``speed_col``.

    Like :func:`label_fast_slow` but keeps the middle (typical) RT-third instead
    of dropping it — needed by the stacked speed histograms, which show how the
    three speeds are composed along each latent. Split is still per-difficulty
    (``plotter._dvQuantileFn``); null-RT trials are excluded.
    """
    parts = _dvQuantileFn(df, rt_col, as_df=True)      # (fast, typical, slow)
    out = []
    for part, name in zip(parts, ("Fast", "Typical", "Slow")):
        part = part.copy()
        part[speed_col] = name
        out.append(part)
    return pd.concat(out)


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------
def _subject_mle_df(subject, col_fit, df_behavior, *, reuse_fit_df,
                    mle_terminal_c):
    """The subject's per-trial ``mle_df`` — reused from the fit or recomputed.

    ``reuse_fit_df=True`` reads the per-trial ``mle_df`` already stored in the
    payload (fast, needs no ``df_behavior``); ``False`` re-evaluates the fitted
    params under MLE (``compare._compute_mle``, needs ``df_behavior``). Returns
    ``(mle_df | None, error | None)``.
    """
    if reuse_fit_df:
        stored = col_fit.payload.get("mle_df")
        if not isinstance(stored, pd.DataFrame):
            raise KeyError(
                f"Subject {subject!r} has no stored 'mle_df' in the fit "
                f"{col_fit.filename!r}; re-run with reuse_fit_df=False.")
        return stored, None
    include_Q, include_RewardRate = compare._include_flags(col_fit.payload)
    mle_df, _nll, _lapse, err = compare._compute_mle(
        subject, col_fit.fid, col_fit.payload, df_behavior,
        include_Q, include_RewardRate, mle_terminal_c, None)
    return mle_df, err


def build_fast_slow_qr(fits, df_behavior=None, *, spec=CHI2_W05_SPEC,
                       reuse_fit_df=True, include_typical=False,
                       bias_scaled=False, min_num_trials=MIN_NUM_TRIALS,
                       mle_terminal_c=MLE_TERMINAL_C.Default, verbose=True):
    """``{subject: per-trial df}`` for every modelling mouse.

    Each frame is the subject's speed-labelled trials with ``RewardRate`` (the
    R-learning latent), ``Q_relative`` and ``Speed`` columns. Only subjects
    with ``>= min_num_trials`` valid trials are kept (the modelling cohort, as
    in ``model_analysis``); the speed split is computed per subject.

    ``bias_scaled`` (default **False**) chooses what ``Q_relative`` measures:

    - **False** — the raw normalized Q log-ratio ``mle_Q_rel_before``, rotated
      onto the DV side. Not comparable across subjects (each weights Q by its
      own ``BIAS_COEF``).
    - **True** — the fitted starting point ``mle_z`` instead (``clip(BIAS_COEF *
      q_rel + Q_VAL_OFFSET, -1, 1)``), rotated onto the DV side. This is the Q
      bias already expressed as a fraction of the decision threshold, so it *is*
      comparable across subjects. Needs the ``mle_z`` column (present in every
      MLE fit's ``mle_df``).

    ``include_typical`` (default **False**) keeps only the fast and slow trials
    (what the scatters use). Set it **True** to also keep the middle (typical)
    third — the scatters filter it back out, but the stacked speed histograms
    need all three.

    ``reuse_fit_df`` (default **True**) reads the per-trial latents already
    stored in each fit payload — recomputing is unnecessary because the fit
    saved the ``mle_df`` at fit time. It needs no ``df_behavior``. Set it
    ``False`` to re-evaluate the fitted params under MLE instead (slower, and
    then ``df_behavior`` from ``compare.prepare_behavior_df()`` is required).
    """
    if not reuse_fit_df and df_behavior is None:
        raise ValueError(
            "reuse_fit_df=False recomputes the MLE latents and needs "
            "df_behavior (from compare.prepare_behavior_df()).")
    labeller = label_speeds if include_typical else label_fast_slow
    q_col = _Z_COL if bias_scaled else _Q_COL
    resolved = resolve_spec(fits, spec)
    out = {}
    for subject in sorted(resolved):
        col_fit = resolved[subject]
        mle_df, err = _subject_mle_df(
            subject, col_fit, df_behavior, reuse_fit_df=reuse_fit_df,
            mle_terminal_c=mle_terminal_c)
        if err is not None:
            print(f"{subject}: {err}")
            continue
        df = mle_df[mle_df.valid].copy()      # valid trials define the cohort
        if len(df) < min_num_trials:
            if verbose:
                print(f"skip {subject}: {len(df)} valid trials "
                      f"< {min_num_trials}")
            continue
        if q_col not in df.columns:
            raise KeyError(
                f"Subject {subject!r}: fit {col_fit.filename!r} has no "
                f"{q_col!r} column needed for bias_scaled Q-relative.")
        df["RewardRate"] = df[_R_COL]
        df = add_q_relative(df, q_col=q_col)
        out[subject] = labeller(df)
        if verbose:
            n = out[subject]
            print(f"{subject}: {len(n)} trials "
                  f"({(n.Speed == 'Fast').sum()} fast / "
                  f"{(n.Speed == 'Typical').sum()} typical / "
                  f"{(n.Speed == 'Slow').sum()} slow)")
    return out


# --------------------------------------------------------------------------
# Plot 1 — per-animal scatter (and the all-subjects pooled scatter)
# --------------------------------------------------------------------------
def _qr_scatter(fs, *, point_size, seed, q_ylim, r_x_lim, fast_color, slow_color,
                title, ax):
    """Draw one shuffled fast/slow scatter of ``fs`` onto ``ax`` (created if
    ``None``); shared by the per-animal and all-subjects scatters."""
    x = fs["RewardRate"].to_numpy(dtype=float)
    y = fs["Q_relative"].to_numpy(dtype=float)
    colors = np.where(fs["Speed"].to_numpy() == "Fast", fast_color, slow_color)
    order = np.random.default_rng(seed).permutation(len(fs))

    if ax is None:
        _fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x[order], y[order], c=colors[order], s=point_size, alpha=0.5,
               edgecolors="none")
    ax.set_xlim(*r_x_lim)
    ax.set_ylim(*q_ylim)
    ax.set_xlabel("R-Learning: Reward Rate")
    ax.set_ylabel("$Q_{Relative}$")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_subject_qr_scatter(subject_df, subject, *, point_size=2, seed=0,
                            q_ylim=(-1.0, 1.0), r_x_lim=(1.0, 0.0),
                            fast_color=FAST_COLOR, slow_color=SLOW_COLOR,
                            ax=None, save_prefix=None):
    """Scatter one animal's fast/slow trials in (reward-rate, Q-relative).

    All points are shuffled into a *single* scatter call with per-point colours
    so neither fast nor slow paints over the other (a two-call plot would let
    the second category hide the first). In Illustrator the two groups are
    still separable via Select > Same > Fill Color. ``seed`` fixes the shuffle.

    The reward-rate x-axis is drawn reversed (1 on the left, 0 on the right).
    ``q_ylim`` sets the Q-relative y-limits — the ``±1`` extremes are rarely
    reached, so pass e.g. ``(-0.5, 0.5)`` to zoom in on where the points are.
    """
    fs = subject_df[subject_df["Speed"].isin(("Fast", "Slow"))]
    ax = _qr_scatter(fs, point_size=point_size, seed=seed, q_ylim=q_ylim,
                     r_x_lim=r_x_lim, fast_color=fast_color, slow_color=slow_color,
                     title=f"Subject: {subject}\nfast (red) / slow (yellow)",
                     ax=ax)
    if save_prefix is not None:
        fp = f"{save_prefix}/FastSlowQR/{safe_filename(subject)}.svg"
        import pathlib
        pathlib.Path(fp).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(fp, bbox_inches="tight")
    return ax


def plot_all_subjects_qr_scatter(subject_dfs, *, point_size=2, seed=0,
                                 q_ylim=(-1.0, 1.0), r_x_lim=(1.0, 0.0),
                                 fast_color=FAST_COLOR, slow_color=SLOW_COLOR,
                                 ax=None, save_prefix=None):
    """Scatter *every* animal's fast/slow trials pooled into one figure.

    Same layout and reversed x-axis as :func:`plot_subject_qr_scatter`, but all
    modelling mice are concatenated so the population-level (reward-rate,
    Q-relative) structure is visible in a single panel. Points are again
    shuffled into one scatter so neither colour paints over the other.
    """
    pooled = pd.concat(subject_dfs.values(), ignore_index=True)
    fs = pooled[pooled["Speed"].isin(("Fast", "Slow"))]
    n = len(subject_dfs)
    # Place out of bound limit values at the limits
    fs = fs.copy()
    fs.loc[fs["Q_relative"] < q_ylim[0], "Q_relative"] = q_ylim[0]
    fs.loc[fs["Q_relative"] > q_ylim[1], "Q_relative"] = q_ylim[1]
    fs.loc[fs["RewardRate"] < r_x_lim[1], "RewardRate"] = r_x_lim[1]
    fs.loc[fs["RewardRate"] > r_x_lim[0], "RewardRate"] = r_x_lim[0]
    ax = _qr_scatter(fs, point_size=point_size, seed=seed, q_ylim=q_ylim,
                     r_x_lim=r_x_lim, fast_color=fast_color,
                     slow_color=slow_color,
                     title=f"All subjects (n = {n})\nfast (red) / slow (yellow)",
                     ax=ax)
    if save_prefix is not None:
        fp = f"{save_prefix}/FastSlowQR/all_subjects.svg"
        import pathlib
        pathlib.Path(fp).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(fp, bbox_inches="tight")
    return ax


# --------------------------------------------------------------------------
# Plot 2 — pooled fastness heatmap
# --------------------------------------------------------------------------
def fastness_grid(df, *, n_bins_r, n_bins_q):
    """``(grid, counts)`` — per-bin slow fraction over (reward-rate, Q-relative).

    ``grid[q_bin, r_bin] = n_slow / (n_fast + n_slow)`` in ``[0, 1]`` (NaN where
    a bin has no fast/slow trials), oriented rows = Q (−1..1), cols = R (0..1).
    Reward-rate is binned over ``[0, 1]`` and Q-relative over ``[-1, 1]``.
    """
    r_edges = np.linspace(0.0, 1.0, n_bins_r + 1)
    q_edges = np.linspace(-1.0, 1.0, n_bins_q + 1)
    r_idx = np.clip(np.digitize(df["RewardRate"].to_numpy(), r_edges) - 1,
                    0, n_bins_r - 1)
    q_idx = np.clip(np.digitize(df["Q_relative"].to_numpy(), q_edges) - 1,
                    0, n_bins_q - 1)
    is_slow = (df["Speed"].to_numpy() == "Slow").astype(float)

    n_slow = np.zeros((n_bins_q, n_bins_r), dtype=float)
    n_total = np.zeros((n_bins_q, n_bins_r), dtype=float)
    np.add.at(n_slow, (q_idx, r_idx), is_slow)
    np.add.at(n_total, (q_idx, r_idx), 1.0)
    grid = np.divide(n_slow, n_total,
                     out=np.full_like(n_slow, np.nan), where=n_total > 0)
    return grid, n_total


def plot_pooled_fastness_heatmap(subject_dfs, *, n_bins_r=20, n_bins_q=20,
                                 ax=None, save_prefix=None):
    """Pool every animal's fast/slow trials into a slow-fraction heatmap.

    x = reward-rate (R-learning), y = Q-relative — the ``model_to_behavior``
    Q/R heatmap with the axes swapped. Colour is the slow fraction under the
    autumn map: red (0) = bin is mostly fast, yellow (1) = mostly slow. ``n_bins_r``
    and ``n_bins_q`` set the resolution.
    """
    pooled = pd.concat(subject_dfs.values(), ignore_index=True)
    grid, _counts = fastness_grid(pooled, n_bins_r=n_bins_r, n_bins_q=n_bins_q)

    cmap = plt.cm.autumn.with_extremes(bad="white")
    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0, origin="lower",
                   extent=[0.0, 1.0, -1.0, 1.0], aspect="auto")
    ax.set_xlim(1.0, 0.0)      # reversed: reward-rate 1 on the left, 0 on right
    ax.set_xlabel("R-Learning: Reward Rate")
    ax.set_ylabel("$Q_{Relative}$")
    ax.set_title("Slow fraction by reward-rate and Q value\n"
                 "(red = fast, yellow = slow)")
    ax.spines[["top", "right"]].set_visible(False)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("slow fraction  n$_{slow}$ / (n$_{fast}$ + n$_{slow}$)")
    if save_prefix is not None:
        fp = f"{save_prefix}/FastSlowQR_heatmap.svg"
        ax.figure.savefig(fp, dpi=300, bbox_inches="tight")
    return ax


# --------------------------------------------------------------------------
# Plot 3 — stacked speed-composition histograms
# --------------------------------------------------------------------------
# Fast -> Typical -> Slow drawn bottom -> top, following the autumn gradient
# red -> orange -> yellow.
_SPEED_ORDER = ("Fast", "Typical", "Slow")
TYPICAL_COLOR = "orange"


def speed_fraction_hist(df, value_col, *, n_bins, value_range):
    """Per-bin composition of Fast/Typical/Slow along ``value_col``.

    Bins ``value_col`` over ``value_range`` into ``n_bins`` and, within each
    bin, returns the *fraction* of trials that are fast, typical and slow (the
    three sum to 1 in every non-empty bin; NaN where a bin is empty). Returns
    ``(edges, fractions, totals)`` where ``fractions`` maps each speed to an
    ``n_bins`` array and ``totals`` is the per-bin trial count.
    """
    lo, hi = value_range
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(df[value_col].to_numpy(dtype=float), edges) - 1,
                  0, n_bins - 1)
    speed = df["Speed"].to_numpy()
    totals = np.zeros(n_bins, dtype=float)
    np.add.at(totals, idx, 1.0)
    fractions = {}
    for s in _SPEED_ORDER:
        counts = np.zeros(n_bins, dtype=float)
        np.add.at(counts, idx[speed == s], 1.0)
        fractions[s] = np.divide(counts, totals,
                                 out=np.full(n_bins, np.nan), where=totals > 0)
    return edges, fractions, totals


def plot_speed_stacked_hist(pooled_df, value_col, *, n_bins, value_range,
                            reverse_x=False, xlabel=None, ax=None,
                            fast_color=FAST_COLOR, typical_color=TYPICAL_COLOR,
                            slow_color=SLOW_COLOR):
    """Stacked bar of the fast/typical/slow fraction along one latent.

    Each bar spans one bin of ``value_col`` and stacks to 1: fast (bottom),
    typical (middle), slow (top), so the plot reads off how the speed mix shifts
    as the latent grows. ``reverse_x`` flips the axis (used for reward-rate, 1
    on the left). Empty bins draw nothing.
    """
    edges, fractions, _totals = speed_fraction_hist(
        pooled_df, value_col, n_bins=n_bins, value_range=value_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    colors = {"Fast": fast_color, "Typical": typical_color, "Slow": slow_color}

    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 4))
    bottom = np.zeros(n_bins, dtype=float)
    for s in _SPEED_ORDER:
        h = np.nan_to_num(fractions[s])           # empty bins -> zero height
        ax.bar(centers, h, width=width, bottom=bottom, color=colors[s],
               label=s, edgecolor="none")
        bottom += h
    ax.set_ylim(0.0, 1.0)
    if reverse_x:
        ax.set_xlim(value_range[1], value_range[0])
    else:
        ax.set_xlim(value_range[0], value_range[1])
    ax.set_xlabel(xlabel if xlabel is not None else value_col)
    ax.set_ylabel("fraction of trials")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_speed_histograms(subject_dfs, *, n_bins_r=20, n_bins_q=20,
                          r_value_range=(0.0, 1.0), q_value_range=(-1.0, 1.0),
                          save_prefix=None):
    """Two pooled stacked histograms: speed composition vs R and vs Q-relative.

    Left panel bins the reward-rate (R-learning) over ``[0, 1]`` with its axis
    reversed (1 on the left, matching the scatter/heatmap); right panel bins
    Q-relative over ``[-1, 1]``. Each bar stacks fast/typical/slow to 1.
    ``n_bins_r`` / ``n_bins_q`` set the resolution. Needs typical trials, so
    build with ``build_fast_slow_qr(..., include_typical=True)``.
    """
    pooled = pd.concat(subject_dfs.values(), ignore_index=True)
    fig, (ax_r, ax_q) = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_speed_stacked_hist(pooled, "RewardRate", n_bins=n_bins_r,
                            value_range=r_value_range, reverse_x=True,
                            xlabel="R-Learning: Reward Rate", ax=ax_r)
    plot_speed_stacked_hist(pooled, "Q_relative", n_bins=n_bins_q,
                            value_range=q_value_range, reverse_x=False,
                            xlabel="$Q_{Relative}$", ax=ax_q)
    ax_r.set_title("Reward rate")
    ax_q.set_title("Q relative")
    ax_q.legend(loc="upper right", frameon=False, title="Speed")
    fig.suptitle("Speed composition along each latent  "
                 "(fast = red, typical = orange, slow = gold)")
    fig.tight_layout()
    if save_prefix is not None:
        fp = f"{save_prefix}/FastSlowQR_speed_histograms.svg"
        fig.savefig(fp, dpi=300, bbox_inches="tight")
    return fig, (ax_r, ax_q)
