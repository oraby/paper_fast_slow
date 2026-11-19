"""Linked latent histogram filtering widgets for MLE notebooks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_LATENT_COLUMNS = {
    "Q-Val": "mle_Q_rel_before",
    "Q-Left": "mle_Q_left_before",
    "Q-Right": "mle_Q_right_before",
    "R-Value": "mle_reward_rate_before",
    # The lapse mixture and terminal-C threshold are constant within one
    # fit but vary across the population (different subjects converge to
    # different λ; users may explore different C settings). Surfacing
    # them here lets the explorer histogram the cross-fit distribution
    # alongside the per-trial latents.
    "λ (lapse)": "mle_lapse_rate",
    "C (terminal)": "mle_terminal_c",
    # Bound-RewardRate per-trial bound + ``--scale-bound`` opt-in. Same
    # 0/1 cross-fit histogram so users can filter by variant.
    "bound·r_t": "mle_uses_per_trial_bound",
    "scale-B": "mle_uses_scaled_bound",
}
FILTER_COLORS = ("orange", "green", "tab:red", "tab:purple", "tab:brown")


@dataclass(frozen=True)
class HistogramFilter:
    kind: str
    column: str
    label: str
    value: tuple[float, float] | str


class HistogramFilterState:
    """Stack-based non-mutating histogram filter state."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.filters: list[HistogramFilter] = []

    def add_bin_filter(self, column: str, low: float, high: float,
                       label: str | None = None) -> None:
        self.filters.append(HistogramFilter(
            kind="bin",
            column=column,
            label=label or column,
            value=(float(low), float(high)),
        ))

    def set_query(self, column: str, query: str, label: str | None = None) -> None:
        self.filters = [
            f for f in self.filters
            if not (f.kind == "query" and f.column == column)
        ]
        if query.strip():
            self.filters.append(HistogramFilter(
                kind="query",
                column=column,
                label=label or column,
                value=query.strip(),
            ))

    def undo(self) -> HistogramFilter | None:
        if not self.filters:
            return None
        return self.filters.pop()

    def mask(self) -> pd.Series:
        mask = pd.Series(True, index=self.df.index)
        for filt in self.filters:
            if filt.column not in self.df.columns:
                mask &= False
                continue
            if filt.kind == "bin":
                low, high = filt.value
                series = pd.to_numeric(self.df[filt.column], errors="coerce")
                mask &= series.ge(low) & series.lt(high)
            elif filt.kind == "query":
                mask &= _query_mask(self.df, filt.column, str(filt.value))
            else:
                raise ValueError(f"Unknown filter kind: {filt.kind!r}")
        return mask.fillna(False)

    def filtered_df(self) -> pd.DataFrame:
        return self.df.loc[self.mask()].copy()


def compute_histogram_layers(
    df: pd.DataFrame,
    state: HistogramFilterState,
    columns: dict[str, str] | None = None,
    *,
    bins: int | Iterable[float] = 30,
) -> dict[str, dict[str, np.ndarray]]:
    """Return total and filtered histogram counts for each configured column."""
    columns = columns or DEFAULT_LATENT_COLUMNS
    mask = state.mask()
    out = {}
    for label, column in columns.items():
        values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy()
        filtered = pd.to_numeric(df.loc[mask, column], errors="coerce").dropna().to_numpy()
        counts, edges = np.histogram(values, bins=bins)
        filt_counts, _ = np.histogram(filtered, bins=edges)
        out[label] = {
            "column": np.asarray(column),
            "edges": edges,
            "total": counts,
            "filtered": filt_counts,
        }
    return out


def show_latent_histogram_explorer(
    data: pd.DataFrame | list,
    *,
    columns: dict[str, str] | None = None,
    bins: int = 30,
):
    """Display linked histograms for MLE latent columns in a notebook."""
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display

    if isinstance(data, list):
        from .data import flatten_mle_results
        df = flatten_mle_results(data)
    else:
        df = data.copy()
    columns = columns or DEFAULT_LATENT_COLUMNS
    state = HistogramFilterState(df)
    query_boxes = {
        label: widgets.Text(description=label, layout=widgets.Layout(width="260px"))
        for label in columns
    }
    run_button = widgets.Button(description="Run/Update", button_style="primary")
    undo_button = widgets.Button(description="Undo")
    status = widgets.HTML()

    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 3.2))
    axes = np.atleast_1d(axes)
    patch_bins: dict[object, tuple[str, float, float]] = {}

    def redraw():
        patch_bins.clear()
        layers = compute_histogram_layers(df, state, columns, bins=bins)
        for ax, (label, column) in zip(axes, columns.items()):
            ax.clear()
            layer = layers[label]
            edges = layer["edges"]
            width = np.diff(edges)
            total_patches = ax.bar(
                edges[:-1], layer["total"], width=width, align="edge",
                color="0.78", edgecolor="0.35", label="all")
            ax.bar(
                edges[:-1], layer["filtered"], width=width, align="edge",
                color="orange", alpha=0.65, label="filtered")
            for patch, low, high in zip(total_patches, edges[:-1], edges[1:]):
                patch_bins[patch] = (column, float(low), float(high))
            for i, filt in enumerate(state.filters):
                if filt.kind == "bin" and filt.column == column:
                    low, high = filt.value
                    ax.axvspan(
                        low, high,
                        color=FILTER_COLORS[i % len(FILTER_COLORS)],
                        alpha=0.28,
                    )
            ax.set_title(label)
            ax.set_ylabel("count")
            ax.legend(loc="upper right", fontsize=8)
        status.value = f"{int(state.mask().sum()):,} / {len(df):,} rows selected"
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is None:
            return
        for patch, (column, low, high) in patch_bins.items():
            contains, _ = patch.contains(event)
            if contains:
                state.add_bin_filter(column, low, high)
                redraw()
                return

    def on_run(_button):
        for label, column in columns.items():
            state.set_query(column, query_boxes[label].value, label=label)
        redraw()

    def on_undo(_button):
        state.undo()
        redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    run_button.on_click(on_run)
    undo_button.on_click(on_undo)
    redraw()
    display(
        widgets.VBox([
            widgets.HBox(list(query_boxes.values())),
            widgets.HBox([run_button]),
            widgets.HBox([undo_button, status]),
        ])
    )
    return {"figure": fig, "state": state, "queries": query_boxes}


def _query_mask(df: pd.DataFrame, column: str, query: str) -> pd.Series:
    series = pd.to_numeric(df[column], errors="coerce")
    local_df = df.copy()
    local_df["x"] = series
    try:
        if column in query or "x" in query:
            result = local_df.eval(query, engine="python")
        else:
            result = local_df.eval(f"x {query}", engine="python")
    except Exception:
        try:
            result = local_df.query(query, engine="python").index
            return df.index.to_series().isin(result)
        except Exception as exc:
            raise ValueError(
                f"Invalid query for {column!r}: {query!r}") from exc
    return pd.Series(result, index=df.index).fillna(False).astype(bool)
