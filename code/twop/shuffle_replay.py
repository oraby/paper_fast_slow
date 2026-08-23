"""Interactive step-by-step replay of one Mallows shuffle.

Visualises the repeated-insertion (RIM) construction the sequence-deviation
calibration uses (:func:`seqdeviation.rim_trace` / ``_mallows_phi``): pick a real
trial and shuffle level, press Run to *record* the insertion steps, then drag a
slider to watch one permutation get built, alongside a small plot of how the level
maps to ``phi``.

The recordable math lives in :mod:`seqdeviation` (pure, uv-tested). ``ipywidgets``
is only in the conda ``py312`` notebook kernel, so it is imported **lazily** inside
:func:`show_shuffle_replay`; the matplotlib drawing helpers stay importable/renderable
under uv (matplotlib is a project dependency), which is what the headless smoke test
exercises.
"""
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from . import seqdeviation as sd

_NEW_CLR = "#d62728"      # the just-inserted neuron
_TOKEN_CLR = "0.85"       # already-placed neurons
_REF_CLR = "0.6"          # the reference row


def _fs(n):
    """Token font size that shrinks as the active set grows."""
    return 9 if n <= 18 else 7 if n <= 32 else 5


def _token(ax, x, y, text, face, textcolor, n, bold=False):
    w = 0.82
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - 0.28), w, 0.56,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=0.6, edgecolor="black", facecolor=face))
    ax.text(x, y, str(text), ha="center", va="center", fontsize=_fs(n),
            color=textcolor, fontweight=("bold" if bold else "normal"))


def _draw_phi_curve(ax, r):
    """Expected inverted-pair fraction vs ``phi``, with the target level and the
    solved ``phi`` marked -- i.e. how ``_mallows_phi`` turns the level into ``phi``."""
    ax.plot(r.phi_grid, r.phi_frac, color="0.3", lw=1.6)
    ax.axhline(r.level, ls="--", color="0.55", lw=1)
    ax.axvline(r.phi, ls="--", color=_NEW_CLR, lw=1)
    ax.plot([r.phi], [r.level], "o", color=_NEW_CLR, ms=7,
            markeredgecolor="black", markeredgewidth=0.6, zorder=5)
    ax.annotate(f"level {r.level * 100:.0f}%  ->  phi = {r.phi:.3f}",
                xy=(r.phi, r.level), xytext=(0.03, 0.92),
                textcoords="axes fraction", fontsize="small", color=_NEW_CLR,
                va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.52)
    ax.set_xlabel("phi  (Mallows dispersion)")
    ax.set_ylabel("expected fraction\nof pairs inverted")
    ax.set_title(f"How the shuffle level sets phi  (n = {r.n} neurons)",
                 fontsize="small")
    ax.spines[["right", "top"]].set_visible(False)


def _draw_step(ax, r, k):
    """The insertion track at recorded step ``k``: the reference row on top and the
    growing shuffled order below, with the just-inserted neuron highlighted and its
    jump (``z`` inversions) annotated. Neurons are labelled by reference position."""
    steps = r.steps
    k = int(np.clip(k, 0, len(steps) - 1))
    s = steps[k]
    n = r.n
    ax.set_xlim(-1.4, n - 0.4)
    ax.set_ylim(-0.4, 2.5)
    ax.axis("off")

    # reference row (all n neurons, in reference order)
    for i in range(n):
        _token(ax, i, 2.0, i + 1, "white", _REF_CLR, n)
    ax.text(-1.3, 2.0, "reference", ha="left", va="center",
            fontsize="small", color=_REF_CLR)

    # current partial order
    for slot, item in enumerate(s["order"]):
        is_new = item == s["inserted"]
        _token(ax, slot, 0.6, item + 1,
               _NEW_CLR if is_new else _TOKEN_CLR,
               "white" if is_new else "black", n, bold=is_new)
    ax.text(-1.3, 0.6, "shuffled", ha="left", va="center",
            fontsize="small", color="0.3")

    # arrow marking the leftward jump (from the append slot j-1 to pos)
    if s["z"] > 0:
        ax.annotate("", xy=(s["pos"], 1.18), xytext=(s["j"] - 1, 1.18),
                    arrowprops=dict(arrowstyle="->", color=_NEW_CLR, lw=1.3))
        ax.text((s["pos"] + s["j"] - 1) / 2, 1.34, f"jumped {s['z']}",
                ha="center", va="bottom", fontsize="x-small", color=_NEW_CLR)

    header = (f"Step {s['j']}/{n}:  insert neuron #{s['j']} at slot {s['pos'] + 1}  "
              f"(z = {s['z']} inversion{'s' if s['z'] != 1 else ''})   |   "
              f"total inversions {s['cum_inv']}/{r.max_d:.0f}")
    if k == len(steps) - 1:
        header += (f"   ->   disorder {r.gap_norm:.3f},  "
                   f"{r.kendall_frac * 100:.0f}% of pairs inverted")
    ax.set_title(header, fontsize="small", loc="left")


def show_shuffle_replay(cross_df):
    """Display the interactive shuffle-replay widget in a notebook.

    ``cross_df`` is the concatenated cross frame from ``build_cross_penalty_df``
    (both references). Requires the conda ``py312`` kernel (ipywidgets); it will not
    render under the headless/uv runner.
    """
    import ipywidgets as widgets
    from IPython.display import display

    regions = sorted(cross_df["BrainRegion"].unique())
    refs = [sd.STRATEGY_FAST, sd.STRATEGY_SLOW]

    region_dd = widgets.Dropdown(options=regions, description="Region")
    session_dd = widgets.Dropdown(description="Session")
    reference_dd = widgets.Dropdown(options=refs, description="Reference")
    trial_dd = widgets.Dropdown(description="Trial")
    level_slider = widgets.FloatSlider(
        value=20, min=0, max=50, step=1, description="Level %",
        continuous_update=False, readout_format=".0f")
    permute_int = widgets.BoundedIntText(value=0, min=0, max=10**6,
                                         description="Permute #")
    run_button = widgets.Button(description="Run / record", button_style="primary")
    iter_slider = widgets.IntSlider(value=0, min=0, max=0, description="step")
    status = widgets.HTML()
    plot_out = widgets.Output()
    state = SimpleNamespace(result=None)

    def refresh_trials(*_):
        if session_dd.value is None:
            trial_dd.options = []
            return
        trials = sd.replay_trials(cross_df, region_dd.value, session_dd.value,
                                  reference_dd.value)
        trial_dd.options = [(f"trial {t}  (n={c})", t) for t, c in trials]

    def refresh_sessions(*_):
        session_dd.options = sd.replay_sessions(cross_df, region_dd.value)
        refresh_trials()

    def render(*_):
        r = state.result
        with plot_out:
            plot_out.clear_output(wait=True)
            if r is None:
                return
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(max(8.0, 0.34 * r.n + 2.0), 6.4),
                gridspec_kw=dict(height_ratios=[1.0, 1.1]))
            _draw_phi_curve(ax1, r)
            _draw_step(ax2, r, iter_slider.value)
            fig.tight_layout()
            display(fig)
            plt.close(fig)

    def on_run(_b):
        if trial_dd.value is None:
            status.value = ("<span style='color:#c00'>Pick a region, session and "
                            "trial, then Run.</span>")
            return
        r = sd.replay_shuffle(cross_df, region_dd.value, session_dd.value,
                              reference_dd.value, trial_dd.value,
                              level_slider.value / 100.0, permute_int.value)
        state.result = r
        iter_slider.max = max(len(r.steps) - 1, 0)
        iter_slider.value = iter_slider.max      # land on the finished permutation
        status.value = (
            f"<b>{r.session}</b> · trial {r.trial_number} · {r.reference} reference"
            f" · n={r.n} · phi={r.phi:.3f} · final disorder {r.gap_norm:.3f}"
            f" ({r.kendall_frac * 100:.0f}% of pairs inverted). "
            f"Drag <i>step</i> to replay the insertions.")
        render()

    region_dd.observe(lambda c: refresh_sessions(), names="value")
    session_dd.observe(lambda c: refresh_trials(), names="value")
    reference_dd.observe(lambda c: refresh_trials(), names="value")
    iter_slider.observe(render, names="value")
    run_button.on_click(on_run)

    refresh_sessions()
    display(widgets.VBox([
        widgets.HBox([region_dd, session_dd, reference_dd, trial_dd]),
        widgets.HBox([level_slider, permute_int, run_button]),
        iter_slider,
        status,
        plot_out,
    ]))
