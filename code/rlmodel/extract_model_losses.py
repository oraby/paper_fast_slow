"""Aggregate per-subject fit losses across saved model result files.

Sweeps the merged ``{subject: payload}`` pickles written by ``fit.simulateDDM``
(``data/RLModel/{fit_mode}_..._.pkl``) and, for each model x subject, extracts
the final loss (``OptimRes.fun``) plus the weights / conditions the fit ran with
and the time it was saved. Emits one tidy DataFrame for downstream loss-
distribution plots.

Usage (from the project root):

    python -m code.rlmodel.extract_model_losses \
        [--results-dir data/RLModel] [--out data/RLModel/loss_summary.pkl]

Writes both ``<out>`` (pickle) and ``<out>.csv``. Pure extraction — no refit.
"""
from __future__ import annotations

import argparse
import importlib
import pathlib
import pickle
import re
import sys

import numpy as np
import pandas as pd


def _loadFit():
    """Resolve ``model.fitio.loadFit`` without hardcoding the checkout's name.

    The fits embed the model's own functions, so they must be read through the
    prefix-agnostic loader rather than a plain ``pickle.load``.
    """
    here = pathlib.Path(__file__).resolve()
    repo_root = here.parents[2]
    if str(repo_root.parent) not in sys.path:
        sys.path.insert(0, str(repo_root.parent))
    mod = importlib.import_module(
        f"{repo_root.name}.code.rlmodel.model.fitio")
    return mod.loadFit


loadFit = _loadFit()

# Matches the ``_mleW{m}_chi2W{c}`` suffix that ``fit.evolveFP`` appends to joint
# MLE+Chi² fits (``{:g}`` formatted floats, so ``1``, ``0.5``, ``2.5`` …).
_WEIGHT_SUFFIX_RE = re.compile(
    r"_mleW(?P<m>[-+0-9.eE]+)_chi2W(?P<c>[-+0-9.eE]+)$")


def _parse_filename(stem):
    """Best-effort ``(fit_mode, mle_weight, chi2_weight)`` from a result-file
    stem. ``fit_mode`` is the token before the first ``_``; the weights come
    from the joint suffix (defaults 1.0 / 0.0 when absent, i.e. pure MLE/chisq).
    These are fallbacks — ``model_config`` on the payload is authoritative."""
    fit_mode = stem.split("_", 1)[0]
    m = _WEIGHT_SUFFIX_RE.search(stem)
    if m:
        return fit_mode, float(m.group("m")), float(m.group("c"))
    return fit_mode, 1.0, 0.0


def _loss_of(payload):
    """The fit's minimized loss: ``OptimRes.fun``, falling back to a stored
    ``neg_loglik``. ``None`` if neither is a finite number (e.g. a dry-run
    payload with ``OptimRes=None``)."""
    optim = payload.get("OptimRes") if isinstance(payload, dict) else None
    loss = getattr(optim, "fun", None)
    if loss is None and isinstance(payload, dict):
        loss = payload.get("neg_loglik")
    if loss is None or not np.isfinite(loss):
        return None
    return float(loss)


def _cfg_get(model_config, name, default=None):
    return getattr(model_config, name, default) if model_config is not None else default


def _row_for(model_file, fit_mode, w_mle_file, w_chi2_file, subject, payload):
    cfg = payload.get("model_config") if isinstance(payload, dict) else None
    conds = _cfg_get(cfg, "mle_condition_columns")
    return {
        "model_file": model_file,
        "fit_mode": fit_mode,
        "subject": subject,
        "loss": _loss_of(payload),
        # Model identity (from model_config when present — MLE; None for chisq,
        # whose payload doesn't store the fn strings — the filename identifies it).
        "drift_fn": _cfg_get(cfg, "drift_fn_str"),
        "bias_fn": _cfg_get(cfg, "bias_fn_str"),
        "noise_fn": _cfg_get(cfg, "noise_fn_str"),
        "dt": payload.get("dt") if isinstance(payload, dict) else None,
        "t_dur": payload.get("t_dur") if isinstance(payload, dict) else None,
        # Weights / conditions (model_config authoritative; filename as fallback).
        "mle_mle_weight": _cfg_get(cfg, "mle_mle_weight", w_mle_file),
        "mle_chi2_weight": _cfg_get(cfg, "mle_chi2_weight", w_chi2_file),
        "mle_choice_weight": _cfg_get(cfg, "mle_choice_weight"),
        "mle_rt_weight": _cfg_get(cfg, "mle_rt_weight"),
        "mle_choice_norm": _cfg_get(cfg, "mle_choice_norm"),
        "mle_condition_columns": str(tuple(conds)) if conds else None,
        # Joint breakdown, when this was a joint fit (else absent → NaN/None).
        "total_loss": payload.get("total_loss") if isinstance(payload, dict) else None,
        "mle_part_loss": payload.get("mle_part_loss") if isinstance(payload, dict) else None,
        "chi2_part_loss": payload.get("chi2_part_loss") if isinstance(payload, dict) else None,
        "ref_mle": payload.get("ref_mle") if isinstance(payload, dict) else None,
        "ref_chi2": payload.get("ref_chi2") if isinstance(payload, dict) else None,
        "save_time": payload.get("fit_finish_time") if isinstance(payload, dict) else None,
    }


def extract_losses(results_dir="data/RLModel"):
    """Build the per-model x subject loss DataFrame from ``results_dir``.

    Only top-level ``mle_*`` / ``chisq_*`` ``{subject: payload}`` pickles are
    read (the per-subject ``subject/`` copies and unrelated pickles are
    skipped). Files that don't unpickle to such a dict are skipped with a note.
    """
    results_dir = pathlib.Path(results_dir)
    rows = []
    for fp in sorted(results_dir.glob("*.pkl")):
        if not (fp.name.startswith("mle_") or fp.name.startswith("chisq_")):
            continue
        try:
            data = loadFit(fp)
        except Exception as exc:  # noqa: BLE001 — tolerate any unreadable pickle
            print(f"Skipping {fp.name}: unreadable ({exc!r})")
            continue
        if not isinstance(data, dict):
            print(f"Skipping {fp.name}: not a subject->payload dict")
            continue
        fit_mode, w_mle_file, w_chi2_file = _parse_filename(fp.stem)
        for subject, payload in data.items():
            rows.append(_row_for(fp.stem, fit_mode, w_mle_file, w_chi2_file,
                                 subject, payload))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="data/RLModel",
                        help="Directory of saved model result pickles.")
    parser.add_argument("--out", default="data/RLModel/loss_summary.pkl",
                        help="Output path (a sibling .csv is also written).")
    args = parser.parse_args()

    df = extract_losses(args.results_dir)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)
    df.to_csv(out.with_suffix(".csv"), index=False)
    print(f"Wrote {len(df)} rows ({df['model_file'].nunique()} model files, "
          f"{df['subject'].nunique()} subjects) to {out} and "
          f"{out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
