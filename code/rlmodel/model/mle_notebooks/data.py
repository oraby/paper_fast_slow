"""Loading and normalization helpers for MLE notebook exploration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from ..fitio import loadFit


@dataclass(frozen=True)
class MLEModelResult:
    """One loaded subject/model MLE result."""

    model_name: str
    subject: str
    result: dict
    posterior_result: object | None = None
    mle_path: Path | None = None
    posterior_path: Path | None = None


def default_result_dir(start: Path | None = None) -> Path:
    """Return the project `data/RLModel` directory from this module location."""
    if start is None:
        start = Path(__file__).resolve()
    # .../code/rlmodel/model/mle_notebooks/data.py -> project root is parents[4]
    project_root = start.parents[4]
    return project_root / "data" / "RLModel"


def load_mle_population_results(
    result_dir: str | Path | None = None,
    *,
    attach_posterior: bool = True,
) -> list[MLEModelResult]:
    """Load every ``mle_*.pkl`` and (optionally) matching ``pp_*.pkl``.

    Loading is intentionally permissive so the population explorer can
    surface every result on disk — not just the "standard" model name
    set:

    - Any file matching ``mle_*.pkl`` is loaded, whatever its model name
      stem. Non-standard variants (``mle_Classic_v2_…``, scratch fits,
      etc.) show up as additional Model dropdown entries for the
      affected subjects.
    - The posterior pickle is optional: a missing ``pp_*.pkl`` is the
      normal case and loading continues with ``posterior_result=None``.
    - Per-file errors (corrupt pickle, schema mismatch, unreadable
      posterior) are caught and reported via ``print`` — they do not
      abort the rest of the load.
    """
    result_dir = Path(result_dir) if result_dir is not None else default_result_dir()
    if not result_dir.exists():
        raise FileNotFoundError(f"MLE result directory not found: {result_dir}")

    loaded: list[MLEModelResult] = []
    for mle_path in sorted(result_dir.glob("mle_*.pkl")):
        try:
            mle_payload = loadFit(mle_path)
        except Exception as exc:  # noqa: BLE001 — surface every load failure
            print(
                f"Warning: skipping {mle_path.name} "
                f"(failed to load MLE pickle: {exc})")
            continue

        pp_path = mle_path.with_name(mle_path.name.replace("mle_", "pp_", 1))
        posterior_payload = None
        pp_available = False
        if attach_posterior and pp_path.exists():
            try:
                with pp_path.open("rb") as f:
                    posterior_payload = pickle.load(f)
                pp_available = True
            except Exception as exc:  # noqa: BLE001
                print(
                    f"Warning: {pp_path.name} exists but failed to load "
                    f"({exc}); continuing with MLE-only result.")
                posterior_payload = None

        model_name = mle_path.stem.removeprefix("mle_")
        posterior_path = pp_path if pp_available else None
        if _looks_like_subject_map(mle_payload):
            for subject, subject_result in mle_payload.items():
                pp_subject = _posterior_for_subject(posterior_payload, subject)
                loaded.append(MLEModelResult(
                    model_name=model_name,
                    subject=str(subject),
                    result=subject_result,
                    posterior_result=pp_subject,
                    mle_path=mle_path,
                    posterior_path=posterior_path,
                ))
        else:
            subject = str(_subject_from_result(mle_payload, fallback=mle_path.stem))
            loaded.append(MLEModelResult(
                model_name=model_name,
                subject=subject,
                result=mle_payload,
                posterior_result=posterior_payload,
                mle_path=mle_path,
                posterior_path=posterior_path,
            ))
    return loaded


def flatten_mle_results(results: list[MLEModelResult]) -> pd.DataFrame:
    """Concatenate all `mle_df` tables with subject/model metadata columns.

    ``mle_lapse_rate`` is already a per-trial column inside each ``mle_df``
    (constant within a fit; written by ``_build_mle_df`` in mle.py).
    ``mle_terminal_c`` is not — it lives on the saved ``model_config`` — so
    we attach it here as a per-row constant. Both columns then surface in
    the population histogram explorer so the user can see the cross-fit
    distribution of C / λ settings alongside per-trial latents.
    """
    pieces = []
    for item in results:
        mle_df = item.result.get("mle_df")
        if mle_df is None:
            continue
        df = mle_df.copy()
        df.insert(0, "mle_model_name", item.model_name)
        df.insert(1, "mle_subject", item.subject)
        model_config = item.result.get("model_config")
        if model_config is not None:
            df["mle_terminal_c"] = float(
                getattr(model_config, "mle_terminal_c", 0.0))
            # Bound-RewardRate per-trial bound + ``--scale-bound`` opt-in.
            # Same backwards-compat pattern: old pickles whose saved
            # MLEModelConfig predates these fields default to False so the
            # population explorer treats them as legacy fixed-bound fits.
            df["mle_uses_per_trial_bound"] = bool(
                getattr(model_config, "uses_per_trial_bound", False))
            df["mle_uses_scaled_bound"] = bool(
                getattr(model_config, "uses_scaled_bound", False))
        pieces.append(df)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True, sort=False)


def result_key(item: MLEModelResult) -> str:
    return f"{item.subject} | {item.model_name}"


def fitted_params_from_result(result: dict) -> dict[str, float]:
    """Extract fitted params from one saved MLE result dict."""
    names = list(result["params_names"])
    optim_res = result.get("OptimRes")
    if optim_res is None:
        values = np.asarray(result["params_init"], dtype=float)
    else:
        values = np.asarray(optim_res.x, dtype=float)
    return {str(name).upper(): float(value) for name, value in zip(names, values)}


def _looks_like_subject_map(payload) -> bool:
    return (
        isinstance(payload, dict)
        and payload
        and "fit_mode" not in payload
        and all(isinstance(v, dict) for v in payload.values())
    )


def _subject_from_result(result: dict, fallback: str) -> str:
    df = result.get("subject_df")
    if isinstance(df, pd.DataFrame) and "Name" in df.columns and len(df):
        return str(df["Name"].iloc[0])
    return fallback


def _posterior_for_subject(posterior_payload, subject):
    if isinstance(posterior_payload, dict) and subject in posterior_payload:
        return posterior_payload[subject]
    if isinstance(posterior_payload, pd.DataFrame) and "Name" in posterior_payload:
        return posterior_payload[posterior_payload["Name"] == subject].copy()
    return posterior_payload
