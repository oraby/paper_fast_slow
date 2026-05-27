"""Loading and normalization helpers for MLE notebook exploration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


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
    """Load `mle_*.pkl` result files and optionally matching `pp_*.pkl` files."""
    result_dir = Path(result_dir) if result_dir is not None else default_result_dir()
    if not result_dir.exists():
        raise FileNotFoundError(f"MLE result directory not found: {result_dir}")

    loaded: list[MLEModelResult] = []
    for mle_path in sorted(result_dir.glob("mle_*.pkl")):
        with mle_path.open("rb") as f:
            mle_payload = pickle.load(f)

        pp_path = mle_path.with_name(mle_path.name.replace("mle_", "pp_", 1))
        posterior_payload = None
        if attach_posterior and pp_path.exists():
            with pp_path.open("rb") as f:
                posterior_payload = pickle.load(f)

        model_name = mle_path.stem.removeprefix("mle_")
        if _looks_like_subject_map(mle_payload):
            for subject, subject_result in mle_payload.items():
                pp_subject = _posterior_for_subject(posterior_payload, subject)
                loaded.append(MLEModelResult(
                    model_name=model_name,
                    subject=str(subject),
                    result=subject_result,
                    posterior_result=pp_subject,
                    mle_path=mle_path,
                    posterior_path=pp_path if pp_path.exists() else None,
                ))
        else:
            subject = str(_subject_from_result(mle_payload, fallback=mle_path.stem))
            loaded.append(MLEModelResult(
                model_name=model_name,
                subject=subject,
                result=mle_payload,
                posterior_result=posterior_payload,
                mle_path=mle_path,
                posterior_path=pp_path if pp_path.exists() else None,
            ))
    return loaded


def flatten_mle_results(results: list[MLEModelResult]) -> pd.DataFrame:
    """Concatenate all `mle_df` tables with subject/model metadata columns."""
    pieces = []
    for item in results:
        mle_df = item.result.get("mle_df")
        if mle_df is None:
            continue
        df = mle_df.copy()
        df.insert(0, "mle_model_name", item.model_name)
        df.insert(1, "mle_subject", item.subject)
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
