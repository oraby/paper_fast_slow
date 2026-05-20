"""Array backend selection for MLE fitting."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable
import importlib
import warnings

import numpy as np


SUPPORTED_ARRAY_BACKENDS = {"auto", "numpy", "cupy"}
SUPPORTED_CUPY_FALLBACKS = {"numpy", "error"}


@dataclass(frozen=True)
class ArrayBackend:
    requested_backend: str
    actual_backend: str
    # ``xp`` is the array module (numpy / cupy / any numpy-API-compatible
    # backend). Typed Any so callers can pass it as the ``xp`` argument to
    # functions whose pyright type hints declare it as ``ModuleType``.
    xp: Any
    normal_cdf: Callable
    device_id: int | None = None
    warning: str | None = None

    def metadata(self):
        return {
            "requested_backend": self.requested_backend,
            "actual_backend": self.actual_backend,
            "device_id": self.device_id,
            "warning": self.warning,
        }


@lru_cache(maxsize=16)
def resolve_array_backend(requested_backend="auto", device_id=None,
                          cupy_fallback="error"):
    """Resolve the requested MLE array backend.

    ``auto`` and ``cupy`` both try to import CuPy and select the requested CUDA
    device. If that fails, fallback behavior is controlled by
    ``cupy_fallback``. The function is cached so unavailable CuPy is not
    re-imported on every objective evaluation.
    """
    print(f"Resolving MLE array backend (requested={requested_backend}, "
          f"device_id={device_id}, cupy_fallback={cupy_fallback})...")
    requested_backend = str(requested_backend).lower()
    cupy_fallback = str(cupy_fallback).lower()
    if requested_backend not in SUPPORTED_ARRAY_BACKENDS:
        raise ValueError(
            f"Unknown MLE array backend {requested_backend!r}. Expected "
            f"{sorted(SUPPORTED_ARRAY_BACKENDS)}.")
    if cupy_fallback not in SUPPORTED_CUPY_FALLBACKS:
        raise ValueError(
            f"Unknown CuPy fallback {cupy_fallback!r}. Expected "
            f"{sorted(SUPPORTED_CUPY_FALLBACKS)}.")
    if requested_backend == "numpy":
        return ArrayBackend(
            requested_backend=requested_backend,
            actual_backend="numpy",
            xp=np,
            normal_cdf=_numpy_ndtr,
            device_id=None,
        )

    try:
        cp = importlib.import_module("cupy")
        # import cupy as cp
        print(f"Testing CuPy on device {device_id}...")
        special = importlib.import_module("cupyx.scipy.special")
        cupy_ndtr = special.ndtr
        print(f"CuPy backend {cupy_ndtr} loaded successfully.")
        if device_id is not None:
            cp.cuda.Device(int(device_id)).use()
        arr = cp.array([1.0])
        _ = cupy_ndtr(arr)
        cp.cuda.Stream.null.synchronize()
        _ = float(cp.asnumpy(arr)[0])
        return ArrayBackend(
            requested_backend=requested_backend,
            actual_backend="cupy",
            xp=cp,
            normal_cdf=cupy_ndtr,
            device_id=None if device_id is None else int(device_id),
        )
    except Exception as exc:
        message = (
            f"Requested MLE array backend {requested_backend!r} could not use "
            f"CuPy/CUDA ({type(exc).__name__}: {exc}).")
        if cupy_fallback == "error":
            raise RuntimeError(message) from exc
        warning = f"{message} Falling back to NumPy."
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
        return ArrayBackend(
            requested_backend=requested_backend,
            actual_backend="numpy",
            xp=np,
            normal_cdf=_numpy_ndtr,
            device_id=None if device_id is None else int(device_id),
            warning=warning,
        )


def asnumpy(xp, value):
    if hasattr(xp, "asnumpy"):
        return xp.asnumpy(value)
    return np.asarray(value)


def assert_gpu_backend(backend):
    """Hard-fail unless ``backend`` actually allocates on the GPU.

    Used as the pre-flight check just before the MLE optimizer runs when the
    user passed ``--mle-backend GPU``. Catches two failure modes that
    ``resolve_array_backend`` alone won't:

    1. The resolver said it picked CuPy, but the actual ``xp`` module is
       NumPy (e.g. a stale module cache, monkey-patched fallback).
    2. CuPy imported but allocations silently land in host memory (e.g. a
       broken CUDA install, no visible device, wrong device id).

    Allocates a four-element probe array and asserts its ``type(...).__module__``
    starts with ``"cupy"``. Returns the probe so callers can log device info.
    """
    if backend.actual_backend != "cupy":
        raise RuntimeError(
            f"GPU was requested for MLE but the resolved backend is "
            f"{backend.actual_backend!r}. Backend metadata: "
            f"{backend.metadata()}")
    probe = backend.xp.zeros((4,), dtype=float)
    module_name = type(probe).__module__
    print(f"Probe array allocated with type {type(probe)} "
          f"from module {module_name} on device {probe.device}.")
    if not module_name.startswith("cupy"):
        raise RuntimeError(
            f"Backend reports 'cupy' but the allocated probe array is from "
            f"{module_name!r}; GPU is not actually in use. Backend metadata: "
            f"{backend.metadata()}")
    return probe


def _numpy_ndtr(value):
    from scipy.special import ndtr
    return ndtr(np.asarray(value))
