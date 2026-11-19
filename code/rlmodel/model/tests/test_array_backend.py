import importlib

import pytest

from ..array_backend import resolve_array_backend


def setup_function():
    resolve_array_backend.cache_clear()


def test_cupy_import_failure_falls_back_to_numpy(monkeypatch):
    real_import = importlib.import_module

    def fake_import(name, package=None):
        if name == "cupy":
            raise ModuleNotFoundError("No module named 'cupy'")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    backend = resolve_array_backend("cupy", device_id=2, cupy_fallback="numpy")

    assert backend.requested_backend == "cupy"
    assert backend.actual_backend == "numpy"
    assert backend.device_id == 2
    assert "Falling back to NumPy" in backend.warning


def test_cupy_import_failure_can_raise(monkeypatch):
    real_import = importlib.import_module

    def fake_import(name, package=None):
        if name == "cupy":
            raise ModuleNotFoundError("No module named 'cupy'")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(RuntimeError, match="could not use CuPy"):
        resolve_array_backend("cupy", device_id=0, cupy_fallback="error")
