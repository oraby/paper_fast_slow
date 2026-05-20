import sys
import types
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


# def test_fake_cupy_selects_device(monkeypatch):
#     selected = []
#     ndtr_called = []

#     class FakeDevice:
#         def __init__(self, device_id):
#             self.device_id = device_id

#         def use(self):
#             selected.append(self.device_id)

#     class FakeStream:
#         @staticmethod
#         def synchronize():
#             return None

#     fake_cupy = types.SimpleNamespace(
#         __name__="cupy",
#         asarray=lambda value: value,
#         asnumpy=lambda value: value,
#         cuda=types.SimpleNamespace(
#             Device=FakeDevice,
#             Stream=types.SimpleNamespace(null=FakeStream()),
#         ),
#     )
#     fake_special = types.SimpleNamespace(
#         ndtr=lambda value: ndtr_called.append(value) or value,
#     )
#     monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
#     monkeypatch.setitem(sys.modules, "cupyx.scipy.special", fake_special)

#     backend = resolve_array_backend("cupy", device_id=3, cupy_fallback="error")

#     assert backend.actual_backend == "cupy"
#     assert backend.device_id == 3
#     assert selected == [3]
#     assert ndtr_called


# def test_fake_cupy_requires_cupyx_ndtr(monkeypatch):
#     class FakeStream:
#         @staticmethod
#         def synchronize():
#             return None

#     fake_cupy = types.SimpleNamespace(
#         __name__="cupy",
#         asarray=lambda value: value,
#         asnumpy=lambda value: value,
#         cuda=types.SimpleNamespace(
#             Device=lambda device_id: types.SimpleNamespace(use=lambda: None),
#             Stream=types.SimpleNamespace(null=FakeStream()),
#         ),
#     )
#     monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
#     monkeypatch.delitem(sys.modules, "cupyx.scipy.special", raising=False)

#     with pytest.raises(RuntimeError, match="cupyx.scipy.special.ndtr"):
#         resolve_array_backend("cupy", device_id=0, cupy_fallback="error")
