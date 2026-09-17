"""Reading and writing saved model fits.

A fit payload is the one artifact under ``data/`` that naturally wants to hold
repo-defined objects: the bias/drift/noise **functions** it was fitted with, and
its ``MLEModelConfig``. Pickled directly, those name this package, so the file
only opens where the checkout still carries the name it had when it was written.

So they are not pickled directly. :func:`toStorable` replaces them with plain
data before the file is written -- the functions by their registry key, the
config by a dict -- and :func:`loadFit` puts them back on the way in. The file
itself names nothing outside numpy/pandas/scipy/stdlib, so

    pd.read_pickle(fit)          # works on any machine, any checkout name

gives you the payload with ``model_config`` as a plain dict. Use
:func:`loadFit` when you want the typed object back; it is a convenience, not a
requirement, which is the whole point.

Writing goes through :func:`saveFit`, which refuses to write a payload that
would not load elsewhere (see ``util.portablepickle.savePortable``).
"""
from __future__ import annotations

import dataclasses
import functools

from ...util.portablepickle import savePortable
from .bias import BIAS_FN_DICT
from .drift import DRIFT_FN_DICT
from .noise import NOISE_FN_DICT

#: Payload key -> the registry its value is looked up in.
_FN_REGISTRIES = {
    "biasFn": BIAS_FN_DICT,
    "driftFn": DRIFT_FN_DICT,
    "noiseFn": NOISE_FN_DICT,
}


def _sameCallable(a, b):
    """Whether two registry candidates are the same model component.

    Identity is not enough. Several registry entries are ``functools.partial``
    objects built once at import (``DRIFT_FN_DICT``'s ``DriftGain``
    family), and unpickling one from an existing fit constructs
    a *new* partial -- so ``is`` fails and ``==`` is false for partials, which
    do not define equality. Compare what actually identifies them: the wrapped
    function plus the bound arguments.
    """
    if a is b:
        return True
    if isinstance(a, functools.partial) and isinstance(b, functools.partial):
        return (a.func is b.func and a.args == b.args
                and a.keywords == b.keywords)
    return False


def _registryKey(fn, registry):
    """The name ``fn`` is registered under."""
    for name, candidate in registry.items():
        if _sameCallable(candidate, fn):
            return name
    raise KeyError(
        f"{fn!r} is not in the registry, so it cannot be stored by name. "
        f"Register it first, or the fit cannot be saved portably.")


def _asConfigDict(config):
    if config is None or isinstance(config, dict):
        return config
    return dataclasses.asdict(config)


def toStorable(payload):
    """A fit payload with its repo-defined objects replaced by plain data.

    Shallow-copies, so the caller's in-memory payload keeps its live objects.
    """
    out = dict(payload)

    names = out.get("fixed_params_names")
    vals = out.get("fixed_params_vals")
    if names is not None and vals is not None:
        vals = list(vals)
        for i, name in enumerate(names):
            registry = _FN_REGISTRIES.get(str(name))
            if registry is not None and callable(vals[i]):
                vals[i] = _registryKey(vals[i], registry)
        out["fixed_params_vals"] = vals

    if "model_config" in out:
        out["model_config"] = _asConfigDict(out["model_config"])
    return out


#: ``MLEModelConfig`` fields of the removed asymmetric-learning-rate
#: experiment. Every shipped fit stores them as False; see
#: ``_withoutRemovedConfigFields``.
_REMOVED_CONFIG_FIELDS = ("uses_asymmetric_alpha", "uses_asymmetric_beta")


def _withoutRemovedConfigFields(config):
    """A saved config dict minus the fields ``MLEModelConfig`` no longer has.

    They are dropped only while False. A fit that actually used separate
    unrewarded learning rates cannot be evaluated by this code, so it is
    refused rather than silently read as a symmetric one.
    """
    used = [name for name in _REMOVED_CONFIG_FIELDS if config.get(name)]
    if used:
        raise ValueError(
            f"This fit was run with {used}, the asymmetric-learning-rate "
            f"experiment, which has been removed from the model code.")
    return {k: v for k, v in config.items() if k not in _REMOVED_CONFIG_FIELDS}


def fromStorable(payload):
    """Inverse of :func:`toStorable` -- rebuild the typed objects.

    Tolerates a payload that never went through ``toStorable`` (an old file
    still holding live objects), so it is safe to apply unconditionally.
    """
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)

    names = out.get("fixed_params_names")
    vals = out.get("fixed_params_vals")
    if names is not None and vals is not None:
        vals = list(vals)
        for i, name in enumerate(names):
            registry = _FN_REGISTRIES.get(str(name))
            if registry is not None and isinstance(vals[i], str):
                vals[i] = registry[vals[i]]
        out["fixed_params_vals"] = vals

    config = out.get("model_config")
    if isinstance(config, dict):
        from .mle import MLEModelConfig
        out["model_config"] = MLEModelConfig(**_withoutRemovedConfigFields(config))

    # The 2026-08-31 portability rewrite turned every OptimizeResult into a
    # plain dict, and every reader asks for ``OptimRes.x`` / ``.fun``. An
    # OptimizeResult is itself a dict, so key access keeps working too.
    optim = out.get("OptimRes")
    if type(optim) is dict:
        from scipy.optimize import OptimizeResult
        out["OptimRes"] = OptimizeResult(optim)
    return out


def loadFit(path):
    """Read one saved fit, with its functions and config rebuilt.

    ``pd.read_pickle(path)`` also works and returns the same payload with
    ``model_config`` left as a dict -- use that when you only need the numbers
    and do not want to import this package.
    """
    import pandas as pd
    payload = pd.read_pickle(path)
    if not isinstance(payload, dict):
        return payload
    # ``{subject: payload}`` at the top level for every fit file on disk.
    return {subject: fromStorable(entry) for subject, entry in payload.items()}


def saveFit(payload, path):
    """Write a ``{subject: payload}`` fit, refusing anything unportable."""
    storable = {subject: toStorable(entry) if isinstance(entry, dict) else entry
                for subject, entry in payload.items()}
    return savePortable(storable, path)
