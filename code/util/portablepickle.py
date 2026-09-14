"""Dependency-free serialisation for the project's dataframes.

Why this exists
---------------
The pickles under ``data/`` embed references to classes that no longer resolve,
so most of them cannot be read in the ``uv`` environment:

* ``pandas.core.indexes.numeric`` -- ``Int64Index``/``Float64Index`` were
  removed in pandas 2.0, so every frame written by the old pandas-1.x kernel is
  unreadable under pandas 2.
* pandas ``string`` extension arrays pickled by a **conda-built** pandas cannot
  be unpickled by the **PyPI-built** pandas of the same version (they disagree
  on ``NDArrayBacked.__setstate__``'s state tuple).
* ``paper_fast_slow.code.twop.genrundata.RunData`` -- a repo-local namedtuple,
  so the pickle only loads when the repo is importable under exactly that name.
* ``caiman.behavior.convert.matreader.states.{States,StartEnd}`` -- from the
  author's earlier ``OneDrive/caiman`` project, which no environment of this
  repo can import; the notebooks already stub it out with a throwaway class, and
  nothing here reads the values.

A *portable payload* sidesteps all four. Frames are decomposed into plain
``numpy`` arrays plus builtins, so the resulting pickle names no class outside
``numpy`` and ``builtins``. It therefore loads on any machine, under any pandas
version, whatever the checkout is called.

This module must stay importable under **pandas 1.5.3 as well as 2.x**: the
legacy frames can only be read by the old pandas, so the conversion runs there.

Round trip::

    payload = toPortable(df)          # under the env that can read the original
    df2     = fromPortable(payload)   # anywhere

``fromPortable(toPortable(x))`` reproduces ``x`` for every structure used in
this project: DataFrame, Series, Index, namedtuple, dict, list, tuple, and the
``traces_sets`` dict-of-dict-of-ndarray held in object columns.
"""
from __future__ import annotations

import functools
import importlib
import inspect
import os as _os
import pickle
import sys as _sys

import numpy as np
import pandas as pd


def _bindStdlibCode():
    """Make sure a bare ``import code`` reaches the standard library.

    Some of the modules this loader has to import (``twop.genrundata`` ->
    ``testscomb`` -> ``IPython``) do ``import code`` transitively. When the
    process was started with the repo root on ``sys.path`` -- which happens for
    any ``python -c`` or heredoc run from the repo root -- that resolves to this
    project's own ``code`` package instead, and IPython dies with
    ``module 'code' has no attribute 'InteractiveConsole'``. See the import
    layout note in ``CLAUDE.md``.

    Binding the stdlib module under the bare name is safe here: the project's
    own modules are always reached as ``<checkout>.code.*``, never as ``code``.
    """
    mod = _sys.modules.get("code")
    if mod is not None and hasattr(mod, "InteractiveConsole"):
        return
    import sysconfig
    import importlib.util
    stdlib_code = _os.path.join(sysconfig.get_paths()["stdlib"], "code.py")
    if not _os.path.isfile(stdlib_code):
        return
    spec = importlib.util.spec_from_file_location("code", stdlib_code)
    if spec is None or spec.loader is None:
        return
    real = importlib.util.module_from_spec(spec)
    _sys.modules["code"] = real
    try:
        spec.loader.exec_module(real)
    except Exception:
        _sys.modules.pop("code", None)


def _importSuffix(module, name):
    """Import ``name`` from the longest suffix of ``module`` resolving here.

    A pickle records the absolute import path in effect on the machine that
    wrote it -- ``paper_fast_slow.code.rlmodel.model.bias``,
    ``code.rlmodel.model.bias``, ``someones_fork.code.rlmodel.model.bias``.
    Walking the path from the right and importing the longest suffix that
    resolves inside *this* package makes the lookup independent of what the
    checkout is called or where it sits. Returns ``None`` when nothing matches.
    """
    _bindStdlibCode()
    parts = module.split(".")
    candidates = [".".join(parts[start:]) for start in range(len(parts))]

    for tail in candidates:
        # Preferred: a relative import, which needs no sys.path surgery.
        if __package__:
            try:
                mod = importlib.import_module(f"..{tail}", __package__)
            except Exception:
                mod = None
            if mod is not None:
                attr = getattr(mod, name, None)
                if attr is not None:
                    return attr

    # Fallback for a bare-script run (``python code/util/migratepickles.py``),
    # where there is no package to anchor a relative import: locate this file's
    # own repo and import through it by absolute name. The checkout may be
    # called anything -- its directory name is read, never assumed.
    here = _os.path.dirname(_os.path.abspath(__file__))       # <repo>/code/util
    code_dir = _os.path.dirname(here)                          # <repo>/code
    repo_dir = _os.path.dirname(code_dir)                      # <repo>
    parent, repo_name = _os.path.split(repo_dir)
    if parent not in _sys.path:
        _sys.path.insert(0, parent)
    for tail in candidates:
        try:
            mod = importlib.import_module(f"{repo_name}.code.{tail}")
        except Exception:
            continue
        attr = getattr(mod, name, None)
        if attr is not None:
            return attr
    return None

#: Payload marker. Presence of this key identifies a wrapped object.
TAG = "__pfs_kind__"

#: Columns whose values are instances of a class we deliberately dropped
#: (see ``DROPPED_CLASS_NAMES``) are removed and recorded here.
DROPPED_COLUMNS = "__pfs_dropped_columns__"

#: Class names that no longer exist anywhere and are stubbed at load time, so
#: their values carry no information. Dropping them is lossless in practice.
DROPPED_CLASS_NAMES = frozenset({"States", "StartEnd", "Dummy", "_PfsStub"})


#: Types whose pickled form names only stable, universally-present modules, so
#: they can be written straight through.
_PASSTHROUGH_ROOTS = ("numpy", "builtins", "datetime", "decimal", "uuid")


def _isPassthrough(obj) -> bool:
    if obj is None or isinstance(obj, (str, bytes, bool, int, float, complex)):
        return True
    return type(obj).__module__.split(".")[0] in _PASSTHROUGH_ROOTS


def _isDroppedValue(value) -> bool:
    return type(value).__name__ in DROPPED_CLASS_NAMES


def _columnIsDropped(series: pd.Series) -> bool:
    """True when a column's non-null values are stubbed foreign objects.

    Only object columns can hold them, and checking a short head is enough --
    these columns are homogeneous by construction (one Bpod states object per
    trial).
    """
    if series.dtype != object:
        return False
    head = series.dropna().head(20)
    return len(head) > 0 and all(_isDroppedValue(v) for v in head)


def _indexValues(index: pd.Index) -> np.ndarray:
    """Index values as a plain array.

    Object-dtype indexes have to go through :func:`toPortable` like any other
    object column -- a ``pd.cut`` groupby key leaves ``pd.Interval`` objects
    here, and they would otherwise be written straight through and put pandas
    back into the payload.
    """
    values = np.asarray(index)
    return toPortable(values) if values.dtype == object else values


def _indexPayload(index: pd.Index) -> dict:
    if isinstance(index, pd.MultiIndex):
        return {TAG: "multiindex",
                "levels": [_indexValues(index.get_level_values(i))
                           for i in range(index.nlevels)],
                "names": list(index.names)}
    return {TAG: "index",
            "values": _indexValues(index),
            "name": index.name,
            "dtype": str(index.dtype)}


def _indexFromPayload(payload: dict) -> pd.Index:
    if payload[TAG] == "multiindex":
        return pd.MultiIndex.from_arrays(
            [fromPortable(lvl) for lvl in payload["levels"]],
            names=payload["names"])
    values = fromPortable(payload["values"])
    dtype = payload.get("dtype", "")
    # Rebuild under the *running* pandas so no removed index class is named.
    # Restore the recorded dtype only when numpy itself recognises it -- a
    # prefix test would misfire, e.g. "interval[float64, left]" starts with
    # "int" but is a pandas extension dtype, not a numpy integer.
    if dtype and getattr(values, "dtype", None) == object:
        try:
            values = values.astype(np.dtype(dtype))
        except (TypeError, ValueError):
            pass
    return pd.Index(values, name=payload["name"])


def _seriesValues(series: pd.Series) -> np.ndarray:
    """Column values as a plain numpy array.

    Extension dtypes (``string``, ``Int64``, ``boolean``, ``Float64``,
    ``category``) are the ones that do not survive a cross-build unpickle, so
    they are flattened to their numpy equivalent. Nullable integers become
    ``float64`` when they hold NA, matching what ``to_numpy`` does.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        return np.asarray(series.astype(object))
    if pd.api.types.is_extension_array_dtype(series.dtype):
        if series.isna().any():
            return np.asarray(series.astype(object).where(series.notna(),
                                                          other=None))
        return np.asarray(series.to_numpy())
    return np.asarray(series.to_numpy())


def _isMatStruct(obj) -> bool:
    """``scipy.io.matlab._mio5_params.mat_struct`` -- the raw Bpod session
    metadata columns (``BlocksInfo``, ``drawParams``, ``soundParams``,
    ``Trials``) hold these.

    Detected by shape rather than by import: the class lives behind a *private*
    scipy path that has already been renamed once (``mio5_params`` ->
    ``_mio5_params``), so naming it in a payload would reintroduce exactly the
    fragility this module exists to remove.
    """
    return (type(obj).__name__ == "mat_struct"
            and hasattr(obj, "_fieldnames"))


def _columnPayload(series: pd.Series):
    """One column's values, recursing when they end up as objects.

    Extension dtypes are flattened first, so an ``interval[...]`` or
    ``category`` column becomes an object array of pandas scalars -- which then
    still has to go through :func:`toPortable` to strip the pandas types out.
    """
    values = _seriesValues(series)
    return toPortable(values) if values.dtype == object else values


def toPortable(obj):
    """Recursively convert ``obj`` into a payload of numpy arrays + builtins."""
    if _isMatStruct(obj):
        return {TAG: "matstruct",
                "fields": {name: toPortable(getattr(obj, name, None))
                           for name in obj._fieldnames}}

    if isinstance(obj, pd.DataFrame):
        dropped = [c for c in obj.columns if _columnIsDropped(obj[c])]
        keep = [c for c in obj.columns if c not in dropped]
        return {TAG: "dataframe",
                "columns": [str(c) for c in keep],
                "data": [_columnPayload(obj[c]) for c in keep],
                "dtypes": [str(obj[c].dtype) for c in keep],
                "index": _indexPayload(obj.index),
                DROPPED_COLUMNS: [str(c) for c in dropped]}

    if isinstance(obj, pd.Series):
        return {TAG: "series",
                "values": _columnPayload(obj),
                "name": obj.name,
                "dtype": str(obj.dtype),
                "index": _indexPayload(obj.index)}

    if isinstance(obj, pd.Index):
        return _indexPayload(obj)

    # namedtuple -- recorded by field name so the class need not exist on load.
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return {TAG: "namedtuple",
                "module": type(obj).__module__,
                "type_name": type(obj).__name__,
                "fields": list(obj._fields),
                "values": [toPortable(v) for v in obj]}

    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            flat = [toPortable(v) for v in obj.ravel().tolist()]
            out = np.empty(len(flat), dtype=object)
            out[:] = flat
            return out.reshape(obj.shape)
        return obj

    if isinstance(obj, dict):
        return {"__pfs_dict__": [(toPortable(k), toPortable(v))
                                 for k, v in obj.items()]}

    if isinstance(obj, list):
        return [toPortable(v) for v in obj]

    if isinstance(obj, tuple):
        return {TAG: "tuple", "values": [toPortable(v) for v in obj]}

    # Callables. The RL-model fit payloads store the bias/drift/noise functions
    # as objects, which is the last hard repo dependency in the data: pickle
    # resolves them through ``find_class``, so the checkout has to be importable
    # under the exact name it had when the fit was written. Recording
    # module+qualname instead moves the lookup into :func:`_importSuffix`,
    # which is prefix-agnostic -- so the payload itself imports nothing but
    # numpy, and the resolution tolerates a renamed or relocated checkout.
    if isinstance(obj, functools.partial):
        return {TAG: "partial",
                "func": toPortable(obj.func),
                "args": [toPortable(a) for a in obj.args],
                "keywords": {k: toPortable(v)
                             for k, v in (obj.keywords or {}).items()},
                # ``partialWithNames`` sets these so introspection still works.
                "attrs": {k: v for k, v in vars(obj).items()
                          if isinstance(v, (str, int, float, bool, type(None)))}}

    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return {TAG: "callable",
                "module": obj.__module__,
                "qualname": getattr(obj, "__qualname__", obj.__name__)}

    if _isDroppedValue(obj):
        return None

    # ``pd.Interval`` shows up in object columns produced by ``pd.cut``. It is
    # public pandas API, but recording it structurally keeps the payload free
    # of *any* pandas import.
    if isinstance(obj, pd.Interval):
        return {TAG: "interval", "left": toPortable(obj.left),
                "right": toPortable(obj.right), "closed": obj.closed}

    # Catch-all for any remaining instance of a non-stdlib class -- the RL-model
    # fits embed ``MLEModelConfig`` (a dataclass of scalars), and this keeps a
    # future one from silently reintroducing a hard import. Recorded the same
    # way as callables: by name, resolved through the prefix-agnostic lookup.
    if not _isPassthrough(obj):
        cls = type(obj)
        state = getattr(obj, "__dict__", None)
        if state is None:
            slots = getattr(cls, "__slots__", ())
            state = {s: getattr(obj, s) for s in slots if hasattr(obj, s)}
        return {TAG: "object",
                "module": cls.__module__,
                "qualname": cls.__qualname__,
                "state": toPortable(dict(state))}

    return obj


def fromPortable(payload):
    """Inverse of :func:`toPortable`."""
    if isinstance(payload, np.ndarray):
        if payload.dtype == object:
            flat = [fromPortable(v) for v in payload.ravel().tolist()]
            out = np.empty(len(flat), dtype=object)
            out[:] = flat
            return out.reshape(payload.shape)
        return payload

    if isinstance(payload, list):
        return [fromPortable(v) for v in payload]

    if not isinstance(payload, dict):
        return payload

    if "__pfs_dict__" in payload:
        return {fromPortable(k): fromPortable(v)
                for k, v in payload["__pfs_dict__"]}

    kind = payload.get(TAG)
    if kind is None:
        return {k: fromPortable(v) for k, v in payload.items()}

    if kind == "dataframe":
        data = {}
        for name, values, dtype in zip(payload["columns"], payload["data"],
                                       payload["dtypes"]):
            col = fromPortable(values)
            data[name] = col
        df = pd.DataFrame(data, columns=payload["columns"],
                          index=_indexFromPayload(payload["index"]))
        return df

    if kind == "series":
        return pd.Series(fromPortable(payload["values"]),
                         index=_indexFromPayload(payload["index"]),
                         name=payload["name"])

    if kind in ("index", "multiindex"):
        return _indexFromPayload(payload)

    if kind == "namedtuple":
        fields = payload["fields"]
        values = [fromPortable(v) for v in payload["values"]]
        # Rebuild the real class when it is importable, so attribute access
        # (``run_data.df_src``) keeps working; fall back to a field-keyed dict
        # when it is not, which keeps the payload readable without the repo.
        cls = (_importSuffix(payload["module"], payload["type_name"])
               if payload.get("module") else None)
        if cls is not None and list(getattr(cls, "_fields", ())) == fields:
            return cls(*values)
        return dict(zip(fields, values))

    if kind == "tuple":
        return tuple(fromPortable(v) for v in payload["values"])

    if kind == "callable":
        module, qualname = payload["module"], payload["qualname"]
        # Nested qualnames ("Outer.method") resolve attribute by attribute.
        head, _, rest = qualname.partition(".")
        resolved = _importSuffix(module, head)
        if resolved is None:
            raise ImportError(
                f"cannot resolve {module}.{qualname} from this checkout; the "
                f"payload records it by name, so the module must exist "
                f"somewhere under this package")
        for part in filter(None, rest.split(".")):
            resolved = getattr(resolved, part)
        return resolved

    if kind == "partial":
        return functools.partial(
            fromPortable(payload["func"]),
            *[fromPortable(a) for a in payload["args"]],
            **{k: fromPortable(v) for k, v in payload["keywords"].items()})

    if kind == "interval":
        return pd.Interval(fromPortable(payload["left"]),
                           fromPortable(payload["right"]),
                           closed=payload["closed"])

    if kind == "object":
        module, qualname = payload["module"], payload["qualname"]
        head, _, rest = qualname.partition(".")
        cls = _importSuffix(module, head)
        if cls is None:
            raise ImportError(
                f"cannot resolve {module}.{qualname} from this checkout")
        for part in filter(None, rest.split(".")):
            cls = getattr(cls, part)
        obj = cls.__new__(cls)
        state = fromPortable(payload["state"])
        try:
            obj.__dict__.update(state)
        except AttributeError:  # __slots__ classes
            for k, v in state.items():
                setattr(obj, k, v)
        return obj

    if kind == "matstruct":
        # A plain dict: nothing in this project reads these MATLAB structs by
        # attribute, and a dict names no class at all.
        return {k: fromPortable(v) for k, v in payload["fields"].items()}

    raise ValueError(f"unknown payload kind: {kind!r}")


# ---------------------------------------------------------------------------
# Reading the legacy artifacts
# ---------------------------------------------------------------------------
#
# Replaces the three ad-hoc ``PickleMissingLoader`` / ``MyUnpickler2`` copies
# that live inline in TwoPTraces.ipynb, 2pAnalysis.ipynb and model_viewer.ipynb.


class _PfsStub:
    """Placeholder for a class this repo cannot import.

    Values of these types were already being discarded by the notebooks' inline
    stubs, so nothing downstream reads them; ``toPortable`` drops the columns
    holding them.
    """
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


def _pandasCompatMap():
    """pandas' own old-location table, as used by ``pandas.read_pickle``.

    It is what remaps ``pandas.core.indexes.numeric.Int64Index`` (removed in
    pandas 2.0) onto ``pandas.core.indexes.base.Index``, and it covers a long
    tail of other pandas-1.x moves besides. ``pickle.Unpickler.find_class``
    does **not** consult it -- only pandas' own unpickler does -- so a plain
    subclass silently loses the compatibility that ``pd.read_pickle`` has.

    Returns an empty map rather than raising if pandas ever drops it (it is
    private, and slated to go with the pandas-1.x pickles it supports).
    """
    try:
        from pandas.compat import pickle_compat
        return dict(getattr(pickle_compat, "_class_locations_map", {}))
    except Exception:
        return {}


_PANDAS_COMPAT = _pandasCompatMap()


class LegacyUnpickler(pickle.Unpickler):
    """Loads the project's older pickles despite their unresolvable imports.

    Handles, in order:

    1. an explicit ``class_map`` from the caller;
    2. pandas' own old-location table (see :func:`_pandasCompatMap`);
    3. ``caiman.*`` and ``__main__.*`` -- stubbed, since the classes exist
       nowhere and their values were already being discarded;
    4. **repo-local modules, whatever prefix they were pickled under.**

    (4) is what makes the loader portable between checkouts. A pickle written
    on one machine records the absolute import path that was in effect there --
    ``paper_fast_slow.code.twop.genrundata``, ``code.twop.genrundata``,
    ``someones_fork.code.twop.genrundata``. Rather than matching a list of
    known prefixes, the loader walks the recorded path from the right and
    imports the longest suffix that resolves inside *this* package. So the
    checkout can be renamed, nested differently, or vendored, and the same
    pickle still loads.
    """

    def __init__(self, fp, *, record=None, class_map=None):
        super().__init__(fp)
        self.record = record if record is not None else set()
        #: ``"module.Name" -> class``. Needed only when the loader runs outside
        #: its package (no ``__package__``, so no relative import to anchor
        #: the suffix search).
        self.class_map = class_map or {}

    def find_class(self, module, name):
        key = f"{module}.{name}"

        remapped = _PANDAS_COMPAT.get((module, name))
        if remapped is not None:
            module, name = remapped
            return super().find_class(module, name)

        if module.startswith("caiman") or module == "__main__":
            self.record.add(key)
            return _PfsStub

        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            pass

        # Prefer the genuine class from this checkout -- a substitute supplied
        # via ``class_map`` would record its own module, which would then be
        # wrong on the way back out. ``class_map`` is the last resort, for when
        # the repo is not importable at all.
        resolved = _importSuffix(module, name)
        if resolved is not None:
            self.record.add(key)
            return resolved
        if key in self.class_map:
            self.record.add(key)
            return self.class_map[key]
        raise ImportError(f"cannot resolve {key}")


def legacyLoad(path, *, record=None, class_map=None):
    """Read one legacy pickle, tolerating its dangling class references."""
    with open(path, "rb") as fp:
        return LegacyUnpickler(fp, record=record, class_map=class_map).load()


# ---------------------------------------------------------------------------
# Writing a pickle that will load anywhere
# ---------------------------------------------------------------------------
#
# The rule this enforces: a file under ``data/`` must name no class this repo
# defines, so a plain ``pd.read_pickle`` works on any machine, under any
# checkout name, without this package installed. See docs/data-portability.md.

#: Module roots a reader is expected to have. A pickle naming anything else
#: cannot be opened by someone who only has the data.
PORTABLE_ROOTS = frozenset({
    "builtins", "collections", "datetime", "decimal", "uuid",
    "copyreg", "_codecs", "numpy", "pandas", "scipy",
})

#: How many elements of one large container to inspect. Columns and object
#: arrays in these frames are homogeneous by construction (one dict of traces
#: per trial, one label per trial), so a bounded sample finds a foreign class
#: if the container holds any. Raise it for a paranoid check.
WALK_SAMPLE = 200


class NotPortableError(TypeError):
    """A payload names a class that would not resolve on another machine."""


def _moduleRootOf(obj):
    """The module root that pickling ``obj`` would name, or ``None``."""
    if isinstance(obj, (str, bytes, bool, int, float, complex, type(None))):
        return None
    target = obj if inspect.isclass(obj) or inspect.isfunction(obj) else type(obj)
    module = getattr(target, "__module__", None)
    return module.split(".")[0] if module else None


def findUnportable(obj, *, sample=WALK_SAMPLE):
    """Every ``module.Qualname`` in ``obj`` that a bare reader could not import.

    Walks the object graph the way pickle would. Large containers are sampled
    (see :data:`WALK_SAMPLE`); cycles are handled by identity.
    """
    bad = {}
    seen = set()

    def visit(node, path):
        if id(node) in seen:
            return
        seen.add(id(node))

        root = _moduleRootOf(node)
        if root is not None and root not in PORTABLE_ROOTS:
            target = (node if inspect.isclass(node) or inspect.isfunction(node)
                      else type(node))
            name = (f"{getattr(target, '__module__', '?')}."
                    f"{getattr(target, '__qualname__', getattr(target, '__name__', '?'))}")
            bad.setdefault(name, path)
            return          # no point descending into a rejected object

        if isinstance(node, dict):
            for i, (k, v) in enumerate(node.items()):
                if i >= sample:
                    break
                visit(k, f"{path}.<key>")
                visit(v, f"{path}[{k!r}]")
        elif isinstance(node, (list, tuple, set, frozenset)):
            for i, v in enumerate(node):
                if i >= sample:
                    break
                visit(v, f"{path}[{i}]")
        elif isinstance(node, np.ndarray):
            if node.dtype == object:
                for i, v in enumerate(node.ravel()[:sample]):
                    visit(v, f"{path}[{i}]")
        elif isinstance(node, pd.DataFrame):
            visit(node.index, f"{path}.index")
            for col in node.columns:
                if node[col].dtype == object:
                    visit(node[col].to_numpy(), f"{path}[{col!r}]")
        elif isinstance(node, pd.Series):
            visit(node.index, f"{path}.index")
            if node.dtype == object:
                visit(node.to_numpy(), path)
        elif isinstance(node, pd.Index):
            if node.dtype == object:
                visit(np.asarray(node), path)
        elif hasattr(node, "__dict__"):
            for k, v in list(vars(node).items())[:sample]:
                visit(v, f"{path}.{k}")

    visit(obj, "<root>")
    return bad


def assertPortable(obj, *, sample=WALK_SAMPLE):
    """Raise :class:`NotPortableError` if ``obj`` would pickle unportably."""
    bad = findUnportable(obj, sample=sample)
    if not bad:
        return
    lines = "\n".join(f"    {name}   at {where}" for name, where in
                      sorted(bad.items()))
    raise NotPortableError(
        "this payload names classes a bare reader could not import:\n"
        f"{lines}\n"
        "  Store plain data instead (a name string, a dict) and rebuild it at "
        "the load site. See docs/data-portability.md.")


def savePortable(obj, path, *, sample=WALK_SAMPLE, protocol=None):
    """Pickle ``obj`` to ``path`` only if it will load on any machine.

    The blessed way to write anything under ``data/``. Checks first and refuses
    rather than writing a file that would need this package to be read.
    """
    assertPortable(obj, sample=sample)
    protocol = pickle.HIGHEST_PROTOCOL if protocol is None else protocol
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fp:
        pickle.dump(obj, fp, protocol=protocol)
    _os.replace(tmp, path)
    return path
