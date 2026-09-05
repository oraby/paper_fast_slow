"""One-off migration of the legacy pickles under ``data/`` to portable payloads.

Most artifacts under ``data/`` can only be read by one specific interpreter --
see :mod:`portablepickle` for why. This script reads them **in that
interpreter** and rewrites them through :func:`portablepickle.toPortable`, so
the result loads anywhere.

Originals are never modified. Output goes to a mirror tree, and the swap is a
separate, deliberate step.

Which interpreter reads what
----------------------------
Almost everything can be read right here. ``LegacyUnpickler`` applies pandas'
own old-location table, which is what rescues the pandas-1.x ``Int64Index``
frames -- 123 of the 139 artifacts needing migration. Only one group resists:

=============================== ================================= ============
Artifacts                       Defect                            Read with
=============================== ================================= ============
``data/RLModel/*.pkl`` (16)     conda-vs-PyPI ``string`` dtype    conda py312
everything else (123)           pandas-1.x / caiman / RunData     ``uv`` here
=============================== ================================= ============

Migrating the first group anyway is the point: the pandas shim is a *private*
module ("Support pre-0.12 series pickle compatibility") that exists to carry
pandas-1.x pickles and will not outlive them, and the repo already pins
``pandas<3.0.0``. After migration nothing depends on it.

Usage::

    # the 123 readable here
    uv run python code/util/migratepickles.py \\
        --repo <REPO> --out <REPO>/data_portable --group legacy-index

    # the 16 that need the conda build; run from a directory that is NOT the
    # repo root, so the local `code` package cannot shadow the stdlib module
    cd code/util
    conda run -n py312 --no-capture-output python -m migratepickles \\
        --repo <REPO> --out <REPO>/data_portable --group string-dtype

    # re-encode with the current numpy, then check every payload loads
    uv run python code/util/migratepickles.py \\
        --repo <REPO> --out <REPO>/data_portable --renormalize
    uv run python code/util/migratepickles.py \\
        --repo <REPO> --out <REPO>/data_portable --verify
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import sys
import time
import traceback

# Importable both as ``python -m migratepickles`` (standalone, for the conda
# envs) and as part of the package.
try:
    from .portablepickle import (legacyLoad, toPortable, fromPortable,
                                 PORTABLE_ROOTS)
except ImportError:  # pragma: no cover - standalone invocation
    from portablepickle import (legacyLoad, toPortable, fromPortable,  # type: ignore
                                PORTABLE_ROOTS)


SUFFIX = ".portable.pkl"

# The only repo-local class any of these pickles embeds. Declared here rather
# than imported so the migration can run standalone under the old conda
# interpreters, where importing the package would drag in its whole dependency
# tree (and put the local ``code`` package on the path, shadowing the stdlib
# module of that name). Fields must match ``twop.genrundata.RunData``.
_RunData = collections.namedtuple(
    "RunData", ["df_src", "df_reduc_stats", "df_reduc_roc", "shortlong_df",
                "shortlong_quantiled_df"])

CLASS_MAP = {
    "paper_fast_slow.code.twop.genrundata.RunData": _RunData,
    "code.twop.genrundata.RunData": _RunData,
}


def _relPaths(repo, group):
    """Every ``data/**/*.pkl`` that the given group should handle."""
    root = os.path.join(repo, "data")
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if not fn.endswith(".pkl") or fn.endswith(SUFFIX):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, repo).replace(os.sep, "/")
            is_model = "/RLModel/" in rel
            if group == "string-dtype" and not is_model:
                continue
            if group == "legacy-index" and is_model:
                continue
            yield rel


def _describe(obj):
    """Small structural fingerprint, used to compare before and after."""
    import pandas as pd
    if isinstance(obj, pd.DataFrame):
        return {"kind": "dataframe", "shape": list(obj.shape),
                "columns": [str(c) for c in obj.columns]}
    if isinstance(obj, pd.Series):
        return {"kind": "series", "len": int(len(obj))}
    if isinstance(obj, dict):
        return {"kind": "dict", "len": len(obj),
                "keys": [str(k) for k in list(obj)[:12]]}
    if isinstance(obj, (list, tuple)):
        return {"kind": type(obj).__name__, "len": len(obj)}
    return {"kind": type(obj).__name__}


def migrate(repo, out_root, group, limit=None, only=None):
    manifest_path = os.path.join(out_root, f"manifest_{group}.json")
    os.makedirs(out_root, exist_ok=True)
    entries = []
    rels = list(_relPaths(repo, group))
    if only:
        rels = [r for r in rels if any(o in r for o in only)]
    if limit:
        rels = rels[:limit]

    for i, rel in enumerate(rels, 1):
        src = os.path.join(repo, rel)
        dst = os.path.join(out_root, rel[len("data/"):]) + SUFFIX
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        entry = {"source": rel, "output": os.path.relpath(dst, out_root),
                 "src_bytes": os.path.getsize(src)}
        t0 = time.time()
        try:
            record = set()
            obj = legacyLoad(src, record=record, class_map=CLASS_MAP)
            entry["foreign_refs"] = sorted(record)
            entry["before"] = _describe(obj)
            payload = toPortable(obj)
            del obj
            with open(dst, "wb") as fp:
                pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
            del payload
            entry["out_bytes"] = os.path.getsize(dst)
            entry["status"] = "ok"
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc()[-800:]
        entry["seconds"] = round(time.time() - t0, 1)
        entries.append(entry)
        mark = "ok " if entry["status"] == "ok" else "FAIL"
        print(f"[{i}/{len(rels)}] {mark} {entry['seconds']:6.1f}s  {rel}",
              flush=True)
        if entry["status"] == "failed":
            print(f"        {entry['error']}", flush=True)

    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(entries, fp, indent=2)
    ok = sum(1 for e in entries if e["status"] == "ok")
    print(f"\n{ok}/{len(entries)} migrated -> {manifest_path}")
    return entries


def verify(out_root):
    """Load every migrated payload in the *current* interpreter."""
    manifests = [f for f in os.listdir(out_root) if f.startswith("manifest_")]
    total = bad = 0
    for name in manifests:
        with open(os.path.join(out_root, name), encoding="utf-8") as fp:
            entries = json.load(fp)
        for entry in entries:
            if entry["status"] != "ok":
                continue
            total += 1
            path = os.path.join(out_root, entry["output"])
            try:
                with open(path, "rb") as fp:
                    obj = fromPortable(pickle.load(fp))
                after = _describe(obj)
                before = entry["before"]
                del obj
                if after.get("shape") and after["shape"] != before.get("shape"):
                    # A dropped stub column legitimately narrows the frame.
                    dropped = len(before.get("columns", [])) - \
                              len(after.get("columns", []))
                    if not (dropped > 0
                            and after["shape"][0] == before["shape"][0]):
                        raise AssertionError(
                            f"shape {before.get('shape')} -> {after['shape']}")
                print(f"ok    {entry['output']}  {after}")
            except Exception as exc:
                bad += 1
                print(f"FAIL  {entry['output']}  {type(exc).__name__}: {exc}")
    print(f"\n{total - bad}/{total} payloads load cleanly here "
          f"(pandas {__import__('pandas').__version__})")
    return bad == 0


def renormalize(out_root):
    """Second pass, run under the *target* interpreter.

    Stage 1 runs under the old conda envs, so the arrays it writes carry
    ``numpy.core.*`` reduction functions -- numpy 2 resolves those only through
    a deprecation shim. Loading and re-dumping here re-encodes them with the
    current numpy, leaving a payload that names nothing deprecated.
    """
    count = 0
    for dirpath, _dirnames, filenames in os.walk(out_root):
        for fn in sorted(filenames):
            if not fn.endswith(SUFFIX):
                continue
            path = os.path.join(dirpath, fn)
            with open(path, "rb") as fp:
                payload = pickle.load(fp)
            tmp = path + ".tmp"
            with open(tmp, "wb") as fp:
                pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
            del payload
            os.replace(tmp, path)
            count += 1
            print(f"renormalized  {os.path.relpath(path, out_root)}")
    print(f"\n{count} payloads re-encoded with "
          f"numpy {__import__('numpy').__version__}")
    return count


def install(repo, out_root, dry_run=True):
    """Rebuild each payload and write it back over ``data/`` as a normal pickle.

    A payload is **not** a drop-in replacement for the file it came from:
    ``pd.read_pickle`` on one returns the payload dict, not the frame, so simply
    renaming ``*.portable.pkl`` onto ``data/`` would break all 73 load sites in
    the repo. Reconstructing first keeps every caller working untouched while
    still discarding what made the originals unportable -- the removed
    ``Int64Index``, the conda-built ``string`` arrays, the ``caiman`` column and
    the private ``mat_struct`` path are all gone once the object is rewritten by
    the current pandas.

    The RL-model fits are the exception: they legitimately hold references to
    the model's own bias/drift/noise functions, so their rewritten pickles name
    this package. Read those through :func:`portablepickle.legacyLoad`, whose
    lookup is prefix-agnostic, rather than a bare ``pickle.load``.
    """
    written = failed = 0
    for dirpath, _dirnames, filenames in os.walk(out_root):
        for fn in sorted(filenames):
            if not fn.endswith(SUFFIX):
                continue
            payload_path = os.path.join(dirpath, fn)
            rel = os.path.relpath(payload_path, out_root)[: -len(SUFFIX)]
            target = os.path.join(repo, "data", rel)
            backup = os.path.join(repo, "data_bak", rel)
            if not os.path.exists(backup):
                print(f"REFUSING {rel}: no backup at data_bak/", flush=True)
                failed += 1
                continue
            if dry_run:
                print(f"would write  {rel}", flush=True)
                written += 1
                continue
            try:
                with open(payload_path, "rb") as fp:
                    obj = fromPortable(pickle.load(fp))
                tmp = target + ".tmp"
                with open(tmp, "wb") as fp:
                    pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)
                del obj
                os.replace(tmp, target)
                written += 1
                print(f"wrote  {rel}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"FAILED {rel}: {type(exc).__name__}: {exc}", flush=True)
    verb = "would write" if dry_run else "written"
    print(f"\n{written} {verb}, {failed} failed")
    return failed == 0


def audit(repo, *, verbose=False):
    """Check every ``data/**/*.pkl`` opens with a bare ``pickle.load``.

    The counterpart of the write-time guard in ``portablepickle.savePortable``:
    that stops *this* repo writing an unportable file, while this catches one
    that arrived some other way -- unzipped from the download site, copied from
    a colleague, produced by a script that bypassed the guard.

    Deliberately not a pytest test. It reads every byte of ~8 GB, so it belongs
    in a release check rather than in a suite that should stay fast. The guard's
    own unit tests are what run on every commit.
    """
    root = os.path.join(repo, "data")
    checked = bad = 0
    failures = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if not fn.endswith(".pkl") or fn.endswith(SUFFIX):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, repo).replace(os.sep, "/")
            checked += 1
            names = set()

            class _Recorder(pickle.Unpickler):
                def find_class(self, module, name):
                    names.add(module.split(".")[0])
                    return super().find_class(module, name)

            try:
                with open(path, "rb") as fp:
                    obj = _Recorder(fp).load()
                del obj
            except Exception as exc:
                bad += 1
                failures.append((rel, f"{type(exc).__name__}: {exc}"))
                print(f"FAIL  {rel}\n        {type(exc).__name__}: {exc}",
                      flush=True)
                continue
            foreign = names - PORTABLE_ROOTS
            if foreign:
                bad += 1
                failures.append((rel, f"imports {sorted(foreign)}"))
                print(f"FAIL  {rel}\n        imports {sorted(foreign)}",
                      flush=True)
            elif verbose:
                print(f"ok    {rel}", flush=True)

    print(f"\n{checked - bad}/{checked} pickles load with a bare pickle.load "
          f"and import only {sorted(PORTABLE_ROOTS)}")
    return not failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", choices=("legacy-index", "string-dtype", "all"),
                    default="all")
    ap.add_argument("--only", nargs="*",
                    help="substring filter on the source path")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--renormalize", action="store_true",
                    help="re-encode payloads with the current numpy")
    ap.add_argument("--install", action="store_true",
                    help="rebuild each payload and write it back over data/")
    ap.add_argument("--audit", action="store_true",
                    help="check every data/**/*.pkl opens with a bare "
                         "pickle.load and imports nothing repo-local")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--yes", action="store_true",
                    help="with --install, actually write (default is a dry run)")
    args = ap.parse_args(argv)

    if args.audit:
        return 0 if audit(args.repo, verbose=args.verbose) else 1
    if args.install:
        return 0 if install(args.repo, args.out, dry_run=not args.yes) else 1
    if args.renormalize:
        renormalize(args.out)
        return 0
    if args.verify:
        return 0 if verify(args.out) else 1
    migrate(args.repo, args.out, args.group, limit=args.limit, only=args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
