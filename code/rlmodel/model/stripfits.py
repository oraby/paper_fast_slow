"""Drop the mid-run optimisation trace from saved model fits.

``fit.simulateDDM`` records every candidate the differential-evolution search
evaluated -- one row per evaluation, holding the loss and the parameter vector
-- as ``candidate_losses_df`` on each subject's payload. It is a debugging aid
for watching a fit converge, and it is **write-only**: ``fit.py`` builds it,
``tests/test_candidate_trace.py`` checks its shape, and nothing reads it back.

It is also most of the file. On the largest fits it is ~83% of every subject's
payload (86,108 rows x 9 columns per subject, 20 subjects), so dropping it takes
``data/RLModel`` from gigabytes to a fraction of that without losing anything a
figure or an analysis uses.

The originals are copied to ``data/RLModel/model_full/`` first, and an existing
backup is never overwritten -- so running this twice cannot destroy the full
copy by backing up an already-stripped file.

Usage (from the project root)::

    uv run python -m code.rlmodel.model.stripfits              # dry run
    uv run python -m code.rlmodel.model.stripfits --yes        # write

What is deliberately *not* stripped
-----------------------------------
Two other keys look strippable and are not. Both were checked against their
readers, not assumed:

``fixed_params_vals["df"]``
    A byte-identical duplicate of ``subject_df`` (``DataFrame.equals`` is
    True), and a further ~184 MB across the set. But
    ``model_to_behavior.ipynb`` reads the frame back out of
    ``fixed_params_vals`` when building Figure 7D, so dropping it needs a
    matching restore on load. ``--report-only`` prints what it would save.

``mle_df``
    ~72% of what remains in each ``mle_*.pkl``, and it *looks* like a
    by-product of fitting. It is not: it carries the per-trial latents the
    model produced, and it is read from the saved payload by
    ``neural_correlate.py`` (Figures 7A-B), ``fast_slow_qr.py``,
    ``mle_notebooks/data.py`` and ``ddm_viewer.py``. **Do not strip it.**
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys

#: Key holding the per-evaluation optimiser trace.
TRACE_KEY = "candidate_losses_df"

#: Where the untouched originals are kept.
BACKUP_DIRNAME = "model_full"


def _isFitPayload(obj):
    """True for a ``{subject: payload}`` fit file.

    Identified by shape rather than by filename so a renamed or newly added fit
    is still picked up, and so ``df_2p_missing.pkl`` (a plain frame) and the
    ``metrics/`` caches are skipped.
    """
    if not isinstance(obj, dict) or not obj:
        return False
    entries = [v for v in obj.values() if isinstance(v, dict)]
    if len(entries) != len(obj):
        return False
    return any("OptimRes" in e or TRACE_KEY in e for e in entries)


def _pickledSize(obj):
    try:
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return 0


def inspectFile(path):
    """Sizes of the strippable parts, without writing anything."""
    import pandas as pd
    payload = pd.read_pickle(path)
    if not _isFitPayload(payload):
        return None
    trace = dupe = 0
    subjects = 0
    for entry in payload.values():
        subjects += 1
        if TRACE_KEY in entry:
            trace += _pickledSize(entry[TRACE_KEY])
        names = list(entry.get("fixed_params_names", []))
        vals = entry.get("fixed_params_vals")
        if vals is not None and "df" in names:
            candidate = vals[names.index("df")]
            subject_df = entry.get("subject_df")
            if (subject_df is not None and hasattr(candidate, "equals")
                    and candidate.equals(subject_df)):
                dupe += _pickledSize(candidate)
    return {"subjects": subjects, "trace_bytes": trace, "dupe_bytes": dupe,
            "file_bytes": os.path.getsize(path)}


def stripPayload(payload):
    """Return the payload without its optimiser traces, and how many were cut.

    Shallow-copies each entry, so the caller's in-memory payload is untouched.
    """
    out = {}
    removed = 0
    for subject, entry in payload.items():
        if isinstance(entry, dict) and TRACE_KEY in entry:
            entry = {k: v for k, v in entry.items() if k != TRACE_KEY}
            removed += 1
        out[subject] = entry
    return out, removed


def stripFile(path, backup_dir, dry_run=True):
    """Back up then rewrite one fit file. Returns a result dict."""
    import pandas as pd
    from .fitio import saveFit

    name = os.path.basename(path)
    before = os.path.getsize(path)
    payload = pd.read_pickle(path)
    if not _isFitPayload(payload):
        return {"name": name, "status": "skipped (not a fit)"}

    stripped, removed = stripPayload(payload)
    if removed == 0:
        return {"name": name, "status": "already stripped",
                "before": before, "after": before}

    if dry_run:
        projected = _pickledSize(stripped)
        return {"name": name, "status": "would strip", "subjects": removed,
                "before": before, "after": projected}

    os.makedirs(backup_dir, exist_ok=True)
    backup = os.path.join(backup_dir, name)
    if os.path.exists(backup):
        # Never overwrite a good backup with an already-processed file.
        if os.path.getsize(backup) < before:
            return {"name": name,
                    "status": f"REFUSED: {BACKUP_DIRNAME}/{name} is smaller "
                              f"than the file it would back up"}
    else:
        shutil.copy2(path, backup)

    saveFit(stripped, path)
    return {"name": name, "status": "stripped", "subjects": removed,
            "before": before, "after": os.path.getsize(path)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="data/RLModel")
    ap.add_argument("--yes", action="store_true",
                    help="actually rewrite (default is a dry run)")
    ap.add_argument("--report-only", action="store_true",
                    help="just show where the bytes are, change nothing")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.results_dir)
    backup_dir = os.path.join(root, BACKUP_DIRNAME)
    paths = sorted(os.path.join(root, f) for f in os.listdir(root)
                   if f.endswith(".pkl"))

    if args.report_only:
        tot_file = tot_trace = tot_dupe = 0
        print(f"{'file':<58}{'size':>9}{'trace':>9}{'dupe df':>9}")
        print("-" * 85)
        for p in paths:
            info = inspectFile(p)
            if info is None:
                continue
            tot_file += info["file_bytes"]
            tot_trace += info["trace_bytes"]
            tot_dupe += info["dupe_bytes"]
            print(f"{os.path.basename(p)[:56]:<58}"
                  f"{info['file_bytes']/1e6:>8.0f}M"
                  f"{info['trace_bytes']/1e6:>8.0f}M"
                  f"{info['dupe_bytes']/1e6:>8.0f}M")
        print("-" * 85)
        print(f"{'TOTAL':<58}{tot_file/1e6:>8.0f}M{tot_trace/1e6:>8.0f}M"
              f"{tot_dupe/1e6:>8.0f}M")
        print(f"\nstripping the trace alone: "
              f"{tot_file/1e6:.0f} MB -> ~{(tot_file-tot_trace)/1e6:.0f} MB")
        return 0

    results = [stripFile(p, backup_dir, dry_run=not args.yes) for p in paths]
    before = after = 0
    for r in results:
        if "before" not in r:
            if r["status"].startswith(("REFUSED", "skipped")):
                print(f"  {r['status']:<30} {r['name'][:56]}")
            continue
        before += r["before"]
        after += r["after"]
        print(f"  {r['status']:<18} {r['before']/1e6:>7.0f}M ->"
              f" {r['after']/1e6:>6.0f}M   {r['name'][:52]}")
    verb = "would save" if not args.yes else "saved"
    print(f"\n{before/1e6:.0f} MB -> {after/1e6:.0f} MB "
          f"({verb} {(before-after)/1e6:.0f} MB)")
    if not args.yes:
        print("dry run -- pass --yes to write "
              f"(originals are copied to {BACKUP_DIRNAME}/ first)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
