"""Figure 7D's surface: z-scored sampling time over Q-relative x reward rate.

Each fitted subject is resampled into pseudo-sessions with fresh stimulus
strengths, simulated under its own fitted parameters, and binned into a
reward-rate x Q-relative x difficulty grid. The published figure built the
whole thing as one frame in the notebook; this module does it a subject at a
time, in parallel, and returns only the per-facet sums -- which is what makes
the full resample count affordable (~10 min on ten cores rather than ~47
single-core, and gigabytes instead of hundreds).

Three things here are deliberate and were measured before being adopted; the
numbers are for the shipped ``chisq_…_4.8s`` fit at 1M simulated trials:

**Binning.** ``np.digitize(v, edges, right=True) - 1`` returns -1 for anything
at or below the first edge, which indexes the *last* row; and a grid sized by
the number of edges leaves its last row and column unreachable. Both were live
in the notebook: 1,846 trials per million with ``Q_relative == -1`` were drawn
at +1, and the reward-rate 1.0 row was never filled. :func:`binIndex` clips
instead of wrapping, and the grid is sized by bins.

**The nudge** (``nudge_sd``, default 0.01). The fitted Q learning rates are
high -- median ALPHA 0.83, and 12 of 22 subjects above 0.8 -- so after a loss
the chosen side collapses to ~(1-ALPHA) while the other stays near 1, putting
``Q_val`` on a handful of values per subject. 63.5% of trials land within ±0.1
of zero and the rest sit on a lattice whose spacing (~0.09-0.13) beats against
the 0.1 bin width, so neighbouring bins hold different populations of trial and
the surface steps. That stepping is not sampling noise: it stays at 0.18
(second-difference units) from 1M to 4.8M trials. Nudging the latents by 0.01
each trial removes half of it and 0.02 removes nearly all, while leaving the
trial-level gradient (-3.03 z per unit reward rate), the simulated RT (1.242 s)
and accuracy (0.729) unchanged. It is a visualisation smoother, not part of the
model -- see ``methods_model_revision.md`` Block 13.

**Subject-balanced facets** (``balanced``). Because each subject's lattice is
its own, a facet can be one animal: the Q -0.7..-0.6 column at low reward rate
was 97% ``Avgat1``, whose z sits 0.36 above the population on hard trials and
~0.13 below on medium and easy ones -- which is exactly the trough that showed
on the Medium and Easy sheets and not on Hard. Averaging per-subject means
first (each mouse once, as n=22 is reported elsewhere) removes it.

Usage::

    from .qrsurface import simulate
    grids = simulate(fit_path, resample_count=1_000, nudge_sd=0.01)
    surface = grids["balanced"]          # (reward rate, Q, difficulty)
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import os
import pathlib
import sys

import numpy as np
import pandas as pd

BIN_SIZE = 0.1
RR_EDGES = np.arange(0, 1 + BIN_SIZE, BIN_SIZE)          # 11 edges -> 10 bins
Q_EDGES = np.arange(-1, 1 + BIN_SIZE, BIN_SIZE)          # 21 edges -> 20 bins
DV_EDGES = np.array([0.0, 1 / 3, 2 / 3, 1.0])            # 4 edges  -> 3 bins
SHAPE = (RR_EDGES.size - 1, Q_EDGES.size - 1, DV_EDGES.size - 1)
DIFFICULTIES = ("Hard", "Medium", "Easy")

#: Default per-trial latent nudge. See the module docstring.
DEFAULT_NUDGE_SD = 0.01

_WORKER = {}


def binCentres(edges):
    """Where a bin's value belongs on the axis -- its middle, not its edge."""
    return edges[:-1] + np.diff(edges) / 2


def binIndex(values, edges):
    """Bin index in ``[0, len(edges) - 2]``.

    ``np.digitize(..., right=True) - 1`` gives -1 at or below the first edge,
    which silently wraps onto the last bin; clip instead.
    """
    return np.clip(np.digitize(values, edges, right=True) - 1, 0, edges.size - 2)


def qRelative(q_val, dv):
    """Q rotated onto the stimulus side: positive means the bias favours it."""
    q_val = np.asarray(q_val, dtype=float)
    dv = np.asarray(dv, dtype=float)
    out = np.abs(q_val) * (np.sign(dv) * np.sign(q_val))
    return np.where(dv == 0, q_val, out)


def subjectSeed(subject):
    """A seed from the subject's name, so a run does not depend on worker count."""
    return int(hashlib.sha256(subject.encode()).hexdigest()[:8], 16)


def resampleSubject(subj_df, resample_count, rng):
    """``resample_count`` pseudo-sessions per real session, with fresh stimuli.

    Each pseudo-session is one of the subject's own sessions copied with a new
    label and a fresh DV per trial -- the published behaviour, since the
    ``random_dv`` path never resampled trials. Built with one positional take
    rather than a frame per pseudo-session, and the labels kept as categories:
    as separate frames, 1,000 resamples cost tens of GiB before a single trial
    is simulated.
    """
    uniq = subj_df[["Date", "SessionNum"]].drop_duplicates().reset_index(drop=True)
    positions = subj_df.groupby(["Date", "SessionNum"], sort=False,
                                observed=True).indices
    chosen = rng.integers(0, len(uniq), size=len(uniq) * resample_count)
    keys = [tuple(uniq.iloc[i]) for i in chosen]
    pos_li = [positions[key] for key in keys]
    lens = np.fromiter((len(pos) for pos in pos_li), dtype=np.int64, count=len(pos_li))

    base_n = len(subj_df)
    out = subj_df.take(np.concatenate([np.arange(base_n)] + pos_li)
                       ).reset_index(drop=True)
    per_pseudo = np.repeat(np.arange(len(chosen)), lens)
    labels = [f"Random_{i}" for i in range(len(chosen))]
    for col in ("SessionNum", "SessId"):
        pseudo = pd.Categorical.from_codes(per_pseudo, labels)
        out[col] = pd.Categorical(np.concatenate(
            [out[col].to_numpy(dtype=object)[:base_n],
             np.asarray(pseudo, dtype=object)]))
    if "DVstr" in out.columns:
        out["DVstr"] = out["DVstr"].astype("category")

    dv = rng.uniform(-1, 1, size=int(lens.sum()))
    # Replace the column rather than assigning into it: the draws are float64
    # and the stored column is float32.
    out["DV"] = np.concatenate([subj_df["DV"].to_numpy(dtype=np.float64), dv])
    return out


def facetsFromTrials(sim_df):
    """Per-facet trial count, sum and sum of squares of the z-scored RT.

    The z-score is within subject, as the figure takes it, so a subject's
    facets can be computed without seeing any other subject.
    """
    from scipy.stats import zscore

    sim_df = sim_df[sim_df["valid"]]
    sim_df = sim_df[sim_df["SimChoiceCorrect"] == True]      # noqa: E712
    z = zscore(sim_df["SimRT"].to_numpy())
    dv = sim_df["DV"].to_numpy()
    idx = (binIndex(sim_df["RewardRate"].to_numpy(), RR_EDGES),
           binIndex(qRelative(sim_df["Q_val"].to_numpy(), dv), Q_EDGES),
           binIndex(np.abs(dv), DV_EDGES))
    counts = np.zeros(SHAPE)
    total = np.zeros(SHAPE)
    total_sq = np.zeros(SHAPE)
    np.add.at(counts, idx, 1)
    np.add.at(total, idx, z)
    np.add.at(total_sq, idx, z ** 2)
    return counts, total, total_sq, len(sim_df)


def runSubject(subject, fit, resample_count, nudge_sd):
    """Resample, simulate and bin one subject. Returns its facet aggregates."""
    from . import logic

    names = list(fit["fixed_params_names"])
    vals = fit["fixed_params_vals"]
    fixed = lambda name: vals[names.index(name)]                 # noqa: E731
    param_names = list(fit["params_names"])
    x = fit["OptimRes"]["x"]
    param = lambda name: float(x[param_names.index(name)])       # noqa: E731

    seed = subjectSeed(subject)
    subj_df = fixed("df").copy()
    subj_df["Name"] = subject
    df = resampleSubject(subj_df, resample_count, np.random.default_rng(seed))

    _, sim_df = logic.makeOneRun(
        df, include_Q=bool(fixed("include_Q")),
        include_RewardRate=bool(fixed("include_RewardRate")),
        biasFn=fixed("biasFn"), driftFn=fixed("driftFn"), noiseFn=fixed("noiseFn"),
        biasFn_df_cols=fixed("biasFn_df_cols"),
        driftFn_df_cols=fixed("driftFn_df_cols"),
        noiseFn_df_cols=fixed("noiseFn_df_cols"),
        biasFn_kwargs={k: param(k) for k in fixed("biasFn_kwargs")},
        driftFn_kwargs={k: param(k) for k in fixed("driftFn_kwargs")},
        noiseFn_kwargs={k: param(k) for k in fixed("noiseFn_kwargs")},
        ALPHA=param("ALPHA"), BETA=param("BETA"),
        NON_DECISION_TIME=param("NON_DECISION_TIME"),
        DRIFT_COEF=param("DRIFT_COEF"), NOISE_SIGMA=param("NOISE_SIGMA"),
        BOUND=param("BOUND"), dt=float(fixed("dt")), t_dur=float(fixed("t_dur")),
        return_df=True,
        # The loss walks every trial to build a fit statistic nothing here reads.
        skip_loss=True,
        latent_nudge_sd=nudge_sd, latent_nudge_seed=seed + 1)
    return facetsFromTrials(sim_df)


def _initializer(fit_path, repo_parent):
    if repo_parent not in sys.path:
        sys.path.insert(0, repo_parent)
    from . import fitio
    _WORKER["fits"] = fitio.loadFit(fit_path)


def _runOne(args):
    subject, resample_count, nudge_sd = args
    return (subject,) + runSubject(subject, _WORKER["fits"][subject],
                                   resample_count, nudge_sd)


def combine(per_subject):
    """Pooled and subject-balanced grids from each subject's facet aggregates.

    ``pooled`` weights every trial equally, ``balanced`` every subject -- the
    one the figure should use, since a facet can otherwise be a single animal.
    """
    counts = sum(c for c, _s, _sq in per_subject)
    total = sum(s for _c, s, _sq in per_subject)
    total_sq = sum(sq for _c, _s, sq in per_subject)
    with np.errstate(invalid="ignore"):
        pooled = np.divide(total, counts, out=np.full(SHAPE, np.nan),
                           where=counts > 1)
        var = np.divide(total_sq, counts, out=np.full(SHAPE, np.nan),
                        where=counts > 1) - pooled ** 2
        sem = np.sqrt(np.divide(np.clip(var, 0, None), counts,
                                out=np.full(SHAPE, np.nan), where=counts > 1))
        means = [np.divide(s, c, out=np.full(SHAPE, np.nan), where=c >= 2)
                 for c, s, _sq in per_subject]
    stacked = np.stack(means)
    # Averaged by hand rather than with nanmean, which warns on all-NaN facets.
    present = np.isfinite(stacked)
    per_facet = present.sum(axis=0)
    balanced = np.divide(np.where(present, stacked, 0).sum(axis=0), per_facet,
                         out=np.full(SHAPE, np.nan), where=per_facet > 0)
    return dict(pooled=pooled, balanced=balanced, sem=sem, counts=counts,
                subjects_per_facet=per_facet)


def _inNotebookKernel():
    """True inside a Jupyter kernel, where spawning a pool is not safe here.

    A spawned worker re-imports the parent's ``__main__`` -- the kernel
    launcher -- and that import reaches ``pdb``, which imports ``code``. This
    repo has its own ``code`` package, and the notebooks' bootstrap cell puts
    it in ``sys.modules`` under that name, so the worker dies before it runs
    anything of ours (``BrokenProcessPool``). Running the pool in a plain
    interpreter avoids the whole question.
    """
    try:
        from IPython import get_ipython
    except ModuleNotFoundError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def _simulateViaSubprocess(fit_path, resample_count, nudge_sd, workers, subjects):
    """Run :func:`simulate` in a clean interpreter and read back its grids."""
    import json
    import subprocess
    import tempfile

    repo_parent = pathlib.Path(__file__).resolve().parents[4]
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "qr_grids.npz"
        cmd = [sys.executable, "-m", __name__,
               "--fit", str(pathlib.Path(fit_path).resolve()),
               "--count", str(resample_count), "--nudge", str(nudge_sd),
               "--out", str(out)]
        if workers is not None:
            cmd += ["--workers", str(workers)]
        if subjects is not None:
            cmd += ["--subjects", *subjects]
        subprocess.run(cmd, cwd=repo_parent, check=True)
        with np.load(out, allow_pickle=False) as data:
            grids = {k: data[k] for k in data.files if k != "meta"}
            meta = json.loads(str(data["meta"]))
    grids.update(meta)
    return grids


def simulate(fit_path, resample_count=1_000, nudge_sd=DEFAULT_NUDGE_SD,
             workers=None, subjects=None, via_subprocess=None):
    """Build the 7D grids from a fit file, one subject per worker.

    ``workers`` defaults to all cores but one, capped at 10 -- each worker
    holds its subject's resampled frame plus the simulator's per-step arrays,
    about 3-4 GiB at ``resample_count=1000``.

    Called from a notebook, the pool runs in a subprocess (see
    :func:`_inNotebookKernel`); pass ``via_subprocess=False`` to force the
    in-process pool.
    """
    from . import fitio

    fit_path = pathlib.Path(fit_path)
    if via_subprocess is None:
        via_subprocess = workers != 1 and _inNotebookKernel()
    if via_subprocess:
        return _simulateViaSubprocess(fit_path, resample_count, nudge_sd,
                                      workers, subjects)
    fits = fitio.loadFit(fit_path)
    names = list(subjects) if subjects is not None else list(fits)
    if workers is None:
        workers = min(max((os.cpu_count() or 2) - 1, 1), 10)
    workers = max(1, min(workers, len(names)))
    tasks = [(name, resample_count, nudge_sd) for name in names]

    if workers == 1:
        results = [(name,) + runSubject(name, fits[name], resample_count, nudge_sd)
                   for name in names]
    else:
        del fits            # each worker loads its own copy
        repo_parent = str(pathlib.Path(__file__).resolve().parents[4])
        with cf.ProcessPoolExecutor(max_workers=workers, initializer=_initializer,
                                    initargs=(str(fit_path), repo_parent)) as pool:
            results = list(pool.map(_runOne, tasks))

    order = {name: i for i, name in enumerate(names)}
    results.sort(key=lambda row: order[row[0]])          # worker order must not matter
    grids = combine([row[1:4] for row in results])
    grids["trials"] = int(sum(row[4] for row in results))
    grids["subjects"] = [row[0] for row in results]
    grids["resample_count"] = resample_count
    grids["nudge_sd"] = nudge_sd
    return grids


def main(argv=None):
    """``python -m …model.qrsurface --fit … --out grids.npz`` (run from the
    directory above the checkout, so the package resolves)."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fit", required=True)
    parser.add_argument("--count", type=int, default=1_000)
    parser.add_argument("--nudge", type=float, default=DEFAULT_NUDGE_SD)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    grids = simulate(args.fit, resample_count=args.count, nudge_sd=args.nudge,
                     workers=args.workers, subjects=args.subjects,
                     via_subprocess=False)
    meta = {k: grids.pop(k) for k in ("trials", "subjects", "resample_count",
                                      "nudge_sd")}
    np.savez(args.out, meta=json.dumps(meta), **grids)
    print(f"{meta['trials']:,} correct trials from {len(meta['subjects'])} subjects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
