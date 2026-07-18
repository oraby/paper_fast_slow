# Slurm launch scripts for the RL model fitter

## Context

`model_runner.py` fits one RL/DDM model to the behavioral dataset, and already
supports `--only-subject S1 [--only-subject S2 …]` to fit a single subject and
**reload-merge** its result into the shared on-disk pickle
([`fit.py:_merge_save_evolve`](../../OneDrive%20-%20Floating%20Reality/Documents/Hatem/paper_fast_slow/code/rlmodel/model/fit.py) — line 620) so parallel
per-subject processes don't clobber each other. Today the only launcher is
[`models_loop.sh`](../../OneDrive%20-%20Floating%20Reality/Documents/Hatem/paper_fast_slow/code/rlmodel/models_loop.sh),
which runs subjects **serially** inside one process.

We want to fan the fit out across a Slurm cluster: **one job per subject** for a
given model + fit mode, so all subjects fit in parallel. Chi²  fits are
CPU-bound and want a whole node; MLE fits use the CuPy GPU backend and want one
≥16 GB GPU. This is exactly the "a few long-running processes each fitting a
different subject" workload the reload-merge save was designed for.

## Invocation facts (from the code / cluster info)

- Runner is invoked as `python -m code.rlmodel.model_runner …` **from the
  project root** (`paper_fast_slow/`, one level above `code/`), same as
  `models_loop.sh`. `DF_FP = "data/behavior/df_behavior.pkl"` resolves relative
  to that cwd.
- Cluster Python = conda; default env **`py314`**, overridable per the
  `--conda-env` flag below.
- `--fit-mode` is `chisq` or `mle`. MLE **requires** `--mle-backend` (runner
  errors otherwise). GPU backend selects `--mle-backend GPU`.
- With `--gres=gpu:1`, Slurm sets `CUDA_VISIBLE_DEVICES` to the allocated GPU and
  CuPy (`array_backend.py:77`, `cp.cuda.Device(id).use()`) indexes **relative**
  to that mask → the allocated GPU is index 0 in-job. So we **do not** pass
  `--mle-device-id` at all; CuPy defaults to the single visible GPU.
- `normal` and `gpu` partitions both cap `MaxTime=04:00:00`, but `--time` is
  **not** set — single-subject jobs finish well under the cap and Slurm falls
  back to the partition default.

## Prerequisite code change: a flag to silence the MLE progress bar

The GPU MLE path force-enables a `tqdm` progress bar
([`model_runner.py:517`](../../OneDrive%20-%20Floating%20Reality/Documents/Hatem/paper_fast_slow/code/rlmodel/model_runner.py):
`mle_show_progress = bool(args.mle_progress or args.mle_backend == "GPU")`),
and the bar is built with `tqdm.auto` + `leave=False` in
[`mle_batch.py:_progress_iter`](../../OneDrive%20-%20Floating%20Reality/Documents/Hatem/paper_fast_slow/code/rlmodel/model/mle_batch.py)
(line 972). It doesn't break batch runs, but under Slurm (stdout → log file, no
TTY) tqdm's `\r` in-place updates don't overwrite — they accumulate, bloating
the per-job log with thousands of progress lines. There is currently **no** CLI
way to disable it for the GPU backend.

Add a **`--mle-no-progress`** store_true flag to `model_runner.py`:
- argparse entry describing it as "force-disable the MLE tqdm bar even under
  `--mle-backend GPU`; use for batch/Slurm runs whose stdout is a log file."
- Change line 517 to:
  `mle_show_progress = bool((args.mle_progress or args.mle_backend == "GPU") and not args.mle_no_progress)`.

The MLE Slurm script passes `--mle-no-progress` by default. (This is preferred
over relying on a `TQDM_DISABLE` env var, whose effect on the Python `tqdm` API —
vs the tqdm CLI — is not guaranteed.)

## Files to create (all under `code/rlmodel/slurm/`)

### 1. `gen_subjects.py` — one-time subject-list generator
Replicates the runner's own subject set: `loadDF(min_valid_trials=0)` from
`model_runner`, then writes `sorted(df.Name.unique())` — one name per line — to
`slurm/subjects.txt`. Reuses `model_runner.loadDF` (do not reimplement the
inclusion logic). Run once (e.g. `conda run -n py314 python -m
code.rlmodel.slurm.gen_subjects` from the project root) and commit the result;
regenerate only if the dataset's subject set changes.

### 2. `subjects.txt` — static list, one subject per line
Produced by the generator. Its **line count N** determines the array range baked
into the two sbatch scripts (`--array=0-(N-1)`).

### 3. `fit_chisq.sbatch` — Chi² array job (exclusive node, `normal`)
`#SBATCH` directives: `--partition=normal`, `--exclusive`,
`--array=0-<N-1>`, `--job-name=rl_chisq`,
`--output=code/rlmodel/slurm/logs/rl_chisq_%A_%a.out`.

Body:
1. Optional-arg scan: pull `--conda-env NAME` out of `"$@"` (default `py314`);
   remaining args are the model spec + extras to forward.
2. `source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$env"`.
3. `cd` to the project root: `cd "${RL_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"`
   (submit from the project root, or export `RL_PROJECT_ROOT`).
4. `subject=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" code/rlmodel/slurm/subjects.txt)`;
   if empty, `exit 0` (guards an oversized array).
5. `python -m code.rlmodel.model_runner --fit-mode chisq \
      --num-cpus "$SLURM_CPUS_ON_NODE" --only-subject "$subject" "$@"` — the
   forwarded `"$@"` carries `--drift/--bias/--noise` and any extras
   (`--asym`, `--scale-bound`, `--mle-*`, …) and comes **last** so a
   user-supplied flag overrides a script default.

### 4. `fit_mle.sbatch` — MLE array job (one ≥16 GB GPU, `gpu`)
`#SBATCH` directives: `--partition=gpu`, `--gres=gpu:1`, `--cpus-per-task=4`,
`--mem=16G`, `--array=0-<N-1>`, `--job-name=rl_mle`,
`--output=code/rlmodel/slurm/logs/rl_mle_%A_%a.out`. (`--gres=gpu:1` on the
`OverSubscribe=NO` `gpu` partition already gives exclusive use of the allocated
GPU; the ≥16 GB minimum is satisfied by the partition's GPUs — no feature exists
to constrain it further.)

Body: same steps 1–4 as chisq, then:
`python -m code.rlmodel.model_runner --fit-mode mle --mle-backend GPU \
   --mle-no-progress --only-subject "$subject" "$@"` — **no** `--mle-device-id`.
`--mle-no-progress` keeps the log clean (see prerequisite change above). Users
pass `--mle-gpu-memory-gb`, `--mle-conditions`, `--mle-mle-weight`, etc. through
`"$@"`.

### 5. `logs/` directory
`#SBATCH --output` paths are resolved by Slurm **before** the job body runs, and
Slurm does **not** create the directory — so it must exist at submit time or the
job's stdout/stderr is silently lost. To make sure it always exists:
- Commit `code/rlmodel/slurm/logs/.gitkeep` so the dir ships with the repo.
- `gen_subjects.py` also does `Path("code/rlmodel/slurm/logs").mkdir(parents=True,
  exist_ok=True)` as part of setup.
- Each sbatch script's header-comment usage line reminds: run
  `mkdir -p code/rlmodel/slurm/logs` before the first submit if the dir is
  missing.

## Usage (documented in a header comment in each script)

```
# from the project root (paper_fast_slow/):
sbatch code/rlmodel/slurm/fit_chisq.sbatch \
    --drift "RewardRate" --bias "Q-Val (Offset)" --noise "Normal(0, 1)" --asym

sbatch code/rlmodel/slurm/fit_mle.sbatch \
    --conda-env py314 \
    --drift "RewardRate" --bias "Q-Val (Offset)" --mle-gpu-memory-gb 16
```

## Notes / caveats to bake into the scripts

- **Line endings:** write the `.sbatch`/`.py` files with **LF** (they run on
  Linux); avoid CRLF from the Windows editor.
- **Concurrency:** `_merge_save_evolve` narrows but does not fully close a TOCTOU
  window between reload and atomic replace. Writes are millisecond-scale vs
  minute/hour fits, so collisions across the array are unlikely; if a subject
  ends up missing from the pickle, just resubmit that array index. Documented in
  a comment, no code change.
- `--only-subject` is always injected by the script; per the request it is never
  user-supplied. The script does **not** try to interpret other extras — it
  forwards them verbatim.

## Verification

1. **Generator:** run `gen_subjects.py` once; confirm `subjects.txt` has the
   expected subjects (spot-check count vs `df.Name.nunique()`), set `<N>` in both
   sbatch `--array` lines.
2. **Dry syntax check (no cluster):** `bash -n fit_chisq.sbatch` /
   `bash -n fit_mle.sbatch` to catch shell errors.
3. **Single-task smoke test on the cluster:** submit with a 1-element array,
   e.g. `sbatch --array=0-0 code/rlmodel/slurm/fit_chisq.sbatch --drift Classic
   --bias None_ --noise "Normal(0, 1)"`; check the log shows the right subject,
   conda env, cwd, and a completed fit, and that the subject's key appears in the
   model's pickle under `data/RLModel/`.
4. **MLE smoke test:** same with `fit_mle.sbatch`; confirm the log reports the
   CuPy GPU backend initialized on the allocated device (no device-id error) and
   the fit ran on GPU.
5. Then submit the full arrays for the real model(s).
