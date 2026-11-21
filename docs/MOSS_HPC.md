# MOSS-Image's full fine-tune on Leonardo: the runbook

Run. Two full fine-tunes have trained on Leonardo's Booster (A100 64 GB, 4 a node): B at 1 MP to step 7,000 (7-8 Nov
2026) and the sample-above model at 2 MP to step 10,257 (from 13 Nov), the second almost wholly on a chain of 30-minute
debug-QOS jobs; the distillation students ran as such chains too (20 Nov). This page is the order of the steps and the
commands; every run's settings, jobs and results are in [MOSS_RUNS.md](MOSS_RUNS.md), the model and the trainer in
[MOSS.md](MOSS.md). The Slurm scripts the runs used live in the project's `moss/` folder on the cluster;
`jobs/moss/hpc_full.sbatch.example` is the template they follow.

## 0. Before anything

- **Budget and account**: `saldo -b` on a login node; `export SBATCH_ACCOUNT=<project>` in the cluster's `~/.bashrc`
  (never in this repo).
- **Login**: `step ssh login`, then `ssh login.leonardo.cineca.it` (the `cineca-hpc` skill has the setup).
- **Queues**: QOS `normal` gives 24 h a job (`boost_qos_bprod` refuses a 2-node job, `QOSMinCpuNotSatisfied`); its
  jobs of 2-8 nodes waited days in November (`sbatch --test-only` on 17 Nov: about 5 days). The debug QOS gives 30
  minutes, 2 jobs running and 4 queued a user: section 5.

## 1. Code, environment and model (login node: internet, at most 10 min CPU a command)

1. **Code and environment** in `$WORK`: clone the repo; `uv sync` (uv from PyPI into `~/.local/bin`: `module load
   python/3.11.7; python3 -m pip install --user uv`; uv fetches Python 3.12 itself, with `UV_PYTHON_INSTALL_DIR` and
   `UV_CACHE_DIR` in `$WORK`). **Leonardo's compute nodes run driver 535.274 (CUDA 12.2)**, so the repo's torch 2.14.1
   built for CUDA 13.0 (driver 580 or newer) does not run there; torch **2.14.1+cu126** does. The cluster's checkout
   carries that as a local edit of `pyproject.toml` and `uv.lock`, kept as a patch beside it and applied again after
   every update of the checkout.
2. **The model**: MOSS-Image Base at `85975793d8b3d432f1b6e27f352abaa4c2a4c0d3` (`hf download` on the login node; the
   compute nodes have no internet, so jobs set `HF_HUB_OFFLINE=1`), then `python -m diffu_moss.init --base <snapshot>
   --out <model dir>`: the model directory the cache is encoded with.

## 2. The training set

Built on CPU by `diffu_page.data.page_pairs` over the whole GT, then its prompt forms and legible samples
([MOSS.md](MOSS.md), "Training data"); the 2 MP set was built on Leonardo itself, by a chain of debug-QOS jobs.

```bash
python -m diffu_page.data.page_pairs --gt-root <GT folder> --model <model dir> --out <set> \
    --megapixels 2.0 --windows-per-page 2 --val-volumes 0.1 --sample-pick nearest --workers 6 --seed 0
```

The 2 MP set is 143,375 training rows (49,504 pages and 95,709 windows before the GT's duplicate collection,
`GT-RA_landskapshandlingar_20260227`, is left out) and 11,919 held out. The sample-above run trains on a set made from
it: every sibling pair of windows (27,350), each window with the other window of its own page as its sample, and as
many whole pages.

## 3. The cache

`diffu-moss-cache-pairs --shard i --shards N` caches one Nth of the plan into the same directory, one run a GPU; a run
without `--shards` over that directory then finds every file, caches any a stopped shard left, and writes
`entries.jsonl` and `cache.json`: the cache a single run writes. The 2 MP cache was built as 32 shards over 4 nodes in
30-minute jobs, each going on where the last stopped, and indexed once on the serial partition (43 minutes for the
first build). `$WORK`
holds it with its prompt embeddings, so the training loads no text encoder; an entry with them is about 10 MB at 1 MP.
`--latents-only` with the trainer's `--encode-text` is the smaller cache (the text encoder's 16.8 GB on every GPU), not
needed there.

Check a cache's mix before training on it: a cache built before 11 Nov without `--limit` tied each pair's prompt form
to its place in the manifest (MOSS_RUNS.md, the note of 11 Nov; fixed in the code, with a test).

## 4. The smoke test, then the run

- **Smoke** (debug QOS, one or two nodes): the template with `STEPS=20 SAVE_EVERY=10` on a small cache, submitted with
  `--time=00:30:00`. It passes if the loss is finite and falls, a checkpoint is written with the settings the run needs
  (the sample-above smoke checked `reference_place: above` in `transformer/config.json` and `ema/config.json`), and it
  resumes (`RESUME=latest`).
- **The run**, as the sample-above model trained: from Base, 2 nodes (8 A100), 16 samples a step, three passes of the
  set (10,257 steps) with the learning rate wound down over the last sixth (`DECAY_STEPS`, `DECAY_TO`), EMA 0.999
  (MOSS's 0.9999 has a horizon of about 10,000 steps), size blocks, the sample above the page:

  ```bash
  REPO=<checkout> MOSS_BASE=<Base snapshot> CACHE=<cache> OUT=<run dir> STEPS=10257 SAVE_EVERY=1000 EMA=0.999 \
      DECAY_STEPS=<the last sixth> SIZE_BLOCK=32 REFERENCE_PLACE=above RESUME=latest \
      sbatch --nodes=2 --qos=normal <the script>
  ```

  `RESUME=latest` goes on from the newest whole checkpoint, so jobs chain (`sbatch --dependency=afterany:<previous>`),
  and the first, finding none, starts from Base.
- **Measured on 8 A100s**: at 1 MP 11.6 s a step and 20 GB a GPU, every block checkpointed (without gradient
  checkpointing every GPU ran out of memory); at 2 MP 18.4 s and 25 GB, 15.2 s with `--size-block 32`. On 4 nodes at
  16 samples 9.7 s (median), on 8 nodes at 32 samples 13.2 s, with `--replicas 8` (HSDP) 11.0 s: per sample 1.4x, 2.3x
  and 2.8x the speed of 2 nodes. A checkpoint with its state and EMA is about 72 GB (state 42, DiT 10, EMA 20).
- **Checked after**: held-out pages and the swap test drawn from the EMA, beside the real scans and the sample pages
  (on the Spark, from a checkpoint copied back, or by a debug-QOS job).

## 5. A run as a chain of debug-QOS jobs

When the normal queue gives no start, a run trains in 30-minute jobs: each resumes from the newest whole checkpoint and
trains `--stop-at` a fixed number of steps further (`STOP_AT`; the schedule stays `STEPS`). What the sample-above run's
chain learned, each from a stop:

- A job submits the next at its own start (`afterany` on itself), not at its end, where it may be out of time; each
  decides at its start whether to go on, and the chain stops after a few jobs in a row without a new checkpoint.
- A self-submitting job must not resubmit after a run of seconds (a broken last step once queued ~27,000 empty jobs).
- An unfinished checkpoint is removed before the next job starts.
- Size the job for a slow day: a resume on 2 nodes takes about 3.5 minutes with the state's files read whole (11 when
  mapped from `$WORK`), and a 2 MP step 17-22 s, so 48 steps a job fit when the shared disk is slow, 64 when it is not.
- A resume on 8 nodes takes about 10 minutes (every node reads the whole ~40 GB state): fine for jobs of hours, not for
  30-minute ones.

Distillation (`diffu-moss-distill`, [MOSS.md](MOSS.md)) chains the same way, with its own `--stop-at` and `--resume
latest`: a step of 8 samples on 8 A100s took about 49 s, so 22 steps a job.

## Still open

- The template's own header still says it is tested on CPU only; the runs used the cluster's copies of it.
- Compiled joint blocks in training (30 steps at 2 MP): waiting for a slot on 11 Nov, no result recorded.
- The next full run's size: 8 nodes with `--replicas 8` is the fastest measured, and needs the normal queue.
