# MOSS-Image's full fine-tune on Leonardo: the runbook

Not run yet. This page is what has to be true, in order, before the HPC run is submitted, and the commands for each
step; every number is measured on the Spark (GB10) unless it says "estimate". [MOSS.md](MOSS.md) has the model and
the trainer; `jobs/moss/run_content_pilot.sh` the pilot this run scales up.

## 0. Before anything

- **The recipe is decided** (TODO.md, "To the HPC run"). Evidence so far (6 Nov): the pilot's look follows the sample
  (swap test at step 2,000: L1 6/6, L2 5/7, L3 3/3); its hand does not, even from the page's own sample (writer ID on
  the six swap pages, drawn with their own sample vs their real windows at the same size: Trolldomskommissionen 0.00
  vs 0.94, Krigsarkivet 0.00 vs 0.94, Svea hovrätt koncept 0.00 vs 0.79, Svea hovrätt renskrivna 0.53 vs 0.89); the
  VAE keeps the hand (47 held-out pages, 874 lines: 0.59 of the real lines named as their own hand, 0.57 of the same
  windows encoded and decoded). Whether the hand improves with steps (own-sample writer ID at steps 500-2,000) decides
  between training the sample form as it is and training look and text only, with hands from per-hand adapters.
- **Budget and account**: `saldo -b` on a login node; `export SBATCH_ACCOUNT=<project>` in the cluster's `~/.bashrc`
  (never in this repo). One pass over ~226,000 rows is estimated at ~23 h on 8 A100s (TODO.md's arithmetic, not
  measured on an A100).
- **Login**: `step ssh login`, then `ssh login.leonardo.cineca.it` (the `cineca-hpc` skill has the setup).

## 1. The full training set (on the Spark, CPU)

The pilot's set (`data_flux3/page_pairs_many_1mp`) read 80 scans a collection (3,623 scans, 15,745 training rows); the
GT has ~52,500 PAGE files, so the full set is ~226,000 rows (estimate, at the pilot's rows a scan). Same commands as
the pilot, without `--max-scans`, each CPU step capped as `run_content_pilot.sh` caps it:

```bash
D=data_flux3/page_pairs_full_1mp
.venv/bin/python -m diffu_page.data.page_pairs --gt-root <GT folder> --model runs/moss/base --out $D \
    --megapixels 1.0 --windows-per-page 2 --val-volumes 0.1 --workers 6 --seed 0
# then, as run_content_pilot.sh steps 1-2 do: leave out GT-RA_landskapshandlingar_20260227 (the same scans as
# _20260220), the content forms (diffu_page.data.page_pair_forms), the legible samples (diffu_page.data.page_pair_samples)
```

Disk: the pilot's set is 17 GB for 3,623 scans, so ~250 GB (estimate); the Spark has 1.8 TB free.

## 2. The cache: where and how big

A cache entry with its prompt's embeddings is 10.5 MB (the pilot's: 126 GB for 12,002 rows), so the full cache would
be ~2.3 TB: more than `$WORK` or `$FAST` hold (1 TB each; the project's `$FAST` was empty on 6 Nov), and `$SCRATCH` (no quota shown) deletes files after 40 days. With
`--latents-only` an entry holds the target's and the sample's latents only (~2 MB, estimate: two `[1, 128, 63, 63]`
bfloat16 tensors at 1 MP), ~450 GB in all, and the trainer encodes each prompt in the step (`--encode-text`, the text
encoder's 16.8 GB on every GPU: ~47 GB a GPU on 4 GPUs, below the A100's 64 GB).

Building it: on the Spark the pilot cached 12,000 rows in ~5.5 h with prompt embeddings (latents alone: 300 rows in
340 s, 1.8 MB a row), so the full set would take days there. On Leonardo, `$WORK` holds 21 TB (checked 6 Nov), so the
cache keeps its prompt embeddings (~1.4 TB for the ~135,000 rows of the 30,983 scans' set) and the training needs no
text encoder. `diffu-moss-cache-pairs --shard i --shards 4` caches a quarter of the plan into the same directory, one
run a GPU; a run without `--shards` over that directory then finds every file, caches any a stopped shard left, and
writes `entries.jsonl` and `cache.json`: the cache a single run writes (tested on the tiny model, file for file).

## 3. On Leonardo (login node: internet, ≤ 10 min CPU a command)

1. **Code and environment** in `$WORK`: clone the repo; `uv sync` (uv from PyPI into `~/.local/bin`: `module load
   python/3.11.7; python3 -m pip install --user uv`; uv fetches Python 3.12 itself, with `UV_PYTHON_INSTALL_DIR` and
   `UV_CACHE_DIR` in `$WORK`). **Leonardo's compute nodes run driver 535.274 (CUDA 12.2)** (checked 6 Nov), so the
   repo's torch 2.14.1 built for CUDA 13.0 (driver 580 or newer) does not run there. torch **2.14.1+cu126** does:
   tested on 4 A100-SXM-64GB in the debug QOS (bf16 matmul, `scaled_dot_product_attention`, `torch.compile`, NCCL
   `all_reduce` across the 4 GPUs under `torch.distributed.run`). **To build**: the environment on Leonardo resolved
   against the cu126 index (a Leonardo variant of `pyproject.toml`'s torch index, or `uv pip install` of the locked
   set with torch from cu126), then the trainer's own smoke below.
2. **The model**: MOSS-Image Base at `85975793d8b3d432f1b6e27f352abaa4c2a4c0d3` into `$FAST` (`hf download` on the
   login node; the compute nodes have no internet, so runs set `HF_HUB_OFFLINE=1`), then `python -m diffu_moss.init
   --base <snapshot> --out $FAST/moss/base` (the model directory `--encode-text` reads).
3. **The data**: the cache and the validation cache copied to `$FAST` (`rsync -a --partial`, from the Spark).

## 4. The smoke test, then the run

- **Smoke** (debug QOS, 30 min, one node): `jobs/moss/hpc_full.sbatch.example` with `STEPS=20 SAVE_EVERY=10` on the
  validation cache. Passes if: the loss is finite and falls, a checkpoint is written and reloads (`--resume`), the
  peak GPU memory is under 64 GB, and seconds a step is measured (to replace the estimate above).
- **The run** (production QOS): the same script with the full cache, `--ema` and `--save-state`, checkpoints every
  ~1,000 steps; a job's walltime is capped, so the run resumes job to job from its last `--save-state` checkpoint.
- **Checked after**: the swap test and held-out pages (`~/diffu_queue/weekend`: `draw_swap.py`, `score_swap.py`,
  `draw_heldout.py`) on the Spark against a checkpoint copied back, beside the real scans.

## Still open (none of it tested)

- The repo's environment with torch from the cu126 index on Leonardo (section 3; torch alone is tested).
- The submit script on real GPUs: tested on CPU only (gloo, world size 2, tiny models).
- Seconds a step on an A100, so the walltime and the number of jobs are estimates until the smoke test.
