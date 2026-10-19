# Training Diffu on the DGX Spark (GB10)

End-to-end: clone → install → smoke test → prep data → train → sample. Stage 1 = flow matching on real
lines. The success metric is **downstream HTR CER on held-out real lines**, not loss or FID.

The box is a DGX Spark: one NVIDIA GB10 GPU (index 0, compute capability 12.1 = sm_121), aarch64, 119 GB of
unified memory shared by CPU and GPU. The Makefile defaults fit it: every target runs on GPU 0 with one
process. On a multi-GPU host override `GPUS` / `NPROC` (`make train`; it adds `--multi_gpu` only when
`NPROC>1`), `GPUS_1` (the `*-1gpu` targets) or `GPUS_2` (the `*-2gpu` targets). For the same runs as Slurm
jobs on CINECA Leonardo (4× A100 per node, offline compute nodes), see [`slurm/README.md`](../slurm/README.md).

## 0. Get the code
```bash
git clone https://github.com/Borg93/diffu2.git && cd diffu2
```

## 1. Install
`uv sync` pulls everything; `torch` / `torchvision` come from the CUDA 13.0 (cu130) wheel index (wired into
`pyproject.toml`):
```bash
make install        # uv sync (runtime deps)
make install-dev    # uv sync --extra dev (adds ruff / pyrefly / pytest)
uv sync --extra dev --extra track --extra hf   # + trackio (>=0.42) and datasets
```
A plain `uv sync` (and so `make install`) removes the extras it isn't given, so list every extra you need in
one call. Check the GPU:
```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
# -> 2.14.1+cu130 NVIDIA GB10 (12, 1)
```

> **DINOv3 is gated.** Accept the model licence on the Hub and run `hf auth login`. REPA (on by default) needs
> it in both style modes, and the dino-mode style encoder is DINOv3; there is no fallback encoder.

> **The HTR recognizer loads from the local Hugging Face cache only** (`local_files_only=True` in the live
> gen_CER gauge, `diffu-eval` and `diffu-vae-gate`). Download it once before the first run:
> `hf download Riksarkivet/trocr-large-handwritten-hist-swe-3-char` (the default) or
> `hf download microsoft/trocr-base-handwritten` (English, for IAM).

## 2. Smoke test (no dataset needed)
A few real training steps on synthetic data, through the full model. The first run downloads the VAE, DINOv3
and torchvision's ImageNet ResNet-18 (the glyph CNN's warm start); after that they are cached.
```bash
make smoke-1gpu     # dino mode, REPA on
make smoke-compile  # the same through the regionally compiled backbone
make smoke-canvas   # --style-canvas --glyph-pad
make smoke-2gpu     # multi-GPU hosts only: GPUS_2 (default 0,1), accelerate --multi_gpu (exercises DDP)
```
`smoke OK — 3 step(s) on 2-line batches (dino +repa)` = good. `diffu-smoke` takes the same architecture flags
as `diffu-train`.

## 3. Prep data
**PAGE-XML / ALTO-XML export:**
```bash
make data XML_DIR=/path/to/xml IMAGE_DIR=/path/to/page_images OUT_DIR=data_out
```
Runs `diffu-data --height 128` (128 is also its default, `Config().data.line_height`) and writes
`data_out/lines/*.png`, `data_out/manifest.jsonl` and writer-disjoint `train/val/test.jsonl`. The writer id is the
page (`diffu-data --split-by volume` splits by volume instead). Crops are clamped to the page, so a box past the edge
gets no black border. It prints the å ä ö Å Ä Ö coverage — check that the diacritics are well represented.
Pre-cropped lines: `make ingest DIR=crops/ OUT_DIR=data_out` (or `diffu-ingest --input lines.jsonl --out-dir
data_out`).
Source-specific recipes live in `scripts/prepare_*.py`.

**IAM with real writer ids** (`data_iam/`): `Teklia/IAM-line` has only image + text, so
`scripts/prepare_iam_writers.py` joins each line to its IAM form, and so to its writer, through DiffusionPen's
preprocessed IAM word files:
```bash
hf download konnik/DiffusionPen --include "saved_iam_data/*" --local-dir data_raw/diffusionpen
uv run --extra hf python scripts/prepare_iam_writers.py --out-dir data_iam
```
Result: `{image, text, writer_id, form_id}` rows — train 6275 lines / 283 writers, val 935 / 56, test 2695 / 161,
writer-disjoint. Teklia's split has 6482 / 976 / 2915 lines; the 3–8 % that couldn't be matched to a form are
dropped. The lines are 128 px tall and wide: p10–p90 ≈ 1.2k–2.6k px, ~95 % wider than 1024. So IAM trains with
`--max-line-width 3072 --max-ref-width 2048 --drop-wide-lines`, which skips the 245 train / 34 val lines wider
than 3072 px.

The width filter and `--bucket-width` read each crop's on-disk width (cached raw in a
`{split}.widths.json` sidecar next to the manifest), so store crops at the training line height —
`make data` does.

## 4. Train
```bash
make train OUT_DIR=data_out STEPS=100000 BATCH=8      # GPU 0: data_out/train.jsonl -> data_out/checkpoints/
```
`make train` runs `accelerate launch --mixed_precision bf16 -m diffu.train` with only `--data-dir`, `--out-dir`,
`--steps`, `--lr` and `--batch-size` (`BATCH` is per GPU, default 64). For anything else call `diffu-train`
directly; on one GPU no `accelerate launch` is needed (bf16 mixed precision is set in the code):
```bash
uv run diffu-train --data-dir data_out --out-dir runs/dino \
    --cond-dropout 0.1 --text-dropout 0.1 \
    --bucket-width --no-grad-checkpoint --compile --batch-size 8 --track
```
The dino run above is the CFG-null recipe of `exp_sd35_128` (trains both the full null and the text-only null
that inference CFG uses) plus the measured speed flags. The canvas recipe is in [`HANDOFF.md`](HANDOFF.md).

**Checkpoints.** Every checkpoint dir holds `config.json`, the full `Config` that every loader reads back:
```
<out-dir>/step_N/   resumable accelerate state (raw weights, optimizer, LR schedule, EMA, RNG; ~17 GB)
                    + config.json + ema_model.safetensors (the EMA weights, what sampling uses)
<out-dir>/final/    config.json + model.safetensors (EMA applied) at the end of the run
<out-dir>/samples/  step_NNNNNN.png sample sheets (REAL / OURS rows + what the recognizer read)
```
The exports (`ema_model.safetensors`, `final/model.safetensors`) leave out the frozen pretrained VAE and DINOv3
weights; those load from the Hub. The `model.safetensors` inside `step_N/` is accelerate's copy of the raw,
non-EMA weights. `--save-every` sets the cadence (default 10000). `--no-ema` drops the EMA (`step_N/` then has
no `ema_model.safetensors`, and `final/` holds the raw weights).

**Loading.** `diffu-generate`, `diffu-synth`, `DiffuPipeline` and `scripts/export_release.py` /
`scripts/extract_ema.py` read the `config.json` next to the weights (`diffu.generate.load_config`, then
`build_model`); architecture flags given on the command line override it. A checkpoint without a `config.json`
gets the `Config` defaults, so pass the architecture flags it trained with. (`--init-from` loads weights only:
the run's architecture is `Config` + its own flags.) Pre-Unifont and SD3.0-era checkpoints (e.g.
`exp_base_weekend`) no longer load.

**Architecture flags** (`config.add_arch_args`; the same on `diffu-train`, `diffu-generate`, `diffu-synth`,
`diffu-smoke`): `--line-height` (128), `--style-canvas`, `--glyph-pad`, `--no-fill-ratio`, `--max-line-width`
(1024), `--max-ref-width` (1024; canvas), `--max-chars` (128).

**Resume.** `--resume <out-dir>/step_N` restores model, optimizer, LR schedule, EMA, RNG and the step
counter (read from the dir name); use the same number of processes the checkpoint was saved with.
`--resume auto` picks the newest complete `step_N` under `--out-dir` (complete = its `config.json`, written
after the state, exists) or starts fresh if there is none — for chained jobs. `--init-from
<weights>.safetensors` loads weights only and restarts optimizer, LR and step. A LoRA run can't `--resume`
(rejected up front: its checkpoints hold only the adapter).

**Speed.** The measured recipe is `--bucket-width --no-grad-checkpoint --compile`:
- `--compile` regionally compiles the MMDiT block (`dynamic=True`, `fullgraph=True`). On the GB10 a compiled
  forward+backward ran 1.41× faster than eager (batch of 4 lines at 1024 px; 36 s cold compile); the full
  recipe measured ~1.9× on the old RTX PRO 6000 box (`PERF_AUDIT.md` §0e). It also works with gradient
  checkpointing (the default), which recomputes each block in the backward: slower, but the fallback when a
  batch doesn't fit in GPU memory. Dynamo's settings are per-thread, so the backbone re-applies them on
  whichever thread runs a compiled block (autograd's backward thread, a server's worker threads).
- `--bucket-width` batches similar-width lines (padding waste 54.5 % → 1.3 % at batch 32, `PERF_AUDIT.md` §0a).
  Batches become width-correlated, a small training-dynamics change. Keep `--white-pad-prob 0` with it.
- Attention goes through diffusers' dispatcher (`--attention-backend`, default torch SDPA). On the GB10 the
  default SDPA was the fastest correct backend (`scripts/bench_attention.py`).
- On the GB10, `--compile-mode max-autotune-no-cudagraphs` can't autotune the GEMMs (Inductor: "not enough
  SMs"); use plain `--compile`. `--compile-vae` also compiles the VAE (the live encode).

**Guidance dropouts** (per sample): `--cond-dropout` drops all conditioning (the full null, incl. the pooled
vector); `--text-dropout` drops the glyph tokens and keeps the style; `--ref-dropout` (canvas) drops the
reference pixels and its transcription and keeps the target text — a dropped reference leaves its canvas
columns pure noise.

**Losses and defaults.** Flow-matching MSE on the velocity with **ink-focal weighting on by default** (latent
cells covered by ink weigh up to 3×; `--no-ink-focal` turns it off). REPA (on; `--no-repa`) aligns block 8 to
DINOv3 features of the target line at its native size, weight 0.5, and stops after 40 % of `--steps`. AdamW
(fused, weight decay 1e-4), LR 1e-4 with 2000 warmup steps then cosine to 1e-6, grad-clip 1.0, EMA with warmup.
`--grad-accum N` counts optimizer steps, and the logged img/s includes it.

**Widths.** `--max-line-width` (default 1024) bounds memory; the model itself has no width limit (2D RoPE +
column-aligned glyph tokens). A wider line is squashed to the cap unless `--drop-wide-lines` skips it. In
canvas mode the reference is a different line by the same writer; it prefers one no wider than
`--max-ref-width`, and a wider one is cropped and its transcription dropped.

> `train.jsonl` must have **at least `batch × num_processes` lines** (`drop_last=True`; a smaller set would
> yield 0 batches and hang DDP). The trainer fails fast with a clear message.

## 5. Monitor (trackio)
`--track` (needs `uv sync --extra track`) logs to project `diffu` under the run name = the `--out-dir`
basename: loss / flow_loss / repa_loss / lr / images_per_s, val_loss, gen_cer / ctrl_gap / gen_cer_cfg and the
sample sheets, plus the run's full config. `--track-space user/name` syncs it to a **private** Space. Alerts:
non-finite loss (ERROR, and the run stops), val loss rising over 3 consecutive evals (WARN), checkpoint saved
(INFO).
```bash
uv run trackio show --project diffu
uv run trackio list alerts --project diffu --json --since 2026-10-19T00:00:00
uv run trackio get metric --project diffu --run canvas --metric val_loss --json
```
**Held-out signals** (rank 0): `val_loss` every `--val-every` (500) steps over 8 batches of `val.jsonl` (the
flow term, comparable to `flow_loss`). Every `--sample-every` (2000) steps 4 lines of a fixed val batch are
generated from the EMA weights and read back by the recognizer: `gen_cer` (unguided), `ctrl_gap` (CER against another
sample's text minus `gen_cer`; > 0 = it follows the text) and `gen_cer_cfg` (CFG scale 5). The default
recognizer is the Swedish `Riksarkivet/trocr-large-handwritten-hist-swe-3-char`; IAM is English, so pass
`--htr-recognizer-eval microsoft/trocr-base-handwritten`. `--no-cer` turns the gauge off.

**Profiling.** `--profile-steps N` profiles N real steps, writes `<out-dir>/profile_step_table.txt` and
`<out-dir>/trace.json`, then exits before training. Feed the trace to the PyTorch `profile-model` skill.
`--profile-stack` adds Python-line attribution (`trace.json.stacks.txt`).

## 6. Evaluate, generate, gate
- **Evaluate:** `diffu-eval --manifest data_out/val.jsonl` reports HTR CER over a manifest (real lines, or the
  synthetic manifest `diffu-synth` writes); `--htr` picks the recognizer.
- **Generate:** `diffu-generate --ckpt data_out/checkpoints/final/model.safetensors --style ref.png --text
  "Smörgåsbord"`. For a dino model, `--width` fixes the canvas width (no ink-crop); for a canvas model it sets
  the written length (the ink extent).
- **Bulk synth:** `diffu-synth --ckpt <weights> --hf-dataset Teklia/IAM-line --n 4000 --out runs/synth`.
- **Stage-0 VAE gate:** `make vae-gate OUT_DIR=data_out` (`diffu-vae-gate`): confirms å ä ö survive
  encode→decode on real ink before you trust the latent.
- **Tests:** `make test` (needs the dev extra).

## 7. Stage 3 — LoRA adapters (adapt a base to a new hand/collection)
`--lora` freezes the base and trains only a low-rank adapter on the backbone's attention projections
(peft `LoraConfig` on the SD3 MMDiT, a `PeftAdapterMixin`). A run writes `pytorch_lora_weights.safetensors`
instead of a ~17 GB state, so per-collection adapters are cheap to store and swap. Pair `--lora` with
`--init-from <base>` — LoRA on a random base learns nothing useful.

You can train straight off a **Hub dataset** (image + text) with `--hf-dataset` — no local manifest. Needs
`uv sync --extra hf`; it doesn't combine with `--bucket-width` / `--drop-wide-lines`.

```bash
# adapt a base checkpoint to IAM (English), rank-16 adapter
uv run diffu-train \
    --init-from data_out/checkpoints/final/model.safetensors \
    --hf-dataset Teklia/IAM-line \
    --lora --lora-rank 16 --lora-alpha 32 \
    --steps 3000 --warmup-steps 150 --batch-size 8 --out-dir runs/iam_lora \
    --htr-recognizer-eval microsoft/trocr-base-handwritten
# -> runs/iam_lora/final/pytorch_lora_weights.safetensors (~6.3M params at rank 16, ~24 MB)
```

Keep the warmup short on a short run: the default 2000 warmup steps left an earlier 4k-step adapter half
undertrained (`EXPERIMENTS.md`). `step_N/` holds the raw adapter, `final/` the EMA-applied one. Inference loads
the adapter onto the same base (`--lora` is a dir or a Hub id; `--lora-scale` in `0..1+`):

```bash
diffu-generate --ckpt data_out/checkpoints/final/model.safetensors \
    --lora runs/iam_lora/final --lora-scale 1.0 \
    --text "the quick brown fox" --style ref_line.png --out iam.png
```

or in Python: `DiffuPipeline.from_pretrained("data_out/checkpoints/final", lora="runs/iam_lora/final")`.
Knobs: `--lora-rank` (capacity), `--lora-alpha` (scale = alpha/r), `--lora-dropout`.

## Gotchas
- The Qwen-Image VAE is video-derived (5-D): each line runs as a one-frame video through a single-frame fast
  path (no frame cache, the causal 3-D convs as 2-D), measured 1.65× faster encode+decode on the GB10.
- If `accelerate` asks to be configured: the Makefile passes its flags explicitly, so `accelerate config` is
  not needed.
- `--hf-dataset Teklia/IAM-line` has no writer column: every style reference is a random other writer. For
  style training use `data_iam/` (section 3).
