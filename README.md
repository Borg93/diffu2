# Diffu

A latent-diffusion model for Swedish (modern + historical) handwritten **line-image** generation, to augment
OCR/HTR training. It writes a given text in the hand of one reference line, is built on 2026 components, and
samples with **classifier-free guidance**.

> Diffu is the package/code name.

## Quickstart (uv)

```bash
git clone https://github.com/Borg93/diffu2.git && cd diffu2
make install          # uv sync — runtime deps; torch 2.14.1 comes from the CUDA 13.0 (cu130) wheel index
hf auth login         # DINOv3 is gated (REPA target + dino-mode style encoder): accept its licence on the Hub
make smoke-1gpu       # sanity: a few training steps on synthetic data (no dataset needed)
make smoke-compile    # the same through the regionally compiled backbone
make smoke-canvas     # the in-canvas style recipe (--style-canvas --glyph-pad)

make data XML_DIR=/path/to/xml IMAGE_DIR=/path/to/page_images OUT_DIR=data_out
hf download Riksarkivet/trocr-large-handwritten-hist-swe-3-char   # the live gen_CER recognizer (local cache only)
make train OUT_DIR=data_out    # Stage-1 training on data_out/train.jsonl -> data_out/checkpoints/
```

The Makefile defaults fit the single-GPU DGX Spark: every target runs on GPU 0 with one process (`make train`
adds `--multi_gpu` only when `NPROC>1`). On a multi-GPU host set `GPUS=0,1 NPROC=2` for `make train`, or
`GPUS_1=` / `GPUS_2=` for the `*-1gpu` / `*-2gpu` targets. `make help` lists every target. The train targets pass
only data dir, steps, LR and batch size; for architecture or speed flags call `uv run diffu-train` directly.
Optional extras: `uv sync --extra track` (trackio), `--extra hf` (Hub datasets), `--extra dev` (ruff / pyrefly /
pytest; `make install-dev`) — list all the extras you want in one `uv sync`, since a plain `uv sync` removes
the others.

Where things stand and what to run next: [`docs/HANDOFF.md`](docs/HANDOFF.md). Full training walkthrough:
[`docs/TRAINING.md`](docs/TRAINING.md). Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
(plain words: [`docs/OVERVIEW.md`](docs/OVERVIEW.md)).

## Inference

There is no downloadable model at the moment. The released `Gabriel/diffu_sd35_128` (128 px, released
2026-07-05) is no longer on the Hub, and its `step_120000` checkpoint stayed on the old training box. Load a
checkpoint you trained:

```python
from diffu.pipeline import DiffuPipeline

pipe = DiffuPipeline.from_pretrained("runs/dino/final")   # a dir (or Hub repo) with config.json + model.safetensors
img = pipe("Göteborgs poliskammare", style="a_line_crop_of_the_target_hand.png", cfg_scale=5.0)
img.save("out.png")
# a --style-canvas model also takes the reference's transcription and a style-guidance scale:
canvas = DiffuPipeline.from_pretrained("runs/canvas/final")
img = canvas("Göteborgs poliskammare", style="ref.png", style_text="what ref.png says", style_cfg_scale=2.0)
```

- A `step_N/` dir loads with `DiffuPipeline.from_pretrained("runs/canvas/step_N", weights_name="ema_model.safetensors")`.
- CLI: `diffu-generate --ckpt runs/canvas/step_N/ema_model.safetensors --style ref.png --text "…"` (plus
  `--style-text "…" --style-cfg-scale 2` for a canvas model; `--cfg-scale` defaults to 5). The architecture
  comes from the `config.json` next to the weights; architecture flags override it, and a checkpoint saved
  without a `config.json` needs the flags it was trained with.
- **Many lines / serving:** `compile_backbone=True` (CLI `--compile`): regional `torch.compile` with
  `dynamic=True`, so one compile covers every line width. On the old RTX PRO 6000 box it measured ~0.37 s/line
  vs ~1 s eager after the one-time cold start (`docs/PERF_AUDIT.md` §0f); not measured on the GB10 yet.
- **Exact speed-ups (diffusers' SD3 recommendations):** fused Q/K/V projections are on by default
  (`fuse_qkv=True`, skipped with a LoRA); `compile_vae=True` (CLI `--compile-vae`) compiles the VAE;
  `compile_mode="reduce-overhead"` / `"max-autotune"` (CLI `--compile-mode`) add CUDA graphs. On the GB10
  max-autotune can't autotune the GEMMs (Inductor: "not enough SMs"), so use plain compile there.
- **Quantization** measured a *loss* on this model (`docs/PERF_AUDIT.md` §0f); `diffu.quantize` was removed
  on 2026-10-19. bf16 + compile is the serving config.
- Publish a checkpoint: `uv run python scripts/export_release.py --ckpt <step_dir> --repo <you>/<name> --push`.

## Two style modes, and padded glyphs

- **dino** (the default; every model trained so far): a frozen DINOv3 ViT-L reads the reference crop squashed
  to 224×224 → Perceiver resampler → one attention-pooled vector that, with the timestep, drives the AdaLN
  modulation of every block. Style is pooled-only: spatial style tokens in the attention carried the
  reference's letter shapes, and the model copied them instead of writing the requested text.
- **canvas** (`--style-canvas`; implemented and unit-tested, **not trained yet**): the reference line's clean
  latent sits **left of the target on one canvas** (extra input channels), with its transcription in the glyph
  tokens; the model continues the line in the same hand and the loss covers only the target columns (F5-TTS /
  DiffInk / Eruku infilling). `--ref-dropout` trains the "no reference" mode that sampling's separate
  `--style-cfg-scale` contrasts against.

`--glyph-pad` (either mode) places each line's glyph tokens over its **own ink extent** and leaves the rest of
the canvas blank, instead of stretching them over the whole canvas: glyph column j lands where letter j is
written, and the extent works as a length control.

```bash
uv run diffu-train --data-dir data_iam --out-dir runs/canvas --style-canvas --glyph-pad \
    --text-dropout 0.1 --ref-dropout 0.2 --cond-dropout 0.05 \
    --max-line-width 3072 --max-ref-width 2048 --drop-wide-lines \
    --bucket-width --no-grad-checkpoint --compile --batch-size 8 \
    --htr-recognizer-eval microsoft/trocr-base-handwritten --track
uv run diffu-generate --ckpt runs/canvas/step_N/ema_model.safetensors \
    --style ref.png --style-text "what ref.png says" --text "Göteborgs poliskammare" \
    --cfg-scale 5 --style-cfg-scale 2
```

`--track` needs `uv sync --extra track`, and the live gauge's recognizer loads from the local Hugging Face cache
only (`hf download microsoft/trocr-base-handwritten` first). The full run order is in
[`docs/HANDOFF.md`](docs/HANDOFF.md) §3.

The reference must be the **same writer** as the target. In the Swedish PAGE-XML data a line's `writer_id` is
its page, so references come from the same page. `data_iam/` (built by `scripts/prepare_iam_writers.py`) carries
real IAM writer ids. `--hf-dataset Teklia/IAM-line` has no writer column, so there every reference is a random
other writer.

## Architecture at a glance

| Part | Diffu |
|------|-------|
| Backbone | **DiT / MMDiT** (diffusers SD3, ~900M params, SD3.5 qk-norm), 2D-RoPE, no width cap |
| Objective | **Rectified flow / flow matching** (logit-normal t), ink-focal loss weighting, REPA for the first 40 % of steps |
| VAE | **16-ch f8 Qwen-Image VAE**, frozen (FLUX.1/SD3 class, *not* FLUX.2) |
| Content text | **Unifont glyph line**: the text rendered with GNU Unifont → ResNet-18 → one token per patch column, shared-column RoPE (Unicode-native å ä ö, historical glyphs) |
| Style | **DINOv3 ViT-L → resampler → pooled AdaLN vector** (dino), or the reference line on the canvas (canvas) |
| Guidance | **Classifier-free guidance** on the text (`cfg_scale` 5 by default), + style guidance in canvas mode |
| Output | **variable-width 128 px lines** |

## The one non-negotiable: diacritics must survive the VAE
On a 64 px line an å-ring / ä-ö dot is sub-8px. SD1.5's 4-ch VAE (and FLUX.2's 64:1 VAE) smear it.
→ Use a **16-ch f8 VAE**, keep it **frozen**, and **gate**: the VAE round-trip reconstruction CER on held-out
real lines must stay within +0.05 of the raw CER *before any diffusion training begins* (`make vae-gate`; the
Qwen-Image VAE measured +0.008).

## Eval = downstream HTR CER on held-out **real** Swedish (writer-disjoint). Not FID.

The live gen_CER during training, `diffu-eval` and `diffu-vae-gate` read lines with the char-level
historical-Swedish TrOCR `Riksarkivet/trocr-large-handwritten-hist-swe-3-char` (`cfg.aux.htr_recognizer_eval`;
`--htr-recognizer-eval` / `--htr` swap it, e.g. for English). They load it from the local Hugging Face cache
only, so `hf download` it once first.

## Layout

```
src/diffu/
  config.py            # pydantic config + the architecture flags shared by every entry point (add_arch_args)
  flow.py              # rectified-flow objective (interpolant, velocity target, loss, REPA cosine loss)
  train.py             # Stage-1 training loop + `diffu-train` (checkpoints, EMA, trackio, live gen_CER)
  smoketest.py         # synthetic forward/backward smoke + `diffu-smoke`
  generate.py          # text + style ref -> line image (CFG flow sampling), the shared loader + `diffu-generate`
  pipeline.py          # DiffuPipeline.from_pretrained — the one-call inference wrapper
  synth.py             # bulk-generate a synthetic line manifest + `diffu-synth`
  eval.py              # HTR Character Error Rate over a manifest + `diffu-eval`
  recognizer.py        # the HTR recognizer wrapper (CER read-back for the live gauge)
  htr_train.py         # fine-tune a TrOCR recognizer + `diffu-htr-train` (the HTR A/B)
  vae_gate.py          # VAE recon-CER gate (stage 0, before any diffusion training) + `diffu-vae-gate`
  profiling.py         # profiler report + trace behind `diffu-train --profile-steps`
  lora.py              # Stage-3 LoRA adapters — add/save/load on the backbone (`diffu-train --lora`)
  model/
    diffu.py           # Diffu — assembles VAE + content + style/canvas + backbone; the CFG flow sampler
    backbone.py        # diffusers SD3 MMDiT wrapper (2D-RoPE, attention backend, fused QKV, regional compile)
    rope.py            # 2D-RoPE + shared-column content RoPE joint-attention processor
    vae.py             # 16-ch Qwen-Image VAE wrapper (frozen, single-frame fast path) + recon-CER gate
    conditioning.py    # text-width estimator · DINOv3 style encoder + Perceiver resampler + attention-pool
    content_encoder.py # Unifont glyph-line content encoder
    metrics.py         # CER
  data/
    parse.py           # PAGE/ALTO XML -> line records (stdlib only, no torch)
    ingest.py          # pre-cropped lines -> manifest + split (`diffu-ingest`)
    dataset.py         # line crops, manifest, writer-disjoint split, torch Dataset
    prepare.py         # `diffu-data` CLI
    collate.py         # padding collate (+ the canvas reference)
    sampler.py         # width-bucket batch sampler (`--bucket-width`)
    hf_dataset.py      # train off a Hub image+text dataset (`diffu-train --hf-dataset Teklia/IAM-line`)
scripts/               # source-specific data recipes (prepare_*), benchmarks, galleries, release export
slurm/                 # the same training as Slurm jobs on CINECA Leonardo (setup, smoke, chained train)
packages/
  diffu-page/          # whole-page composition: PageSpec templates, compositor, augmentation, HTR replay
  diffu-runner/        # Ray line-bank generation + CPU bank-mode page composition (the 1M-page path)
  diffu-studio/        # gradio control panel (single-line + whole-page synthesis, PiD upscaler)
docs/                  # HANDOFF (current state), OVERVIEW, ARCHITECTURE (+ architecture.html), TRAINING;
                       # dated records: STATUS, PERF_AUDIT, EXPERIMENTS, RL_PLAN; scratchpad/ (frozen notes)
tests/                 # pytest suite (data, collate/sampler, flow, metrics, eval, canvas, LoRA, VAE, attention)
```

Sources: SD3 rectified flow (arXiv 2403.03206) · FLUX.1 VAE 16-ch f8 · DINOv3 (arXiv 2508.10104) ·
Riksarkivet TrOCR (Swedish Lion Libre) · Quo Vadis (arXiv 2508.09936).
