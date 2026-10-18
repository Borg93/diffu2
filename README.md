# Diffu

A latent-diffusion model for Swedish (modern + historical) handwritten **line-image**
generation, to augment OCR/HTR training. Few-shot writer style, content-conditioned,
built on 2026 components, sampled with **classifier-free guidance**.

> Diffu is the package/code name.

## Quickstart (uv)

```bash
git clone https://github.com/AI-Riksarkivet/diffu.git && cd diffu
make install          # uv sync — runtime deps; torch 2.14.1 comes from the CUDA 13.0 (cu130) wheel index
make smoke-1gpu       # sanity: a few training steps on synthetic data, GPU 1 (no dataset needed)
make smoke-2gpu       # same, across GPU 1 + 2 (exercises the multi-GPU path)

make data XML_DIR=/path/to/xml IMAGE_DIR=/path/to/page_images OUT_DIR=data_out
make train-1gpu DATA_DIR=data_out      # Stage-1 training on GPU 1
make train-2gpu DATA_DIR=data_out      # Stage-1 training on GPU 1 + 2
```

`make help` lists every target. The train/smoke targets pin `CUDA_VISIBLE_DEVICES` (GPU 1, or 1+2)
so the busy GPU 0 is never touched — override with `GPUS_1=` / `GPUS_2=`. Full walkthrough:
[`docs/TRAINING.md`](docs/TRAINING.md); architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Inference — the released model

`exp_sd35_128/step_120000` (128 px, CFG-null fix) is published as
[`Gabriel/diffu_sd35_128`](https://huggingface.co/Gabriel/diffu_sd35_128) (released 2026-07-05;
~2–5 % CER read-back through the Riksarkivet HTR recogniser at `cfg_scale=5`). It supersedes the
earlier 64 px [`Gabriel/test_sd35`](https://huggingface.co/Gabriel/test_sd35).

```python
from diffu.pipeline import DiffuPipeline

pipe = DiffuPipeline.from_pretrained("Gabriel/diffu_sd35_128")     # or a local release dir
img = pipe("Göteborgs poliskammare", style="a_line_crop_of_the_target_hand.png", cfg_scale=5.0)
img.save("out.png")
```

- **Many lines / serving:** `DiffuPipeline.from_pretrained(..., compile_backbone=True)` — regional
  `torch.compile` (dynamic; one compile covers variable widths), ~0.37 s/line vs ~1 s eager after the
  one-time cold start.
- **Quantization:** measured a *loss* on this model (see `docs/PERF_AUDIT.md` §0f) — don't; bf16 + compile
  is the shipping config. `diffu.quantize` stays available for hardware where low-bit GEMMs win.
- CLI equivalent: `diffu-generate --text "…" --style ref.png --ckpt <weights> [--compile]`.
- Publish a new checkpoint: `uv run python scripts/export_release.py --ckpt <step_dir> --repo <you>/<name> --push`.

## In-canvas style + padded glyphs (new, opt-in)

Two flags fix why the released model ignores the style reference. Both need `--glyph-line`; old
checkpoints and commands are unchanged.

- `--style-canvas`: no style encoder. The reference line's clean latent sits **left of the target on one
  canvas** (an extra input channel), with its transcription in the glyph tokens; the model continues the
  line in the same hand, and the loss covers only the target columns (F5-TTS / DiffInk / Eruku infilling).
  `--ref-dropout 0.2` trains the "no reference" mode so sampling can use a separate `--style-cfg-scale`.
- `--glyph-pad`: glyph tokens cover each line's **real ink width**, the rest of the canvas stays blank,
  instead of being stretched over the whole canvas — glyph column j lands where letter j is written, and
  the extent works as a length control.

```bash
diffu-train --data-dir data_out --glyph-line --glyph-pad --style-canvas \
    --text-dropout 0.1 --ref-dropout 0.2 --cond-dropout 0.05
diffu-generate --ckpt <weights> --glyph-pad --style-canvas --line-height 128 \
    --style ref_line.png --style-text "what the ref line says" --text "Göteborgs poliskammare" \
    --cfg-scale 5 --style-cfg-scale 2
```

The reference must be the **same writer** as the target: lines from the same page (the `writer_id`
default) are fine. `Teklia/IAM-line` has **no writer column**, so its references are random other
writers — use an IAM source with writer ids to test style there.

## Architecture at a glance

| Part | Diffu |
|------|-------|
| Backbone | **DiT / MMDiT** (transformer), 2D-RoPE, variable-length lines |
| Objective | **Rectified flow / flow matching** |
| VAE | **16-ch f8 VAE** (FLUX.1/SD3 class, *not* FLUX.2) + decoder fine-tuned on ink |
| Content text | **glyph-line / GNU-Unifont glyph rendering** → ResNet tokens (Unicode-native å ä ö, historical glyphs) |
| Style | **DINOv3 ViT-L** features → resampler |
| Guidance | **Classifier-free guidance (CFG)** — `cfg_scale ≈ 5` at inference |
| Output | **variable-length 128px lines** (match Transkribus) |

## The one non-negotiable: diacritics must survive the VAE
On a 64px line an å-ring / ä-ö dot is sub-8px. SD1.5's 4-ch VAE (and FLUX.2's 64:1 VAE) smear it.
→ Use a **16-ch f8 VAE**, **fine-tune only the decoder** on Riksarkivet ink (encoder frozen), and
**gate**: VAE round-trip reconstruction CER on held-out real Swedish must be ≈ raw real CER
*before any diffusion training begins.*

## Eval = downstream HTR CER on held-out **real** Swedish (writer-disjoint). Not FID.

## Layout

```
src/diffu/
  config.py            # pydantic config for the whole model + data pipeline
  flow.py              # rectified-flow objective (interpolant, velocity target, loss)
  train.py             # Stage-1 training loop (REPA on by default, --no-repa to disable) + `diffu-train`
  smoketest.py         # synthetic forward/backward smoke + `diffu-smoke`
  generate.py          # text + style ref -> line image (CFG flow sampling) + `diffu-generate`
  pipeline.py          # DiffuPipeline.from_pretrained — the released-model inference wrapper
  eval.py              # HTR Character Error Rate over a manifest + `diffu-eval`
  recognizer.py        # Swedish HTR recogniser wrapper (CER read-back for eval)
  htr_train.py         # fine-tune / evaluate the HTR recogniser
  vae_gate.py          # VAE decoder recon-CER gate (stage 0, before any diffusion training)
  quantize.py          # optional low-bit quantization (bf16 + compile is the shipping config)
  profiling.py         # perf-profiling helpers
  lora.py              # Stage-3 LoRA adapters — add/save/load on the backbone (`diffu-train --lora`)
  model/
    diffu.py           # Diffu — assembles VAE + content + style + backbone; CFG flow sampler
    backbone.py        # diffusers SD3 MMDiT wrapper
    rope.py            # 2D-RoPE (shared-column, variable-length lines)
    vae.py             # 16-ch Qwen-Image VAE wrapper + recon-CER gate
    conditioning.py    # glyph-line width renderer · DINOv3 style + Perceiver resampler + attn-pool
    content_encoder.py # Unifont glyph content encoder
    metrics.py         # CER
  data/
    parse.py           # PAGE/ALTO XML -> line records (stdlib only, no torch)
    ingest.py          # scan+XML -> line-image dataset
    dataset.py         # manifest build + writer-disjoint split + torch Dataset
    prepare.py         # `diffu-data` CLI
    collate.py         # width-bucketing collate
    sampler.py         # width-bucket batch sampler
    latent_cache.py    # precomputed VAE-latent cache
    hf_dataset.py      # train off a Hub image+text dataset (`diffu-train --hf-dataset Teklia/IAM-line`)
  rl/                  # recognizer-reward finetuning scaffolding (Flow-OPD; see docs/RL_PLAN.md)
packages/
  diffu-page/          # whole-page composition: PageSpec templates, compositor, augmentation, HTR replay
  diffu-runner/        # Ray line-bank generation + CPU bank-mode page composition (the 1M-page path)
  diffu-studio/        # gradio control panel (single-line + whole-page synthesis, PiD upscaler)
docs/                  # STATUS.md, ARCHITECTURE.md, TRAINING.md, PERF_AUDIT.md, EXPERIMENTS.md, RL_PLAN.md
tests/                 # pytest suite (parse, dataset, collate, flow, metrics, eval)
```

Sources: SD3 rectified flow (arXiv 2403.03206) · FLUX.1 VAE 16-ch f8 · DINOv3 (arXiv 2508.10104) ·
Riksarkivet TrOCR (Swedish Lion Libre) · Quo Vadis (arXiv 2508.09936).
