---
title: Diffu Studio
emoji: ✒️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.19.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Swedish historical handwriting synthesis + HTR labels
---

# ✒️ Diffu Studio

A control panel for **Diffu** — an SD3.5 MMDiT handwriting generator for Swedish historical HTR data
augmentation. Type text, pick a writer's hand, and generate:

- **Single line** — watch the model denoise *noise → handwriting* live.
- **Whole page** — a full document page streamed **line by line** (each line denoises, then seats; the
  page grows top→bottom; then a scan/aging pass), with **PAGE / ALTO / JSON** ground-truth labels emitted
  alongside — ready for HTR training.
- **Every layout / furniture / aging knob** is a control; look/aging changes re-composite instantly (no
  re-generation).

Model: the Hub repo named by the **`DIFFU_CKPT_REPO`** Space variable (required) — `model.safetensors` +
the `config.json` describing its architecture (override the weights filename with `DIFFU_CKPT_FILE`),
downloaded at boot.

## ⚡ ZeroGPU + ahead-of-time compilation

This Space runs on **ZeroGPU** (H200). The MMDiT backbone is **AoT-compiled with
[`aokit`](https://github.com/huggingface/aokit)** on first model load (weightless `.pt2` package,
lazy-loaded per forked request, with a dynamic line-width shape). If the export doesn't like the model's
custom 2D-RoPE graph, it falls back — fixed-width AoT → JIT `compile_blocks` → eager — so the Space always
works; only the speedup is lost. See the startup logs for which path was taken.

> Note: this is the first ZeroGPU deploy of a custom (non-diffusers-pipeline) backbone; the AoT export is
> being validated/tuned on-Space. The base handwriting synthesis works regardless.
