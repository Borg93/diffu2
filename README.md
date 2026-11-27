⚠️ Papers are incoming! ⚠️

# Diffu

Synthetic archive pages with ground truth, to train page-reading vision-language models end to end (Unlimited-OCR 2,
with MiniCPM-OCR as the crop control): millions of whole pages, each from a layout (a PAGE or ALTO file, or boxes), a
text and a sample page whose paper, scan and pen it takes. Every page comes with its labels (ALTO and PAGE XML, and
the page readers' own formats); every drawn line is read back by an HTR model and the labels mark the lines that came
out wrong.

The pages are drawn by **MOSS-Image** (Apache-2.0): our own implementation around its released weights, and our own
full fine-tune of its DiT.

## What works today

- **The model**: [Gabriel/diffu-moss-above](https://huggingface.co/Gabriel/diffu-moss-above), a full fine-tune of
  MOSS-Image's DiT on Swedish archival handwriting of the 16th to 19th centuries. It draws a whole page at 2 MP from a
  layout (every line's exact text and box) and a sample page whose tokens are placed above the page's. The text is
  written as given, the paper and the kind of scan follow the sample, the ink and the pen in part; the hand's letter
  forms mostly do not follow yet. What was measured: [docs/MOSS_RUNS.md](docs/MOSS_RUNS.md).
- **The demo Space**: [Gabriel/diffu-archive-pages](https://huggingface.co/spaces/Gabriel/diffu-archive-pages)
  (Gradio on ZeroGPU, private while it is tested), its source in `packages/studio/space`: a page editor of boxes and
  text, a sample page, the page drawn and read back, its ALTO and PAGE labels returned.
- **Distillation** (`diffu-moss-distill`): LoRA students that draw a page in 8 steps without guidance; with `--swap` a
  share of the samples come from another page. The Space's fast mode draws with the first student.
- **Faster drawing**, each an option of the draw settings and off by default: compiled joint blocks, MXFP8 (Blackwell
  GPUs), AOTInductor packages (`diffu-moss-aoti`), step caches and the UniPC sampler.
  [docs/INFERENCE.md](docs/INFERENCE.md), section 12.
- **The readers** (`packages/htr_train`): training and scoring the page and line readers the pages are made for, and
  curating real scans with machine-made labels into training sets (`packages/curate`).

Not done yet: generation at scale. `packages/runner` (Ray Data, Lance) is written and waits until testing on one
machine is finished. Open work: [TODO.md](TODO.md).

## The repo

A uv workspace of seven members under `packages/` (the root is the workspace alone), and one package outside it.
Everything that draws is a Modular Diffusers pipeline: our blocks around a model's own.

- **`packages/core`** (diffu-core): what every member shares about a model directory: its model index and loading its
  components.
- **`packages/moss`** (diffu-moss): MOSS-Image: our DiT and blocks, `diffu-moss-init` (a model directory), the
  reference training (`diffu-moss-cache-pairs`, `diffu-moss-train-ref`, `diffu-moss-export-ref`), `diffu-moss-distill`,
  hand LoRAs as PEFT adapters (`diffu-moss-convert-lora`), the speed options, `diffu-moss-aoti` and
  `diffu-moss-export-hf`. [docs/MOSS.md](docs/MOSS.md)
- **`packages/page`** (diffu-page): a page as data (layouts, crops, labels), captions, read-back and the judge, the
  page and crop pipelines, the set builders; `diffu-generate`, `diffu-evaluate`, `diffu-draw-pairs`,
  `diffu-reader-set`. [docs/PIPELINE.md](docs/PIPELINE.md)
- **`packages/studio`** (diffu-studio): `diffu-studio`, MOSS-Image models drawing the same page side by side on a
  Gradio canvas, and the demo Space (`space/`, `publish_space.sh`).
- **`packages/runner`** (diffu-runner): `diffu-run` and `diffu-run-export`, pages drawn on a Ray cluster into a Lance
  table. [docs/RAY.md](docs/RAY.md), [docs/INFERENCE.md](docs/INFERENCE.md)
- **`packages/htr_train`** (diffu-htr-train): `htr-train` and `htr-eval`: Unlimited-OCR 2, MiniCPM-OCR, its
  proof-reader Jev and a DSpark draft, each tool in its own environment. [docs/HTR_TRAIN.md](docs/HTR_TRAIN.md),
  the runs in [docs/HTR_RUNS.md](docs/HTR_RUNS.md)
- **`packages/curate`** (diffu-curate): `diffu-curate`, the pools of old-labelled scans scored page by page in a Lance
  table and sampled into training sets. [docs/CURATION.md](docs/CURATION.md)
- **`packages/sglang_moss`** (diffu-sglang-moss, not a workspace member: its own environment): a plugin that lets
  SGLang Diffusion's MOSS-Image draw from a sample page. [docs/INFERENCE.md](docs/INFERENCE.md), section 13

```
jobs/         spark_guard.sh (one GPU job at a time on the DGX Spark, 128 GB shared by CPU and GPU),
              moss/ (MOSS-Image training on the Spark; a Slurm example for the HPC run)
deploy/       kuberay/ (the manifests of docs/INFERENCE.md)
docs/         MOSS.md (the model and its training), MOSS_RUNS.md (the experiment log), MOSS_AUDIT.md (why the hand
              did not follow, 11 Nov), MOSS_HPC.md (the HPC runbook), PIPELINE.md, INFERENCE.md, RAY.md,
              HTR_TRAIN.md (with HTR_RUNS.md, HTR_HANDOVER.md), CURATION.md
conftest.py   test fixtures shared by the members (pages, a scanned form, tiny random models)
TODO.md       open work
```

`uv sync --all-packages` installs the workspace; the tests run on CPU with tiny random models
(`CUDA_VISIBLE_DEVICES="" uv run pytest -n 4`). `DIFFU_HTR_MODEL` and `DIFFU_HTR_REVISION` choose the read-back model
and pin its revision.

## Drawing a page

The quickest way is the demo Space. On a machine with a GPU, once (the fine-tuned DiT, with MOSS-Image-Base's text
encoder, tokenizer and VAE around it):

```bash
uv sync --all-packages
hf download OpenMOSS-Team/MOSS-Image-Base --revision 85975793d8b3d432f1b6e27f352abaa4c2a4c0d3 \
    --include "text_encoder/*" --include "tokenizer/*" --include "vae/*" --local-dir models/MOSS-Image-Base
hf download Gabriel/diffu-moss-above --revision 3d5d3d966a0d32d4b9573c3d6a6d2324d19bb5a8 \
    --local-dir models/dit/diffu-moss-above
uv run diffu-moss-init --base models/MOSS-Image-Base --transformer models/dit/diffu-moss-above --out models/moss-above
```

A run file holds one page job per line, as JSON. A page's layout is a PAGE or ALTO file (`template`) or its lines'
boxes in pixels (`layout`); its text is the `text` given, poured into the running-text lines, or, left out, the
layout's own. `page_pixels` draws the page whole at that size, and `reference` names the sample page. One line of
`run.jsonl`, a layout of boxes with new text (its heading keeps its own):

```json
{"id": "boxes-0001", "layout": {"width": 1184, "height": 1680, "lines": [{"box": [130, 170, 1050, 226], "kind": "heading", "text": "Anno 1650"}, {"box": [130, 270, 1050, 326]}, {"box": [130, 340, 1050, 396]}]}, "text": "Tå laga tingh höltz medh Allmogen ...", "style": {"provenance": {"title": "the court records of Jämtlands domsaga", "years": "1650"}, "look": {"greyscale": false, "paper": "cream", "ink": "black"}}, "reference": {"image": "/data/samples/court.jpg", "caption": "sample"}, "page_pixels": 2000000}
```

Then draw; each page is written as an image with its labels, and `summary.json` lists them:

```bash
uv run diffu-generate --model models/moss-above --jobs run.jsonl --steps 28 --guidance 2 --out out/run1
```

28 steps at guidance 2 are the Space's guided settings (at MOSS-Image's own guidance 3 the ink comes out darker than
the sample's). On the DGX Spark every GPU command goes through `jobs/spark_guard.sh`. Every field of a page job and
of the draw settings: [docs/INFERENCE.md](docs/INFERENCE.md), sections 5 and 6; every option of `diffu-generate`, and
`diffu-reader-set`, which writes pages as fine-tuning sets in the page readers' formats:
[docs/PIPELINE.md](docs/PIPELINE.md).

## Earlier models

- [Gabriel/diffu-moss-r6b](https://huggingface.co/Gabriel/diffu-moss-r6b) (run B, 1 MP,
  the sample laid over the page: the look follows the sample, the hand does not) and the four hand adapters
  (`Gabriel/diffu-moss-hand-*`, PEFT adapters of MOSS-Image-Base's DiT, `--lora hand=<folder>`:
  [docs/MOSS.md](docs/MOSS.md), "Hand LoRAs"). Their runs are one line each under "Superseded runs" in
  [docs/MOSS_RUNS.md](docs/MOSS_RUNS.md).
