# Diffu

Synthetic archive pages with ground truth, for training page-reading vision-language models (DeepSeek-OCR 2,
PaddleOCR-VL) and HTR: any real page layout (from a PAGE or ALTO file), any text, a chosen archive's hand, paper and
scan look. Every drawn line is read back by an HTR model, and the labels (PAGE or ALTO XML, and the page readers' own
formats) mark the lines that came out wrong. The pages are drawn by **MOSS-Image** (Apache-2.0: our own implementation
around its released weights, and our own training to teach it to draw from a sample page), the main path, or by
**FLUX.3 Klein** with our LoRAs, the research reference.

## The repo

A uv workspace: the root project and five members under `packages/`. Everything that draws is a Modular Diffusers
pipeline, our blocks around a model's own, the model chosen by its directory alone.

- **`src/diffu`** (diffu): what every member shares about a model directory: its `modular_model_index.json`
  (`diffu.repo.ModelIndex`) and loading its components without letting a failed one through
  (`diffu.modular_pipelines.loading`).
- **`packages/moss`** (diffu-moss): MOSS-Image: our DiT, its blocks, `diffu-moss-init` (a model directory), and its
  reference training (`train/`: `diffu-moss-cache-pairs`, `diffu-moss-train-ref`, a LoRA or every weight, on one GPU
  or sharded over several, and `diffu-moss-export-ref`). [docs/MOSS.md](docs/MOSS.md)
- **`packages/page`** (diffu-page): pages as data (layouts, crops, captions, labels, the training-set builders in
  `data/`), a template marked from its scan, our blocks and the page and crop pipelines; `diffu-generate`,
  `diffu-evaluate`, `diffu-draw-pairs`, `diffu-reader-set`. [docs/PIPELINE.md](docs/PIPELINE.md)
- **`packages/train`** (diffu-train): FLUX.3 LoRA training with NeMo AutoModel: the configs, our dataloaders and the
  style-pair cache (NeMo's environment). [docs/FLUX3.md](docs/FLUX3.md)
- **`packages/runner`** (diffu-runner): generation at scale on a Ray cluster. [docs/RAY.md](docs/RAY.md),
  [docs/INFERENCE.md](docs/INFERENCE.md) (KubeRay inference)
- **`packages/studio`** (diffu-studio): MOSS-Image models drawing the same page side by side on a `gr.Workflow` canvas.

```
jobs/         spark_guard.sh (one GPU job at a time on the DGX Spark, 128 GB shared by CPU and GPU),
              moss/ (MOSS-Image training on the Spark, a LoRA or every weight; a Slurm template for the HPC run),
              flux3/run_v3d.sh (FLUX.3's style-reference LoRA), setup_nemo_env.sh (the members in NeMo's environment)
docs/         PIPELINE.md, MOSS.md, FLUX3.md, RAY.md, INFERENCE.md
styles.json   style presets: the FLUX.3 hand LoRAs, each in its archive's look
conftest.py   test fixtures shared by the members (pages, a scanned form, diffusers' tiny FLUX.3, a tiny MOSS-Image
              and a tiny reference training on it)
TODO.md       open work
```

Package names keep the `diffu-` prefix (imports `diffu_page`, ...) so they stay unique in NeMo's shared environment.

## Environments

- **This repo** (`uv sync`): drawing with MOSS-Image, MOSS's training, CPU checks and tests
  (`CUDA_VISIBLE_DEVICES="" uv run pytest -n 4`; the pipeline tests draw with tiny random models on CPU).
- **NeMo AutoModel** (`~/Desktop/repos/Automodel/.venv-managed`): FLUX.3 LoRA training and FLUX.3's GPU jobs;
  `jobs/setup_nemo_env.sh` installs diffu, diffu-moss, diffu-page and diffu-train into it.
- `FLUX3_MODEL`: the FLUX.3 Klein directory, from the environment or a `.env` file; there is no default, so a command
  that needs a model and is given no `--model` stops before loading anything. `DIFFU_HTR_MODEL` and
  `DIFFU_HTR_REVISION` choose the read-back model and pin its revision.

## Drawing pages

```bash
BASE=~/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Image-Base/snapshots/85975793d8b3d432f1b6e27f352abaa4c2a4c0d3
uv run diffu-moss-init --base $BASE --transformer runs/moss_hand_1715/merged_step350 --out runs/moss/hand_1715
jobs/spark_guard.sh --need-gb 50 --name pages -- uv run diffu-generate --model runs/moss/hand_1715 \
    --templates page_a.xml --text-from text.txt --title "the records of Överståthållarämbetet" --paper yellowed \
    --ink brown --whole-page --page-megapixels 2 --reader-labels --out runs/pages
uv run diffu-reader-set --run runs/pages --model paddleocr-vl --out data/readers/paddleocr_vl.jsonl
```

Every option, job files and style presets: [docs/PIPELINE.md](docs/PIPELINE.md). Drawing with FLUX.3 and its LoRAs:
[docs/FLUX3.md](docs/FLUX3.md).
