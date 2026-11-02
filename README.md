# Diffu

Synthetic archive pages with ground truth, for training page-reading vision-language models (DeepSeek-OCR 2,
PaddleOCR-VL) and HTR: any real page layout (from a PAGE or ALTO file), any text, a chosen archive's hand, paper and
scan look. Every drawn line is read back by an HTR model, and the labels (PAGE or ALTO XML, and the page readers' own
formats) mark the lines that came out wrong. The pages are drawn by **MOSS-Image** (Apache-2.0: our own implementation
around its released weights, and our own training to teach it to draw from a sample page), the main path, or by
**FLUX.3 Klein** with our LoRAs, the research reference.

## The repo

A uv workspace of six members under `packages/`; the root is the workspace alone (no package of its own). Everything
that draws is a Modular Diffusers pipeline, our blocks around a model's own, the model chosen by its directory alone.

- **`packages/core`** (diffu-core): what every member shares about a model directory: its `modular_model_index.json`
  (`diffu_core.repo.ModelIndex`) and loading its components without letting a failed one through
  (`diffu_core.loading`).
- **`packages/moss`** (diffu-moss): MOSS-Image: our DiT, its blocks, `diffu-moss-init` (a model directory), and its
  reference training (`train/`: `diffu-moss-cache-pairs`, `diffu-moss-train-ref`, a LoRA or every weight, on one GPU
  or sharded over several, and `diffu-moss-export-ref`). [docs/MOSS.md](docs/MOSS.md)
- **`packages/page`** (diffu-page), in layers: `page/` (a page as data: layouts, crops, labels, a template marked from
  its scan), `captions/`, `quality/` (read-back, the judge, hand fidelity), `draw/` (planning, our blocks, the page and
  crop pipelines) and `data/` (the training- and eval-set builders); `diffu-generate`, `diffu-evaluate`,
  `diffu-draw-pairs`, `diffu-reader-set` (`cli/`). [docs/PIPELINE.md](docs/PIPELINE.md)
- **`packages/flux3`** (diffu-flux3): FLUX.3 LoRA training with NeMo AutoModel: the configs, our dataloaders and the
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
  `jobs/setup_nemo_env.sh` installs diffu-core, diffu-moss, diffu-page and diffu-flux3 into it.
- `FLUX3_MODEL`: the FLUX.3 Klein directory, from the environment or a `.env` file; there is no default, so a command
  that needs a model and is given no `--model` stops before loading anything. `DIFFU_HTR_MODEL` and
  `DIFFU_HTR_REVISION` choose the read-back model and pin its revision.

## Drawing pages

The model is [Gabriel/diffu-moss-r6b](https://huggingface.co/Gabriel/diffu-moss-r6b) (public): our fine-tuned
MOSS-Image DiT. Its text encoder, tokenizer and VAE are MOSS-Image-Base's. Once:

```bash
uv sync --all-packages
hf download OpenMOSS-Team/MOSS-Image-Base --revision 85975793d8b3d432f1b6e27f352abaa4c2a4c0d3 \
    --include "text_encoder/*" --include "tokenizer/*" --include "vae/*" --local-dir models/MOSS-Image-Base
hf download Gabriel/diffu-moss-r6b --revision 2246c79aa65bf5a16409f185671fd2877efdb142 \
    --local-dir models/dit/diffu-moss-r6b
uv run diffu-moss-init --base models/MOSS-Image-Base --transformer models/dit/diffu-moss-r6b --out models/moss-r6b
```

A run file holds one page job per line, as JSON. A page's layout is a PAGE or ALTO file (`template`) or its lines'
boxes in pixels given as JSON (`layout`); its text is the `text` given, poured into the running-text lines, or, left
out, the layout's own. Two lines of `run.jsonl`, a template with new text and a layout of boxes with new text (its
heading keeps its own), each drawn in the look of a sample page:

```json
{"id": "court-0001", "template": "/data/templates/0105.xml", "text": "Anno 1649 den 6 Aprilis Tå laga tingh höltz medh Allmogen ...", "style": {"provenance": {"title": "the court records of Jämtlands domsaga", "years": "1649"}, "look": {"greyscale": false, "paper": "cream", "ink": "black"}}, "reference": {"image": "/data/samples/court.jpg", "caption": "sample"}}
{"id": "boxes-0001", "layout": {"width": 1400, "height": 2000, "lines": [{"box": [150, 200, 1250, 270], "kind": "heading", "text": "Anno 1650"}, {"box": [150, 320, 1250, 390]}, {"box": [150, 410, 1250, 480]}]}, "text": "Tå laga tingh höltz medh Allmogen ...", "style": {"provenance": {"title": "the court records of Jämtlands domsaga", "years": "1650"}, "look": {"greyscale": false, "paper": "cream", "ink": "black"}}, "reference": {"image": "/data/samples/court.jpg", "caption": "sample"}}
```

Then draw and write each page out as an image with its labels (`<id>.jpg`, `<id>.alto.xml`, `<id>.page.xml`):

```bash
echo '{"steps": 50, "device": "cuda"}' > draw.json  # Blackwell: add "precision": "mxfp8", "compile_blocks": true
uv run diffu-run --jobs run.jsonl --draw-settings draw.json --model models/moss-r6b --out out/run1 --gpus 1
uv run diffu-run-export --out out/run1 --to out/run1/files
```

Every field of a page job, with a worked example of each combination (template or layout, new text or its own), the
draw settings, outputs and running on KubeRay: [docs/INFERENCE.md](docs/INFERENCE.md). What the model does well and
what not yet: its [model card](https://huggingface.co/Gabriel/diffu-moss-r6b). Drawing on one machine without Ray, and
every option of `diffu-generate`: [docs/PIPELINE.md](docs/PIPELINE.md); its `--reader-labels` and `diffu-reader-set`
also write the labels in the page readers' own formats (DeepSeek-OCR 2, PaddleOCR-VL), which `diffu-run` does not yet.
Drawing with FLUX.3 and its LoRAs: [docs/FLUX3.md](docs/FLUX3.md).
