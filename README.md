# Diffu

Synthetic archive pages with ground truth, for training HTR and vision-language models: any real page layout (from a
PAGE or ALTO file), any text, a chosen archive's hand, paper and scan look. Every generated line is read back by an
HTR model, and the labels (PAGE or ALTO XML) mark the lines that came out wrong. It runs today with **FLUX.3 Klein**
and our LoRAs (the research reference); **diffu** is our own model on FLUX.3's architecture, trained only on our data,
so the same pipeline can run on a model we may ship; **MOSS-Image** (Apache-2.0, our own implementation around its
released weights) writes every line's exact text in its box, and with a hand LoRA merged into it, the hand. The old
from-scratch line model is under the git tag `diffu-mmdit-v1`.

How the pieces fit together, and what keeps training and generation consistent: [docs/PIPELINE.md](docs/PIPELINE.md).

## Layout

The repo's root project is **diffu**, the model; the uv workspace members under `packages/` are MOSS-Image, the page
pipeline and what runs it. Everything that draws is a Modular Diffusers pipeline: our blocks (diffu-page) around the
model's own (FLUX.3 Klein from diffusers, diffu, or MOSS-Image), chosen by the model directory alone.

| Where | Package | What it is | Heavy dependencies |
| --- | --- | --- | --- |
| `src/diffu`, `tests/` | diffu | The model, laid out like a diffusers model (`modular_pipelines/diffu/`): FLUX.3's transformer with our own weights, FLUX.3's text encoder (Qwen3-VL-4B-Instruct) and the Qwen-Image VAE; `diffu-init` writes a new model repo, `diffu-export` a trained checkpoint's (sampled at the shift it trained at) | torch, diffusers |
| `packages/moss` | diffu-moss | MOSS-Image as the third model family: our DiT (`transformer.py`; the released weights load into it by name; reference images as FLUX.3 Klein takes them), its blocks (`modular_pipelines/moss/`: `MossAutoBlocks`, `MossModularPipeline`), a crop's caption compiled as MOSS's Layout prompt (`layout_prompt.py`); `diffu-moss-init` writes a MOSS model repo; our reference training (`train/`: `diffu-moss-cache-pairs`, `diffu-moss-train-ref`, `diffu-moss-export-ref`) | torch, diffusers, transformers, peft |
| `packages/page` | diffu-page | Archive pages as data (`layout/`, `crops/`, `captions/`, plans, labels, the training-data builders in `data/`), a template marked from its scan (`marking/`: printed lines and ruled lines), our blocks (`blocks/`), the page and crop pipelines (`pipeline.py`) and their CLIs: `diffu-generate`, `diffu-evaluate`, `diffu-sheets` | torch, diffusers, transformers, OpenCV |
| `packages/train` | diffu-train | NeMo AutoModel glue: the training configs (`configs/flux3/`, `configs/diffu/`), dataloader configs, the v3 pair cache and diffu's training cache | NeMo AutoModel (its environment only) |
| `packages/runner` | diffu-runner | Generation at scale on a Ray cluster: pages planned on CPU workers, crops drawn by GPU actors each holding a crop pipeline | ray |
| `packages/studio` | diffu-studio | The pipeline as a `gr.Workflow` canvas: a plan node feeding one draw node per drawer (a model and its LoRAs), so models draw the same page side by side; app view and REST endpoint included | gradio |

Package names keep the `diffu-` prefix (imports `diffu_page`, ...) so they stay unique in NeMo's shared environment.

```
jobs/           GPU job scripts (flux3/, diffu/), spark_guard.sh (one GPU job at a time on the DGX Spark, 128 GB unified
                memory), setup_nemo_env.sh (installs the members into NeMo's environment)
docs/           PIPELINE.md
conftest.py     test fixtures shared by the members (pages, a form beside its scan and a stand-in print/hand classifier,
                diffusers' tiny FLUX.3 test model, a tiny diffu repo, a tiny MOSS-Image repo)
```

## Environments

- **This repo** (`uv sync` builds diffu and every member): CPU checks and tests, `CUDA_VISIBLE_DEVICES="" uv run pytest -n 4`.
  The generation tests run diffusers' tiny FLUX.3 test model end to end on CPU.
- **NeMo AutoModel** (`~/Desktop/repos/Automodel/.venv-managed`): training and the GPU jobs.
  `jobs/setup_nemo_env.sh` installs diffu, diffu-moss, diffu-page and diffu-train into it.
- `FLUX3_MODEL`: the FLUX.3 Klein directory, from the environment or a `.env` file; there is no default, so a command
  that needs it and is given no `--model` stops before loading anything. `DIFFU_HTR_MODEL` and `DIFFU_HTR_REVISION`
  choose the read-back model and pin its revision.

## Generate

```bash
cd ~/Desktop/repos/Automodel && ~/Desktop/repos/diffu2/jobs/spark_guard.sh --need-gb 45 --name pages -- \
    .venv-managed/bin/diffu-generate \
    --templates page_a.xml --templates page_b.xml --text-from other_page.xml \
    --lora v2=runs/flux3_layout_lora_v2/epoch_0_step_1599/model --out runs/pages \
    --title "the court records of Jämtlands domsaga" --paper cream --ink "dark brown"
```

`--text-from` also takes a plain text file, written as it stands (a blank line ends a paragraph). `--whole-page`
draws each template whole (`--page-megapixels`, 2.5 by default) instead of as training-scale windows. A template that
is a two-page opening, as most GT scans are, is drawn as its single pages, never as a spread (`summary.json` lists
them; a job line can ask for one, `"side": "left"` or `"right"`). `--mark-from-scan` marks each template from the
scan beside it before planning (off by default): a GT PAGE file mostly says neither which lines are printed
(`@production`) nor where the ruled lines run (`SeparatorRegion`), so a printed form filled in by hand would be drawn
as handwriting without its grid. Each line the file says nothing of is read as printed or handwritten by a small CNN
(`--print-hand-model`, by default `runs/print_hand/print_hand.pt`; less than 90 % sure stays handwritten), and the
page's ruled lines are found on the scan if the file has none; `summary.json` names the classifier. It is trained on
CPU from the OCR and HTR ground truths and validated on collections it never saw (`report.json`; on 25,086 lines of 77
collections, 27 Oct: 89.8 % of the printed lines read as printed, 0.17 % of the handwritten ones):

```bash
OMP_NUM_THREADS=4 nice -n 10 uv run python -m diffu_page.data.print_hand_set \
    --printed ~/Desktop/data/GT-Lejonet-OCR-202606 --handwritten ~/Desktop/data/GT-Lejonet_HTR_202602 \
    --out runs/print_hand/lines.npz
OMP_NUM_THREADS=4 nice -n 10 uv run python -m diffu_page.marking.train --lines runs/print_hand/lines.npz \
    --out runs/print_hand
```

To give every page its own template, text, style, seed, reference and size, write them to a job file, one
`diffu_page.jobs.PageJob` per line (the run file diffu-runner reads), and pass `--jobs run.jsonl` instead of the page
options:

```json
{"template": "page_a.xml", "text": "Anno 1720 den 5 maj ...", "style": {"provenance": {"title": "the court records of Jämtlands domsaga", "years": "1720"}, "look": {"greyscale": false, "paper": "cream", "ink": "dark brown"}}, "page_pixels": 2500000, "id": "jamt_1720_p1"}
```

A style can also be named. `styles.json` holds a preset per hand LoRA (`diffu_page.styles.StylePreset`: the archive
and years, the look, the hand's LoRA and an optional reference, paths relative to the file); `--style` draws the
templates in one, in place of `--title` and the look options, by its hand alone:

```bash
cd ~/Desktop/repos/Automodel && ~/Desktop/repos/diffu2/jobs/spark_guard.sh --need-gb 45 --name pages -- \
    .venv-managed/bin/diffu-generate --templates page_a.xml --text-from text.txt \
    --styles ~/Desktop/repos/diffu2/styles.json --style pahlman --out runs/pages
```

With `--styles`, a job line's `style` may be a preset's name, so one run draws several hands, one pipeline per hand in
turn (`summary.json` lists each group's LoRAs and pages):

```json
{"template": "page_a.xml", "text": "Anno 1825 den 3 junii ...", "style": "pahlman", "id": "pahlman_p1"}
{"template": "page_b.xml", "text": "Anno 1743 den 12 martii ...", "style": "kopparberg", "id": "kopparberg_p1"}
```

`--fuse-loras` fuses the LoRAs into the transformer's weights at load: faster for a run that keeps one LoRA (the
studio keeps its adapters, to switch them per drawer). `--compile-blocks` then compiles the transformer's repeated
blocks (the first crop pays for the compilation). `--repair N` draws the lines the read-back rejects again in place, up
to N times, each pass from its own noise and kept only where it reads back; everything else in the crop stays as drawn
(on four whole 1715 pages one pass fixed 14 of 45 rejected lines and spoiled none; a pass costs about one drawing).
`summary.json` gives each crop's lines repaired.

### MOSS-Image

A MOSS-Image model directory is a `modular_model_index.json` naming `MossAutoBlocks` and the folders its components
load from; `diffu-moss-init` writes one from the Base snapshot in the Hugging Face cache, drawing with the snapshot's
own DiT or a trained one (`--transformer`: a hand LoRA merged into Base, or a full fine-tune, exported by
KohakuMossTrainer as a diffusers folder). Nothing is copied. Then `--model` draws with it as with FLUX.3, here in this
repo's environment, where MOSS was verified:

```bash
BASE=~/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Image-Base/snapshots/85975793d8b3d432f1b6e27f352abaa4c2a4c0d3
uv run diffu-moss-init --base $BASE --out runs/moss/base
uv run diffu-moss-init --base $BASE --transformer runs/moss_hand_1715/merged_step350 --out runs/moss/hand_1715
jobs/spark_guard.sh --need-gb 50 --name pages -- uv run diffu-generate --model runs/moss/hand_1715 \
    --templates page_a.xml --text-from text.txt --title "the records of Överståthållarämbetet" --paper yellowed \
    --ink brown --whole-page --page-megapixels 2 --out runs/pages_moss
```

Each crop's caption is compiled as MOSS's Layout prompt (the caption's prose as the summary, then a text item per line
with its exact text in its pixel box, its paper and ruled lines as objects); the prompt may take 4,096 tokens (a 44-47
line page takes 1,500-2,500), and a longer one is refused, not cut. MOSS draws at its own guidance (3) and sampling
shift (5) unless `--guidance` or `--sampling-shift` name others, as FLUX.3 draws at its 4 and 6.93. It takes no
`--lora` at draw time (merge the LoRA into the DiT folder the model directory names); `--repair` redraws its rejected
lines in place as FLUX.3's. It takes a `--reference` (a sample page of the same volume) only when its DiT was trained
to read one: the released models were not, so `diffu-moss-init` marks a directory `takes_reference` only for a DiT
exported by our reference training (`diffu-moss-train-ref`, `diffu-moss-export-ref`; the pilot is
`jobs/moss/run_ref_pilot.sh`), and the caption's citation of the reference then names "the sample image" as in
training. `diffu-evaluate --without-reference` draws a reference set's crops from their captions alone, beside a run
with them.

In Python, a page planned into crop jobs on CPU, then each crop drawn by the crop pipeline (a Modular Diffusers
pipeline), as diffu-generate and diffu-runner do it:

```python
from pathlib import Path

from diffu_page.captions.look import ScanLook
from diffu_page.captions.provenance import Provenance
from diffu_page.jobs import PageJob, crop_jobs
from diffu_page.pipeline import DrawSettings, crop_blocks, draw_crop, load_pipeline
from diffu_page.plan import PageStyle

style = PageStyle(
    provenance=Provenance(title="the court records of Jämtlands domsaga", years="1690–1700"),
    look=ScanLook(greyscale=False, paper="cream", ink="dark brown"),
)
settings = DrawSettings(steps=50)
pipe = load_pipeline(model_dir, crop_blocks, loras={"v2": lora_dir}, settings=settings)
page = PageJob(template=Path("page.xml"), text=text, style=style, seed=0)  # a blank line in text ends a paragraph
crops = [
    draw_crop(pipe, job.plan, name=job.name, seed=job.seed, reference=job.reference_image(), settings=settings)
    for job in crop_jobs(page, 0)
]
# each crop: .image, .labels_xml (ALTO), .line_reads, .line_cers, .unclear_lines
# with a reference-style LoRA, give the page the crop to draw in the hand of, captioned as that LoRA was trained:
#   PageJob(..., reference=Reference.of_crop(Path("crop.jpg"), ReferenceCaption.PLAIN))
#   (diffu_page.plan, diffu_page.captions.restyle)
```
