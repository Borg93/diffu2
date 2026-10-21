# Diffu

Synthetic archive pages with ground truth, for training HTR and vision-language models: any real page layout (from a
PAGE or ALTO file), any text, a chosen archive's hand, paper and scan look. Every generated line is read back by an
HTR model, and the labels (PAGE or ALTO XML) mark the lines that came out wrong. It runs today with **FLUX.3 Klein**
and our LoRAs (the research reference); **diffu** is our own model on FLUX.3's architecture, trained only on our data,
so the same pipeline can run on a model we may ship. The old from-scratch line model is under the git tag
`diffu-mmdit-v1`.

How the pieces fit together, and what keeps training and generation consistent: [docs/PIPELINE.md](docs/PIPELINE.md).

## Workspace

A uv workspace: one member per concern, each with its own dependencies.

| Member | What it is | Heavy dependencies |
| --- | --- | --- |
| `packages/diffu-page` | Archive pages as data: PAGE/ALTO reading (`layout/`), refill, crops (`crops/`), captions (`captions/`), plans, labels, the training-data builders (`data/`). CPU only. | none |
| `packages/diffu-gen` | Generation with Modular Diffusers (`modular/`: blocks, pipelines, read-back), `CropGenerator`, settings, the `diffu-generate` / `diffu-evaluate` / `diffu-sheets` CLIs | torch, diffusers, transformers |
| `packages/diffu-train` | NeMo AutoModel glue: dataloader configs, v3 pair caching | NeMo AutoModel (its environment only) |
| `packages/diffu` | The diffu model: FLUX.3's architecture with our own weights | torch, diffusers |
| `packages/diffu-runner` | Generation at scale on a Ray cluster: `CropGenerator`'s encode and draw stages as Ray Data actors | ray |
| `packages/diffu-studio` | A Gradio app: a template, a text and a style in, crops with their labels out | gradio |

```
configs/flux3/  NeMo AutoModel training configs
jobs/           GPU job scripts (flux3/), spark_guard.sh (one GPU job at a time on the DGX Spark, 128 GB unified
                memory), setup_nemo_env.sh (installs the members into NeMo's environment)
docs/           PIPELINE.md
conftest.py     test fixtures shared by the members (diffusers' tiny FLUX.3 test model)
```

## Environments

- **This repo** (`uv sync` builds every member): CPU checks and tests, `CUDA_VISIBLE_DEVICES="" uv run pytest`. The
  generation tests run diffusers' tiny FLUX.3 test model end to end on CPU.
- **NeMo AutoModel** (`~/Desktop/repos/Automodel/.venv-managed`): training and the GPU jobs.
  `jobs/setup_nemo_env.sh` installs diffu-page, diffu-gen and diffu-train into it.
- `FLUX3_MODEL`: the FLUX.3 Klein directory (default `~/Desktop/repos/flux3-models/flux-3-image-klein-base`).

## Generate

```bash
cd ~/Desktop/repos/Automodel && ~/Desktop/repos/diffu2/jobs/spark_guard.sh --need-gb 45 --name pages -- \
    .venv-managed/bin/diffu-generate \
    --templates page_a.xml --templates page_b.xml --text-from other_page.xml \
    --lora v2=runs/flux3_layout_lora_v2/epoch_0_step_1599/model --out runs/pages \
    --title "the court records of Jämtlands domsaga" --paper cream --ink "dark brown"
```

In Python:

```python
from pathlib import Path

from diffu_page.captions.look import ScanLook
from diffu_page.captions.provenance import Provenance
from diffu_gen.generator import CropGenerator, GeneratorSettings
from diffu_page.layout import load_xml_layout, refill
from diffu_page.plan import PageStyle, plan_page

style = PageStyle(
    provenance=Provenance(title="the court records of Jämtlands domsaga", years="1690–1700"),
    look=ScanLook(greyscale=False, paper="cream", ink="dark brown"),
)
generator = CropGenerator(model_dir, loras={"v2": lora_dir}, settings=GeneratorSettings(steps=50))
for plan in plan_page(refill(load_xml_layout(Path("page.xml")), [text]), style):
    crop = generator(plan, seed=0, image_name="crop.jpg")  # crop.image, crop.labels_xml, crop.line_cers
```
