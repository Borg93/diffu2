# Diffu

Synthetic archive pages with ground truth, for training HTR and vision-language models: any page layout (from a
real PAGE/ALTO page or a template), any text, a chosen archive's hand, paper and scan look. Every generated line
is read back by an HTR model, and the labels (PAGE or ALTO XML) mark the lines that came out wrong.

Two parts:

- **The page pipeline** (`packages/page-synth`): layouts, crop planning, captions, labels, and the generator built
  from Modular Diffusers blocks. It runs today with **FLUX.3 Klein** and our LoRAs, the research reference.
- **diffu, the model** (planned, see [TODO.md](TODO.md)): FLUX.3's architecture with our own weights, trained only
  on our data, so the pipeline can run on a model we may ship. The old from-scratch line model is under the git tag
  `diffu-mmdit-v1`.

How the pieces fit together, and what keeps training and generation consistent: [docs/PIPELINE.md](docs/PIPELINE.md).

## Layout

The package follows the pipeline: a page **layout** is cut into **crops**, each crop gets a **caption** (its prompt),
the **modular** pipelines draw it, read every line back and write its **labels**.

```
packages/page-synth/src/page_synth/
  layout/       a real PAGE/ALTO page as its lines (page_xml.py: the one XML reader, shared with the training-crop
                builder), and refill: new text in its running-text lines
  crops/        geometry.py (the crop windows), grid.py (the 16 px latent grid)
  plan.py       a page as CropPlans: the crops training cut, each with its caption
  captions/     prompt.py (the FLUX.3 layout prompt), look.py, provenance.py, records.py, restyle.py
  modular/      blocks.py (our Modular Diffusers blocks), pipelines.py (CropTextBlocks, CropBlocks, loading, LoRAs),
                readback.py (TrOCR read-back and CER)
  generator.py  CropGenerator: one crop plan in, one image + labels out
  labels.py     the label data and its PAGE / ALTO exporters
  cli/          generate.py, evaluate.py, sheets.py (also the page-synth-generate / -evaluate / -sheets commands)
  data/         training and evaluation sets (crops, steering set, v3 reference pairs), NeMo caching and dataloaders
  settings.py   typed settings from the environment (FLUX3_MODEL: the FLUX.3 model directory)
configs/flux3/  NeMo AutoModel training configs
jobs/           GPU job scripts (flux3/), spark_guard.sh (one GPU job at a time on the DGX Spark, 128 GB unified
                memory), setup_nemo_env.sh (installs page-synth into NeMo's environment)
```

## Environments

- **This repo** (`uv sync`): page-synth with torch, diffusers (pinned to the FLUX.3 commit), transformers. CPU
  checks and tests: `CUDA_VISIBLE_DEVICES="" uv run pytest`. The generator tests run diffusers' tiny FLUX.3 test
  model end to end on CPU.
- **NeMo AutoModel** (`~/Desktop/repos/Automodel/.venv-managed`): training and the GPU jobs.
  `jobs/setup_nemo_env.sh` installs page-synth into it.

## Generate

```bash
cd ~/Desktop/repos/Automodel && ~/Desktop/repos/diffu2/jobs/spark_guard.sh --need-gb 45 --name pages -- \
    .venv-managed/bin/page-synth-generate \
    --templates page_a.xml --templates page_b.xml --text-from other_page.xml \
    --lora v2=runs/flux3_layout_lora_v2/epoch_0_step_1599/model --out runs/pages \
    --title "the court records of Jämtlands domsaga" --paper cream --ink "dark brown"
```

In Python:

```python
from pathlib import Path

from page_synth.captions.look import ScanLook
from page_synth.captions.provenance import Provenance
from page_synth.generator import CropGenerator, GeneratorSettings
from page_synth.layout import load_xml_layout, refill
from page_synth.plan import PageStyle, plan_page

style = PageStyle(
    provenance=Provenance(title="the court records of Jämtlands domsaga", years="1690–1700"),
    look=ScanLook(greyscale=False, paper="cream", ink="dark brown"),
)
generator = CropGenerator(model_dir, loras={"v2": lora_dir}, settings=GeneratorSettings(steps=50))
for plan in plan_page(refill(load_xml_layout(Path("page.xml")), [text]), style):
    crop = generator(plan, seed=0, image_name="crop.jpg")  # crop.image, crop.labels_xml, crop.line_cers
```
