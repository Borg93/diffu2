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

## Layout

```
packages/page-synth/src/page_synth/
  layout/              page layouts: replay a PAGE/ALTO page, templates, refill with new text
  labels.py            the label data and its PAGE / ALTO exporters
  geometry.py, plan.py the page cut into training-scale crops
  caption.py, look.py, provenance.py, records.py
                       the caption grammar (layout boxes + style) shared by training data and generation
  blocks.py            our Modular Diffusers blocks: crop prompt, HTR read-back, label export
  pipelines.py         the text pipeline and the crop pipeline over one ComponentsManager; LoRA loading
  generator.py         CropGenerator: one crop plan in, one image + labels out (the Ray Data actor unit)
  generate.py          CLI: pages from templates and text
  evaluate.py          CLI: held-out crops, read-back CER against the real crops
  sheets.py, restyle.py
  data/                training-data builders (crops, steering set, v3 reference pairs, NeMo caching)
  nemo.py              NeMo AutoModel dataloader configs (interleaved aspect buckets)
configs/flux3/         NeMo AutoModel training configs
jobs/flux3/            GPU job scripts; jobs/setup_nemo_env.sh installs page-synth into NeMo's environment
scripts/spark_guard.sh one GPU job at a time on the DGX Spark (128 GB unified memory)
```

## Environments

- **This repo** (`uv sync`): page-synth with torch, diffusers (pinned to the FLUX.3 commit), transformers. CPU
  checks and tests: `CUDA_VISIBLE_DEVICES="" uv run pytest`. The generator tests run diffusers' tiny FLUX.3 test
  model end to end on CPU.
- **NeMo AutoModel** (`~/Desktop/repos/Automodel/.venv-managed`): training and the GPU jobs.
  `jobs/setup_nemo_env.sh` installs page-synth into it.

## Generate

```bash
cd ~/Desktop/repos/Automodel && ~/Desktop/repos/diffu2/scripts/spark_guard.sh --need-gb 45 --name pages -- \
    .venv-managed/bin/python -m page_synth.generate \
    --templates page_a.xml page_b.xml --text-from other_page.xml \
    --lora v2=runs/flux3_layout_lora_v2/epoch_0_step_1599/model --out runs/pages \
    --title "the court records of Jämtlands domsaga" --paper cream --ink "dark brown"
```

In Python:

```python
from page_synth.generator import CropGenerator, GeneratorSettings
from page_synth.layout import load_xml_layout, refill
from page_synth.look import ScanLook
from page_synth.plan import PageStyle, plan_page
from page_synth.provenance import Provenance

style = PageStyle(
    provenance=Provenance(title="the court records of Jämtlands domsaga", years="1690–1700"),
    look=ScanLook(greyscale=False, paper="cream", ink="dark brown"),
)
generator = CropGenerator(model_dir, loras={"v2": lora_dir}, settings=GeneratorSettings(steps=50))
for plan in plan_page(refill(load_xml_layout("page.xml", single_page=False), [text]), style):
    crop = generator(plan, seed=0, image_name="crop.jpg")  # crop.image, crop.labels_xml, crop.line_cers
```
