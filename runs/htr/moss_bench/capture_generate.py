"""diffu-generate, with each page's prompt and its sample on the grid written down as it draws (for the SGLang runs).

Nothing of the drawing is changed: the two functions that make the prompt and size the sample are wrapped to save what
they return to CAPTURE_DIR (page_K.txt, page_K.png, page_K.json with the page's width and height).

    CAPTURE_DIR=<dir> python capture_generate.py <diffu-generate's arguments>
"""

import json
import os
from pathlib import Path

import diffu_moss.modular_pipelines.moss.encoders as encoders
import diffu_page.draw.blocks.crop as crop
from diffu_page.cli.generate import app

OUT = Path(os.environ["CAPTURE_DIR"])
OUT.mkdir(parents=True, exist_ok=True)
prompt_of, sample_of = encoders.sample_layout_prompt, crop.sized_sample
count = {"prompt": 0, "sample": 0}


def prompt(caption, texts, size, **more):
    made = prompt_of(caption, texts, size, **more)
    (OUT / f"page_{count['prompt']}.txt").write_text(made, encoding="utf-8")
    (OUT / f"page_{count['prompt']}.json").write_text(json.dumps({"width": size[0], "height": size[1]}))
    count["prompt"] += 1
    return made


def sample(image, rule, **more):
    made = sample_of(image, rule, **more)
    made.save(OUT / f"page_{count['sample']}.png")
    count["sample"] += 1
    return made


encoders.sample_layout_prompt, crop.sized_sample = prompt, sample
app()
