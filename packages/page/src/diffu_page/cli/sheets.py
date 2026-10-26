"""Contact sheets of a steering run: per template, every variant's generated crop beside a real crop of the archive
it was asked to look like.

Reads the steering set (``diffu_page.data.steer_set`` or ``style_pairs``' by_example) and the
``diffu_page.cli.evaluate`` run over it (``generated/NNN.png``, ``results.json``); writes ``steer_template_<t>.jpg``
into the run dir.

    diffu-sheets --steer data_flux3/steer_v2 --run runs/flux3_layout_v2/steer
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image, ImageDraw
from pydantic import BaseModel

from diffu_page.evalset import generated_path, read_jsonl
from diffu_page.output import JPEG_QUALITY, console, load_rgb, side_by_side

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

PANEL_HEIGHT = 420
LABEL_HEIGHT = 22
GAP = 10


class Prompt(BaseModel):
    """What a sheet needs of a set's prompt (a steering or by-example ``diffu_page.evalset.EvalCrop``): its template and
    variant, which an eval row may leave out and a steering row may not."""

    template: int
    variant: str
    example: str | None  # a real crop of the variant's style, relative to the set's dir


class Sample(BaseModel):
    """What a sheet needs of an evaluated crop: its index in the set and its read-back CER."""

    index: int
    gen_cer: float


class Results(BaseModel):
    """What a sheet needs of ``diffu_page.cli.evaluate``'s ``results.json``."""

    samples: list[Sample]


def label_of(prompt: Prompt, sample: Sample) -> str:
    """A row's label: the variant, its read-back CER, and what the panel beside it shows."""
    beside = "   | right: a real crop of that style" if prompt.example else ""
    return f"{prompt.variant}   (TrOCR CER {sample.gen_cer:.3f}){beside}"


def labelled_row(label: str, panels: list[Image.Image]) -> Image.Image:
    """The panels side by side at ``PANEL_HEIGHT``, under one line of label."""
    row = side_by_side(panels, height=PANEL_HEIGHT, gap=GAP, top=LABEL_HEIGHT)
    ImageDraw.Draw(row).text((4, 4), label, fill="black")
    return row


def stack(rows: list[Image.Image]) -> Image.Image:
    """The rows one under the other."""
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height + GAP for r in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + GAP
    return sheet


@app.command()
def sheets(
    steer: Annotated[Path, typer.Option(help="the steering set (diffu_page.data.steer_set)")],
    run: Annotated[Path, typer.Option(help="the diffu_page.cli.evaluate output dir of that set")],
) -> None:
    """Lay out one contact sheet per template of a steering run."""
    prompts = read_jsonl(steer / "metadata.jsonl", Prompt)
    results = Results.model_validate_json((run / "results.json").read_text(encoding="utf-8"))
    by_template: dict[int, list[Image.Image]] = defaultdict(list)
    for sample in results.samples:
        prompt = prompts[sample.index]
        panels = [load_rgb(generated_path(run, sample.index))]
        if prompt.example:
            panels.append(load_rgb(steer / prompt.example))
        by_template[prompt.template].append(labelled_row(label_of(prompt, sample), panels))
    for template, rows in by_template.items():
        out = run / f"steer_template_{template}.jpg"
        stack(rows).save(out, quality=JPEG_QUALITY)
        console.print(out)


if __name__ == "__main__":
    app()
