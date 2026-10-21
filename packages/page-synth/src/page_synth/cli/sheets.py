"""Contact sheets of the v2 steering test: per template, every variant's generated crop beside a real crop of the
archive it was asked to look like.

Reads the steering set (``page_synth.data.steer_set``) and the ``page_synth.cli.evaluate`` run over it (``generated/NNN.png``,
``results.json``); writes ``steer_template_<t>.jpg`` into the run dir.

    python -m page_synth.cli.sheets --steer data_flux3/steer_v2 --run runs/flux3_layout_v2/steer
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import BaseModel

PANEL_HEIGHT = 420
LABEL_HEIGHT = 22
GAP = 10


class Prompt(BaseModel):
    """What a sheet needs of a set's prompt (``page_synth.data.steer_set.SteerRow`` or ``page_synth.data.style_pairs.EvalRow``)."""

    template: int
    variant: str
    example: str | None  # a real crop of the variant's style, relative to the set's dir


class Sample(BaseModel):
    gen_cer: float


class Results(BaseModel):
    """What a sheet needs of ``page_synth.cli.evaluate``'s ``results.json``."""

    samples: list[Sample]


def label_of(prompt: Prompt, sample: Sample) -> str:
    beside = "   | right: a real crop of that style" if prompt.example else ""
    return f"{prompt.variant}   (TrOCR CER {sample.gen_cer:.3f}){beside}"


def labelled_row(label: str, panels: list[Image.Image]) -> Image.Image:
    """The panels side by side at ``PANEL_HEIGHT``, under one line of label."""
    scaled = [p.resize((round(p.width * PANEL_HEIGHT / p.height), PANEL_HEIGHT)) for p in panels]
    row = Image.new(
        "RGB", (sum(p.width for p in scaled) + GAP * (len(scaled) - 1), LABEL_HEIGHT + PANEL_HEIGHT), "white"
    )
    ImageDraw.Draw(row).text((4, 4), label, fill="black")
    x = 0
    for panel in scaled:
        row.paste(panel, (x, LABEL_HEIGHT))
        x += panel.width + GAP
    return row


def stack(rows: list[Image.Image]) -> Image.Image:
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height + GAP for r in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + GAP
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steer", type=Path, required=True, help="the steering set (page_synth.data.steer_set)")
    ap.add_argument("--run", type=Path, required=True, help="the page_synth.cli.evaluate output dir of that set")
    args = ap.parse_args()

    with (args.steer / "metadata.jsonl").open(encoding="utf-8") as metadata:
        prompts = [Prompt.model_validate_json(line) for line in metadata]
    results = Results.model_validate_json((args.run / "results.json").read_text())
    by_template: dict[int, list[Image.Image]] = defaultdict(list)
    for k, (prompt, sample) in enumerate(zip(prompts, results.samples, strict=True)):
        panels: list[Image.Image] = [Image.open(args.run / "generated" / f"{k:03d}.png")]
        if prompt.example:
            panels.append(Image.open(args.steer / prompt.example))
        by_template[prompt.template].append(labelled_row(label_of(prompt, sample), panels))
    for template, rows in by_template.items():
        out = args.run / f"steer_template_{template}.jpg"
        stack(rows).save(out, quality=88)
        print(out)


if __name__ == "__main__":
    main()
