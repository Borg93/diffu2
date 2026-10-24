"""Measure the hands of an evaluate run from its saved drawings, without the GPU: each drawn crop
(``generated/NNN.png``) beside its real crop and, where the eval set lists the reference's line boxes, its reference,
as ``diffu_page.cli.evaluate`` measures them now. For runs evaluated before evaluate measured hands, and to compare
runs on one eval set. Writes ``hands.json`` (each crop's ``Hands`` and the run's ``HandFidelity``) into the run.

    python -m diffu_page.cli.hands --crops data_flux3/style_pairs_v3/val_eval \\
        --run runs/flux3_style_v3/eval_val_step_1599
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel

from diffu_page.cli.evaluate import EvalCrop, HandFidelity, Hands, Results, crop_hands, hand_fidelity, read_crops
from diffu_page.output import load_rgb

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


class CropHands(BaseModel):
    """One drawn crop of the run, by its index in the eval set, and its hands."""

    index: int
    file: str
    variant: str | None = None
    hands: Hands


class HandsReport(BaseModel):
    """``hands.json``: the run's medians and every drawn crop's hands."""

    fidelity: HandFidelity
    samples: list[CropHands]


def drawn_hands(crop: EvalCrop, crops: Path, drawing: Path) -> Hands:
    """The hands of ``crop`` drawn as ``drawing``: its real crop resized to the drawing, as evaluate reads it back."""
    drawn = load_rgb(drawing)
    reference = load_rgb(crops / crop.reference) if crop.reference else None
    real = load_rgb(crops / crop.file_name).resize(drawn.size)
    return crop_hands(drawn, real, crop.lines, reference, crop.reference_lines)


def measure_run(crops: Path, run: Path) -> HandsReport:
    """Every drawn crop of the evaluate run in ``run`` measured against the eval set in ``crops``."""
    eval_crops = read_crops(crops, sys.maxsize)
    results = Results.model_validate_json((run / "results.json").read_text(encoding="utf-8"))
    samples = [
        CropHands(
            index=sample.index,
            file=sample.file,
            variant=sample.variant,
            hands=drawn_hands(eval_crops[sample.index], crops, run / "generated" / f"{sample.index:03d}.png"),
        )
        for sample in results.samples
    ]
    return HandsReport(fidelity=hand_fidelity([sample.hands for sample in samples]), samples=samples)


@app.command()
def main(
    crops: Annotated[Path, typer.Option(help="the eval set the run drew (a diffu_page.cli.evaluate --crops dir)")],
    run: Annotated[Path, typer.Option(help="the evaluate run: its results.json and generated/")],
) -> None:
    """Measure the hands of an evaluate run's drawings against their real crops and references."""
    report = measure_run(crops, run)
    (run / "hands.json").write_text(report.model_dump_json(indent=1), encoding="utf-8")
    print(report.fidelity.model_dump_json(indent=1))  # machine output: rich would wrap it at the terminal's width


if __name__ == "__main__":
    app()
