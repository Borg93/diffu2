"""The eval sets diffu-evaluate draws: one row model that every builder writes (build_crops' held-out splits,
steer_set, style_pairs, probe_set) and every reader reads (evaluate, the hands CLI), and the JSON-lines files they
live in."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Self

from pydantic import BaseModel, model_validator

from diffu_page.captions.prompt import LineRecord
from diffu_page.plan import CropPlan


class CropRow(BaseModel):
    """What every row of an eval set has: the real crop to compare with, its caption and line boxes, and optionally a
    noise seed and how a steering sheet labels it."""

    file_name: str  # the real crop, relative to the set's dir
    caption: str
    width: int
    height: int
    lines: list[LineRecord]
    seed: int | None = None  # in place of evaluate's --seed + index (a steering template draws every variant alike)
    template: int | None = None  # the steering template the row was made from
    variant: str | None = None
    example: str | None = None  # a real crop diffu_page.cli.sheets shows beside the drawing

    def plan(self) -> CropPlan:
        """The crop as a plan: its stored caption and boxes, lines read top to bottom as numbered."""
        return CropPlan(
            rect=(0, 0, self.width, self.height),
            width=self.width,
            height=self.height,
            caption=self.caption,
            lines=self.lines,
            reading_order=list(range(len(self.lines))),
        )


class EvalCrop(CropRow):
    """One crop of any eval set as diffu-evaluate reads it: drawn from a reference image when it names one, the
    reference's line boxes beside it (they measure the reference's hand)."""

    reference: str | None = None  # relative to the set's dir
    reference_lines: list[list[int]] | None = None  # the reference's line boxes (0-1000, y first)

    @model_validator(mode="after")
    def _reference_with_its_lines(self) -> Self:
        if (self.reference is None) != (self.reference_lines is None):
            raise ValueError(f"{self.file_name}: a reference and its line boxes go together, one without the other")
        return self


class ReferencedCrop(CropRow):
    """A crop of a set drawn from references (style_pairs' and probe_set's): its reference and the reference's line
    boxes required, so such a set without them fails when it is read. Its rows read as ``EvalCrop`` too."""

    reference: str
    reference_lines: list[list[int]]


def read_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    """Every line of the JSON-lines file at ``path`` as a ``model``."""
    with path.open(encoding="utf-8") as lines:
        return [model.model_validate_json(line) for line in lines if line.strip()]


def write_jsonl(path: Path, rows: Sequence[BaseModel]) -> None:
    """``rows`` as the JSON-lines file at ``path``, its dir made if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row.model_dump_json()}\n" for row in rows), encoding="utf-8")


def read_crops(crops_dir: Path, n: int | None = None) -> list[EvalCrop]:
    """The first ``n`` crops of an eval set (every one when None).

    Raises:
        ValueError: If some but not all of them name a reference image.
    """
    crops = read_jsonl(crops_dir / "metadata.jsonl", EvalCrop)[:n]
    with_reference = sum(crop.reference is not None for crop in crops)
    if with_reference not in (0, len(crops)):
        raise ValueError(f"{with_reference} of {len(crops)} crops name a reference image: all or none must")
    return crops


def generated_path(run: Path, index: int) -> Path:
    """Where an evaluate run saves its drawing of the set's crop ``index``."""
    return run / "generated" / f"{index:03d}.png"
