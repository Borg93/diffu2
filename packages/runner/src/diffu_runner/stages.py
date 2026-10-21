"""The Ray Data stages of a run, as plain callables (no Ray import, so they run and test off-cluster).

Rows are dicts of plain values: a page row carries its index and its :class:`PageJob` as JSON, a crop row its
:class:`CropJob` as JSON, a drawn row the crop's outputs and ``error`` ('' when it was drawn).
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel

from diffu_page.jobs import CropJob, PageJob, crop_jobs
from diffu_page.pipeline import DrawSettings, crop_blocks, draw_crop, load_pipeline

type Row = dict[str, object]


class DrawnRow(BaseModel):
    """One drawn crop as written to the run's Parquet: its job, its outputs, or the error that stopped it."""

    name: str
    template: str
    seed: int
    image: bytes  # JPEG; empty when the crop failed
    labels_xml: str
    line_cers: list[float]
    unclear_lines: int
    seconds: float
    error: str  # '' when drawn, else "<ExceptionType>: <message>"


def page_rows(pages: list[PageJob]) -> list[Row]:
    """A run's pages as Ray Data rows, each with its index in the run (its crops' names and noise seeds depend on
    it)."""
    return [{"index": index, "page": page.model_dump_json()} for index, page in enumerate(pages)]


def plan_rows(row: Row) -> list[Row]:
    """A page row planned into crop rows (Ray Data ``flat_map``)."""
    page = PageJob.model_validate_json(str(row["page"]))
    return [{"job": job.model_dump_json()} for job in crop_jobs(page, int(str(row["index"])))]


def _jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def _reference(path: Path | None) -> Image.Image | None:
    if path is None:
        return None
    with Image.open(path) as image:
        return image.convert("RGB")


class DrawActor:
    """A Ray Data actor holding one crop pipeline (:func:`diffu_page.pipeline.crop_blocks`): loaded once, then each
    batch of crop rows drawn.

    Args:
        model: The model directory (FLUX.3 Klein or diffu).
        loras: NeMo LoRA checkpoint ``model/`` dirs by adapter name.
        settings: How every crop is drawn.
    """

    def __init__(self, model: Path, loras: dict[str, Path], settings: DrawSettings) -> None:
        self._pipe = load_pipeline(model, crop_blocks, loras=loras, settings=settings)
        self._settings = settings

    def _draw(self, job: CropJob) -> DrawnRow:
        start = time.perf_counter()
        try:
            crop = draw_crop(
                self._pipe,
                job.plan,
                name=job.name,
                seed=job.seed,
                reference=_reference(job.reference),
                settings=self._settings,
            )
        except Exception as error:  # flag, don't drop: a failed crop stays in the run with its error
            return DrawnRow(
                name=job.name,
                template=str(job.template),
                seed=job.seed,
                image=b"",
                labels_xml="",
                line_cers=[],
                unclear_lines=0,
                seconds=time.perf_counter() - start,
                error=f"{type(error).__name__}: {error}",
            )
        return DrawnRow(
            name=job.name,
            template=str(job.template),
            seed=job.seed,
            image=_jpeg(crop.image),
            labels_xml=crop.labels_xml,
            line_cers=crop.line_cers,
            unclear_lines=crop.unclear_lines,
            seconds=time.perf_counter() - start,
            error="",
        )

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, list[object]]:
        rows = [self._draw(CropJob.model_validate_json(str(job))) for job in batch["job"]]
        return {field: [getattr(row, field) for row in rows] for field in DrawnRow.model_fields}
