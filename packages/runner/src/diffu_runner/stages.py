"""The Ray Data stages of a run, as plain callables (no Ray import, so they run and test off-cluster).

Rows are dicts of plain values. A run file's row is one of its lines, a :class:`PageJob` as JSON (Ray Data's
``read_text``); :func:`plan_rows` turns it into a row for each single page it draws (a two-page opening is two pages:
:func:`diffu_page.jobs.single_pages`) that the run's table does not hold yet, each planned whole as one crop (its
:class:`CropJob` as JSON) or carrying the error that stopped it. A :class:`DrawActor` draws, judges and labels each
planned page and returns it as the page table's rows (:mod:`diffu_runner.table`).
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager, nullcontext
from enum import StrEnum
from pathlib import Path

import pyarrow as pa
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch.nn.attention import SDPBackend, sdpa_kernel

from diffu_page.jobs import CropJob, PageJob, crop_jobs, single_pages
from diffu_page.judge import TrOCRJudge
from diffu_page.labels import LabelFormat, export
from diffu_page.local_run import describe
from diffu_page.output import err_console, jpeg_bytes
from diffu_page.pipeline import DrawSettings, crop_blocks, draw_crop, load_pipeline
from diffu_page.settings import settings
from diffu_runner.table import LineRow, PageRow, page_table

type Row = dict[str, object]

DEVICE_FAULT_EXIT = 1  # the exit status of an actor whose device failed: a crash to Ray, which restarts the actor


class Attention(StrEnum):
    """The attention kernel the models' attention (PyTorch's ``scaled_dot_product_attention``) runs on: ``auto`` lets
    PyTorch pick one per call, any other value allows that kernel alone, so a call it cannot take fails."""

    AUTO = "auto"
    FLASH = "flash"
    CUDNN = "cudnn"
    EFFICIENT = "efficient"
    MATH = "math"

    def kernel(self) -> AbstractContextManager[object]:
        """A context in which attention runs on this kernel."""
        backends = {
            Attention.FLASH: SDPBackend.FLASH_ATTENTION,
            Attention.CUDNN: SDPBackend.CUDNN_ATTENTION,
            Attention.EFFICIENT: SDPBackend.EFFICIENT_ATTENTION,
            Attention.MATH: SDPBackend.MATH,
        }
        return nullcontext() if self is Attention.AUTO else sdpa_kernel(backends[self])


class JudgeModel(BaseModel):
    """The HTR model every drawn line is read back with: a local directory (held by its absolute path), or a Hub id at
    a pinned revision in the Hugging Face cache (by default ``DIFFU_HTR_MODEL`` and ``DIFFU_HTR_REVISION``)."""

    model_config = ConfigDict(frozen=True)

    model: str = settings.htr_model
    revision: str | None = settings.htr_revision

    @field_validator("model")
    @classmethod
    def _local_directory_absolute(cls, model: str) -> str:
        local = Path(model)
        return str(local.absolute()) if local.is_dir() else model

    @field_validator("revision")
    @classmethod
    def _local_directory_has_none(cls, revision: str | None) -> str | None:
        return revision or None


class DrawConfig(BaseModel):
    """What decides how a run's pages come out: the model directory (MOSS-Image's, ``diffu-moss-init``'s; or FLUX.3
    Klein's with its ``loras``, NeMo ``model/`` dirs by adapter name), the draw settings, the size in pixels a page is
    drawn whole at unless its job gives its own, and the judge. Every run into one table draws with the same. Paths are
    held absolute: the actors load them from working directories of their own."""

    model_config = ConfigDict(frozen=True)

    model: Path
    loras: dict[str, Path] = {}
    settings: DrawSettings
    page_pixels: int = Field(default=1_000_000, gt=0)
    judge: JudgeModel = JudgeModel()

    @field_validator("model")
    @classmethod
    def _absolute_model(cls, model: Path) -> Path:
        return model.absolute()

    @field_validator("loras")
    @classmethod
    def _absolute_loras(cls, loras: dict[str, Path]) -> dict[str, Path]:
        return {name: path.absolute() for name, path in loras.items()}


def plan_rows(row: Row, *, drawn: frozenset[str], page_pixels: int) -> list[Row]:
    """A run file's row (``text``: a :class:`PageJob` as JSON) planned into a row for each single page it draws that
    ``drawn`` (the ids the run's table holds drawn) lacks, each planned whole: at its job's ``page_pixels``, else at
    about ``page_pixels`` pixels (Ray Data ``flat_map``). A page that cannot be planned keeps its row, with its error
    and no plan."""
    page = PageJob.model_validate_json(str(row["text"]))
    planned: list[Row] = []
    for single in single_pages(page):
        if single.id in drawn:
            continue
        whole = single if single.page_pixels is not None else single.model_copy(update={"page_pixels": page_pixels})
        try:
            [job] = crop_jobs(whole, 0)  # a page planned whole is one crop, named and seeded by the page's id
        except Exception as error:  # flag, don't drop: a page that cannot be planned stays in the run with its error
            planned.append({"page": whole.model_dump_json(), "job": "", "error": describe(error)})
        else:
            planned.append({"page": whole.model_dump_json(), "job": job.model_dump_json(), "error": ""})
    return planned


def quiet_progress(blocks: object) -> None:
    """Turn off the progress bar of every denoise loop in ``blocks`` (a pipeline's blocks): each actor's per-step bars
    would flood the run's logs."""
    for block in getattr(blocks, "sub_blocks", {}).values():
        if hasattr(block, "set_progress_bar_config"):
            block.set_progress_bar_config(disable=True)
        quiet_progress(block)


class DrawActor:
    """A Ray Data actor holding the judge and one crop pipeline (:func:`diffu_page.pipeline.crop_blocks`), each loaded
    once, then drawing each planned page whole, judging it line by line and labelling it (ALTO and PAGE), as the page
    table's rows.

    A page that cannot be drawn becomes a row with its error. A CUDA fault ends the actor's process instead: the device
    can no longer be trusted, so Ray restarts the actor (``max_restarts``) and draws the page again. Running out of
    GPU memory is a page's error: drawing it again would run out again.

    Args:
        draw: How the run's pages are drawn.
        run: The run the rows are drawn in.
        attention: The kernel the models' attention runs on while a page is drawn and judged (an
            :class:`Attention` value, given as its plain string: see :func:`diffu_runner.run.draw_dataset`).
    """

    def __init__(self, draw: DrawConfig, run: str, attention: str = Attention.AUTO.value) -> None:
        self._attention = Attention(attention)
        self._judge = TrOCRJudge(draw.judge.model, revision=draw.judge.revision, device=draw.settings.device)
        self._pipe = load_pipeline(draw.model, crop_blocks, loras=draw.loras, settings=draw.settings)
        quiet_progress(self._pipe.blocks)
        self._draw = draw
        self._run = run

    def __call__(self, batch: Mapping[str, Iterable[object]]) -> pa.Table:
        planned = zip(batch["page"], batch["job"], batch["error"], strict=True)
        return page_table([self._page_row(str(page), str(job), str(error)) for page, job, error in planned])

    def _page_row(self, page_json: str, job_json: str, plan_error: str) -> PageRow:
        page = PageJob.model_validate_json(page_json)
        if plan_error:
            return self._failed(page, error=plan_error, seed=None, seconds=0.0)
        job = CropJob.model_validate_json(job_json)
        start = time.perf_counter()
        try:
            with self._attention.kernel():
                crop = draw_crop(
                    self._pipe,
                    job.plan,
                    name=str(page.id),
                    seed=job.seed,
                    reference=job.reference_image(),
                    settings=self._draw.settings,
                    judge=self._judge,
                )
        except torch.AcceleratorError as fault:
            err_console.print(f"{page.id}: the device failed, so the actor exits for Ray to restart it: {fault}")
            os._exit(DEVICE_FAULT_EXIT)
        except Exception as error:  # a page's error must not end a long run; its row keeps the error
            return self._failed(page, error=describe(error), seed=job.seed, seconds=time.perf_counter() - start)
        lines = [
            LineRow(
                text=crop.plan.lines[k].text,
                read=crop.verdicts[k].read,
                cer=crop.verdicts[k].cer,
                numbers_differ=crop.verdicts[k].numbers_differ,
            )
            for k in crop.plan.reading_order
        ]
        return PageRow(
            id=str(page.id),
            image=jpeg_bytes(crop.image),
            width=crop.image.width,
            height=crop.image.height,
            alto_xml=export(crop.labels, LabelFormat.ALTO),
            page_xml=export(crop.labels, LabelFormat.PAGE),
            text="\n".join(line.text for line in lines),
            lines=lines,
            unclear_lines=crop.unclear_lines,
            seed=job.seed,
            template=str(page.template),
            job=page.model_dump_json(),
            model=str(self._draw.model),
            run=self._run,
            seconds=time.perf_counter() - start,
        )

    def _failed(self, page: PageJob, *, error: str, seed: int | None, seconds: float) -> PageRow:
        return PageRow(
            id=str(page.id),
            seed=seed,
            template=str(page.template),
            job=page.model_dump_json(),
            model=str(self._draw.model),
            run=self._run,
            seconds=seconds,
            error=error,
        )
