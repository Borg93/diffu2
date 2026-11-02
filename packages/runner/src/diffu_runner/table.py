"""A run's page table: one row per page in a Lance table (``<out>/pages.lance``), keyed by the page id, the image in
a Blob v2 column (Lance file format 2.2) and the labels as ALTO and PAGE XML.

A page that failed is a row too, with its error and no image, until a run into the same table draws it again: a run
resumes from what the table holds, skipping the pages drawn (:func:`drawn_ids`) and drawing again those that failed
(:func:`drop_failed`). The table is written by the run's driver alone (lance-ray's streaming write), so only the
machine that runs ``diffu-run`` needs it mounted.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from pathlib import Path

import lance
import pyarrow as pa
from pydantic import BaseModel, ConfigDict

PAGES = "pages.lance"
FILE_FORMAT = "2.2"  # Lance's file format; Blob v2 columns need 2.2 or later

LINE = pa.struct(
    [
        pa.field("text", pa.large_utf8()),
        pa.field("read", pa.large_utf8()),
        pa.field("cer", pa.float32()),
        pa.field("numbers_differ", pa.bool_()),
    ]
)
SCHEMA = pa.schema(
    [
        pa.field("id", pa.utf8(), nullable=False),
        lance.blob_field("image"),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
        pa.field("alto_xml", pa.large_utf8()),
        pa.field("page_xml", pa.large_utf8()),
        pa.field("text", pa.large_utf8()),
        pa.field("lines", pa.list_(LINE)),
        pa.field("unclear_lines", pa.int32()),
        pa.field("seed", pa.int64()),
        pa.field("template", pa.utf8()),
        pa.field("job", pa.large_utf8()),
        pa.field("model", pa.utf8()),
        pa.field("run", pa.utf8()),
        pa.field("seconds", pa.float32()),
        pa.field("error", pa.utf8()),
    ]
)


class LineRow(BaseModel):
    """One line of a drawn page: the text its labels carry, what the judge read in its box, the read's CER against
    the text, and whether a number in it reads otherwise."""

    model_config = ConfigDict(frozen=True)

    text: str
    read: str
    cer: float
    numbers_differ: bool


class PageRow(BaseModel):
    """A page as the table holds it.

    Attributes:
        id: The page's id (its job's, an opening's side appended).
        image: The page as JPEG; None when it failed.
        alto_xml: Its labels as ALTO; ``page_xml`` as PAGE. Both name the image ``<id>.jpg``.
        text: Its lines' texts in reading order, a line each.
        lines: Each line's text and the judge's verdict on it, in reading order.
        unclear_lines: How many lines the labels tag ``unclear``: the judge could not confirm them.
        seed: The noise seed it was drawn from (None if it never reached drawing).
        template: Its template's path; None for a page given as a layout.
        job: The page job as drawn, as JSON (:class:`diffu_page.draw.jobs.PageJob`, planned whole): it draws the page
            again.
        model: The model directory that drew it.
        run: The run that drew it (``<out>/runs/<run>.json``).
        seconds: How long drawing, judging and labelling it took on its actor.
        error: Why it failed, as ``<ExceptionType>: <message>``; None when it was drawn.
    """

    id: str
    image: bytes | None = None
    width: int | None = None
    height: int | None = None
    alto_xml: str | None = None
    page_xml: str | None = None
    text: str | None = None
    lines: list[LineRow] = []
    unclear_lines: int | None = None
    seed: int | None = None
    template: str | None = None
    job: str
    model: str
    run: str
    seconds: float
    error: str | None = None


class RunCounts(BaseModel):
    """What one run put into the table: the pages it drew, those that failed, and the median time a page took."""

    drawn: int
    failed: int
    median_seconds: float | None


def page_table(rows: Sequence[PageRow]) -> pa.Table:
    """``rows`` as an Arrow table of the page table's schema, the images in a Blob v2 column."""
    values = {name: [getattr(row, name) for row in rows] for name in SCHEMA.names}
    values["lines"] = [[line.model_dump() for line in row.lines] for row in rows]
    arrays = [
        lance.blob_array(values[field.name]) if field.name == "image" else pa.array(values[field.name], field.type)
        for field in SCHEMA
    ]
    return pa.Table.from_arrays(arrays, schema=SCHEMA)


def quoted(value: str) -> str:
    """``value`` as a string literal of a Lance filter."""
    return "'" + value.replace("'", "''") + "'"


def drawn_ids(table: Path) -> frozenset[str]:
    """The ids of the pages ``table`` holds drawn; none when there is no table yet."""
    if not table.exists():
        return frozenset()
    ids = lance.dataset(str(table)).to_table(columns=["id"], filter="error IS NULL").column("id")
    return frozenset(ids.to_pylist())


def drop_failed(table: Path) -> int:
    """Delete the rows of the pages that failed from ``table``, so that a run draws them again; how many there were."""
    if not table.exists():
        return 0
    dataset = lance.dataset(str(table))
    failed = dataset.count_rows(filter="error IS NOT NULL")
    if failed:
        dataset.delete("error IS NOT NULL")
    return failed


def run_counts(table: Path, run: str) -> RunCounts:
    """What the run ``run`` put into ``table``."""
    if not table.exists():
        return RunCounts(drawn=0, failed=0, median_seconds=None)
    rows = lance.dataset(str(table)).to_table(columns=["seconds", "error"], filter=f"run = {quoted(run)}").to_pydict()
    drawn = [seconds for seconds, error in zip(rows["seconds"], rows["error"], strict=True) if error is None]
    return RunCounts(
        drawn=len(drawn),
        failed=len(rows["error"]) - len(drawn),
        median_seconds=statistics.median(drawn) if drawn else None,
    )
