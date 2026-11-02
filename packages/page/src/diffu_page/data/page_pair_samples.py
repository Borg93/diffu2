"""Legible samples for the rows of a page-pair manifest, written with a new manifest beside it: each row's sample page
cut from its scan at its target's writing scale (``SampleSize.LEGIBLE``, :mod:`diffu_page.draw.sample`) in place of
its ``refs/`` image, shrunk under half a megapixel, beside which a window's letters are several times larger.

A page row's sample is the sample page at its target's pixel count (about 1 MP); a window row's is a window of the
sample page at the window's line pitch and size, cut around the middle of the page's lines. Both pitches are measured
as drawing measures them (:func:`diffu_page.draw.sample.line_pitch`): the window's on its lines' boxes (its scan's
record holds its :class:`diffu_page.data.page_pairs.WindowRecord`), the sample page's on the lines of its whole-page
plan (its own record's), which ``val_eval`` gives as a sample's lines and drawing reads, so a sample is cut here as
``ReferenceGridStep`` cuts one from the same page and lines when a model trained on legible samples draws. Each sample
page is cut whole from its scan, read from the GT root the set's ``scan.json`` names; the rows are grouped by their
sample's scan, each scan read once, on ``--workers`` processes.

Written in the set's dir: ``samples/<row id>.jpg``, each written whole before it is named, so a stopped run resumes
where it stopped, and ``--out``, a new manifest: the rows in their order, each as it was but for its sample (its
``reference`` its own image, its ``sample_size`` legible and its sample page's id ``sample``), which
``diffu-moss-cache-pairs`` records, and a training and a model after it. A row whose sample is legible already is
refused.

    python -m diffu_page.data.page_pair_samples --pairs data_flux3/page_pairs_many_1mp/train_content.jsonl \\
        --out data_flux3/page_pairs_many_1mp/train_content_legible.jsonl --workers 4
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image
from pydantic import BaseModel, ConfigDict

from diffu_moss.train.pairs import PagePairRow, SampleSize
from diffu_page.data.build_crops import TILE_QUALITY
from diffu_page.data.page_pair_forms import read_rows, row_target, scan_xml
from diffu_page.data.page_pairs import (
    SCAN_RECORD,
    SCANNED,
    ScanResult,
    in_processes,
    load_scan,
    record_reader,
    recorded_settings,
    scan_digest,
)
from diffu_page.draw.sample import legible_sample, line_pitch
from diffu_page.output import console
from diffu_page.page.layout import load_xml_layout, scan_pages
from diffu_page.page.layout.page_xml import find_scan
from diffu_page.page.layout.pages import digest, page_id

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

SAMPLES = "samples"


class RowSample(BaseModel):
    """A row's legible sample to cut: the row's id, its sample page's id, its target's size (width, height) px and, for
    a window, its line pitch px (None for a whole page)."""

    model_config = ConfigDict(frozen=True)

    row: str
    sample: str
    size: tuple[int, int]
    pitch: float | None


class SampleTask(BaseModel):
    """The samples to cut from one scan of the page-pair set in ``directory``: its PAGE file (relative to the GT root
    ``gt_root``) and the rows' samples of its pages."""

    directory: Path
    gt_root: str
    xml: str
    samples: list[RowSample]


class SamplesWritten(BaseModel):
    """What a manifest given legible samples holds: its rows, of them its windows, and the samples cut now (the others
    a stopped run cut)."""

    rows: int
    windows: int
    cut: int


def sample_file(directory: Path, row: str) -> Path:
    """Where the legible sample of the row ``row`` of the set in ``directory`` is written."""
    return directory / SAMPLES / f"{row}.jpg"


def cut_samples(task: SampleTask) -> int:
    """The legible samples of ``task``'s rows not written yet, each cut from its sample page, the page cut whole from
    the scan (:func:`diffu_page.draw.sample.legible_sample`), written whole before it is named.

    Returns:
        How many it cut.

    Raises:
        ValueError: If the PAGE file names no scan beside it, the scan is not the size it gives, or a sample is no page
            of it.
    """
    left = [sample for sample in task.samples if not sample_file(task.directory, sample.row).exists()]
    if not left:
        return 0
    xml = Path(task.gt_root) / task.xml
    scan_path = find_scan(xml)
    if scan_path is None:
        raise ValueError(f"{task.xml} names no scan beside it")
    layout = load_xml_layout(xml)
    pages = {page_id(Path(task.xml), page): page for page in scan_pages(layout)}
    record = task.directory / SCANNED / f"{digest(task.xml)}.json"
    recorded = ScanResult.model_validate_json(record.read_text(encoding="utf-8"))
    lines = {page.id: [line.bbox for line in page.lines] for page in recorded.pages}
    missing = sorted({sample.sample for sample in left} - (pages.keys() & lines.keys()))
    if missing:
        raise ValueError(f"{task.xml} gives no recorded single page {', '.join(missing)}")
    scan = load_scan(scan_path, layout)
    cuts: dict[str, Image.Image] = {}
    for sample in left:
        if sample.sample not in cuts:
            cuts[sample.sample] = scan.crop(pages[sample.sample].rect)
        image = legible_sample(cuts[sample.sample], lines[sample.sample], size=sample.size, pitch=sample.pitch)
        final = sample_file(task.directory, sample.row)
        partial = final.with_suffix(".partial")
        image.save(partial, format="JPEG", quality=TILE_QUALITY)
        partial.rename(final)
    return len(left)


def row_samples(manifest: Path) -> Iterator[RowSample]:
    """Each row's legible sample to cut, a row at a time, a window's pitch measured on its lines' boxes as its scan's
    record holds them.

    Raises:
        ValueError: If a row's sample is legible already, or its window is not recorded in its set.
    """
    page_of = record_reader(manifest.parent)
    for row in read_rows(manifest):
        if row.sample_size is not SampleSize.HALF:
            raise ValueError(f"row {row.id}: its sample is sized {row.sample_size} already")
        pitch = None
        if row.window is not None:
            page, target = row_target(row)
            window = next((window for window in page_of(page).windows if window.id == target), None)
            if window is None:
                raise ValueError(f"row {row.id}: its set records no window {target} of page {page}")
            pitch = line_pitch([line.bbox for line in window.lines], row.size[1])
        yield RowSample(row=row.id, sample=Path(row.reference).stem, size=row.size, pitch=pitch)


def legible_row(row: PagePairRow) -> PagePairRow:
    """``row`` with its legible sample: its reference the sample cut for it, its sample page named by ``sample``."""
    return row.model_copy(
        update={
            "reference": f"{SAMPLES}/{row.id}.jpg",
            "sample_size": SampleSize.LEGIBLE,
            "sample": Path(row.reference).stem,
        }
    )


def write_legible_samples(manifest: Path, out: Path, *, workers: int) -> SamplesWritten:
    """The rows of ``manifest``, a manifest in a page-pair set's dir, given legible samples (the module's docstring),
    cut on ``workers`` processes and written to ``out``.

    Raises:
        ValueError: If ``out`` exists (the manifest itself among others), ``manifest`` is in no page-pair set's dir, the
            set holds no record of a sample's scan, a row's sample is legible already or its window is not recorded,
            or a sample cannot be cut (:func:`cut_samples`).
    """
    if out.exists():
        raise ValueError(f"{out} exists: give a new file, the manifest is kept as it is")
    directory = manifest.parent
    if not (directory / SCAN_RECORD).is_file():
        raise ValueError(f"{manifest} is in no page-pair set's dir: no {SCAN_RECORD} beside it")
    settings = recorded_settings(directory / SCAN_RECORD)
    by_scan: dict[str, list[RowSample]] = {}
    for sample in row_samples(manifest):
        by_scan.setdefault(scan_digest(sample.sample), []).append(sample)
    xmls = {scan: scan_xml(directory, scan) for scan in by_scan}  # every sample's scan recorded, before one is cut
    (directory / SAMPLES).mkdir(exist_ok=True)
    tasks = (
        SampleTask(directory=directory, gt_root=settings.gt_root, xml=xmls[scan], samples=samples)
        for scan, samples in by_scan.items()
    )
    cut = 0
    for k, count in enumerate(in_processes(cut_samples, tasks, workers), start=1):
        cut += count
        if k % 100 == 0:
            console.print(f"{k} of {len(by_scan)} scans cut")
    rows = windows = 0
    partial = out.with_name(f".{out.name}.partial")
    with partial.open("w", encoding="utf-8") as lines:
        for row in read_rows(manifest):
            lines.write(f"{legible_row(row).model_dump_json()}\n")
            rows += 1
            windows += row.window is not None
    partial.rename(out)
    return SamplesWritten(rows=rows, windows=windows, cut=cut)


@app.command()
def main(
    pairs: Annotated[
        Path,
        typer.Option(
            help="a manifest in a page-pair set's dir: its train.jsonl or val.jsonl, or one made of their rows"
        ),
    ],
    out: Annotated[Path, typer.Option(help="the manifest to write, a new file: its rows with legible samples")],
    workers: Annotated[int, typer.Option(min=1, help="processes cutting the scans: fewer beside a training")] = max(
        1, (os.cpu_count() or 2) // 2
    ),
) -> None:
    """Give the rows of a page-pair manifest samples at their targets' writing scale, cut from the scans: a new
    manifest."""
    try:
        written = write_legible_samples(pairs, out, workers=workers)
    except ValueError as error:  # a used --out, a manifest of no set, a row legible already: the arguments' fault
        raise typer.BadParameter(str(error)) from error
    console.print(f"{written.model_dump_json()} -> {out}")


if __name__ == "__main__":
    app()
