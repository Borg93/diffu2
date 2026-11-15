"""The table of single pages: a row a page, pointing at its scan and its place on it.

Most scans are openings, and every reader here reads single pages. A scan is cut as the set builder cuts it
(:func:`diffu_page.page.layout.pages.scan_pages`, by its labelled lines alone), and what that cut cannot be trusted
with is said on the row, for a cut read from the image to settle:

- ``alone``: the page's facing page holds no labelled line. The cut is the middle of the band without text, which
  then reaches from this page's text far into the facing page, so the page's image holds a strip of it.
- ``whole`` with ``lines == 0``: nothing to cut by. Such a scan is as likely a full ledger the old segmenter missed as
  an empty opening.
- ``whole`` and nearly square: a gutter is only looked for in a scan wider than tall.

Given the gutters read from the scans (:mod:`diffu_curate.gutter`), a scan is cut at its gutter instead (``cut`` is
``image``): both pages are rows, one without a labelled line too. A scan the network finds no gutter in is still cut
there when its lines' gutter falls at the same place (two readings that agree), is one page when its lines see no
opening either, and is otherwise given whole with ``spread`` set: its lines say opening, the image does not say where,
so it is not a page. A row holds no image: :func:`open_page` crops the scan when a page is wanted.
"""

from __future__ import annotations

from collections.abc import Iterator
from multiprocessing import Pool as ProcessPool
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from diffu_curate.gutter import Gutter
from diffu_curate.settings import Pool, collection_of
from diffu_curate.table import append_rows, compact, held_values
from diffu_page.data.reader_pages import LabelFile, gt_split, label_files
from diffu_page.page.layout.model import Layout
from diffu_page.page.layout.pages import ScanPage, gutter, is_unsplit_spread, page_id, scan_pages, split_at, whole_page
from diffu_page.page.layout.xml_replay import load_xml_layout

GROUND_TRUTH = "gt"  # the ground truth's name as a pool in the table
WHOLE = "whole"  # the side of a page that is its scan whole
BY_LINES, BY_IMAGE = "lines", "image"  # how a scan was cut: by its labelled lines, or at the gutter read from its image
VOLUMES_A_WRITE = 200
AGREE = 0.04  # the lines' gutter and the image's are the same place within this share of the scan's width

Image.MAX_IMAGE_PIXELS = None  # the scans are large; they are our own


class PageRow(BaseModel):
    """A single page: its id (in the ground truth the id the reader sets give it, in a pool ``<scan>_<side>``), its
    pool, volume and scan, its side of an opening (``whole``: the scan is the page), where its labels and its scan
    are, the scan's size, its rectangle on the scan, how many labelled lines and characters it holds, and what makes
    its cut doubtful (the module's docstring). ``spread`` marks a scan wider than tall whose lines leave no gutter:
    not a page. ``split`` and ``collection`` are the ground truth's (None in a pool); ``error`` is set, and the rest
    left empty, when the label file cannot be read."""

    page: str
    pool: str
    volume: str
    scan: str
    side: str
    labels: str
    image: str
    error: str | None = None
    scan_width: int | None = None
    scan_height: int | None = None
    x0: int | None = None
    y0: int | None = None
    x1: int | None = None
    y1: int | None = None
    lines: int | None = None
    chars: int | None = None
    cut: str | None = None
    gutter_score: float | None = None
    alone: bool | None = None
    spread: bool | None = None
    split: str | None = None
    collection: str | None = None


class Scan(BaseModel):
    """A scan to cut into its pages: where its labels and its image are, what names it, and, in the ground truth, the
    folder its pages' ids are taken below, its collection and its split."""

    pool: str
    volume: str
    name: str
    labels: Path
    image: Path
    gutter: Gutter | None = None
    ids_below: Path | None = None
    collection: str | None = None
    split: str | None = None


def _row(scan: Scan, page: ScanPage, *, alone: bool | None, spread: bool | None, size: tuple[int, int]) -> PageRow:
    side = page.side.value if page.side else WHOLE
    name = page_id(scan.labels.relative_to(scan.ids_below), page) if scan.ids_below else f"{scan.name}_{side}"
    x0, y0, x1, y1 = (round(value) for value in page.rect)
    return PageRow(
        page=name,
        pool=scan.pool,
        volume=scan.volume,
        scan=scan.name,
        side=side,
        labels=str(scan.labels),
        image=str(scan.image),
        scan_width=size[0],
        scan_height=size[1],
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        lines=len(page.layout.lines),
        chars=sum(len(line.text) for line in page.layout.lines),
        cut=BY_IMAGE if scan.gutter else BY_LINES,
        gutter_score=scan.gutter.score if scan.gutter else None,
        alone=alone,
        spread=spread,
        split=scan.split,
        collection=scan.collection,
    )


def page_rows(scan: Scan) -> list[PageRow]:
    """The rows of a scan's pages; one row with its error if its labels cannot be read. A scan given with its gutter
    is cut as :func:`_cut_at_the_image` says, both pages given even when one holds no labelled line; a scan given
    without is cut by its labelled lines, as the set builder cuts it."""
    try:
        layout = load_xml_layout(scan.labels)
    except ValueError as error:
        return [
            PageRow(
                page=f"{scan.name}_{WHOLE}", pool=scan.pool, volume=scan.volume, scan=scan.name, side=WHOLE,
                labels=str(scan.labels), image=str(scan.image), error=str(error)[:200],
                split=scan.split, collection=scan.collection,
            )
        ]  # fmt: skip
    size = (layout.width, layout.height)
    if scan.gutter is None:
        pages = scan_pages(layout)
        alone = len(pages) == 1 and pages[0].side is not None
        return [_row(scan, page, alone=alone, spread=is_unsplit_spread(layout), size=size) for page in pages]
    pages, opening_uncut = _cut_at_the_image(layout, scan.gutter)
    return [_row(scan, page, alone=None, spread=opening_uncut, size=size) for page in pages]


def _cut_at_the_image(layout: Layout, found: Gutter) -> tuple[list[ScanPage], bool]:
    """The scan's pages by the gutter read from its image, and whether it is an opening left uncut. Cut at the image's
    gutter when the network finds one, or when it is unsure and the lines' gutter falls within :data:`AGREE` of it.
    Else whole: an opening left uncut when the lines see a gutter elsewhere or text across the middle of a scan wider
    than tall (of 36 such scans looked at, all held two pages or sheets, the right cut the lines' in some and the
    image's in others)."""
    by_lines = gutter(layout)
    agreed = by_lines is not None and abs(by_lines / layout.width - found.x) <= AGREE
    if found.found or agreed:
        return list(split_at(layout, found.x * layout.width)), False
    return [whole_page(layout)], by_lines is not None or is_unsplit_spread(layout)


def pool_scans(pool: Pool, volume: Path, gutters: dict[str, Gutter] | None = None) -> list[Scan]:
    """The scans of one of a pool's volumes, by name, each with its gutter where ``gutters`` (by image path) is
    given and holds it."""
    scans = []
    for path in sorted(volume.glob("*.xml")):
        image = pool.scan_file(path.stem)
        gutter = gutters.get(str(image)) if gutters is not None else None
        scans.append(Scan(pool=pool.name, volume=volume.name, name=path.stem, labels=path, image=image, gutter=gutter))
    return scans


def _truth_scan(label: LabelFile, ground_truth: Path, test_list: str, gutters: dict[str, Gutter] | None) -> Scan:
    return Scan(
        gutter=gutters.get(str(label.scan)) if gutters is not None else None,
        pool=GROUND_TRUTH,
        volume=label.path.parent.parent.name,
        name=label.scan.stem,
        labels=label.path,
        image=label.scan,
        ids_below=ground_truth,
        collection=collection_of(label.path),
        split=gt_split(label.path, test_list).value,
    )


def truth_scans(ground_truth: Path, test_list: str, gutters: dict[str, Gutter] | None = None) -> list[Scan]:
    """The ground truth's scans, one a scan as the set builder takes them (its second export of a collection and
    files without their scan left out), each with its collection and its split by ``test_list``."""
    root = ground_truth.resolve()
    labels, _ = label_files(root)
    return [_truth_scan(label, root, test_list, gutters) for label in labels]


def _rows_of(scans: list[Scan]) -> list[PageRow]:
    return [row for scan in scans for row in page_rows(scan)]


def write_pages(groups: list[list[Scan]], table: Path, workers: int) -> Iterator[tuple[int, int, int]]:
    """Append the pages of every scan in ``groups`` (a pool's volumes, or the ground truth in chunks) to ``table``.
    Yields (groups done, groups to do, rows written) after each write."""
    pending: list[PageRow] = []
    written = 0
    with ProcessPool(workers) as processes:
        for count, rows in enumerate(processes.imap_unordered(_rows_of, groups), 1):
            pending += rows
            if count % VOLUMES_A_WRITE == 0 or count == len(groups):
                append_rows(table, pending)
                written += len(pending)
                pending = []
                yield count, len(groups), written
    compact(table)


def pool_groups(pool: Pool, table: Path, gutters: dict[str, Gutter] | None = None) -> list[list[Scan]]:
    """A pool's scans a volume, the volumes ``table`` already holds left out, so a stopped run goes on."""
    held = held_values(table, "volume", f"pool = '{pool.name}'")
    return [pool_scans(pool, volume, gutters) for volume in pool.volumes() if volume.name not in held]


def open_page(row: PageRow) -> Image.Image:
    """The page's image, cropped from its scan."""
    if row.x0 is None or row.y0 is None or row.x1 is None or row.y1 is None:
        raise ValueError(f"{row.page}: no rectangle ({row.error})")
    with Image.open(row.image) as scan:
        return scan.convert("RGB").crop((row.x0, row.y0, row.x1, row.y1))
