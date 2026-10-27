"""The print/hand classifier's lines (CPU): text lines of ground-truth pages cut from their scans as the classifier
reads them (:func:`diffu_page.marking.lines.line_images`), an OCR ground truth's lines as printed and an HTR ground
truth's as handwritten, each with its collection, so the classifier can be validated on collections it never saw.

Each root holds a folder per collection, its PAGE files in ``page/`` folders beside their scans (the Transkribus export
layout). Pages are drawn at random (seeded) and from each its lines of at least three characters, in random order, a
share of the collection's lines per page. A scan is decoded at the smallest scale that keeps its median line at least
64 px high. An HTR ground truth transcribes a form's printed headings with its entries, so a few of its lines are
printed: the handwritten lines are a little noisy.

    python -m diffu_page.data.print_hand_set --printed ~/Desktop/data/GT-Lejonet-OCR-202606 \\
        --handwritten ~/Desktop/data/GT-Lejonet_HTR_202602 --out runs/print_hand/lines.npz

Writes one ``.npz`` (no pickles, :func:`read_line_set`): ``images``, every line image side by side (64 px high, uint8,
ink dark), each line's ``widths``, whether it is ``printed``, the index of its ``collection`` in ``collections`` and
its ``pages`` (the PAGE file, relative to its root).
"""

from __future__ import annotations

import math
import random
import statistics
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from pydantic import BaseModel, ConfigDict

from diffu_page.layout import load_xml_layout
from diffu_page.layout.page_xml import find_scan
from diffu_page.local_run import describe
from diffu_page.marking.lines import LINE_HEIGHT, REDUCTIONS, line_images, on_scan, read_scan
from diffu_page.output import console, err_console

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

MIN_TEXT = 3  # characters: a shorter line (a number, a mark) shows too little of its hand or type
MAX_WIDTH = 3 * 256  # px: a longer line image keeps its start, three of the classifier's windows


class Sample(BaseModel):
    """How much of each collection is cut: lines from up to ``pages`` random pages, up to ``lines`` in all."""

    model_config = ConfigDict(frozen=True)

    pages: int
    lines: int


class GroundTruth(BaseModel):
    """A ground truth to cut lines from: its root (a folder per collection), whether its lines are printed, and how much
    of each collection to cut."""

    model_config = ConfigDict(frozen=True)

    root: Path
    printed: bool
    sample: Sample


class LineSet(BaseModel):
    """Line images (uint8, ink dark, :data:`LINE_HEIGHT` px high), whether each is printed, its collection (an index
    into ``collections``) and the PAGE file it was cut from."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    images: list[np.ndarray]
    printed: np.ndarray  # bool, one per image
    collection: np.ndarray  # int, one per image
    collections: list[str]
    pages: list[str]


def page_lines(xml_path: Path, rng: random.Random) -> list[np.ndarray]:
    """The line images of a ground-truth page (lines of at least :data:`MIN_TEXT` characters), in random order, cut from
    its scan decoded at the smallest scale that keeps its median line :data:`LINE_HEIGHT` px high; none without a
    scan."""
    layout = load_xml_layout(xml_path)
    lines = [line for line in layout.lines if len(line.text.strip()) >= MIN_TEXT]
    scan_path = find_scan(xml_path)
    if scan_path is None or not lines:
        return []
    median = statistics.median(line.box[3] - line.box[1] for line in lines)
    reduction = max(r for r in REDUCTIONS if r == 1 or median / r >= LINE_HEIGHT)
    scan = read_scan(scan_path, reduction=reduction)
    rng.shuffle(lines)
    return [image[:, :MAX_WIDTH] for line in lines for image in line_images(scan, on_scan(line.box, layout, scan))]


def collection_lines(
    collection: Path, root: Path, *, sample: Sample, rng: random.Random
) -> tuple[list[tuple[np.ndarray, str]], dict[str, str]]:
    """Up to ``sample.lines`` line images of a collection, each with its PAGE file (relative to ``root``), from up to
    ``sample.pages`` random pages, an even share from each; and the pages that could not be read, by path, with their
    error."""
    xml_paths = sorted(p for p in collection.rglob("page/*.xml") if not p.name.startswith("._"))  # "._*": macOS
    rng.shuffle(xml_paths)
    share = math.ceil(sample.lines / sample.pages)
    cut: list[tuple[np.ndarray, str]] = []
    failed: dict[str, str] = {}
    pages = 0
    for xml_path in xml_paths:
        if pages == sample.pages or len(cut) >= sample.lines:
            break
        page = str(xml_path.relative_to(root))
        try:
            images = page_lines(xml_path, rng)
        except Exception as error:  # one unreadable page must not end the build; it is reported
            failed[page] = describe(error)
            continue
        if images:
            pages += 1
            cut += [(image, page) for image in images[:share]]
    return cut[: sample.lines], failed


def build(truths: list[GroundTruth], out: Path, *, seed: int) -> dict[str, str]:
    """Cut every collection of each ground truth and write the set to ``out``.

    Returns:
        The pages that could not be read, by path, with their error.
    """
    rng = random.Random(seed)
    images: list[np.ndarray] = []
    printed: list[bool] = []
    collection_of: list[int] = []
    collections: list[str] = []
    pages: list[str] = []
    failed: dict[str, str] = {}
    for truth in truths:
        for collection in sorted(p for p in truth.root.iterdir() if p.is_dir()):
            cut, unread = collection_lines(collection, truth.root, sample=truth.sample, rng=rng)
            failed |= unread
            if not cut:
                continue
            collections.append(collection.name)
            for image, page in cut:
                images.append(image)
                printed.append(truth.printed)
                collection_of.append(len(collections) - 1)
                pages.append(page)
            console.print(f"{collection.name}: {len(cut)} {'printed' if truth.printed else 'handwritten'} lines")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        images=np.concatenate(images, axis=1),
        widths=np.array([image.shape[1] for image in images]),
        printed=np.array(printed),
        collection=np.array(collection_of),
        collections=np.array(collections),
        pages=np.array(pages),
    )
    return failed


def read_line_set(path: Path) -> LineSet:
    """The line set :func:`build` wrote to ``path``."""
    with np.load(path, allow_pickle=False) as saved:
        bounds = np.cumsum(saved["widths"])[:-1]
        return LineSet(
            images=np.split(saved["images"], bounds, axis=1),
            printed=saved["printed"],
            collection=saved["collection"],
            collections=saved["collections"].tolist(),
            pages=saved["pages"].tolist(),
        )


@app.command()
def main(
    printed: Annotated[Path, typer.Option(help="the OCR ground truth: a folder per collection of printed pages")],
    handwritten: Annotated[Path, typer.Option(help="the HTR ground truth: a folder per collection of written pages")],
    out: Annotated[Path, typer.Option(help="the .npz to write")],
    printed_pages: Annotated[int, typer.Option(help="pages drawn per printed collection")] = 60,
    printed_lines: Annotated[int, typer.Option(help="lines cut per printed collection")] = 900,
    handwritten_pages: Annotated[int, typer.Option(help="pages drawn per handwritten collection")] = 30,
    handwritten_lines: Annotated[int, typer.Option(help="lines cut per handwritten collection")] = 300,
    seed: int = 0,
) -> None:
    """Cut the print/hand classifier's lines from an OCR and an HTR ground truth."""
    truths = [
        GroundTruth(root=printed.expanduser(), printed=True, sample=Sample(pages=printed_pages, lines=printed_lines)),
        GroundTruth(
            root=handwritten.expanduser(),
            printed=False,
            sample=Sample(pages=handwritten_pages, lines=handwritten_lines),
        ),
    ]
    failed = build(truths, out, seed=seed)
    for page, error in failed.items():
        err_console.print(f"{page}: {error}")
    console.print(f"-> {out}")


if __name__ == "__main__":
    app()
