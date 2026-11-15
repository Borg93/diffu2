"""How wrong a pool's labels are, measured on the scans the ground truth also holds.

A scan of the ground truth whose label file in the pool is still the old pipeline's has its old label and the
transcribers' truth side by side. The old lines are matched to the true ones by their boxes; a matched pair gives a
line's error, and what is left unmatched on either side gives the lines the old pipeline missed and the lines it found
where the transcribers transcribed none. These scans come from volumes the transcribers chose, so they are easier than
a pool at large: a threshold read off them is a floor, not the pool's level.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel

from diffu_curate.alto import AltoLine, AltoPage, Box, read_alto
from diffu_curate.settings import COLLECTION_PREFIX, Pool, collection_of, volume_of
from diffu_curate.table import write_rows
from diffu_page.data.reader_pages import gt_split
from diffu_page.quality.readback import levenshtein, numbers_differ

OLD_PIPELINE = "htrflow"  # the writer whose labels are calibrated; a transcriber's export in the pool is truth already
MATCH_IOU = 0.2  # an old line and a true line are the same line from this overlap of their boxes
MASK_SCALE = 8  # outlines are compared on masks drawn at 1/8 of the scan


class TruthScan(BaseModel):
    """A ground-truth scan a pool also holds: its name, its ALTO file, its collection and its split."""

    scan: str
    truth: Path
    collection: str
    split: str


class CalibrationScan(BaseModel):
    """A ground-truth scan in a pool and who wrote the pool's labels of it."""

    scan: str
    collection: str
    split: str
    source: str


class CalibrationLine(BaseModel):
    """An old line and the true line it was matched to: their texts, their boxes' overlap, the old text's errors
    against the truth, whether a number in it reads otherwise, and the old line's size and density."""

    scan: str
    old: str
    truth: str
    iou: float
    errors: int
    cer: float
    numbers_differ: bool
    width: float
    height: float
    chars_per_100px: float


class CalibrationPage(BaseModel):
    """A scan's old label against its truth: lines on each side, matched, missed by the old pipeline and found by it
    alone (``astray``), the error rate over the matched lines, and how the two sets of outlines cover each other."""

    scan: str
    collection: str
    split: str
    old_lines: int
    true_lines: int
    matched: int
    missed: int
    astray: int
    cer_matched: float
    old_on_true: float
    true_covered: float
    mask_iou: float


def box_iou(a: Box, b: Box) -> float:
    """The overlap of two boxes: their intersection over their union."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match_lines(old: tuple[AltoLine, ...], true: tuple[AltoLine, ...]) -> list[tuple[int, int, float]]:
    """Old and true lines paired one to one, the most overlapping pairs first, as (old index, true index, overlap);
    a pair overlapping less than :data:`MATCH_IOU` is no pair."""
    overlaps = sorted(
        ((box_iou(o.box, t.box), i, j) for i, o in enumerate(old) for j, t in enumerate(true)), reverse=True
    )
    used_old: set[int] = set()
    used_true: set[int] = set()
    pairs = []
    for iou, i, j in overlaps:
        if iou < MATCH_IOU:
            break
        if i in used_old or j in used_true:
            continue
        used_old.add(i)
        used_true.add(j)
        pairs.append((i, j, iou))
    return pairs


def outline_mask(page: AltoPage) -> np.ndarray:
    """The page's line outlines filled, on a grid :data:`MASK_SCALE` times coarser than the scan."""
    image = Image.new("1", (int(page.width) // MASK_SCALE + 1, int(page.height) // MASK_SCALE + 1))
    draw = ImageDraw.Draw(image)
    for line in page.lines:
        draw.polygon([(x / MASK_SCALE, y / MASK_SCALE) for x, y in line.outline], fill=1)
    return np.asarray(image, dtype=bool)


def compare(scan: TruthScan, old: AltoPage, true: AltoPage) -> tuple[CalibrationPage, list[CalibrationLine]]:
    """A scan's old label against its truth: the page's row and a row for each matched line."""
    pairs = match_lines(old.lines, true.lines)
    lines = []
    for i, j, iou in pairs:
        ours, theirs = old.lines[i], true.lines[j]
        errors = levenshtein(ours.text, theirs.text)
        lines.append(
            CalibrationLine(
                scan=scan.scan,
                old=ours.text,
                truth=theirs.text,
                iou=iou,
                errors=errors,
                cer=errors / max(1, len(theirs.text)),
                numbers_differ=numbers_differ(ours.text, theirs.text),
                width=ours.width,
                height=ours.height,
                chars_per_100px=100 * len(ours.text) / max(1.0, ours.width),
            )
        )
    old_mask = outline_mask(old.model_copy(update={"width": true.width, "height": true.height}))
    true_mask = outline_mask(true)
    inter = int(np.logical_and(old_mask, true_mask).sum())
    page = CalibrationPage(
        scan=scan.scan,
        collection=scan.collection,
        split=scan.split,
        old_lines=len(old.lines),
        true_lines=len(true.lines),
        matched=len(pairs),
        missed=len(true.lines) - len(pairs),
        astray=len(old.lines) - len(pairs),
        cer_matched=sum(line.errors for line in lines) / max(1, sum(len(line.truth) for line in lines)),
        old_on_true=inter / max(1, int(old_mask.sum())),
        true_covered=inter / max(1, int(true_mask.sum())),
        mask_iou=inter / max(1, int(np.logical_or(old_mask, true_mask).sum())),
    )
    return page, lines


def truth_scans(ground_truth: Path, pool: Pool, test_list: str) -> list[TruthScan]:
    """The ground truth's scans whose volume ``pool`` holds and whose label file is in it, by name; a scan
    ``test_list`` names in its volume is a test scan."""
    volumes = {volume.name for volume in pool.volumes()}
    scans = {}
    for truth in sorted(ground_truth.glob(f"{COLLECTION_PREFIX}*/*/*/alto/*.xml")):
        if truth.name.startswith("._") or volume_of(truth.stem) not in volumes or truth.stem in scans:
            continue
        if not pool.label_file(truth.stem).exists():
            continue
        collection = collection_of(truth)
        scans[truth.stem] = TruthScan(
            scan=truth.stem, truth=truth, collection=collection, split=gt_split(truth.resolve(), test_list).value
        )
    return list(scans.values())


class Calibration(BaseModel):
    """A pool's calibration: every ground-truth scan it holds, and the pages and lines of those the old pipeline
    labelled."""

    scans: list[CalibrationScan]
    pages: list[CalibrationPage]
    lines: list[CalibrationLine]


def calibrate(ground_truth: Path, pool: Pool, test_list: str) -> Calibration:
    """Measure ``pool``'s old labels against the ground truth on the scans both hold."""
    result = Calibration(scans=[], pages=[], lines=[])
    for scan in truth_scans(ground_truth, pool, test_list):
        old = read_alto(pool.label_file(scan.scan))
        source = old.writer or "other"
        result.scans.append(
            CalibrationScan(scan=scan.scan, collection=scan.collection, split=scan.split, source=source)
        )
        if source != OLD_PIPELINE:
            continue
        page, lines = compare(scan, old, read_alto(scan.truth))
        result.pages.append(page)
        result.lines += lines
    return result


def write_calibration(calibration: Calibration, out: Path) -> None:
    """Write a calibration as three Lance tables in ``out``: ``scans.lance``, ``pages.lance`` and ``lines.lance``."""
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "scans.lance", calibration.scans, CalibrationScan)
    write_rows(out / "pages.lance", calibration.pages, CalibrationPage)
    write_rows(out / "lines.lance", calibration.lines, CalibrationLine)
