"""A scan and its text lines as the print/hand classifier reads them: the scan in grey, each line cut by its box and
reduced to its ink, 64 px high, so that neither the paper, stains, the light nor the scan's resolution tell print from
hand."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from diffu_page.layout import Layout, PageBox

LINE_HEIGHT = 64  # px: every line image's height
UPRIGHT = 1.5  # a box this many times taller than wide holds a line set sideways, as a column heading can be
SMALLEST = 8  # px: a box narrower or lower than this on the scan holds nothing to read
REDUCTIONS = (1, 2, 4, 8)  # the scales a scan can be decoded at: JPEG decodes them directly


def read_scan(path: Path, *, reduction: int = 1) -> np.ndarray:
    """The scan at ``path`` in grey (uint8), decoded at 1/``reduction`` of its size (one of :data:`REDUCTIONS`).

    Raises:
        ValueError: If ``reduction`` is not one of :data:`REDUCTIONS`, or the file is not an image OpenCV reads.
    """
    flags = {
        1: cv2.IMREAD_GRAYSCALE,
        2: cv2.IMREAD_REDUCED_GRAYSCALE_2,
        4: cv2.IMREAD_REDUCED_GRAYSCALE_4,
        8: cv2.IMREAD_REDUCED_GRAYSCALE_8,
    }
    if reduction not in flags:
        raise ValueError(f"a scan is decoded at 1/{'/'.join(map(str, REDUCTIONS))} of its size, not 1/{reduction}")
    scan = cv2.imread(str(path), flags[reduction])
    if scan is None:
        raise ValueError(f"{path} is not an image that can be read")
    return scan


def on_scan(box: PageBox, layout: Layout, scan: np.ndarray) -> PageBox:
    """``box`` (page px of ``layout``) in ``scan``'s pixels: a scan can be stored at another size than its page's."""
    sx, sy = scan.shape[1] / layout.width, scan.shape[0] / layout.height
    return (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)


def line_images(scan: np.ndarray, box: PageBox) -> list[np.ndarray]:
    """The views of a text line the classifier reads, from a grey ``scan`` and the line's ``box`` in its pixels: the
    box's ink on white, :data:`LINE_HEIGHT` px high (uint8, ink dark); a box standing upright (a column heading set
    sideways) turned both ways, as which way its text runs is not known; none for a box too small to read."""
    height, width = scan.shape
    x0, y0 = max(0, round(box[0])), max(0, round(box[1]))
    x1, y1 = min(width, round(box[2])), min(height, round(box[3]))
    if x1 - x0 < SMALLEST or y1 - y0 < SMALLEST:
        return []
    crop = scan[y0:y1, x0:x1]
    if y1 - y0 > UPRIGHT * (x1 - x0):
        return [_ink(np.rot90(crop, turns)) for turns in (1, 3)]
    return [_ink(crop)]


def _ink(crop: np.ndarray) -> np.ndarray:
    """A line's crop scaled to :data:`LINE_HEIGHT` and divided by its paper (what a closing wider than any stroke leaves
    of it, blurred), its contrast stretched from the darkest ink to the paper most of it shows: stains, uneven light and
    the paper's tone go, the strokes stay."""
    scale = LINE_HEIGHT / crop.shape[0]
    size = (max(1, round(crop.shape[1] * scale)), LINE_HEIGHT)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    line = cv2.resize(np.ascontiguousarray(crop), size, interpolation=interpolation).astype(np.float32)
    stroke_free = cv2.morphologyEx(line, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    paper = cv2.GaussianBlur(stroke_free, (0, 0), sigmaX=LINE_HEIGHT / 8)
    ratio = line / np.maximum(paper, 1.0)
    high = float(np.percentile(ratio, 60))  # the paper: a line's ink rarely covers 40 % of its box
    low = min(float(np.percentile(ratio, 1)), high - 0.25)  # a line of little ink is not stretched into noise
    return (np.clip((ratio - low) / (high - low), 0.0, 1.0) * 255).astype(np.uint8)
