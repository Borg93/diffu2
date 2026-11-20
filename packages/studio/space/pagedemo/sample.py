"""A sample page's writing scale, so the model reads the sample at the scale of the page it draws.

The model was trained on samples at their target's writing scale (``legible``): a window's sample is the sample page
scaled so that its line pitch is the target's and cut at the target's size around the middle of its lines
(:func:`diffu_page.draw.sample.sample_window`), from the sample page's line boxes. An uploaded sample comes without
line boxes, so its pitch is measured here: the lines of a page of handwriting darken its rows at one period, the
strongest period of the rows' ink (:func:`line_pitch`). :func:`sample_lines` then states that pitch and where the
writing lies as line boxes, which the pipeline cuts the sample by. Its ``zoom`` shows the model the sample's writing
that many times larger than the page's own: more of the pen's detail, less of the page.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from diffu_page.page.crops.geometry import LayoutBox

MEASURED_SIDE = 1000  # px: the longer side of the copy the pitch is measured on
INK_BELOW = 0.78  # a pixel is ink when darker than this share of the paper around it
SHORTEST_PITCH, TALLEST_PITCH = 1 / 90, 1 / 5  # of the page's height: the pitches looked for
STRIPS = 6  # the page's width is measured in this many strips, each for its own rows
CLEAR_PERIOD = 0.12  # the rows' autocorrelation at a period that counts as one
FIRST_OF = 0.75  # a shorter period within this share of the strongest is taken for it (a pitch, not its double)


def _ink(image: Image.Image) -> np.ndarray:
    """Where ``image`` is ink: darker than the paper around it (its blurred copy), whatever the scan's brightness."""
    grey = image.convert("L")
    paper = grey.filter(ImageFilter.GaussianBlur(max(grey.size) / 40))
    return np.asarray(grey, dtype=np.float32) < INK_BELOW * np.asarray(paper, dtype=np.float32)


def line_pitch(image: Image.Image) -> float | None:
    """The line pitch of a page of handwriting, px of ``image``: the period at which its rows' ink repeats, from the
    rows' autocorrelation, taken strip by strip across the page (lines that slope smear the rows of the whole width,
    not of a strip); None when no period stands out (a blank page, a drawing, a single line)."""
    scale = MEASURED_SIDE / max(image.size)
    small = image.resize((max(8, round(image.width * scale)), max(8, round(image.height * scale))))
    ink = _ink(small)
    height, width = ink.shape
    window = max(3, height // 6) | 1
    correlation, weight = np.zeros(height), 0.0
    for strip in np.array_split(np.arange(width // 10, width - width // 10), STRIPS):
        rows = ink[:, strip].mean(axis=1)
        trend = np.convolve(np.pad(rows, window // 2, mode="edge"), np.ones(window) / window, mode="valid")
        wave = rows - trend
        energy = float(wave @ wave)
        if energy > 1e-9:  # a strip's own periods, each strip counted by how much it is written on
            correlation += np.correlate(wave, wave, mode="full")[height - 1 :]
            weight += energy
    if weight <= 0:
        return None
    correlation /= weight
    shortest, tallest = max(3, round(height * SHORTEST_PITCH)), round(height * TALLEST_PITCH)
    peaks = [
        lag
        for lag in range(shortest, min(tallest, height - 2))
        if correlation[lag] >= correlation[lag - 1] and correlation[lag] > correlation[lag + 1]
    ]
    if not peaks:
        return None
    strongest = max(correlation[lag] for lag in peaks)
    if strongest < CLEAR_PERIOD:
        return None
    first = next(lag for lag in peaks if correlation[lag] >= FIRST_OF * strongest)
    return first / scale


def _written_area(image: Image.Image) -> tuple[float, float, float, float]:
    """The part of ``image`` its writing lies in, as ``x0, y0, x1, y1`` on 0-1 of it: the rows and columns between
    which the middle nine tenths of its ink lie (the whole image when it has none)."""
    scale = MEASURED_SIDE / max(image.size)
    ink = _ink(image.resize((max(8, round(image.width * scale)), max(8, round(image.height * scale)))))
    if not ink.any():
        return 0.0, 0.0, 1.0, 1.0

    def span(counts: np.ndarray) -> tuple[float, float]:
        share = np.cumsum(counts) / counts.sum()
        low, high = int(np.searchsorted(share, 0.05)), int(np.searchsorted(share, 0.95))
        return low / len(counts), min(1.0, (high + 1) / len(counts))

    (x0, x1), (y0, y1) = span(ink.sum(axis=0)), span(ink.sum(axis=1))
    return x0, y0, x1, y1


def sample_lines(image: Image.Image, pitch: float, *, zoom: float = 1.0) -> list[LayoutBox]:
    """Line boxes (``y0, x0, y1, x1`` on 0-1000 of ``image``) that state the sample's line pitch ``pitch`` px and
    where its writing lies, for the pipeline to cut the sample by: two boxes one pitch tall, at the top and the bottom
    of the written area. With ``zoom`` the pitch stated is that much smaller than measured, so the sample is shown
    that many times larger than the page's writing.

    Raises:
        ValueError: If the pitch or the zoom is not positive.
    """
    if pitch <= 0 or zoom <= 0:
        raise ValueError(f"a line pitch ({pitch}) and a zoom ({zoom}) are positive")
    x0, y0, x1, y1 = _written_area(image)
    tall = max(1, round(1000 * pitch / zoom / image.height))
    left, right = round(1000 * x0), max(round(1000 * x0) + 1, round(1000 * x1))
    top = min(round(1000 * y0), 1000 - tall)
    bottom = max(top + tall, min(1000, round(1000 * y1)))
    return [(top, left, top + tall, right), (bottom - tall, left, bottom, right)]
