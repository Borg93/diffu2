"""What a crop's handwriting looks like, measured from its pixels inside its line boxes: slant, pen width, letter
size, density, and the paper's and ink's colours.

BFL's FLUX guidance asks a caption to name a style by its visible properties (a hand's slant and stroke, colours as
hex) rather than by an era or an archive, which the model reads as a colour grade. These numbers are what the
described reference captions (``diffu_page.captions.restyle``) say about a reference's hand, and what the
hand-fidelity check compares between a drawing and the crop it should look like.

Each line is cut out of its box and scaled to ``LINE_HEIGHT`` pixels before it is measured, so the numbers do not
depend on the crop's resolution: a 0.5 MP reference and the 1 MP crop it was shrunk from measure alike. The boxes are
the crop's PAGE lines (0-1000 of the crop, y first, as the captions' layout rows): there is no fallback for an image
without them, because text bands found in an image's row ink profile cut through lines on most GT crops. CPU only,
numpy and PIL.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel

from diffu_page.captions.look import ScanLook, band_name, scan_look
from diffu_page.crops.geometry import pixel_box
from diffu_page.crops.grid import as_reference

LINE_HEIGHT = 80  # px a line is scaled to before it is measured: its strokes then span about 4-12 px
COARSE_SLANTS = range(-30, 64, 4)  # degrees searched first; then +-3 around the best, in 1 degree steps
MIN_INK_PIXELS = 30  # a scaled line with less ink than this does not count towards the slant
MIN_BOX = (8, 16)  # rows and columns a line box needs in the image to be measured

type Box = Sequence[int]  # y0, x0, y1, x1 on 0-1000 of the image

# The words a caption names a hand with, by level. Measured on 2,000 v2 training crops: slant
# quantiles 5/20/33/67/80/95% are 0/14/27/40/43/49 degrees, so the slant bands are named by angle; the stroke,
# letter and density bands are the terciles. Checked by eye on real lines from each band.
SLANT_WORDS = (
    (-5.0, "left-slanted"),
    (10.0, "upright"),
    (25.0, "slightly right-slanted"),
    (40.0, "right-slanted"),
    (math.inf, "steeply right-slanted"),
)
STROKE_WORDS = ((0.061, "fine"), (0.084, "medium-width"), (math.inf, "broad"))
LETTER_WORDS = ((0.26, "small"), (0.39, "medium-sized"), (math.inf, "large"))
DENSITY_WORDS = ((0.104, "widely spaced"), (0.146, "evenly spaced"), (math.inf, "closely packed"))


class HandLook(BaseModel):
    """A crop's handwriting, measured. Lengths are shares of the line height. A measure the image does not allow is
    None: no measurable line box, no ink in the line boxes (the density is then 0), no paper (an image of nothing but
    ink) or no ink at all."""

    model_config = {"frozen": True}

    slant: float | None  # degrees, positive when the strokes lean right
    stroke: float | None  # the pen's width
    letters: float | None  # the height of the letters' core band (x-height)
    density: float | None  # the share of a line's area covered by ink
    paper: str | None  # "#RRGGBB"
    ink: str | None  # "#RRGGBB"


class ReferenceLook(BaseModel):
    """What a reference crop looks like: its scan, paper and ink in words, and its hand measured."""

    model_config = {"frozen": True}

    scan: ScanLook
    hand: HandLook


class HandDistance(BaseModel):
    """How far one hand is from another, 0 where they measure alike: the slant in degrees, each length as the
    |log2| of the two lengths' ratio (1.0: one is twice the other), the colours as CIE76 colour differences (about
    2.3 is just noticeable). A measure one of the hands lacks (no measurable line, no ink) is None."""

    slant: float | None
    stroke: float | None
    letters: float | None
    density: float | None
    paper: float | None
    ink: float | None


# Grey levels between the two sides of Otsu's split below which an image holds no ink: Otsu always splits, paper grain
# too. Real line boxes measure 47 and up (1,265 boxes of five collections' held-out windows, faint inks and microfilm
# among them; median 105), blank paper about 16 (24 Oct).
MIN_INK_CONTRAST = 30


def otsu(gray: np.ndarray) -> int:
    """The grey level that best separates ink from paper (Otsu's threshold): pixels below it are ink. 0 (no pixel is
    ink) when the two sides differ by less than ``MIN_INK_CONTRAST``; an image of one grey level is all ink if it is
    black, else all paper."""
    hist = np.bincount(np.clip(gray, 0, 255).astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    between = (mu[-1] * omega - mu) ** 2 / np.maximum(omega * (1 - omega), 1e-12)
    split = int(np.argmax(between))  # the last level on the ink side
    ink_share = omega[split]
    if ink_share <= 0 or ink_share >= 1:  # one grey level: no split to judge
        return split + 1
    ink_mean, paper_mean = mu[split] / ink_share, (mu[-1] - mu[split]) / (1 - ink_share)
    return split + 1 if paper_mean - ink_mean >= MIN_INK_CONTRAST else 0


def hand_look(image: Image.Image, boxes: Sequence[Box]) -> HandLook:
    """The hand of ``image`` measured inside its line ``boxes``; the paper and ink colours over the whole image."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    lines = _scaled_lines(gray, boxes)
    slant = _slant(lines)
    tan = math.tan(math.radians(slant or 0.0))
    upright = [_sheared(line, tan) < threshold for line, threshold in lines]
    ink = gray < otsu(gray)
    return HandLook(
        slant=slant,
        stroke=_stroke(upright),
        letters=_letters(upright),
        # unsheared: the shear pads each row with paper in proportion to the slant
        density=float(np.mean([(line < threshold).mean() for line, threshold in lines])) if lines else None,
        paper=_hex(rgb[~ink]),
        ink=_hex(rgb[ink & (gray <= np.percentile(gray[ink], 50))]) if ink.any() else None,
    )


def slant_words(slant: float | None) -> str:
    """How a caption names a hand's slant: "right-slanted cursive", or plain "cursive" when it was not measured."""
    return "cursive" if slant is None else f"{band_name(slant, SLANT_WORDS)} cursive"


def hand_bands(hand: HandLook) -> tuple[str, str, str]:
    """The words a caption names a hand's pen, letters and spacing by: ("fine", "small", "closely packed").

    Raises:
        ValueError: If the hand has no ink in its line boxes: a caption would describe a hand that is not there.
    """
    if hand.stroke is None or hand.letters is None or hand.density is None:
        raise ValueError("a hand with no ink in its line boxes cannot be described")
    return (
        band_name(hand.stroke, STROKE_WORDS),
        band_name(hand.letters, LETTER_WORDS),
        band_name(hand.density, DENSITY_WORDS),
    )


def hand_words(hand: HandLook) -> str:
    """A hand in a caption's words, its visible features named: "right-slanted cursive, about 35 degrees, with fine
    pen strokes and small, closely packed letters".

    Raises:
        ValueError: If the hand has no ink in its line boxes (``hand_bands``).
    """
    stroke, letters, density = hand_bands(hand)
    slant = slant_words(hand.slant)
    if hand.slant is not None:
        slant += f", about {abs(5 * round(hand.slant / 5))} degrees,"
    return f"{slant} with {stroke} pen strokes and {letters}, {density} letters"


def reference_look(image: Image.Image, boxes: Sequence[Box]) -> ReferenceLook:
    """What the reference ``image`` looks like, its hand measured inside its line ``boxes``."""
    return ReferenceLook(scan=scan_look(image), hand=hand_look(image, boxes))


def measure_reference(path: Path, boxes: Sequence[Box]) -> ReferenceLook:
    """What the image at ``path`` looks like as a reference: shrunk to the reference size as training and generation
    draw from it (``as_reference``), its hand measured inside ``boxes``. Training captions and planned crops are both
    described through here, so a reference reads the same in both."""
    with Image.open(path) as image:
        return reference_look(as_reference(image), boxes)


def hand_distance(a: HandLook, b: HandLook) -> HandDistance:
    """How far hand ``a`` is from hand ``b``."""
    return HandDistance(
        slant=abs(a.slant - b.slant) if a.slant is not None and b.slant is not None else None,
        stroke=_log_ratio(a.stroke, b.stroke),
        letters=_log_ratio(a.letters, b.letters),
        density=_log_ratio(a.density, b.density),
        paper=_delta_e(a.paper, b.paper),
        ink=_delta_e(a.ink, b.ink),
    )


def _scaled_lines(gray: np.ndarray, boxes: Sequence[Box]) -> list[tuple[np.ndarray, int]]:
    """Each measurable line box cut out, scaled to ``LINE_HEIGHT`` rows, with its own ink threshold."""
    height, width = gray.shape
    lines: list[tuple[np.ndarray, int]] = []
    for box in boxes:
        left, top, right, bottom = (round(v) for v in pixel_box(box, (width, height)))
        if bottom - top < MIN_BOX[0] or right - left < MIN_BOX[1]:
            continue
        crop = Image.fromarray(gray[top:bottom, left:right].astype(np.uint8))
        scale = LINE_HEIGHT / crop.height
        scaled = np.asarray(
            crop.resize((max(MIN_BOX[1], round(crop.width * scale)), LINE_HEIGHT), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
        lines.append((scaled, otsu(scaled)))
    return lines


def _sheared(line: np.ndarray, tan: float) -> np.ndarray:
    """``line`` with each row shifted by (centre - row) * ``tan``, interpolated, so strokes leaning by atan(tan)
    stand upright; white beyond the edges."""
    rows, cols = line.shape
    pad = int(abs(tan) * rows / 2) + 2
    x = np.arange(-pad, cols + pad, dtype=np.float32)
    source = np.arange(cols, dtype=np.float32)
    out = np.empty((rows, x.size), np.float32)
    for row in range(rows):
        out[row] = np.interp(x - (row - rows / 2) * tan, source, line[row], left=255, right=255)
    return out


def _continuity(ink: np.ndarray) -> float:
    """The sum of squared column heights over the columns whose ink is one unbroken vertical run (Vinciarelli and
    Luettin's deslant criterion): largest when the strokes stand upright."""
    count = ink.sum(axis=0)
    rows = np.arange(ink.shape[0])[:, None]
    top = np.where(ink, rows, ink.shape[0]).min(axis=0)
    bottom = np.where(ink, rows, -1).max(axis=0)
    unbroken = (count > 0) & (count == bottom - top + 1)
    return float((count[unbroken].astype(np.float64) ** 2).sum())


def _slant(lines: list[tuple[np.ndarray, int]]) -> float | None:
    """The shear angle that makes the lines' strokes stand most upright, summed over the lines with ink (each
    normalised by its ink), searched coarse then fine; None when no line has ``MIN_INK_PIXELS`` of ink.

    The shear is interpolated, not rounded per row: a rounded shear makes unbroken columns likeliest where the shift
    is a whole pixel per row (45 degrees), and the estimates piled up there.
    """
    inked = [(line, threshold) for line, threshold in lines if (line < threshold).sum() > MIN_INK_PIXELS]

    def uprightness(angle: int) -> float:
        tan = math.tan(math.radians(angle))
        total = 0.0
        for line, threshold in inked:
            ink = _sheared(line, tan) < threshold
            total += _continuity(ink) / max(1, ink.sum())
        return total

    if not inked:
        return None
    coarse = max(COARSE_SLANTS, key=uprightness)
    return float(max(range(coarse - 3, coarse + 4), key=uprightness))


def _stroke(lines: list[np.ndarray]) -> float | None:
    """The interquartile mean of the horizontal ink runs in the upright lines, as a share of the line height: the
    pen's width across its upright strokes, without the joins and flourishes in the long tail; None without ink."""
    runs = [
        np.nonzero(edges == -1)[1] - np.nonzero(edges == 1)[1]
        for edges in (np.diff(ink.astype(np.int8), axis=1, prepend=0, append=0) for ink in lines)
    ]
    joined = np.sort(np.concatenate(runs)) if runs else np.array([])
    if not joined.size:
        return None
    middle = joined[len(joined) // 4 : max(len(joined) // 4 + 1, 3 * len(joined) // 4)]
    return float(middle.mean()) / LINE_HEIGHT


def _letters(lines: list[np.ndarray]) -> float | None:
    """The median share of an inked line's rows in its core band (the rows inked at least half as much as its
    busiest); None without ink."""
    shares = [float((ink.mean(axis=1) >= 0.5 * ink.mean(axis=1).max()).mean()) for ink in lines if ink.any()]
    return float(np.median(shares)) if shares else None


def _log_ratio(a: float | None, b: float | None) -> float | None:
    return abs(math.log2(a / b)) if a and b else None


def _hex(pixels: np.ndarray) -> str | None:
    """The median colour of ``pixels`` (N x RGB) as "#RRGGBB"; None when there are none."""
    if not len(pixels):
        return None
    red, green, blue = (round(float(v)) for v in np.median(pixels, axis=0))
    return f"#{red:02X}{green:02X}{blue:02X}"


def lab(hex_colour: str) -> tuple[float, float, float]:
    """CIE L*a*b* (D65) of an sRGB hex colour: lightness L* (0-100), then a* (green-red) and b* (blue-yellow)."""
    rgb = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    x = (0.4124 * linear[0] + 0.3576 * linear[1] + 0.1805 * linear[2]) / 0.95047
    y = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    z = (0.0193 * linear[0] + 0.1192 * linear[1] + 0.9505 * linear[2]) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116

    return 116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))


def _delta_e(a: str | None, b: str | None) -> float | None:
    return math.dist(lab(a), lab(b)) if a is not None and b is not None else None
