"""What a scan looks like, in the words a caption uses: greyscale or colour, the paper's and the ink's colour."""

from __future__ import annotations

import math
from enum import StrEnum

import numpy as np
from PIL import Image
from pydantic import BaseModel

# Mean HSV saturation (0-1) below which a crop is a greyscale (microfilm) scan: those measure 0.0, colour
# scans 0.08-0.39.
GREY_SATURATION = 8.0 / 255


class Paper(StrEnum):
    """The paper colours a caption names: neutral paper by its brightness, tinted paper by its saturation."""

    GREY = "grey"
    LIGHT_GREY = "light grey"
    WHITE = "white"
    OFF_WHITE = "off-white"
    CREAM = "cream"
    YELLOWED = "yellowed"


class Ink(StrEnum):
    """The ink colours a caption names: grey to black, or brown, each by its brightness."""

    BLACK = "black"
    DARK_GREY = "dark grey"
    FADED_GREY = "faded grey"
    DARK_BROWN = "dark brown"
    BROWN = "brown"
    FADED_BROWN = "faded brown"


# Colour names by level, from the GT collections: colour paper is hue 28-48 degrees, saturation 0.09-0.35;
# ink is brown (hue 22-45) at value 0.23-0.49; greyscale paper is value 0.71-0.91 and its ink 0.05-0.45.
NEUTRAL_PAPER = 0.06  # paper saturation below which it is named by brightness


NEUTRAL_PAPERS = ((0.78, Paper.GREY), (0.88, Paper.LIGHT_GREY), (math.inf, Paper.WHITE))  # by value


TINTED_PAPERS = ((0.12, Paper.OFF_WHITE), (0.22, Paper.CREAM), (math.inf, Paper.YELLOWED))  # by saturation


BROWN_INK_SATURATION = 0.15


BROWN_INK_HUES = (15.0, 55.0)  # degrees


BROWN_INKS = ((0.30, Ink.DARK_BROWN), (0.42, Ink.BROWN), (math.inf, Ink.FADED_BROWN))  # by value


GREY_INKS = ((0.15, Ink.BLACK), (0.35, Ink.DARK_GREY), (math.inf, Ink.FADED_GREY))  # by value


class ScanLook(BaseModel):
    """What a crop's scan looks like, in plain words for its caption."""

    model_config = {"frozen": True}

    greyscale: bool
    paper: Paper
    ink: Ink


def scan_look(img: Image.Image) -> ScanLook:
    """Greyscale or colour, and names for the paper's colour (the median of the lightest 40% of pixels) and
    the ink's (the median of the darkest 2%)."""
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32).reshape(-1, 3) / 255
    value = hsv[:, 2]
    greyscale = float(hsv[:, 1].mean()) < GREY_SATURATION
    _, paper_saturation, paper_value = np.median(hsv[value >= np.percentile(value, 60)], axis=0)
    ink_hue, ink_saturation, ink_value = np.median(hsv[value <= np.percentile(value, 2)], axis=0)
    if greyscale:  # a microfilm's leftover saturation is scanner noise
        paper_saturation = ink_saturation = 0.0
    return ScanLook(
        greyscale=greyscale,
        paper=_paper_colour(paper_saturation, paper_value),
        ink=_ink_colour(ink_hue * 360, ink_saturation, ink_value),
    )


def _paper_colour(saturation: float, value: float) -> Paper:
    if saturation < NEUTRAL_PAPER:
        return band_name(value, NEUTRAL_PAPERS)
    return band_name(saturation, TINTED_PAPERS)


def _ink_colour(hue: float, saturation: float, value: float) -> Ink:
    low, high = BROWN_INK_HUES
    if saturation < BROWN_INK_SATURATION or not low <= hue <= high:
        return band_name(value, GREY_INKS)
    return band_name(value, BROWN_INKS)


def band_name[T](level: float, bands: tuple[tuple[float, T], ...]) -> T:
    """The name of the first band ``level`` is below; the last band's limit is infinite."""
    return next(name for limit, name in bands if level < limit)
