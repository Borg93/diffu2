"""What a scan looks like, in the words a caption uses: greyscale or colour, the paper's and the ink's colour."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image
from pydantic import BaseModel

# Mean HSV saturation (0-1) below which a crop is a greyscale (microfilm) scan: those measure 0.0, colour
# scans 0.08-0.39.
GREY_SATURATION = 8.0 / 255


# Colour names by level, from the GT collections: colour paper is hue 28-48 degrees, saturation 0.09-0.35;
# ink is brown (hue 22-45) at value 0.23-0.49; greyscale paper is value 0.71-0.91 and its ink 0.05-0.45.
NEUTRAL_PAPER = 0.06  # paper saturation below which it is named by brightness


NEUTRAL_PAPERS = ((0.78, "grey"), (0.88, "light grey"), (math.inf, "white"))  # by value


TINTED_PAPERS = ((0.12, "off-white"), (0.22, "cream"), (math.inf, "yellowed"))  # by saturation


BROWN_INK_SATURATION = 0.15


BROWN_INK_HUES = (15.0, 55.0)  # degrees


BROWN_INKS = ((0.30, "dark brown"), (0.42, "brown"), (math.inf, "faded brown"))  # by value


GREY_INKS = ((0.15, "black"), (0.35, "dark grey"), (math.inf, "faded grey"))  # by value


class ScanLook(BaseModel):
    """What a crop's scan looks like, in plain words for its caption."""

    model_config = {"frozen": True}

    greyscale: bool
    paper: str  # e.g. "yellowed", "light grey"
    ink: str  # e.g. "dark brown", "black"


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


def _paper_colour(saturation: float, value: float) -> str:
    if saturation < NEUTRAL_PAPER:
        return _band(value, NEUTRAL_PAPERS)
    return _band(saturation, TINTED_PAPERS)


def _ink_colour(hue: float, saturation: float, value: float) -> str:
    low, high = BROWN_INK_HUES
    if saturation < BROWN_INK_SATURATION or not low <= hue <= high:
        return _band(value, GREY_INKS)
    return _band(value, BROWN_INKS)


def _band(level: float, bands: tuple[tuple[float, str], ...]) -> str:
    """The name of the first band ``level`` is below; the last band's limit is infinite."""
    return next(name for limit, name in bands if level < limit)
