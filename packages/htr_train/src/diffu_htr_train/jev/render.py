"""Our renderer of Jev's candidate texts, frozen once pairs are built with it (a new renderer is a new input
distribution): a candidate drawn black on white on one line, as tall and as wide as its crop, in one font at the
largest size that fits both (no smaller than :data:`MIN_SIZE`), laid out by FreeType's basic layout (not Raqm's, which
may or may not be installed). Its version (:func:`renderer_version`) names the rule, the font's digest and the Pillow
and FreeType that draw it; every pair's meta records it, and a Jev run refuses pairs of another."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, features

RULE = "htr-train-jev-render-1"  # the rule below; change it with the rule
MARGIN = 4  # px left of the text
MIN_SIZE = 8  # pt: a long candidate is drawn this small, and runs over the right edge


def renderer_version(font: Path) -> str:
    """The renderer's version: its rule, the first 12 hex digits of the font file's sha256, and the Pillow and
    FreeType versions (``htr-train-jev-render-1+<font>+pillow-<v>+freetype-<v>``)."""
    digest = hashlib.sha256(font.read_bytes()).hexdigest()[:12]
    return f"{RULE}+{digest}+pillow-{version('pillow')}+freetype-{features.version('freetype2')}"


def _face(font: Path, points: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font), points, layout_engine=ImageFont.Layout.BASIC)


def _fitting_font(text: str, font: Path, size: tuple[int, int]) -> ImageFont.FreeTypeFont:
    """The font at the largest size, from 70 % of the height down, whose ``text`` fits the width less the margins."""
    width, height = size
    points = max(MIN_SIZE, round(height * 0.7))
    face = _face(font, points)
    while points > MIN_SIZE and face.getlength(text) > width - 2 * MARGIN:
        points -= 1
        face = _face(font, points)
    return face


def render_text(text: str, font: Path, size: tuple[int, int]) -> Image.Image:
    """``text`` drawn black on white in an RGB image of ``size`` (width, height), on one line, left at
    :data:`MARGIN` and centred on the height."""
    image = Image.new("RGB", size, "white")
    face = _fitting_font(text, font, size)
    ImageDraw.Draw(image).text((MARGIN, size[1] / 2), text, fill="black", font=face, anchor="lm")
    return image
