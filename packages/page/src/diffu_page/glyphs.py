"""The glyph map: a window's line texts drawn as plain glyphs into their boxes, for diffu to trace (experiment E2).

diffu reads the map as a reference image (FLUX.3's reference stream, E2 and E2b) or, a glyph model (``diffu-init
--glyph``, E2c), beside its latents on channels, each image token with its own cell of the map; either way the words it
draws are spelled by construction and each line is as long as its box. Black DejaVu Sans on white: the map carries the
text and its place, never a hand. It is drawn at the target's own size, which must lie on the latent grid, because
FLUX.3's preprocessing center-crops a reference that does not, and a glyph model's map must lie on the target's grid.

DejaVu Sans covers 165 of the 167 characters in our transcriptions; it lacks ꝕ and a stray Malayalam ഽ. A character
the font lacks is left out rather than drawn as the font's empty box, which the model would learn to copy.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

from diffu_page.captions.prompt import LineRecord
from diffu_page.crops.geometry import SLOT_ABOVE_BASELINE, SLOT_SIDE_MARGIN, pixel_box
from diffu_page.crops.grid import LATENT_GRID

GLYPH_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
EM_PER_PITCH = 0.6  # the glyphs' em size, as a share of the line's slot height (its pitch)
_PROBE_SIZE = 40  # px: the size characters are checked for a glyph at


@cache
def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(GLYPH_FONT), size)


@cache
def _has_glyph(char: str) -> bool:
    """Whether the font draws ``char`` itself, not its empty box for a missing glyph (U+10FFFD has none)."""
    return char.isspace() or _probe(char) != _probe("\U0010fffd")


@cache
def _probe(char: str) -> bytes:
    """``char``'s pixels at the probe size."""
    image = Image.new("L", (2 * _PROBE_SIZE, 2 * _PROBE_SIZE), 0)
    ImageDraw.Draw(image).text((0, 0), char, fill=255, font=_font(_PROBE_SIZE))
    return image.tobytes()


def drawable(text: str) -> str:
    """``text`` without the characters the glyph font lacks."""
    return "".join(char for char in text if _has_glyph(char))


def glyph_map(lines: Sequence[LineRecord], size: tuple[int, int]) -> Image.Image:
    """The glyph map of a window of ``size`` (``width, height``) pixels: each line's text in black on white, from its
    box's left edge to its right (less the slot's side margins), its baseline at the line's own or, in records
    without one, where the slot puts it.

    Raises:
        ValueError: If a side of ``size`` is not a multiple of the latent grid.
    """
    if size[0] % LATENT_GRID or size[1] % LATENT_GRID:
        raise ValueError(f"a glyph map is drawn at the target's size, a multiple of {LATENT_GRID} px; got {size}")
    page = Image.new("L", size, 255)
    for line in lines:
        text = drawable(line.text).strip()
        if text:
            page = ImageChops.darker(page, _line_layer(text, line, size))
    return page.convert("RGB")


def _line_layer(text: str, line: LineRecord, size: tuple[int, int]) -> Image.Image:
    """A white layer of the window's size with ``text`` set into ``line``'s box."""
    x0, y0, x1, y1 = pixel_box(line.bbox, size)
    pitch = y1 - y0
    baseline = line.baseline * size[1] / 1000 if line.baseline is not None else y0 + SLOT_ABOVE_BASELINE * pitch
    font = _font(max(1, round(EM_PER_PITCH * pitch)))
    above = round(SLOT_ABOVE_BASELINE * pitch)
    strip = Image.new("L", (max(1, round(font.getlength(text))), max(1, round(pitch))), 255)
    ImageDraw.Draw(strip).text((0, above), text, fill=0, font=font, anchor="ls")
    left, _, right, _ = ImageChops.invert(strip).getbbox() or (0, 0, strip.width, 0)
    strip = strip.crop(
        (left, 0, right, strip.height)
    )  # the ink's own width: no side bearings, rows kept for the baseline
    margin = SLOT_SIDE_MARGIN * pitch
    width = max(1, round(x1 - x0 - 2 * margin))
    layer = Image.new("L", size, 255)
    layer.paste(
        strip.resize((width, strip.height), Image.Resampling.LANCZOS), (round(x0 + margin), round(baseline) - above)
    )
    return layer
