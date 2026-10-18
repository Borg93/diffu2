"""Draw the ground-truth labels over a generated page, so polygons + transcriptions can be eyeballed.

The studio emits PAGE/ALTO/JSON labels; this renders them back ON the page (line polygon, baseline, word
boxes, and the transcription text) so a human can confirm the labels actually match the ink. Uses a TrueType
font so Swedish å ä ö render in the overlay text (Hershey/cv2 can't).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from diffu_page.labels import LineLabel

_LINE = (255, 40, 40, 255)  # line polygon — red
_WORD = (40, 120, 255, 160)  # word boxes — blue
_BASE = (0, 190, 0, 230)  # baseline — green
_TEXT = (200, 0, 0, 255)  # transcription — red


def render_label_overlay(
    page: Image.Image, lines: list[LineLabel], *, show_words: bool = True
) -> Image.Image:
    """Return a copy of ``page`` with each line's polygon, baseline, word boxes and transcription drawn on."""
    out = page.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    font_size = max(12, out.height // 90)
    font = _load_font(font_size)
    for line in lines:
        poly = [(int(x), int(y)) for x, y in line.polygon]
        if len(poly) >= 3:
            draw.polygon(poly, outline=_LINE)
        if show_words:
            for word in line.words:
                wp = [(int(x), int(y)) for x, y in word.polygon]
                if len(wp) >= 3:
                    draw.polygon(wp, outline=_WORD)
        if len(line.baseline) >= 2:
            draw.line([(int(x), int(y)) for x, y in line.baseline], fill=_BASE, width=1)
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        if xs and ys:
            draw.text((min(xs), max(0, min(ys) - font_size - 2)), line.text, fill=_TEXT, font=font)
    return out


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()
