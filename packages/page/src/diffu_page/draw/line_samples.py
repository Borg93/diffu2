"""Line samples: the hand given box by box. Each line box of a target gets a crop of a real line in the hand to copy,
cut from a source page at the target's writing scale and placed on the DiT's position grid against that box
(``diffu_moss.transformer.token_positions``' offsets), each crop a reference of its own.

The source is the target's own page (or another window of it), its lines that the target does not hold: one page is
almost always one hand, so in training the crops are the target's scribe for certain, and none shows a line the target
draws. In drawing it is a sample page with its line boxes (a page job's ``reference.lines``).

A crop is its source line's slot (one line pitch tall, :func:`diffu_page.draw.sample.line_pitch`) scaled so the
source's pitch is the target's, its height rounded onto the latent grid, its width the source line's but at most its
target box's (and ``max_width``), floored onto the grid, cut from the line's left end. It lies over its box
(``over``: its top-left cell at the box's) or beside the page in the box's rows (``beside``: its last column just
left of the page's first, as ``above`` lays a sample just above the page), so a crop is tied to its line by its rows,
and laid over it only if asked.

A sample per text region (:func:`plan_region_crops`) gives each handwritten text region of the target a crop of another
text region of the source instead: a region is the lines of one PAGE ``TextRegion`` (a line's ``block``; a table's
lines are one region, by their table), its box the bounding box of its lines' slots. A region's crop is cut from a
source region none of whose lines the target region holds (on a whole page drawn from itself: another region, never
its own), scaled so the source region's pitch is the target region's, at most as wide and tall as the target region's
box (and ``max_width``, ``max_height``), each side floored onto the grid, cut from the source region's top-left
corner; it lies over the region's box (its top-left cell at the box's) or beside the page in the box's rows.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from PIL import Image
from pydantic import BaseModel, ConfigDict

from diffu_moss.transformer import CropPlace
from diffu_page.captions.prompt import LineRecord
from diffu_page.draw.sample import line_pitch
from diffu_page.page.crops.grid import LATENT_GRID
from diffu_page.page.layout.model import PageBox, Production


class LineCrop(BaseModel):
    """One line's or text region's crop and where it lies: the source line or region it is cut from and the target
    line or region whose box it styles (lines' ids, regions' blocks), its rectangle on the source image (px), its size (px, on the latent grid) and its centre's offset
    from the target's centre in latent rows and columns (the DiT's ``reference_offsets``)."""

    model_config = ConfigDict(frozen=True)

    source: str
    target: str
    rect: PageBox  # x0, y0, x1, y1 px of the source image
    size: tuple[int, int]  # (width, height) px
    offset: tuple[float, float]  # (rows, columns)

    @property
    def tokens(self) -> int:
        """The tokens the DiT reads for it."""
        return (self.size[0] // LATENT_GRID) * (self.size[1] // LATENT_GRID)


def _written(lines: Sequence[LineRecord]) -> list[LineRecord]:
    """The handwritten lines that hold text."""
    return [line for line in lines if line.production is Production.HANDWRITTEN and line.text.strip()]


def region_of(line: LineRecord) -> str:
    """The text region a line is written in: its table for a table cell's line, else its block."""
    return f"table|{line.cell.table}" if line.cell is not None else line.block


def _written_regions(lines: Sequence[LineRecord]) -> dict[str, list[LineRecord]]:
    """The handwritten lines that hold text, by their region, in the order the regions first come."""
    regions: dict[str, list[LineRecord]] = {}
    for line in _written(lines):
        regions.setdefault(region_of(line), []).append(line)
    return regions


def _box(lines: Sequence[LineRecord], size: tuple[int, int]) -> tuple[float, float, float, float]:
    """The bounding box ``x0, y0, x1, y1`` px, of an image of ``size`` (width, height), of ``lines``' boxes."""
    width, height = size
    return (
        min(line.bbox[1] for line in lines) / 1000 * width,
        min(line.bbox[0] for line in lines) / 1000 * height,
        max(line.bbox[3] for line in lines) / 1000 * width,
        max(line.bbox[2] for line in lines) / 1000 * height,
    )


def _offset(top_cell: int, left_cell: int, size: tuple[int, int], target_size: tuple[int, int]) -> tuple[float, float]:
    """The offset (rows, columns) from the target's centre of the centre of a crop of ``size`` px whose top-left cell
    is at ``top_cell``, ``left_cell`` of the target's grid."""
    rows, columns = size[1] // LATENT_GRID, size[0] // LATENT_GRID
    return (
        top_cell + rows / 2 - target_size[1] / LATENT_GRID / 2,
        left_cell + columns / 2 - target_size[0] / LATENT_GRID / 2,
    )


def _on_grid(side: float, *, floor: bool) -> int:
    """A side in px onto the latent grid, rounded (or ``floor``ed), at least one cell."""
    cells = math.floor(side / LATENT_GRID) if floor else round(side / LATENT_GRID)
    return max(1, cells) * LATENT_GRID


def plan_line_crops(
    source_size: tuple[int, int],
    source_lines: Sequence[LineRecord],
    target_size: tuple[int, int],
    target_lines: Sequence[LineRecord],
    *,
    place: CropPlace = "over",
    max_crops: int | None = None,
    max_width: int | None = None,
    seed: int = 0,
) -> list[LineCrop]:
    """The crops for a target of ``target_size`` (width, height) px whose lines are ``target_lines``, cut from a
    source image of ``source_size`` whose lines are ``source_lines`` (each line's box 0-1000 of its own image, y
    first): one for each of the target's handwritten lines in order (the first ``max_crops``), each from a source line
    whose text the target does not hold, those taken in an order shuffled by ``seed`` and again from the first when
    the target has more lines (the module's docstring for their size and place).

    Returns:
        The crops, in the target's line order; none when the source holds no line the target does not.

    Raises:
        ValueError: If the target has no handwritten line (none to give a crop to, nor to measure its pitch by).
    """
    targets = _written(target_lines)
    if not targets:
        raise ValueError("the target has no handwritten line to give a crop to")
    texts = {line.text for line in target_lines}
    sources = [line for line in _written(source_lines) if line.text not in texts]
    if not sources:
        return []
    random.Random(seed).shuffle(sources)
    target_width, target_height = target_size
    source_width, source_height = source_size
    pitch = line_pitch([line.bbox for line in targets], target_height)
    scale = pitch / line_pitch([line.bbox for line in sources], source_height)  # target px per source px
    crops = []
    for k, target in enumerate(targets[:max_crops]):
        source = sources[k % len(sources)]
        y0, x0, y1, x1 = (
            edge / 1000 * side for edge, side in zip(source.bbox, (source_height, source_width) * 2, strict=True)
        )
        ty0, tx0, ty1, tx1 = (
            edge / 1000 * side for edge, side in zip(target.bbox, (target_height, target_width) * 2, strict=True)
        )
        widest = min((x1 - x0) * scale, tx1 - tx0, math.inf if max_width is None else max_width)
        width, height = _on_grid(widest, floor=True), _on_grid((y1 - y0) * scale, floor=False)
        across, down = min(width / scale, source_width), min(height / scale, source_height)
        left = min(max(0.0, x0), source_width - across)
        top = min(max(0.0, (y0 + y1 - down) / 2), source_height - down)
        rows, columns = height // LATENT_GRID, width // LATENT_GRID
        top_cell = round((ty0 + ty1) / 2 / LATENT_GRID - rows / 2)
        left_cell = round(tx0 / LATENT_GRID) if place == "over" else -columns
        offset = _offset(top_cell, left_cell, (width, height), target_size)
        crops.append(
            LineCrop(
                source=source.id,
                target=target.id,
                rect=(left, top, left + across, top + down),
                size=(width, height),
                offset=offset,
            )
        )
    return crops


def plan_region_crops(
    source_size: tuple[int, int],
    source_lines: Sequence[LineRecord],
    target_size: tuple[int, int],
    target_lines: Sequence[LineRecord],
    *,
    place: CropPlace = "over",
    max_width: int = 512,
    max_height: int = 512,
    seed: int = 0,
) -> list[LineCrop]:
    """The crops for a target of ``target_size`` (width, height) px whose lines are ``target_lines``, cut from a
    source image of ``source_size`` whose lines are ``source_lines`` (each line's box 0-1000 of its own image, y
    first): one for each of the target's handwritten text regions in order, each from a source region none of whose
    lines' texts that region holds, those taken in an order shuffled by ``seed`` and again from the first when the
    target has more regions (the module's docstring for their size and place). The source may be the target itself.

    Returns:
        The crops, in the target's region order; none for a region the source holds no other region for.

    Raises:
        ValueError: If the target has no handwritten line (no region to give a crop to).
    """
    targets = _written_regions(target_lines)
    if not targets:
        raise ValueError("the target has no handwritten line to give a crop to")
    regions = list(_written_regions(source_lines).items())
    random.Random(seed).shuffle(regions)
    source_width, source_height = source_size
    crops = []
    for k, (target, lines) in enumerate(targets.items()):
        texts = {line.text for line in lines}
        others = [(name, region) for name, region in regions if texts.isdisjoint(line.text for line in region)]
        if not others:
            continue
        source, region = others[k % len(others)]
        pitch = line_pitch([line.bbox for line in lines], target_size[1])
        scale = pitch / line_pitch([line.bbox for line in region], source_height)  # target px per source px
        x0, y0, x1, y1 = _box(region, source_size)
        tx0, ty0, tx1, ty1 = _box(lines, target_size)
        width = _on_grid(min((x1 - x0) * scale, tx1 - tx0, max_width), floor=True)
        height = _on_grid(min((y1 - y0) * scale, ty1 - ty0, max_height), floor=True)
        across, down = min(width / scale, source_width), min(height / scale, source_height)
        left = min(max(0.0, x0), source_width - across)
        top = min(max(0.0, y0), source_height - down)
        columns = width // LATENT_GRID
        top_cell = round(ty0 / LATENT_GRID)
        left_cell = round(tx0 / LATENT_GRID) if place == "over" else -columns
        crops.append(
            LineCrop(
                source=source,
                target=target,
                rect=(left, top, left + across, top + down),
                size=(width, height),
                offset=_offset(top_cell, left_cell, (width, height), target_size),
            )
        )
    return crops


def cut_line_crops(source: Image.Image, crops: Sequence[LineCrop]) -> list[Image.Image]:
    """``crops`` cut from ``source``, each its rectangle at its size."""
    image = source.convert("RGB")
    return [image.resize(crop.size, Image.Resampling.LANCZOS, box=crop.rect) for crop in crops]
