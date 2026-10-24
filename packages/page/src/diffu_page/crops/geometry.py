"""Where a page's lines are and how a page is cut into crops at the scale the FLUX.3 LoRAs train on.

Crops cover consecutive whole lines of one text column (a wide region plus the margin notes, headings and page
numbers beside it), scaled to a ~64 px line pitch and saved at the exact size of a NeMo bucket. A table is one region:
its cells' lines side by side make one row, so its windows hold whole rows across the table, and its pitch is measured
from row to row. A crop keeps the parts of the page's ruled lines inside it.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from itertools import pairwise
from operator import attrgetter

from pydantic import BaseModel, Field

from diffu_page.layout.model import ROLE_NAMES, CellRef, Layout, LineBox, LineKind, Production

type Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in page pixels


# NeMo's image buckets for --resolution_preset 1024p (MultiTierBucketCalculator, quantization 64), (w, h).
# Preprocessing fits an image to the bucket nearest its aspect by a cover-resize and a center crop, which
# would trim a crop's edge lines out from under their boxes; a crop saved at a bucket's exact size is not cut.
BUCKETS: list[tuple[int, int]] = [
    (768, 1344), (832, 1216), (896, 1152), (1024, 1024), (1152, 896),
    (1216, 832), (1280, 768), (1344, 768), (1408, 704), (1472, 640),
]  # fmt: skip


WIDEST_BUCKET = max(width for width, _ in BUCKETS)


# A line's box is its slot on the page, in line pitches: PAGE polygons include ascenders and flourishes, so
# their own boxes overlap the neighbouring lines'.
SLOT_ABOVE_BASELINE = 0.7


SLOT_BELOW_BASELINE = 0.3


SLOT_SIDE_MARGIN = 0.15


WHOLE_LINE = 0.9  # share of a line's width inside a crop that counts as the whole line


CUT_LINE = 0.15  # above this (and below WHOLE_LINE) a line is cut at the side and the crop is dropped


SAME_ROW = 0.3  # of a table's median line height: baselines closer than this are one row of cells


SHORTEST_RULE = 1.0  # line pitches: a ruled line with less than this inside a crop is left out of it


RULE_THICKNESS = 2  # the thinnest a ruled line's layout box is, on 0-1000


class CropSettings(BaseModel):
    """How pages are cut into crops (pixels are after scaling unless noted)."""

    line_spacing: float = 64.0  # target line pitch
    min_line_spacing: float = 40.0  # crops scaled below this pitch are skipped
    min_merge_spacing: float = 48.0  # smallest pitch a margin note may shrink its column to
    min_width: int = 640  # narrower regions are only cropped together with a wide one
    min_lines: int = 3
    overlap_lines: int = 2  # lines shared by consecutive windows of a column


class Line(BaseModel):
    """A text line on a page: its outline and baseline height (page px), its text, kind, source block, how it was
    made and, in a table, its cell."""

    polygon: list[tuple[float, float]]
    baseline_y: float
    text: str
    kind: LineKind = LineKind.BODY
    source: str = ""  # the region / block it came from, when a caller tracks reading order by block
    production: Production = Production.HANDWRITTEN
    cell: CellRef | None = None

    @property
    def role(self) -> LineKind | None:
        """The line's role on the page (a margin note, a page number, ...); None for running text and table cells."""
        return self.kind if self.kind in ROLE_NAMES else None

    @property
    def box(self) -> Box:
        """The line's bounding box."""
        xs, ys = zip(*self.polygon, strict=True)
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def middle_x(self) -> float:
        """The x of the line's middle."""
        return (self.box[0] + self.box[2]) / 2


class Region(BaseModel):
    """A text region's lines, or a whole table's."""

    lines: list[Line]

    @property
    def is_table(self) -> bool:
        """Whether the region is a table: its lines are cells'."""
        return any(line.cell is not None for line in self.lines)

    @property
    def box(self) -> Box:
        """The bounding box of the region's lines."""
        boxes = [line.box for line in self.lines]
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )


def regions_of(layout: Layout) -> list[Region]:
    """The layout's text lines grouped into the regions crops are cut from: a table's lines by table, the others by
    region and block, in reading order, each region's lines top to bottom. A table line without a cell (an old
    layout) is left out."""
    grouped: dict[tuple[str, str], list[Line]] = {}
    for box in layout.boxes:
        if box.text.strip() and (box.region is not LineKind.TABLE or box.cell is not None):
            grouped.setdefault(_region_key(box), []).append(_line(box))
    return [Region(lines=sorted(lines, key=attrgetter("baseline_y"))) for lines in grouped.values()]


def _region_key(box: LineBox) -> tuple[str, str]:
    """The region a line is cut in: its table for a table cell's line, else its region and block."""
    if box.cell is not None:
        return LineKind.TABLE, box.cell.table
    return box.region, box.block


def _line(box: LineBox) -> Line:
    x0, y0, x1, y1 = box.x, box.y, box.x + box.w, box.y + box.h
    return Line(
        polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        baseline_y=box.baseline_y,
        text=box.text,
        kind=box.region,
        source=box.source,
        production=box.production,
        cell=box.cell,
    )


class Column(BaseModel):
    """A wide text region and the narrow regions (margin notes, headings, page numbers) that joined it."""

    anchor: Box  # the wide region's own box
    left: float
    right: float
    lines: list[Line]

    @classmethod
    def of(cls, region: Region) -> Column:
        """A column anchored on a wide region."""
        box = region.box
        return cls(anchor=box, left=box[0], right=box[2], lines=list(region.lines))

    def gap_to(self, box: Box) -> float:
        """Vertical distance from the anchor region to ``box``; 0 when they overlap."""
        return max(0.0, max(self.anchor[1], box[1]) - min(self.anchor[3], box[3]))

    def width_with(self, box: Box) -> float:
        """The column's width if ``box`` joined it."""
        return max(self.right, box[2]) - min(self.left, box[0])

    def add(self, region: Region) -> None:
        """Join a narrow region's lines to the column."""
        self.lines.extend(region.lines)
        self.left = min(self.left, region.box[0])
        self.right = max(self.right, region.box[2])


class PageGeometry(BaseModel):
    """What cropping a column needs to know about its page: size (px), line pitch, every line and every ruled line."""

    size: tuple[int, int]
    pitch: float
    lines: list[Line]
    rules: list[Box] = Field(default_factory=list)


class ColumnFrame(BaseModel):
    """A column's padded x-range (page px) and the bucket its crops are saved at."""

    left: float
    right: float
    bucket: tuple[int, int]

    @property
    def scale(self) -> float:
        """Crop pixels per page pixel."""
        return self.bucket[0] / (self.right - self.left)

    @property
    def span(self) -> float:
        """The crop height in page pixels."""
        return self.bucket[1] / self.scale


class Crop(BaseModel):
    """A crop of a page: its rectangle (page px), bucket size, whole lines, the lines its top or bottom cuts and the
    parts of the page's ruled lines inside it."""

    rect: Box
    size: tuple[int, int]
    pitch: float
    lines: list[Line]
    fragments: list[Line]
    rules: list[Box] = Field(default_factory=list)

    def layout_box(self, box: Box) -> list[int]:
        """A page rectangle as a FLUX.3 layout box: ``[y_min, x_min, y_max, x_max]`` on 0-1000, clamped."""
        x0, y0, x1, y1 = self.rect
        width, height = x1 - x0, y1 - y0
        relative = (
            (box[1] - y0) / height,
            (box[0] - x0) / width,
            (box[3] - y0) / height,
            (box[2] - x0) / width,
        )
        return [min(1000, max(0, round(1000 * v))) for v in relative]

    def slot(self, line: Line) -> list[int]:
        """A line's layout box: its x-extent plus a small margin, one line pitch around its baseline."""
        margin = SLOT_SIDE_MARGIN * self.pitch
        top = line.baseline_y - SLOT_ABOVE_BASELINE * self.pitch
        bottom = line.baseline_y + SLOT_BELOW_BASELINE * self.pitch
        return self.layout_box((line.box[0] - margin, top, line.box[2] + margin, bottom))

    def baseline(self, line: Line) -> int:
        """A line's baseline as a layout y, 0-1000 of the crop height. A crop holds only lines whose baseline lies
        inside it, so unlike the line's slot it is never clamped."""
        return round(1000 * (line.baseline_y - self.rect[1]) / (self.rect[3] - self.rect[1]))

    def visible_box(self, line: Line) -> list[int]:
        """The layout box of the part of a cut-off line inside the crop."""
        return self.layout_box(
            (line.box[0], max(line.box[1], self.rect[1]), line.box[2], min(line.box[3], self.rect[3]))
        )

    def rule_box(self, rule: Box) -> list[int]:
        """A ruled line's layout box, at least ``RULE_THICKNESS`` units across so a hairline stays a box."""
        y0, x0, y1, x1 = self.layout_box(rule)
        y0, x0 = min(y0, 1000 - RULE_THICKNESS), min(x0, 1000 - RULE_THICKNESS)
        return [y0, x0, max(y1, y0 + RULE_THICKNESS), max(x1, x0 + RULE_THICKNESS)]

    def place(self, x: float, y: float) -> str:
        """A page point's position on the crop in words: "in the upper right corner", "at the top center"."""
        x0, y0, x1, y1 = self.rect
        return _place((x - x0) / (x1 - x0), (y - y0) / (y1 - y0))

    def cut_edge(self, line: Line) -> str:
        """The edge (``top`` or ``bottom``) that cuts a line off the crop."""
        return "top" if line.baseline_y < self.rect[1] else "bottom"


def line_spacing(regions: list[Region]) -> float | None:
    """Median vertical distance between consecutive baselines within regions (the page's line pitch); a table's are
    its rows' (see ``_row_baselines``)."""
    gaps = [b - a for region in regions for a, b in pairwise(_pitch_baselines(region)) if b > a]
    return statistics.median(gaps) if gaps else None


def _pitch_baselines(region: Region) -> list[float]:
    if region.is_table:
        return _row_baselines(region.lines)
    return [line.baseline_y for line in region.lines]


def _row_baselines(lines: list[Line]) -> list[float]:
    """A table's baselines, top to bottom, the lines of one row (cells side by side, baselines within ``SAME_ROW`` of
    the median line height) counted once, at their median."""
    tolerance = SAME_ROW * statistics.median(line.box[3] - line.box[1] for line in lines)
    rows: list[list[float]] = []
    for baseline in sorted(line.baseline_y for line in lines):
        if rows and baseline - rows[-1][-1] <= tolerance:
            rows[-1].append(baseline)
        else:
            rows.append([baseline])
    return [statistics.median(row) for row in rows]


def _x_overlap(box: Box, rect: Box) -> float:
    """Fraction of ``box``'s width inside ``rect``'s x-range."""
    inside = max(0.0, min(box[2], rect[2]) - max(box[0], rect[0]))
    return inside / max(box[2] - box[0], 1e-6)


def text_columns(regions: list[Region], pitch: float, settings: CropSettings) -> list[list[Line]]:
    """The lines of each wide text region together with the narrow regions that belong beside it.

    A region too narrow to be cropped on its own (a margin note, heading, page number) joins the nearest
    wide region it overlaps vertically or sits within two line pitches of, if the merged column, scaled to
    fit the widest bucket, keeps a line pitch of at least ``min_merge_spacing``; it is then cropped with that
    region's lines, so the crops show page structure and not just body text. Lines come back sorted by
    baseline.
    """
    natural_scale = min(1.0, settings.line_spacing / pitch)
    is_wide = [(r.box[2] - r.box[0] + pitch) * natural_scale >= settings.min_width for r in regions]
    columns = [Column.of(region) for region, wide in zip(regions, is_wide, strict=True) if wide]
    for region, wide in zip(regions, is_wide, strict=True):
        if wide:
            continue
        box = region.box
        homes = [c for c in columns if c.gap_to(box) <= 2 * pitch and _merge_fits(c, box, pitch, settings=settings)]
        if homes:
            min(homes, key=lambda c: c.gap_to(box)).add(region)
    return [sorted(c.lines, key=lambda line: line.baseline_y) for c in columns]


def _merge_fits(column: Column, box: Box, pitch: float, *, settings: CropSettings) -> bool:
    """Whether ``column`` plus ``box``, scaled to fit the widest bucket, keeps a legible line pitch."""
    span = column.width_with(box) + pitch
    scale = min(1.0, settings.line_spacing / pitch, WIDEST_BUCKET / span)
    return pitch * scale >= settings.min_merge_spacing


def crops_for_page(
    regions: list[Region], size: tuple[int, int], settings: CropSettings, *, rules: Sequence[Box] = ()
) -> list[Crop]:
    """Crops over consecutive whole lines of each text column (see ``text_columns``).

    A column's crops all take the size of one NeMo bucket (the one nearest the column's width at the target
    line pitch whose height still fits on the page). Windows of lines overlap by ``overlap_lines``. A crop
    holds every line whose baseline lies inside it and whose width is (almost) whole; a crop that cuts a line
    at the side (unlabelled visible text) is dropped. Lines cut by the top or bottom edge are its
    ``fragments``. Each of the page's ruled ``rules`` (page px) at least ``SHORTEST_RULE`` line pitches long
    inside a crop is kept in it, cut to the crop.

    A page where this finds no crop at all (a form of printed labels and handwritten fields, a page of one-line
    regions: nothing anchors a column or gives a pitch inside a region) is cropped as one block of all its lines,
    its pitch measured from row to row across the page; every other page is cropped as above.
    """
    lines = [line for region in regions for line in region.lines]
    pitch = line_spacing(regions)
    if pitch is not None:
        page = PageGeometry(size=size, pitch=pitch, lines=lines, rules=list(rules))
        crops = [
            crop for column in text_columns(regions, pitch, settings) for crop in _column_crops(column, page, settings)
        ]
        if crops:
            return crops
    return _block_crops(lines, size, settings, rules=rules)


def _block_crops(
    lines: list[Line], size: tuple[int, int], settings: CropSettings, *, rules: Sequence[Box]
) -> list[Crop]:
    """Crops of all of a page's lines as one block, at the pitch between its rows."""
    rows = _row_baselines(lines) if lines else []
    gaps = [b - a for a, b in pairwise(rows)]
    if not gaps:
        return []
    page = PageGeometry(size=size, pitch=statistics.median(gaps), lines=lines, rules=list(rules))
    return _column_crops(sorted(lines, key=lambda line: line.baseline_y), page, settings)


def _column_crops(lines: list[Line], page: PageGeometry, settings: CropSettings) -> list[Crop]:
    frame = _column_frame(lines, page, settings)
    if frame is None:
        return []
    per_window = _lines_per_window(lines, frame.span, page.pitch)
    if per_window < settings.min_lines:
        return []
    starts = _window_starts(len(lines), per_window, settings.overlap_lines)
    crops = (_crop(_window_rect(lines[i : i + per_window], frame, page), frame.bucket, page) for i in starts)
    return [crop for crop in crops if crop is not None and len(crop.lines) >= settings.min_lines]


def _column_frame(lines: list[Line], page: PageGeometry, settings: CropSettings) -> ColumnFrame | None:
    """The column's padded x-range and the bucket nearest its width at the target pitch; None if unfit."""
    pad = 0.5 * page.pitch
    left = max(0.0, min(line.box[0] for line in lines) - pad)
    right = min(float(page.size[0]), max(line.box[2] for line in lines) + pad)
    natural_width = (right - left) * min(1.0, settings.line_spacing / page.pitch)
    fitting = [b for b in BUCKETS if b[1] * (right - left) / b[0] <= page.size[1]]
    if natural_width < settings.min_width or not fitting:
        return None

    def width_error(bucket: tuple[int, int]) -> float:
        return abs(bucket[0] - natural_width)

    frame = ColumnFrame(left=left, right=right, bucket=min(fitting, key=width_error))
    if page.pitch * frame.scale < settings.min_line_spacing:
        return None
    return frame


def _lines_per_window(lines: list[Line], span: float, pitch: float) -> int:
    count = len(lines)
    while count > 1 and lines[count - 1].baseline_y - lines[0].baseline_y + 2 * pitch > span:
        count -= 1
    return count


def _window_starts(total: int, size: int, overlap: int) -> list[int]:
    """First-line indices of windows of ``size`` lines sharing ``overlap``; the last ends at the column's end."""
    starts = list(range(0, max(total - size, 0) + 1, max(1, size - overlap)))
    if starts[-1] != total - size:
        starts.append(total - size)
    return starts


def _window_rect(window: list[Line], frame: ColumnFrame, page: PageGeometry) -> Box:
    middle = (window[0].baseline_y + window[-1].baseline_y) / 2 - 0.25 * page.pitch
    top = min(max(0.0, middle - frame.span / 2), page.size[1] - frame.span)
    return (frame.left, top, frame.right, top + frame.span)


def _crop(rect: Box, bucket: tuple[int, int], page: PageGeometry) -> Crop | None:
    """The crop at ``rect``; None when it cuts a line at the side."""
    inside = [line for line in page.lines if rect[1] <= line.baseline_y <= rect[3]]
    if any(CUT_LINE < _x_overlap(line.box, rect) < WHOLE_LINE for line in inside):
        return None
    return Crop(
        rect=rect,
        size=bucket,
        pitch=page.pitch,
        lines=[line for line in inside if _x_overlap(line.box, rect) >= WHOLE_LINE],
        fragments=[line for line in page.lines if _is_cut_off(line, rect, page.pitch)],
        rules=[part for rule in page.rules if (part := _rule_inside(rule, rect, page.pitch)) is not None],
    )


def _rule_inside(rule: Box, rect: Box, pitch: float) -> Box | None:
    """The part of a ruled line inside ``rect``; None when less than ``SHORTEST_RULE`` line pitches of it is."""
    part = (max(rule[0], rect[0]), max(rule[1], rect[1]), min(rule[2], rect[2]), min(rule[3], rect[3]))
    if part[2] < part[0] or part[3] < part[1]:
        return None
    return part if max(part[2] - part[0], part[3] - part[1]) >= SHORTEST_RULE * pitch else None


def _is_cut_off(line: Line, rect: Box, pitch: float) -> bool:
    """Whether the crop's top or bottom edge cuts the line: baseline outside, a visible part of its ink inside."""
    if rect[1] <= line.baseline_y <= rect[3] or _x_overlap(line.box, rect) < 0.5:
        return False
    return min(line.box[3], rect[3]) - max(line.box[1], rect[1]) > 0.25 * pitch


def _place(across: float, down: float) -> str:
    """Words for a position given as fractions of the crop's width and height."""
    side = _third(across, ("left", "center", "right"))
    level = _third(down, ("upper", "middle", "lower"))
    if side == "center":
        return {"upper": "at the top center", "middle": "in the middle", "lower": "at the bottom center"}[level]
    if level == "middle":
        return f"on the {side} side"
    return f"in the {level} {side} corner"


def _third(fraction: float, names: tuple[str, str, str]) -> str:
    if fraction < 1 / 3:
        return names[0]
    if fraction > 2 / 3:
        return names[2]
    return names[1]
