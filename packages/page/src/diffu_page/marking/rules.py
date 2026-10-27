"""A page's ruled lines found on its scan: the long straight lines of a printed form, a table's grid or a ledger's
columns, horizontal and vertical, told apart from handwriting, the page's edges, the fold of an opening and the scan's
border.

The scan is read at about 40 px a text line, min-pooled so that a hairline stays dark. Its thin dark strokes are picked
out (a black-hat: what a closing wider than any stroke fills in) and its straight runs of ink at least two text lines
long kept (an opening along each direction in turn). Runs are joined into strokes across small breaks, and a stroke is
no part of a rule when it is thick (a vertical one: a fold's shadow), wavy, has no paper beside it (a page's edge, the
scan's border) or lies on a text line (a dash in it, a filler running on from it). The strokes left are joined across
larger breaks (a rule crossed by writing, or faded), and a rule is at least six text lines long and runs near the text;
a vertical one neither reaches up over the head margin (an edge of the book block) nor is the fold of an opening: the
line by its gutter, between its pages' text, that runs along half the text's height. Lengths are in text lines (the
median line box's height), so scans of any resolution are read alike.
"""

from __future__ import annotations

import statistics
from enum import StrEnum

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from diffu_page.layout import Layout, PageBox, Rule, gutter


class Direction(StrEnum):
    """Which way a ruled line runs."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


# Every length here is in text lines (the median text line box's height) but WORK_PITCH's
WORK_PITCH = 40.0  # px: the text line height a scan is read at
INK = 0.22  # how much darker than its paper a pixel of ink is, at least
RUN = 2.0  # the shortest straight run of ink kept
SMALL_GAP = 0.5  # the widest break inside a stroke
GAP = 2.0  # the widest break inside a rule
LEAN = 0.15  # how far a stroke may lie off the course of the one it continues
MIN_LENGTH = 6.0  # the shortest rule
MIN_COVER = 0.7  # the least share of a rule's length its ink covers
MAX_THICK = {Direction.HORIZONTAL: 0.35, Direction.VERTICAL: 0.15}  # a thicker vertical line is a fold's shadow
MAX_WAVE = 0.05  # the most a rule's ink strays from its course (rms)
MAX_BEND = 1.0  # the most a rule bows from its chord: a page curving into its fold bends its rules
BESIDE = (0.2, 0.3, 0.4, 0.5, 0.6)  # where the paper beside a stroke is looked for
PAPER_BESIDE = 0.8  # the paper beside a rule is at least this bright, of the page's paper
TOUCH = 0.25  # a horizontal stroke starting this close to a text line's end, at its height, runs on from it
NEAR = 1.5  # a rule runs at most this far from a text line
HEAD = 1.0  # a vertical line reaching further above the text and the horizontal rules is an edge or a fold
FOLD = 0.5  # a vertical line this close to an opening's fold is the fold
FOLD_SPAN = 0.5  # the least share of the text's height a fold runs along, whole or in pieces


class Sheet(BaseModel):
    """A scan as ruled lines are looked for on it, turned so that the lines sought run along its rows: grey, min-pooled
    to about :data:`WORK_PITCH` px a text line; its text line boxes in its pixels (along, across, along, across), its
    text line height and the grey of its paper."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    image: np.ndarray
    boxes: list[PageBox]
    pitch: float
    paper: float
    direction: Direction

    def turned(self) -> Sheet:
        """The sheet turned a quarter (transposed), its vertical lines along its rows."""
        return Sheet(
            image=np.ascontiguousarray(self.image.T),
            boxes=[(top, left, bottom, right) for left, top, right, bottom in self.boxes],
            pitch=self.pitch,
            paper=self.paper,
            direction=Direction.VERTICAL,
        )


class Stroke(BaseModel):
    """A straight run of ink along a sheet's rows: for each column it covers, the middle of its ink across and how much
    ink there is; and its course, a quadratic fitted to the middles, of the column from the stroke's centre column."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    columns: np.ndarray
    middles: np.ndarray
    widths: np.ndarray
    centre: float
    course: np.ndarray

    @classmethod
    def of(cls, columns: np.ndarray, middles: np.ndarray, widths: np.ndarray) -> Stroke:
        """A stroke of these columns, its course fitted (from its centre column: a fit over far columns is
        ill-conditioned)."""
        centre = float(columns.mean())
        course = np.polyfit(columns - centre, middles, min(2, len(columns) - 1))
        return cls(columns=columns, middles=middles, widths=widths, centre=centre, course=course)

    @classmethod
    def joined(cls, strokes: list[Stroke]) -> Stroke:
        """One stroke of all of ``strokes``' columns."""
        return cls.of(
            *(
                np.concatenate([getattr(stroke, name) for stroke in strokes])
                for name in ("columns", "middles", "widths")
            )
        )

    @property
    def start(self) -> float:
        return float(self.columns.min())

    @property
    def end(self) -> float:
        return float(self.columns.max())

    @property
    def length(self) -> float:
        return self.end - self.start

    def at(self, columns: np.ndarray) -> np.ndarray:
        """Where the stroke's course runs across at ``columns``."""
        return np.polyval(self.course, columns - self.centre)

    def reach(self) -> tuple[float, float]:
        """The lowest and the highest its course runs across."""
        across = self.at(np.linspace(self.start, self.end, 64))
        return float(across.min()), float(across.max())

    @property
    def across(self) -> float:
        """Where the stroke runs across, midway between its reach."""
        return sum(self.reach()) / 2

    @property
    def thickness(self) -> float:
        return float(self.widths.mean())

    @property
    def wave(self) -> float:
        """How far its ink strays from its course (rms)."""
        return float(np.sqrt(np.mean((self.middles - self.at(self.columns)) ** 2)))

    @property
    def bend(self) -> float:
        """How far its course bows from the chord between its ends."""
        chord = np.interp(self.columns, [self.start, self.end], self.at(np.array([self.start, self.end])))
        return float(np.abs(self.at(self.columns) - chord).max())

    @property
    def cover(self) -> float:
        """The share of its length its ink covers."""
        return len(self.columns) / max(1.0, self.length)


def find_rules(scan: np.ndarray, layout: Layout) -> list[Rule]:
    """The ruled lines of a page (in its page px), found on its grey ``scan`` (of any size: the layout is scaled to it):
    the horizontal ones top to bottom, then the vertical ones left to right. A page without text lines has none: what
    tells a rule from writing is the text."""
    if not layout.lines:
        return []
    sheet = _sheet(scan, layout)
    horizontal = _rules(_joined(_runs(sheet), sheet, gap=SMALL_GAP), sheet)
    upright = sheet.turned()
    strokes = _joined(_runs(upright), upright, gap=SMALL_GAP)
    head = min([min(box[0] for box in upright.boxes), *(rule.reach()[0] for rule in horizontal)])
    fold = _fold(layout, upright, strokes, crossing=horizontal)
    vertical = [
        rule
        for rule in _rules(strokes, upright)
        if rule.start >= head - HEAD * sheet.pitch and (fold is None or abs(rule.across - fold) > FOLD * sheet.pitch)
    ]
    to_page = (layout.width / sheet.image.shape[1], layout.height / sheet.image.shape[0])
    return [_page_rule(rule, Direction.HORIZONTAL, to_page) for rule in horizontal] + [
        _page_rule(rule, Direction.VERTICAL, to_page) for rule in vertical
    ]


def _sheet(scan: np.ndarray, layout: Layout) -> Sheet:
    """The scan min-pooled to about :data:`WORK_PITCH` px a text line (a hairline stays as dark as it was), with its
    text line boxes, text line height and paper (the 90th percentile of the grey around its text)."""
    factor = max(1, int(_line_height(layout, scan) / WORK_PITCH))
    darkest = cv2.erode(scan, np.ones((factor, factor), np.uint8)) if factor > 1 else scan
    image = cv2.resize(darkest, (scan.shape[1] // factor, scan.shape[0] // factor), interpolation=cv2.INTER_AREA)
    sx, sy = image.shape[1] / layout.width, image.shape[0] / layout.height
    boxes = [(line.box[0] * sx, line.box[1] * sy, line.box[2] * sx, line.box[3] * sy) for line in layout.lines]
    left, top = int(min(box[0] for box in boxes)), int(min(box[1] for box in boxes))
    right, bottom = int(max(box[2] for box in boxes)) + 1, int(max(box[3] for box in boxes)) + 1
    return Sheet(
        image=image,
        boxes=boxes,
        pitch=_line_height(layout, image),
        paper=float(np.percentile(image[top:bottom, left:right], 90)),
        direction=Direction.HORIZONTAL,
    )


def _line_height(layout: Layout, image: np.ndarray) -> float:
    """The page's text line height in ``image``'s pixels: its median line box's."""
    return statistics.median(line.box[3] - line.box[1] for line in layout.lines) * image.shape[0] / layout.height


def _runs(sheet: Sheet) -> list[Stroke]:
    """The sheet's straight runs of ink along its rows, at least :data:`RUN` text lines long: its thin dark strokes (a
    black-hat), thickened across so that a slightly slanted line keeps long runs, opened along."""
    image = sheet.image.astype(np.float32)
    size = max(3, round(sheet.pitch / 3) | 1)
    paper = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)))
    ink = ((paper - image) / np.maximum(paper, 1.0) > INK).astype(np.uint8)
    run = max(9, round(RUN * sheet.pitch))
    thickened = cv2.dilate(ink, np.ones((3, 1), np.uint8))
    opened = cv2.morphologyEx(thickened, cv2.MORPH_OPEN, np.ones((1, run), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    runs: list[Stroke] = []
    for k in range(1, count):
        x, y, w, h = (int(v) for v in stats[k, :4])
        own = (labels[y : y + h, x : x + w] == k) & (ink[y : y + h, x : x + w] > 0)
        columns = np.flatnonzero(own.any(axis=0))
        if len(columns) < run // 2:
            continue
        weight = own[:, columns].astype(np.float32)
        widths = weight.sum(axis=0)
        middles = (weight * np.arange(h, dtype=np.float32)[:, None]).sum(axis=0) / widths + y
        runs.append(Stroke.of(columns.astype(np.float32) + x, middles, widths))
    return runs


def _joined(strokes: list[Stroke], sheet: Sheet, *, gap: float) -> list[Stroke]:
    """The strokes joined where one starts at most ``gap`` text lines after another ends, on its course."""
    groups: list[list[Stroke]] = []
    for stroke in sorted(strokes, key=lambda s: s.start):
        group = next((group for group in groups if _continues(group[-1], stroke, sheet, gap=gap)), None)
        if group is None:
            groups.append([stroke])
        else:
            group.append(stroke)
    return [group[0] if len(group) == 1 else Stroke.joined(group) for group in groups]


def _continues(last: Stroke, stroke: Stroke, sheet: Sheet, *, gap: float) -> bool:
    """Whether ``stroke`` continues ``last``: it starts at most ``gap`` text lines after ``last`` ends, within
    :data:`LEAN` text lines of where the straight line through ``last``'s final stretch leads."""
    if not 0 <= stroke.start - last.end <= gap * sheet.pitch:
        return False
    final, first = slice(-64, None), slice(0, 64)
    heading = np.polyfit(last.columns[final] - last.end, last.middles[final], min(1, len(last.columns[final]) - 1))
    led_to = float(np.polyval(heading, stroke.columns[first].mean() - last.end))
    off = abs(led_to - float(stroke.middles[first].mean()))
    return off <= max(2.0, LEAN * sheet.pitch)


def _rules(strokes: list[Stroke], sheet: Sheet) -> list[Stroke]:
    """The ruled lines among a sheet's strokes, in the order they lie across: those that can be parts of rules, joined
    across :data:`GAP`, at least :data:`MIN_LENGTH` long, covered by their ink, straight and near the text."""
    parts = [stroke for stroke in strokes if _part_of_rule(stroke, sheet)]
    rules = [
        rule
        for rule in _joined(parts, sheet, gap=GAP)
        if rule.length >= MIN_LENGTH * sheet.pitch
        and rule.cover >= MIN_COVER
        and _straight(rule, sheet)
        and _near_text(rule, sheet)
    ]
    return sorted(rules, key=lambda rule: (rule.across, rule.start))


def _part_of_rule(stroke: Stroke, sheet: Sheet) -> bool:
    """Whether a stroke can be part of a rule: thin, straight, with paper beside it and not on a text line."""
    return (
        stroke.thickness <= MAX_THICK[sheet.direction] * sheet.pitch
        and _straight(stroke, sheet)
        and min(_paper_beside(stroke, sheet)) >= PAPER_BESIDE * sheet.paper
        and not _on_text_line(stroke, sheet)
    )


def _straight(stroke: Stroke, sheet: Sheet) -> bool:
    return stroke.wave <= MAX_WAVE * sheet.pitch and stroke.bend <= MAX_BEND * sheet.pitch


def _paper_beside(stroke: Stroke, sheet: Sheet) -> tuple[float, float]:
    """The grey on either side of a stroke: along it, the brightest a little way off it (:data:`BESIDE`, so that a
    neighbouring rule is looked past), its median."""
    image = sheet.image
    columns = np.arange(int(stroke.start), int(stroke.end) + 1)
    course = stroke.at(columns)
    sides: list[float] = []
    for sign in (-1, 1):
        greys = [image[_rows(course + sign * offset * sheet.pitch, image), columns] for offset in BESIDE]
        sides.append(float(np.median(np.max(greys, axis=0))))
    return sides[0], sides[1]


def _rows(across: np.ndarray, image: np.ndarray) -> np.ndarray:
    return np.clip(np.round(across).astype(int), 0, image.shape[0] - 1)


def _on_text_line(stroke: Stroke, sheet: Sheet) -> bool:
    """Whether a stroke lies on a text line: across, inside the middle half of the line's box, and along, three
    quarters of it within the box or (a horizontal stroke) starting where the line ends or ending where it starts: a
    filler running on from it."""
    across = stroke.across
    touch = TOUCH * sheet.pitch
    for start, low, end, high in sheet.boxes:
        quarter = (high - low) / 4
        if not low + quarter <= across <= high - quarter:
            continue
        if min(end, stroke.end) - max(start, stroke.start) >= 0.75 * stroke.length:
            return True
        runs_on = abs(stroke.start - end) <= touch or abs(stroke.end - start) <= touch
        if runs_on and sheet.direction is Direction.HORIZONTAL:
            return True
    return False


def _near_text(rule: Stroke, sheet: Sheet) -> bool:
    """Whether a text line lies within :data:`NEAR` text lines of the rule, beside some stretch of it."""
    low, high = rule.reach()
    near = NEAR * sheet.pitch
    return any(
        start < rule.end and end > rule.start and box_low - near < high and box_high + near > low
        for start, box_low, end, box_high in sheet.boxes
    )


def _fold(layout: Layout, upright: Sheet, strokes: list[Stroke], *, crossing: list[Stroke]) -> float | None:
    """Where across the upright sheet an opening's fold runs: of the places by its gutter (:func:`gutter`, else its
    middle) between its pages' text and a text line either side, the nearest the gutter where vertical strokes, whole
    or in pieces, run along :data:`FOLD_SPAN` of the text's height and no horizontal rule (``crossing``) runs across.
    None for a page no wider than tall, or one where no stroke so runs: a line between two columns that a rule crosses
    is the form's."""
    if layout.width <= layout.height:
        return None
    width = upright.image.shape[0]
    split = gutter(layout)
    middle = width / 2 if split is None else split * width / layout.width
    pitch = upright.pitch
    left = max((box[3] for box in upright.boxes if (box[1] + box[3]) / 2 < middle), default=middle)
    right = min((box[1] for box in upright.boxes if (box[1] + box[3]) / 2 >= middle), default=middle)
    low, high = min(left, middle) - pitch, max(right, middle) + pitch  # transcribers' boxes may reach over the fold
    top, bottom = min(box[0] for box in upright.boxes), max(box[2] for box in upright.boxes)
    by_gutter = [stroke for stroke in strokes if low <= stroke.across <= high]
    folds = [
        stroke.across
        for stroke in by_gutter
        if _span([s for s in by_gutter if abs(s.across - stroke.across) <= FOLD * pitch], top, bottom)
        >= FOLD_SPAN * (bottom - top)
        and not any(rule.start < stroke.across - pitch and rule.end > stroke.across + pitch for rule in crossing)
    ]
    return min(folds, key=lambda across: abs(across - middle)) if folds else None


def _span(strokes: list[Stroke], top: float, bottom: float) -> float:
    """How much of ``top`` to ``bottom`` the strokes run along, together."""
    covered, reached = 0.0, top
    for start, end in sorted((max(stroke.start, top), min(stroke.end, bottom)) for stroke in strokes):
        if end > reached:
            covered += end - max(start, reached)
            reached = end
    return covered


def _page_rule(rule: Stroke, direction: Direction, to_page: tuple[float, float]) -> Rule:
    """A rule found on a sheet as a :class:`Rule` in page px."""
    low, high = rule.reach()
    if direction is Direction.HORIZONTAL:
        return Rule(x0=rule.start * to_page[0], y0=low * to_page[1], x1=rule.end * to_page[0], y1=high * to_page[1])
    return Rule(x0=low * to_page[0], y0=rule.start * to_page[1], x1=high * to_page[0], y1=rule.end * to_page[1])
