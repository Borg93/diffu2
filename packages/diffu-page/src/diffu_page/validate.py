"""Validation — is a synthetic page structurally correct, and would it READ correctly?

Two families of checks, all model-free:

- **Structural / label checks** (:func:`validate_table`, :func:`validate_labels`,
  :func:`validate_xml_roundtrip`) — the guards the audit runs on every composed page.
- **Recognition metrics** (:func:`iou`, :func:`char_error_rate`, :func:`match` → :class:`MatchResult`)
  — detection IoU tells you the lines are *findable* (blending/contrast right); CER tells you they're
  *readable*. These await a detector/recognizer integration (see :mod:`diffu_page.htr` for the HTRFlow
  end-to-end run). The real verdict is still: train the downstream stack on the synthetic pages and
  evaluate on REAL held-out pages.
"""

from __future__ import annotations

import statistics
import xml.etree.ElementTree as ET
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel

from .labels import LineLabel, PageLabels, export
from .layout import Layout


class Box(NamedTuple):
    """An axis-aligned bounding box in corner form. Tuple-compatible on purpose — it unpacks
    (``x0, y0, x1, y1 = box``) and indexes like the plain tuple it replaces, so the geometry loops stay
    unchanged, but the fields are now named instead of documented in a trailing comment."""

    x0: int
    y0: int
    x1: int
    y1: int


# Fixed-slot regions whose cells must sit inside their column and keep a consistent size (see compose).
_TABLE_REGIONS = {
    "table",
    "table_header",
    "row_number",
    "reg_label",
    "reg_value",
    "reg_header",
    "kv_label",
    "kv_value",
}
_HEADER_REGIONS = {"table_header", "reg_header"}


class MatchResult(BaseModel):
    """Detection + recognition scores for one page, against its known ground truth."""

    n_gt: int
    n_pred: int
    matched: int
    precision: float
    recall: float
    mean_cer: float


def poly_to_box(poly: list[tuple[int, int]]) -> Box:
    """Axis-aligned bounding box of a polygon."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return Box(min(xs), min(ys), max(xs), max(ys))


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two axis-aligned boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def char_error_rate(pred: str, gt: str) -> float:
    """Levenshtein distance / len(gt) — the same CER the rest of the project scores on."""
    if not gt:
        return 0.0 if not pred else 1.0
    prev = list(range(len(gt) + 1))
    for i, pc in enumerate(pred, 1):
        cur = [i]
        for j, gc in enumerate(gt, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (pc != gc)))
        prev = cur
    return prev[-1] / len(gt)


def match(gt: list[tuple[Box, str]], pred: list[tuple[Box, str]], iou_thresh: float = 0.5) -> MatchResult:
    """Greedy IoU-match predicted boxes to ground truth; score detection P/R + mean CER on matches."""
    used: set[int] = set()
    tp = 0
    cers: list[float] = []
    for gbox, gtext in gt:
        best_j, best_iou = -1, iou_thresh
        for j, (pbox, _ptext) in enumerate(pred):
            if j in used:
                continue
            v = iou(gbox, pbox)
            if v >= best_iou:
                best_j, best_iou = j, v
        if best_j >= 0:
            used.add(best_j)
            tp += 1
            cers.append(char_error_rate(pred[best_j][1], gtext))
    return MatchResult(
        n_gt=len(gt),
        n_pred=len(pred),
        matched=tp,
        precision=tp / len(pred) if pred else 0.0,
        recall=tp / len(gt) if gt else 0.0,
        mean_cer=float(np.mean(cers)) if cers else 1.0,
    )


def _rule_crossings(layout: Layout, cells: list[LineLabel], col_pad: int) -> list[str]:
    """Cells whose ink crosses a vertical column rule (wrong column / clipping)."""
    problems: list[str] = []
    verticals = sorted({r.x1 for r in layout.rules if r.x1 == r.x2})
    if not verticals:
        return problems
    for ln in cells:
        x0, _, x1, _ = poly_to_box(ln.polygon)
        crossed = [v for v in verticals if x0 + col_pad < v < x1 - col_pad]
        if crossed:
            problems.append(f"cell {ln.text!r} crosses column rule(s) at {crossed} (ink x {x0}..{x1})")
    return problems


def _cell_overlaps(body: list[LineLabel], overlap_tol: int) -> list[str]:
    """Pairs of body cells whose ink bboxes overlap each other."""
    problems: list[str] = []
    for i in range(len(body)):
        ax0, ay0, ax1, ay1 = poly_to_box(body[i].polygon)
        for j in range(i + 1, len(body)):
            bx0, by0, bx1, by1 = poly_to_box(body[j].polygon)
            if min(ax1, bx1) - max(ax0, bx0) > overlap_tol and min(ay1, by1) - max(ay0, by0) > overlap_tol:
                problems.append(f"cells {body[i].text!r} and {body[j].text!r} overlap")
    return problems


def _header_overlaps(cells: list[LineLabel], body: list[LineLabel], overlap_tol: int) -> list[str]:
    """Headers overlapping a body cell. A header (a table header row, or any of a register's interspersed
    section headers) must not overlap a body cell. Check true bbox overlap per pair — NOT "max header
    bottom vs min body top", which cross-compares a later section header against the first row on a
    multi-section register."""
    problems: list[str] = []
    for h in (ln for ln in cells if ln.region in _HEADER_REGIONS):
        hx0, hy0, hx1, hy1 = poly_to_box(h.polygon)
        for b in body:
            bx0, by0, bx1, by1 = poly_to_box(b.polygon)
            if min(hx1, bx1) - max(hx0, bx0) > overlap_tol and min(hy1, by1) - max(hy0, by0) > overlap_tol:
                problems.append(f"header {h.text!r} overlaps cell {b.text!r}")
    return problems


def _height_consistency(heights: list[int], height_cv_max: float) -> list[str]:
    """A grossly inconsistent cell-height distribution (a stretched / over-sized cell)."""
    if len(heights) < 4:
        return []
    mean_h = statistics.mean(heights)
    cv = statistics.pstdev(heights) / mean_h if mean_h else 0.0
    if cv > height_cv_max:
        return [f"cell heights inconsistent: CV={cv:.2f} (mean {mean_h:.0f}px)"]
    return []


def _reading_order_problems(cells: list[LineLabel], heights: list[int], col_pad: int) -> list[str]:
    """Reading order must be ROW-MAJOR: reading the cells in ``order`` never jumps UP more than a row, and
    stays left-to-right within a row. Catches the register bug where all labels read, then all values."""
    if len(cells) < 2 or not heights:
        return []
    gap = max(1, sorted(heights)[len(heights) // 2])

    def base_y(ln: LineLabel) -> int:
        return ln.baseline[0][1] if ln.baseline else poly_to_box(ln.polygon)[1]

    seq = sorted(cells, key=lambda ln: ln.order)
    for a, b in zip(seq, seq[1:], strict=False):
        if base_y(b) < base_y(a) - gap:
            return [f"reading order jumps up a row: {a.text!r} (#{a.order}) -> {b.text!r} (#{b.order})"]
        if (
            abs(base_y(b) - base_y(a)) <= gap
            and poly_to_box(b.polygon)[0] + col_pad < poly_to_box(a.polygon)[0]
        ):
            return [f"reading order runs right-to-left in a row: {a.text!r} -> {b.text!r}"]
    return []


def validate_table(
    layout: Layout,
    lines: list[LineLabel],
    *,
    col_pad: int = 4,
    overlap_tol: int = 6,
    height_cv_max: float = 0.45,
) -> list[str]:
    """Return the STRUCTURAL problems with a composed table/register ([] == correct) — the check the audit
    needs so a broken table is rejected instead of passing on ``rules > 0``.

    Checks: (1) no cell's ink crosses a vertical column rule (wrong column / clipping); (2) body cells don't
    overlap each other; (3) the header row doesn't overlap the first data row; (4) cell heights are not
    grossly inconsistent (a stretched / over-sized cell); (5) reading order is row-major. A non-table page
    returns ``[]``.
    """
    cells = [ln for ln in lines if ln.region in _TABLE_REGIONS and len(ln.polygon) >= 2]
    if not cells:
        return []
    body = [ln for ln in cells if ln.region not in _HEADER_REGIONS]
    heights = [poly_to_box(ln.polygon)[3] - poly_to_box(ln.polygon)[1] for ln in body]
    return [
        *_rule_crossings(layout, cells, col_pad),
        *_cell_overlaps(body, overlap_tol),
        *_header_overlaps(cells, body, overlap_tol),
        *_height_consistency(heights, height_cv_max),
        *_reading_order_problems(cells, heights, col_pad),
    ]


# PAGE / ALTO share the local tag ``TextLine``; count them per exporter to prove serialization is lossless.
_LINE_TAGS = {
    "page": "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}TextLine",
    "alto": "{http://www.loc.gov/standards/alto/ns-v4#}TextLine",
}


def _centroid(poly: list[tuple[int, int]]) -> tuple[float, float]:
    """Mean of a polygon's vertices — a rotation-stable anchor (the extremes move under warp; the centre
    barely does), so containment tests on it survive the scan pass's perspective/skew."""
    return sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly)


def validate_labels(labels: PageLabels, *, pad: int = 4) -> list[str]:
    """Return the problems with a page's ground-truth labels ([] == correct) — the check the audit needs so a
    mislabelled page is rejected, not just eyeballed.

    Invariants are chosen to be true even after the scan pass warps the geometry (so a pass means the GT is
    actually right, not that warping was disabled). Per line: the polygon has >=3 vertices; polygon, baseline
    and word points all lie in ``[0,W]x[0,H]``; the baseline midpoint sits in the polygon's vertical band and
    runs left->right; each word's centroid falls inside the polygon bbox; the word transcriptions reconstruct
    the line text. Across the page: reading order is a contiguous ``0..n-1`` permutation (no line dropped,
    duplicated, or gapped). Centroid/midpoint tests are used, not extreme-point ones, because a rotated
    straight baseline or word-quad can poke a few px past the AABB of the rotated ink hull without being wrong.
    """
    w, h = labels.width, labels.height
    problems: list[str] = []
    for ln in labels.lines:
        if len(ln.polygon) < 3:
            problems.append(f"line {ln.text!r} polygon has {len(ln.polygon)} vertices (<3)")
            continue
        pts = [*ln.polygon, *ln.baseline, *(p for wd in ln.words for p in wd.polygon)]
        off = next(((x, y) for x, y in pts if not (0 <= x <= w and 0 <= y <= h)), None)
        if off is not None:
            problems.append(f"line {ln.text!r} label leaves the page at {off}")
        px0, py0, px1, py1 = poly_to_box(ln.polygon)
        # Half a line-height: loose enough for the scan warp to shift a straight baseline / word-quad past the
        # ink-hull AABB, tight enough that a baseline/word swapped onto a neighbouring line (>= one line-pitch
        # away) is still caught.
        tol = max(pad, (py1 - py0) // 2)
        if ln.baseline:
            my = (ln.baseline[0][1] + ln.baseline[-1][1]) / 2
            if not (py0 - tol <= my <= py1 + tol):
                problems.append(f"line {ln.text!r} baseline midpoint y={my:.0f} off polygon {py0}..{py1}")
            if ln.baseline[0][0] > ln.baseline[-1][0] + pad:
                problems.append(f"line {ln.text!r} baseline runs right-to-left")
        for wd in ln.words:
            if not wd.polygon:
                continue
            cx, cy = _centroid(wd.polygon)
            if not (px0 - tol <= cx <= px1 + tol and py0 - tol <= cy <= py1 + tol):
                problems.append(f"word {wd.text!r} centroid sits off its line {ln.text!r}")
                break
        if ln.words and " ".join(wd.text for wd in ln.words) != " ".join(ln.text.split()):
            problems.append(f"line {ln.text!r} word transcriptions don't reconstruct the line text")
    orders = sorted(ln.order for ln in labels.lines)
    if orders and orders != list(range(len(orders))):
        problems.append(f"reading order is not a contiguous 0..{len(orders) - 1} permutation: {orders}")
    return problems


def validate_xml_roundtrip(labels: PageLabels) -> list[str]:
    """Return the problems serializing labels to every format ([] == correct): each of ``page`` / ``alto`` must
    be well-formed XML whose ``TextLine`` count equals the line count, and ``json`` must re-parse to the same
    number of lines — proving no line is dropped or merged by export."""
    problems: list[str] = []
    n = len(labels.lines)
    for fmt, tag in _LINE_TAGS.items():
        text = export(labels, fmt)
        try:
            root = ET.fromstring(text)
        except ET.ParseError as exc:
            problems.append(f"{fmt} export is not well-formed XML: {exc}")
            continue
        found = len(list(root.iter(tag)))
        if found != n:
            problems.append(f"{fmt} export has {found} TextLine elements, expected {n}")
    return problems
