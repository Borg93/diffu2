"""Compose stage — seat rendered ink lines on a paper page, then make it cohere into one scan.

The blend recipe (why it looks authentic, no inpainter needed — this is what docTR does): for each line,
render at a **constant font size**, **baseline-align**, align within the slot, optional micro-skew, then
**multiply the paper THROUGH the model's verbatim ink** — its matte maps the tan halo + white to 0, so only
real strokes darken the page and paper fibre shows through (translucent, no per-line shadow). The seated
page is then handed to :func:`diffu_page.augment.scan_pass` to become one coherent scan across the real
pristine -> filthy range. Background paper comes from :func:`diffu_page.paper.paper_background`.

Returns the page (BGR) and one :class:`~diffu_page.labels.LineLabel` of ground truth per seated line.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import cv2
import numpy as np
from pydantic import BaseModel

from .augment import AugmentConfig, scan_pass
from .labels import LineLabel, Point, WordLabel
from .layout import Layout, LineBox, PageSpec, Rule
from .paper import PaperTone, paper_background
from .render import render_boxes

if TYPE_CHECKING:
    from .render import LineRenderer

_INK = 0.25  # alpha coverage above which a pixel counts as ink (matte boundary)

# Fixed-width slots: a table cell / register entry / form field must keep its natural size, NOT be stretched
# to fill its slot the way a flowing prose line is — stretching a short numeric cell blows it up wide and
# breaks column consistency. These regions opt out of the underflow stretch in `_place_and_blend`.
_FIXED_CELL_REGIONS = {
    "table",
    "table_header",
    "row_number",
    "reg_label",
    "reg_value",
    "reg_header",
    "kv_label",
    "kv_value",
    "margin_key",  # a short keyed note stays its natural width, not stretched to fill the gutter
}


class _Placed(BaseModel):
    """Geometry recovered from one seated line's ink: a tight polygon, a baseline, and word boxes."""

    polygon: list[Point]
    baseline: list[Point]
    words: list[WordLabel]


def _rect(x0: int, y0: int, x1: int, y1: int) -> list[Point]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _ink_polygon(ink: np.ndarray, ox: int, oy: int, bins: int = 16) -> list[Point]:
    """A tight polygon hugging the ink: a top profile (left->right) + bottom profile (right->left),
    sampled over ``bins`` column slices. Falls back to the bbox when the ink is too sparse to profile."""
    cols = ink > _INK
    xs = np.where(cols.any(axis=0))[0]
    if len(xs) == 0:
        return []
    xmin, xmax = int(xs.min()), int(xs.max())
    edges = np.linspace(xmin, xmax + 1, bins + 1).astype(int)
    top: list[Point] = []
    bot: list[Point] = []
    for i in range(bins):
        c0, c1 = int(edges[i]), max(int(edges[i]) + 1, int(edges[i + 1]))
        rows = np.where(cols[:, c0:c1].any(axis=1))[0]
        if len(rows) == 0:
            continue
        cx = (c0 + c1) // 2
        top.append((ox + cx, oy + int(rows.min())))
        bot.append((ox + cx, oy + int(rows.max()) + 1))
    if len(top) < 2:
        ys = np.where(cols.any(axis=1))[0]
        return _rect(ox + xmin, oy + int(ys.min()), ox + xmax + 1, oy + int(ys.max()) + 1)
    top[0], top[-1] = (ox + xmin, top[0][1]), (ox + xmax + 1, top[-1][1])  # pin the hull to the true
    bot[0], bot[-1] = (ox + xmin, bot[0][1]), (ox + xmax + 1, bot[-1][1])  # ink width, not the bin centres
    return top + bot[::-1]


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` runs of True in a 1-D boolean array."""
    runs: list[tuple[int, int]] = []
    n = len(flags)
    i = 0
    while i < n:
        if flags[i]:
            j = i + 1
            while j < n and flags[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _word_boxes(ink: np.ndarray, ox: int, oy: int, text: str) -> list[WordLabel]:
    """Per-word boxes from the REAL inter-word gaps in the ink (the widest gaps split words), falling back
    to a proportional split only when glyph runs are too few/joined to separate cleanly."""
    words = text.split()
    cols = np.asarray((ink > _INK).any(axis=0))
    xs = np.where(cols)[0]
    if len(xs) == 0:
        return []
    if len(words) <= 1:
        ys = np.where((ink > _INK).any(axis=1))[0]
        box = _rect(ox + int(xs.min()), oy + int(ys.min()), ox + int(xs.max()) + 1, oy + int(ys.max()) + 1)
        return [WordLabel(text=text, polygon=box)]
    runs = _runs(cols)
    if len(runs) >= len(words):  # merge glyph runs into len(words) groups at the widest gaps
        widest = sorted(range(len(runs) - 1), key=lambda k: runs[k + 1][0] - runs[k][1], reverse=True)
        cuts = sorted(widest[: len(words) - 1])
        groups: list[tuple[int, int]] = []
        start = 0
        for c in cuts:
            groups.append((runs[start][0], runs[c][1]))
            start = c + 1
        groups.append((runs[start][0], runs[-1][1]))
        out: list[WordLabel] = []
        for (c0, c1), word in zip(groups, words, strict=True):
            rows = np.where((ink[:, c0:c1] > _INK).any(axis=1))[0]
            out.append(
                WordLabel(
                    text=word, polygon=_rect(ox + c0, oy + int(rows.min()), ox + c1, oy + int(rows.max()) + 1)
                )
            )
        return out
    return _proportional_words(words, ink, ox, oy)  # cursive/joined: can't separate -> estimate


def _proportional_words(words: list[str], ink: np.ndarray, ox: int, oy: int) -> list[WordLabel]:
    """Split the ink bbox by word length — the fallback when ink gaps don't isolate words."""
    rows, cols = np.where(ink > _INK)
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    y0, y1 = oy + int(rows.min()), oy + int(rows.max()) + 1
    total = sum(len(w) for w in words) + (len(words) - 1)
    out: list[WordLabel] = []
    cx = float(x0)
    for i, word in enumerate(words):
        frac = (len(word) + (1 if i < len(words) - 1 else 0)) / total
        nx = min(x1, cx + frac * (x1 - x0))
        out.append(WordLabel(text=word, polygon=_rect(ox + int(cx), y0, ox + int(nx), y1)))
        cx = nx
    return out


def _skew(ink: np.ndarray, deg: float) -> np.ndarray:
    """Rotate an ink raster by ``deg`` degrees, growing the canvas so nothing is clipped."""
    h, w = ink.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2
    return cv2.warpAffine(ink, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def _fit_ink_to_slot(ink: np.ndarray, box: LineBox, spec: PageSpec) -> np.ndarray:
    """Fit one ink raster to its slot: shrink an overflow, stretch a prose underflow, apply micro-skew.

    If the ink's natural width overflows the slot it is scaled DOWN (aspect-preserving), and if it falls
    SHORT of the slot it is stretched horizontally to fill it (real archival lines run to the margin) —
    capped at ``spec.max_stretch`` so a short/ragged line is never blown up, and inset by a small pad so
    ink never touches the slot edge. Both the shrink and the stretch are bounded by ``box.w`` (which the
    layout already fit inside the page margins / template region), so a placed line can never spill
    outside its slot.
    """
    ih, iw = ink.shape[:2]
    pad = max(2, int(0.05 * box.h))  # keep ink off the exact slot edge (never touches the margin)
    slot_w = max(1, box.w - 2 * pad)
    if iw > slot_w:  # overflow: shrink aspect-preserving to fit the slot
        scale = slot_w / iw
        ink = cv2.resize(
            ink, (max(1, int(iw * scale)), max(1, int(ih * scale))), interpolation=cv2.INTER_AREA
        )
        ih, iw = ink.shape[:2]
    elif box.region not in _FIXED_CELL_REGIONS and iw < slot_w:
        # underflow: a flowing prose line stretches horizontally to fill the slot (capped; short lines stay
        # ragged). Fixed table/register/form cells opt out — they keep their natural size for column consistency.
        target = min(slot_w, int(iw * spec.max_stretch))
        if target > iw:
            ink = cv2.resize(ink, (target, ih), interpolation=cv2.INTER_LINEAR)
    if abs(box.rotation) > 1e-3:
        ink = _skew(ink, box.rotation)
    return ink


def _place_and_blend(
    page: np.ndarray, ink: np.ndarray, box: LineBox, spec: PageSpec, ink_tint: np.ndarray
) -> _Placed | None:
    """Place one ink raster aligned within its slot; return its ink geometry.

    The line is rendered at ``box.h`` for a consistent font size and fit to its slot by
    :func:`_fit_ink_to_slot` (shrink overflow / stretch prose underflow / micro-skew). It is then aligned
    within ``box`` and multiply-blended through ``ink_tint`` so the paper fibre shows through.
    """
    ih, iw = ink.shape[:2]
    if iw == 0 or ih == 0:
        return None
    ink = _fit_ink_to_slot(ink, box, spec)
    ih, iw = ink.shape[:2]

    if box.align == "right":
        left = box.x + box.w - iw
    elif box.align == "center":
        left = box.x + (box.w - iw) // 2
    else:
        left = box.x
    top = int(box.baseline_y - spec.baseline_frac * ih)  # baseline-align, not bbox-top

    H, W = page.shape[:2]
    y0, x0 = max(0, top), max(0, left)
    y1, x1 = min(H, top + ih), min(W, left + iw)
    if y1 <= y0 or x1 <= x0:
        return None

    alpha = ink[..., 3].astype(np.float32) / 255.0
    a = alpha[y0 - top : y1 - top, x0 - left : x1 - left][..., None]
    roi = page[y0:y1, x0:x1].astype(np.float32)
    # Alpha-OVER the ink colour onto the local paper. Opacity is the matte coverage, gamma-lifted so the
    # model's faint/thin strokes still read as real ink; the colour is `ink_tint`, a DARK iron-gall/sepia
    # core, so full coverage lands near real near-black ink. Paper shows through partial coverage. (The old
    # `coverage * darkness**1.7 * (1-tint)` triple-attenuation washed the ink out to near-invisible.)
    cover = a**0.8
    page[y0:y1, x0:x1] = np.clip(roi * (1.0 - cover * (1.0 - ink_tint)), 0, 255).astype(np.uint8)

    return _recover_geometry(a[..., 0], box, spec, x0=x0, y0=y0, top=top, ih=ih)


def _recover_geometry(
    matte: np.ndarray, box: LineBox, spec: PageSpec, *, x0: int, y0: int, top: int, ih: int
) -> _Placed | None:
    """A seated line's ink geometry: tight polygon, baseline clamped to the ink band, and word boxes."""
    cols = np.asarray((matte > _INK).any(axis=0))
    xs = np.where(cols)[0]
    if len(xs) == 0:
        return None
    ink_rows = np.where(np.asarray((matte > _INK).any(axis=1)))[0]  # actual ink band (raster is padded)
    iy0, iy1 = y0 + int(ink_rows.min()), y0 + int(ink_rows.max())
    baseline_y = int(min(max(top + spec.baseline_frac * ih, iy0), iy1))  # seated baseline, clamped to ink
    baseline = [(x0 + int(xs.min()), baseline_y), (x0 + int(xs.max()), baseline_y)]
    return _Placed(
        polygon=_ink_polygon(matte, x0, y0),
        baseline=baseline,
        words=_word_boxes(matte, x0, y0, box.text),
    )


_REGION_READING_PRIORITY = {"page_number": 0, "header": 1, "marginalia": 3}

# Table/register cells interleave spatially across a row (a name and its value share a line), so they must
# read ROW-MAJOR as ONE stream — grouping them per region would read every label, then every value, then
# every header (a garbled register). All row-flow regions collapse to one group ordered (row, x).
_ROW_FLOW_REGIONS = {
    "table",
    "table_header",
    "row_number",
    "reg_label",
    "reg_value",
    "reg_header",
    "kv_label",
    "kv_value",
}
_ROW_FLOW_KEY = "\x00row_flow"


def _poly_top(ln: LineLabel) -> int:
    return min(p[1] for p in ln.polygon)


def _poly_left(ln: LineLabel) -> int:
    return min(p[0] for p in ln.polygon)


def _base_y(ln: LineLabel) -> int:
    return ln.baseline[0][1] if ln.baseline else min(p[1] for p in ln.polygon)


def _cluster_rows(group: list[LineLabel]) -> list[LineLabel]:
    """Order table/register cells ROW-MAJOR: cluster them into rows by BASELINE gaps (a new row starts when
    the baseline jumps more than a cell height — larger than the within-row jitter, smaller than the row
    pitch), then read each row left-to-right. So a label precedes its value and rows read top-to-bottom."""
    if not group:
        return group
    ordered = sorted(group, key=_base_y)
    heights = sorted(max(p[1] for p in ln.polygon) - min(p[1] for p in ln.polygon) for ln in group)
    gap = max(1, heights[len(heights) // 2])  # a full cell height: > within-row jitter, < the row pitch
    out: list[LineLabel] = []
    row: list[LineLabel] = []
    last_y: int | None = None
    for ln in ordered:
        y = _base_y(ln)
        if last_y is not None and y - last_y > gap:
            out.extend(sorted(row, key=_poly_left))
            row = []
        row.append(ln)
        last_y = y
    out.extend(sorted(row, key=_poly_left))
    return out


def _assign_reading_order(lines: list[LineLabel]) -> list[LineLabel]:
    """Reassign ``order`` as a true page reading sequence: page-number first, then body/table read row-major,
    marginalia last. Table/register cells read as one row-major stream so label/value interleave per row."""

    def region_priority(region: str) -> int:
        return _REGION_READING_PRIORITY.get(region, 2)  # body / columns / table cells read in the middle

    groups: dict[str, list[LineLabel]] = {}
    for ln in lines:
        if ln.region in _ROW_FLOW_REGIONS:
            key = _ROW_FLOW_KEY
        elif ln.region in ("rubric", "margin_key", "initial"):
            # a mid-text rubric heading, a keyed marginal note, and an enlarged opening initial all read in
            # FLOW position with the body (sorted by baseline then x, so a left-gutter key / a line-start
            # initial precedes its body line), not appended last like an unkeyed running marginalia note
            key = "body"
        else:
            key = ln.region
        groups.setdefault(key, []).append(ln)

    def region_key(item: tuple[str, list[LineLabel]]) -> tuple[int, int, int]:
        region, group = item
        prio = 2 if region == _ROW_FLOW_KEY else region_priority(region)
        return prio, min(_poly_left(ln) for ln in group), min(_poly_top(ln) for ln in group)

    ordered: list[LineLabel] = []
    for region, group in sorted(groups.items(), key=region_key):
        seq = (
            _cluster_rows(group)
            if region == _ROW_FLOW_KEY
            else sorted(group, key=lambda ln: (_base_y(ln), _poly_left(ln)))
        )
        for ln in seq:
            ordered.append(ln.model_copy(update={"order": len(ordered)}))
    return ordered


def _draw_furniture(page: np.ndarray, rules: list[Rule]) -> None:
    """Print the layout's column separators / table rules / register leaders onto the paper, like a
    pre-ruled ledger. ``style="dashed"`` draws a dotted leader (a register's name-to-value dashes) instead
    of a solid rule."""
    for r in rules:
        colour = (70, 70, 82)
        if r.style == "dashed":  # leader dots/dashes bridging a register entry to its value
            length = max(1, int(np.hypot(r.x2 - r.x1, r.y2 - r.y1)))
            for t in range(0, length, 14):  # 8px dash + 6px gap
                f0, f1 = t / length, min(1.0, (t + 8) / length)
                p0 = (int(r.x1 + (r.x2 - r.x1) * f0), int(r.y1 + (r.y2 - r.y1) * f0))
                p1 = (int(r.x1 + (r.x2 - r.x1) * f1), int(r.y1 + (r.y2 - r.y1) * f1))
                cv2.line(page, p0, p1, colour, r.width, cv2.LINE_AA)
        else:
            cv2.line(page, (r.x1, r.y1), (r.x2, r.y2), colour, r.width, cv2.LINE_AA)


def _draw_edits(page: np.ndarray, lines: list[LineLabel]) -> None:
    """Ink the manuscript edits the layout tagged: a strikethrough across each ``del`` line, a small
    insertion caret under each ``add`` line, and a thin underline under each ``underline`` line. The text
    stays in the label — this only marks it visually."""
    for ln in lines:
        if not ln.polygon or ln.tag not in ("del", "add", "underline"):
            continue
        xs = [p[0] for p in ln.polygon]
        ys = [p[1] for p in ln.polygon]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        if ln.tag == "underline":  # a thin ink underline just below the ink (names / dates / places)
            yb = min(page.shape[0] - 1, y1 + max(2, (y1 - y0) // 12))
            cv2.line(page, (x0, yb), (x1, yb), (55, 60, 82), max(1, (y1 - y0) // 16), cv2.LINE_AA)
        elif ln.tag == "del":  # a thin, semi-transparent ink-brown strike — not a hard black bar
            yc = (y0 + y1) // 2
            overlay = page.copy()
            cv2.line(overlay, (x0, yc), (x1, yc), (55, 60, 82), max(1, (y1 - y0) // 18), cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.6, page, 0.4, 0, page)
        else:  # add: a caret '^' tucked under the start of the interlinear word
            cv2.line(page, (x0 - 6, y1), (x0, y1 - 7), (50, 50, 62), 1, cv2.LINE_AA)
            cv2.line(page, (x0, y1 - 7), (x0 + 6, y1), (50, 50, 62), 1, cv2.LINE_AA)


def _sample_ink(rng: random.Random, aug: AugmentConfig) -> np.ndarray:
    """One page's body-ink BGR colour: a base from the palette, lightened toward paper by a random fade.
    Returned as a ``(1, 1, 3)`` multiplier so :func:`_place_and_blend` can broadcast it over an ink ROI."""
    base = rng.choice(aug.ink_colors) if aug.ink_colors else (0.05, 0.05, 0.06)
    fade = rng.uniform(*aug.ink_fade)
    tint = [channel + fade * (1.0 - channel) for channel in base]
    return np.array(tint, np.float32).reshape(1, 1, 3)


class PageComposer:
    """Seat line inks onto a paper page ONE AT A TIME so a caller can render (and stream) lines lazily.

    :func:`compose_page_from_inks` is the batch form; this exposes the very same steps as a stateful
    object: the paper, printed furniture and the page's single ink colour are built up front; each
    :meth:`seat` blends one BGRA ink into its slot (``page`` grows after every call); :meth:`finish`
    then assigns reading order, draws the manuscript edits and runs the scan/aging pass. Deterministic
    given ``seed`` — seating N inks in box order then :meth:`finish` is byte-identical to
    :func:`compose_page_from_inks`, so the streaming studio and the batch corpus generator agree.
    """

    def __init__(
        self,
        layout: Layout,
        *,
        paper_dir: str | None = None,
        seed: int = 0,
        aug: AugmentConfig | None = None,
        paper_tone: PaperTone | None = None,
    ) -> None:
        self.layout = layout
        self.spec: PageSpec = layout.spec
        self.aug = aug or AugmentConfig()
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.page = paper_background(
            self.spec.width, self.spec.height, paper_dir, self._rng, self._np_rng, tone=paper_tone
        )
        _draw_furniture(self.page, layout.rules)  # rules printed first; text written over them
        self._ink_tint = _sample_ink(self._rng, self.aug)  # one writer/ink colour for the whole page
        self.lines: list[LineLabel] = []

    def seat(self, box: LineBox, ink: np.ndarray) -> LineLabel | None:
        """Blend one line's BGRA ``ink`` into ``box``'s slot; return its recovered label (or None if empty)."""
        placed = _place_and_blend(self.page, ink, box, self.spec, self._ink_tint)
        if placed is None:
            return None
        label = LineLabel(
            text=box.text,
            polygon=placed.polygon,
            baseline=placed.baseline,
            region=box.region,
            words=placed.words,
            tag=box.tag,
            printed=box.printed,
        )
        self.lines.append(label)
        return label

    def finish(self, *, trace: dict[str, float] | None = None) -> tuple[np.ndarray, list[LineLabel]]:
        """Assign reading order, draw edits, run the scan pass; return ``(page_bgr, lines)``.

        ``trace`` is forwarded to :func:`scan_pass` so an audit can see which scan features fired.
        """
        self.lines = _assign_reading_order(self.lines)  # true reading order, not seating order
        _draw_edits(self.page, self.lines)  # strike "del" + caret "add", before aging so they warp too
        return scan_pass(
            self.page,
            self.lines,
            spec=self.spec,
            rng=self._rng,
            np_rng=self._np_rng,
            aug=self.aug,
            trace=trace,
        )


def compose_page_from_inks(
    layout: Layout,
    inks: list[np.ndarray],
    *,
    paper_dir: str | None = None,
    seed: int = 0,
    aug: AugmentConfig | None = None,
    paper_tone: PaperTone | None = None,
    trace: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[LineLabel]]:
    """Seat already-rendered line ``inks`` (1:1 with ``layout.boxes``, BGRA) onto a paper page and run the
    scan pass. Deterministic given ``seed``. Splitting render from blend lets the renderer batch every line
    of every page in a single GPU pass before any compositing happens. ``aug`` retunes the scan look,
    ``paper_tone`` sets the paper hue/luminance, and ``trace`` collects which scan features fired.
    """
    composer = PageComposer(layout, paper_dir=paper_dir, seed=seed, aug=aug, paper_tone=paper_tone)
    for box, ink in zip(layout.boxes, inks, strict=True):
        composer.seat(box, ink)
    return composer.finish(trace=trace)


def compose_page(
    layout: Layout,
    renderer: LineRenderer,
    *,
    paper_dir: str | None = None,
    seed: int = 0,
    aug: AugmentConfig | None = None,
    paper_tone: PaperTone | None = None,
    trace: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[LineLabel]]:
    """Assemble a page from a layout + a per-line renderer (placeholder / Eruku / single-line Diffu).

    For the fast Diffu path, render every box of every page up-front in one batched pass and call
    :func:`compose_page_from_inks` directly instead. ``aug`` retunes the augmentation pipeline (see
    :class:`~diffu_page.augment.AugmentConfig`); ``paper_tone`` sets the scan-profile paper hue; ``trace``
    collects the scan features that fired. Returns ``(page_bgr, lines)``.
    """
    inks = render_boxes(layout.boxes, renderer)
    return compose_page_from_inks(
        layout, inks, paper_dir=paper_dir, seed=seed, aug=aug, paper_tone=paper_tone, trace=trace
    )
