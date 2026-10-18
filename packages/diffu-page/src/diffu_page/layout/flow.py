"""Text flow — wrap paragraphs and place line boxes down a region's rect.

Pure geometry helpers shared by every template: character/width estimation, word-wrap (optionally by
a renderer's real measured width), box construction, top-to-bottom flow, the shared page furniture
(page number / header / marginalia / catchword), and replay height normalization.
"""

from __future__ import annotations

import random
import textwrap

from .model import Align, LineBox, Measure, PageSpec


def _chars_per_line(region_w: int, spec: PageSpec) -> int:
    return max(1, int(region_w / spec.px_per_char))


def _wrap(paragraphs: list[str], region_w: int, spec: PageSpec, measure: Measure | None = None) -> list[str]:
    """Word-wrap paragraphs to ``region_w``.

    With ``measure`` (a renderer's width predictor) lines are fit by ACTUAL rendered width, so they
    never overflow the slot and never get shrunk — keeping font size uniform within a paragraph.
    Without it, falls back to a ``px_per_char`` character-count estimate.
    """
    if measure is None:
        width = _chars_per_line(region_w, spec)
        lines: list[str] = []
        for para in paragraphs:
            lines.extend(textwrap.wrap(para, width=width) or [""])
        return lines

    lines = []
    max_chars = _chars_per_line(region_w, spec)  # backstop: measure() saturates on long strings in a wide
    for para in paragraphs:  # column and never breaks -> the whole paragraph crams onto one scaled-tiny line
        words = para.split()
        if not words:
            lines.append("")
            continue
        cur = words[0]
        for w in words[1:]:
            cand = f"{cur} {w}"
            if len(cand) <= max_chars and measure(cand, spec.line_height) <= region_w:
                cur = cand
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def _box(
    text: str,
    x: int,
    y: int,
    w: int,
    h: int,
    spec: PageSpec,
    rng: random.Random,
    region: str,
    align: Align = "left",
    printed: bool = False,
) -> LineBox:
    return LineBox(
        text=text,
        x=x,
        y=y,
        w=w,
        h=h,
        region=region,
        align=align,
        baseline_y=int(y + spec.baseline_frac * h),
        rotation=0.0 if printed else (rng.uniform(-spec.skew, spec.skew) if spec.skew else 0.0),
        printed=printed,
    )


def _flow(
    lines: list[str],
    rect: tuple[int, int, int, int],
    spec: PageSpec,
    rng: random.Random,
    region: str,
    align: Align = "left",
) -> tuple[list[LineBox], int]:
    """Place text lines top-to-bottom inside ``rect``; return (boxes, the y reached).

    The slot spans the full region width so the compositor can ``align`` the (narrower) line within
    it; the emitted label polygon is always the tight ink extent regardless of alignment.
    """
    x0, y0, x1, y1 = rect
    region_w = x1 - x0
    step = int(spec.line_height * (1.0 + spec.line_gap))
    boxes: list[LineBox] = []
    y = y0
    for line in lines:
        if y + spec.line_height > y1:
            break  # region full
        boxes.append(_box(line, x0, y, region_w, spec.line_height, spec, rng, region, align))
        y += step
    return boxes, y


def _page_number_box(spec: PageSpec, rng: random.Random) -> LineBox:
    """The page number in the OUTER-top corner (right for recto, left for verso) as its own region."""
    m, W = spec.margin, spec.width
    pw = max(spec.line_height, int(len(spec.page_number or "") * spec.px_per_char * 1.4))
    ph = int(spec.line_height * 1.1)
    py = int(m * 0.35)
    px = (W - m - pw) if spec.recto else m  # outer corner: recto=right, verso=left
    return _box(
        spec.page_number or "", px, py, pw, ph, spec, rng, "page_number", "right" if spec.recto else "left"
    )


def _marginalia_boxes(spec: PageSpec, rng: random.Random, by0: int, on_left: bool) -> list[LineBox]:
    """Unkeyed running notes down the OUTER margin (spaced out, read last)."""
    m, W, H = spec.margin, spec.width, spec.height
    mh = max(8, int(spec.line_height * spec.marg_scale))  # marginalia <= body size (marg_scale<=1)
    boxes: list[LineBox] = []
    y = by0 + int(spec.line_height * 1.5)
    for note in spec.marginalia:
        if y + mh > H - m:  # guard: don't run off the page bottom
            break
        nw = min(spec.marg_gutter - 16, max(mh, int(len(note) * spec.px_per_char * spec.marg_scale)))
        mx = int(m * 0.3) if on_left else (W - m - nw)
        boxes.append(_box(note, mx, y, nw, mh, spec, rng, "marginalia"))
        y += int(mh * (1 + spec.line_gap)) + rng.randint(spec.line_height * 2, spec.line_height * 4)
    return boxes


def _header_boxes(
    spec: PageSpec, rng: random.Random, by0: int, bx0: int, bx1: int
) -> tuple[list[LineBox], int]:
    """A centered header block atop the body, larger; return (its boxes, the y reached below it)."""
    m, H = spec.margin, spec.height
    hh = int(spec.line_height * spec.header_scale)
    boxes: list[LineBox] = []
    y = by0
    for hl in spec.header_lines:
        if y + hh > H - m:  # guard: don't run header off the page
            break
        hw = min(bx1 - bx0, max(hh, int(len(hl) * spec.px_per_char * spec.header_scale)))
        hx = bx0 + max(0, ((bx1 - bx0) - hw) // 2)  # centered
        boxes.append(_box(hl, hx, y, hw, hh, spec, rng, "header", "center"))
        y += hh + int(spec.line_height * 0.25)  # small gap, not 0.3*hh (which ballooned with scale)
    return boxes, y


def _catchword_box(spec: PageSpec, rng: random.Random, bx0: int, bx1: int, by1: int) -> LineBox:
    """The custos at the bottom-outer corner (right for recto, left for verso)."""
    cw_h = max(8, int(spec.line_height * 0.95))
    cw_w = min(bx1 - bx0, max(spec.line_height, int(len(spec.catchword or "") * spec.px_per_char)))
    cw_y = by1 - cw_h
    cw_x = (bx1 - cw_w) if spec.recto else bx0
    return _box(
        spec.catchword or "",
        cw_x,
        cw_y,
        cw_w,
        cw_h,
        spec,
        rng,
        "catchword",
        "right" if spec.recto else "left",
    )


def _page_elements(spec: PageSpec, rng: random.Random) -> tuple[list[LineBox], tuple[int, int, int, int]]:
    """Add page number / header / marginalia; return (their boxes, the remaining body rect).

    Page number sits in the OUTER-top corner (right for recto, left for verso) as its own region.
    Marginalia reserve a left gutter and indent the body. A header block sits atop the body, larger,
    and pushes the body down. Body templates then flow into the returned rect.
    """
    m, W, H = spec.margin, spec.width, spec.height
    boxes: list[LineBox] = []
    bx0, by0, bx1, by1 = m, m, W - m, H - m

    if spec.binding:  # reserve the spine gutter — NO text/furniture sits on the binding
        if spec.binding_left:
            bx0 += spec.binding_gutter
        else:
            bx1 -= spec.binding_gutter

    if spec.page_number:
        pn = _page_number_box(spec, rng)
        boxes.append(pn)
        by0 = max(by0, pn.y + pn.h + int(m * 0.3))

    if spec.marginalia or spec.margin_keys:  # both reserve the outer gutter; keyed notes are seated in-flow
        on_left = (not spec.binding_left) if spec.binding else True  # OUTER margin, opposite the spine
        boxes.extend(_marginalia_boxes(spec, rng, by0, on_left))
        if on_left:  # indent the body away from the margin, but keep it usable
            bx0 = min(bx0 + spec.marg_gutter, bx1 - 4 * spec.line_height)
        else:
            bx1 = max(bx1 - spec.marg_gutter, bx0 + 4 * spec.line_height)

    if spec.header_lines:
        header, y = _header_boxes(spec, rng, by0, bx0, bx1)
        boxes.extend(header)
        by0 = min(y + int(spec.line_height * 0.5), by1)  # body can't start past its own bottom

    if spec.catchword:  # custos at the bottom-outer corner (right for recto, left for verso)
        cw = _catchword_box(spec, rng, bx0, bx1, by1)
        boxes.append(cw)
        by1 = max(by0 + spec.line_height, cw.y - int(spec.line_height * 0.4))  # reserve the foot strip

    return boxes, (bx0, by0, bx1, by1)


def _margin_key_box(
    text: str, y: int, rect: tuple[int, int, int, int], spec: PageSpec, rng: random.Random
) -> LineBox:
    """A keyed marginal note seated in the reserved gutter beside its paragraph (whichever side the body was
    indented from). Region ``margin_key`` so the compositor reads it interleaved with the body at this ``y``."""
    x0, _, x1, _ = rect
    left_gutter, right_gutter = x0 - spec.margin, (spec.width - spec.margin) - x1
    mh = max(8, int(spec.line_height * spec.marg_scale))
    if left_gutter >= right_gutter:  # body indented from the left -> note sits in the left gutter
        gx = int(spec.margin * 0.3)
        gw = min(max(mh, x0 - gx - 8), max(mh, int(len(text) * spec.px_per_char * spec.marg_scale)))
    else:  # body indented from the right (verso / binding-left) -> right gutter
        gw = min(right_gutter - 8, max(mh, int(len(text) * spec.px_per_char * spec.marg_scale)))
        gx = spec.width - spec.margin - gw
    return _box(text, gx, y, max(mh, gw), mh, spec, rng, "margin_key")


def _flow_paragraphs(
    paragraphs: list[str],
    rect: tuple[int, int, int, int],
    spec: PageSpec,
    rng: random.Random,
    region: str,
    measure: Measure | None = None,
    margin_keys: list[str] | None = None,
) -> list[LineBox]:
    """Flow each paragraph as its own block of lines, with a blank ``para_gap`` between paragraphs.

    A centered rubric ``section_heading`` is inserted before every third paragraph, and (per ``enum_prob``)
    a ``§. N.`` article enumerator is prefixed to a paragraph's text — the in-prose sectioning the corpus
    carries; both are ordinary labelled boxes the recognizer reads and segments around. When ``margin_keys``
    is given, note ``i`` is seated in the gutter beside paragraph ``i`` (region ``margin_key``, read in flow).
    """
    x0, y0, x1, y1 = rect
    region_w = x1 - x0
    step = int(spec.line_height * (1.0 + spec.line_gap))
    gap = int(spec.line_height * spec.para_gap)
    boxes: list[LineBox] = []
    y = y0
    for pi, para in enumerate(paragraphs):
        if spec.section_headings and pi > 0 and pi % 3 == 0 and y + step <= y1:  # centered rubric heading
            htext = spec.section_headings[(pi // 3 - 1) % len(spec.section_headings)]
            hw = min(region_w, max(spec.line_height, int(len(htext) * spec.px_per_char)))
            boxes.append(
                _box(htext, x0 + (region_w - hw) // 2, y, hw, spec.line_height, spec, rng, "rubric", "center")
            )
            y += step
        if spec.enum_prob and rng.random() < spec.enum_prob:  # an article enumerator, read inline
            para = f"{spec.enum_marker} {pi + 1}. {para}"
        if margin_keys and pi < len(margin_keys) and y + spec.line_height <= y1:  # key seated at para start
            boxes.append(_margin_key_box(margin_keys[pi], y, rect, spec, rng))
        for line in _wrap([para], region_w, spec, measure):
            if y + spec.line_height > y1:
                return boxes
            boxes.append(_box(line, x0, y, region_w, spec.line_height, spec, rng, region))
            y += step
        y += gap
    return boxes


def _fit_to_width(text: str, width: int, spec: PageSpec) -> str:
    """Trim a cell's text — whole words first, then a hard character cap — so its natural ink width fits
    ``width``. A ruled table cell renders at its natural width (fixed-cell, never stretched), so an
    over-long note would otherwise spill across the column rule; this keeps every cell inside its column."""
    max_chars = max(1, int(width / spec.px_per_char))
    if len(text) <= max_chars:
        return text
    kept = ""
    for word in text.split():
        candidate = f"{kept} {word}".strip()
        if len(candidate) > max_chars:
            break
        kept = candidate
    return kept or text[:max_chars]


def _normalize_line_heights(boxes: list[LineBox], spec: PageSpec) -> list[LineBox]:
    """Cap each replayed line's height to the gap to the NEXT line in its own column, so a tall ALTO/PAGE
    bbox (inflated by ascenders/descenders) doesn't render text over the line below. Per-column (via
    x-overlap) so a two-column spread or a table isn't collapsed to one uniform height; baselines are kept."""
    if len(boxes) < 2:
        return boxes
    out: list[LineBox] = []
    for b in boxes:
        below = [
            o.y - b.y
            for o in boxes
            if o.y > b.y + 2 and min(b.x + b.w, o.x + o.w) - max(b.x, o.x) > 0.3 * min(b.w, o.w)
        ]
        h = min(b.h, max(spec.line_height, int(min(below) * 1.05))) if below else b.h
        out.append(b.model_copy(update={"h": h, "baseline_y": b.y + int(spec.baseline_frac * h)}))
    return out
