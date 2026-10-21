"""Manuscript-edit passes — labelled GT edits applied over a built layout.

Interlinear insertions, strikethrough deletions, underlines (:func:`_apply_edits`) and the
versal/initial drop cap (:func:`_apply_drop_cap`). All output stays exactly transcribed; the tags
travel to PAGE (custom) + JSON.
"""

from __future__ import annotations

import random

from .flow import _box
from .model import Layout, LineBox

_BODY_REGIONS = ("body", "left", "right")  # regions eligible for manuscript edits (plus any col*)
_INSERTS = ["icke", "afven", "samt", "som", "den", "ej", "ater", "vid", "ock", "har"]


def is_body(region: str) -> bool:
    """Whether a region holds running text (a body or column paragraph, not page furniture)."""
    return region in _BODY_REGIONS or region.startswith("col")


def _apply_edits(layout: Layout, rng: random.Random) -> Layout:
    """Inject manuscript edits as labelled GT: strike some body lines (tag ``del``) and squeeze a small
    interlinear word above others (region ``insertion``, tag ``add``). The struck/added text stays
    transcribed — HTR must read around it — and the tags travel to PAGE (custom) + JSON."""
    spec = layout.spec
    boxes = list(layout.boxes)
    insertions: list[LineBox] = []
    for i, box in enumerate(boxes):
        if not is_body(box.region):
            continue
        if rng.random() < spec.deletion_prob:
            boxes[i] = box.model_copy(update={"tag": "del"})
        elif rng.random() < spec.underline_prob:  # underline a name/date line (elif: never also struck)
            boxes[i] = box.model_copy(update={"tag": "underline"})
        if rng.random() < spec.insertion_prob:
            ih = max(8, int(box.h * 0.55))
            iy = max(0, box.y - int(box.h * 0.45))  # squeezed into the gap above the line
            word = rng.choice(_INSERTS)
            ix = box.x + rng.randint(0, max(1, box.w // 2))
            iw = min(box.w - (ix - box.x), max(ih, int(len(word) * spec.px_per_char * 0.75)))
            ins = _box(word, ix, iy, max(ih, iw), ih, spec, rng, "insertion")
            insertions.append(ins.model_copy(update={"tag": "add"}))
    return layout.model_copy(update={"boxes": boxes + insertions})


_INITIAL_REGIONS = ("body", "reg_header", "header")  # a section can open with any of these; enlarge its first


def _apply_drop_cap(layout: Layout, rng: random.Random) -> Layout:
    """Enlarge the opening line's first letter into a versal/initial — a labelled ``initial`` box ~2x the body
    height, the rest of the line shifted right to clear it. The enlarged capital that opens a section in the
    corpus's oldest hands. The letter and the remainder stay separately transcribed, so the GT is still exact.
    """
    spec = layout.spec
    boxes = list(layout.boxes)
    openers = [(i, b) for i, b in enumerate(boxes) if b.region in _INITIAL_REGIONS and b.text.strip()]
    if not openers:
        return layout
    i, box = min(openers, key=lambda t: (t[1].y, t[1].x))
    first, rest = box.text[0], box.text[1:].lstrip()
    cap_h = int(box.h * 2.0)  # a display initial that clearly reads as a versal against the body line
    cap_w = int(cap_h * 0.8)
    # A versal SHARES its opening line's baseline and rises above it. Seating it 2px proud of that baseline
    # (placement is baseline-driven) is what makes _assign_reading_order — body sorted (baseline, x) — read
    # the initial immediately BEFORE the remainder of its line; a taller box with its own y+frac*h baseline
    # would land ~¼ line BELOW and read after it (the bug this comment guards against).
    base_y = box.baseline_y - 2
    cap_y = max(0, int(base_y - spec.baseline_frac * cap_h))
    initial = _box(first, box.x, cap_y, cap_w, cap_h, spec, rng, "initial")
    initial = initial.model_copy(update={"baseline_y": base_y})
    boxes[i] = box.model_copy(
        update={"text": rest, "x": box.x + cap_w + 6, "w": max(spec.line_height, box.w - cap_w - 6)}
    )
    boxes.append(initial)
    return layout.model_copy(update={"boxes": boxes})
