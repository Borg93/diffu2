"""Document + handling marks — corrections, stamps, dust, gloved fingers, holes, curled corners,
and the scribal furniture (paraphs, keyed margin marks). All photometric/edge — labels never move."""

from __future__ import annotations

import random

import cv2
import numpy as np

from ..labels import LineLabel
from .bed import PaperBounds
from .config import AugmentConfig


def _ink_corrections(
    page: np.ndarray, lines: list[LineLabel], rng: random.Random, aug: AugmentConfig
) -> None:
    """Occasional ink strikethroughs over a line + a stray blot — archival corrections. The label text is
    unchanged (struck/blotted text is still transcribed); only the imaging gets harder, like real data."""
    if rng.random() >= aug.correction_prob or not lines:
        return
    for ln in rng.sample(lines, k=min(len(lines), rng.randint(1, 2))):
        xs = [p[0] for p in ln.polygon]  # polygon is a tight hull, not a 4-pt rect -> take its bbox
        ys_ = [p[1] for p in ln.polygon]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys_), max(ys_)
        if x1 > x0 and rng.random() < 0.7:  # strikethrough through the middle of the line
            ys = int((y0 + y1) / 2 + rng.uniform(-0.15, 0.15) * (y1 - y0))
            cv2.line(page, (x0, ys), (x1, ys), (40, 40, 48), max(1, (y1 - y0) // 14), cv2.LINE_AA)
    for _ in range(rng.randint(0, 2)):  # stray ink blots
        cx, cy = rng.randint(0, page.shape[1] - 1), rng.randint(0, page.shape[0] - 1)
        cv2.circle(page, (cx, cy), rng.randint(3, 9), (28, 28, 34), -1, cv2.LINE_AA)


def _stamps(page: np.ndarray, rng: random.Random, aug: AugmentConfig) -> None:
    """An occasional archival stamp/seal — a faded rectangular or oval ink mark with a line or two of
    'text', near a margin/corner. A semi-transparent overlay, so text underneath stays labelled."""
    if rng.random() >= aug.stamp_prob:
        return
    h, w = page.shape[:2]
    sw, sh = rng.randint(w // 8, w // 4), rng.randint(h // 22, h // 12)
    x = rng.choice((rng.randint(0, max(1, w // 6)), rng.randint(w * 2 // 3, max(2, w - sw))))
    y = rng.randint(0, max(1, h - sh))
    colour = rng.choice([(120, 90, 70), (70, 70, 130), (60, 60, 60)])  # faded blue / purple ink / black
    over = page.copy()
    thick = max(2, sh // 10)
    if rng.random() < 0.5:
        cv2.rectangle(over, (x, y), (x + sw, y + sh), colour, thick, cv2.LINE_AA)
    else:
        cv2.ellipse(
            over, (x + sw // 2, y + sh // 2), (sw // 2, sh // 2), 0, 0, 360, colour, thick, cv2.LINE_AA
        )
    for k in range(rng.randint(1, 2)):  # faint stamp 'text' rules
        ly = y + sh * (k + 1) // 3
        cv2.line(over, (x + sw // 6, ly), (x + sw * 5 // 6, ly), colour, max(1, sh // 16), cv2.LINE_AA)
    alpha = rng.uniform(0.3, 0.6)
    cv2.addWeighted(over, alpha, page, 1 - alpha, 0, page)


def _gloved_fingers(
    page: np.ndarray, rng: random.Random, aug: AugmentConfig, trace: dict[str, float] | None = None
) -> None:
    """A cotton-gloved operator fingertip intruding from a page edge — ubiquitous in archive scans where
    someone holds the leaf flat while it is photographed. Drawn as one or two soft cream rounded tips with
    a faint drop shadow on the paper, entering from a side or the bottom (rarely the top) near a corner, so
    it sits over the outer margin and only grazes text. Photometric only — no label moves."""
    if rng.random() >= aug.glove_prob:
        return
    h, w = page.shape[:2]
    m = min(h, w)
    edge = rng.choice(("bottom", "left", "right"))  # held from the sides / bottom, seldom the top
    span = w if edge == "bottom" else h
    base = rng.randint(int(0.12 * span), int(0.80 * span))
    fw = rng.randint(int(m * 0.05), int(m * 0.09))  # fingertip width across
    fl = rng.randint(int(m * 0.10), int(m * 0.20))  # how far it reaches onto the page
    cream = (rng.randint(182, 214), rng.randint(192, 222), rng.randint(200, 230))  # BGR cotton-glove cream
    mask = np.zeros((h, w), np.float32)
    for k in range(rng.randint(1, 2)):
        off = base + k * int(fw * 1.4)
        ang = rng.uniform(-14, 14)
        if edge == "bottom":
            center, axes = (off, h + fl // 3), (fw // 2, fl)
        elif edge == "left":
            center, axes = (-fl // 3, off), (fl, fw // 2)
        else:
            center, axes = (w + fl // 3, off), (fl, fw // 2)
        cv2.ellipse(mask, center, axes, ang, 0, 360, 1.0, -1, cv2.LINE_AA)
    shadow = cv2.GaussianBlur(mask, (0, 0), m * 0.012)  # soft cast shadow just outside the tip
    page[:] = np.clip(
        page.astype(np.float32) * (1.0 - 0.16 * np.clip(shadow - mask, 0.0, 1.0))[..., None], 0, 255
    ).astype(np.uint8)
    tip = cv2.GaussianBlur(mask, (0, 0), m * 0.004)[..., None]  # feather the tip edge onto the paper
    page[:] = np.clip(
        page.astype(np.float32) * (1.0 - tip) + np.array(cream, np.float32) * tip, 0, 255
    ).astype(np.uint8)
    if trace is not None:
        trace["gloved_finger"] = 1.0  # C6


def _hole_center(band: str, b: PaperBounds, pw: int, ph: int, rng: random.Random) -> tuple[int, int]:
    """One hole centre inside the chosen outer margin ``band`` of the paper bounds."""
    if band == "left":
        return rng.randint(b.x0 + int(pw * 0.02), b.x0 + int(pw * 0.08)), rng.randint(b.y0, b.y1 - 1)
    if band == "right":
        return rng.randint(b.x0 + int(pw * 0.92), b.x0 + int(pw * 0.98)), rng.randint(b.y0, b.y1 - 1)
    if band == "top":
        return rng.randint(b.x0, b.x1 - 1), rng.randint(b.y0 + int(ph * 0.02), b.y0 + int(ph * 0.08))
    return rng.randint(b.x0, b.x1 - 1), rng.randint(b.y0 + int(ph * 0.92), b.y0 + int(ph * 0.98))


def _holes(
    page: np.ndarray,
    rng: random.Random,
    aug: AugmentConfig,
    trace: dict[str, float] | None = None,
    bounds: PaperBounds | None = None,
) -> None:
    """Small holes — worm damage, a punch-hole, a nibbled corner — along a margin: a dark irregular
    perforation (the dark scanner backing showing through the leaf) rimmed by a lighter torn edge. Kept in
    the outer margin bands of the PAPER (``bounds``, defaulting to the whole canvas) so a bed-mounted sheet
    gets its holes on the leaf, not on the surrounding scanner bed. Photometric only."""
    if rng.random() >= aug.hole_prob:
        return
    b = bounds or PaperBounds.full(page)
    pw, ph = b.x1 - b.x0, b.y1 - b.y0
    m = min(pw, ph)
    band = rng.choice(("left", "right", "top", "bottom"))
    for _ in range(rng.randint(1, 3)):
        r = max(2, rng.randint(int(m * 0.004), int(m * 0.02)))
        cx, cy = _hole_center(band, b, pw, ph, rng)
        axes = (r, max(2, int(r * rng.uniform(0.6, 1.2))))
        cv2.ellipse(page, (cx, cy), axes, rng.randint(0, 180), 0, 360, (18, 18, 22), -1, cv2.LINE_AA)
        cv2.ellipse(
            page,
            (cx, cy),
            (int(r * 1.3), int(r * 1.15)),
            0,
            0,
            360,
            (150, 158, 172),
            max(1, r // 4),
            cv2.LINE_AA,
        )
    if trace is not None:
        trace["hole"] = 1.0  # C8


def _curled_corner(
    page: np.ndarray, rng: random.Random, aug: AugmentConfig, trace: dict[str, float] | None = None
) -> None:
    """A lifted / curled brittle corner: the leaf's corner peels off the platen, so a triangular wedge
    catches extra light (a bright highlight) and casts a soft shadow along its fold line. Confined to a
    corner, over the margin. Photometric only — no label moves."""
    if rng.random() >= aug.curl_prob:
        return
    h, w = page.shape[:2]
    m = min(h, w)
    size = rng.randint(int(m * 0.08), int(m * 0.18))
    tri = {
        "tl": np.array([[0, 0], [size, 0], [0, size]]),
        "tr": np.array([[w - 1, 0], [w - 1 - size, 0], [w - 1, size]]),
        "bl": np.array([[0, h - 1], [size, h - 1], [0, h - 1 - size]]),
        "br": np.array([[w - 1, h - 1], [w - 1 - size, h - 1], [w - 1, h - 1 - size]]),
    }[rng.choice(("tl", "tr", "bl", "br"))]
    mask = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(mask, tri.astype(np.int32), 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), m * 0.003)[..., None]
    out = page.astype(np.float32) + mask * rng.uniform(22, 55)  # lifted paper catches light
    fold = np.zeros((h, w), np.float32)
    cv2.line(fold, tuple(tri[1]), tuple(tri[2]), 1.0, max(2, int(m * 0.006)), cv2.LINE_AA)  # the fold seam
    fold = cv2.GaussianBlur(fold, (0, 0), m * 0.005)[..., None]
    page[:] = np.clip(out * (1.0 - 0.22 * fold), 0, 255).astype(np.uint8)
    if trace is not None:
        trace["curled_corner"] = 1.0  # C8


def _dust(page: np.ndarray, rng: random.Random, aug: AugmentConfig) -> None:
    """Dust specks + the odd scratch on the scanner glass — tiny dark dots and thin dark lines."""
    if rng.random() >= aug.dust_prob:
        return
    h, w = page.shape[:2]
    for _ in range(rng.randint(8, 40)):  # specks
        cv2.circle(page, (rng.randint(0, w - 1), rng.randint(0, h - 1)), rng.randint(0, 1), (22, 22, 22), -1)
    for _ in range(rng.randint(0, 2)):  # scratches
        x0, y0 = rng.randint(0, w - 1), rng.randint(0, h - 1)
        x1, y1 = x0 + rng.randint(-w // 4, w // 4), y0 + rng.randint(-h // 4, h // 4)
        cv2.line(page, (x0, y0), (x1, y1), (30, 30, 30), 1, cv2.LINE_AA)


def _paraph_stroke(
    page: np.ndarray,
    x: int,
    y: int,
    scale: float,
    colour: tuple[int, int, int],
    thick: int,
    rng: random.Random,
) -> None:
    """One looping cadel flourish: an open loop + a swept tail + a terminal hook."""
    cv2.ellipse(
        page,
        (x, y),
        (max(1, int(scale)), max(1, int(scale * 0.6))),
        rng.randint(-20, 20),
        rng.randint(-30, 30),
        rng.randint(300, 360),
        colour,
        thick,
        cv2.LINE_AA,
    )
    x2, y2 = x + int(scale * rng.uniform(2.0, 4.0)), y + int(scale * rng.uniform(-0.5, 0.5))
    cv2.line(page, (x, y), (x2, y2), colour, thick, cv2.LINE_AA)  # swept tail
    cv2.line(
        page, (x2, y2), (x2 - int(scale * 0.5), y2 - int(scale * 0.6)), colour, thick, cv2.LINE_AA
    )  # hook


def _paraphs(
    page: np.ndarray,
    lines: list[LineLabel],
    rng: random.Random,
    aug: AugmentConfig,
    trace: dict[str, float] | None = None,
) -> None:
    """A scribal paraph / flourish at the foot of the text block — decoration the recognizer must segment
    around. Non-text ink, so no label; drawn before aging so it warps + tones with the page."""
    if (
        aug.paraph_prob <= 0.0 or not lines or rng.random() >= aug.paraph_prob
    ):  # short-circuit: no rng when off
        return
    h, w = page.shape[:2]
    m = min(h, w)
    colour = (rng.randint(35, 62), rng.randint(40, 68), rng.randint(55, 88))  # dark iron-gall ink
    xs = [p[0] for ln in lines for p in ln.polygon]
    ymax = max(p[1] for ln in lines for p in ln.polygon)
    thick = max(1, int(m * 0.0016))
    lo, mid = int(min(xs)), int((min(xs) + max(xs)) / 2)
    for _ in range(rng.randint(1, 2)):
        cx = rng.randint(lo, max(lo + 1, mid))
        cy = min(h - 1, ymax + rng.randint(int(m * 0.008), int(m * 0.05)))
        _paraph_stroke(page, cx, cy, rng.uniform(0.6, 1.4) * m * 0.03, colour, thick, rng)
    if trace is not None:
        trace["paraph"] = 1.0


def _draw_mark(
    page: np.ndarray, x: int, y: int, s: int, colour: tuple[int, int, int], thick: int, rng: random.Random
) -> None:
    """One keyed reference glyph: a cross, an ex, a tick, a crude manicule (pointing hand), or 'NB'."""
    kind = rng.choice(("cross", "x", "tick", "manicule", "nb"))
    if kind == "cross":
        cv2.line(page, (x, y - s), (x, y + s), colour, thick, cv2.LINE_AA)
        cv2.line(page, (x - s, y), (x + s, y), colour, thick, cv2.LINE_AA)
        return
    if kind == "x":
        cv2.line(page, (x - s, y - s), (x + s, y + s), colour, thick, cv2.LINE_AA)
        cv2.line(page, (x - s, y + s), (x + s, y - s), colour, thick, cv2.LINE_AA)
        return
    if kind == "tick":
        cv2.line(page, (x, y - s), (x, y + s), colour, thick, cv2.LINE_AA)
        return
    if kind == "manicule":  # a small fist + a pointing finger
        cv2.circle(page, (x, y), max(1, int(s * 0.6)), colour, thick, cv2.LINE_AA)
        cv2.line(page, (x + int(s * 0.6), y), (x + 2 * s, y), colour, thick, cv2.LINE_AA)
        return
    cv2.putText(
        page, "NB", (x - s, y + s // 2), cv2.FONT_HERSHEY_SIMPLEX, s / 26.0, colour, thick, cv2.LINE_AA
    )


def _margin_marks(
    page: np.ndarray,
    lines: list[LineLabel],
    rng: random.Random,
    aug: AugmentConfig,
    trace: dict[str, float] | None = None,
) -> None:
    """Keyed reference marks in the OUTER margin next to random body lines — the attention/reference marks a
    recognizer must find and segment. Photometric ink in the text-free margin."""
    if aug.margin_mark_prob <= 0.0 or rng.random() >= aug.margin_mark_prob:  # short-circuit: no rng when off
        return
    body = [ln for ln in lines if ln.region in ("body", "xml") and ln.baseline]
    if not body:
        return
    h, w = page.shape[:2]
    m = min(h, w)
    colour = (rng.randint(35, 62), rng.randint(40, 68), rng.randint(55, 88))
    s = max(3, int(m * 0.014))
    thick = max(1, s // 6)
    mx = rng.randint(int(w * 0.02), int(w * 0.045))  # outer margin, clear of the body
    for ln in rng.sample(body, k=min(len(body), rng.randint(1, 3))):
        _draw_mark(page, mx, ln.baseline[0][1], s, colour, thick, rng)
    if trace is not None:
        trace["margin_mark"] = 1.0
