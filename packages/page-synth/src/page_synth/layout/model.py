"""Layout data model — the value objects the layout stage produces and validates.

:class:`LineBox` is the entire interface to the renderer/compositor; :class:`PageSpec` is the
generation control surface; :class:`Layout` bundles the result. No geometry is computed here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field, field_validator

type Measure = Callable[[str, int], int]  # renderer width predictor: (text, height) -> px width
type Align = Literal["left", "right", "center"]  # how a rendered line sits in its slot


class LineBox(BaseModel):
    """One handwriting line's slot on the page: where it goes + what it says + which region it's in."""

    text: str
    x: int  # left edge of the slot
    y: int  # top of the slot
    w: int  # target render width (~len*px_per_char, capped to the region)
    h: int  # nominal line height (px)
    baseline_y: int  # absolute y the line's baseline should land on
    region: str = "body"  # body | col0.. | table | page_number | header | marginalia | kv_* | xml
    align: Align = "left"
    rotation: float = 0.0  # small per-line skew in degrees (cosmetic realism)
    printed: bool = False  # render as TYPESET print (form headers / row numbers), not handwriting
    tag: str = ""  # editorial tag carried to the label: add (interlinear) | del (struck) | unclear
    block: str = ""  # the source block a replayed line came from (its PAGE TextRegion / ALTO TextBlock id)


class Rule(BaseModel):
    """A furniture line drawn by the compositor (column separator / table rule / register leader)."""

    x1: int
    y1: int
    x2: int
    y2: int
    width: int = 2
    style: Literal["solid", "dashed"] = "solid"  # solid ruled line | dashed (register leader dots/dashes)


class Column(BaseModel):
    """A borderless column for ``template="columns"`` — position/width as page fractions + alignment."""

    x_frac: float
    w_frac: float
    align: Align = "left"


class TableColumn(BaseModel):
    """One column of a typed ruled grid (``template="semi_table"``): a header title, a width ``weight``
    (proportioned across the grid), a content ``kind`` and cell ``align``."""

    model_config = {"frozen": True}

    title: str
    weight: float  # relative width; columns are laid out proportionally to the sum of weights
    kind: str  # rownum | name | place | date | text | money
    align: Align = "left"


class PageSpec(BaseModel):
    """Page geometry + every knob a template samples — THE generation control surface.

    Pass to :func:`build_layout` to synthesize a layout, or to :func:`load_xml_layout` to replay a
    real one. The engine is deterministic given ``(spec, seed)``; vary fields per page for diversity.
    """

    # --- canvas ---
    width: int = 1654  # ~A4 @ 200 dpi, portrait
    height: int = 2339
    margin_ratio: float = 0.06  # page margin as a fraction of min(W,H) — scales with page size (docTR-style)
    # --- line metrics ---
    line_height: int = 48
    line_gap: float = 0.45  # extra vertical gap as a fraction of line_height
    px_per_char: float = 15.0  # corpus density (matches generate.py's width bias)
    max_stretch: float = 2.0  # cap on horizontally stretching a short line to fill its slot (compose); the
    # 8:1 model aspect ceiling means a single crop tops out below a wide column, so allow a firmer fill-out
    baseline_frac: float = 0.78  # baseline sits this far down the line box
    skew: float = 0.6  # max |per-line skew| in degrees
    para_gap: float = 0.6  # blank gap between paragraphs, as a fraction of line_height
    # --- body template ---
    template: str = "single_column"  # single_column|multi_column|semi_table|columns|key_value
    n_columns: int = 2
    gutter_ratio: float = 0.025  # column gutter as a fraction of page width
    table_rows: int = 14
    table_cols: int = 3
    typeset_furniture: bool = False  # True only for genuinely PRE-PRINTED forms; else table furniture
    # (headers, row numbers, money cells) is handwritten through the model, like the real archival registers
    columns: list[Column] = []  # template="columns": arbitrary borderless columns (no rules)
    kv_label_frac: float = 0.34  # template="key_value": label-column width as a fraction of the body
    kv_delim: str = ":"  # split each paragraph into (label, value) on this delimiter
    # --- page elements (real-document furniture; each becomes its own labeled region) ---
    recto: bool = True  # right page -> page number top-RIGHT; verso (False) -> top-LEFT
    page_number: str | None = None  # e.g. "2." ; placed in the outer-top corner
    header_lines: list[str] = []  # optional header/date block at the top (larger font)
    # a header is NEVER smaller than the body (ge=1.0); vary it per page within [1.0, ~2.0].
    header_scale: float = Field(default=1.6, ge=1.0)
    marginalia: list[str] = []  # short notes down the OUTER margin (opposite the binding), read LAST
    # keyed marginal notes (case №, "Resol.", a date) — one per body paragraph, each seated beside the
    # paragraph it annotates and read INTERLEAVED with the body at that point (not appended at the end).
    margin_keys: list[str] = []
    marg_gutter_ratio: float = 0.17  # marginalia gutter as a fraction of page width
    # marginalia is NEVER bigger than the body (le=1.0); vary it per page within (~0.8, 1.0].
    marg_scale: float = Field(default=0.95, gt=0.0, le=1.0)
    # --- binding (book spine): an inner-margin gutter with NO text + a corner-aware spine shadow ---
    binding: bool = False  # draw a book binding/spine on the inner edge
    binding_ratio: float = 0.045  # inner gutter (no-text spine margin) as a fraction of width; real bound
    # text often runs close to the gutter — keep this small and vary it per page (see the CLI)
    # --- manuscript edits (labelled GT): interlinear additions + struck-through deletions + underline ---
    insertion_prob: float = Field(default=0.0, ge=0.0, le=1.0)  # per body line: an interlinear "add" above it
    deletion_prob: float = Field(default=0.0, ge=0.0, le=1.0)  # per body line: strike it through, tag "del"
    underline_prob: float = Field(default=0.0, ge=0.0, le=1.0)  # per body line: underline it (names/dates), tag
    # --- catchword / custos: the next leaf's first word, written at the bottom-outer corner (binding cue) ---
    catchword: str | None = None
    # --- in-prose sectioning (labelled GT): centered rubric headings between paragraphs + § enumerators ---
    section_headings: list[str] = []  # e.g. ["Resolutio", "Utslag:"]; inserted centered every few paragraphs
    enum_prob: float = Field(default=0.0, ge=0.0, le=1.0)  # per paragraph: prefix an article enumerator
    enum_marker: str = "§"  # the enumerator glyph ("§" for the real model; "N:o" renders in the placeholder)
    # enlarge the opening line's first letter into a versal/initial (region "initial"), the enlarged capital
    # that opens a section in the corpus's hands; the rest of the line shifts right to make room.
    drop_cap: bool = False

    @field_validator("template")
    @classmethod
    def _template_is_registered(cls, value: str) -> str:
        # NOT a Literal: TEMPLATES is the open registry (adding a template must stay a one-dict edit),
        # so validate against it instead. Imported lazily — templates.py imports this module.
        from .templates import TEMPLATES

        if value not in TEMPLATES:
            raise ValueError(f"unknown template {value!r}; options: {sorted(TEMPLATES)}")
        return value

    @property
    def spread(self) -> bool:
        """A two-page opening (template ``spread``): a central gutter, not a single-edge spine."""
        return self.template == "spread"

    @property
    def binding_left(self) -> bool:
        """Spine on the LEFT for a recto (right-hand) page, RIGHT for a verso — facing the centre gutter."""
        return self.recto

    @property
    def binding_gutter(self) -> int:
        """Spine gutter in px — a fraction of page width; no text sits in it."""
        return int(self.binding_ratio * self.width)

    @property
    def margin(self) -> int:
        """Page margin in px — a fraction of the smaller page dimension (scales with page size)."""
        return int(self.margin_ratio * min(self.width, self.height))

    @property
    def gutter(self) -> int:
        """Column gutter in px — a fraction of page width."""
        return int(self.gutter_ratio * self.width)

    @property
    def marg_gutter(self) -> int:
        """Marginalia gutter in px — a fraction of page width."""
        return int(self.marg_gutter_ratio * self.width)


class Layout(BaseModel):
    """The full result of the layout stage: line boxes + furniture rules + the spec used."""

    spec: PageSpec
    boxes: list[LineBox] = []
    rules: list[Rule] = []


def validate_layout(layout: Layout, *, overlap_tol: int = 3) -> list[str]:
    """Check a layout is valid for training data; return a list of problems ([] == valid).

    Catches the failure modes that make bad labels: non-positive box sizes, boxes off the page,
    and real overlaps between boxes (e.g. marginalia bleeding into the body, header onto text).
    Lines stacked within one region flow don't overlap by construction; this catches the cross-region
    and degenerate-parameter cases. NOTE: it validates the LABEL boxes — ascender/descender ink may
    extend a few px beyond a box, which is intended (the emitted polygon is the true ink extent).
    """
    spec = layout.spec
    probs: list[str] = []
    b = layout.boxes
    for i, x in enumerate(b):
        if x.w <= 0 or x.h <= 0:
            probs.append(f"box {i} [{x.region}] non-positive size {x.w}x{x.h}")
        if x.x < 0 or x.y < 0 or x.x + x.w > spec.width or x.y + x.h > spec.height:
            probs.append(f"box {i} [{x.region}] off page: ({x.x},{x.y}) {x.w}x{x.h} > {spec.width}x{spec.height}")
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            ix = min(b[i].x + b[i].w, b[j].x + b[j].w) - max(b[i].x, b[j].x)
            iy = min(b[i].y + b[i].h, b[j].y + b[j].h) - max(b[i].y, b[j].y)
            if ix > overlap_tol and iy > overlap_tol:
                probs.append(f"overlap: box {i} [{b[i].region}] & box {j} [{b[j].region}] by {ix}x{iy}px")
    return probs
