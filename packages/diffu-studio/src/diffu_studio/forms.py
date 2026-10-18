"""Frozen Pydantic value-objects for the studio's UI state — one validated form per generation mode.

The Gradio views collect raw widget values into these forms at the request boundary (see
:mod:`diffu_studio.controls`), so the streaming services in :mod:`diffu_studio.handlers` take ONE typed form
instead of the 24-plus positional arguments the old god-handlers carried. Each form knows how to lower itself
to the real ``diffu-page`` config it drives, so the knob names live in exactly one place.
"""

from __future__ import annotations

import re

from diffu_page.augment import AugmentConfig
from diffu_page.config import GenConfig
from pydantic import BaseModel, Field

from diffu_studio.configs import SampleSettings, build_augment, build_gen_config
from diffu_studio.page import LayoutDetail, PageRequest

# Presence gates: on a corpus these are "fraction of pages that get the effect", but in the single-page studio
# a probability is meaningless — for one page + one seed the gate either fires or not, so nudging it looks dead.
# The studio forces each to a deterministic on/off (>0 → always apply), so a knob visibly changes THIS page.
# (The seed still re-rolls WHERE/how the effect lands — see the Reroll button.)
_AUGMENT_GATES = frozenset({
    "bed_prob", "microfilm_prob", "correction_prob", "stamp_prob", "fold_prob", "dust_prob",
    "clasp_prob", "glove_prob", "hole_prob", "curl_prob", "paraph_prob", "margin_mark_prob",
})  # fmt: skip


def _force(prob: float) -> float:
    """A presence probability → a deterministic gate: any positive value always applies, 0 never does."""
    return 1.0 if prob > 0 else 0.0


def _csv(text: str) -> list[str]:
    """A comma-separated textbox value → a clean list (empty box = empty list = the feature off)."""
    return [part.strip() for part in text.split(",") if part.strip()]


class AugmentForm(BaseModel):
    """The curated scan-look knobs — mirrors :func:`diffu_studio.configs.build_augment` field-for-field."""

    model_config = {"frozen": True}

    bed_prob: float = 0.7
    microfilm_prob: float = 0.12
    correction_prob: float = 0.15
    stamp_prob: float = 0.12
    fold_prob: float = 0.2
    dust_prob: float = 0.35
    rotate_max: float = 2.5
    wash_max: float = 0.35
    ink_fade_max: float = 0.18
    skew: float = 0.008
    ink_color: str | None = None
    vignette_max: float = 0.42
    bleed_max: float = 0.13
    spine_shadow_max: float = 0.72
    spine_warp_max: float = 0.15
    gutter_blur_max: float = 3.0
    edge_damage_max: float = 0.007
    clasp_prob: float = 0.2
    glove_prob: float = 0.15
    hole_prob: float = 0.06
    curl_prob: float = 0.15
    paraph_prob: float = 0.0
    margin_mark_prob: float = 0.0
    foxing_tint: str | None = None

    def to_augment(self) -> AugmentConfig:
        """Lower to the real :class:`AugmentConfig`, forcing presence gates deterministic (studio single-page)."""
        values = self.model_dump()
        for name in _AUGMENT_GATES:
            values[name] = _force(values[name])
        return build_augment(**values)


class LineForm(BaseModel):
    """Single-line generation state — the text plus the shared sampling settings (no model choice)."""

    model_config = {"frozen": True}

    text: str
    settings: SampleSettings = Field(default_factory=SampleSettings)


class PageForm(BaseModel):
    """The whole-page control state — flat knob values plus a nested :class:`AugmentForm`."""

    model_config = {"frozen": True}

    text: str = ""
    seed: int = 0
    page_number: str | None = None
    xml_layout: str | None = None
    formats: list[str] = Field(default_factory=lambda: ["page", "alto", "json"])
    steps: int = 24
    cfg_scale: float = 5.0
    template: str = "single_column"
    n_columns: int = 2
    width: int = 1240
    height: int = 1754
    line_height: int = 100
    line_gap: float = 0.18
    para_gap: float = 0.4
    furniture_prob: float = 0.5
    typeset_furniture: bool = False
    binding_prob: float = 0.4
    insertion_prob: float = 0.012
    deletion_prob: float = 0.008
    # --- layout detail (PageSpec-only knobs; lowered to LayoutDetail — see layout_detail()) ---
    header_scale: float = 1.6
    marg_scale: float = 0.95
    margin_ratio: float = 0.06
    px_per_char: float = 15.0
    drop_cap: bool = False
    catchword: str = ""  # blank = off
    section_headings: str = ""  # comma-separated rubrics; blank = off
    enum_prob: float = 0.0
    enum_marker: str = "§"
    underline_prob: float = 0.0
    margin_keys: str = ""  # comma-separated keyed margin notes; blank = off
    table_rows: int = 14
    table_cols: int = 3
    kv_label_frac: float = 0.34
    # --- "Surprise me": corpus-mix sampled spec instead of the hand-set layout fields ---
    surprise: bool = False
    spread_frac: float = 0.35
    augment: AugmentForm = Field(default_factory=AugmentForm)

    def gen_config(self) -> GenConfig:
        """Lower the page knobs to a validated :class:`GenConfig` (raises on out-of-range values)."""
        return build_gen_config(
            width=self.width,
            height=self.height,
            template=self.template,
            n_columns=self.n_columns,
            line_height=self.line_height,
            line_gap=self.line_gap,
            para_gap=self.para_gap,
            furniture_prob=_force(
                self.furniture_prob
            ),  # deterministic on/off in the studio (see _AUGMENT_GATES)
            typeset_furniture=self.typeset_furniture,
            binding_prob=_force(self.binding_prob),  # "binding prob" → a real toggle for this page
            insertion_prob=self.insertion_prob,
            deletion_prob=self.deletion_prob,
            steps=self.steps,
            cfg_scale=self.cfg_scale,
            formats=self.formats,
            augment=self.augment.to_augment(),
        )

    def layout_detail(self) -> LayoutDetail:
        """Lower the flat widget values to the typed PageSpec-side knobs (comma lists split, blank = off).

        Validation (ranges, positivity) happens here on the :class:`LayoutDetail` model, mirroring how
        :meth:`gen_config` validates via :class:`GenConfig`.
        """
        return LayoutDetail(
            header_scale=self.header_scale,
            marg_scale=self.marg_scale,
            margin_ratio=self.margin_ratio,
            px_per_char=self.px_per_char,
            drop_cap=self.drop_cap,
            catchword=self.catchword.strip() or None,
            section_headings=_csv(self.section_headings),
            enum_prob=self.enum_prob,
            enum_marker=self.enum_marker,
            underline_prob=self.underline_prob,
            margin_keys=_csv(self.margin_keys),
            table_rows=self.table_rows,
            table_cols=self.table_cols,
            kv_label_frac=self.kv_label_frac,
        )

    def to_request(self, *, animate: bool) -> PageRequest:
        """Assemble the :class:`PageRequest` the page stream consumes.

        Paragraphs are separated by BLANK lines (standard prose convention); single newlines flow together
        into one paragraph and re-wrap to the layout. (Previously every newline made its own paragraph, so
        a few lines of text got a paragraph gap between each — the page looked far airier than intended.)
        """
        return PageRequest(
            paragraphs=[
                " ".join(block.split()) for block in re.split(r"\n\s*\n", self.text) if block.strip()
            ],
            gen=self.gen_config(),
            settings=SampleSettings(steps=self.steps, cfg_scale=self.cfg_scale),
            seed=self.seed,
            page_number=self.page_number or None,
            detail=self.layout_detail(),
            surprise=self.surprise,
            spread_frac=self.spread_frac,
            xml_layout=self.xml_layout,
            animate=animate,
        )
