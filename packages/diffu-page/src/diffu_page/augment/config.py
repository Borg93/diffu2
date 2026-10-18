"""The scan-augmentation control surface — every probability and strength range in one value object."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AugmentConfig(BaseModel):
    """Tunable knobs for the scan-augmentation pipeline — every probability and strength range in one
    documented, overridable place instead of magic numbers scattered through the helpers. Tuples are
    ``(min, max)`` ranges sampled per page; pass a custom instance to ``compose_page`` to retune."""

    model_config = {"frozen": True}

    # photometric aging (_degrade)
    vignette: tuple[float, float] = (0.08, 0.42)  # uneven platen light falling off toward the edges
    bleed: tuple[float, float] = (
        0.03,
        0.13,
    )  # verso show-through strength (kept subtle; 0.28 read as a loud mirror)
    foxing_tint: tuple[float, float, float] = (1.0, 0.78, 0.45)  # BGR stain weights -> brown/sepia foxing

    # binding (_binding_shadow, _spine_warp, _gutter_focus)
    spine_shadow: tuple[float, float] = (
        0.20,
        0.72,
    )  # gutter valley darkness (deep enough to read as a spine)
    spine_warp: tuple[float, float] = (0.06, 0.15)  # localized binding compression (fraction of the zone)
    gutter_blur: tuple[float, float] = (1.2, 3.0)  # focus-falloff blur sigma near the spine

    # scanner bed (_scanner_warp, _mount_on_bed, _draw_clasps)
    skew: float = Field(default=0.008, ge=0.0)  # global perspective jitter (fraction of the page)
    bed_prob: float = Field(default=0.7, ge=0.0, le=1.0)  # fraction of pages mounted on a dark scanner bed
    edge_damage: tuple[float, float] = (0.0015, 0.007)  # fore-edge eat-in depth (fraction of min dimension)
    clasp_prob: float = Field(default=0.2, ge=0.0, le=1.0)  # fraction of bedded pages with a clasp

    # occasional document marks (probability-gated per page)
    correction_prob: float = Field(default=0.15, ge=0.0, le=1.0)  # ink strikethroughs / blots over text
    stamp_prob: float = Field(default=0.12, ge=0.0, le=1.0)  # an archival stamp / seal overlay
    fold_prob: float = Field(default=0.2, ge=0.0, le=1.0)  # a fold/crease shadow or brown water tide-line
    dust_prob: float = Field(default=0.35, ge=0.0, le=1.0)  # dust specks + the odd scratch on the glass

    # physical-handling marks (probability-gated per page; all photometric/edge -> labels never move)
    glove_prob: float = Field(default=0.15, ge=0.0, le=1.0)  # cotton-gloved operator fingertip at an edge
    hole_prob: float = Field(default=0.06, ge=0.0, le=1.0)  # worm/punch holes in the margin (backing shows)
    curl_prob: float = Field(default=0.15, ge=0.0, le=1.0)  # a lifted/curled brittle corner + fold shadow

    # scribal furniture (ink, drawn over the seated page before aging; decoration the recognizer segments
    # around). Default OFF so the vanilla config stays byte-identical; the scan profiles enable them.
    paraph_prob: float = Field(default=0.0, ge=0.0, le=1.0)  # a looping paraph/flourish at a column foot
    margin_mark_prob: float = Field(default=0.0, ge=0.0, le=1.0)  # keyed reference marks in the outer margin

    # body-ink colour (compose blend): full-coverage BGR multipliers (0..1) sampled per page. These are the
    # DARK iron-gall/sepia STROKE-CORE colours (R > G < B warm, near-black), NOT the light per-stroke
    # average — real Swedish archival ink reads dark and high-contrast against the paper.
    ink_colors: list[tuple[float, float, float]] = Field(
        default_factory=lambda: [
            (0.12, 0.14, 0.20),  # dark iron-gall brown-black (the common one)
            (0.08, 0.09, 0.12),  # near-black
            (0.16, 0.20, 0.31),  # mid brown-black
            (0.20, 0.27, 0.42),  # warm sepia (still clearly legible)
            (0.14, 0.16, 0.23),  # warm dark brown
            (0.22, 0.20, 0.16),  # blue-black (B > R)
        ]
    )
    ink_fade: tuple[float, float] = (0.0, 0.18)  # per-page lighten toward paper; small so ink stays legible

    # whole-sheet geometry + alternate imaging modalities
    rotate: tuple[float, float] = (0.0, 2.5)  # whole-page rotation magnitude in degrees (sign random)
    wash: tuple[float, float] = (
        0.0,
        0.45,
    )  # low-contrast brown damp wash on dirtier pages (the degraded tail)
    microfilm_prob: float = Field(default=0.12, ge=0.0, le=1.0)  # render as high-contrast bitonal microfilm
