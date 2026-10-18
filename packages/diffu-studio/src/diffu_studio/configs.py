"""Bridge the UI/API controls to the real ``diffu-page`` configs — one place that knows the knob names.

The control panel binds to :class:`~diffu_page.config.GenConfig` and :class:`~diffu_page.augment.AugmentConfig`
so a slider genuinely changes the corpus config (not a parallel copy that drifts). The UI exposes a curated
set; the API accepts a full ``augment`` dict, so *everything* is reachable.
"""

from __future__ import annotations

from diffu_page.augment import AugmentConfig
from diffu_page.config import Fmt, GenConfig, Template
from pydantic import BaseModel, Field


class SampleSettings(BaseModel):
    """Denoise-sampling knobs shared by single-line and whole-page generation."""

    model_config = {"frozen": True}

    steps: int = Field(default=24, ge=1, le=80)  # denoise iterations — more = smoother, slower
    cfg_scale: float = Field(
        default=5.0, ge=0.0, le=12.0
    )  # text-following strength (~5 is the legible sweet spot)
    auto_width: bool = True  # size the canvas to the text, then ink-crop the blank tail
    width: int = Field(default=448, ge=64, le=2048)  # fixed canvas width, only used when auto_width is off


def list_templates() -> list[str]:
    return [t.value for t in Template]


def list_formats() -> list[str]:
    return [f.value for f in Fmt]


def default_configs() -> dict[str, object]:
    """Defaults for both configs (JSON-mode dicts) — the initial state of the control panel + the API."""
    return {
        "gen": GenConfig().model_dump(mode="json"),
        "augment": AugmentConfig().model_dump(mode="json"),
        "templates": list_templates(),
        "formats": list_formats(),
    }


def _hex_to_bgr(color: str | None) -> tuple[float, float, float] | None:
    """A ``#rrggbb`` UI colour → a BGR triple in ``[0,1]`` (AugmentConfig's ink-colour convention)."""
    if not color:
        return None
    h = color.lstrip("#")
    if len(h) != 6:
        return None
    r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (b, g, r)


def build_augment(
    *,
    bed_prob: float = 0.7,
    microfilm_prob: float = 0.12,
    correction_prob: float = 0.15,
    stamp_prob: float = 0.12,
    fold_prob: float = 0.2,
    dust_prob: float = 0.35,
    rotate_max: float = 2.5,
    wash_max: float = 0.35,
    ink_fade_max: float = 0.18,
    skew: float = 0.008,
    ink_color: str | None = None,
    # --- newly exposed knobs (all photometric/edge → the cheap recompose path, no model) ---
    vignette_max: float = 0.42,
    bleed_max: float = 0.13,
    spine_shadow_max: float = 0.72,
    spine_warp_max: float = 0.15,
    gutter_blur_max: float = 3.0,
    edge_damage_max: float = 0.007,
    clasp_prob: float = 0.2,
    glove_prob: float = 0.15,
    hole_prob: float = 0.06,
    curl_prob: float = 0.15,
    paraph_prob: float = 0.0,
    margin_mark_prob: float = 0.0,
    foxing_tint: str | None = None,  # "#rrggbb" → BGR stain weights
) -> AugmentConfig:
    """Curated scan-look controls → an :class:`AugmentConfig`. Range fields (rotate/wash/fade/vignette/bleed/…)
    are driven by their upper bound from one slider; ``ink_color``/``foxing_tint`` set a colour from one hue.
    Every knob here is compositing-only, so the UI wires them all to the cheap re-composite path (no model)."""
    aug = AugmentConfig(
        bed_prob=bed_prob,
        microfilm_prob=microfilm_prob,
        correction_prob=correction_prob,
        stamp_prob=stamp_prob,
        fold_prob=fold_prob,
        dust_prob=dust_prob,
        rotate=(0.0, rotate_max),
        wash=(0.0, wash_max),
        ink_fade=(0.0, ink_fade_max),
        skew=skew,
        vignette=(0.0, vignette_max),
        bleed=(0.0, bleed_max),
        spine_shadow=(0.0, spine_shadow_max),
        spine_warp=(0.0, spine_warp_max),
        gutter_blur=(0.0, gutter_blur_max),
        edge_damage=(0.0, edge_damage_max),
        clasp_prob=clasp_prob,
        glove_prob=glove_prob,
        hole_prob=hole_prob,
        curl_prob=curl_prob,
        paraph_prob=paraph_prob,
        margin_mark_prob=margin_mark_prob,
    )
    updates: dict[str, object] = {}
    if (bgr := _hex_to_bgr(ink_color)) is not None:
        updates["ink_colors"] = [bgr]
    if (fox := _hex_to_bgr(foxing_tint)) is not None:
        updates["foxing_tint"] = fox
    return aug.model_copy(update=updates) if updates else aug


def build_gen_config(
    *,
    width: int = 1240,
    height: int = 1754,
    template: str = "single_column",
    n_columns: int = 2,
    line_height: int = 100,  # GT scale: big script, ~13 lines/page (the compositor fill fix)
    line_gap: float = 0.18,
    para_gap: float = 0.4,
    furniture_prob: float = 0.5,
    typeset_furniture: bool = False,
    binding_prob: float = 0.4,
    insertion_prob: float = 0.012,
    deletion_prob: float = 0.008,
    steps: int = 24,
    cfg_scale: float = 5.0,
    formats: list[str] | None = None,
    augment: AugmentConfig | None = None,
) -> GenConfig:
    """Assemble a validated :class:`GenConfig` from the page-mode controls (raises on out-of-range values)."""
    return GenConfig(
        width=width,
        height=height,
        template=Template(template),
        n_columns=n_columns,
        line_height=line_height,
        line_gap=line_gap,
        para_gap=para_gap,
        furniture_prob=furniture_prob,
        typeset_furniture=typeset_furniture,
        binding_prob=binding_prob,
        insertion_prob=insertion_prob,
        deletion_prob=deletion_prob,
        steps=steps,
        cfg_scale=cfg_scale,
        formats=[Fmt(f) for f in (formats or ["page", "alto", "json"])],
        augment=augment or AugmentConfig(),
    )
