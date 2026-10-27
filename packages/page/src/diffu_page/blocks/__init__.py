"""Our Modular Diffusers blocks: the page steps before any model (layout, marking from the scan, the page of an opening,
refill, crop plans) and the crop steps before
a model's own blocks (crop prompt, a glyph model's glyph map and the reference grid). :mod:`diffu_page.pipeline`
assembles them around a model's blocks; the judge (:mod:`diffu_page.judge`) and the labels
(:mod:`diffu_page.blocks.ground_truth`) run after."""

from diffu_page.blocks.crop import CropPromptStep, GlyphMapStep, ReferenceGridStep
from diffu_page.blocks.drawn import DrawnCrop, crop_identity
from diffu_page.blocks.page import LoadLayoutStep, MarkFromScanStep, OpeningPageStep, PlanCropsStep, RefillStep

__all__ = [
    "CropPromptStep",
    "DrawnCrop",
    "GlyphMapStep",
    "LoadLayoutStep",
    "MarkFromScanStep",
    "OpeningPageStep",
    "PlanCropsStep",
    "ReferenceGridStep",
    "RefillStep",
    "crop_identity",
]
