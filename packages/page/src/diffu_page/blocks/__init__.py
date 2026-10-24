"""Our Modular Diffusers blocks: the page steps before any model (layout, refill, crop plans) and the crop steps around
a model's own blocks (crop prompt and reference grid before; HTR read-back and labels after). :mod:`diffu_page.pipeline`
assembles them around a model's blocks."""

from diffu_page.blocks.crop import CropPromptStep, ReferenceGridStep
from diffu_page.blocks.drawn import CROP_OUTPUTS, DrawnCrop, crop_identity
from diffu_page.blocks.ground_truth import MAX_LINE_CER, LabelExportStep, LineReadbackStep
from diffu_page.blocks.page import LoadLayoutStep, PlanCropsStep, RefillStep

__all__ = [
    "CROP_OUTPUTS",
    "MAX_LINE_CER",
    "CropPromptStep",
    "DrawnCrop",
    "LabelExportStep",
    "LineReadbackStep",
    "LoadLayoutStep",
    "PlanCropsStep",
    "ReferenceGridStep",
    "RefillStep",
    "crop_identity",
]
