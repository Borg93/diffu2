"""Our Modular Diffusers blocks: the page steps before any model (layout, refill, crop plans), the crop steps around a
model's own blocks (crop prompt and reference grid before; HTR read-back and labels after), and the loop that draws
every crop of a page. :mod:`diffu_page.pipeline` assembles them around a model's blocks."""

from diffu_page.blocks.crop import CropPromptStep, ReferenceGridStep
from diffu_page.blocks.ground_truth import LabelExportStep, LineReadbackStep
from diffu_page.blocks.loop import CROP_OUTPUTS, CropLoopStep, DrawnCrop, crop_identity
from diffu_page.blocks.page import LoadLayoutStep, LoadReferenceStep, PlanCropsStep, RefillStep

__all__ = [
    "CROP_OUTPUTS",
    "CropLoopStep",
    "CropPromptStep",
    "DrawnCrop",
    "LabelExportStep",
    "LineReadbackStep",
    "LoadLayoutStep",
    "LoadReferenceStep",
    "PlanCropsStep",
    "ReferenceGridStep",
    "RefillStep",
    "crop_identity",
]
