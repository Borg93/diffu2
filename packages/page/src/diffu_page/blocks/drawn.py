"""A drawn crop, as the crop blocks give it back, and how the crops of a page are named and seeded."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict

from diffu_page.plan import CropPlan

CROP_OUTPUTS = ["images", "line_reads", "line_cers", "labels_xml", "unclear_lines"]  # what a drawn crop keeps


def crop_identity(page_name: str, seed: int, k: int) -> tuple[str, int]:
    """Crop ``k`` of a page: its name and the noise seed it is drawn from."""
    return f"{page_name}_{k}", seed + k


class DrawnCrop(BaseModel):
    """One drawn crop: its name and plan, its image, what the HTR model read in each line box and that read's CER
    against the line's text, and its labels."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    plan: CropPlan
    image: Image.Image
    line_reads: list[str]
    line_cers: list[float]
    labels_xml: str
    unclear_lines: int

    @classmethod
    def from_outputs(cls, name: str, plan: CropPlan, outputs: Mapping[str, Any]) -> DrawnCrop:
        """The crop from the crop blocks' :data:`CROP_OUTPUTS`."""
        return cls(
            name=name,
            plan=plan,
            image=outputs["images"][0],
            line_reads=outputs["line_reads"],
            line_cers=outputs["line_cers"],
            labels_xml=outputs["labels_xml"],
            unclear_lines=outputs["unclear_lines"],
        )
