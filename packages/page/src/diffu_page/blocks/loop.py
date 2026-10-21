"""Drawing every crop of a page: the crop blocks run once per planned crop, each crop on its own pipeline state."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, override

import torch
from diffusers.modular_pipelines import (
    InputParam,
    ModularPipeline,
    OutputParam,
    PipelineState,
    SequentialPipelineBlocks,
)
from PIL import Image
from pydantic import BaseModel, ConfigDict

from diffu_page.plan import CropPlan

CROP_OUTPUTS = ["images", "line_reads", "line_cers", "labels_xml", "unclear_lines"]  # what a drawn crop keeps
_PER_CROP = frozenset({"crop_plan", "generator", "image_name"})  # crop blocks' inputs the loop sets for each crop


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


class CropLoopStep(SequentialPipelineBlocks):
    """Runs its crop blocks (see :func:`diffu_page.pipeline.crop_blocks`) once per crop plan of a page.

    Crop ``k`` is named ``<page_name>_<k>`` and drawn from noise seed ``seed + k``; every other input of the crop
    blocks (steps, reference image, label format, ...) is the page's.
    """

    @property
    @override
    def description(self) -> str:
        return "Draws every planned crop of a page with the crop blocks, crop k from noise seed `seed + k`."

    @property
    @override
    def inputs(self) -> list[InputParam]:
        # optional here: the crop blocks check what they require when they run, as a pipeline call does (their
        # combined inputs also list values that their own steps produce, such as the latents)
        shared = [dataclasses.replace(param, required=False) for param in super().inputs if param.name not in _PER_CROP]
        return [
            InputParam("crop_plans", type_hint=list, required=True, description="The page's crops to draw."),
            InputParam("page_name", type_hint=str, default="page", description="Crop k is named <page_name>_<k>."),
            InputParam("seed", type_hint=int, default=0, description="Crop k is drawn from noise seed `seed + k`."),
            *shared,
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("crops", type_hint=list, description="Every crop drawn (DrawnCrop), in plan order.")]

    def _crop_state(self, state: PipelineState, own: dict[str, object]) -> PipelineState:
        """A crop's pipeline state: its ``own`` inputs, and the page's value (or the default) of every other input
        the crop blocks take."""
        crop_state = PipelineState()
        for param in super().inputs:
            if param.name is None:
                continue
            value = own[param.name] if param.name in own else state.get(param.name)
            crop_state.set(param.name, param.default if value is None else value, param.kwargs_type)
        return crop_state

    @override
    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        crops: list[DrawnCrop] = []
        for k, plan in enumerate(block_state.crop_plans):
            name, seed = crop_identity(block_state.page_name, block_state.seed, k)
            own = {"crop_plan": plan, "generator": torch.Generator().manual_seed(seed), "image_name": f"{name}.jpg"}
            _, crop_state = super().__call__(components, self._crop_state(state, own))
            crops.append(DrawnCrop.from_outputs(name, plan, crop_state.get(CROP_OUTPUTS)))
        block_state.crops = crops
        self.set_block_state(state, block_state)
        return components, state
