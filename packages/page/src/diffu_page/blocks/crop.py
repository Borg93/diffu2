"""The crop steps that run before a model's own blocks: a crop plan as the model's inputs, and a reference image on the
grid it was trained on."""

from __future__ import annotations

from typing import override

from diffusers.modular_pipelines import InputParam, ModularPipeline, ModularPipelineBlocks, OutputParam, PipelineState

from diffu_page.crops.grid import as_reference
from diffu_page.plan import CropPlan


class CropPromptStep(ModularPipelineBlocks):
    """A planned crop as a model's inputs: its caption, its line boxes and its size."""

    @property
    @override
    def description(self) -> str:
        return "Turns a diffu_page.plan.CropPlan into the prompt, the line boxes (page_lines) and the image size."

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [InputParam("crop_plan", type_hint=CropPlan, required=True, description="The crop to draw.")]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("prompt", type_hint=str, description="The crop's layout caption."),
            OutputParam("page_lines", type_hint=list, description="The crop's LineRecords (boxes on 0-1000)."),
            OutputParam("height", type_hint=int, description="The crop's height in pixels."),
            OutputParam("width", type_hint=int, description="The crop's width in pixels."),
        ]

    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        plan: CropPlan = block_state.crop_plan
        block_state.prompt = plan.caption
        block_state.page_lines = plan.lines
        block_state.height, block_state.width = plan.height, plan.width
        self.set_block_state(state, block_state)
        return components, state


class ReferenceGridStep(ModularPipelineBlocks):
    """Puts a reference image at the size and on the grid the v3 LoRAs were trained on (``as_reference``).

    The training references were shrunk to the reference size (half a target's pixels) and NeMo's cache encoder
    squashed them onto the 16-pixel grid, while FLUX.3's own preprocessing would keep a 1 MP crop at 1 MP and pad it
    out to the grid with a reflected border. Resizing here first makes FLUX.3's preprocessing a no-op, so a reference
    of any size is drawn from as in training. Without a reference the step does nothing.
    """

    @property
    @override
    def description(self) -> str:
        return "Resizes the reference image to the reference size and 16-px grid it was trained at (as_reference)."

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [InputParam("image", required=False, description="The reference image (PIL), if any.")]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("image", description="The reference image on the training grid.")]

    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        if block_state.image is not None:
            block_state.image = as_reference(block_state.image)
        self.set_block_state(state, block_state)
        return components, state
