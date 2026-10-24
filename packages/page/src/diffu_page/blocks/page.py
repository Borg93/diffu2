"""The page steps that run before any model: a template page read as its layout, refilled with new text and planned
into the crops a model draws. CPU only; they need no components."""

from __future__ import annotations

from pathlib import Path
from typing import override

from diffusers.modular_pipelines import InputParam, ModularPipeline, ModularPipelineBlocks, OutputParam, PipelineState

from diffu_page.layout import Layout, load_xml_layout, refill
from diffu_page.plan import PageStyle, Reference, plan_page


class LoadLayoutStep(ModularPipelineBlocks):
    """A PAGE or ALTO page read as its layout: its size and its text lines in reading order."""

    @property
    @override
    def description(self) -> str:
        return "Reads a PAGE or ALTO file as its layout: the page size, and each line's text, box, baseline and region."

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [InputParam("template", type_hint=Path, required=True, description="A PAGE or ALTO file to redraw.")]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("layout", type_hint=Layout, description="The template's size and text lines.")]

    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        block_state.layout = load_xml_layout(Path(block_state.template))
        self.set_block_state(state, block_state)
        return components, state


class RefillStep(ModularPipelineBlocks):
    """New text written into a layout's running-text lines, at the page's own writing density."""

    @property
    @override
    def description(self) -> str:
        return "Rewrites the layout's running-text lines with the text; margin notes, headings and page numbers stay."

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("layout", type_hint=Layout, required=True, description="The page to write into."),
            InputParam("text", type_hint=str, required=True, description="The running text to write."),
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("layout", type_hint=Layout, description="The layout holding the new text.")]

    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        block_state.layout = refill(block_state.layout, [block_state.text])
        self.set_block_state(state, block_state)
        return components, state


class PlanCropsStep(ModularPipelineBlocks):
    """A layout cut into the crops the models were trained on, each with its caption in the page's style, or as a
    crop drawn from the page's reference is captioned."""

    @property
    @override
    def description(self) -> str:
        return "Plans the layout as training-scale crops (diffu_page.plan): size, caption, line boxes, reading order."

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("layout", type_hint=Layout, required=True, description="The page to cut."),
            InputParam("style", type_hint=PageStyle, required=True, description="The archive and look to name."),
            InputParam(
                "reference",
                type_hint=Reference,
                description="A crop to draw the page in the hand of: the captions then take the form its LoRA was "
                "trained on (diffu_page.plan.Reference).",
            ),
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("crop_plans", type_hint=list, description="The page's CropPlans, in page order.")]

    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        block_state.crop_plans = plan_page(block_state.layout, block_state.style, reference=block_state.reference)
        self.set_block_state(state, block_state)
        return components, state
