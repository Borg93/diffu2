"""The page steps that run before any model: a template page read as its layout, marked from its scan if asked, the
page of an opening asked for taken, refilled with new text and planned into the crops a model draws. CPU only; they
need no components."""

from __future__ import annotations

from pathlib import Path
from typing import override

from diffusers.modular_pipelines import InputParam, ModularPipeline, ModularPipelineBlocks, OutputParam, PipelineState

from diffu_page.captions.prompt import PageStyle
from diffu_page.layout import Layout, PageSide, load_xml_layout, refill, scan_pages
from diffu_page.marking import PrintHandClassifier, marked, template_scan
from diffu_page.plan import Reference, plan_page, plan_whole_page


class LoadLayoutStep(ModularPipelineBlocks):
    """A PAGE or ALTO page read as its layout: its size, its text lines in reading order and its ruled lines."""

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


class MarkFromScanStep(ModularPipelineBlocks):
    """A template's layout marked from the scan beside it (:func:`diffu_page.marking.marked`) when a print/hand
    classifier is given: its lines printed or handwritten where the template does not say, its ruled lines where it has
    none. Without a classifier the layout passes as it is."""

    @property
    @override
    def description(self) -> str:
        return (
            "Marks the template from its scan when a print/hand classifier is given: each line it says nothing of read "
            "as printed or handwritten, and its ruled lines found on the scan if it has none."
        )

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("template", type_hint=Path, required=True, description="The PAGE file, its scan beside it."),
            InputParam("layout", type_hint=Layout, required=True, description="The template's layout."),
            InputParam(
                "print_hand",
                type_hint=PrintHandClassifier,
                description="The classifier to read the lines with (diffu_page.marking); none leaves the layout as is.",
            ),
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("layout", type_hint=Layout, description="The layout, marked from its scan.")]

    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        if block_state.print_hand is not None:
            template = Path(block_state.template)
            block_state.layout = marked(block_state.layout, template_scan(template), classifier=block_state.print_hand)
        self.set_block_state(state, block_state)
        return components, state


class OpeningPageStep(ModularPipelineBlocks):
    """With a side, that page of a two-page opening, in its own coordinates; without one, the layout as it is."""

    @property
    @override
    def description(self) -> str:
        return "Of an opening, takes the page on the side asked for, as a page of its own."

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("template", type_hint=Path, required=True, description="The PAGE or ALTO file, for errors."),
            InputParam("layout", type_hint=Layout, required=True, description="The template's layout."),
            InputParam(
                "side",
                type_hint=PageSide,
                description="The page of a two-page opening to redraw, as a single page (diffu_page.layout.pages).",
            ),
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("layout", type_hint=Layout, description="The page's layout.")]

    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        side = block_state.side
        if side is not None:
            page = next((page for page in scan_pages(block_state.layout) if page.side == side), None)
            if page is None:
                raise ValueError(
                    f"{block_state.template} has no {side} page to draw: it is a single page, not a two-page opening, "
                    "or nothing is written on that page"
                )
            block_state.layout = page.layout
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
            InputParam(
                "text", type_hint=str, required=True, description="The running text; a blank line ends a paragraph."
            ),
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("layout", type_hint=Layout, description="The layout holding the new text.")]

    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        block_state.layout = refill(block_state.layout, block_state.text)
        self.set_block_state(state, block_state)
        return components, state


class PlanCropsStep(ModularPipelineBlocks):
    """A layout cut into the crops the models were trained on, each with its caption in the page's style, or as a
    crop drawn from the page's reference is captioned."""

    @property
    @override
    def description(self) -> str:
        return (
            "Plans the layout as training-scale crops, or whole at a pixel budget (diffu_page.plan): size, caption, "
            "line boxes, reading order."
        )

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
            InputParam(
                "page_pixels",
                type_hint=int,
                description="Plan the page whole, as one crop of about this many pixels, instead of as windows.",
            ),
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("crop_plans", type_hint=list, description="The page's CropPlans, in page order.")]

    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        layout, style, reference = block_state.layout, block_state.style, block_state.reference
        if block_state.page_pixels is None:
            block_state.crop_plans = plan_page(layout, style, reference=reference)
        else:
            block_state.crop_plans = [
                plan_whole_page(layout, style, pixels=block_state.page_pixels, reference=reference)
            ]
        self.set_block_state(state, block_state)
        return components, state
