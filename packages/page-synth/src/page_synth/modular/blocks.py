"""Our Modular Diffusers blocks around FLUX.3's own: a crop plan in, a crop's ground truth out.

- ``CropPromptStep`` turns a :class:`page_synth.plan.CropPlan` into the prompt, the line boxes and the image size;
- ``ReferenceGridStep`` resizes a reference image the way it was resized for v3 training;
- ``LineReadbackStep`` reads every line box of the generated image with a Swedish HTR TrOCR;
- ``LabelExportStep`` writes the crop's labels from the plan's boxes and words, each line tagged ``unclear`` when
  its read-back error rate is over ``max_line_cer`` (its words may not be on the page as the label says).

:mod:`page_synth.modular.pipelines` assembles them with FLUX.3's blocks.
"""

from __future__ import annotations

from typing import override

from diffusers.modular_pipelines import (
    ComponentSpec,
    InputParam,
    ModularPipeline,
    ModularPipelineBlocks,
    OutputParam,
    PipelineState,
)
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from page_synth.crops.geometry import SLOT_ABOVE_BASELINE, SLOT_BELOW_BASELINE
from page_synth.crops.grid import LARGEST_BUCKET_PIXELS, fit_to_grid
from page_synth.labels import LabelFormat, LineLabel, PageLabels, export
from page_synth.layout import RUNNING_TEXT
from page_synth.modular.readback import HTR, pixel_box, read_lines
from page_synth.plan import CropPlan


def training_grid(image: Image.Image) -> Image.Image:
    """A reference image as NeMo's FLUX.3 cache encoder resized it for v3 training (see ``fit_to_grid``)."""
    return fit_to_grid(image, LARGEST_BUCKET_PIXELS)


class ReferenceGridStep(ModularPipelineBlocks):
    """Puts a reference image on the grid the v3 LoRA was trained on.

    NeMo's cache encoder squashed every training reference onto the 16-pixel grid, while FLUX.3's own preprocessing
    pads a reference out to it with a reflected border. Resizing first makes FLUX.3's preprocessing a no-op, so a
    reference is drawn from exactly as in training. Without a reference the step does nothing.
    """

    model_name = "flux3"

    @property
    @override
    def description(self) -> str:
        return "Resizes the reference image onto the 16-px grid the way NeMo's FLUX.3 cache encoder did in training."

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
            block_state.image = training_grid(block_state.image)
        self.set_block_state(state, block_state)
        return components, state


class CropPromptStep(ModularPipelineBlocks):
    """A planned crop as the FLUX.3 inputs: its caption, its line boxes and its size."""

    model_name = "flux3"

    @property
    @override
    def description(self) -> str:
        return "Turns a page_synth.plan.CropPlan into the prompt, the line boxes (page_lines) and the image size."

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


class LineReadbackStep(ModularPipelineBlocks):
    """Reads every line box of the generated image with an HTR model: the per-line CER against its text."""

    model_name = "flux3"

    @property
    @override
    def description(self) -> str:
        return "Crops each line's box from the generated image and reads it with TrOCR (per-line CER)."

    @property
    @override
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("htr_model", VisionEncoderDecoderModel, pretrained_model_name_or_path=HTR),
            ComponentSpec("htr_processor", TrOCRProcessor, pretrained_model_name_or_path=HTR),
        ]

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("images", required=True, description="The generated images (PIL)."),
            InputParam("page_lines", type_hint=list, required=True, description="The crop's LineRecords."),
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("line_reads", type_hint=list, description="What the recognizer read in each line box."),
            OutputParam("line_cers", type_hint=list, description="Per-line CER against the line's text."),
        ]

    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        block_state.line_reads, block_state.line_cers = read_lines(
            block_state.images[0],
            block_state.page_lines,
            model=components.htr_model,
            processor=components.htr_processor,
        )
        self.set_block_state(state, block_state)
        return components, state


def line_labels(plan: CropPlan, line_cers: list[float], max_line_cer: float) -> list[LineLabel]:
    """The plan's lines as labels in crop pixels: the slot as the polygon, the baseline across it, and a line
    whose read-back error rate is over ``max_line_cer`` tagged ``unclear``."""
    order_of = {index: order for order, index in enumerate(plan.reading_order)}
    labels: list[LineLabel] = []
    for index, (line, line_cer) in enumerate(zip(plan.lines, line_cers, strict=True)):
        x0, y0, x1, y1 = (round(v) for v in pixel_box(line.bbox, (plan.width, plan.height)))
        baseline_y = round(y0 + SLOT_ABOVE_BASELINE / (SLOT_ABOVE_BASELINE + SLOT_BELOW_BASELINE) * (y1 - y0))
        labels.append(
            LineLabel(
                text=line.text,
                polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                baseline=[(x0, baseline_y), (x1, baseline_y)],
                region=line.role or RUNNING_TEXT,
                order=order_of[index],
                tag="unclear" if line_cer > max_line_cer else "",
            )
        )
    return labels


class LabelExportStep(ModularPipelineBlocks):
    """The crop's ground truth: its lines in ALTO or PAGE XML, lines that read back badly tagged ``unclear``."""

    model_name = "flux3"

    @property
    @override
    def description(self) -> str:
        return "Writes the crop's labels (ALTO or PAGE XML) from its plan, flagging lines the read-back failed."

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("crop_plan", type_hint=CropPlan, required=True),
            InputParam("line_cers", type_hint=list, required=True, description="LineReadbackStep's per-line CER."),
            InputParam("label_format", type_hint=LabelFormat, default="alto", description="alto | page"),
            InputParam("max_line_cer", type_hint=float, default=0.25, description="Above it a line is unclear."),
            InputParam("image_name", type_hint=str, default="", description="The image file the labels name."),
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("labels_xml", type_hint=str, description="The crop's labels."),
            OutputParam("unclear_lines", type_hint=int, description="Lines over max_line_cer."),
        ]

    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        plan: CropPlan = block_state.crop_plan
        lines = line_labels(plan, block_state.line_cers, block_state.max_line_cer)
        page = PageLabels(width=plan.width, height=plan.height, image=block_state.image_name, lines=lines)
        block_state.labels_xml = export(page, block_state.label_format)
        block_state.unclear_lines = sum(line.tag == "unclear" for line in lines)
        self.set_block_state(state, block_state)
        return components, state
