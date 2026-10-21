"""Our Modular Diffusers blocks around FLUX.3's own: a crop plan in, a crop's ground truth out.

- ``CropPromptStep`` turns a :class:`page_synth.crops.plan.CropPlan` into the prompt, the line boxes and the image size;
- ``LineReadbackStep`` reads every line box of the generated image with a Swedish HTR TrOCR;
- ``LabelExportStep`` writes the crop's labels from the plan's boxes and words, each line tagged ``unclear`` when
  its read-back error rate is over ``max_line_cer`` (its words may not be on the page as the label says).

:mod:`page_synth.modular.pipelines` assembles them with FLUX.3's blocks.
"""

from __future__ import annotations

from typing import Literal

from diffusers.modular_pipelines import (
    ComponentSpec,
    InputParam,
    ModularPipeline,
    ModularPipelineBlocks,
    OutputParam,
    PipelineState,
)
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from page_synth.crops.plan import ROLES, CropPlan
from page_synth.labels import LineLabel, PageLabels, export
from page_synth.modular.readback import HTR, pixel_box, read_lines

type LabelFormat = Literal["alto", "page"]

SLOT_BASELINE = 0.7  # a line slot runs 0.7 line pitch above its baseline and 0.3 below (page_synth.crops.geometry)
_REGION_OF_ROLE = {role: region for region, role in reversed(ROLES.items())}  # the first region naming each role


class CropPromptStep(ModularPipelineBlocks):
    """A planned crop as the FLUX.3 inputs: its caption, its line boxes and its size."""

    model_name = "flux3"

    @property
    def description(self) -> str:
        return "Turns a page_synth.crops.plan.CropPlan into the prompt, the line boxes (page_lines) and the image size."

    @property
    def inputs(self) -> list[InputParam]:
        return [InputParam("crop_plan", type_hint=CropPlan, required=True, description="The crop to draw.")]

    @property
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
    def description(self) -> str:
        return "Crops each line's box from the generated image and reads it with TrOCR (per-line CER)."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("htr_model", VisionEncoderDecoderModel, pretrained_model_name_or_path=HTR),
            ComponentSpec("htr_processor", TrOCRProcessor, pretrained_model_name_or_path=HTR),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("images", required=True, description="The generated images (PIL)."),
            InputParam("page_lines", type_hint=list, required=True, description="The crop's LineRecords."),
        ]

    @property
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
        baseline_y = round(y0 + SLOT_BASELINE * (y1 - y0))
        labels.append(
            LineLabel(
                text=line.text,
                polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                baseline=[(x0, baseline_y), (x1, baseline_y)],
                region=_REGION_OF_ROLE.get(line.role, "body"),
                order=order_of[index],
                tag="unclear" if line_cer > max_line_cer else "",
            )
        )
    return labels


class LabelExportStep(ModularPipelineBlocks):
    """The crop's ground truth: its lines in ALTO or PAGE XML, lines that read back badly tagged ``unclear``."""

    model_name = "flux3"

    @property
    def description(self) -> str:
        return "Writes the crop's labels (ALTO or PAGE XML) from its plan, flagging lines the read-back failed."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("crop_plan", type_hint=CropPlan, required=True),
            InputParam("line_cers", type_hint=list, required=True, description="LineReadbackStep's per-line CER."),
            InputParam("label_format", type_hint=str, default="alto", description="alto | page"),
            InputParam("max_line_cer", type_hint=float, default=0.25, description="Above it a line is unclear."),
            InputParam("image_name", type_hint=str, default="", description="The image file the labels name."),
        ]

    @property
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
