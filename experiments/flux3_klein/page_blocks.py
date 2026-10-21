"""Modular Diffusers blocks that turn a planned crop into a FLUX.3 image with its ALTO/PAGE ground truth.

The generator is FLUX.3's own ``text2image`` workflow (``image_conditioned`` with a reference image) with three
blocks of ours around it::

    CropPromptStep -> [FLUX.3 text encoder, denoise, decode] -> LineReadbackStep -> LabelExportStep

- ``CropPromptStep`` turns a :class:`page_plan.CropPlan` into the prompt, the line boxes and the image size;
- ``LineReadbackStep`` (``eval_layout``) reads every line box of the generated image with a Swedish HTR TrOCR;
- ``LabelExportStep`` writes the crop's labels from the plan's boxes and words, each line flagged when its
  read-back error rate is over ``max_line_cer`` (the words may not be on the page as the label says).

The prompt upsampler is popped so the quoted lines are encoded exactly. One pipeline call draws one crop; a page
is the loop over :func:`page_plan.plan_page`'s crops (``generate_pages.py``). Runs in the NeMo AutoModel
environment (diffusers main with FLUX.3, diffu-page installed).
"""

from __future__ import annotations

from typing import Literal

import torch
from diffu_page import PageLabels, export
from diffu_page.labels import LineLabel
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.modular_pipelines import (
    Flux3AutoBlocks,
    InputParam,
    ModularPipeline,
    ModularPipelineBlocks,
    OutputParam,
    PipelineState,
)
from eval_layout import HTR, LineReadbackStep, load_trocr, pixel_box
from page_plan import ROLES, CropPlan
from transformers import TrOCRProcessor

type LabelFormat = Literal["alto", "page"]
_REGION_OF_ROLE = {
    role: region for region, role in reversed(ROLES.items())
}  # the first region naming each role
SLOT_BASELINE = 0.7  # a line slot runs 0.7 line pitch above its baseline and 0.3 below (build_layout_crops)


class CropPromptStep(ModularPipelineBlocks):
    """A planned crop as the FLUX.3 inputs: its caption, its line boxes and its size."""

    model_name = "flux3"

    @property
    def description(self) -> str:
        return "Turns a page_plan.CropPlan into the prompt, the line boxes (page_lines) and the image size."

    @property
    def inputs(self) -> list[InputParam]:
        return [InputParam("crop_plan", type_hint=CropPlan, required=True, description="The crop to draw.")]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("prompt", type_hint=str, description="The crop's layout caption."),
            OutputParam(
                "page_lines", type_hint=list, description="[{'id', 'text', 'role', 'bbox' (0-1000)}]"
            ),
            OutputParam("height", type_hint=int, description="The crop's height in pixels."),
            OutputParam("width", type_hint=int, description="The crop's width in pixels."),
        ]

    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        plan: CropPlan = block_state.crop_plan
        block_state.prompt = plan.caption
        block_state.page_lines = [line.model_dump() for line in plan.lines]
        block_state.height, block_state.width = plan.height, plan.width
        self.set_block_state(state, block_state)
        return components, state


def line_labels(plan: CropPlan, line_cers: list[float], max_line_cer: float) -> list[LineLabel]:
    """The plan's lines as labels in crop pixels: the slot as the polygon, the baseline across it, and a line
    whose read-back error rate is over ``max_line_cer`` tagged ``unclear``."""
    labels: list[LineLabel] = []
    order_of = {index: order for order, index in enumerate(plan.reading_order)}
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
        return (
            "Writes the crop's labels (ALTO or PAGE XML) from its plan, flagging lines the read-back failed."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("crop_plan", type_hint=CropPlan, required=True),
            InputParam(
                "line_cers", type_hint=list, required=True, description="LineReadbackStep's per-line CER."
            ),
            InputParam("label_format", type_hint=str, default="alto", description="alto | page"),
            InputParam(
                "max_line_cer", type_hint=float, default=0.25, description="Above it a line is unclear."
            ),
            InputParam(
                "image_name", type_hint=str, default="", description="The image file the labels name."
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("labels_xml", type_hint=str, description="The crop's labels."),
            OutputParam("unclear_lines", type_hint=int, description="Lines over max_line_cer."),
        ]

    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        plan: CropPlan = block_state.crop_plan
        lines = line_labels(plan, block_state.line_cers, block_state.max_line_cer)
        page = PageLabels(width=plan.width, height=plan.height, image=block_state.image_name, lines=lines)
        block_state.labels_xml = export(page, block_state.label_format)
        block_state.unclear_lines = sum(line.tag == "unclear" for line in lines)
        self.set_block_state(state, block_state)
        return components, state


def build_generator(
    model_path: str, *, guidance_scale: float = 4.0, with_reference: bool = False
) -> ModularPipeline:
    """The crop generator: our prompt block, FLUX.3 without its prompt upsampler, read-back, label export."""
    blocks = Flux3AutoBlocks().get_workflow("image_conditioned" if with_reference else "text2image")
    blocks.sub_blocks.pop("prompt_upsampler")
    blocks.sub_blocks.insert("crop_prompt", CropPromptStep(), 0)
    blocks.sub_blocks.insert("line_readback", LineReadbackStep(), len(blocks.sub_blocks))
    blocks.sub_blocks.insert("label_export", LabelExportStep(), len(blocks.sub_blocks))
    pipe = blocks.init_pipeline(model_path)
    pipe.load_components(
        names=[name for name in pipe.pretrained_component_names if not name.startswith("htr_")],
        dtype={"vae": torch.float32, "default": torch.bfloat16},
    )
    pipe.update_components(htr_model=load_trocr(), htr_processor=TrOCRProcessor.from_pretrained(HTR))
    pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=guidance_scale))
    pipe.to("cuda")
    return pipe
