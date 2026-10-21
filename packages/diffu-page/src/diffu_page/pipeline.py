"""The page pipeline: our blocks around a model's own Modular Diffusers blocks.

A model repo (FLUX.3 Klein for the reference runs; diffu, FLUX.3's architecture with our own weights) names its blocks
in ``modular_model_index.json``; :data:`FAMILIES` maps those to the model's blocks and pipeline class, so a model is
chosen by its directory alone. Three sets of blocks are built:

- :class:`PagePlanBlocks` (no model): ``LoadLayoutStep`` -> ``RefillStep`` -> ``PlanCropsStep``. A template page, a
  text and a style in; the crops to draw out.
- :func:`crop_blocks`: ``CropPromptStep`` -> ``ReferenceGridStep`` -> the model's text encoder, VAE encoder, denoise
  and decode -> ``LineReadbackStep`` -> ``LabelExportStep``. A crop plan in; its image, per-line read-back and labels
  out. The model's auto blocks pick text-to-image or reference-conditioned denoising from the inputs (``image``).
- :func:`page_blocks`: the plan steps, then ``CropLoopStep`` running the crop blocks once per planned crop.

``diffu-generate`` and diffu-studio run the page pipeline; ``diffu-evaluate`` and diffu-runner's GPU actors run the
crop pipeline, after diffu-runner has planned the pages on CPU workers with :class:`PagePlanBlocks`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import override

import torch
from diffusers import Flux3AutoBlocks, Flux3ModularPipeline
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.modular_pipelines import ModularPipeline, OutputParam, SequentialPipelineBlocks
from PIL import Image
from pydantic import BaseModel, ConfigDict

from diffu.modular_pipelines.diffu import DiffuAutoBlocks, DiffuModularPipeline
from diffu_page.blocks import (
    CROP_OUTPUTS,
    CropLoopStep,
    CropPromptStep,
    DrawnCrop,
    LabelExportStep,
    LineReadbackStep,
    LoadLayoutStep,
    PlanCropsStep,
    ReferenceGridStep,
    RefillStep,
)
from diffu_page.labels import LabelFormat
from diffu_page.lora import load_nemo_lora
from diffu_page.plan import CropPlan
from diffu_page.readback import materialize_position_tables


class ModelFamily(BaseModel):
    """A model family's own blocks and pipeline class, and the blocks of its that generation leaves out."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    blocks: type[SequentialPipelineBlocks]
    pipeline: type[ModularPipeline]
    unused: tuple[str, ...] = ()


FAMILIES: dict[str, ModelFamily] = {
    # FLUX.3's prompt upsampler rewrites a prompt as prose: it would rewrite the quoted lines of a layout caption
    "Flux3AutoBlocks": ModelFamily(blocks=Flux3AutoBlocks, pipeline=Flux3ModularPipeline, unused=("prompt_upsampler",)),
    "DiffuAutoBlocks": ModelFamily(blocks=DiffuAutoBlocks, pipeline=DiffuModularPipeline),
}


def model_family(model: Path) -> ModelFamily:
    """The family of the Modular Diffusers repo at ``model``, by the blocks its ``modular_model_index.json`` names.

    Raises:
        ValueError: If no family has those blocks.
    """
    index = json.loads((model / "modular_model_index.json").read_text(encoding="utf-8"))
    blocks = index["_blocks_class_name"]
    if blocks not in FAMILIES:
        raise ValueError(f"{model} names the blocks {blocks!r}; the known model families are {sorted(FAMILIES)}")
    return FAMILIES[blocks]


class CropBlocks(SequentialPipelineBlocks):
    """A crop plan in; the drawn crop, every line of it read back, and its labels out (built by :func:`crop_blocks`)."""

    _workflow_map = {
        "text2image": {"crop_plan": True},
        "image_conditioned": {"crop_plan": True, "image": True},
    }

    @property
    @override
    def description(self) -> str:
        return (
            "Draws a planned crop with a model's own blocks (reference-conditioned when an `image` is given), reads "
            "every line back with TrOCR and writes the crop's ALTO/PAGE labels."
        )

    @property
    @override
    def outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("images"),
            OutputParam("line_reads", type_hint=list, description="What the recognizer read in each line box."),
            OutputParam("line_cers", type_hint=list, description="Per-line CER against the line's text."),
            OutputParam("labels_xml", type_hint=str, description="The crop's labels."),
            OutputParam("unclear_lines", type_hint=int, description="Lines over max_line_cer."),
        ]


def crop_blocks(family: ModelFamily) -> CropBlocks:
    """Our crop steps around ``family``'s own blocks: the crop prompt and the reference grid before them, the read-back
    and the labels after."""
    model = family.blocks().sub_blocks
    for name in family.unused:
        model.pop(name)
    return CropBlocks.from_blocks_dict(
        {
            "crop_prompt": CropPromptStep(),
            "reference_grid": ReferenceGridStep(),
            **model,
            "line_readback": LineReadbackStep(),
            "label_export": LabelExportStep(),
        }
    )


class PagePlanBlocks(SequentialPipelineBlocks):
    """A template page, a text and a style in; the crops to draw out. No model: CPU only."""

    block_classes = [LoadLayoutStep, RefillStep, PlanCropsStep]
    block_names = ["load_layout", "refill", "plan_crops"]

    @property
    @override
    def description(self) -> str:
        return "Reads a PAGE/ALTO template, writes the text into its running-text lines and plans it as crops."

    @property
    @override
    def outputs(self) -> list[OutputParam]:
        return [OutputParam("crop_plans", type_hint=list, description="The page's CropPlans, in page order.")]


class PageBlocks(SequentialPipelineBlocks):
    """A template page, a text and a style in; every crop of it drawn, read back and labelled out (built by
    :func:`page_blocks`)."""

    _workflow_map = {
        "text2image": {"template": True, "text": True, "style": True},
        "image_conditioned": {"template": True, "text": True, "style": True, "image": True},
    }

    @property
    @override
    def description(self) -> str:
        return (
            "Plans a PAGE/ALTO template refilled with new text as crops, then draws, reads back and labels every crop "
            "with a model's own blocks."
        )

    @property
    @override
    def outputs(self) -> list[OutputParam]:
        return [OutputParam("crops", type_hint=list, description="Every crop drawn (DrawnCrop), in plan order.")]


def page_blocks(family: ModelFamily) -> PageBlocks:
    """The plan steps, then ``family``'s crop blocks once per planned crop."""
    draw_crops = CropLoopStep.from_blocks_dict({"draw_crop": crop_blocks(family)})
    return PageBlocks.from_blocks_dict({**PagePlanBlocks().sub_blocks, "draw_crops": draw_crops})


class DrawSettings(BaseModel):
    """How every crop is drawn.

    Attributes:
        steps: Denoising steps per crop.
        guidance_scale: Classifier-free guidance scale; 1 or less samples without guidance.
        label_format: ``alto`` or ``page``.
        text_layers: The text-encoder hidden layers stacked into the prompt embeddings; ``None`` is the model's own
            choice (FLUX.3: every 4th of Qwen3-VL's).
        device: Where the components run.
    """

    model_config = ConfigDict(frozen=True)

    steps: int = 50
    guidance_scale: float = 4.0
    label_format: LabelFormat = "alto"
    text_layers: tuple[int, ...] | None = None
    device: str = "cuda"

    def pipeline_inputs(self) -> dict[str, object]:
        """The settings a pipeline takes per call, by input name (the guidance scale and device are set at load)."""
        layers = {} if self.text_layers is None else {"text_encoder_out_layers": self.text_layers}
        return {"num_inference_steps": self.steps, "label_format": self.label_format, **layers}


type BlocksFor = Callable[[ModelFamily], SequentialPipelineBlocks]


def _require_loaded(pipe: ModularPipeline) -> None:
    """Fail fast: ``load_components`` logs a component that failed to load and leaves it ``None``."""
    missing = [name for name in pipe.pretrained_component_names if getattr(pipe, name, None) is None]
    if missing:
        raise RuntimeError(f"components failed to load (see the log above): {missing}")


def load_pipeline(model: Path, blocks: BlocksFor, *, loras: dict[str, Path], settings: DrawSettings) -> ModularPipeline:
    """``blocks`` built for the model at ``model``, its components loaded (VAE in float32, the rest in bfloat16) on
    ``settings.device``, the guider at ``settings.guidance_scale`` and every LoRA of ``loras`` (NeMo ``model/`` dirs by
    adapter name) active.

    Raises:
        ValueError: If the model's blocks have no family.
        RuntimeError: If a component failed to load.
    """
    family = model_family(model)
    pipe = family.pipeline(blocks=blocks(family), pretrained_model_name_or_path=str(model))
    pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
    _require_loaded(pipe)
    pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=settings.guidance_scale))
    pipe.to(settings.device)
    materialize_position_tables(pipe.htr_model)
    for name, lora_dir in loras.items():
        load_nemo_lora(pipe, lora_dir, name)
    if loras:
        pipe.set_adapters(list(loras))
    return pipe


def draw_crop(
    pipe: ModularPipeline,
    plan: CropPlan,
    *,
    name: str,
    seed: int,
    reference: Image.Image | None,
    settings: DrawSettings,
) -> DrawnCrop:
    """One crop drawn by a crop pipeline (:func:`crop_blocks`) from noise seed ``seed``, conditioned on ``reference``
    when there is one."""
    reference_input = {} if reference is None else {"image": reference}
    outputs = pipe(
        crop_plan=plan,
        generator=torch.Generator().manual_seed(seed),
        image_name=f"{name}.jpg",
        **settings.pipeline_inputs(),
        **reference_input,
        output=CROP_OUTPUTS,
    )
    return DrawnCrop.from_outputs(name, plan, outputs)
