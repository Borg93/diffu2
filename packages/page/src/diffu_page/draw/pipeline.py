"""The page pipeline: our blocks around MOSS-Image's own Modular Diffusers blocks.

A model directory is MOSS-Image's (``diffu-moss-init``'s): its ``modular_model_index.json`` names MOSS-Image's blocks,
and :func:`require_moss` refuses any other. Two sets of blocks are built:

- :class:`PagePlanBlocks` (no model): ``LoadLayoutStep`` -> ``MarkFromScanStep`` -> ``OpeningPageStep`` ->
  ``RefillStep`` -> ``PlanCropsStep``. A template page (or one page of a two-page opening) or a layout given in its
  place, a text (or none: the lines keep their own), a style and optionally a reference crop and a print/hand
  classifier (to mark the template from its scan) in; the crops to draw out, captioned for the reference if there is
  one.
- :func:`crop_blocks`: ``CropPromptStep`` -> ``ReferenceGridStep`` -> MOSS-Image's Layout prompt, text encoder,
  reference encoder, denoise and decode. A crop plan in; its image out. A line repair sits in the denoise step
  (:mod:`diffu_page.draw.blocks.repair`). MOSS-Image compiles the plan's caption and lines into its own Layout prompt
  first, and takes a reference (``image``) only when its DiT was trained to (``takes_reference`` in its model index:
  ``diffu_moss.train``).

A page is planned with :class:`PagePlanBlocks`; each of its crops is drawn on its own with the crop pipeline, judged
line by line (:mod:`diffu_page.quality.judge`) and labelled (:func:`diffu_page.draw.blocks.ground_truth.crop_labels`).
Draw, judge and labels are separate stages, so each can run as its own worker; :func:`draw_crop` runs the three in one
process, and :func:`repair_lines` draws a crop's rejected lines again in place.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Self, override

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.modular_pipelines import ModularPipeline, OutputParam, SequentialPipelineBlocks
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from diffu_core.loading import COMPONENT_DTYPES, require_loaded
from diffu_core.repo import ModelIndex
from diffu_moss.adapters import fuse_adapters, load_adapters, read_adapter
from diffu_moss.inference import MossInference, check_inference, prepare_inference
from diffu_moss.latents import encode_latents
from diffu_moss.modular_pipelines.moss import MossAutoBlocks, MossModularPipeline
from diffu_moss.modular_pipelines.moss.before_denoise import Sampler
from diffu_moss.modular_pipelines.moss.denoise import GUIDANCE_SCALE as MOSS_GUIDANCE_SCALE
from diffu_moss.mxfp8 import Precision
from diffu_moss.step_cache import StepCache
from diffu_page.draw.blocks import (
    CropPromptStep,
    DrawnCrop,
    LoadLayoutStep,
    MarkFromScanStep,
    OpeningPageStep,
    PlanCropsStep,
    ReferenceGridStep,
    RefillStep,
)
from diffu_page.draw.blocks.ground_truth import crop_labels
from diffu_page.draw.blocks.repair import RepairBlendStep, RepairPrepareStep
from diffu_page.draw.guidance import ReferenceGuidance
from diffu_page.draw.plan import CropPlan
from diffu_page.page.crops.geometry import LayoutBox, pixel_box
from diffu_page.page.crops.grid import LATENT_GRID
from diffu_page.page.labels import LabelFormat
from diffu_page.quality.judge import MAX_LINE_CER, Judge, default_judge

# Errors after which the device cannot be trusted: a run stops there instead of failing every later crop, and is not
# retried into the same out-of-memory (which once froze the DGX Spark's unified memory)
DEVICE_ERRORS = (torch.OutOfMemoryError, torch.AcceleratorError)
REPAIR_SEED_STRIDE = 7919  # between a crop's seed and each repair pass's, so no pass redraws from the crop's own noise


def require_moss(model: Path) -> None:
    """Check that the Modular Diffusers repo at ``model`` is MOSS-Image's, by the blocks its
    ``modular_model_index.json`` names.

    Raises:
        ValueError: If the directory holds no model index, or its index names other blocks.
    """
    blocks = ModelIndex.read(model).blocks_class_name
    if blocks != MossAutoBlocks.__name__:
        raise ValueError(
            f"{model} is no MOSS-Image model directory (diffu-moss-init's): its index names the blocks {blocks!r}"
        )


class CropBlocks(SequentialPipelineBlocks):
    """A crop plan in; the drawn crop out (built by :func:`crop_blocks`)."""

    _workflow_map = {
        "text2image": {"crop_plan": True},
        "image_conditioned": {"crop_plan": True, "image": True},
    }

    @property
    @override
    def description(self) -> str:
        return "Draws a planned crop with MOSS-Image's own blocks (reference-conditioned when an `image` is given)."

    @property
    @override
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images")]


def crop_blocks() -> CropBlocks:
    """Our crop steps before MOSS-Image's own blocks (the crop prompt and the reference grid), and a line repair in its
    denoise step (:mod:`diffu_page.draw.blocks.repair`), idle unless a repair mask is given.

    Raises:
        TypeError: If a denoise workflow of MOSS-Image's has no scheduler step (``after_denoiser``) in its denoise loop.
    """
    model = MossAutoBlocks().sub_blocks
    denoise = copy.deepcopy(model["denoise"])  # diffusers shares its blocks' instances process-wide: change a copy
    for branch in denoise.sub_blocks.values():
        steps = branch.sub_blocks
        loop = steps["denoise"].sub_blocks
        if "after_denoiser" not in loop:
            raise TypeError(f"{type(branch).__name__}'s denoise loop has no scheduler step to blend after")
        steps.insert("repair_prepare", RepairPrepareStep(), list(steps).index("prepare_latents") + 1)
        loop.insert("repair_blend", RepairBlendStep(), list(loop).index("after_denoiser") + 1)
    model["denoise"] = denoise
    return CropBlocks.from_blocks_dict(
        {
            "crop_prompt": CropPromptStep(),
            "reference_grid": ReferenceGridStep(),
            **model,
        }
    )


class PagePlanBlocks(SequentialPipelineBlocks):
    """A template page (or one page of a two-page opening) or a layout, a text (or none), a style and optionally a
    reference crop and a print/hand classifier in; the crops to draw out. No model but the small CNN that marks a
    template: CPU only."""

    block_classes = [LoadLayoutStep, MarkFromScanStep, OpeningPageStep, RefillStep, PlanCropsStep]
    block_names = ["load_layout", "mark_from_scan", "opening_page", "refill", "plan_crops"]

    @property
    @override
    def description(self) -> str:
        return (
            "Reads a PAGE/ALTO template (marked from its scan if asked) or takes a layout, writes the text, if given, "
            "into its running-text lines and plans it as crops."
        )

    @property
    @override
    def outputs(self) -> list[OutputParam]:
        return [OutputParam("crop_plans", type_hint=list, description="The page's CropPlans, in page order.")]


# A page's guidance as measured on the sample-above model against its sample pages (docs/MOSS_RUNS.md, 20-21 Nov
# 2026): at MOSS-Image's own 3 the ink is too dark and the strokes too thick, at 2 they are close to the sample's and
# every line is still written; guidance over the first half of the steps draws the same page a quarter sooner. None
# takes MOSS-Image's own scale.
PAGE_GUIDANCE_SCALE = 2.0
PAGE_GUIDANCE_STOP = 0.5


class DrawSettings(BaseModel):
    """How every crop is drawn.

    Attributes:
        steps: Denoising steps per crop.
        guidance_scale: Classifier-free guidance scale; exactly 1 samples without guidance (one pass a step). ``None``
            is MOSS-Image's own (3, its demo's).
        reference_guidance_scale: How hard guidance also pushes toward the reference image
            (:mod:`diffu_page.draw.guidance`, a third pass per step); 1 guides the caption alone.
        guidance_stop: The fraction of the steps after which guidance stops: one pass a step after it; 1 guides every
            step.
        label_format: ``alto`` or ``page``.
        max_line_cer: The read-back CER above which a line's label is tagged ``unclear``.
        device: Where the components run.
        sampling_shift: The timestep shift the scheduler samples at; ``None`` is MOSS-Image's own (5, its
            demo's).
        lora_scale: How strongly every LoRA applies: 1 at its trained strength, 0 not at all (diffusers'
            ``set_adapters`` weights, which fusing keeps).
        fuse_loras: Fuse the LoRAs into the transformer's weights at load and drop their adapter layers: no LoRA
            matmuls per step, and the transformer can be compiled, quantized or run from a package; the LoRAs can't be switched or rescaled after.
        compile_blocks: Compile the DiT's joint blocks (torch.compile) at load; the first crop pays for the
            compilation. A run with LoRAs must fuse them first. The blocks are compiled after any MXFP8 swap, so the quantization of their inputs fuses into them. With a
            step cache (``cache``) each block's own forward is compiled and the cache's hooks decide eagerly around it.
        precision: The DiT's joint-block projections in their own bfloat16, or in MXFP8: weights and
            inputs in blocks of 32 FP8 values sharing a power-of-two scale, multiplied by a Blackwell GPU's tensor cores
            (:mod:`diffu_moss.mxfp8`; emulated on the CPU). It pays off with ``compile_blocks`` or ``aoti_package``,
            which fuse each layer's quantization of its input: run eagerly, that is some fifteen small kernels. A run
            with LoRAs must fuse them first.
        text_encoder_precision: The text encoder's decoder-layer projections alike (half their
            memory).
        aoti_package: Run the DiT's joint blocks from this AOTInductor package (``diffu-moss-aoti``), built
            for ``precision``, this GPU's architecture and this torch version (any other is refused); instead of
            ``compile_blocks``. A run with LoRAs must fuse them first.
        sampler: What steps the sampling grid: ``euler``, MOSS-Image's own sampler, or ``unipc``, diffusers'
            multistep UniPC solver, which draws as well in fewer ``steps``. Either calls the DiT once a pass a step, at
            the grid's levels.
        cache: A step cache that skips the DiT's joint blocks at some steps
            (:mod:`diffu_moss.step_cache`), by ``kind``: ``{"kind": "first_block", "threshold": 0.05}``,
            ``{"kind": "taylorseer", "cache_interval": 5, "disable_cache_before_step": 3}``, ``{"kind": "magcache",
            "mag_ratios": [...]}``, or ``{"kind": "magcache_calibration", "ratios_file": ...}``, which skips nothing
            and measures MagCache's ratios for this sampler and these ``steps``. The first-block cache also takes
            ``warmup_steps`` (a pass's first steps always computed) and ``max_skipped_in_a_row``.
        repair_passes: How many times a drawn crop's rejected lines are drawn again in place (:func:`repair_lines`),
            each pass from its own noise (:func:`repair_seed`); 0 keeps every crop as drawn.
    """

    model_config = ConfigDict(frozen=True)

    steps: int = Field(default=50, ge=1)
    guidance_scale: float | None = PAGE_GUIDANCE_SCALE
    reference_guidance_scale: float = 1.0
    guidance_stop: float = Field(default=PAGE_GUIDANCE_STOP, ge=0.0, le=1.0)
    label_format: LabelFormat = LabelFormat.ALTO
    max_line_cer: float = MAX_LINE_CER
    device: str = "cuda"
    sampling_shift: float | None = Field(default=None, gt=0.0)
    lora_scale: float = Field(default=1.0, ge=0.0)
    fuse_loras: bool = False
    compile_blocks: bool = False
    precision: Precision = Precision.BF16
    text_encoder_precision: Precision = Precision.BF16
    aoti_package: Path | None = None
    sampler: Sampler = Sampler.EULER
    cache: StepCache | None = None
    repair_passes: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _moss_inference_fits(self) -> Self:
        _ = self.moss_inference  # MossInference validates itself: a package refuses compile_blocks
        return self

    @property
    def moss_inference(self) -> MossInference:
        """How the model computes (:mod:`diffu_moss.inference`)."""
        return MossInference(
            precision=self.precision,
            text_encoder_precision=self.text_encoder_precision,
            compile_blocks=self.compile_blocks,
            aoti_package=self.aoti_package,
            sampler=self.sampler,
            cache=self.cache,
        )

    @property
    def needs_fused_loras(self) -> list[str]:
        """The settings asked that rework the transformer's layers once its LoRAs are loaded, by name: its blocks
        compiled, its joint-block projections in MXFP8, its joint blocks run from a package. Each would leave
        unfused adapter layers out of what it builds, or refuse them."""
        asked = {
            "compile_blocks": self.compile_blocks,
            "precision": self.precision is not Precision.BF16,
            "aoti_package": self.aoti_package is not None,
        }
        return [name for name, is_asked in asked.items() if is_asked]

    @property
    def guides_reference(self) -> bool:
        """Whether guidance also pushes toward the reference image (a third pass a step)."""
        return not math.isclose(self.reference_guidance_scale, 1.0)

    def pipeline_inputs(self) -> dict[str, object]:
        """The settings a pipeline takes per call, by input name (the guidance scale and device are set at load; the
        label format and the unclear limit apply after the judge)."""
        return {"num_inference_steps": self.steps}


type BuildBlocks = Callable[[], SequentialPipelineBlocks]


def load_pipeline(
    model: Path, blocks: BuildBlocks, *, loras: dict[str, Path], settings: DrawSettings
) -> ModularPipeline:
    """``blocks`` built for the MOSS-Image model at ``model``, its components loaded (VAE in float32, the rest in
    bfloat16) on ``settings.device``, the guider at ``settings.guidance_scale`` (MOSS-Image's own when None, and
    ``settings.reference_guidance_scale``) until ``settings.guidance_stop``, and every LoRA of ``loras`` (PEFT adapter
    folders of its DiT by adapter name, ``diffu-moss-convert-lora``'s) active at ``settings.lora_scale``, fused into
    the DiT's weights when ``settings.fuse_loras`` (:mod:`diffu_moss.adapters`). The model computes as
    ``settings.moss_inference`` asks (:func:`diffu_moss.inference.prepare_inference`: its layers in MXFP8, UniPC, its
    joint blocks from an AOTInductor package or compiled, a step cache around them, for pages of ``settings.steps``),
    its LoRAs loaded first.

    Everything that can be checked without the model's weights is checked first: its index, the guider, every LoRA as
    read, and the precision and package against the device.

    Raises:
        ValueError: If the directory is not MOSS-Image's, the guidance settings do not go together, a LoRA is no
            adapter of the DiT or does not fit it, the transformer's layers are to be compiled, quantized or packaged
            around LoRAs that are not fused, or the precision or package does not fit the device, this process or the
            model.
        RuntimeError: If a component failed to load.
    """
    reworked = settings.needs_fused_loras
    if loras and reworked and not settings.fuse_loras:
        raise ValueError(
            f"{', '.join(reworked)} only with the LoRAs fused (fuse_loras): they rework the transformer's layers, and "
            "unfused adapter layers would stand in the way"
        )
    require_moss(model)
    check_inference(settings.moss_inference, torch.device(settings.device))
    guidance_scale = MOSS_GUIDANCE_SCALE if settings.guidance_scale is None else settings.guidance_scale
    guider = ReferenceGuidance(
        guidance_scale=guidance_scale,
        reference_scale=settings.reference_guidance_scale,
        stop=settings.guidance_stop,
    )
    adapters = {name: read_adapter(lora_dir) for name, lora_dir in loras.items()}
    pipe = MossModularPipeline(blocks=blocks(), pretrained_model_name_or_path=str(model))
    pipe.load_components(dtype=COMPONENT_DTYPES)
    require_loaded(pipe)
    pipe.update_components(guider=guider)
    if settings.sampling_shift is not None:
        pipe.update_components(
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift=settings.sampling_shift)
        )
    pipe.to(settings.device)
    load_adapters(pipe.transformer, adapters, scale=settings.lora_scale)
    if adapters and settings.fuse_loras:
        fuse_adapters(pipe.transformer)
    prepare_inference(pipe, settings.moss_inference, steps=settings.steps)
    return pipe


def draw_crop(
    pipe: ModularPipeline,
    plan: CropPlan,
    *,
    name: str,
    seed: int,
    reference: Image.Image | None,
    settings: DrawSettings,
    judge: Judge | None = None,
    reference_lines: Sequence[LayoutBox] = (),
) -> DrawnCrop:
    """One crop drawn by a crop pipeline (:func:`crop_blocks`) from noise seed ``seed``, conditioned on ``reference``
    (its line boxes ``reference_lines``, if known: a model trained on legible samples cuts a window's sample by them)
    when there is one, every line judged by ``judge`` (the default TrOCR judge on ``settings.device`` if none), and
    labelled with the lines the judge could not confirm tagged ``unclear``.

    Raises:
        ValueError: If ``settings`` guide toward a reference and there is none, before anything is drawn; from drawing,
            if the model reads a window's sample cut from its line boxes and none are given.
    """
    image = _draw(pipe, plan, seed=seed, reference=reference, settings=settings, reference_lines=reference_lines)
    verdicts = (judge or default_judge(settings.device))(image, plan.lines)
    labels = crop_labels(plan, verdicts, max_line_cer=settings.max_line_cer, image_name=f"{name}.jpg")
    return DrawnCrop(
        name=name, plan=plan, image=image, verdicts=verdicts, labels=labels, label_format=settings.label_format
    )


def repair_seed(seed: int, repair_pass: int) -> int:
    """The noise seed of a crop's repair pass ``repair_pass`` (1, 2, ...), given the seed that drew the crop."""
    return seed + repair_pass * REPAIR_SEED_STRIDE


def repair_lines(
    pipe: ModularPipeline,
    crop: DrawnCrop,
    *,
    seed: int,
    reference: Image.Image | None,
    settings: DrawSettings,
    judge: Judge | None = None,
    reference_lines: Sequence[LayoutBox] = (),
) -> DrawnCrop:
    """``crop`` with the lines its verdicts reject drawn again in place (:mod:`diffu_page.draw.blocks.repair`) from
    noise seed ``seed``, which should not be the seed that drew them, from ``reference`` as :func:`draw_crop` draws.
    Each line is redrawn over the latent cells its box touches, and a redrawn line replaces those cells of the crop
    only if the judge passes it, so every other pixel stays the crop's; the labels then follow the judge's verdicts on
    the result, and a line still rejected stays ``unclear``.

    Raises:
        ValueError: If ``settings`` guide toward a reference and there is none, before anything is drawn.
    """
    rejected = [k for k, verdict in enumerate(crop.verdicts) if not verdict.passes(settings.max_line_cer)]
    if not rejected:
        return crop
    judge = judge or default_judge(settings.device)
    lines = [crop.plan.lines[k] for k in rejected]
    cells = [_latent_cells(line.bbox, crop.image.size) for line in lines]
    redrawn = _draw(
        pipe,
        crop.plan,
        seed=seed,
        reference=reference,
        settings=settings,
        reference_lines=reference_lines,
        repair_latents=page_latents(pipe, crop.image),
        repair_mask=_latent_mask(cells, crop.image.size),
    )
    image = crop.image.copy()
    kept = 0
    for area, verdict in zip(cells, judge(redrawn, lines), strict=True):
        if verdict.passes(settings.max_line_cer):
            image.paste(redrawn.crop(area), area[:2])  # the redraw meets the kept latents at the cells' edges
            kept += 1
    if not kept:
        return crop
    verdicts = judge(image, crop.plan.lines)
    labels = crop_labels(crop.plan, verdicts, max_line_cer=settings.max_line_cer, image_name=f"{crop.name}.jpg")
    return DrawnCrop(
        name=crop.name,
        plan=crop.plan,
        image=image,
        verdicts=verdicts,
        labels=labels,
        label_format=settings.label_format,
    )


def page_latents(pipe: ModularPipeline, image: Image.Image) -> torch.Tensor:
    """``image`` as the model's latents ``(1, C, height / 16, width / 16)``, encoded as MOSS-Image's reference encoder
    and its decoder do (:func:`diffu_moss.latents.encode_latents`), without the reference preprocessing (a drawn crop
    is already on the latent grid)."""
    processor = pipe.image_processor
    pixels = processor.normalize(processor.numpy_to_pt(processor.pil_to_numpy(image.convert("RGB"))))
    return encode_latents(pipe.vae, pixels)


def _draw(
    pipe: ModularPipeline,
    plan: CropPlan,
    *,
    seed: int,
    reference: Image.Image | None,
    settings: DrawSettings,
    reference_lines: Sequence[LayoutBox] = (),
    **repair: torch.Tensor,
) -> Image.Image:
    """``plan`` drawn by a crop pipeline from noise seed ``seed``, conditioned on ``reference`` (and its line boxes
    ``reference_lines``, when given) when there is one; with ``repair`` (``repair_latents`` and ``repair_mask``), drawn
    again only inside the mask."""
    if reference is None and settings.guides_reference:
        raise ValueError("reference guidance needs a reference image: draw this crop at reference_guidance_scale 1")
    reference_input: dict[str, object] = {}
    if reference is not None:
        reference_input["image"] = reference
        if reference_lines:
            reference_input["reference_lines"] = list(reference_lines)
    # The text encoder step encodes the negative caption only if the guider guides at its current step, which until the
    # denoise loop starts is the previous crop's last: past guidance_stop, this crop would lose its negative caption.
    pipe.guider.set_state(step=0, num_inference_steps=settings.steps, timestep=None)
    return pipe(
        crop_plan=plan,
        generator=torch.Generator().manual_seed(seed),
        **settings.pipeline_inputs(),
        **reference_input,
        **repair,
        output="images",
    )[0]


def _latent_cells(bbox: Sequence[int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """The pixels ``(x0, y0, x1, y1)`` of the whole latent cells (16 px) that a layout box touches in an image of
    ``size``."""
    x0, y0, x1, y1 = pixel_box(bbox, size)
    width, height = size
    return (
        math.floor(x0 / LATENT_GRID) * LATENT_GRID,
        math.floor(y0 / LATENT_GRID) * LATENT_GRID,
        min(math.ceil(x1 / LATENT_GRID) * LATENT_GRID, width),
        min(math.ceil(y1 / LATENT_GRID) * LATENT_GRID, height),
    )


def _latent_mask(cells: Sequence[tuple[int, int, int, int]], size: tuple[int, int]) -> torch.Tensor:
    """``cells`` (pixels on the latent grid) as a mask ``(1, 1, height / 16, width / 16)``: 1 inside, 0 elsewhere."""
    width, height = size
    mask = torch.zeros(1, 1, height // LATENT_GRID, width // LATENT_GRID)
    for x0, y0, x1, y1 in cells:
        mask[:, :, y0 // LATENT_GRID : y1 // LATENT_GRID, x0 // LATENT_GRID : x1 // LATENT_GRID] = 1.0
    return mask
