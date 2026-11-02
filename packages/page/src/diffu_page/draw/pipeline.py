"""The page pipeline: our blocks around a model's own Modular Diffusers blocks.

A model repo (MOSS-Image, ``diffu-moss-init``'s; FLUX.3 Klein for the reference runs) names its blocks in
``modular_model_index.json``; :data:`FAMILIES` maps those to the model's blocks and pipeline class, so a model is
chosen by its directory alone. Two sets of blocks are built:

- :class:`PagePlanBlocks` (no model): ``LoadLayoutStep`` -> ``MarkFromScanStep`` -> ``OpeningPageStep`` ->
  ``RefillStep`` -> ``PlanCropsStep``. A template page (or one page of a two-page opening), a text, a style and
  optionally a reference crop and a print/hand classifier (to mark the template from its scan) in; the crops to draw
  out, captioned for the reference if there is one.
- :func:`crop_blocks`: ``CropPromptStep`` -> ``ReferenceGridStep`` -> the model's text encoder, VAE encoders, denoise
  and decode. A crop plan in; its image out. The model's auto blocks pick text-to-image or reference-conditioned
  denoising from the inputs (``image``); a line repair sits in both (:mod:`diffu_page.draw.blocks.repair`). MOSS-Image
  compiles the plan's caption and lines into its own Layout prompt first, and takes a reference only when its DiT was
  trained to (``takes_reference`` in its model index: ``diffu_moss.train``).

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
from diffusers import FlowMatchEulerDiscreteScheduler, Flux3AutoBlocks, Flux3ModularPipeline
from diffusers.modular_pipelines import ModularPipeline, OutputParam, PipelineState, SequentialPipelineBlocks
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from diffu_core.loading import COMPONENT_DTYPES, require_loaded
from diffu_core.repo import ModelIndex
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
from diffu_page.draw.lora import nemo_lora_state_dict
from diffu_page.draw.plan import CropPlan
from diffu_page.page.crops.geometry import pixel_box
from diffu_page.page.crops.grid import LATENT_GRID
from diffu_page.page.labels import LabelFormat
from diffu_page.quality.judge import MAX_LINE_CER, Judge, default_judge

# Errors after which the device cannot be trusted: a run stops there instead of failing every later crop, and is not
# retried into the same out-of-memory (which once froze the DGX Spark's unified memory)
DEVICE_ERRORS = (torch.OutOfMemoryError, torch.AcceleratorError)
REPAIR_SEED_STRIDE = 7919  # between a crop's seed and each repair pass's, so no pass redraws from the crop's own noise


class ModelFamily(BaseModel):
    """A model family's own blocks and pipeline class, the blocks of its that generation leaves out, the guidance
    scale it draws at unless the settings name one, and whether LoRAs load into it at draw time."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    blocks: type[SequentialPipelineBlocks]
    pipeline: type[ModularPipeline]
    unused: tuple[str, ...] = ()
    guidance_scale: float = 4.0
    takes_loras: bool = True


FAMILIES: dict[str, ModelFamily] = {
    # FLUX.3's prompt upsampler rewrites a prompt as prose: it would rewrite the quoted lines of a layout caption
    "Flux3AutoBlocks": ModelFamily(blocks=Flux3AutoBlocks, pipeline=Flux3ModularPipeline, unused=("prompt_upsampler",)),
    # a MOSS LoRA is merged into the DiT a MOSS repo names (diffu-moss-init --transformer)
    "MossAutoBlocks": ModelFamily(
        blocks=MossAutoBlocks, pipeline=MossModularPipeline, guidance_scale=MOSS_GUIDANCE_SCALE, takes_loras=False
    ),
}


def model_family(model: Path) -> ModelFamily:
    """The family of the Modular Diffusers repo at ``model``, by the blocks its ``modular_model_index.json`` names.

    Raises:
        ValueError: If no family has those blocks.
    """
    blocks = ModelIndex.read(model).blocks_class_name
    if blocks not in FAMILIES:
        raise ValueError(f"{model} names the blocks {blocks!r}; the known model families are {sorted(FAMILIES)}")
    return FAMILIES[blocks]


class CropBlocks(SequentialPipelineBlocks):
    """A crop plan in; the drawn crop out (built by :func:`crop_blocks`)."""

    _workflow_map = {
        "text2image": {"crop_plan": True},
        "image_conditioned": {"crop_plan": True, "image": True},
    }

    @property
    @override
    def description(self) -> str:
        return "Draws a planned crop with a model's own blocks (reference-conditioned when an `image` is given)."

    @property
    @override
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images")]


def crop_blocks(family: ModelFamily) -> CropBlocks:
    """Our crop steps before ``family``'s own blocks (the crop prompt and the reference grid), and a line repair in each
    of its denoise branches (:mod:`diffu_page.draw.blocks.repair`), idle unless a repair mask is given.

    Raises:
        TypeError: If a denoise branch of ``family`` has no scheduler step (``after_denoiser``) in its denoise loop.
    """
    model = family.blocks().sub_blocks
    for name in family.unused:
        model.pop(name)
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
    """A template page (or one page of a two-page opening), a text, a style and optionally a reference crop and a
    print/hand classifier in; the crops to draw out. No model but the small CNN that marks a template: CPU only."""

    block_classes = [LoadLayoutStep, MarkFromScanStep, OpeningPageStep, RefillStep, PlanCropsStep]
    block_names = ["load_layout", "mark_from_scan", "opening_page", "refill", "plan_crops"]

    @property
    @override
    def description(self) -> str:
        return (
            "Reads a PAGE/ALTO template (marked from its scan if asked), writes the text into its running-text lines "
            "and plans it as crops."
        )

    @property
    @override
    def outputs(self) -> list[OutputParam]:
        return [OutputParam("crop_plans", type_hint=list, description="The page's CropPlans, in page order.")]


class DrawSettings(BaseModel):
    """How every crop is drawn.

    Attributes:
        steps: Denoising steps per crop.
        guidance_scale: Classifier-free guidance scale; exactly 1 samples without guidance (one pass a step). ``None``
            is the model's own (4 for FLUX.3; 3 for MOSS-Image, its demo's).
        reference_guidance_scale: How hard guidance also pushes toward the reference image
            (:mod:`diffu_page.draw.guidance`, a third pass per step); 1 guides the caption alone.
        guidance_stop: The fraction of the steps after which guidance stops: one pass a step after it; 1 guides every
            step.
        label_format: ``alto`` or ``page``.
        max_line_cer: The read-back CER above which a line's label is tagged ``unclear``.
        text_layers: The text-encoder hidden layers stacked into the prompt embeddings; ``None`` is the model's own
            choice (FLUX.3: every 4th of Qwen3-VL's).
        device: Where the components run.
        sampling_shift: The timestep shift the scheduler samples at; ``None`` is the model's own (FLUX.3's 6.93, which
            its LoRAs trained at; MOSS-Image's 5, its demo's).
        lora_scale: How strongly every LoRA applies: 1 at its trained strength, 0 not at all (diffusers'
            ``set_adapters`` weights, which fusing keeps).
        fuse_loras: Fuse the LoRAs into the transformer's weights at load and drop their adapter layers: no LoRA
            matmuls per step, and the transformer can be compiled; the LoRAs can't be switched or rescaled after.
        compile_blocks: Compile the transformer's repeated blocks (torch.compile, diffusers' regional compilation) at
            load; the first crop pays for the compilation. A run with LoRAs must fuse them first. MOSS-Image's joint
            blocks are compiled after any MXFP8 swap, so the quantization of their inputs fuses into them; not with a
            step cache (``cache``), whose hooks decide eagerly around each block.
        precision: MOSS-Image only: its DiT's joint-block projections in their own bfloat16, or in MXFP8: weights and
            inputs in blocks of 32 FP8 values sharing a power-of-two scale, multiplied by a Blackwell GPU's tensor cores
            (:mod:`diffu_moss.mxfp8`; emulated on the CPU). It pays off with ``compile_blocks`` or ``aoti_package``,
            which fuse each layer's quantization of its input: run eagerly, that is some fifteen small kernels.
        text_encoder_precision: MOSS-Image only: its text encoder's decoder-layer projections alike (half their
            memory).
        aoti_package: MOSS-Image only: run its joint blocks from this AOTInductor package (``diffu-moss-aoti``), built
            for ``precision``, this GPU's architecture and this torch version (any other is refused); instead of
            ``compile_blocks``.
        sampler: MOSS-Image only: what steps its sampling grid: ``euler``, its own sampler, or ``unipc``, diffusers'
            multistep UniPC solver, which draws as well in fewer ``steps``. Either calls the DiT once a pass a step, at
            the grid's levels.
        cache: MOSS-Image only: a step cache that skips its DiT's joint blocks at some steps
            (:mod:`diffu_moss.step_cache`), by ``kind``: ``{"kind": "first_block", "threshold": 0.05}``,
            ``{"kind": "taylorseer", "cache_interval": 5, "disable_cache_before_step": 3}``, ``{"kind": "magcache",
            "mag_ratios": [...]}``, or ``{"kind": "magcache_calibration", "ratios_file": ...}``, which skips nothing
            and measures MagCache's ratios for this sampler and these ``steps``. Not with ``compile_blocks``.
        repair_passes: How many times a drawn crop's rejected lines are drawn again in place (:func:`repair_lines`),
            each pass from its own noise (:func:`repair_seed`); 0 keeps every crop as drawn.
    """

    model_config = ConfigDict(frozen=True)

    steps: int = Field(default=50, ge=1)
    guidance_scale: float | None = None
    reference_guidance_scale: float = 1.0
    guidance_stop: float = Field(default=1.0, ge=0.0, le=1.0)
    label_format: LabelFormat = LabelFormat.ALTO
    max_line_cer: float = MAX_LINE_CER
    text_layers: tuple[int, ...] | None = None
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
        _ = self.moss_inference  # MossInference validates itself: a package or a step cache refuses compile_blocks
        return self

    @property
    def moss_inference(self) -> MossInference:
        """How a MOSS-Image model computes (:mod:`diffu_moss.inference`)."""
        return MossInference(
            precision=self.precision,
            text_encoder_precision=self.text_encoder_precision,
            compile_blocks=self.compile_blocks,
            aoti_package=self.aoti_package,
            sampler=self.sampler,
            cache=self.cache,
        )

    @property
    def moss_only(self) -> list[str]:
        """The settings asked that only MOSS-Image takes, by name: MXFP8 for its DiT or text encoder, a package of its
        joint blocks, UniPC, a step cache."""
        asked = {
            "precision": self.precision is not Precision.BF16,
            "text_encoder_precision": self.text_encoder_precision is not Precision.BF16,
            "aoti_package": self.aoti_package is not None,
            "sampler": self.sampler is not Sampler.EULER,
            "cache": self.cache is not None,
        }
        return [name for name, is_asked in asked.items() if is_asked]

    @property
    def guides_reference(self) -> bool:
        """Whether guidance also pushes toward the reference image (a third pass a step)."""
        return not math.isclose(self.reference_guidance_scale, 1.0)

    def pipeline_inputs(self) -> dict[str, object]:
        """The settings a pipeline takes per call, by input name (the guidance scale and device are set at load; the
        label format and the unclear limit apply after the judge)."""
        layers = {} if self.text_layers is None else {"text_encoder_out_layers": self.text_layers}
        return {"num_inference_steps": self.steps, **layers}


type BlocksFor = Callable[[ModelFamily], SequentialPipelineBlocks]


def load_pipeline(model: Path, blocks: BlocksFor, *, loras: dict[str, Path], settings: DrawSettings) -> ModularPipeline:
    """``blocks`` built for the model at ``model``, its components loaded (VAE in float32, the rest in bfloat16) on
    ``settings.device``, the guider at ``settings.guidance_scale`` (the model family's own when None, and
    ``settings.reference_guidance_scale``) until ``settings.guidance_stop``, and every
    LoRA of ``loras`` (NeMo ``model/`` dirs by adapter name) active, fused into the transformer when
    ``settings.fuse_loras``, and the transformer's repeated blocks compiled when ``settings.compile_blocks``. A
    MOSS-Image model computes as ``settings.moss_inference`` asks (:func:`diffu_moss.inference.prepare_inference`: its
    layers in MXFP8, UniPC, its joint blocks from an AOTInductor package or compiled, a step cache around them, for
    pages of ``settings.steps``).

    Everything that can be checked without the model is checked first: its family, the guider, every LoRA, and
    MOSS-Image's precision and package against the device.

    Raises:
        ValueError: If the model's blocks have no family, the guidance settings do not go together, a LoRA would
            load at the wrong strength or into a model that takes none at draw time (MOSS-Image), the blocks are to
            be compiled around LoRAs that are not fused, a setting only MOSS-Image takes is asked of another model, or
            MOSS-Image's precision or package does not fit the device, this process or the model.
        RuntimeError: If a component failed to load.
    """
    if settings.compile_blocks and loras and not settings.fuse_loras:
        raise ValueError("compile the transformer's blocks only with its LoRAs fused: adapter layers would recompile")
    family = model_family(model)
    if loras and not family.takes_loras:
        raise ValueError(
            f"{model} takes no LoRAs at draw time: merge them into the transformer it draws with (diffu-moss-init "
            "--transformer)"
        )
    moss = family.pipeline is MossModularPipeline
    if settings.moss_only and not moss:
        raise ValueError(f"{model} is not MOSS-Image, which alone takes {', '.join(settings.moss_only)}")
    if moss:
        check_inference(settings.moss_inference, torch.device(settings.device))
    guidance_scale = family.guidance_scale if settings.guidance_scale is None else settings.guidance_scale
    guider = ReferenceGuidance(
        guidance_scale=guidance_scale,
        reference_scale=settings.reference_guidance_scale,
        stop=settings.guidance_stop,
    )
    adapters = {name: nemo_lora_state_dict(lora_dir) for name, lora_dir in loras.items()}
    pipe = family.pipeline(blocks=blocks(family), pretrained_model_name_or_path=str(model))
    pipe.load_components(dtype=COMPONENT_DTYPES)
    require_loaded(pipe)
    pipe.update_components(guider=guider)
    if settings.sampling_shift is not None:
        pipe.update_components(
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift=settings.sampling_shift)
        )
    pipe.to(settings.device)
    for name, state_dict in adapters.items():
        pipe.load_lora_weights(state_dict, adapter_name=name)
    if loras:
        pipe.set_adapters(list(loras), adapter_weights=[settings.lora_scale] * len(loras))
    if loras and settings.fuse_loras:
        pipe.fuse_lora(adapter_names=list(loras))
        pipe.unload_lora_weights()
    if moss:
        prepare_inference(pipe, settings.moss_inference, steps=settings.steps)
    elif settings.compile_blocks:  # dynamic: crop sizes and caption lengths vary, and each new shape would recompile
        pipe.transformer.compile_repeated_blocks(fullgraph=True, dynamic=True)
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
) -> DrawnCrop:
    """One crop drawn by a crop pipeline (:func:`crop_blocks`) from noise seed ``seed``, conditioned on ``reference``
    when there is one, every line judged by ``judge`` (the default TrOCR judge on ``settings.device`` if none), and
    labelled with the lines the judge could not confirm tagged ``unclear``.

    Raises:
        ValueError: If ``settings`` guide toward a reference and there is none, before anything is drawn.
    """
    image = _draw(pipe, plan, seed=seed, reference=reference, settings=settings)
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
) -> DrawnCrop:
    """``crop`` with the lines its verdicts reject drawn again in place (:mod:`diffu_page.draw.blocks.repair`) from
    noise seed ``seed``, which should not be the seed that drew them. Each line is redrawn over the latent cells its box
    touches, and a redrawn line replaces those cells of the crop only if the judge passes it, so every other pixel stays
    the crop's; the labels then follow the judge's verdicts on the result, and a line still rejected stays ``unclear``.

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
    """``image`` as the model's latents ``(1, C, height / 16, width / 16)``, made by the model's own VAE encoder step as
    it encodes a reference image, without the reference preprocessing (a drawn crop is already on the latent grid).
    MOSS-Image, whose reference encoder takes a reference only for a DiT trained with them, encodes as that step and
    its decoder do (:func:`diffu_moss.latents.encode_latents`)."""
    processor = pipe.image_processor
    pixels = processor.normalize(processor.numpy_to_pt(processor.pil_to_numpy(image.convert("RGB"))))
    if isinstance(pipe, MossModularPipeline):
        return encode_latents(pipe.vae, pixels)
    encode = pipe.blocks.sub_blocks["vae_encoder"].sub_blocks["image_conditioning"].sub_blocks["encode"]
    state = PipelineState()
    state.set("condition_images", [pixels])
    encode(pipe, state)
    return state.get("image_cond_latents")[0]


def _draw(
    pipe: ModularPipeline,
    plan: CropPlan,
    *,
    seed: int,
    reference: Image.Image | None,
    settings: DrawSettings,
    **repair: torch.Tensor,
) -> Image.Image:
    """``plan`` drawn by a crop pipeline from noise seed ``seed``, conditioned on ``reference`` when there is one; with
    ``repair`` (``repair_latents`` and ``repair_mask``), drawn again only inside the mask."""
    if reference is None and settings.guides_reference:
        raise ValueError("reference guidance needs a reference image: draw this crop at reference_guidance_scale 1")
    reference_input = {} if reference is None else {"image": reference}
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
