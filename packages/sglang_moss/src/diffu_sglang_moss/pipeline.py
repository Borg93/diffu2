"""SGLang's MOSS-Image pipeline drawing from a reference image: its config and stages.

The config is SGLang's MOSS-Image config with a second task, text and image to image (TI2I, chosen when a request
names an ``image_path``), the reference never resized and the drawn image's size the request's own, the DiT
:class:`~diffu_sglang_moss.dit.DiffuMOSSImageTransformer` in place of SGLang's, and the reference latents handed to
the DiT in both passes of guidance, the conditional and the unconditional, as ``diffu_moss``'s denoiser does, with
the place the DiT was trained to read them at (``reference_place``: over the drawn image, or above it).

Its sampling parameters default to ``diffu_moss``'s guidance, so a request drawn with the defaults is drawn as our
pipeline draws it.

The pipeline runs SGLang's MOSS-Image stages with the reference encoder (:mod:`diffu_sglang_moss.reference`) after the
text encoder: input validation, the prompts and the noise, the reference, the denoising loop (with our guidance stop,
:mod:`diffu_sglang_moss.denoising`), the decoder.
"""

from __future__ import annotations

from dataclasses import dataclass  # noqa: TID251  SGLang's pipeline configs are dataclasses
from pathlib import Path
from typing import Any, ClassVar

import PIL.Image
import torch
from sglang.multimodal_gen.configs.pipeline_configs.moss_image import MOSSImagePipelineConfig
from sglang.multimodal_gen.configs.sample.moss_image import MOSSImageSamplingParams
from sglang.multimodal_gen.configs.task_type import ModelTaskType
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines.moss_image import MOSSImagePipeline
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.moss_image import (
    MOSSImageBeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

from diffu_sglang_moss.denoising import DiffuMOSSImageDenoisingStage
from diffu_sglang_moss.reference import (
    REFERENCE_LATENTS,
    REFERENCE_PLACE,
    MOSSImageReferenceStage,
    reference_place,
    takes_reference,
)

TRANSFORMER = "DiffuMOSSImageTransformer"  # the registered name of diffu_sglang_moss.dit's DiT
NEGATIVE_PROMPT = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"  # the official demo's, ours


def _dit_kwargs(batch: Req) -> dict[str, Any]:
    """The DiT's keyword arguments of the plugin for ``batch``: its reference latents and their place, if it has any,
    and ``fused_qkv`` if it asks for it."""
    kwargs: dict[str, Any] = {"fused_qkv": True} if getattr(batch.sampling_params, "fused_qkv", False) else {}
    latents = batch.extra.get(REFERENCE_LATENTS)
    if latents:
        kwargs |= {"reference_latents": latents, "reference_place": batch.extra.get(REFERENCE_PLACE, "over")}
    return kwargs


@dataclass
class DiffuMOSSImageSamplingParams(MOSSImageSamplingParams):
    """SGLang's MOSS-Image sampling parameters with diffu_moss's defaults: the official demo's negative prompt (not an
    empty one), and the unconditional pass skipping its blocks from the first step (SGLang's from 1% of the steps).

    ``guidance_stop`` is our guider's ``stop`` (:mod:`diffu_sglang_moss.denoising`): the fraction of the steps after
    which only the conditional pass runs and its prediction is taken as it is; 1, every step guided, by default.
    ``fused_qkv`` computes each joint block's queries, keys and values by one GEMM (:mod:`diffu_sglang_moss.dit`),
    equal to the three GEMMs' within rounding; off by default.
    """

    negative_prompt: str = NEGATIVE_PROMPT
    slg_start: float = 0.0
    guidance_stop: float = 1.0
    fused_qkv: bool = False

    @classmethod
    def image_request_extra_fields(cls) -> frozenset[str]:
        return super().image_request_extra_fields() | {"guidance_stop", "fused_qkv"}


@dataclass
class DiffuMOSSImagePipelineConfig(MOSSImagePipelineConfig):
    """SGLang's MOSS-Image config, taking a reference image."""

    supported_task_types: ClassVar[tuple[ModelTaskType, ...] | None] = (ModelTaskType.T2I, ModelTaskType.TI2I)
    transformer_class_override: str | None = TRANSFORMER

    def calculate_condition_image_size(self, image: PIL.Image.Image, width: int, height: int) -> None:
        """None: the reference is read on its own grid, as it was trained (an off-grid one is refused)."""
        return None

    def prepare_calculated_size(self, image: PIL.Image.Image) -> None:
        """None: the drawn image's size is the request's, not the reference's."""
        return None

    def prepare_pos_cond_kwargs(
        self, batch: Req, device: torch.device, rotary_emb: object, dtype: torch.dtype
    ) -> dict[str, Any]:
        return {**super().prepare_pos_cond_kwargs(batch, device, rotary_emb, dtype), **_dit_kwargs(batch)}

    def prepare_neg_cond_kwargs(
        self, batch: Req, device: torch.device, rotary_emb: object, dtype: torch.dtype
    ) -> dict[str, Any]:
        return {**super().prepare_neg_cond_kwargs(batch, device, rotary_emb, dtype), **_dit_kwargs(batch)}


class DiffuMOSSImagePipeline(MOSSImagePipeline):
    """SGLang's MOSS-Image pipeline with the reference encoder; registered as ``MOSSImagePipeline``, the class a
    MOSS-Image folder's ``model_index.json`` names, so a model folder needs nothing of ours to be drawn from."""

    pipeline_config_cls = DiffuMOSSImagePipelineConfig

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        transformer = Path(self.model_path) / "transformer"
        reference = takes_reference(transformer)
        self.add_stage(InputValidationStage())
        self.add_stage(
            MOSSImageBeforeDenoisingStage(
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                scheduler=self.get_module("scheduler"),
            )
        )
        self.add_stage(
            MOSSImageReferenceStage(
                self.get_module("vae"), takes_reference=reference, place=reference_place(transformer)
            )
        )
        self.add_stage_factory(
            RoleType.DENOISER,
            lambda: DiffuMOSSImageDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                pipeline=self,
            ),
            "denoising_stage",
        )
        self.add_standard_decoding_stage()
