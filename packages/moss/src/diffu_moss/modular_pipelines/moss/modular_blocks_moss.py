"""MOSS-Image's blocks: the Layout prompt and text encoder steps, the reference encoder, the denoise step and the
decoder.

The denoise step is laid out as FLUX.3's (``Flux3AutoCoreDenoiseStep``): an auto step whose one workflow runs
``prepare_latents``, ``set_timesteps``, the loop of ``denoiser`` and ``after_denoiser`` and ``after_denoise`` (FLUX.3's
own step, back from tokens to the latent grid). MOSS-Image's denoising is the same with a reference image or without:
a reference (only for a DiT trained to take one) is encoded before it and read by the DiT in every pass. The page
pipeline (``diffu_page.pipeline.crop_blocks``) inserts its line repair by those names, as it does into FLUX.3's.
"""

from __future__ import annotations

from typing import override

from diffusers.modular_pipelines import AutoPipelineBlocks, OutputParam, SequentialPipelineBlocks
from diffusers.modular_pipelines.flux3.decoders import Flux3UnpackLatentsStep
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict

from diffu_moss.modular_pipelines.moss.before_denoise import MossPrepareLatentsStep, MossSetTimestepsStep
from diffu_moss.modular_pipelines.moss.decoders import MossDecodeStep
from diffu_moss.modular_pipelines.moss.denoise import MossDenoiseStep
from diffu_moss.modular_pipelines.moss.encoders import (
    MossLayoutPromptStep,
    MossReferenceEncodeStep,
    MossTextEncoderStep,
)


class MossCoreDenoiseStep(SequentialPipelineBlocks):
    """The noise drawn, the sampling grid set, the loop run and the latents unpacked to their grid."""

    model_name = "moss"
    block_classes = [MossPrepareLatentsStep(), MossSetTimestepsStep(), MossDenoiseStep(), Flux3UnpackLatentsStep()]
    block_names = ["prepare_latents", "set_timesteps", "denoise", "after_denoise"]

    @property
    @override
    def description(self) -> str:
        return (
            "Core denoise step that performs the denoising process for MOSS-Image, for text-to-image generation, "
            "conditioned on the reference latents when a reference-trained model is given some."
        )

    @property
    @override
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("latents")]


class MossAutoCoreDenoiseStep(AutoPipelineBlocks):
    """The denoise step by workflow: MOSS-Image has one, with a reference image or without."""

    model_name = "moss"
    block_classes = [MossCoreDenoiseStep]
    block_names = ["text2image"]
    block_trigger_inputs = [None]

    @property
    @override
    def description(self) -> str:
        return (
            "Core denoise step that selects the MOSS-Image denoising workflow by the inputs.\n"
            " - `MossCoreDenoiseStep` is used always (text-to-image, the only workflow; a reference image's latents, "
            "for a model trained to take them, condition it)."
        )


AUTO_BLOCKS = InsertableDict(
    [
        ("layout_prompt", MossLayoutPromptStep()),
        ("text_encoder", MossTextEncoderStep()),
        ("reference_encoder", MossReferenceEncodeStep()),
        ("denoise", MossAutoCoreDenoiseStep()),
        ("decode", MossDecodeStep()),
    ]
)


class MossAutoBlocks(SequentialPipelineBlocks):
    """MOSS-Image's text-to-image generation, conditioned on a reference image for a model trained to take one; a crop
    plan's caption (with ``page_lines``) is drawn from its Layout prompt."""

    model_name = "moss"
    block_classes = AUTO_BLOCKS.values()
    block_names = AUTO_BLOCKS.keys()
    _workflow_map = {"text2image": {"prompt": True}, "image_conditioned": {"prompt": True, "image": True}}

    @property
    @override
    def description(self) -> str:
        return (
            "Auto blocks that perform text-to-image generation using MOSS-Image: its DiT, the Qwen3.5 text encoder and "
            "the FLUX.2 VAE. A crop's layout caption with its lines (`page_lines`) is compiled as MOSS's Layout "
            "prompt first. A model trained with references (`takes_reference`) also takes a reference `image`."
        )

    @property
    @override
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images"), OutputParam.template("latents")]
