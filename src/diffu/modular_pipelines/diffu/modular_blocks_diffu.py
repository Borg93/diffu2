"""diffu's blocks: its text encoder, VAE encoder and decoder around FLUX.3's own denoise blocks.

The denoise step is diffusers' ``Flux3AutoCoreDenoiseStep`` unchanged: it reads the latent channels from the transformer
and the latent grid from the pipeline's ``vae_scale_factor``, so it runs diffu's 64-channel Qwen-Image latents and
its Qwen3-VL embeddings (FLUX.3's text encoder) as it runs FLUX.3's.
"""

from __future__ import annotations

from typing import override

from diffusers.modular_pipelines import AutoPipelineBlocks, OutputParam, SequentialPipelineBlocks
from diffusers.modular_pipelines.flux3.encoders import Flux3ProcessImagesInputStep
from diffusers.modular_pipelines.flux3.modular_blocks_flux3 import Flux3AutoCoreDenoiseStep
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict

from diffu.modular_pipelines.diffu.decoders import DiffuDecodeStep
from diffu.modular_pipelines.diffu.encoders import DiffuTextEncoderStep, DiffuVaeEncoderStep


class DiffuVaeEncoderSequentialStep(SequentialPipelineBlocks):
    """Reference images preprocessed as FLUX.3 preprocesses them, then encoded with the Qwen-Image VAE."""

    model_name = "diffu"
    block_classes = [Flux3ProcessImagesInputStep(), DiffuVaeEncoderStep()]
    block_names = ["preprocess", "encode"]

    @property
    @override
    def description(self) -> str:
        return "VAE encoder step that preprocesses and encodes the reference images into their latent representations."


class DiffuAutoVaeEncoderStep(AutoPipelineBlocks):
    """The reference images' latents when an ``image`` is given; skipped otherwise."""

    model_name = "diffu"
    block_classes = [DiffuVaeEncoderSequentialStep]
    block_names = ["image_conditioning"]
    block_trigger_inputs = ["image"]

    @property
    @override
    def description(self) -> str:
        return (
            "VAE encoder step that encodes the reference images into their latent representations.\n"
            " - `DiffuVaeEncoderSequentialStep` is used when `image` is provided (image-to-image).\n"
            " - If `image` is not provided, step will be skipped."
        )


AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", DiffuTextEncoderStep()),
        ("vae_encoder", DiffuAutoVaeEncoderStep()),
        ("denoise", Flux3AutoCoreDenoiseStep()),
        ("decode", DiffuDecodeStep()),
    ]
)


class DiffuAutoBlocks(SequentialPipelineBlocks):
    """diffu's text-to-image and reference-conditioned generation, FLUX.3's pipeline with diffu's encoders."""

    model_name = "diffu"
    block_classes = AUTO_BLOCKS.values()
    block_names = AUTO_BLOCKS.keys()
    _workflow_map = {
        "text2image": {"prompt": True},
        "image_conditioned": {"prompt": True, "image": True},
    }

    @property
    @override
    def description(self) -> str:
        return (
            "Auto blocks that perform text-to-image and image-to-image generation using diffu: FLUX.3's architecture "
            "with diffu's weights, FLUX.3's text encoder (Qwen3-VL) and the Qwen-Image VAE."
        )

    @property
    @override
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images"), OutputParam.template("latents")]
