"""diffu's blocks: its text encoder, VAE encoders and decoder around FLUX.3's own denoise blocks.

The denoise step is a copy of diffusers' ``Flux3AutoCoreDenoiseStep`` with a glyph model's glyph map concatenated to
its latents on channels (:func:`diffu.modular_pipelines.diffu.denoise.glyph_denoise_step`), which a model without the
glyph map runs as FLUX.3's own. It reads the latent channels from the pipeline (the transformer's, less a glyph model's
glyph channels) and the latent grid from its ``vae_scale_factor``, so it runs diffu's 64-channel Qwen-Image latents and
its Qwen3-VL embeddings (FLUX.3's text encoder) as it runs FLUX.3's.
"""

from __future__ import annotations

from typing import override

from diffusers.modular_pipelines import AutoPipelineBlocks, OutputParam, SequentialPipelineBlocks
from diffusers.modular_pipelines.flux3.encoders import Flux3ProcessImagesInputStep
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict

from diffu.modular_pipelines.diffu.decoders import DiffuDecodeStep
from diffu.modular_pipelines.diffu.denoise import glyph_denoise_step
from diffu.modular_pipelines.diffu.encoders import DiffuGlyphEncoderStep, DiffuTextEncoderStep, DiffuVaeEncoderStep


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
        ("glyph_encoder", DiffuGlyphEncoderStep()),
        ("denoise", glyph_denoise_step()),
        ("decode", DiffuDecodeStep()),
    ]
)


class DiffuAutoBlocks(SequentialPipelineBlocks):
    """diffu's text-to-image and reference-conditioned generation, FLUX.3's pipeline with diffu's encoders; a glyph
    model draws from its glyph map (``glyph_image``) in either."""

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
            "with diffu's weights, FLUX.3's text encoder (Qwen3-VL) and the Qwen-Image VAE. A glyph model draws from "
            "its glyph map (`glyph_image`), concatenated to its latents on channels."
        )

    @property
    @override
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images"), OutputParam.template("latents")]
