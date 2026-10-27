"""MOSS-Image's decoder block: the denoised latents through the FLUX.2 VAE (:func:`diffu_moss.latents.decode_latents`),
as images."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from diffusers import AutoencoderKLFlux2
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.modular_pipelines import ComponentSpec, InputParam, ModularPipelineBlocks, OutputParam, PipelineState

from diffu_moss.latents import PIXELS_PER_LATENT, decode_latents

if TYPE_CHECKING:
    from diffu_moss.modular_pipelines.moss.modular_pipeline import MossModularPipeline


class MossDecodeStep(ModularPipelineBlocks):
    """The denoised latents ``[1, latent_channels, h, w]`` as images: un-normalized by the VAE's batch-norm statistics,
    un-patched 2x2, decoded and clamped to [-1, 1]."""

    model_name = "moss"

    @property
    @override
    def description(self) -> str:
        return (
            "Step that decodes the denoised latents into images with the FLUX.2 VAE: un-normalized by its batch-norm "
            "statistics, un-patched 2x2, decoded."
        )

    @property
    @override
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLFlux2),
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": PIXELS_PER_LATENT}),
                default_creation_method="from_config",
            ),
        ]

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("output_type"),
            InputParam("latents", required=True, type_hint=torch.Tensor, description="The denoised latents"),
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images")]

    @torch.no_grad()
    def __call__(
        self, components: MossModularPipeline, state: PipelineState
    ) -> tuple[MossModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        pixels = decode_latents(components.vae, block_state.latents)
        block_state.images = components.image_processor.postprocess(pixels, output_type=block_state.output_type)
        self.set_block_state(state, block_state)
        return components, state
