"""diffu's decoder block: the denoised latents through the Qwen-Image VAE, as FLUX.3's decoder does with its own."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from diffusers import AutoencoderKLQwenImage
from diffusers.modular_pipelines import ComponentSpec, InputParam, ModularPipelineBlocks, OutputParam, PipelineState
from diffusers.modular_pipelines.flux3.image_processor import Flux3ImageProcessor

from diffu.latents import decode_latents

if TYPE_CHECKING:
    from diffu.modular_pipelines.diffu.modular_pipeline import DiffuModularPipeline


class DiffuDecodeStep(ModularPipelineBlocks):
    """The denoised latents as images (:func:`diffu.latents.decode_latents`), cropped back to a padded reference."""

    model_name = "diffu"

    @property
    @override
    def description(self) -> str:
        return (
            "Step that decodes the denoised latents into images with the Qwen-Image VAE: unpacked, denormalized with "
            "the VAE's latent statistics, decoded."
        )

    @property
    @override
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLQwenImage),
            ComponentSpec("image_processor", Flux3ImageProcessor, default_creation_method="from_config"),
        ]

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("output_type"),
            InputParam("latents", required=True, type_hint=torch.Tensor, description="The denoised latents"),
            InputParam(
                "reference_crop",
                type_hint=list[tuple[int, int, int, int] | None] | None,
                description="For image-to-image, per reference, the (left, top, right, bottom) box to crop the "
                "decoded image back to (a single padded reference) or None",
            ),
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images")]

    @torch.no_grad()
    def __call__(
        self, components: DiffuModularPipeline, state: PipelineState
    ) -> tuple[DiffuModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        images = decode_latents(components.vae, block_state.latents)
        block_state.images = components.image_processor.postprocess(
            images, output_type=block_state.output_type, reference_crop=block_state.reference_crop
        )
        self.set_block_state(state, block_state)
        return components, state
