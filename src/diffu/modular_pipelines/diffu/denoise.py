"""diffu's denoise step: FLUX.3's, with a glyph model's glyph map concatenated to its latents on channels.

A glyph model (``diffu-init --glyph``) takes the glyph map's latents beside its own, token by token, as FLUX.1's Canny
and Depth models take their control image (diffusers' ``FluxControlPipeline``): each image token carries its own cell
of the map, in every pass of classifier-free guidance. FLUX.3's transformer has no ``out_channels``, so its prediction
is as wide as its input; the latents are stepped with its first channels. :func:`glyph_denoise_step` puts three steps
into a copy of diffusers' ``Flux3AutoCoreDenoiseStep``, in both branches: :class:`GlyphPrepareStep` packs the map's
latents as FLUX.3 packs the noise, :class:`GlyphConcatStep` puts them beside the latents before the transformer, and
:class:`GlyphSplitStep` cuts the latents and the prediction back to the latents' channels before the scheduler step. A
model without the glyph map (``glyph_channels`` 0, as every diffu repo before E2c) draws through them as through
FLUX.3's own step.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, override

import torch
from diffusers.modular_pipelines import InputParam, ModularPipelineBlocks, OutputParam, PipelineState
from diffusers.modular_pipelines.flux3.modular_blocks_flux3 import Flux3AutoCoreDenoiseStep
from diffusers.modular_pipelines.modular_pipeline import BlockState
from diffusers.modular_pipelines.modular_pipeline_utils import ConfigSpec

if TYPE_CHECKING:
    from diffu.modular_pipelines.diffu.modular_pipeline import DiffuModularPipeline

GLYPH_CHANNELS = ConfigSpec(
    "glyph_channels",
    0,
    description="The glyph map's channels a glyph model takes beside its latents (its modular_model_index.json); 0 "
    "for a model without",
)


def _tokens(grid: torch.Tensor) -> torch.Tensor:
    """``(B, C, h, w)`` on the latent grid as tokens ``(B, h * w, C)``, in the order FLUX.3 packs its latents."""
    return grid.flatten(2).transpose(1, 2)


class GlyphPrepareStep(ModularPipelineBlocks):
    """A glyph model's glyph map as tokens, one beside each image token, in the model inputs' dtype; none for a model
    without the glyph map.

    Raises:
        ValueError: From the call: if a glyph model has no glyph map, a model without one is given one, or the map's
            latents are on another latent grid than the image's.
    """

    model_name = "diffu"

    @property
    @override
    def description(self) -> str:
        return (
            "Prepares a glyph model's glyph map: its latents packed into tokens as the image latents are (row by row), "
            "one beside each image token, repeated to the batch. A model without the glyph map takes none."
        )

    @property
    @override
    def expected_configs(self) -> list[ConfigSpec]:
        return [GLYPH_CHANNELS]

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "glyph_latents",
                type_hint=torch.Tensor,
                description="The glyph map's latents (1, C, height // 16, width // 16), from the glyph encoder step",
            ),
            InputParam("latents", required=True, type_hint=torch.Tensor, description="The initial noise as tokens."),
            InputParam("height", type_hint=int, description="The height in pixels of the image drawn"),
            InputParam("width", type_hint=int, description="The width in pixels of the image drawn"),
            InputParam("dtype", type_hint=torch.dtype, description="The dtype of the model inputs"),
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "glyph_tokens",
                type_hint=torch.Tensor,
                description="The glyph map's latents as tokens (B, seq_len, glyph_channels), one beside each image "
                "token; None for a model without the glyph map",
            )
        ]

    @torch.no_grad()
    def __call__(
        self, components: DiffuModularPipeline, state: PipelineState
    ) -> tuple[DiffuModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        glyph_channels: int = components.config.glyph_channels
        glyph: torch.Tensor | None = block_state.glyph_latents
        if glyph is None and glyph_channels:
            raise ValueError("a glyph model draws from a glyph map: give it `glyph_image`, the map at the image's size")
        if glyph is not None and not glyph_channels:
            raise ValueError("this model takes no glyph map: its model index names no glyph_channels")
        block_state.glyph_tokens = None
        if glyph is not None:
            latents: torch.Tensor = block_state.latents
            token = components.vae_scale_factor * 2
            grid = (block_state.height // token, block_state.width // token)
            if tuple(glyph.shape[-2:]) != grid:
                raise ValueError(
                    f"the glyph map's latents are on a {tuple(glyph.shape[-2:])} grid, the image's on {grid}: draw the "
                    "map at the image's size"
                )
            tokens = _tokens(glyph).to(latents.device, block_state.dtype)
            block_state.glyph_tokens = tokens.expand(latents.shape[0], -1, -1)
        self.set_block_state(state, block_state)
        return components, state


class GlyphConcatStep(ModularPipelineBlocks):
    """Before the transformer: a glyph model's latents with the glyph map's tokens beside them on channels."""

    model_name = "diffu"

    @property
    @override
    def description(self) -> str:
        return (
            "Step within the denoising loop, before the denoiser: concatenates the glyph map's tokens to the latents "
            "on channels, so the transformer sees the map in every pass of the guider. Does nothing for a model "
            "without the glyph map."
        )

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor, description="The latents as tokens."),
            InputParam("glyph_tokens", type_hint=torch.Tensor, description="The glyph map's latents as tokens."),
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents", type_hint=torch.Tensor, description="The latents, the glyph map's tokens beside them"
            )
        ]

    @torch.no_grad()
    def __call__(
        self, components: DiffuModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> tuple[DiffuModularPipeline, BlockState]:
        if block_state.glyph_tokens is not None:
            latents: torch.Tensor = block_state.latents
            block_state.latents = torch.cat([latents, block_state.glyph_tokens.to(latents.dtype)], dim=-1)
        return components, block_state


class GlyphSplitStep(ModularPipelineBlocks):
    """After the transformer: a glyph model's latents and prediction cut back to the latents' own channels, which the
    scheduler steps (the prediction's other channels, the glyph map's, are never trained)."""

    model_name = "diffu"

    @property
    @override
    def description(self) -> str:
        return (
            "Step within the denoising loop, after the denoiser: keeps the latents' own channels of the latents and of "
            "the predicted velocity, for the scheduler to step. Does nothing for a model without the glyph map."
        )

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latents", required=True, type_hint=torch.Tensor, description="The latents and the glyph map's tokens."
            ),
            InputParam("noise_pred", required=True, type_hint=torch.Tensor, description="The predicted velocity."),
            InputParam("glyph_tokens", type_hint=torch.Tensor, description="The glyph map's latents as tokens."),
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor, description="The latents being denoised"),
            OutputParam("noise_pred", type_hint=torch.Tensor, description="The predicted velocity of the latents"),
        ]

    @torch.no_grad()
    def __call__(
        self, components: DiffuModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> tuple[DiffuModularPipeline, BlockState]:
        if block_state.glyph_tokens is not None:
            channels = block_state.latents.shape[-1] - block_state.glyph_tokens.shape[-1]
            block_state.latents = block_state.latents[..., :channels]
            block_state.noise_pred = block_state.noise_pred[..., :channels]
        return components, block_state


def glyph_denoise_step() -> Flux3AutoCoreDenoiseStep:
    """FLUX.3's denoise step (``Flux3AutoCoreDenoiseStep``) with a glyph model's glyph map in both its branches:
    prepared after the noise, beside the latents from before the denoiser to before the scheduler step. A copy, made
    once (:data:`diffu.modular_pipelines.diffu.modular_blocks_diffu.AUTO_BLOCKS`): diffusers shares its blocks'
    instances process-wide, so FLUX.3's own denoise step stays as it is."""
    denoise = copy.deepcopy(Flux3AutoCoreDenoiseStep())
    for branch in denoise.sub_blocks.values():
        steps = branch.sub_blocks
        steps.insert("prepare_glyph", GlyphPrepareStep(), list(steps).index("prepare_latents") + 1)
        loop = steps["denoise"].sub_blocks
        loop.insert("glyph_concat", GlyphConcatStep(), list(loop).index("denoiser"))
        loop.insert("glyph_split", GlyphSplitStep(), list(loop).index("after_denoiser"))
    return denoise
