"""MOSS-Image's ModularPipeline."""

from __future__ import annotations

from typing import Any

from diffusers.modular_pipelines import ModularPipeline, ModularPipelineBlocks

from diffu_moss.modular_pipelines.moss.modular_blocks_moss import MossAutoBlocks

FLUX2_VAE_SCALE = 8  # the FLUX.2 VAE's downsampling, before a VAE is loaded
LATENT_CHANNELS = 128  # the DiT's latent channels (the VAE's 32, packed 2x2), before a DiT is loaded


class MossModularPipeline(ModularPipeline):
    """A ModularPipeline for MOSS-Image.

    Without ``blocks`` it runs :class:`MossAutoBlocks`, so ``MossModularPipeline.from_pretrained(<MOSS repo>)`` builds
    the whole model.
    """

    default_blocks_name = None  # diffusers resolves block names only in its own package; __init__ defaults instead

    def __init__(self, blocks: ModularPipelineBlocks | None = None, **kwargs: Any) -> None:
        super().__init__(blocks=blocks if blocks is not None else MossAutoBlocks(), **kwargs)

    @property
    def vae_scale_factor(self) -> int:
        """The VAE's downsampling: its latents are half a latent cell (16 px) on a side."""
        vae = getattr(self, "vae", None)
        return FLUX2_VAE_SCALE if vae is None else 2 ** (len(vae.config.block_out_channels) - 1)

    @property
    def num_channels_latents(self) -> int:
        """The channels of the latents drawn: the DiT's."""
        transformer = getattr(self, "transformer", None)
        return LATENT_CHANNELS if transformer is None else transformer.config.latent_channels

    @property
    def requires_unconditional_embeds(self) -> bool:
        """Whether the guider guides at its current step, so the negative prompt must be encoded."""
        guider = getattr(self, "guider", None)
        return guider is not None and guider._enabled and guider.num_conditions > 1
