"""diffu's ModularPipeline: FLUX.3's with the Qwen-Image VAE's scale, and a glyph model's latent channels."""

from __future__ import annotations

from typing import Any, override

from diffusers import Flux3ModularPipeline
from diffusers.modular_pipelines import ModularPipelineBlocks

from diffu.modular_pipelines.diffu.modular_blocks_diffu import DiffuAutoBlocks

QWEN_IMAGE_SCALE = 8  # the Qwen-Image VAE's downsampling, before a VAE is loaded


class DiffuModularPipeline(Flux3ModularPipeline):
    """A ModularPipeline for diffu.

    FLUX.3's pipeline (its LoRA loader, the guider's need for unconditional embeddings) with the Qwen-Image VAE's scale
    and the latent channels of the transformer's input, less a glyph model's glyph map's. Without ``blocks`` it runs
    :class:`DiffuAutoBlocks`, so ``DiffuModularPipeline.from_pretrained(<diffu repo>)`` builds the whole model.
    """

    default_blocks_name = None  # diffusers resolves block names only in its own package; __init__ defaults instead

    def __init__(self, blocks: ModularPipelineBlocks | None = None, **kwargs: Any) -> None:
        super().__init__(blocks=blocks if blocks is not None else DiffuAutoBlocks(), **kwargs)

    @property
    @override
    def vae_scale_factor(self) -> int:
        vae = getattr(self, "vae", None)
        return QWEN_IMAGE_SCALE if vae is None else vae.spatial_compression_ratio

    @property
    @override
    def num_channels_latents(self) -> int:
        """The channels of the latents drawn: the transformer's input channels, less the glyph map's that a glyph model
        takes beside them (``glyph_channels``, a config of diffu's denoise step; 0 for blocks without it)."""
        return super().num_channels_latents - self.config.get("glyph_channels", 0)
