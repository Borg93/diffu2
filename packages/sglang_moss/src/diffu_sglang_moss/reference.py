"""The reference image (a sample page) of a request, as the clean latents the DiT reads beside the drawn image's.

As ``diffu_moss`` reads it (``diffu_moss.latents.image_pixels`` and ``encode_latents``): on its own 16-px grid, never
resized, in [-1, 1]; the VAE's mode, patched 2x2 and normalized by the VAE's batch-norm statistics. A request names it
as any SGLang image request does (``image_path``: a path, a URL, or several); its latents go into ``batch.extra`` and
from there into both passes of guidance (:mod:`diffu_sglang_moss.config`).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import ComponentUse
from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_flux2 import AutoencoderKLFlux2
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from torch import Tensor

PIXELS_PER_LATENT = 16  # 8x VAE downsampling, then 2x2 patches
REFERENCE_LATENTS = "diffu_reference_latents"  # the key of batch.extra the latents are kept under
EXPORT_RECORD = "export.json"  # beside the DiT's weights: diffu_moss's record of whether it takes a reference


def takes_reference(transformer: Path) -> bool:
    """Whether the DiT folder ``transformer`` was trained to take a reference image, by its export's record."""
    record = transformer / EXPORT_RECORD
    return record.is_file() and bool(json.loads(record.read_text(encoding="utf-8")).get("takes_reference"))


def image_pixels(image: Image.Image) -> Tensor:
    """An image on the latent grid as the VAE encodes it: ``[1, 3, H, W]`` in [-1, 1], never resized.

    Raises:
        ValueError: If a side is not a multiple of 16 px.
    """
    if image.width % PIXELS_PER_LATENT or image.height % PIXELS_PER_LATENT:
        raise ValueError(f"the reference image's size {image.size} is not on the {PIXELS_PER_LATENT}-px latent grid")
    pixels = VaeImageProcessor.numpy_to_pt(VaeImageProcessor.pil_to_numpy(image.convert("RGB")))
    return VaeImageProcessor.normalize(pixels)


def encode_latents(vae: AutoencoderKLFlux2, pixels: Tensor) -> Tensor:
    """Images ``[B, 3, H, W]`` in [-1, 1] as the DiT's latents ``[B, 4 * channels, H / 16, W / 16]``."""
    parameter = next(vae.parameters())
    z = vae.encode(pixels.to(parameter.device, parameter.dtype)).mode()
    batch, channels, height, width = z.shape
    z = z.reshape(batch, channels, height // 2, 2, width // 2, 2).permute(0, 1, 3, 5, 2, 4)
    z = z.reshape(batch, channels * 4, height // 2, width // 2)
    stats = vae.bn
    mean = stats.running_mean.to(z).view(1, -1, 1, 1)
    std = (stats.running_var + vae.config.arch_config.batch_norm_eps).sqrt().to(z).view(1, -1, 1, 1)
    return (z - mean) / std


class MOSSImageReferenceStage(PipelineStage):
    """The request's reference images (``batch.condition_image``, which SGLang loads from ``image_path``), each
    encoded into ``batch.extra[REFERENCE_LATENTS]``; without one it does nothing.

    Raises:
        ValueError: From the call, if a reference is given to a model that takes none (``takes_reference``), or is
            off the 16-px grid.
    """

    def __init__(self, vae: AutoencoderKLFlux2, *, takes_reference: bool) -> None:
        super().__init__()
        self.vae = vae
        self.takes_reference = takes_reference

    def component_uses(self, server_args: ServerArgs, stage_name: str | None = None) -> list[ComponentUse]:
        return [ComponentUse(self._component_stage_name(stage_name), "vae")]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        images = batch.condition_image
        if images is None:
            return batch
        if not self.takes_reference:
            raise ValueError(
                "this MOSS-Image model takes no reference image: only a DiT trained with references does (its "
                "transformer/export.json says takes_reference)"
            )
        references: list[Image.Image] = images if isinstance(images, list) else [images]
        with self.use_declared_component(component_name="vae", module=self.vae) as vae:
            latents = [encode_latents(vae, image_pixels(image)) for image in references]
        batch.extra[REFERENCE_LATENTS] = latents
        return batch
