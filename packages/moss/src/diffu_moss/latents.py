"""MOSS-Image's latents: the FLUX.2 VAE's, each 2x2 patch packed on channels, normalized by the VAE's batch-norm
running statistics.

The one place an image becomes the DiT's latents and back: the decode step decodes with :func:`decode_latents`, a
drawn crop is encoded with :func:`encode_latents` for a line repair (``diffu_page.pipeline.page_latents``), and a
reference image is read by :func:`image_pixels` and encoded by :func:`encode_latents` alike in drawing (the reference
encode step) and in training (``diffu_moss.train.cache``).
"""

from __future__ import annotations

import torch
from diffusers import AutoencoderKLFlux2
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from torch import Tensor

PIXELS_PER_LATENT = 16  # 8x VAE downsampling, then 2x2 patches


def image_pixels(image: Image.Image) -> Tensor:
    """An image on the latent grid as the VAE encodes it: ``[1, 3, H, W]`` in [-1, 1], never resized.

    Raises:
        ValueError: If a side is not a multiple of 16 px (``diffu_page``'s ``ReferenceGridStep`` and its training
            pairs put a reference on the grid; resizing here would read it otherwise than training did).
    """
    if image.width % PIXELS_PER_LATENT or image.height % PIXELS_PER_LATENT:
        raise ValueError(f"the image's size {image.size} is not on the {PIXELS_PER_LATENT}-px latent grid")
    pixels = VaeImageProcessor.numpy_to_pt(VaeImageProcessor.pil_to_numpy(image.convert("RGB")))
    return VaeImageProcessor.normalize(pixels)


def _batch_norm(vae: AutoencoderKLFlux2, like: Tensor) -> tuple[Tensor, Tensor]:
    """The VAE's batch-norm mean and standard deviation per packed channel, to broadcast over ``like``."""
    stats = vae.bn
    mean = stats.running_mean.to(like).view(1, -1, 1, 1)
    std = (stats.running_var + vae.config.batch_norm_eps).sqrt().to(like).view(1, -1, 1, 1)
    return mean, std


def encode_latents(vae: AutoencoderKLFlux2, pixels: Tensor) -> Tensor:
    """Images ``[B, 3, H, W]`` in [-1, 1] (``H`` and ``W`` multiples of 16) as the DiT's latents
    ``[B, 4 * latent_channels, H / 16, W / 16]``: the VAE's mode, patched 2x2, normalized by its batch-norm
    statistics. The inverse of :func:`decode_latents` up to the VAE's reconstruction."""
    with torch.inference_mode():
        z = vae.encode(pixels.to(vae.device, vae.dtype)).latent_dist.mode()
    batch, channels, height, width = z.shape
    z = z.reshape(batch, channels, height // 2, 2, width // 2, 2).permute(0, 1, 3, 5, 2, 4)
    z = z.reshape(batch, channels * 4, height // 2, width // 2)
    mean, std = _batch_norm(vae, z)
    return (z - mean) / std


def decode_latents(vae: AutoencoderKLFlux2, latents: Tensor) -> Tensor:
    """The DiT's latents ``[B, 4 * latent_channels, h, w]`` as images ``[B, 3, 16h, 16w]`` in [-1, 1]: un-normalized
    by the VAE's batch-norm statistics, un-patched 2x2, decoded."""
    z = latents.to(vae.dtype)
    mean, std = _batch_norm(vae, z)
    z = z * std + mean
    batch, channels, height, width = z.shape
    z = z.reshape(batch, channels // 4, 2, 2, height, width).permute(0, 1, 4, 2, 5, 3)
    z = z.reshape(batch, channels // 4, 2 * height, 2 * width)
    with torch.inference_mode():
        return vae.decode(z, return_dict=False)[0].clamp(-1, 1)
