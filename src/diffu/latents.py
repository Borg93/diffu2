"""diffu's image latents: Qwen-Image VAE latents, normalized per channel, each 2x2 patch packed on channels.

The one place an image becomes the transformer's latents and back, for the inference blocks and the training cache
alike.
"""

from __future__ import annotations

import torch
from diffusers import AutoencoderKLQwenImage


def patchify(latents: torch.Tensor) -> torch.Tensor:
    """``(B, C, H, W)`` latents as ``(B, 4C, H/2, W/2)``: each 2x2 patch on channels in FLUX.3's order, channel
    ``c * 4 + dy * 2 + dx``."""
    batch, channels, height, width = latents.shape
    patches = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    return patches.permute(0, 1, 3, 5, 2, 4).reshape(batch, channels * 4, height // 2, width // 2)


def unpatchify(latents: torch.Tensor) -> torch.Tensor:
    """``(B, 4C, H, W)`` packed latents back to ``(B, C, 2H, 2W)``: the inverse of :func:`patchify`."""
    batch, channels, height, width = latents.shape
    patches = latents.reshape(batch, channels // 4, 2, 2, height, width)
    return patches.permute(0, 1, 4, 2, 5, 3).reshape(batch, channels // 4, height * 2, width * 2)


def _latent_stats(vae: AutoencoderKLQwenImage, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The VAE's per-channel latent mean and standard deviation, to broadcast over ``like``."""
    shape = (1, vae.config.z_dim, 1, 1)
    mean = torch.tensor(vae.config.latents_mean).view(shape).to(like.device, like.dtype)
    std = torch.tensor(vae.config.latents_std).view(shape).to(like.device, like.dtype)
    return mean, std


def encode_latents(vae: AutoencoderKLQwenImage, images: torch.Tensor) -> torch.Tensor:
    """Images ``(B, 3, H, W)`` in [-1, 1] as the transformer's latents ``(B, 4 * z_dim, H/16, W/16)``: the VAE's mode
    (a one-frame video to Qwen-Image's VAE), normalized per channel, patchified."""
    latents = vae.encode(images.unsqueeze(2)).latent_dist.mode()[:, :, 0]
    mean, std = _latent_stats(vae, latents)
    return patchify((latents - mean) / std)


def decode_latents(vae: AutoencoderKLQwenImage, latents: torch.Tensor) -> torch.Tensor:
    """The transformer's latents back to images ``(B, 3, H, W)`` in [-1, 1]: the inverse of :func:`encode_latents`
    up to the VAE's reconstruction."""
    latents = unpatchify(latents.to(vae.dtype))
    mean, std = _latent_stats(vae, latents)
    return vae.decode((latents * std + mean).unsqueeze(2), return_dict=False)[0][:, :, 0]
