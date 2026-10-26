"""diffu's image latents: Qwen-Image VAE latents, normalized per channel, each 2x2 patch packed on channels.

The one place an image becomes the transformer's latents and back, for the inference blocks and the training cache
alike. A repo's VAE config holds the normalization (``latents_mean``, ``latents_std``): Qwen-Image's own, measured on
photos and video, or our pages' (``diffu-init --latent-stats``). :func:`renormalized` moves latents cached under one
normalization to another, so a cache trains a repo whose stats differ without being made again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import torch
from diffusers import AutoencoderKLQwenImage
from pydantic import BaseModel, ConfigDict, model_validator

from diffu.repo import ModelIndex

PATCH_CHANNELS = 4  # a 2x2 patch packed on channels: the VAE's channel c is packed channels 4c to 4c + 3


class LatentStats(BaseModel):
    """A VAE's latent mean and standard deviation per channel, which diffu's latents are normalized with: the VAE
    config's ``latents_mean`` and ``latents_std``.

    Raises:
        ValueError: If the two differ in length, or a standard deviation is not positive.
    """

    model_config = ConfigDict(frozen=True)

    latents_mean: tuple[float, ...]
    latents_std: tuple[float, ...]

    @model_validator(mode="after")
    def _one_pair_per_channel(self) -> Self:
        if len(self.latents_mean) != len(self.latents_std):
            raise ValueError(f"{len(self.latents_mean)} means for {len(self.latents_std)} standard deviations")
        if min(self.latents_std, default=1.0) <= 0:
            raise ValueError(f"a standard deviation is not positive: {self.latents_std}")
        return self


def vae_stats(model: Path) -> LatentStats:
    """The stats the diffu repo at ``model`` normalizes its latents with: its VAE's config, wherever its index loads the
    VAE from."""
    vae = ModelIndex.read(model).source("vae")
    config = AutoencoderKLQwenImage.load_config(vae.repo, revision=vae.revision, subfolder=vae.subfolder)
    return LatentStats.model_validate(dict(config))


def renormalized(latents: torch.Tensor, source: LatentStats, target: LatentStats) -> torch.Tensor:
    """Packed latents ``(..., 4 * channels, H, W)`` normalized with ``source``, normalized with ``target`` instead: what
    :func:`encode_latents` makes of the same image with a VAE whose config holds ``target``.

    Raises:
        ValueError: If the latents' packed channels are not four for each channel of both stats.
    """
    channels = latents.shape[-3]
    if not channels == PATCH_CHANNELS * len(source.latents_mean) == PATCH_CHANNELS * len(target.latents_mean):
        raise ValueError(
            f"latents of {channels} packed channels, stats of {len(source.latents_mean)} and "
            f"{len(target.latents_mean)} channels"
        )
    source_mean, source_std, target_mean, target_std = (
        torch.tensor(values, dtype=torch.float64).repeat_interleave(PATCH_CHANNELS).view(-1, 1, 1)
        for values in (source.latents_mean, source.latents_std, target.latents_mean, target.latents_std)
    )
    raw = latents.double() * source_std + source_mean
    return ((raw - target_mean) / target_std).to(latents.dtype)


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
