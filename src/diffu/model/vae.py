"""16-ch f8 VAE wrapper (diffusers).

PICK = Qwen-Image VAE 1.0 (f8, 16-ch, text-rich decoder, Apache-2.0); FLUX.2 VAE is an Apache
fallback. Encoder frozen; decoder fine-tunable on Swedish ink. Latent normalization is read from
the loaded VAE's own config (scalar scaling/shift for FLUX/SD, per-channel mean/std for Qwen/Wan)
so we never hardcode the wrong constants. Includes the round-trip reconstruction-CER gate.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from diffusers import AutoencoderKL

from ..config import VAEConfig

try:  # Qwen-Image uses a dedicated VAE class in recent diffusers
    from diffusers import AutoencoderKLQwenImage
except ImportError:  # older diffusers / not installed locally
    AutoencoderKLQwenImage = None


def _load_vae(pretrained: str, subfolder: str) -> AutoencoderKL:
    """Load the latent VAE, falling back to the Qwen-Image (video-derived) VAE class.

    ``AutoencoderKL.from_pretrained`` fails on the Qwen-Image checkpoint because it declares a
    different model class; we catch the load/config error and retry with the dedicated class.

    Args:
        pretrained: HF repo id (e.g. ``"Qwen/Qwen-Image"``).
        subfolder: Subfolder holding the VAE weights (e.g. ``"vae"``).

    Returns:
        The instantiated diffusers VAE module.
    """
    try:
        return AutoencoderKL.from_pretrained(pretrained, subfolder=subfolder)
    except (OSError, ValueError, KeyError):
        # Class/config mismatch (Qwen is a Wan/video-derived VAE) or a load error -> retry.
        if AutoencoderKLQwenImage is not None:
            return AutoencoderKLQwenImage.from_pretrained(pretrained, subfolder=subfolder)
        raise


class VAEWrapper(nn.Module):
    """Frozen-encoder / trainable-decoder VAE with latent (de)normalization and a recon-CER gate."""

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vae = _load_vae(cfg.pretrained, cfg.subfolder)
        self.latent_channels = cfg.latent_channels
        self.video_vae = cfg.video_vae  # Qwen/Wan VAE wants 5D [B, C, T, H, W]

        c = self.vae.config
        # Two normalization conventions across modern VAEs:
        #   FLUX/SD : (z - shift_factor) * scaling_factor           (scalars)
        #   Qwen/Wan: (z - latents_mean) / latents_std              (per-channel)
        self._per_channel = getattr(c, "latents_mean", None) is not None
        if self._per_channel:
            self.register_buffer("lat_mean", torch.tensor(c.latents_mean).view(1, -1, 1, 1))
            self.register_buffer("lat_std", torch.tensor(c.latents_std).view(1, -1, 1, 1))
        else:
            self.scaling = getattr(c, "scaling_factor", 1.0)
            self.shift = getattr(c, "shift_factor", None) or 0.0

        self.vae.requires_grad_(False)
        if cfg.finetune_decoder_only:
            self.vae.decoder.requires_grad_(True)
            if getattr(self.vae, "post_quant_conv", None) is not None:
                self.vae.post_quant_conv.requires_grad_(True)

    def _normalize(self, z: torch.Tensor) -> torch.Tensor:
        if self._per_channel:
            return (z - self.lat_mean) / self.lat_std
        return (z - self.shift) * self.scaling

    def _denormalize(self, z: torch.Tensor) -> torch.Tensor:
        if self._per_channel:
            return z * self.lat_std + self.lat_mean
        return z / self.scaling + self.shift

    @torch.no_grad()
    def encode(self, x: torch.Tensor, *, sample: bool = True) -> torch.Tensor:
        """Encode images to a normalized 4D latent.

        Args:
            x: Images ``[B, 3, H, W]`` in ``[-1, 1]``.
            sample: Draw from the posterior (``True``, training default — one reparam sample) or return its
                MODE/mean (``False``). Use ``sample=False`` to precompute a DETERMINISTIC latent cache
                (data/latent_cache.py); the posterior std is tiny for an f8 VAE and flow matching injects its
                own noise, so the mode is the conventional, behavior-safe cache choice.

        Returns:
            Normalized latent ``[B, C, H/8, W/8]`` (encoder frozen). For video VAEs (Qwen/Wan) a
            ``T=1`` frame dim is added then squeezed so the latent stays 4D for the DiT.
        """
        if self.video_vae:
            x = x.unsqueeze(2)  # [B, 3, 1, H, W]
        dist = self.vae.encode(x).latent_dist
        z = dist.sample() if sample else dist.mode()
        if self.video_vae:
            z = z.squeeze(2)  # back to 4D [B, C, h, w]
        return self._normalize(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a normalized 4D latent to an image ``[B, 3, H, W]`` in ``[-1, 1]`` (decoder trainable)."""
        z = self._denormalize(z)
        if self.video_vae:
            z = z.unsqueeze(2)  # [B, C, 1, h, w]
        img = self.vae.decode(z).sample
        if self.video_vae:
            img = img.squeeze(2)  # back to [B, 3, H, W]
        return img

    @torch.no_grad()
    def reconstruction_cer_gate(
        self,
        images: torch.Tensor,
        transcriptions: list[str],
        recognizer: Callable[[torch.Tensor], list[str]],
        batch_size: int = 8,
    ) -> dict[str, float | bool]:
        """Round-trip real lines (encode -> decode) and measure HTR CER on the reconstruction.

        Must be within ``cfg.recon_cer_gate`` of the raw-image CER before any diffusion training
        begins (otherwise the latent is dropping diacritics).

        Args:
            images: Real line images ``[B, 3, H, W]`` in ``[-1, 1]``.
            transcriptions: Ground-truth strings, one per image.
            recognizer: Callable mapping a batch of images to predicted strings.
            batch_size: Lines per VAE round-trip + recognizer call (chunked to bound GPU memory).

        Returns:
            ``{"raw_cer", "recon_cer", "passed"}``.
        """
        from .metrics import cer as _cer

        # Chunked: the Qwen/Wan VAE's 3D convs allocate >80 GB if the whole set is round-tripped at
        # once (the gate OOMed on 500 lines). Encode/decode + recognize in small batches instead.
        raw_preds: list[str] = []
        recon_preds: list[str] = []
        for i in range(0, images.shape[0], batch_size):
            chunk = images[i : i + batch_size]
            recon = self.decode(self.encode(chunk))
            raw_preds.extend(recognizer(chunk))
            recon_preds.extend(recognizer(recon))
        raw_cer = _cer(raw_preds, transcriptions)
        recon_cer = _cer(recon_preds, transcriptions)
        return {
            "raw_cer": raw_cer,
            "recon_cer": recon_cer,
            "passed": (recon_cer - raw_cer) <= self.cfg.recon_cer_gate,
        }
