"""16-ch f8 VAE wrapper (diffusers).

PICK = Qwen-Image VAE 1.0 (f8, 16-ch, text-rich decoder, Apache-2.0), frozen. Latent normalization is read
from the loaded VAE's own config (scalar scaling/shift for SD-style VAEs, per-channel mean/std for Qwen/Wan)
so we never hardcode the wrong constants. Includes the round-trip reconstruction-CER gate.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, AutoencoderKLQwenImage
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageCausalConv3d, QwenImageUpsample
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from ..config import VAEConfig
from .backbone import configure_compile, ensure_compile_config
from .metrics import cer


class _SingleFrameCausalConv3d(QwenImageCausalConv3d):
    """A causal 3D conv that runs a single frame as the 2D conv it amounts to.

    For T=1 the causal conv pads time with ``k_t - 1`` zero frames in front, so only the LAST temporal
    kernel slice ever touches real data — the full conv3d spends ``k_t``x (3x) the FLOPs multiplying zeros.
    This runs ``conv2d`` with that slice instead: same weights and state-dict keys, rounding-level identical
    output, gradients reach the same (only non-zero-grad) slice. Anything else uses the stock path.
    """

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        k_t = self.kernel_size[0]
        if (
            cache_x is None
            and x.shape[2] == 1
            and self._padding[4] == k_t - 1
            and self.stride[0] == 1
            and self.dilation[0] == 1
        ):
            frame = F.pad(x.squeeze(2), self._padding[:4])  # (w_left, w_right, h_top, h_bottom)
            out = F.conv2d(
                frame, self.weight[:, :, -1], self.bias, self.stride[1:], 0, self.dilation[1:], self.groups
            )
            return out.unsqueeze(2)
        return super().forward(x, cache_x)


class _NativeDtypeUpsample(QwenImageUpsample):
    """``QwenImageUpsample`` without its fp32 round trip for nearest modes (a pure copy: bit-identical)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode in ("nearest", "nearest-exact"):
            return nn.Upsample.forward(self, x)
        return super().forward(x)


def _load_vae(cfg: VAEConfig) -> AutoencoderKL | AutoencoderKLQwenImage:
    """The diffusers VAE: the Qwen-Image (Wan/video-derived) class when ``cfg.video_vae``, else ``AutoencoderKL``."""
    cls = AutoencoderKLQwenImage if cfg.video_vae else AutoencoderKL
    return cls.from_pretrained(cfg.pretrained, subfolder=cfg.subfolder)


class VAEWrapper(nn.Module):
    """Frozen VAE with latent (de)normalization and a recon-CER gate."""

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vae = _load_vae(cfg)
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

        if self.video_vae:
            for module in self.vae.modules():  # same parameters, faster single-frame forwards
                if type(module) is QwenImageCausalConv3d:
                    module.__class__ = _SingleFrameCausalConv3d
                elif type(module) is QwenImageUpsample:
                    module.__class__ = _NativeDtypeUpsample

        self.vae.requires_grad_(False)
        self._compiled = False  # compile_codec() ran: encode/decode re-apply the per-thread compile settings
        self._compile_mode: str | None = None

    def compile_codec(self, mode: str | None = None) -> None:
        """``torch.compile`` the VAE encode + decode (``fullgraph``, dynamic width) — diffusers' SD3 docs compile
        the decode for inference; the encode runs every training step. In place: state-dict keys unchanged."""
        configure_compile(mode)
        self._compiled, self._compile_mode = True, mode
        self.vae.encoder.compile(fullgraph=True, dynamic=True, mode=mode)
        self.vae.decoder.compile(fullgraph=True, dynamic=True, mode=mode)

    def _direct_single_frame(self) -> bool:
        """Images are one frame: call the encoder/decoder directly instead of diffusers' frame loop.

        ``AutoencoderKLQwenImage._encode/_decode`` loop over frame chunks with a per-conv frame cache (a full
        activation ``.clone()`` per cached conv). With one frame every cached conv sees no previous frame, so
        the direct call is bit-identical and skips the clones. Slicing / tiling keep the stock path.
        """
        return self.video_vae and not (
            getattr(self.vae, "use_slicing", False) or getattr(self.vae, "use_tiling", False)
        )

    def _single_frame_encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        return DiagonalGaussianDistribution(self.vae.quant_conv(self.vae.encoder(x)))

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
                MODE/mean (``False``, deterministic — e.g. an inference reference line).

        Returns:
            Normalized latent ``[B, C, H/8, W/8]`` (encoder frozen). For video VAEs (Qwen/Wan) a
            ``T=1`` frame dim is added then squeezed so the latent stays 4D for the DiT.
        """
        if self._compiled:  # the compile settings are per thread: apply them in this one too
            ensure_compile_config(self._compile_mode)
        if self.video_vae:
            x = x.unsqueeze(2)  # [B, 3, 1, H, W]
        dist = self._single_frame_encode(x) if self._direct_single_frame() else self.vae.encode(x).latent_dist
        z = dist.sample() if sample else dist.mode()
        if self.video_vae:
            z = z.squeeze(2)  # back to 4D [B, C, h, w]
        return self._normalize(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a normalized 4D latent to an image ``[B, 3, H, W]`` in ``[-1, 1]``."""
        if self._compiled:
            ensure_compile_config(self._compile_mode)
        z = self._denormalize(z)
        if self.video_vae:
            z = z.unsqueeze(2)  # [B, C, 1, h, w]
        if self._direct_single_frame():
            img = self.vae.decoder(self.vae.post_quant_conv(z)).clamp(-1.0, 1.0)
        else:
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
        # Chunked: the Qwen/Wan VAE's 3D convs allocate >80 GB if the whole set is round-tripped at
        # once (the gate OOMed on 500 lines). Encode/decode + recognize in small batches instead.
        raw_preds: list[str] = []
        recon_preds: list[str] = []
        for i in range(0, images.shape[0], batch_size):
            chunk = images[i : i + batch_size]
            recon = self.decode(self.encode(chunk))
            raw_preds.extend(recognizer(chunk))
            recon_preds.extend(recognizer(recon))
        raw_cer = cer(raw_preds, transcriptions)
        recon_cer = cer(recon_preds, transcriptions)
        return {
            "raw_cer": raw_cer,
            "recon_cer": recon_cer,
            "passed": (recon_cer - raw_cer) <= self.cfg.recon_cer_gate,
        }
