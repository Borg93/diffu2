"""MOSS-Image's DiT, written for this project from the published design so the released weights load by name: 42
single-stream blocks of joint attention over the image's tokens (one per 16-px cell, 128 latent channels) followed by
the prompt's, all modulated by one shared projection of the timestep; the prompt enters through a linear map and two
text-only refiner blocks; positions are a 4-axis rotary table (reference, text index, row, column; the image's rows
and columns centred). Each sample runs at its own prompt length, so no padding reaches the attention.

Reference images (sample pages), as FLUX.3 Klein takes them: each one's clean latents become tokens after the prompt's,
embedded as the drawn image's are, placed on the reference axis (the drawn image and the prompt at 0, reference ``k``
at ``k + 1``) with their own grid centred, and modulated as at timestep 0 while the drawn image and the prompt are
modulated at the flow timestep; they join the joint attention and only the drawn image's tokens are predicted. The
released models were trained without them: only a DiT trained with references (``diffu_moss.train``) reads them.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from torch import Tensor, nn
from torch.nn import functional as F

TIME_PERIOD = 10_000
NORM_EPS = 1e-6
MODULATIONS = 6  # shift, scale and gate of the attention half, then of the feed-forward half


class RMSNorm(nn.Module):
    """Root-mean-square norm computed in fp32, scaled by its weight in the input's dtype."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(  # ty: ignore[invalid-mutable-override]  torch declares forward an attribute; Modules override it
        self,
        x: Tensor,
    ) -> Tensor:
        wide = x.float()
        return (wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + NORM_EPS)).to(x.dtype) * self.weight


def soft_gated(hidden: Tensor, gate: Tensor) -> Tensor:
    """MOSS's feed-forward activation: a gate capped at 4 (tanh) times its sigmoid, times the value capped at 25."""
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def rotated(x: Tensor, rope: Tensor) -> Tensor:
    """``x`` ``[B, L, heads, head_dim]`` turned by the rotary table ``[1, L, head_dim]`` (its sines, then its
    cosines), the halves of each head paired."""
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    first, second = x.chunk(2, -1)
    return x * cos + torch.cat((-second, first), -1) * sin


def joint_attention(q: Tensor, k: Tensor, v: Tensor, rope: Tensor, heads: int) -> Tensor:
    """Full attention over one sample's tokens, queries and keys turned by the rotary table."""
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = rotated(q.reshape(shape), rope).transpose(1, 2)
    k = rotated(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v).transpose(1, 2).flatten(2)


def grid_positions(height: int, width: int, text_length: int, reference: int, device: torch.device) -> Tensor:
    """The 4-axis positions ``[height * width, 4]`` of an image's tokens, row by row: (``reference``, the prompt's
    length, its row, its column), rows and columns centred on the image's own grid."""
    rows, cols = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    image = torch.stack((torch.full_like(rows, reference), torch.full_like(rows, text_length), rows, cols), -1)
    image = image.flatten(0, 1).float()
    image[:, 2] -= (height - 1) / 2
    image[:, 3] -= (width - 1) / 2
    return image


def token_positions(
    height: int,
    width: int,
    text_length: int,
    device: torch.device,
    references: Sequence[tuple[int, int]] = (),
) -> Tensor:
    """The 4-axis position of every token in the order the DiT holds them: the drawn image's (:func:`grid_positions`
    on the reference axis at 0), the prompt's at (0, their index, 0, 0), then each reference's, of grid ``references[k]``
    (rows, columns), on the reference axis at ``k + 1``."""
    text = torch.zeros(text_length, 4, device=device)
    text[:, 1] = torch.arange(text_length, device=device)
    images = [grid_positions(h, w, text_length, k + 1, device) for k, (h, w) in enumerate(references)]
    return torch.cat((grid_positions(height, width, text_length, 0, device), text, *images))


def rope_table(positions: Tensor, head_dim: int, base: float) -> Tensor:
    """The rotary table ``[1, L, head_dim]`` of ``positions``: ``head_dim / 8`` frequencies per axis (computed in fp32
    on the CPU), all the sines, then all the cosines."""
    bands = head_dim // 8
    frequencies = 1 / (base ** (torch.arange(bands, dtype=torch.float32) / bands))
    phase = positions.unsqueeze(-1) * frequencies.to(positions.device)
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


def time_features(timestep: Tensor, dim: int) -> Tensor:
    """The timestep's cosines and sines over ``dim / 2`` frequencies, computed in float64, returned in fp32."""
    half = dim // 2
    frequencies = TIME_PERIOD ** (-torch.arange(half, device=timestep.device, dtype=torch.float64) / half)
    phase = timestep.double()[:, None] * frequencies
    return torch.cat((phase.cos(), phase.sin()), -1).float()


class TextRefinerBlock(nn.Module):
    """A text-only transformer block over the projected prompt (biased linears, unmodulated)."""

    def __init__(self, dim: int, ffn_dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(  # ty: ignore[invalid-mutable-override]  torch declares forward an attribute; Modules override it
        self,
        x: Tensor,
        rope: Tensor,
    ) -> Tensor:
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(joint_attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(soft_gated(self.ffn_up(y), self.ffn_gate(y)))


class TextEmbedder(nn.Module):
    """The prompt's hidden states projected to the model's width, then refined by text-only blocks."""

    def __init__(self, text_dim: int, dim: int, ffn_dim: int, heads: int, depth: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextRefinerBlock(dim, ffn_dim, heads) for _ in range(depth)])

    def forward(  # ty: ignore[invalid-mutable-override]  torch declares forward an attribute; Modules override it
        self,
        text: Tensor,
        rope: Tensor,
    ) -> Tensor:
        x = self.proj(text.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x


def modulation_by_token(modulation: Tensor) -> tuple[Tensor, ...]:
    """The six modulations (shift, scale and gate of each half) to broadcast over a block's tokens: from the shared
    modulation ``[B, 6, dim]`` of every token, each ``[B, 1, dim]``; from a modulation per token ``[B, tokens, 6, dim]``,
    each ``[B, tokens, dim]``."""
    if modulation.dim() == 3:
        return tuple(part[:, None] for part in modulation.unbind(1))
    return modulation.unbind(2)


class JointBlock(nn.Module):
    """A single-stream block over the image's and the prompt's tokens (and any reference's), its attention and
    feed-forward halves shifted, scaled and gated by the time modulation: one shared by every token (``[B, 6, dim]``),
    or one per token (``[B, tokens, 6, dim]``: the references' at timestep 0)."""

    def __init__(self, dim: int, ffn_dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(  # ty: ignore[invalid-mutable-override]  torch declares forward an attribute; Modules override it
        self,
        x: Tensor,
        rope: Tensor,
        modulation: Tensor,
    ) -> Tensor:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation_by_token(modulation)
        dtype = self.proj["q"].weight.dtype
        y = (self.norm1(x) * (1 + scale_a) + shift_a).to(dtype)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](joint_attention(q, k, self.proj["v"](y), rope, self.heads)) * gate_a
        y = self.norm2(x) * (1 + scale_f) + shift_f
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](soft_gated(hidden, gate)) * gate_f


class MossTransformer(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """MOSS-Image's DiT. ``forward`` takes noisy latents ``[B, latent_channels, H, W]`` (one per 16-px cell), the flow
    timestep (sigma x 1000), the prompts' hidden states ``[B, S, text_dim]`` and their mask ``[B, S]``, and returns the
    predicted velocity (noise minus image) in the latents' shape. ``skip_layers`` leaves out joint blocks by index.

    ``references`` are reference images' clean latents, each ``[B, latent_channels, h, w]`` on its own grid of any
    size (row ``b`` is sample ``b``'s): their tokens follow the prompt's on the reference axis at ``k + 1``, modulated
    as at timestep 0, and are not predicted. Without references the forward is exactly the released model's.

    As diffusers' own transformers, it takes PEFT adapters (``add_adapter``, ``fuse_lora``) and checkpoints its joint
    blocks' activations while training once ``enable_gradient_checkpointing`` is called.

    Raises:
        ValueError: From the constructor, if ``dim`` does not split into heads whose width is a multiple of 8 or
            ``freq_dim`` is odd; from ``forward``, if a sample has no prompt token, or a reference is not a batch of
            latents like the drawn image's (as many samples, as many channels).
    """

    _keys_to_ignore_on_load_unexpected = [r"^rope\.bands$"]  # the rotary frequencies are recomputed in fp32
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        dim: int = 3072,
        ffn_dim: int = 8192,
        num_heads: int = 24,
        num_layers: int = 42,
        text_dim: int = 4096,
        freq_dim: int = 256,
        latent_channels: int = 128,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError("dim must divide into heads whose width is a multiple of 8, and freq_dim must be even")
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = TextEmbedder(text_dim, dim, ffn_dim, num_heads)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, MODULATIONS * dim)
        self.layers = nn.ModuleList([JointBlock(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)
        self.gradient_checkpointing = False

    def forward(
        self,
        latents: Tensor,
        timestep: Tensor,
        text: Tensor,
        text_mask: Tensor,
        skip_layers: frozenset[int] = frozenset(),
        references: Sequence[Tensor] | None = None,
    ) -> Tensor:
        batch, channels = latents.shape[:2]
        references = list(references or ())
        for reference in references:
            if reference.dim() != 4 or tuple(reference.shape[:2]) != (batch, channels):
                raise ValueError(
                    f"a reference must be latents [{batch}, {channels}, h, w] like the drawn image's, "
                    f"not {list(reference.shape)}"
                )
        dtype = self.video_embedder.weight.dtype
        times = torch.as_tensor(timestep, device=latents.device, dtype=torch.float32).reshape(-1).expand(batch)
        time = self._time_embedding(times)
        modulation = self.shared_block_modulation(time.to(dtype)).view(batch, MODULATIONS, -1)
        head = self.video_head_modulation(time.to(dtype)).view(batch, 2, -1)
        clean = None
        if references:  # clean latents: the modulation of timestep 0
            clean_time = self._time_embedding(torch.zeros_like(times))
            clean = self.shared_block_modulation(clean_time.to(dtype)).view(batch, MODULATIONS, -1)
        return torch.cat(
            [
                self._sample(
                    latents[k : k + 1],
                    text[k : k + 1, text_mask[k].bool()],
                    modulation[k : k + 1],
                    head[k : k + 1],
                    skip_layers,
                    [reference[k : k + 1] for reference in references],
                    None if clean is None else clean[k : k + 1],
                )
                for k in range(batch)
            ]
        )

    def _time_embedding(self, times: Tensor) -> Tensor:
        """The timesteps' embedding ``[B, dim]`` in fp32."""
        dtype = self.video_embedder.weight.dtype
        return self.time_embedding(time_features(times, self.config.freq_dim).to(dtype)).float()

    def _embedded(self, latents: Tensor) -> Tensor:
        """Latents ``[1, latent_channels, h, w]`` as tokens ``[1, h * w, dim]``, row by row."""
        return self.video_embedder(latents.flatten(2).transpose(1, 2).to(self.video_embedder.weight.dtype).contiguous())

    def _sample(
        self,
        latents: Tensor,
        text: Tensor,
        modulation: Tensor,
        head: Tensor,
        skip_layers: frozenset[int],
        references: list[Tensor],
        clean_modulation: Tensor | None,
    ) -> Tensor:
        _, _, height, width = latents.shape
        if text.shape[1] == 0:
            raise ValueError("every sample needs at least one prompt token")
        image_tokens, text_tokens = height * width, text.shape[1]
        video = self._embedded(latents)
        grids = [(reference.shape[2], reference.shape[3]) for reference in references]
        positions = token_positions(height, width, text_tokens, latents.device, grids)
        rope = rope_table(positions, self.config.dim // self.config.num_heads, self.config.rope_base).to(video.dtype)
        prompt = self.text_embedder(text, rope[:, image_tokens : image_tokens + text_tokens])
        x = torch.cat((video, prompt, *(self._embedded(reference) for reference in references)), 1)
        if clean_modulation is not None:
            drawn, clean = image_tokens + text_tokens, x.shape[1] - image_tokens - text_tokens
            modulation = torch.cat(
                (
                    modulation[:, None].expand(-1, drawn, -1, -1),
                    clean_modulation[:, None].expand(-1, clean, -1, -1),
                ),
                1,
            )
        for index, layer in enumerate(self.layers):
            if index in skip_layers:
                continue
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = self._gradient_checkpointing_func(layer, x, rope, modulation)
            else:
                x = layer(x, rope, modulation)
        x = self.final_norm_video(x[:, :image_tokens].to(self.final_norm_video.weight.dtype))
        x = x * (1 + head[:, 1, None]) + head[:, 0, None]
        x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
        return x.transpose(1, 2).reshape(1, -1, height, width)
