"""MOSS-Image's DiT, written for this project from the published design so the released weights load by name: 42
single-stream blocks of joint attention over the image's tokens (one per 16-px cell, 128 latent channels) followed by
the prompt's, all modulated by one shared projection of the timestep; the prompt enters through a linear map and two
text-only refiner blocks; positions are a 4-axis rotary table (reference, text index, row, column; the image's rows
and columns centred). Each sample runs at its own prompt length, so no padding reaches the attention."""

from __future__ import annotations

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch import Tensor, nn
from torch.nn import functional as F

TIME_PERIOD = 10_000
NORM_EPS = 1e-6


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


def token_positions(height: int, width: int, text_length: int, device: torch.device) -> Tensor:
    """The 4-axis position of every token ``[height * width + text_length, 4]``: an image token at (0, the prompt's
    length, its row, its column), rows and columns centred; the prompt's tokens at (0, their index, 0, 0)."""
    rows, cols = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    image = torch.stack((torch.zeros_like(rows), torch.full_like(rows, text_length), rows, cols), -1)
    image = image.flatten(0, 1).float()
    image[:, 2] -= (height - 1) / 2
    image[:, 3] -= (width - 1) / 2
    text = torch.zeros(text_length, 4, device=device)
    text[:, 1] = torch.arange(text_length, device=device)
    return torch.cat((image, text))


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


class JointBlock(nn.Module):
    """A single-stream block over the image's and the prompt's tokens, its attention and feed-forward halves
    shifted, scaled and gated by the shared time modulation."""

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
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        dtype = self.proj["q"].weight.dtype
        y = (self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]).to(dtype)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](joint_attention(q, k, self.proj["v"](y), rope, self.heads)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](soft_gated(hidden, gate)) * gate_f[:, None]


class MossTransformer(ModelMixin, ConfigMixin):
    """MOSS-Image's DiT. ``forward`` takes noisy latents ``[B, latent_channels, H, W]`` (one per 16-px cell), the flow
    timestep (sigma x 1000), the prompts' hidden states ``[B, S, text_dim]`` and their mask ``[B, S]``, and returns the
    predicted velocity (noise minus image) in the latents' shape. ``skip_layers`` leaves out joint blocks by index.

    Raises:
        ValueError: From the constructor, if ``dim`` does not split into heads whose width is a multiple of 8 or
            ``freq_dim`` is odd; from ``forward``, if a sample has no prompt token.
    """

    _keys_to_ignore_on_load_unexpected = [r"^rope\.bands$"]  # the rotary frequencies are recomputed in fp32

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
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([JointBlock(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def forward(
        self,
        latents: Tensor,
        timestep: Tensor,
        text: Tensor,
        text_mask: Tensor,
        skip_layers: frozenset[int] = frozenset(),
    ) -> Tensor:
        batch = latents.shape[0]
        dtype = self.video_embedder.weight.dtype
        times = torch.as_tensor(timestep, device=latents.device, dtype=torch.float32).reshape(-1).expand(batch)
        time = self.time_embedding(time_features(times, self.config.freq_dim).to(dtype)).float()
        modulation = self.shared_block_modulation(time.to(dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(dtype)).view(batch, 2, -1)
        return torch.cat(
            [
                self._sample(
                    latents[k : k + 1],
                    text[k : k + 1, text_mask[k].bool()],
                    modulation[k : k + 1],
                    head[k : k + 1],
                    skip_layers,
                )
                for k in range(batch)
            ]
        )

    def _sample(
        self, latents: Tensor, text: Tensor, modulation: Tensor, head: Tensor, skip_layers: frozenset[int]
    ) -> Tensor:
        _, _, height, width = latents.shape
        if text.shape[1] == 0:
            raise ValueError("every sample needs at least one prompt token")
        image_tokens = height * width
        video = self.video_embedder(
            latents.flatten(2).transpose(1, 2).to(self.video_embedder.weight.dtype).contiguous()
        )
        positions = token_positions(height, width, text.shape[1], latents.device)
        rope = rope_table(positions, self.config.dim // self.config.num_heads, self.config.rope_base).to(video.dtype)
        x = torch.cat((video, self.text_embedder(text, rope[:, image_tokens:])), 1)
        for index, layer in enumerate(self.layers):
            if index not in skip_layers:
                x = layer(x, rope, modulation)
        x = self.final_norm_video(x[:, :image_tokens].to(self.final_norm_video.weight.dtype))
        x = x * (1 + head[:, 1, None]) + head[:, 0, None]
        x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
        return x.transpose(1, 2).reshape(1, -1, height, width)
