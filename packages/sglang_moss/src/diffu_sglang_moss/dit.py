"""SGLang's MOSS-Image DiT reading reference images as ``diffu_moss.transformer.MossTransformer`` does.

Each reference's clean latents become tokens after the prompt's, embedded as the drawn image's are, placed on the
reference axis of the rotary table at ``k + 1`` for the ``k``-th (the drawn image and the prompt at 0) with its own
grid centred, and modulated as at timestep 0 while the drawn image and the prompt are modulated at the flow timestep;
they join every joint block's attention and only the drawn image's tokens are predicted. No weight is added: a DiT
trained with references (``diffu-moss-train-ref``) reads them with the released model's weights' names.

Without references the forward is SGLang's own. With them it computes the blocks with SGLang's own pieces
(``_norm_rope``, ``_attention``, ``_situ``), but the modulation per token in plain torch, as SGLang's fused kernels
compute it bit for bit; sequence parallelism is not taken, as its shards assume the image and prompt alone.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from sglang.multimodal_gen.runtime.models.dits.moss_image import (
    MOSSImageConditioning,
    MOSSImageTransformer,
    _attention,
    _Block,
    _norm_rope,
    _prepare_rope,
    _situ,
    validate_skip_layers,
)
from torch import Tensor

MODULATIONS = 6  # shift, scale and gate of the attention half, then of the feed-forward half


def reference_positions(height: int, width: int, text_length: int, reference: int, device: torch.device) -> Tensor:
    """The 4-axis positions ``[height * width, 4]`` of a reference's tokens, row by row: (``reference``, the prompt's
    length, its row, its column), rows and columns centred on the reference's own grid, as the drawn image's are."""
    rows, cols = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    grid = torch.stack((torch.full_like(rows, reference), torch.full_like(rows, text_length), rows, cols), -1)
    grid = grid.flatten(0, 1).float()
    grid[:, 2] -= (height - 1) / 2
    grid[:, 3] -= (width - 1) / 2
    return grid


class DiffuMOSSImageTransformer(MOSSImageTransformer):
    """:class:`MOSSImageTransformer` whose forward also takes ``reference_latents``: reference images' clean latents,
    each ``[1 or B, latent_channels, h, w]`` on its own grid of any size (one row is shared by every sample).

    Raises:
        ValueError: From ``forward``, if a reference is not latents like the drawn image's (as many channels, one row
            or as many as the batch).
        NotImplementedError: From ``forward``, if references are given under sequence parallelism.
    """

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor | None = None,
        prepared_conditioning: MOSSImageConditioning | None = None,
        skip_layers: Sequence[int] = (),
        reference_latents: Sequence[Tensor] | None = None,
    ) -> Tensor:
        if not reference_latents:
            return super().forward(
                hidden_states,
                timestep,
                encoder_hidden_states,
                encoder_attention_mask,
                prepared_conditioning,
                skip_layers,
            )
        if self._sequence_parallel:
            raise NotImplementedError("reference images under sequence parallelism are not supported")
        skip = frozenset(validate_skip_layers(skip_layers, self.config.num_layers))
        batch, channels, height, width = hidden_states.shape
        references = [self._batched(reference, batch, channels) for reference in reference_latents]
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1).expand(batch)
        modulation, head = self._modulation(times)
        clean, _ = self._modulation(torch.zeros_like(times))
        if prepared_conditioning is None:
            prepared_conditioning = self.prepare_conditioning(
                encoder_hidden_states, encoder_attention_mask, height, width, self.video_embedder.weight.dtype
            )
        outputs = torch.empty_like(hidden_states, dtype=torch.float32)
        for group in prepared_conditioning.groups:
            indices = group.indices
            if indices is None:
                indices = torch.arange(batch, device=hidden_states.device)
            latents, drawn, last, still = (
                tensor.index_select(0, indices) for tensor in (hidden_states, modulation, head, clean)
            )
            refs = [reference.index_select(0, indices) for reference in references]
            sample = self._sample(latents, drawn, last, still, group.text, group.rope, refs, skip)
            outputs.index_copy_(0, indices, sample)
        return outputs

    @staticmethod
    def _batched(reference: Tensor, batch: int, channels: int) -> Tensor:
        """A reference's latents as one row per sample."""
        if reference.dim() != 4 or reference.shape[1] != channels or reference.shape[0] not in (1, batch):
            raise ValueError(
                f"a reference must be latents [1 or {batch}, {channels}, h, w] like the drawn image's, "
                f"not {list(reference.shape)}"
            )
        return reference.expand(batch, -1, -1, -1)

    def _modulation(self, times: Tensor) -> tuple[Tensor, Tensor]:
        """The joint blocks' modulation ``[B, 6, dim]`` and the head's ``[B, 2, dim]`` at ``times``, computed as
        SGLang's forward computes them (the sinusoid in float64, the embedding fed float32)."""
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        batch = times.shape[0]
        blocks = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype))
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype))
        return blocks.view(batch, MODULATIONS, -1), head.view(batch, 2, -1)

    def _embedded(self, latents: Tensor) -> Tensor:
        """Latents ``[B, latent_channels, h, w]`` as tokens ``[B, h * w, dim]``, row by row."""
        return self.video_embedder(latents.flatten(2).transpose(1, 2).to(self.video_embedder.weight.dtype).contiguous())

    def _reference_rope(self, grids: list[tuple[int, int]], text_length: int, like: Tensor) -> tuple[Tensor, Tensor]:
        """The rotary coefficients of the references' tokens, laid out as ``prepare_conditioning``'s (``like``'s
        batch and dtype)."""
        if self.rope.bands.dtype != torch.float32:
            self.rope.reset_bands()
        positions = torch.cat(
            [reference_positions(h, w, text_length, k + 1, like.device) for k, (h, w) in enumerate(grids)]
        )
        phase = positions.unsqueeze(-1) * self.rope.bands
        table = torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0).to(like.dtype)
        sin, cos = _prepare_rope(table)
        return sin.expand(like.shape[0], -1, -1).contiguous(), cos.expand(like.shape[0], -1, -1).contiguous()

    def _sample(
        self,
        latents: Tensor,
        modulation: Tensor,
        head: Tensor,
        clean: Tensor,
        text: Tensor,
        rope: tuple[Tensor, Tensor],
        references: list[Tensor],
        skip: frozenset[int],
    ) -> Tensor:
        """One group's velocity: the drawn image's tokens, the prompt's and the references' through the joint blocks,
        the references modulated by ``clean`` and the rest by ``modulation``."""
        batch, _, height, width = latents.shape
        image_tokens = height * width
        drawn = image_tokens + text.shape[1]
        x = torch.cat((self._embedded(latents), text, *(self._embedded(reference) for reference in references)), 1)
        grids = [(reference.shape[2], reference.shape[3]) for reference in references]
        extra = self._reference_rope(grids, text.shape[1], rope[0])
        rope = (torch.cat((rope[0], extra[0]), 1), torch.cat((rope[1], extra[1]), 1))
        by_token = torch.cat(
            (modulation[:, None].expand(-1, drawn, -1, -1), clean[:, None].expand(-1, x.shape[1] - drawn, -1, -1)), 1
        )
        for index, layer in enumerate(self.layers):
            if index not in skip:
                x = self._block(layer, x, rope, by_token)
        x = self.final_norm_video(x[:, :image_tokens].to(self.final_norm_video.weight.dtype))
        x = x * (1 + head[:, 1, None]) + head[:, 0, None]
        x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
        return x.transpose(1, 2).reshape(batch, -1, height, width).float()

    @staticmethod
    def _block(layer: _Block, x: Tensor, rope: tuple[Tensor, Tensor], modulation: Tensor) -> Tensor:
        """SGLang's joint block (``_Block``) with a modulation per token ``[B, tokens, 6, dim]``."""
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(2)
        dtype = layer.proj["q"].weight.dtype
        y = (layer.norm1(x) * (1 + scale_a) + shift_a).to(dtype)
        q = _norm_rope(layer.q_norm, layer.proj["q"](y), rope, layer.heads)
        k = _norm_rope(layer.k_norm, layer.proj["k"](y), rope, layer.heads)
        x = x + layer.proj["o"](_attention(q, k, layer.proj["v"](y), None, layer.heads, layer.attention)) * gate_a
        y = layer.norm2(x) * (1 + scale_f) + shift_f
        hidden, gate = layer.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + layer.proj["ffn_down"](_situ(hidden, gate)) * gate_f
