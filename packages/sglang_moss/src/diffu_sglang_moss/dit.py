"""SGLang's MOSS-Image DiT reading reference images as ``diffu_moss.transformer.MossTransformer`` does.

Each reference's clean latents become tokens after the prompt's, embedded as the drawn image's are, placed on the
reference axis of the rotary table at ``k + 1`` for the ``k``-th (the drawn image and the prompt at 0) with its own
grid centred (``over``) or its rows lifted above the drawn image's (``above``, as
``diffu_moss.transformer.token_positions`` lays a sample page), and modulated as at timestep 0 while the drawn image and the prompt are modulated at the flow timestep;
they join every joint block's attention and only the drawn image's tokens are predicted. No weight is added: a DiT
trained with references (``diffu-moss-train-ref``) reads them with the released model's weights' names.

Without references the forward is SGLang's own. With them it computes the blocks with SGLang's own pieces
(``_norm_modulate``, ``_norm_rope``, ``_attention``, ``residual_gate_add``, ``_situ``), the modulated ones on the two
parts of the tokens that share a modulation (:class:`TokenModulation`), so SGLang's fused kernels run on each;
sequence parallelism is not taken, as its shards assume the image and prompt alone.

Cache-DiT (SGLang's step cache, a request's ``enable_cache_dit``) reads a model's blocks as one list of modules that
each take the hidden states and return them. :class:`JointLayer` is a joint block in that shape (it also skips itself
in a pass that leaves its index out, which the forward's own loop did by index), ``joint_layers`` the list, registered
with SGLang's Cache-DiT integration below; once the cache is mounted every request, with a reference or without, runs
through them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
    _CUSTOM_BLOCK_ADAPTER_SPECS,
    CustomBlockAdapterSpec,
    ForwardPattern,
)
from sglang.multimodal_gen.runtime.models.dits.moss_image import (
    MOSSImageConditioning,
    MOSSImageTransformer,
    _attention,
    _Block,
    _norm_modulate,
    _norm_rope,
    _prepare_rope,
    _RMSNorm,
    _rotate_native,
    _situ,
    _situ_native,
    residual_gate_add,
    validate_skip_layers,
)
from torch import Tensor, nn

from diffu_sglang_moss.reference import ReferencePlace

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


def placed_positions(
    grids: Sequence[tuple[int, int]], text_length: int, height: int, place: ReferencePlace, device: torch.device
) -> list[Tensor]:
    """Each reference's positions (:func:`reference_positions`, the ``k``-th on the reference axis at ``k + 1``) by
    ``place`` against a drawn image of ``height`` latent rows: ``over``, centre on centre; ``above``, its rows moved
    up so its last lies one above the drawn image's first, several stacked upwards in order, as
    ``diffu_moss.transformer.token_positions`` lays them."""
    placed, lifted = [], 0
    for k, (rows, cols) in enumerate(grids):
        positions = reference_positions(rows, cols, text_length, k + 1, device)
        if place == "above":
            positions[:, 2] -= (height + rows) / 2 + lifted
            lifted += rows
        placed.append(positions)
    return placed


def _traced_norm_rope(norm: _RMSNorm, x: Tensor, rope: tuple[Tensor, Tensor], heads: int) -> Tensor:
    """``_norm_rope``'s plain path (``_rotate_native`` of ``forward_native``), for torch.compile to trace."""
    return _rotate_native(norm.forward_native(x.unflatten(-1, (heads, -1))), rope).flatten(2)


class TokenModulation(NamedTuple):
    """The joint blocks' modulation of one group's tokens: ``image`` ``[B, 6, dim]`` (the flow timestep's) for the
    first ``drawn`` tokens (the drawn image's and the prompt's), ``clean`` ``[B, 6, dim]`` (timestep 0's) for the rest
    (the references'). A named tuple, not a model: it is handed to every block of every step, and torch.compile reads
    it as a tuple."""

    drawn: int
    image: Tensor
    clean: Tensor


def _segments(x: Tensor, modulation: TokenModulation) -> list[tuple[Tensor, Tensor]]:
    """``x`` ``[B, tokens, ...]`` cut where the modulation changes, each part contiguous (SGLang's fused kernels take
    no other) with its modulation ``[B, 6, dim]``; one part when there are no references."""
    drawn = modulation.drawn
    if drawn >= x.shape[1]:
        return [(x.contiguous(), modulation.image)]
    return [(x[:, :drawn].contiguous(), modulation.image), (x[:, drawn:].contiguous(), modulation.clean)]


def _joined(parts: list[Tensor]) -> Tensor:
    return parts[0] if len(parts) == 1 else torch.cat(parts, 1)


def _norm_modulated(norm: _RMSNorm, x: Tensor, modulation: TokenModulation, half: int) -> Tensor:
    """``norm(x) * (1 + scale) + shift`` of the attention half (``half`` 0) or the feed-forward half (3), by SGLang's
    fused RMSNorm + modulation (``_norm_modulate``) on each part of :func:`_segments`."""
    return _joined(
        [_norm_modulate(norm, part, mod[:, half + 1], mod[:, half]) for part, mod in _segments(x, modulation)]
    )


def _gated_added(x: Tensor, update: Tensor, modulation: TokenModulation, half: int) -> Tensor:
    """``x + update * gate`` of the attention half (``half`` 0) or the feed-forward half (3), by SGLang's fused
    ``residual_gate_add`` on each part of :func:`_segments`."""
    updates = [part for part, _ in _segments(update, modulation)]
    return _joined(
        [
            residual_gate_add(part, change, mod[:, half + 2, None])
            for (part, mod), change in zip(_segments(x, modulation), updates, strict=True)
        ]
    )


QKV = "_diffu_qkv"  # the attribute of a block that holds its Q, K and V weights as one [3 * dim, dim]


def qkv_weight(layer: _Block) -> Tensor:
    """The block's Q, K and V weights as one ``[3 * dim, dim]``, built once: the three projections' weights become
    views of its rows, so it costs no memory. Built again if they no longer are (moved, offloaded, reloaded)."""
    weights = [layer.proj[name].weight for name in ("q", "k", "v")]
    held = getattr(layer, QKV, None)
    if held is None or any(
        weight.data_ptr() != part.data_ptr() for weight, part in zip(weights, held.chunk(3), strict=True)
    ):
        held = torch.cat([weight.detach() for weight in weights])
        for weight, part in zip(weights, held.chunk(3), strict=True):
            weight.data = part
        setattr(layer, QKV, held)
    return held


def _qkv(layer: _Block, y: Tensor, fused: bool) -> tuple[Tensor, Tensor, Tensor]:
    """The block's queries, keys and values of ``y``: three GEMMs, or with ``fused`` one on :func:`qkv_weight`, its
    output split (the queries and keys copied contiguous, as SGLang's fused RMSNorm + RoPE takes no other)."""
    if not fused:
        return layer.proj["q"](y), layer.proj["k"](y), layer.proj["v"](y)
    q, k, v = nn.functional.linear(y, qkv_weight(layer)).chunk(3, -1)
    return q.contiguous(), k.contiguous(), v


def joint_block(
    layer: _Block, x: Tensor, rope: tuple[Tensor, Tensor], modulation: TokenModulation, fused_qkv: bool = False
) -> Tensor:
    """SGLang's joint block (``_Block``) with the tokens modulated in two parts (:class:`TokenModulation`).

    Each part runs SGLang's fused RMSNorm + modulation and gated residual (``_norm_modulate``, ``residual_gate_add``),
    whose fast paths take one modulation row ``[B, dim]`` a sample, and the parts are joined again: on CPU, or where a
    fast path does not apply, these are the plain torch ops SGLang's kernels reproduce bit for bit. With
    ``fused_qkv`` the queries, keys and values come from one GEMM (:func:`qkv_weight`), equal to the three's within
    rounding, not bit for bit.

    Under torch.compile it runs the plain paths of SGLang's fused kernels, and three GEMMs: their bit-exact gates
    compare shapes symbolically, which PyTorch's shape solver fails on ("BooleanAtom not allowed in this context"),
    and Inductor fuses the plain paths itself.
    """
    if torch.compiler.is_compiling():
        return _traced_joint_block(layer, x, rope, modulation)
    dtype = layer.proj["q"].weight.dtype
    y = _norm_modulated(layer.norm1, x, modulation, 0).to(dtype)
    q, k, v = _qkv(layer, y, fused_qkv)
    q = _norm_rope(layer.q_norm, q, rope, layer.heads)
    k = _norm_rope(layer.k_norm, k, rope, layer.heads)
    attended = layer.proj["o"](_attention(q, k, v, None, layer.heads, layer.attention))
    x = _gated_added(x, attended, modulation, 0)
    y = _norm_modulated(layer.norm2, x, modulation, 3)
    hidden, gate = layer.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
    return _gated_added(x, layer.proj["ffn_down"](_situ(hidden, gate)), modulation, 3)


def _by_token(modulation: TokenModulation, tokens: int) -> Tensor:
    """The modulation per token ``[B, tokens, 6, dim]``."""
    image, clean, drawn = modulation.image, modulation.clean, modulation.drawn
    if drawn >= tokens:
        return image[:, None].expand(-1, tokens, -1, -1)
    return torch.cat((image[:, None].expand(-1, drawn, -1, -1), clean[:, None].expand(-1, tokens - drawn, -1, -1)), 1)


def _traced_joint_block(layer: _Block, x: Tensor, rope: tuple[Tensor, Tensor], modulation: TokenModulation) -> Tensor:
    """:func:`joint_block` by the plain paths alone, the modulation laid out per token."""
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = _by_token(modulation, x.shape[1]).unbind(2)
    dtype = layer.proj["q"].weight.dtype
    y = (layer.norm1.forward_native(x) * (1 + scale_a) + shift_a).to(dtype)
    q = _traced_norm_rope(layer.q_norm, layer.proj["q"](y), rope, layer.heads)
    k = _traced_norm_rope(layer.k_norm, layer.proj["k"](y), rope, layer.heads)
    x = x + layer.proj["o"](_attention(q, k, layer.proj["v"](y), None, layer.heads, layer.attention)) * gate_a
    y = layer.norm2.forward_native(x) * (1 + scale_f) + shift_f
    hidden, gate = layer.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
    return x + layer.proj["ffn_down"](_situ_native(hidden, gate)) * gate_f


class JointLayer(nn.Module):
    """Joint block ``index`` as a module from hidden states to hidden states (Cache-DiT's ``Pattern_3``): it returns
    them untouched in a pass whose ``skip`` holds its index. It holds its block without owning it, so the DiT's
    weights keep their names under ``layers``."""

    def __init__(self, block: _Block, index: int) -> None:
        super().__init__()
        self._held = (block,)  # a tuple is not registered as a submodule
        self.index = index

    def forward(
        self,
        hidden_states: Tensor,
        rope: tuple[Tensor, Tensor],
        modulation: TokenModulation,
        skip: frozenset[int],
        fused_qkv: bool = False,
    ) -> Tensor:
        if self.index in skip:
            return hidden_states
        return joint_block(self._held[0], hidden_states, rope, modulation, fused_qkv)


JOINT_LAYERS = "joint_layers"  # the attribute Cache-DiT finds the DiT's blocks under


def is_joint_layer(name: str, module: nn.Module) -> bool:
    """Regional compile's condition (``_compile_conditions``) for the plugin's joint blocks."""
    return isinstance(module, JointLayer)


def is_sglang_block(name: str, module: nn.Module) -> bool:
    """Regional compile's condition for SGLang's own joint blocks (``_Block``, not the prompt's ``_TextBlock``)."""
    return isinstance(module, _Block)


def _compiled(module: nn.Module) -> bool:
    """Whether ``module`` was compiled in place (``nn.Module.compile``, as regional compile does)."""
    return getattr(module, "_compiled_call_impl", None) is not None


class DiffuMOSSImageTransformer(MOSSImageTransformer):
    """:class:`MOSSImageTransformer` whose forward also takes ``reference_latents``: reference images' clean latents,
    each ``[1 or B, latent_channels, h, w]`` on its own grid of any size (one row is shared by every sample).

    With ``fused_qkv`` each block's queries, keys and values come from one GEMM (:func:`joint_block`).

    Regional compile (``regional_compile``) compiles the joint blocks only: the plugin's (:class:`JointLayer`) and
    SGLang's (``_Block``). Under torch.compile, regional or whole, every request runs the plugin's blocks, whose
    traced path takes the plain ops SGLang's fused kernels stand for, and SGLang's blocks are never called.

    Raises:
        ValueError: From ``forward``, if a reference is not latents like the drawn image's (as many channels, one row
            or as many as the batch).
        NotImplementedError: From ``forward``, if references are given under sequence parallelism.
    """

    _compile_conditions = [is_joint_layer, is_sglang_block]

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.joint_layers = nn.ModuleList(JointLayer(block, index) for index, block in enumerate(self.layers))

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor | None = None,
        prepared_conditioning: MOSSImageConditioning | None = None,
        skip_layers: Sequence[int] = (),
        reference_latents: Sequence[Tensor] | None = None,
        reference_place: ReferencePlace = "over",
        fused_qkv: bool = False,
    ) -> Tensor:
        cached = bool(getattr(self, "_is_cached", False))  # Cache-DiT's mark: its blocks are joint_layers
        compiled = torch.compiler.is_compiling() or _compiled(self.joint_layers[0])
        if not (reference_latents or cached or compiled or fused_qkv):
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
        references = [self._batched(reference, batch, channels) for reference in reference_latents or ()]
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
            sample = self._sample(
                latents, drawn, last, still, group.text, group.rope, refs, skip, reference_place, fused_qkv
            )
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

    def _reference_rope(
        self, grids: list[tuple[int, int]], text_length: int, like: Tensor, height: int, place: ReferencePlace
    ) -> tuple[Tensor, Tensor]:
        """The rotary coefficients of the references' tokens, laid out as ``prepare_conditioning``'s (``like``'s
        batch and dtype), each reference by ``place`` against a drawn image of ``height`` latent rows."""
        if self.rope.bands.dtype != torch.float32:
            self.rope.reset_bands()
        positions = torch.cat(placed_positions(grids, text_length, height, place, like.device))
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
        place: ReferencePlace,
        fused_qkv: bool = False,
    ) -> Tensor:
        """One group's velocity: the drawn image's tokens, the prompt's and the references' through the joint blocks,
        the references modulated by ``clean`` and the rest by ``modulation``."""
        batch, _, height, width = latents.shape
        image_tokens = height * width
        drawn = image_tokens + text.shape[1]
        x = torch.cat((self._embedded(latents), text, *(self._embedded(reference) for reference in references)), 1)
        grids = [(reference.shape[2], reference.shape[3]) for reference in references]
        if references:
            extra = self._reference_rope(grids, text.shape[1], rope[0], height, place)
            rope = (torch.cat((rope[0], extra[0]), 1), torch.cat((rope[1], extra[1]), 1))
        by_part = TokenModulation(drawn, modulation, clean)
        for layer in self.joint_layers:
            x = layer(x, rope, by_part, skip, fused_qkv)
        x = self.final_norm_video(x[:, :image_tokens].to(self.final_norm_video.weight.dtype))
        x = x * (1 + head[:, 1, None]) + head[:, 0, None]
        x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
        return x.transpose(1, 2).reshape(batch, -1, height, width).float()


# SGLang's Cache-DiT integration keeps its own table of models cache-dit does not know: ours joins it by name.
_CUSTOM_BLOCK_ADAPTER_SPECS[DiffuMOSSImageTransformer.__name__] = CustomBlockAdapterSpec(
    blocks_attr=JOINT_LAYERS, forward_pattern=ForwardPattern.Pattern_3
)
