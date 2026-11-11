"""DSpark speculative decoding for the MiniCPM-OCR HTR reader in PyTorch and transformers, on CUDA graphs.

vLLM serves this draft natively; this module is for hosts where vLLM cannot run (ZeroGPU). It follows the speculators
DSpark recipe the draft was trained with (speculators 0.10, ``models/dflash`` and ``models/dspark``):

* Context: the reader's residual stream entering layers 4, 8, 12, 16 and 20 at every position before the anchor,
  concatenated, through ``fc`` and ``hidden_norm``; each draft layer projects it to keys and values (``k_norm`` and
  rotary on the keys). A sliding window of 2,048 positions.
* Query block: the anchor token and 14 mask tokens at positions p..p+14, causal inside the block; slot k predicts the
  token at p+k+1 (``sample_from_anchor``).
* Sampling: the block's base logits (a 32,000-token draft vocabulary) plus the Markov bias of the previous token,
  argmax, left to right; draft ids map to the reader's vocabulary through ``d2t``.

Profiled with torch.profiler, eager decoding was launch-bound (about 1,500 kernels a reader pass, the GPU busy 12 % of
the time). So the reader runs on transformers' StaticCache with each pass replayed from a CUDA graph
(``GraphedReader``), the draft's whole proposal is one graph (``GraphedDraft``), and a draft cycle takes one reader
pass (``FastDSpark``): the reader is hybrid, its linear-attention states cannot be cut back after rejected tokens, so
the verifying graph keeps each linear layer's block inputs and the states are rebuilt over the accepted tokens only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load_file
from torch import nn
from transformers import PreTrainedModel, Qwen3Config, StaticCache
from transformers.cache_utils import LinearAttentionCacheLayerMixin, StaticLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen3_5

AUX_LAYERS = (4, 8, 12, 16, 20)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding of x [heads, n, head_dim] with cos/sin [n, head_dim]."""
    return x * cos[None] + rotate_half(x) * sin[None]


class DraftAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, eps: float):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.q_norm = Qwen3RMSNorm(head_dim, eps=eps)
        self.k_norm = Qwen3RMSNorm(head_dim, eps=eps)

    def query(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        return rope(self.q_norm(self.q_proj(x).view(n, self.heads, self.head_dim)).transpose(0, 1), cos, sin)

    def kv(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys and values of x [n, hidden], each [kv_heads, n, head_dim]."""
        n = x.shape[0]
        k = self.k_norm(self.k_proj(x).view(n, self.kv_heads, self.head_dim)).transpose(0, 1)
        v = self.v_proj(x).view(n, self.kv_heads, self.head_dim).transpose(0, 1)
        return rope(k, cos, sin), v


class DraftLayer(nn.Module):
    def __init__(self, cfg: Qwen3Config, heads: int, kv_heads: int, head_dim: int):
        super().__init__()
        eps = float(cfg.rms_norm_eps)
        self.self_attn = DraftAttention(cfg.hidden_size, heads, kv_heads, head_dim, eps)
        self.mlp = Qwen3MLP(cfg)
        self.input_layernorm = Qwen3RMSNorm(cfg.hidden_size, eps=eps)
        self.post_attention_layernorm = Qwen3RMSNorm(cfg.hidden_size, eps=eps)


class DSparkDraft(nn.Module):
    """The draft's weights as speculators saves them; the input embeddings are the reader's."""

    def __init__(self, config: dict[str, Any], embed_tokens: nn.Embedding):
        super().__init__()
        layer_config = {
            k: v for k, v in config["transformer_layer_config"].items() if k not in ("architectures", "dtype")
        }
        self.cfg = Qwen3Config(**layer_config)
        hidden = self.cfg.hidden_size
        self.heads = int(self.cfg.num_attention_heads)
        self.kv_heads = int(self.cfg.num_key_value_heads or self.heads)
        self.head_dim = int(self.cfg.head_dim or hidden // self.heads)
        eps = float(self.cfg.rms_norm_eps)
        self.block_size: int = config["block_size"]
        self.mask_token_id: int = config["mask_token_id"]
        self.window: int | None = self.cfg.sliding_window if self.cfg.use_sliding_window else None
        self.embed_tokens = embed_tokens
        self.fc = nn.Linear(len(config["aux_hidden_state_layer_ids"]) * hidden, hidden, bias=False)
        self.hidden_norm = Qwen3RMSNorm(hidden, eps=eps)
        blocks = [
            DraftLayer(self.cfg, self.heads, self.kv_heads, self.head_dim) for _ in range(self.cfg.num_hidden_layers)
        ]
        self.layers = nn.ModuleList(blocks)
        self.blocks: tuple[DraftLayer, ...] = tuple(blocks)  # the same modules, typed
        self.norm = Qwen3RMSNorm(hidden, eps=eps)
        self.lm_head = nn.Linear(hidden, config["draft_vocab_size"], bias=False)
        self.markov_w1 = nn.Embedding(embed_tokens.num_embeddings, config["markov_rank"])
        self.markov_w2 = nn.Linear(config["markov_rank"], config["draft_vocab_size"], bias=False)
        self.register_buffer("d2t", torch.zeros(config["draft_vocab_size"], dtype=torch.long), persistent=False)
        self.rotary = Qwen3RotaryEmbedding(self.cfg)

    @classmethod
    def from_folder(cls, folder: str | Path, embed_tokens: nn.Embedding) -> DSparkDraft:
        """The draft a speculators checkpoint folder holds (config.json and model.safetensors)."""
        folder = Path(folder)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if config.get("markov_head_type", "vanilla") != "vanilla" or not config.get("sample_from_anchor", True):
            raise ValueError("only drafts with the vanilla Markov head that sample from the anchor are supported")
        draft = cls(config, embed_tokens)
        state: dict[str, torch.Tensor] = {}
        for name, tensor in load_file(folder / "model.safetensors").items():
            if name == "d2t":
                draft.d2t_map.copy_(tensor)
            elif name != "t2d" and not name.startswith("confidence_head."):  # training and adaptive verification only
                state[name.replace("markov_head.", "")] = tensor
        missing, unexpected = draft.load_state_dict(state, strict=False)
        missing = [m for m in missing if not m.startswith(("embed_tokens.", "rotary."))]
        if missing or unexpected:
            raise ValueError(f"the draft's weights do not match: missing {missing}, unexpected {unexpected}")
        return draft

    @property
    def d2t_map(self) -> torch.Tensor:
        return cast(torch.Tensor, self.d2t)

    def cos_sin(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.rotary(torch.empty(0, dtype=dtype, device=positions.device), positions[None])
        return cos[0], sin[0]


def aux_states(hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """The residual stream entering each aux layer, [n, 5 * hidden]; hidden_states[0] are the embeddings."""
    return torch.cat([hidden_states[i][0] for i in AUX_LAYERS], dim=-1)


class Decoded(BaseModel):
    """A reading's tokens, when each became known (seconds from the start) and whether the draft proposed it; its
    seconds and reader passes; with a draft, the tokens proposed and accepted over how many verifications."""

    tokens: list[int] = Field(default_factory=list)
    times: list[float] = Field(default_factory=list)
    from_draft: list[bool] = Field(default_factory=list)
    seconds: float = 0.0
    reader_passes: int = 0
    drafted: int = 0
    accepted: int = 0
    cycles: int = 0

    @property
    def acceptance_length(self) -> float:
        """vLLM's measure: 1 + accepted draft tokens per verification."""
        return 1 + self.accepted / self.cycles if self.cycles else float("nan")


def _stop(tokens: list[int], eos: set[int], max_new: int) -> tuple[list[int], bool]:
    for i, token in enumerate(tokens):
        if token in eos:
            return tokens[: i + 1], True
    if len(tokens) >= max_new:
        return tokens[:max_new], True
    return tokens, False


class _Graph(BaseModel):
    """A captured reader pass: its graph, the static inputs refilled before each replay, the static outputs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph: torch.cuda.CUDAGraph
    ids: torch.Tensor
    pos: torch.Tensor
    out: Any


class _LinearState(BaseModel):
    """A linear-attention layer and the copy of its states `save` took."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    layer_idx: int
    layer: Any
    conv: torch.Tensor
    recurrent: torch.Tensor


class _AttentionState(BaseModel):
    """A full-attention layer and the length `save` took."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    layer: Any
    length: torch.Tensor


class GraphedReader:
    """The reader on a StaticCache, each pass length replayed from a CUDA graph captured on first use.

    The cache is reset in place per request (its addresses, and so the graphs, kept). Full-attention layers keep
    preallocated keys and values and an on-device length; linear-attention layers fixed-address states; the attention
    mask is built inside the graph from the positions."""

    def __init__(self, model: PreTrainedModel, max_len: int):
        self.model, self.max_len = model, max_len
        self.device = model.get_output_embeddings().weight.device
        self.cache = StaticCache(config=model.config, max_cache_len=max_len)
        self.kv_index = torch.arange(max_len, device=self.device)
        self.graphs: dict[tuple[int, bool], _Graph] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.linear: list[_LinearState] = []
        self.attention: list[_AttentionState] = []

    def prefill(self, inputs: dict[str, Any], hidden: bool) -> Any:
        """A new request: the cache emptied in place, then the prompt (image and text) read eagerly."""
        if inputs["input_ids"].shape[1] + 64 > self.max_len:
            raise ValueError(
                f"a prompt of {inputs['input_ids'].shape[1]} tokens is too long for a cache of {self.max_len}"
            )
        self.cache.reset()
        return self.model(**inputs, past_key_values=self.cache, use_cache=True, output_hidden_states=hidden)

    def save(self) -> None:
        """Remember the cache's states and lengths, into buffers allocated on the first call."""
        if not self.linear and not self.attention:
            for idx, layer in enumerate(self.cache.layers):
                if isinstance(layer, LinearAttentionCacheLayerMixin):
                    conv, recurrent = layer.conv_states[0], layer.recurrent_states[0]
                    if conv is None or recurrent is None:
                        raise RuntimeError("save() before the prompt was read")
                    self.linear.append(
                        _LinearState(layer_idx=idx, layer=layer, conv=conv.clone(), recurrent=recurrent.clone())
                    )
                elif isinstance(layer, StaticLayer):
                    self.attention.append(_AttentionState(layer=layer, length=_length(layer).clone()))
            return
        for state in self.linear:
            state.conv.copy_(state.layer.conv_states[0])
            state.recurrent.copy_(state.layer.recurrent_states[0])
        for state in self.attention:
            state.length.copy_(_length(state.layer))

    def restore(self) -> None:
        """The cache as `save` left it (keys and values written since stay behind the length, masked by position)."""
        for state in self.linear:
            state.layer.conv_states[0].copy_(state.conv)
            state.layer.recurrent_states[0].copy_(state.recurrent)
        for state in self.attention:
            _length(state.layer).copy_(state.length)

    def _forward(self, ids: torch.Tensor, pos: torch.Tensor, hidden: bool) -> Any:
        mask = (self.kv_index[None, :] <= pos[0][:, None])[None, None]
        return self.model(
            input_ids=ids,
            position_ids=pos,
            past_key_values=self.cache,
            use_cache=True,
            attention_mask={"full_attention": mask, "linear_attention": None},
            output_hidden_states=hidden,
        )

    def _capture(self, ids: torch.Tensor, pos: torch.Tensor, hidden: bool) -> tuple[torch.cuda.CUDAGraph, Any]:
        self.save()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):  # kernels autotune and allocate outside the capture
                self._forward(ids, pos, hidden)
                self.restore()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out = self._forward(ids, pos, hidden)
        self.restore()  # capture records without running; the cache stays as it was
        return graph, out

    def run(self, tokens: list[int], start: int, hidden: bool) -> Any:
        """A pass over `tokens` at positions start.. (the cache must hold exactly the positions before start)."""
        key = (len(tokens), hidden)
        ids = torch.tensor([tokens], device=self.device)
        pos = torch.arange(start, start + len(tokens), device=self.device)[None]
        graph = self.graphs.get(key)
        if graph is None:
            captured, out = self._capture(ids, pos, hidden)
            graph = self.graphs[key] = _Graph(graph=captured, ids=ids, pos=pos, out=out)
        else:
            graph.ids.copy_(ids)
            graph.pos.copy_(pos)
        graph.graph.replay()
        return graph.out


def _length(layer: StaticLayer) -> torch.Tensor:
    return cast(torch.Tensor, layer.cumulative_length)


class _Recorder:
    """While a verifying graph is captured: each linear layer's block inputs, by layer index."""

    def __init__(self) -> None:
        self.active = False
        self.layer = -1
        self.inputs: dict[int, dict[str, torch.Tensor]] = {}


_RECORDER = _Recorder()
_chunk_rule = qwen3_5.torch_chunk_gated_delta_rule


def _recording_chunk_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if _RECORDER.active:
        _RECORDER.inputs[_RECORDER.layer].update(q=query, k=key, v=value, g=g, beta=beta)
    return _chunk_rule(query, key, value, g, beta, *args, **kwargs)


qwen3_5.torch_chunk_gated_delta_rule = _recording_chunk_rule  # the layers look the rule up at call time


class OnePassReader(GraphedReader):
    """A GraphedReader whose verifying graph (more than one token, hidden states out) records the linear layers' block
    inputs, so a partly accepted block can be kept without a second pass."""

    def __init__(self, model: PreTrainedModel, max_len: int):
        super().__init__(model, max_len)
        update = self.cache.update_conv_state

        def recording_update(conv_input: torch.Tensor, layer_idx: int, *args: Any, **kwargs: Any) -> Any:
            if _RECORDER.active:
                _RECORDER.layer = layer_idx
                _RECORDER.inputs[layer_idx] = {"conv": conv_input}
            return update(conv_input, layer_idx, *args, **kwargs)

        self.cache.update_conv_state = recording_update
        self.recorded: dict[int, dict[str, torch.Tensor]] = {}

    def _capture(self, ids: torch.Tensor, pos: torch.Tensor, hidden: bool) -> tuple[torch.cuda.CUDAGraph, Any]:
        if not (hidden and ids.shape[1] > 1):
            return super()._capture(ids, pos, hidden)
        self.save()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._forward(ids, pos, hidden)
                self.restore()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        _RECORDER.active, _RECORDER.inputs = True, {}
        try:
            with torch.cuda.graph(graph, pool=self.pool):
                out = self._forward(ids, pos, hidden)
        finally:
            _RECORDER.active = False
        self.recorded = dict(_RECORDER.inputs)
        self.restore()
        return graph, out


class GraphedDraft:
    """The draft's context in fixed buffers and its whole proposal (the parallel backbone and the Markov sampling)
    replayed from one CUDA graph: context keys and values sit at their positions in buffers of the reader's cache
    length, the block's own at the end; the window and causal masks come from the anchor's position on the GPU."""

    def __init__(self, draft: DSparkDraft, max_len: int):
        self.draft, self.max_len = draft, max_len
        weight = draft.lm_head.weight
        device, dtype = weight.device, weight.dtype
        b = draft.block_size
        size = (draft.kv_heads, max_len + b, draft.head_dim)
        self.keys = [torch.zeros(size, device=device, dtype=dtype) for _ in draft.blocks]
        self.values = [torch.zeros(size, device=device, dtype=dtype) for _ in draft.blocks]
        self.length = 0
        self.anchor = torch.zeros(1, dtype=torch.long, device=device)
        self.anchor_pos = torch.zeros((), dtype=torch.long, device=device)
        self.kv_index = torch.arange(max_len, device=device)
        self.offsets = torch.arange(b, device=device)
        self.mask_ids = torch.full((b,), draft.mask_token_id, dtype=torch.long, device=device)
        self.tril = torch.ones(b, b, dtype=torch.bool, device=device).tril()
        self.graph = torch.cuda.CUDAGraph()
        self.captured = False
        self.out = torch.zeros(b, dtype=torch.long, device=device)

    def reset(self) -> None:
        self.length = 0  # older entries stay behind, masked out by position

    @torch.inference_mode()
    def extend(self, aux: torch.Tensor, start: int) -> None:
        """The context of positions start.. (aux: the reader's states entering the aux layers)."""
        if start != self.length or start + aux.shape[0] > self.max_len:
            raise ValueError(f"context from {start} does not follow {self.length} or overflows {self.max_len}")
        d, n = self.draft, aux.shape[0]
        c = d.hidden_norm(d.fc(aux))
        cos, sin = d.cos_sin(torch.arange(start, start + n, device=aux.device), c.dtype)
        for i, layer in enumerate(d.blocks):
            k, v = layer.self_attn.kv(c, cos, sin)
            self.keys[i][:, start : start + n].copy_(k)
            self.values[i][:, start : start + n].copy_(v)
        self.length += n

    def _propose(self) -> torch.Tensor:
        d, b = self.draft, self.draft.block_size
        x = d.embed_tokens(torch.where(self.offsets == 0, self.anchor, self.mask_ids)).to(d.lm_head.weight.dtype)
        cos, sin = d.cos_sin(self.anchor_pos + self.offsets, x.dtype)
        context = self.kv_index < self.anchor_pos
        if d.window is not None:
            context = context & (self.kv_index >= self.anchor_pos - d.window)
        mask = torch.cat([context[None, :].expand(b, -1), self.tril], dim=1)[None, None]
        for i, layer in enumerate(d.blocks):
            attn = layer.self_attn
            h = layer.input_layernorm(x)
            k_block, v_block = attn.kv(h, cos, sin)
            self.keys[i][:, self.max_len :].copy_(k_block)
            self.values[i][:, self.max_len :].copy_(v_block)
            out = F.scaled_dot_product_attention(
                attn.query(h, cos, sin)[None],
                self.keys[i][None],
                self.values[i][None],
                attn_mask=mask,
                scale=attn.head_dim**-0.5,
                enable_gqa=True,
            )[0]
            x = x + attn.o_proj(out.transpose(0, 1).reshape(b, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        base = d.lm_head(d.norm(x)).float()
        prev, tokens = self.anchor, []
        for k in range(b):
            # index_select, not d2t[choice]: indexing by a 0-d tensor reads it on the host, which breaks a capture
            choice = (base[k][None] + d.markov_w2(d.markov_w1(prev)).float()).argmax(dim=-1)
            prev = choice + d.d2t_map.index_select(0, choice)
            tokens.append(prev)
        return torch.cat(tokens)

    @torch.inference_mode()
    def propose(self, anchor: int, anchor_pos: int) -> torch.Tensor:
        """The draft's block_size tokens after the anchor (the reader's vocabulary), left on the GPU."""
        if anchor_pos != self.length:
            raise ValueError(f"the anchor at {anchor_pos} does not follow the context's {self.length} positions")
        self.anchor.fill_(anchor)
        self.anchor_pos.fill_(anchor_pos)
        if not self.captured:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._propose()
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(self.graph):
                self.out = self._propose()
            self.captured = True
        self.graph.replay()
        return self.out


class FastDSpark:
    """The reader and its draft, every step of a cycle replayed from CUDA graphs: the draft's proposal goes into the
    verifying graph on the GPU, one sync reads the drafts and the reader's tokens together, and the work after the
    check (the draft's context extended, the reader's states rebuilt) is one graph per number of tokens kept. That work
    depends only on the snapshot and the recorded block inputs, so the warm-up runs before each capture need no undoing.
    """

    def __init__(self, model: PreTrainedModel, draft: DSparkDraft, max_len: int):
        self.reader = OnePassReader(model, max_len)
        self.gdraft = GraphedDraft(draft, max_len)
        self.block = draft.block_size + 1
        self.start = torch.zeros((), dtype=torch.long, device=self.reader.device)
        self.after: dict[int, torch.cuda.CUDAGraph] = {}
        self.verify_out: Any = None

    def _after_check(self, keep: int) -> None:
        g, r = self.gdraft, self.reader
        d = g.draft
        aux = aux_states(self.verify_out.hidden_states)[:keep]
        positions = self.start + torch.arange(keep, device=self.start.device)
        c = d.hidden_norm(d.fc(aux))
        cos, sin = d.cos_sin(positions, c.dtype)
        for i, layer in enumerate(d.blocks):
            k, v = layer.self_attn.kv(c, cos, sin)
            g.keys[i].index_copy_(1, positions, k)
            g.values[i].index_copy_(1, positions, v)
        if keep == self.block:
            return  # every state is already exact
        for state in r.linear:
            inputs = r.recorded[state.layer_idx]
            conv = torch.cat([state.conv, inputs["conv"][..., :keep]], dim=-1)[..., -state.conv.shape[-1] :]
            state.layer.conv_states[0].copy_(conv)
            _, recurrent = qwen3_5.torch_recurrent_gated_delta_rule(
                inputs["q"][:, :keep],
                inputs["k"][:, :keep],
                inputs["v"][:, :keep],
                g=inputs["g"][:, :keep],
                beta=inputs["beta"][:, :keep],
                initial_state=state.recurrent,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            state.layer.recurrent_states[0].copy_(recurrent)
        for state in r.attention:
            _length(state.layer).copy_(self.start + keep)

    def after_check(self, keep: int) -> None:
        """Keep the verified block's first `keep` tokens (the reader's states and the draft's context)."""
        graph = self.after.get(keep)
        if graph is None:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._after_check(keep)
            torch.cuda.current_stream().wait_stream(stream)
            graph = self.after[keep] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.reader.pool):
                self._after_check(keep)
        graph.replay()
        self.gdraft.length += keep

    def _verify(self, anchor: int, drafts: torch.Tensor, anchor_pos: int) -> Any:
        r = self.reader
        graph = r.graphs.get((self.block, True))
        if graph is None:
            return r.run([anchor, *drafts.tolist()], anchor_pos, hidden=True)
        graph.ids[0, 0] = anchor
        graph.ids[0, 1:].copy_(drafts)
        graph.pos.copy_(torch.arange(anchor_pos, anchor_pos + self.block, device=r.device)[None])
        graph.graph.replay()
        return graph.out

    @torch.inference_mode()
    def decode(self, inputs: dict[str, Any], max_new: int, eos: set[int]) -> Decoded:
        """The reader's greedy reading of `inputs`, with the draft: the same text, fewer reader passes."""
        torch.cuda.synchronize()
        started = time.perf_counter()
        r, g = self.reader, self.gdraft
        n = inputs["input_ids"].shape[1]
        max_new = min(max_new, r.max_len - n - 2 * self.block - 2)
        out = r.prefill(inputs, hidden=True)
        g.reset()
        g.extend(aux_states(out.hidden_states), 0)
        result = Decoded(reader_passes=1)
        tokens, done = _stop([int(out.logits[0, -1].argmax())], eos, max_new)
        times, from_draft = [time.perf_counter() - started], [False]
        while not done:
            anchor_pos = n + len(tokens) - 1
            drafts = g.propose(tokens[-1], anchor_pos)
            r.save()
            self.verify_out = self._verify(tokens[-1], drafts, anchor_pos)
            both = torch.cat([drafts, self.verify_out.logits[0].argmax(-1)]).tolist()  # the cycle's one sync
            proposed, verified = both[: self.block - 1], both[self.block - 1 :]
            accepted = 0
            while accepted < len(proposed) and proposed[accepted] == verified[accepted]:
                accepted += 1
            result.reader_passes += 1
            result.cycles += 1
            result.drafted += len(proposed)
            result.accepted += accepted
            times += [time.perf_counter() - started] * (accepted + 1)  # known at this cycle's sync
            from_draft += [True] * accepted + [False]
            self.start.fill_(anchor_pos)
            self.after_check(accepted + 1)
            tokens, done = _stop(tokens + [*proposed[:accepted], verified[accepted]], eos, max_new)
        torch.cuda.synchronize()
        result.tokens, result.seconds = tokens, time.perf_counter() - started
        result.times, result.from_draft = times[: len(tokens)], from_draft[: len(tokens)]
        return result

    @torch.inference_mode()
    def warm(self, inputs: dict[str, Any], eos: set[int]) -> None:
        """Capture every graph (one reading with and one without the draft, then each number of tokens kept)."""
        greedy(self.reader, inputs, 32, eos)
        self.decode(inputs, 128, eos)
        for keep in range(1, self.block + 1):  # idempotent: depends only on the snapshot and the recorded inputs
            self.start.fill_(0)
            self.after_check(keep)
        self.gdraft.reset()


@torch.inference_mode()
def greedy(reader: GraphedReader, inputs: dict[str, Any], max_new: int, eos: set[int]) -> Decoded:
    """The reader alone on CUDA graphs: one replay per token, greedy."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    n = inputs["input_ids"].shape[1]
    max_new = min(max_new, reader.max_len - n - 1)
    out = reader.prefill(inputs, hidden=False)
    tokens, times = [int(out.logits[0, -1].argmax())], [time.perf_counter() - started]
    while tokens[-1] not in eos and len(tokens) < max_new:
        out = reader.run([tokens[-1]], n + len(tokens) - 1, hidden=False)
        tokens.append(int(out.logits[0, -1].argmax()))
        times.append(time.perf_counter() - started)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    tokens = _stop(tokens, eos, max_new)[0]
    return Decoded(
        tokens=tokens,
        times=times[: len(tokens)],
        from_draft=[False] * len(tokens),
        seconds=seconds,
        reader_passes=len(times),
    )
