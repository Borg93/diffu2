"""RoPEJointAttnProcessor routes through diffusers' dispatch_attention_fn (attention backends)."""

from __future__ import annotations

import threading

import pytest
import torch
import torch._dynamo
import torch.nn.functional as F

from diffu.config import BackboneConfig
from diffu.model.backbone import Backbone
from diffu.model.rope import RoPEJointAttnProcessor


def _backbone() -> Backbone:
    cfg = BackboneConfig(dim=64, num_layers=2, heads=2, context_dim=32)
    torch.manual_seed(0)
    return Backbone(cfg, in_channels=16).eval()


def _inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1)
    return torch.randn(2, 16, 4, 12), torch.full((2,), 500.0), torch.randn(2, 6, 32), torch.randn(2, 32)


def test_dispatch_matches_direct_sdpa(monkeypatch):
    model = _backbone()
    latent, t, ctx, pooled = _inputs()
    with torch.no_grad():
        dispatched = model(latent, t, ctx, pooled)

    def direct_sdpa(query, key, value, **_kwargs):  # the pre-0.41 path: SDPA on [B, H, S, D]
        out = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        )
        return out.transpose(1, 2)

    monkeypatch.setattr("diffu.model.rope.dispatch_attention_fn", direct_sdpa)
    with torch.no_grad():
        direct = model(latent, t, ctx, pooled)
    torch.testing.assert_close(dispatched, direct)


def test_set_attention_backend_reaches_the_rope_processor():
    model = _backbone()
    model.set_attention_backend("_native_math")
    processors = set(model.transformer.attn_processors.values())
    assert processors and all(isinstance(p, RoPEJointAttnProcessor) for p in processors)
    assert all(str(p._attention_backend.value) == "_native_math" for p in processors)
    latent, t, ctx, pooled = _inputs()
    with torch.no_grad():
        assert torch.isfinite(model(latent, t, ctx, pooled)).all()


def test_fused_qkv_matches_separate_projections():
    model = _backbone()
    latent, t, ctx, pooled = _inputs()
    with torch.no_grad():
        separate = model(latent, t, ctx, pooled)
        model.fuse_qkv_projections()
        fused = model(latent, t, ctx, pooled)
    fused_attns = [m for m in model.transformer.modules() if getattr(m, "fused_projections", False)]
    assert fused_attns and all(hasattr(m, "to_qkv") and hasattr(m, "to_added_qkv") for m in fused_attns)
    torch.testing.assert_close(fused, separate)


def _processor(model: Backbone) -> RoPEJointAttnProcessor:
    (processor,) = set(model.transformer.attn_processors.values())
    assert isinstance(processor, RoPEJointAttnProcessor)
    return processor


def test_concat_qkv_matches_separate_projections_and_gradients():
    latent, t, ctx, pooled = _inputs()
    grads = []
    outs = []
    for concat in (False, True):
        model = _backbone()
        _processor(model).concat_qkv = concat
        out = model(latent, t, ctx, pooled)
        out.square().mean().backward()
        attn = model.transformer.transformer_blocks[0].attn
        grads.append((attn.to_q.weight.grad.clone(), attn.add_v_proj.weight.grad.clone()))
        outs.append(out.detach())
    torch.testing.assert_close(outs[1], outs[0])
    for separate, concat in zip(grads[0], grads[1], strict=True):
        torch.testing.assert_close(concat, separate)


def test_attention_gets_bf16_qkv_under_autocast(monkeypatch):
    import diffu.model.rope as rope

    seen: list[tuple[torch.dtype, ...]] = []
    real = rope.dispatch_attention_fn

    def spy(query, key, value, **kwargs):
        seen.append((query.dtype, key.dtype, value.dtype))
        return real(query, key, value, **kwargs)

    monkeypatch.setattr(rope, "dispatch_attention_fn", spy)
    model = _backbone()  # fp32 weights, as in training
    latent, t, ctx, pooled = _inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        model(latent, t, ctx, pooled)
    assert seen and all(dtypes == (torch.bfloat16,) * 3 for dtypes in seen)


def test_concat_qkv_falls_back_for_lora_layers():
    from diffu.lora import add_lora

    class Wrap(torch.nn.Module):
        def __init__(self, backbone: Backbone) -> None:
            super().__init__()
            self.backbone = backbone

    model = _backbone()
    latent, t, ctx, pooled = _inputs()
    add_lora(Wrap(model), rank=4, alpha=8)
    with torch.no_grad():
        separate = model(latent, t, ctx, pooled)
        _processor(model).concat_qkv = True
        concat = model(latent, t, ctx, pooled)
    torch.testing.assert_close(concat, separate)


def test_compile_blocks_turns_on_the_qkv_concat():
    model = _backbone()
    assert not _processor(model).concat_qkv
    model.compile_blocks()
    assert _processor(model).concat_qkv


def test_an_attention_mask_raises_instead_of_being_ignored():
    model = _backbone()
    attn = model.transformer.transformer_blocks[0].attn
    hidden, context = torch.randn(2, 24, 64), torch.randn(2, 6, 64)
    with pytest.raises(NotImplementedError, match="attention mask"):
        attn(hidden, context, attention_mask=torch.ones(2, 30, dtype=torch.bool))


def test_compiled_backbone_applies_the_compile_settings_on_other_threads():
    # torch keeps Dynamo's settings per thread; a server or autograd thread would otherwise run the compiled
    # blocks against the default recompile limit (8), which fullgraph=True makes fatal.
    model = _backbone()
    model._compiled = True  # what compile_blocks() sets, without paying for a compile here
    seen: list[int] = []

    def worker() -> None:
        with torch.no_grad():
            model(*_inputs())
        seen.append(torch._dynamo.config.recompile_limit)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert seen == [256]


def test_checkpoint_recompute_applies_the_compile_settings_on_its_thread(monkeypatch: pytest.MonkeyPatch):
    # Gradient checkpointing re-runs each block during backward, on autograd's device thread for CUDA; a
    # compiled block must get the compile settings there too. CPU autograd runs on the calling thread, so a
    # backward on a worker thread stands in for it.
    import diffu.model.backbone as backbone_module

    threads: list[int] = []
    monkeypatch.setattr(
        backbone_module, "ensure_compile_config", lambda mode=None: threads.append(threading.get_ident())
    )
    model = _backbone().train()
    model.enable_gradient_checkpointing()
    model._compiled = True
    loss = model(*_inputs()).square().mean()
    worker = threading.Thread(target=loss.backward)
    worker.start()
    worker.join()
    assert worker.ident in threads
