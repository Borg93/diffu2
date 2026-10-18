"""RoPEJointAttnProcessor routes through diffusers' dispatch_attention_fn (attention backends)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from diffu.config import BackboneConfig
from diffu.model.backbone import Backbone
from diffu.model.rope import RoPEJointAttnProcessor


def _backbone() -> Backbone:
    cfg = BackboneConfig(dim=64, num_layers=2, heads=2, context_dim=32)
    torch.manual_seed(0)
    return Backbone(cfg, in_channels=16, context_dim=32, pooled_dim=32).eval()


def _inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1)
    return torch.randn(2, 16, 4, 12), torch.full((2,), 500.0), torch.randn(2, 6, 32), torch.randn(2, 32)


def test_dispatch_matches_direct_sdpa(monkeypatch):
    model = _backbone()
    latent, t, ctx, pooled = _inputs()
    with torch.no_grad():
        dispatched = model(latent, t, ctx, pooled, n_content=6)

    def direct_sdpa(query, key, value, **_kwargs):  # the pre-0.41 path: SDPA on [B, H, S, D]
        out = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        )
        return out.transpose(1, 2)

    monkeypatch.setattr("diffu.model.rope.dispatch_attention_fn", direct_sdpa)
    with torch.no_grad():
        direct = model(latent, t, ctx, pooled, n_content=6)
    torch.testing.assert_close(dispatched, direct)


def test_set_attention_backend_reaches_the_rope_processor():
    model = _backbone()
    model.set_attention_backend("_native_math")
    processors = set(model.transformer.attn_processors.values())
    assert processors and all(isinstance(p, RoPEJointAttnProcessor) for p in processors)
    assert all(str(p._attention_backend.value) == "_native_math" for p in processors)
    latent, t, ctx, pooled = _inputs()
    with torch.no_grad():
        assert torch.isfinite(model(latent, t, ctx, pooled, n_content=6)).all()
