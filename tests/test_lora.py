"""LoRA adapter mechanics on a tiny backbone (no heavy VAE/DINO): add -> freeze -> save/load."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from diffu.config import BackboneConfig
from diffu.lora import DEFAULT_TARGETS, LORA_WEIGHTS_NAME, add_lora, load_lora, save_lora
from diffu.model import Diffu
from diffu.model.backbone import Backbone


class _Wrap(nn.Module):
    """Minimal stand-in exposing ``.backbone.transformer`` — all the LoRA helpers touch."""

    def __init__(self, backbone: Backbone) -> None:
        super().__init__()
        self.backbone = backbone


def _tiny() -> _Wrap:
    cfg = BackboneConfig(dim=32, num_layers=2, heads=2, context_dim=16)
    return _Wrap(Backbone(cfg, in_channels=16, out_channels=16))


def test_add_lora_freezes_base_trains_only_adapter() -> None:
    model = _tiny()
    n_trainable = add_lora(cast(Diffu, model), rank=4, alpha=8, targets=DEFAULT_TARGETS)
    assert n_trainable > 0
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable, "no trainable params after add_lora"
    assert all("lora_" in name for name in trainable), "add_lora must freeze everything except LoRA"


def test_lora_save_load_applies_after_load(tmp_path) -> None:
    # A non-identity adapter (lora_B != 0) must CHANGE the forward after load — guards the prefix bug
    # where load_lora_adapter silently matched no keys (adapter loaded but never applied).
    src = _tiny()
    add_lora(cast(Diffu, src), rank=4, alpha=8)
    with torch.no_grad():
        for name, p in src.named_parameters():
            if "lora_B" in name:
                p.fill_(0.05)  # lora_B inits to 0 (identity); make it non-identity so it must perturb output
    path = save_lora(cast(Diffu, src), str(tmp_path))
    assert path.endswith(LORA_WEIGHTS_NAME)
    assert (tmp_path / LORA_WEIGHTS_NAME).is_file()

    dst = _tiny()
    lat, ts = torch.randn(1, 16, 4, 4), torch.tensor([500.0])
    ctx, pooled = torch.randn(1, 5, 16), torch.randn(1, 16)
    with torch.no_grad():
        before = dst.backbone(lat, ts, ctx, pooled).clone()
    load_lora(cast(Diffu, dst), str(tmp_path))
    with torch.no_grad():
        after = dst.backbone(lat, ts, ctx, pooled)
    assert after.shape == (1, 16, 4, 4)
    assert not torch.allclose(before, after), "LoRA loaded but not applied — check load_lora prefix"
