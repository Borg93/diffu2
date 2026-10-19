"""Shared test fixtures: a fake f8 VAE so model-level tests run CPU-fast and offline."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

C = 16  # latent channels of the real (and fake) VAE


class FakeVAE(nn.Module):
    """f8 stand-in for the Qwen VAE: average-pool to a C-channel latent, nearest-upsample back."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))  # gives the module a device/dtype

    def encode(self, x: torch.Tensor, *, sample: bool = True) -> torch.Tensor:
        return F.avg_pool2d(x.mean(1, keepdim=True), 8).expand(-1, C, -1, -1).contiguous()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return F.interpolate(z[:, :3], scale_factor=8, mode="nearest")


@pytest.fixture
def fake_vae(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("diffu.model.diffu.VAEWrapper", lambda _cfg: FakeVAE())
