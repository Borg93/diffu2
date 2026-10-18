"""Rectified-flow objective: interpolant, timestep sampling, losses."""

from __future__ import annotations

import torch

from diffu.flow import (
    flow_loss,
    interpolate,
    repa_loss,
    sample_timesteps,
)


def test_interpolate_endpoints():
    x0 = torch.randn(4, 16, 8, 8)
    eps = torch.randn_like(x0)
    t0 = torch.zeros(4)
    x_t, target_v = interpolate(x0, eps, t0)
    assert torch.allclose(x_t, x0)  # t=0 -> data
    assert torch.allclose(target_v, eps - x0)  # v = eps - x0 (t-independent)

    x_t1, _ = interpolate(x0, eps, torch.ones(4))
    assert torch.allclose(x_t1, eps)  # t=1 -> noise


def test_sample_timesteps_in_unit_interval():
    t = sample_timesteps(1024, torch.device("cpu"))
    assert t.shape == (1024,)
    assert (t >= 0).all() and (t <= 1).all()


def test_flow_loss_matches_mse_and_weighting():
    v = torch.randn(2, 3)
    tgt = torch.randn(2, 3)
    assert torch.allclose(flow_loss(v, tgt), ((v - tgt) ** 2).mean())
    assert torch.allclose(flow_loss(v, tgt, weight=torch.zeros_like(v)), torch.tensor(0.0))


def test_repa_loss_zero_when_aligned_one_when_orthogonal():
    a = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    assert torch.allclose(repa_loss(a, a.clone()), torch.tensor(0.0), atol=1e-6)
    orth = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert torch.allclose(repa_loss(a, orth), torch.tensor(1.0), atol=1e-6)
