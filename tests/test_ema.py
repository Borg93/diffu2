"""LerpEMAModel: diffusers' EMAModel with a one-pass foreach update — same EMA weights."""

from __future__ import annotations

import torch
from diffusers.training_utils import EMAModel

from diffu.ema import LerpEMAModel


def _params(seed: int) -> list[torch.nn.Parameter]:
    torch.manual_seed(seed)
    params = [torch.nn.Parameter(torch.randn(16, 8)), torch.nn.Parameter(torch.randn(8))]
    params.append(
        torch.nn.Parameter(torch.randn(4), requires_grad=False)
    )  # a frozen one: copied, not averaged
    return params


def _train(ema_cls: type[EMAModel], steps: int = 50) -> tuple[EMAModel, list[torch.nn.Parameter]]:
    params = _params(0)
    ema = ema_cls(params, use_ema_warmup=True, foreach=True)
    gen = torch.Generator().manual_seed(1)
    for _ in range(steps):
        with torch.no_grad():
            for p in params:
                p.add_(torch.randn(p.shape, generator=gen) * 0.1)
        ema.step(params)
    return ema, params


def test_lerp_ema_matches_diffusers_ema():
    ours, _ = _train(LerpEMAModel)
    stock, _ = _train(EMAModel)
    assert ours.optimization_step == stock.optimization_step == 50
    assert ours.cur_decay_value == stock.cur_decay_value
    for a, b in zip(ours.shadow_params, stock.shadow_params, strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_lerp_ema_step_is_bit_identical_once_past_warmup():
    params = _params(0)
    ours = LerpEMAModel(params, decay=0.99, foreach=True)
    stock = EMAModel(params, decay=0.99, foreach=True)
    for ema in (ours, stock):
        ema.optimization_step = 1000  # past warmup: 1 - decay < 0.5
    with torch.no_grad():
        for p in params:
            p.add_(1.0)
    ours.step(params)
    stock.step(params)
    for a, b in zip(ours.shadow_params, stock.shadow_params, strict=True):
        assert torch.equal(a, b)


def test_state_dict_round_trips_into_the_stock_class():
    ours, params = _train(LerpEMAModel, steps=5)
    stock = EMAModel(params, use_ema_warmup=True, foreach=True)
    stock.load_state_dict(ours.state_dict())
    for a, b in zip(ours.shadow_params, stock.shadow_params, strict=True):
        assert torch.equal(a, b)


def test_shadow_weights_stay_out_of_autograd():
    ours, _ = _train(LerpEMAModel, steps=3)
    assert all(s.grad_fn is None and not s.requires_grad for s in ours.shadow_params)
