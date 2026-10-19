"""The VAE's single-frame fast path matches diffusers' stock frame loop (tiny random Qwen VAE, no download)."""

from __future__ import annotations

import copy

import pytest
import torch
from diffusers import AutoencoderKLQwenImage

from diffu.config import VAEConfig
from diffu.model.vae import VAEWrapper, _SingleFrameCausalConv3d


@pytest.fixture
def tiny_vae() -> AutoencoderKLQwenImage:
    torch.manual_seed(0)
    return AutoencoderKLQwenImage(base_dim=16, z_dim=16, dim_mult=[1, 2, 2, 2], num_res_blocks=1).eval()


def _wrapper(monkeypatch: pytest.MonkeyPatch, vae: AutoencoderKLQwenImage) -> VAEWrapper:
    monkeypatch.setattr("diffu.model.vae._load_vae", lambda *_args: copy.deepcopy(vae))
    return VAEWrapper(VAEConfig()).eval()


def test_fast_path_matches_the_stock_frame_loop(monkeypatch, tiny_vae):
    wrapper = _wrapper(monkeypatch, tiny_vae)
    assert any(isinstance(m, _SingleFrameCausalConv3d) for m in wrapper.vae.modules())
    x = torch.rand(2, 3, 64, 96) * 2 - 1
    with torch.no_grad():
        stock_z = wrapper._normalize(tiny_vae.encode(x.unsqueeze(2)).latent_dist.mode().squeeze(2))
        fast_z = wrapper.encode(x, sample=False)
        stock_img = tiny_vae.decode(wrapper._denormalize(stock_z).unsqueeze(2)).sample.squeeze(2)
        fast_img = wrapper.decode(stock_z)
    torch.testing.assert_close(fast_z, stock_z, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(fast_img, stock_img, atol=1e-5, rtol=1e-4)
    assert set(wrapper.vae.state_dict()) == set(tiny_vae.state_dict())  # same checkpoint keys


def test_tiling_keeps_the_stock_path(monkeypatch, tiny_vae):
    wrapper = _wrapper(monkeypatch, tiny_vae)
    wrapper.vae.use_tiling = True
    assert not wrapper._direct_single_frame()


def test_decoder_gradients_match_the_stock_path(monkeypatch, tiny_vae):
    wrapper = _wrapper(monkeypatch, tiny_vae).train()
    stock = copy.deepcopy(tiny_vae).train()
    z = torch.randn(1, 16, 8, 8)
    wrapper.decode(z).square().mean().backward()
    stock.decode(wrapper._denormalize(z).unsqueeze(2)).sample.square().mean().backward()
    ours = dict(wrapper.vae.decoder.named_parameters())
    for name, param in stock.decoder.named_parameters():
        if param.grad is None:  # e.g. temporal-upsampling convs, unused for one frame on both paths
            assert ours[name].grad is None, name
        else:
            torch.testing.assert_close(ours[name].grad, param.grad, atol=1e-5, rtol=1e-3)
