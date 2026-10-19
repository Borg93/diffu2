"""The FLUX.2-style backbone (``backbone.arch="flux2"``) and its cached canvas reference.

Tiny sizes + the fake VAE / DINO from test_style_canvas keep these CPU-fast and offline.
"""

from __future__ import annotations

import tempfile
from functools import partial

import pytest
import torch
from conftest import C
from test_style_canvas import FakeDino, _cfg, _item, _model
from torch.utils.data import DataLoader

from diffu.config import BackboneConfig, Config, flux2_preset
from diffu.data.collate import collate_lines
from diffu.model.backbone_flux2 import Flux2Backbone


def _flux2(cfg: Config) -> Config:
    cfg.backbone = cfg.backbone.model_copy(
        update={
            "arch": "flux2",
            "dim": 64,
            "heads": 2,
            "num_layers": 1,
            "num_single_layers": 2,
            "context_dim": 64,
        }
    )
    return cfg


def _batch() -> dict:
    return collate_lines([_item(96, "Lind", 48, "Lund"), _item(64, "Åsa", 64, "Erik")], max_ref_width=128)


def test_preset_matches_the_sd3_default_size():
    preset = flux2_preset(BackboneConfig())
    assert (preset.arch, preset.dim, preset.heads, preset.num_layers, preset.num_single_layers) == (
        "flux2",
        1536,
        12,
        4,
        20,
    )
    assert preset.num_layers + preset.num_single_layers == BackboneConfig().num_layers  # same depth (24)


def test_patchify_round_trips():
    backbone = Flux2Backbone(
        BackboneConfig(dim=64, heads=2, num_layers=1, num_single_layers=1), 16, 64, 64, 16
    )
    latent = torch.randn(2, 16, 4, 10)
    assert torch.equal(backbone._unpatchify(backbone._patchify(latent), 4, 10), latent)


class StubStyle(torch.nn.Module):
    """A DINO-mode style encoder stand-in: 4 learned style tokens + a pooled vector, no download."""

    def __init__(self, _cond, dim: int) -> None:
        super().__init__()
        self.tokens = torch.nn.Parameter(torch.zeros(4, dim))
        self.enc = FakeDino().enc

    def forward(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokens.expand(pixels.shape[0], -1, -1) + pixels.mean()
        return tokens, tokens.mean(1)


def test_flux2_trains_and_generates_in_dino_mode(fake_vae, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("diffu.model.diffu.StyleEncoder", StubStyle)
    model = _model(_flux2(_cfg(canvas=False)))
    batch = _batch()
    model.train()
    loss = model(
        batch["images"],
        batch["texts"],
        batch["style_pixel_values"],
        text_widths=batch["widths"],
        cond_dropout=0.1,
    )
    assert torch.isfinite(loss)
    loss.backward()
    model.eval()
    style = torch.zeros(1, 3, 224, 224)
    out = model.generate(["Lind"], style, latent_hw=(4, 10), num_steps=2, cfg_scale=5.0, text_widths=[64])
    assert out.shape == (1, 3, 32, 80)


def test_flux2_canvas_reference_is_tokens_not_channels(fake_vae):
    model = _model(_flux2(_cfg()))
    assert model.ref_tokens
    assert model.backbone.transformer.x_embedder.in_features == C * 4  # no reference channels on flux2
    batch = _batch()
    model.train()
    loss = model(
        batch["images"],
        batch["texts"],
        None,
        text_widths=batch["widths"],
        ref_images=batch["ref_images"],
        ref_texts=batch["ref_texts"],
        ref_widths=batch["ref_widths"],
        cond_dropout=0.1,
        text_dropout=0.1,
        ref_dropout=0.5,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.backbone.transformer.x_embedder.weight.grad is not None


def test_flux2_reference_changes_the_prediction(fake_vae):
    model = _model(_flux2(_cfg())).eval()
    images = torch.rand(1, 3, 32, 64) * 2 - 1
    common = {"text_widths": [64], "ref_texts": ["Lund"], "ref_widths": [48]}
    losses = []
    for ref in (torch.ones(1, 3, 32, 48), -torch.ones(1, 3, 32, 48)):
        torch.manual_seed(0)
        losses.append(model(images, ["Lind"], None, ref_images=ref, **common).item())
    assert losses[0] != pytest.approx(losses[1])


@pytest.mark.parametrize(("cfg_scale", "style_cfg_scale"), [(0.0, 0.0), (5.0, 0.0), (5.0, 2.0)])
def test_cached_reference_sampling_equals_recomputing_it(fake_vae, monkeypatch, cfg_scale, style_cfg_scale):
    model = _model(_flux2(_cfg())).eval()
    kwargs = {
        "latent_hw": (4, 10),
        "num_steps": 3,
        "cfg_scale": cfg_scale,
        "style_cfg_scale": style_cfg_scale,
        "text_widths": [64, 48],
        "ref_images": torch.ones(1, 3, 32, 48),
        "ref_texts": ["Lund"],
        "ref_widths": [48],
    }
    torch.manual_seed(0)
    cached = model.generate(["Lind", "Åsa"], None, **kwargs)

    # Reference run: recompute the reference tokens on EVERY step instead of reusing the cache.
    seen: dict[str, torch.Tensor] = {}
    real_with_ref = Flux2Backbone.forward_with_ref

    def remember(self, latent, timestep, context, pooled, ref_latent, n_content=None):
        seen["ref"] = ref_latent
        return real_with_ref(self, latent, timestep, context, pooled, ref_latent, n_content=n_content)

    def recompute(self, latent, timestep, context, pooled, cache, ref_cols, n_content=None):
        return real_with_ref(self, latent, timestep, context, pooled, seen["ref"], n_content=n_content)[0]

    monkeypatch.setattr(Flux2Backbone, "forward_with_ref", remember)
    monkeypatch.setattr(Flux2Backbone, "forward_cached", recompute)
    torch.manual_seed(0)
    recomputed = model.generate(["Lind", "Åsa"], None, **kwargs)
    assert cached.shape == (2, 3, 32, 80)
    torch.testing.assert_close(cached, recomputed)


def test_flux2_canvas_repa(fake_vae, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("diffu.model.diffu.StyleEncoder", FakeDino)
    cfg = _flux2(_cfg())
    for layer in (0, 2):  # a dual-stream block, then a single-stream block
        cfg.aux.repa, cfg.aux.repa_layer = True, layer
        model = _model(cfg).train()
        batch = _batch()
        out = model(
            batch["images"],
            batch["texts"],
            None,
            text_widths=batch["widths"],
            ref_images=batch["ref_images"],
            ref_texts=batch["ref_texts"],
            ref_widths=batch["ref_widths"],
            return_losses=True,
        )
        assert torch.isfinite(out["repa"]) and torch.isfinite(out["flow"])


def test_flux2_lora_targets_the_fused_single_stream_projections(fake_vae):
    from diffu.lora import add_lora

    model = _model(_flux2(_cfg()))
    assert add_lora(model, rank=4, alpha=8) > 0
    names = [n for n, _ in model.backbone.transformer.named_modules()]
    assert any(n.endswith("single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A") for n in names)
    assert any(n.endswith("single_transformer_blocks.0.attn.to_out.lora_A") for n in names)
    assert any(n.endswith("transformer_blocks.0.attn.add_q_proj.lora_A") for n in names)


def test_flux2_canvas_trains_end_to_end(fake_vae):
    from accelerate.state import AcceleratorState

    from diffu.train import train

    AcceleratorState._reset_state(reset_partial_state=True)
    cfg = _flux2(_cfg())
    loader = DataLoader(
        [_item(96, "Lind", 48, "Lund"), _item(64, "Åsa", 64, "Erik")] * 2,
        batch_size=2,
        collate_fn=partial(collate_lines, max_ref_width=cfg.data.max_ref_width),
    )
    with tempfile.TemporaryDirectory() as tmp:
        train(
            cfg,
            loader,
            steps=2,
            log_every=1,
            use_ema=False,
            out_dir=tmp,
            save_final=False,
            grad_checkpoint=True,
            log_cer=False,
            ref_dropout=0.2,
        )
