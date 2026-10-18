"""glyph_pad (padded, not stretched, glyph tokens) + style_mode="canvas" (in-context reference line).

A tiny backbone and a fake f8 VAE keep these CPU-fast and offline; they check wiring and shapes, not quality.
"""

from __future__ import annotations

import tempfile
from functools import partial

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffu.config import Config
from diffu.data.collate import collate_lines

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


def _cfg(*, canvas: bool = True, glyph_pad: bool = True) -> Config:
    cfg = Config()
    cfg.backbone.dim, cfg.backbone.num_layers, cfg.backbone.heads = 64, 2, 2
    cfg.backbone.context_dim = 64
    cfg.cond.glyph_line = True
    cfg.cond.glyph_cnn_pretrained = False
    cfg.cond.glyph_pad = glyph_pad
    cfg.cond.style_mode = "canvas" if canvas else "dino"
    cfg.aux.repa = False
    cfg.data.line_height = 32
    cfg.data.max_line_width = 256
    cfg.data.max_ref_width = 128
    cfg.flow.sample_steps = 2
    return cfg


@pytest.fixture
def fake_vae(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("diffu.model.diffu.VAEWrapper", lambda _cfg: FakeVAE())


def _model(cfg: Config):
    from diffu.model import Diffu

    torch.manual_seed(0)
    return Diffu(cfg)


def _item(width: int, text: str, ref_width: int, ref_text: str) -> dict[str, object]:
    return {
        "image": torch.rand(3, 32, width) * 2 - 1,
        "text": text,
        "writer_id": "w0",
        "style_pixel_values": torch.zeros(3, 224, 224),
        "style_text": ref_text,
        "style_image": torch.rand(3, 32, ref_width) * 2 - 1,
    }


# --- glyph_pad ---------------------------------------------------------------------------------------


def test_glyph_pad_blanks_columns_past_each_lines_extent():
    from diffu.model.content_encoder import GlyphLineContentEncoder

    enc = GlyphLineContentEncoder(_cfg().cond, out_dim=32)
    tokens = enc(["Lind", "Lewenhaupt"], torch.device("cpu"), w_tokens=12, extents=[3, 9])
    assert tokens.shape == (2, 12, 32)
    assert tokens[0, 3:].abs().sum() == 0 and tokens[1, 9:].abs().sum() == 0  # padded, not stretched
    assert tokens[0, :3].abs().sum() > 0 and tokens[1, :9].abs().sum() > 0


def test_glyph_line_without_extents_still_stretches():
    from diffu.model.content_encoder import GlyphLineContentEncoder

    enc = GlyphLineContentEncoder(_cfg().cond, out_dim=32)
    tokens = enc(["Lind", "Lewenhaupt"], torch.device("cpu"), w_tokens=12)
    assert tokens.shape == (2, 12, 32)
    assert (tokens[0].abs().sum(-1) > 0).all()  # legacy: the short line spans every column


# --- collate -----------------------------------------------------------------------------------------


def test_collate_reports_real_widths_and_pads_refs_to_one_width():
    batch = [_item(100, "a", 40, "ref a"), _item(130, "b", 70, "ref b")]
    out = collate_lines(batch, width_multiple=16, max_width=1024, max_ref_width=128)
    assert out["widths"] == [100, 130]
    assert out["ref_images"].shape == (2, 3, 32, 80)  # max ref width 70 -> 80
    assert out["ref_widths"] == [40, 70]
    assert out["ref_texts"] == ["ref a", "ref b"]
    assert torch.allclose(out["ref_images"][0, :, :, 40:], torch.ones(3, 32, 40))  # white pad


def test_collate_crops_an_overwide_ref_and_drops_its_text():
    out = collate_lines([_item(64, "a", 300, "far too long")], max_ref_width=128)
    assert out["ref_images"].shape[-1] == 128
    assert out["ref_widths"] == [128] and out["ref_texts"] == [""]


def test_collate_widths_ignore_the_white_pad_augmentation():
    out = collate_lines([_item(100, "a", 40, "r")], white_pad_prob=1.0, white_pad_max_frac=0.5)
    assert out["widths"] == [100]  # the glyph extent must stay the real ink extent


# --- model -------------------------------------------------------------------------------------------


def test_canvas_needs_glyph_line(fake_vae):
    cfg = _cfg()
    cfg.cond.glyph_line = False
    with pytest.raises(ValueError, match="glyph_line"):
        _model(cfg)


def test_canvas_forward_takes_the_loss_on_target_columns_only(fake_vae, monkeypatch: pytest.MonkeyPatch):
    import diffu.model.diffu as diffu_mod

    model = _model(_cfg())
    model.train()
    shapes: list[torch.Size] = []
    real_flow_loss = diffu_mod.flow_loss

    def spy(v_pred, target_v, weight=None):
        shapes.append(v_pred.shape)
        return real_flow_loss(v_pred, target_v, weight=weight)

    monkeypatch.setattr(diffu_mod, "flow_loss", spy)
    batch = collate_lines([_item(96, "Lind", 48, "Lund"), _item(64, "Åsa", 64, "Erik")], max_ref_width=128)
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
    assert shapes == [torch.Size([2, C, 32 // 8, 96 // 8])]  # target columns, not the 64+96 px canvas
    loss.backward()
    assert model.backbone.transformer.pos_embed.proj.weight.grad is not None


def test_canvas_reference_changes_the_prediction(fake_vae):
    model = _model(_cfg()).eval()
    images = torch.rand(1, 3, 32, 64) * 2 - 1
    common = {"text_widths": [64], "ref_texts": ["Lund"], "ref_widths": [48]}
    losses = []
    for ref in (torch.ones(1, 3, 32, 48), -torch.ones(1, 3, 32, 48)):
        torch.manual_seed(0)
        losses.append(model(images, ["Lind"], None, ref_images=ref, **common).item())
    assert losses[0] != pytest.approx(losses[1])  # the reference pixels reach the backbone


@pytest.mark.parametrize(("cfg_scale", "style_cfg_scale"), [(0.0, 0.0), (5.0, 0.0), (5.0, 2.0)])
def test_canvas_generate_returns_only_the_target_line(fake_vae, cfg_scale, style_cfg_scale):
    model = _model(_cfg()).eval()
    out = model.generate(
        ["Lind", "Åsa"],
        None,
        latent_hw=(4, 10),
        num_steps=2,
        cfg_scale=cfg_scale,
        style_cfg_scale=style_cfg_scale,
        text_widths=[64, 48],
        ref_images=torch.ones(1, 3, 32, 48),  # one reference broadcast over the batch
        ref_texts=["Lund"],
        ref_widths=[48],
    )
    assert out.shape == (2, 3, 32, 80)


def test_canvas_generate_line_sizes_the_canvas_from_the_reference(fake_vae):
    from PIL import Image

    from diffu.generate import generate_line, load_ref_line

    cfg = _cfg()
    model = _model(cfg).eval()
    ref = load_ref_line(Image.new("RGB", (120, 32), "white"), cfg, text="abcd")  # 30 px per char
    assert ref.image.shape == (1, 3, 32, 128) and ref.width == 120
    img = generate_line(model, "abcdefgh", None, cfg=cfg, num_steps=2, ref=ref)
    assert img.height == 32 and img.width <= 288  # 8 chars × 30 px × slack, rounded to 16


def test_canvas_trains_end_to_end(fake_vae):
    from accelerate.state import AcceleratorState

    from diffu.train import train

    AcceleratorState._reset_state(reset_partial_state=True)  # another test may have set a different precision
    cfg = _cfg()
    batch = [_item(96, "Lind", 48, "Lund"), _item(64, "Åsa", 64, "Erik")] * 2
    loader = DataLoader(
        batch, batch_size=2, collate_fn=partial(collate_lines, max_ref_width=cfg.data.max_ref_width)
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
            grad_checkpoint=False,
            log_cer=False,
            text_dropout=0.1,
            ref_dropout=0.2,
        )


class FakeDino(nn.Module):
    """Stand-in for the frozen DINOv3 encoder REPA reads (224px input, 16px patches, 1 CLS token)."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.enc = nn.Module()
        self.enc.config = type("Cfg", (), {"hidden_size": 24, "patch_size": 16})()

    def _features(self, pixels: torch.Tensor) -> torch.Tensor:
        side = pixels.shape[-1] // 16
        return torch.randn(pixels.shape[0], 1 + side * side, 24)


def test_canvas_repa_aligns_only_the_target_tokens(fake_vae, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("diffu.model.diffu.StyleEncoder", FakeDino)
    cfg = _cfg()
    cfg.aux.repa = True
    model = _model(cfg).train()
    batch = collate_lines([_item(96, "Lind", 48, "Lund"), _item(64, "Åsa", 64, "Erik")], max_ref_width=128)
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
