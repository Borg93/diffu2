"""glyph_pad (padded, not stretched, glyph tokens) + style_mode="canvas" (in-context reference line).

A tiny backbone and a fake f8 VAE keep these CPU-fast and offline; they check wiring and shapes, not quality.
"""

from __future__ import annotations

import os
import tempfile
import traceback
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
    img = generate_line(model, "abcdefgh", None, num_steps=2, ref=ref)
    assert img.height == 32 and img.width <= 288  # 8 chars × 30 px × slack, rounded to 16
    short = generate_line(model, "abcdefgh", None, num_steps=2, ref=ref, ink_width=64)  # a set length
    assert short.width <= 80  # 64 px × slack, rounded to 16


def test_style_guidance_without_text_guidance_keeps_the_text(fake_vae):
    model = _model(_cfg()).eval()
    calls = _backbone_calls(model)
    model.generate(
        ["Lind"],
        None,
        latent_hw=(4, 10),
        num_steps=2,
        cfg_scale=0.0,  # text guidance off = scale 1, not "drop the text"
        style_cfg_scale=2.0,
        text_widths=[64],
        ref_images=torch.ones(1, 3, 32, 48),
        ref_texts=["Lund"],
        ref_widths=[48],
    )
    assert all(latent.shape[0] == 2 for latent, _ in calls)  # full + text-only: no null pass


def test_checkpoints_carry_config_and_the_ema_export(fake_vae, tmp_path):
    from accelerate.state import AcceleratorState
    from safetensors import safe_open

    from diffu.generate import build_model, load_config
    from diffu.train import train

    AcceleratorState._reset_state(reset_partial_state=True)
    cfg = _cfg()
    batch = [_item(96, "Lind", 48, "Lund"), _item(64, "Åsa", 64, "Erik")] * 2
    loader = DataLoader(batch, batch_size=2, collate_fn=partial(collate_lines, max_ref_width=128))
    train(
        cfg,
        loader,
        steps=2,
        save_every=1,
        log_every=1,
        out_dir=str(tmp_path),
        grad_checkpoint=False,
        log_cer=False,
    )
    for ckpt in ("step_1", "final"):
        assert load_config(tmp_path / ckpt).model_dump() == cfg.model_dump()  # config.json round-trips
    assert (
        tmp_path / "step_1" / "ema_model.safetensors"
    ).is_file()  # EMA weights next to the resumable state
    final = str(tmp_path / "final" / "model.safetensors")
    with safe_open(final, framework="pt") as handle:
        keys = set(handle.keys())
    assert keys and not any(k.startswith("vae.") for k in keys)  # the frozen pretrained VAE is not exported
    assert build_model(load_config(tmp_path / "final"), final, device="cpu").canvas


def _backbone_calls(model) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Record each backbone call's (input latent, pooled vector)."""
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    real = model.backbone.forward

    def spy(latent, timestep, context, pooled, **kwargs):
        calls.append((latent.detach().clone(), pooled.detach().clone()))
        return real(latent, timestep, context, pooled, **kwargs)

    model.backbone.forward = spy
    return calls


@pytest.mark.parametrize(("ref_dropout", "cond_dropout"), [(1.0, 0.0), (0.0, 1.0)])
def test_a_dropped_reference_leaves_its_canvas_columns_pure_noise(fake_vae, ref_dropout, cond_dropout):
    model = _model(_cfg()).train()
    calls = _backbone_calls(model)
    images = torch.rand(1, 3, 32, 64) * 2 - 1
    common = {"text_widths": [64], "ref_texts": ["Lund"], "ref_widths": [48]}
    for dropout in ({"ref_dropout": ref_dropout, "cond_dropout": cond_dropout}, {}):
        for ref in (torch.ones(1, 3, 32, 48), -torch.ones(1, 3, 32, 48)):
            torch.manual_seed(0)
            model(images, ["Lind"], None, ref_images=ref, **common, **dropout)
    ref_cols = 48 // 8
    noisy = [latent[:, :C, :, :ref_cols] for latent, _ in calls]
    torch.testing.assert_close(noisy[0], noisy[1])  # dropped: the same noise whatever the reference
    assert calls[0][0][:, C:].abs().sum() == 0  # and the clean-reference channels are blank
    assert not torch.allclose(noisy[2], noisy[3])  # kept: the columns carry the (noised) reference


def test_sampling_rebuilds_the_reference_columns_each_step(fake_vae):
    from diffusers import FlowMatchEulerDiscreteScheduler

    model = _model(_cfg()).eval()
    calls = _backbone_calls(model)
    ref = torch.ones(1, 3, 32, 48)
    model.generate(
        ["Lind"],
        None,
        latent_hw=(4, 10),
        num_steps=3,
        cfg_scale=5.0,
        style_cfg_scale=2.0,
        text_widths=[64],
        ref_images=ref,
        ref_texts=["Lund"],
        ref_widths=[48],
    )
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(3)
    ref_latent, ref_cols = model.vae.encode(ref), 48 // 8
    eps_ref = calls[0][0][:1, :C, :, :ref_cols]  # the first step is all noise (sigma = 1)
    for (latent, pooled), sigma in zip(calls, scheduler.sigmas, strict=False):
        full, text_only, null = latent[:, :C, :, :ref_cols].chunk(3)
        torch.testing.assert_close(full, (1 - sigma) * ref_latent + sigma * eps_ref)  # as forward() noises it
        torch.testing.assert_close(text_only, eps_ref)  # the reference-dropped variants see pure noise
        torch.testing.assert_close(null, eps_ref)
        assert pooled[2].abs().sum() == 0 and torch.equal(pooled[0], pooled[1])  # null = cond-dropout mode


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

    def _features(self, pixels: torch.Tensor) -> torch.Tensor:  # CLS + one token per 16-px patch
        patches = (pixels.shape[-2] // 16) * (pixels.shape[-1] // 16)
        return torch.randn(pixels.shape[0], 1 + patches, 24)


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


def test_warm_start_widens_the_input_layer_with_zeroed_new_channels(
    fake_vae, monkeypatch: pytest.MonkeyPatch
):
    from safetensors.torch import save_file

    from diffu.generate import load_checkpoint

    monkeypatch.setattr("diffu.model.diffu.StyleEncoder", FakeDino)  # the legacy model's DINO, offline
    legacy = _model(_cfg(canvas=False, glyph_pad=False))
    canvas = _model(_cfg())
    key = "backbone.transformer.pos_embed.proj.weight"
    assert legacy.state_dict()[key].shape[1] == C and canvas.state_dict()[key].shape[1] == 2 * C
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/legacy.safetensors"
        save_file({k: v.contiguous() for k, v in legacy.state_dict().items()}, path)
        load_checkpoint(canvas, path)
    widened = canvas.state_dict()[key]
    torch.testing.assert_close(widened[:, :C], legacy.state_dict()[key])
    assert widened[:, C:].abs().sum() == 0  # the reference channels start silent


@pytest.mark.parametrize("canvas", [False, True])
def test_fused_qkv_generation_is_unchanged(fake_vae, monkeypatch: pytest.MonkeyPatch, canvas: bool):
    monkeypatch.setattr("diffu.model.diffu.StyleEncoder", FakeDinoStyle)
    model = _model(_cfg(canvas=canvas)).eval()
    kwargs = {"latent_hw": (4, 10), "num_steps": 2, "cfg_scale": 5.0, "text_widths": [64]}
    if canvas:
        kwargs |= {"ref_images": torch.ones(1, 3, 32, 48), "ref_texts": ["Lund"], "ref_widths": [48]}
    style = None if canvas else torch.zeros(1, 3, 224, 224)
    torch.manual_seed(0)
    separate = model.generate(["Lind"], style, **kwargs)
    model.backbone.fuse_qkv_projections()
    torch.manual_seed(0)
    fused = model.generate(["Lind"], style, **kwargs)
    torch.testing.assert_close(fused, separate)


class FakeDinoStyle(FakeDino):
    """FakeDino that also acts as the dino-mode style encoder (a zero pooled style vector)."""

    def __init__(self, _cond, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return pixels.new_zeros(pixels.shape[0], self.dim)


def _train_two_log_steps(monkeypatch: pytest.MonkeyPatch, loss_value: float) -> list[int]:
    """Run 4 optimizer steps with log_every=2 on a stubbed loss; return train.py's .item() call count."""
    from accelerate.state import AcceleratorState

    import diffu.model.diffu as diffu_mod
    from diffu.train import train

    AcceleratorState._reset_state(reset_partial_state=True)

    def fake_forward(self, images, texts, *_args, **_kwargs):
        loss = self.backbone.transformer.proj_out.weight.sum() * 0.0 + loss_value
        return {"loss": loss, "flow": loss.detach()}

    monkeypatch.setattr(diffu_mod.Diffu, "forward", fake_forward)
    calls = [0]
    real_item = torch.Tensor.item

    def counting_item(self):
        caller = traceback.extract_stack(limit=2)[0].filename
        if caller.endswith(os.path.join("diffu", "train.py")):  # not torch internals (e.g. CPU AdamW)
            calls[0] += 1
        return real_item(self)

    monkeypatch.setattr(torch.Tensor, "item", counting_item)
    cfg = _cfg()
    loader = DataLoader([_item(64, "a", 32, "b")] * 8, batch_size=2, collate_fn=collate_lines)
    with tempfile.TemporaryDirectory() as tmp:
        train(cfg, loader, steps=4, log_every=2, use_ema=False, out_dir=tmp, save_final=False, log_cer=False)
    return calls


def test_training_reads_losses_only_at_log_steps(fake_vae, monkeypatch: pytest.MonkeyPatch):
    calls = _train_two_log_steps(monkeypatch, 1.0)
    assert calls[0] == 4  # (loss, flow) at steps 0 and 2 — not at every step


def test_non_finite_loss_still_stops_training(fake_vae, monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(RuntimeError, match="non-finite loss"):
        _train_two_log_steps(monkeypatch, float("nan"))


# --- model guards --------------------------------------------------------------------------------------


def test_a_grid_first_seen_while_sampling_still_trains(fake_vae):
    # generate() runs under inference_mode; a RoPE table first built there used to be cached as an inference
    # tensor, which autograd can't save, so the next training step on that canvas width failed.
    model = _model(_cfg())
    ref = torch.ones(1, 3, 32, 48)
    model.eval().generate(["Lind"], None, latent_hw=(4, 10), ref_images=ref, ref_texts=["Lund"])
    model.train()
    images = torch.rand(1, 3, 32, 80) * 2 - 1  # latent width 10: the canvas grid sampled above
    loss = model(images, ["Lind"], text_widths=[80], ref_images=ref, ref_texts=["Lund"], ref_widths=[48])
    loss.backward()


@pytest.mark.parametrize("text", ["a", ""])  # "" = a reference with an unknown transcription
def test_one_glyph_strip_trains_at_batch_size_one(text: str):
    from diffu.model.content_encoder import GlyphLineContentEncoder

    enc = GlyphLineContentEncoder(_cfg().cond, out_dim=32).train()  # train-mode BatchNorm
    tokens = enc([text], torch.device("cpu"), w_tokens=4, extents=[2])
    assert tokens.shape == (1, 4, 32) and tokens[0, 2:].abs().sum() == 0
    assert enc([text], torch.device("cpu"), w_tokens=4).shape == (1, 4, 32)


def test_line_height_must_fit_the_token_grid(fake_vae):
    cfg = _cfg()
    cfg.data.line_height = 40
    with pytest.raises(ValueError, match="multiple of 16"):
        _model(cfg)


def test_style_guidance_needs_a_canvas_model(fake_vae, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("diffu.model.diffu.StyleEncoder", FakeDinoStyle)
    model = _model(_cfg(canvas=False)).eval()
    with pytest.raises(ValueError, match="canvas"):
        model.generate(["Lind"], torch.zeros(1, 3, 224, 224), latent_hw=(4, 10), style_cfg_scale=2.0)
