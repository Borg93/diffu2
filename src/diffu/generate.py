"""Generate handwritten line images from text + a writer style reference.

Wraps ``Diffu.generate`` (CFG-guided flow sampling) as the ``diffu-generate`` CLI:

    diffu-generate --text "Smörgåsbord" --style ref_line.png --out smorgas.png --ckpt run/final/model.safetensors

The architecture comes from the ``config.json`` saved next to the checkpoint (arch flags override it).
Without ``--ckpt`` it runs an untrained model (plumbing smoke only). The style reference is any line crop
of the target hand; it is ImageNet-normalized to the DINO input the model expects — or, for a
``--style-canvas`` checkpoint, placed at line height left of the generated line (give its transcription
with ``--style-text`` so the model can match letters to strokes).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict
from safetensors import safe_open
from safetensors.torch import load_file
from torchvision.transforms import functional as TF

from .config import Config, add_arch_args, apply_arch_args
from .data.collate import round_up
from .data.dataset import style_tensor
from .model import Diffu

_WIDTH_BIAS = 0.45  # canvas-density knob for dino-mode models: they have NO length control and FILL any
# canvas they are given (no internal "write N chars and stop"): an over-wide canvas is filled by LOOPING
# characters (garbled repeats), an over-tight one crams. CRUCIAL: the raw estimator scales width with
# line-height (natural_width: size = height*0.7), so this bias is HEIGHT-COUPLED and must be re-tuned whenever
# line_height changes. 0.55 hit the GT density (~15 px/char) at the original 64px, but at the current 128px
# that same 0.55 lands at ~29 px/char — the catastrophic looping regime (CER-vs-density is monotonic:
# 30/24/18/15 px/char -> 0.41/0.18/0.09/0.037). 0.45 (~24 px/char at 128px) is the empirically clean point
# across Swedish + English (the width sweeps in docs/EXPERIMENTS.md). glyph_pad is the real length control.
_INK_THRESHOLD = 200  # 0-255 grayscale; a column with any pixel below this counts as inked
_CANVAS_SLACK = 1.15  # canvas-mode canvas = expected ink width × this (room for a longer hand; ink-cropped)
_INPUT_PROJ = "backbone.transformer.pos_embed.proj.weight"


def load_style(source: str | Path | Image.Image) -> torch.Tensor:
    """A style reference line crop (path or image) -> the DINO input ``[3, S, S]`` (dino-mode models)."""
    if isinstance(source, Image.Image):
        return style_tensor(source)
    with Image.open(source) as im:
        return style_tensor(im)


class RefLine(BaseModel):
    """A canvas-mode style reference: the line at model height, white-padded to a multiple of 16."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: torch.Tensor  # [1, 3, H, W16] in [-1, 1]
    width: int  # real ink width in px (before the padding)
    text: str = ""  # its transcription ("" = unknown)


def load_ref_line(source: str | Path | Image.Image, cfg: Config, text: str = "") -> RefLine:
    """A reference line crop -> :class:`RefLine` at ``cfg.data.line_height`` (aspect kept, capped width).

    A crop wider than ``cfg.data.max_ref_width`` is cut and its transcription dropped, as in training.
    """
    if isinstance(source, Image.Image):
        img = source.convert("RGB")
    else:
        with Image.open(source) as handle:
            img = handle.convert("RGB")
    height = cfg.data.line_height
    width = min(cfg.data.max_line_width, max(16, round(img.width * height / max(img.height, 1))))
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    if width > cfg.data.max_ref_width:
        img, width, text = img.crop((0, 0, cfg.data.max_ref_width, height)), cfg.data.max_ref_width, ""
    tensor = TF.normalize(TF.to_tensor(img), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    tensor = torch.nn.functional.pad(tensor, (0, round_up(width, 16) - width), value=1.0)
    return RefLine(image=tensor.unsqueeze(0), width=width, text=text)


def ref_text_width(ref: RefLine, text: str, model: Diffu) -> int:
    """Expected ink width (px) of ``text`` in the reference's hand: its px-per-char × len(text).

    Falls back to the glyph renderer's raw width estimate when the reference has no transcription.
    """
    if not ref.text.strip():
        return model.guidance_renderer.natural_width(text)
    return max(16, round(len(text) * ref.width / len(ref.text)))


def to_pil(image: torch.Tensor) -> Image.Image:
    """``[3, H, W]`` in ``[-1, 1]`` -> PIL RGB. ``.float()``: torchvision can't convert bf16 tensors."""
    return TF.to_pil_image(((image.clamp(-1, 1) + 1) / 2).float().cpu())


def auto_canvas_width(model: Diffu, text: str) -> int:
    """The dino-mode auto canvas width (px) for ``text``: the glyph renderer's raw estimate × ``_WIDTH_BIAS``,
    rounded to 16 and capped at ``max_line_width``. Its blank tail is removed by :func:`ink_crop`."""
    renderer = model.guidance_renderer
    biased = int(renderer.natural_width(text) * _WIDTH_BIAS)
    return min(model.cfg.data.max_line_width, round_up(max(renderer.height, biased), 16))


def ink_crop(img: Image.Image, margin: int = 8) -> Image.Image:
    """Crop a generated line to its rightmost ink + a small white margin (trim the blank tail).

    Guards against over-cropping an undertrained (near-white) output: if no column passes the ink
    threshold, the image is returned unchanged.
    """
    inked = (np.asarray(img.convert("L")) < _INK_THRESHOLD).any(axis=0)  # [W] True where a column has ink
    if not inked.any():
        return img  # blank / near-white output: don't crop to a sliver
    right = int(np.flatnonzero(inked)[-1]) + 1
    return img.crop((0, 0, min(img.width, right + margin), img.height))


@torch.inference_mode()
def generate_line(
    model: Diffu,
    text: str,
    style: torch.Tensor | None,
    *,
    num_steps: int | None = None,
    cfg_scale: float = 0.0,
    ref: RefLine | None = None,
    style_cfg_scale: float = 0.0,
    ink_width: int | None = None,
) -> Image.Image:
    """Turn one text + style ref into a line image at the CANONICAL geometry (auto-width + ink-crop).

    The single source of truth for "text + style -> line image", so the deploy path (``generate`` /
    ``DiffuPipeline`` / the studio) and the eval scripts produce identical images and the measured CER is the
    CER of what actually ships. ``style`` is a preprocessed ``[1, 3, S, S]`` reference tensor; ``model`` is an
    already-built, loaded :class:`Diffu`.

    Canvas-mode models take ``ref`` instead (``style`` unused): the canvas is sized from the reference's
    own px-per-char — or ``ink_width`` (px) when given — which also sets the glyph extent with ``glyph_pad``.
    """
    cfg = model.cfg
    latent_h = cfg.data.line_height // cfg.vae.downscale_factor
    if not model.canvas:
        canvas_w = auto_canvas_width(model, text)
        img = model.generate(
            [text],
            style,
            latent_hw=(latent_h, canvas_w // cfg.vae.downscale_factor),
            num_steps=num_steps,
            cfg_scale=cfg_scale,
        )[0]
        return ink_crop(to_pil(img))
    if ref is None:
        raise ValueError("this checkpoint uses style_mode='canvas': pass ref=load_ref_line(...)")
    ink_w = min(cfg.data.max_line_width, ink_width or ref_text_width(ref, text, model))
    canvas_w = min(cfg.data.max_line_width, round_up(round(ink_w * _CANVAS_SLACK), 16))
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    img = model.generate(
        [text],
        None,
        latent_hw=(latent_h, canvas_w // cfg.vae.downscale_factor),
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        style_cfg_scale=style_cfg_scale,
        text_widths=[ink_w],
        ref_images=ref.image.to(device=dev, dtype=dtype),
        ref_texts=[ref.text],
        ref_widths=[ref.width],
    )[0]
    return ink_crop(to_pil(img))


def load_config(ckpt: str | Path | None) -> Config:
    """The training config saved with a checkpoint: ``config.json`` in the checkpoint dir (or next to the
    weights file). A checkpoint without one (older runs) gets the defaults — pass its arch flags then."""
    if ckpt is None:
        return Config()
    path = Path(ckpt)
    cfg_path = (path if path.is_dir() else path.parent) / "config.json"
    if not cfg_path.is_file():
        return Config()
    return Config.model_validate_json(cfg_path.read_text(encoding="utf-8"))


def load_checkpoint(model: Diffu, ckpt: str) -> None:
    """Load model weights from a safetensors file (non-strict).

    Keys whose tensor SHAPE no longer matches are dropped rather than loaded (``load_state_dict`` errors on a
    shape mismatch even with ``strict=False``), except the input projection: a checkpoint reading a different
    number of input channels than the model was trained with other architecture flags, and dropping it would
    leave the input layer random — so that raises (a narrower one is widened, see ``_widen_input_channels``).
    """
    if not ckpt.endswith(".safetensors"):
        raise ValueError(f"expected a .safetensors checkpoint, got {ckpt}")
    state = load_file(ckpt)
    own = model.state_dict()
    widened = _widen_input_channels(state, own)
    mismatched = [k for k, v in state.items() if k in own and v.shape != own[k].shape]
    if _INPUT_PROJ in mismatched:
        raise RuntimeError(
            f"{ckpt} reads {state[_INPUT_PROJ].shape[1]} input channels but this model reads "
            f"{own[_INPUT_PROJ].shape[1]}: it was trained with a different --style-canvas setting"
        )
    for k in mismatched:
        del state[k]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if widened:
        print(f"widened {widened}: old input channels kept, new ones zero-initialised")
    if missing or unexpected or mismatched:
        print(
            f"loaded with {len(missing)} missing / {len(unexpected)} unexpected / "
            f"{len(mismatched)} shape-mismatched keys (expected for frozen enc + resized seq len)"
        )


def _widen_input_channels(state: dict[str, torch.Tensor], own: dict[str, torch.Tensor]) -> list[str]:
    """Warm-start a backbone that READS more input channels than the checkpoint (``style_mode="canvas"``
    adds the reference channels): copy the trained patch-embed weights into the leading channels and zero
    the new ones, so the model starts out exactly as the checkpoint (the extra input contributes nothing)
    and learns to use it — the SD-inpainting / ControlNet recipe — instead of re-initialising the whole input
    layer at random.
    """
    old, new = state.get(_INPUT_PROJ), own.get(_INPUT_PROJ)
    if old is None or new is None or old.shape == new.shape:
        return []
    if (
        old.dim() != 4
        or old.shape[0] != new.shape[0]
        or old.shape[2:] != new.shape[2:]
        or old.shape[1] > new.shape[1]
    ):
        return []
    weight = torch.zeros(new.shape, dtype=old.dtype)
    weight[:, : old.shape[1]] = old
    state[_INPUT_PROJ] = weight
    return [_INPUT_PROJ]


def assert_conditioner_loaded(model: Diffu, ckpt: str) -> None:
    """Fail fast if a checkpoint's TEXT-CONDITIONER (``glyph_content``) doesn't fit this code's architecture.

    ``load_checkpoint`` is non-strict, so a checkpoint trained with a different conditioner silently leaves
    that module at its RANDOM init — the model then generates confident-looking GIBBERISH that no metric on
    the output image catches (this masked a whole debugging session). Raise only on a genuine architecture
    change: much of the conditioner absent AND the checkpoint carrying differently-named conditioner
    weights — never the benign frozen-and-legitimately-absent case.
    """
    with safe_open(ckpt, framework="pt") as handle:
        ckpt_keys = set(handle.keys())
    own = set(model.state_dict())
    cond = [k for k in own if k.startswith("glyph_content.")]
    missing = [k for k in cond if k not in ckpt_keys]
    renamed = [k for k in ckpt_keys - own if k.startswith("glyph_content.")]
    if cond and len(missing) > 0.5 * len(cond) and renamed:
        raise RuntimeError(
            f"checkpoint architecture mismatch: {len(missing)}/{len(cond)} text-conditioner (glyph_content) "
            f"weights are absent and {len(renamed)} differently-named ones are present, so the conditioner "
            f"stays random and the model generates gibberish. This checkpoint predates the glyph-line content "
            f"encoder or was trained with other arch flags.\n  checkpoint: {ckpt}"
        )


def build_model(
    cfg: Config,
    ckpt: str | None = None,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    lora: str | None = None,
    lora_scale: float = 1.0,
    fuse_qkv: bool = True,
    attention_backend: str | None = None,
    compile_mode: str | None = None,
    compile_vae: bool = False,
    compile_backbone: bool = False,
) -> Diffu:
    """The inference model: built from ``cfg`` (REPA off — training-only), weights loaded and checked, then
    the optional LoRA adapter, fused Q/K/V, attention backend and compiles, in the order they require.

    The one loader behind ``diffu-generate``, ``DiffuPipeline``, ``diffu-synth`` and the studio/page/runner.
    ``compile_backbone`` regionally compiles the MMDiT blocks (``dynamic=True``: one compile covers every
    width) — a cold-start cost that only pays off over many lines.
    """
    cfg = cfg.model_copy(deep=True)
    cfg.aux.repa = False  # training-only: don't build its DINOv3 (a canvas model needs no DINO at all)
    if ckpt:  # the checkpoint holds the trained glyph CNN: skip the ImageNet download (offline hosts)
        cfg.cond.glyph_cnn_pretrained = False
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # TF32 matmuls for fp32 inference — standalone inference would otherwise run every GEMM strict-fp32.
    torch.set_float32_matmul_precision("high")
    model = Diffu(cfg).to(device=device, dtype=dtype).eval()
    if ckpt:
        load_checkpoint(model, ckpt)
        assert_conditioner_loaded(model, ckpt)  # fail fast: a mismatched arch generates gibberish
    if lora:  # a Stage-3 LoRA adapter (a dir or Hub id) on the base — before compile
        from .lora import load_lora

        load_lora(model, lora, scale=lora_scale)
    if fuse_qkv and not lora:  # exact; a LoRA wraps the unfused projections, so keep those then
        model.backbone.fuse_qkv_projections()
    if attention_backend:  # before compile: the backend is traced into the compiled graph
        model.backbone.set_attention_backend(attention_backend)
    if compile_vae:
        model.vae.compile_codec(compile_mode)
    if compile_backbone:
        model.backbone.compile_blocks(compile_mode)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate handwritten line images with Diffu.")
    ap.add_argument("--text", nargs="+", required=True, help="one or more line strings to render")
    ap.add_argument("--style", required=True, help="reference line crop of the target hand")
    ap.add_argument("--out", default="generated.png", help="output PNG (or prefix when >1 text)")
    ap.add_argument("--ckpt", default=None, help="model weights (.safetensors); omit for an untrained smoke")
    ap.add_argument(
        "--width",
        type=int,
        default=None,
        help="fixed width in px instead of the per-text auto width: the canvas for dino models (no "
        "ink-crop), the written length for canvas models",
    )
    ap.add_argument("--steps", type=int, default=None, help="denoising steps (default: the config's 24)")
    ap.add_argument(
        "--cfg-scale",
        type=float,
        default=5.0,
        help="classic CFG (text-following); 0=off. cfg=5 measured best on a val sweep (gen_CER 0.25 vs 0.33 "
        "at cfg=3, step_160000); cfg=7 over-saturates. See docs/PERF_AUDIT.md §0c.",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the transformer blocks (−70%% kernel launches; best for many lines — one "
        "compile with dynamic shapes covers every width). See docs/PERF_AUDIT.md §0e.",
    )
    ap.add_argument("--style-text", default="", help="--style-canvas: transcription of the --style line")
    ap.add_argument(
        "--style-cfg-scale",
        type=float,
        default=0.0,
        help="--style-canvas: style guidance (needs a model trained with --ref-dropout); 0=off",
    )
    ap.add_argument(
        "--attention-backend", default=None, help="diffusers attention kernel (e.g. _native_cudnn, flash_hub)"
    )
    ap.add_argument(
        "--compile-mode",
        choices=("reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
        default=None,
        help="torch.compile mode for --compile / --compile-vae: reduce-overhead / max-autotune add CUDA graphs "
        "(one per line width)",
    )
    ap.add_argument("--compile-vae", action="store_true", help="torch.compile the VAE decode")
    ap.add_argument(
        "--no-fuse-qkv",
        dest="fuse_qkv",
        action="store_false",
        help="keep separate Q/K/V matmuls (default: fused)",
    )
    ap.add_argument(
        "--lora", default=None, help="LoRA adapter dir or Hub id to load onto the base checkpoint"
    )
    ap.add_argument("--lora-scale", type=float, default=1.0, help="how strongly to apply the LoRA (0..1+)")
    add_arch_args(ap)
    args = ap.parse_args()

    cfg = apply_arch_args(load_config(args.ckpt), args)
    model = build_model(
        cfg,
        args.ckpt,
        lora=args.lora,
        lora_scale=args.lora_scale,
        fuse_qkv=args.fuse_qkv,
        attention_backend=args.attention_backend,
        compile_mode=args.compile_mode,
        compile_vae=args.compile_vae,
        compile_backbone=args.compile,
    )
    imgs = _generate_all(model, args)
    out = Path(args.out)
    if len(imgs) == 1:
        imgs[0].save(out)
        print(f"wrote {out}")
    else:
        for i, img in enumerate(imgs):
            p = out.with_name(f"{out.stem}_{i}{out.suffix}")
            img.save(p)
            print(f"wrote {p}")


@torch.inference_mode()
def _generate_all(model: Diffu, args: argparse.Namespace) -> list[Image.Image]:
    """One image per ``--text`` at its own canonical geometry (``generate_line``), or — dino models with
    ``--width`` — all texts on one fixed-width canvas without ink-crop."""
    if model.canvas:
        ref = load_ref_line(args.style, model.cfg, text=args.style_text)
        return [
            generate_line(
                model,
                text,
                None,
                num_steps=args.steps,
                cfg_scale=args.cfg_scale,
                ref=ref,
                style_cfg_scale=args.style_cfg_scale,
                ink_width=args.width,
            )
            for text in args.text
        ]
    style = load_style(args.style).unsqueeze(0).to(next(model.parameters()).device)  # [1, 3, S, S]
    if args.width is None:
        return [
            generate_line(model, t, style, num_steps=args.steps, cfg_scale=args.cfg_scale) for t in args.text
        ]
    cfg = model.cfg
    latent_hw = (
        cfg.data.line_height // cfg.vae.downscale_factor,
        round_up(args.width, 16) // cfg.vae.downscale_factor,
    )
    imgs = model.generate(
        args.text,
        style.expand(len(args.text), -1, -1, -1),
        latent_hw=latent_hw,
        num_steps=args.steps,
        cfg_scale=args.cfg_scale,
    )  # [B, 3, H, W]
    return [to_pil(img) for img in imgs]


if __name__ == "__main__":
    main()
