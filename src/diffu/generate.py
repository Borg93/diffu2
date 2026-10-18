"""Generate handwritten line images from text + a writer style reference.

Wraps ``Diffu.generate`` (CFG-guided flow sampling) as the ``diffu-generate`` CLI:

    diffu-generate --text "Smörgåsbord" --style ref_line.png --out smorgas.png --ckpt run/final/model.safetensors

Without ``--ckpt`` it runs an untrained model (plumbing smoke only). The style reference is any line
crop of the target hand; it is ImageNet-normalized to the DINO input the model expects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .config import Config
from .model import Diffu
from .model.conditioning import GlyphLineRenderer

# DINOv2/v3 expect ImageNet-normalized 224x224 input (matches data/dataset.py style refs).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_WIDTH_BIAS = 0.45  # canvas-density knob. The model has NO length control — it FILLS any canvas it is
# given (no internal "write N chars and stop"): an over-wide canvas is filled by LOOPING characters
# (garbled repeats — see scripts/diag_width_sweep.py), an over-tight one crams. CRUCIAL: the raw estimator
# scales width with line-height (natural_width: size = height*0.7), so this bias is HEIGHT-COUPLED and must
# be re-tuned whenever line_height changes. 0.55 hit the GT density (~15 px/char) at the original 64px, but
# at the current 128px that same 0.55 lands at ~29 px/char — the catastrophic looping regime (CER-vs-density
# is monotonic: 30/24/18/15 px/char -> 0.41/0.18/0.09/0.037). 0.45 (~24 px/char at 128px) is the empirically
# clean point across Swedish + English — no looping, no cramming (verified: scripts/diag_bias_pick.py). The
# proper fix is a real length-conditioning channel; until then, RE-TUNE this bias per line_height.
_INK_THRESHOLD = 200  # 0-255 grayscale; a column with any pixel below this counts as inked


def load_style(path: str, size: int = 224) -> torch.Tensor:
    """Load a style reference line crop -> ImageNet-normalized ``[3, S, S]`` tensor."""
    with Image.open(path) as im:
        img = im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return TF.normalize(TF.to_tensor(img), _IMAGENET_MEAN, _IMAGENET_STD)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def to_pil(image: torch.Tensor) -> Image.Image:
    """``[3, H, W]`` in ``[-1, 1]`` -> PIL RGB. ``.float()``: torchvision can't convert bf16 tensors."""
    return TF.to_pil_image(((image.clamp(-1, 1) + 1) / 2).float().cpu())


def natural_width(renderer: GlyphLineRenderer, text: str, cfg: Config) -> int:
    """Auto canvas width (px) for ``text``: glyph natural width × bias, rounded to 16, capped.

    Shared by generate.py and app.py so the CLI and UI compute width identically. Generously wide so
    the model has room; the resulting blank tail is removed by :func:`ink_crop`.
    """
    biased = int(renderer.natural_width(text) * _WIDTH_BIAS)
    return min(cfg.data.max_line_width, _round_up(max(renderer.height, biased), 16))


def ink_crop(img: Image.Image, margin: int = 8) -> Image.Image:
    """Crop a generated line to its rightmost ink + a small white margin (trim the blank tail).

    Guards against over-cropping an undertrained (near-white) output: if no column passes the ink
    threshold, the image is returned unchanged.
    """
    import numpy as np

    arr = np.asarray(img.convert("L"))
    inked = (arr < _INK_THRESHOLD).any(axis=0)  # [W] True where a column has ink
    if not inked.any():
        return img  # blank / near-white output: don't crop to a sliver
    right = int(inked.nonzero()[0].max()) + 1
    crop_w = min(img.width, right + margin)
    return img.crop((0, 0, crop_w, img.height))


@torch.inference_mode()
def generate_line(
    model: Diffu,
    text: str,
    style: torch.Tensor,
    *,
    cfg: Config,
    num_steps: int = 24,
    cfg_scale: float = 0.0,
) -> Image.Image:
    """Turn one text + style ref into a line image at the CANONICAL geometry (auto-width + ink-crop).

    The single source of truth for "text + style -> line image", so the deploy path (``generate`` / the
    app) and the eval ruler (``scripts/eval_cer.py``) produce identical images and the measured CER is
    the CER of what actually ships. ``style`` is a preprocessed ``[1, 3, S, S]`` reference tensor;
    ``model`` is an already-built, loaded :class:`Diffu`.
    """
    canvas_w = natural_width(model.guidance_renderer, text, cfg)
    latent_h = cfg.data.line_height // cfg.vae.downscale_factor
    latent_w = canvas_w // cfg.vae.downscale_factor
    img = model.generate(
        [text], style, latent_hw=(latent_h, latent_w), num_steps=num_steps, cfg_scale=cfg_scale
    )[0]
    return ink_crop(to_pil(img))


def load_checkpoint(model: Diffu, ckpt: str) -> None:
    """Load model weights from a safetensors or torch state_dict file (non-strict).

    Keys whose tensor SHAPE no longer matches (e.g. the glyph positional table ``glyph_content.pos``
    after changing ``cfg.cond.max_chars``) are dropped rather than loaded — ``load_state_dict`` errors
    on a shape mismatch even with ``strict=False`` — so a warm-start (``--init-from``) survives a
    changed sequence length, re-initialising only the resized parameter.
    """
    if ckpt.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(ckpt)
    else:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    own = model.state_dict()
    mismatched = [k for k, v in state.items() if k in own and v.shape != own[k].shape]
    for k in mismatched:
        del state[k]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected or mismatched:
        print(
            f"loaded with {len(missing)} missing / {len(unexpected)} unexpected / "
            f"{len(mismatched)} shape-mismatched keys (expected for frozen enc + resized seq len)"
        )


def assert_conditioner_loaded(model: Diffu, ckpt: str) -> None:
    """Fail fast if a checkpoint's TEXT-CONDITIONER (``glyph_content``) doesn't fit this code's architecture.

    ``load_checkpoint`` is non-strict, so a checkpoint trained with a different conditioner silently leaves
    that module at its RANDOM init — the model then generates confident-looking GIBBERISH that no metric on
    the output image catches (this masked a whole debugging session). Raise only on a genuine architecture
    change: much of the conditioner absent AND the checkpoint carrying differently-named conditioner
    weights — never the benign frozen-and-legitimately-absent case. (Promoted from diffu_page.render so the
    core loaders — CLI, DiffuPipeline — get the same guard the page renderer and studio rely on.)
    """
    if not ckpt.endswith(".safetensors"):
        return
    from safetensors import safe_open

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
            f"stays random and the model generates gibberish. This checkpoint was trained with a different "
            f"model version/arch flags — fix the flags (--line-height/--no-glyph-line/--style-tokens) or "
            f"generate from a code-matching checkpoint.\n  checkpoint: {ckpt}"
        )


@torch.inference_mode()
def generate(
    texts: list[str],
    style_path: str,
    *,
    cfg: Config,
    ckpt: str | None = None,
    width: int = 512,
    auto_width: bool = True,
    num_steps: int = 24,
    cfg_scale: float = 0.0,
    device: str | None = None,
    compile_model: bool = False,
    lora: str | None = None,
    lora_scale: float = 1.0,
) -> list[Image.Image]:
    """Generate one line image per text in the given writer's style.

    With ``auto_width`` (default), each line's canvas width comes from its own text length (the glyph
    renderer's natural width, biased wide and capped at ``cfg.data.max_line_width``); since
    ``model.generate`` takes one latent grid per call, lines are generated one at a time at their own
    width, then ink-cropped so the generously-wide canvas yields a tight line. With ``auto_width=False``
    the fixed ``width`` is used for all texts (no ink-crop) — the manual override.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # TF32 matmuls for fp32 inference — train.py sets this for training, but standalone inference
    # (CLI/pipeline/eval) never did, leaving every GEMM on the strict-fp32 path (several-fold slower).
    torch.set_float32_matmul_precision("high")
    model = Diffu(cfg).to(device).eval()
    if ckpt:
        load_checkpoint(model, ckpt)
        assert_conditioner_loaded(model, ckpt)  # fail fast: a mismatched arch generates gibberish
    if lora:  # load a Stage-3 LoRA adapter (a dir or Hub id) onto the base — before compile
        from .lora import load_lora

        load_lora(model, lora, scale=lora_scale)
    if compile_model:
        # Fuse the per-block glue (−70% kernel launches, §0e). compile_blocks is dynamic=True — ONE
        # compile covers the variable widths — but the cold-start cost still only pays off over MANY
        # lines (batch / served / page rendering), not a one-shot generation.
        model.backbone.compile_blocks()

    style1 = load_style(style_path).unsqueeze(0).to(device)  # [1, 3, S, S]
    latent_h = cfg.data.line_height // cfg.vae.downscale_factor

    if not auto_width:  # manual override: one fixed width for the whole batch, no ink-crop
        style = style1.expand(len(texts), -1, -1, -1)
        latent_w = _round_up(width, 16) // cfg.vae.downscale_factor
        imgs = model.generate(
            texts,
            style,
            latent_hw=(latent_h, latent_w),
            num_steps=num_steps,
            cfg_scale=cfg_scale,
        )  # [B, 3, H, W]
        return [to_pil(img) for img in imgs]

    out: list[Image.Image] = []
    for text in texts:  # per-text canonical geometry (auto-width + ink-crop), shared with the eval ruler
        out.append(generate_line(model, text, style1, cfg=cfg, num_steps=num_steps, cfg_scale=cfg_scale))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate handwritten line images with Diffu.")
    ap.add_argument("--text", nargs="+", required=True, help="one or more line strings to render")
    ap.add_argument("--style", required=True, help="reference line crop of the target hand")
    ap.add_argument("--out", default="generated.png", help="output PNG (or prefix when >1 text)")
    ap.add_argument("--ckpt", default=None, help="model weights (.safetensors/.pt); omit for untrained smoke")
    ap.add_argument(
        "--width", type=int, default=512, help="manual line width in px (only used with --no-auto-width)"
    )
    ap.add_argument(
        "--no-auto-width",
        action="store_true",
        help="disable per-text auto width + ink-crop; use the fixed --width for every text",
    )
    ap.add_argument("--steps", type=int, default=24)
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
        help="torch.compile the transformer blocks (−70%% kernel launches; best for many lines — adds "
        "one-time compile latency per distinct width). See docs/PERF_AUDIT.md §0e.",
    )
    # Arch flags — Config() defaults track the CURRENT training run, not the checkpoint being loaded;
    # the defaults here match the best released run (exp_sd35_fast: 64px, glyph-line, pooled style).
    ap.add_argument("--line-height", type=int, default=64, help="crop height the checkpoint trained at")
    ap.add_argument("--no-glyph-line", dest="glyph_line", action="store_false")
    ap.add_argument("--style-tokens", dest="style_in_context", action="store_true")
    ap.add_argument("--lora", default=None, help="LoRA adapter dir or Hub id to load onto the base checkpoint")
    ap.add_argument("--lora-scale", type=float, default=1.0, help="how strongly to apply the LoRA (0..1+)")
    ap.set_defaults(glyph_line=True, style_in_context=False)
    args = ap.parse_args()

    cfg = Config()
    cfg.cond.glyph_line = args.glyph_line
    cfg.cond.style_in_context = args.style_in_context
    cfg.data.line_height = args.line_height

    imgs = generate(
        args.text,
        args.style,
        cfg=cfg,
        ckpt=args.ckpt,
        width=args.width,
        auto_width=not args.no_auto_width,
        num_steps=args.steps,
        cfg_scale=args.cfg_scale,
        compile_model=args.compile,
        lora=args.lora,
        lora_scale=args.lora_scale,
    )

    out = Path(args.out)
    if len(imgs) == 1:
        imgs[0].save(out)
        print(f"wrote {out}")
    else:
        for i, img in enumerate(imgs):
            p = out.with_name(f"{out.stem}_{i}{out.suffix}")
            img.save(p)
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
