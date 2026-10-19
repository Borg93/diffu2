"""Config for Diffu (pydantic models) + the architecture CLI flags shared by training and every loader."""

from __future__ import annotations

import argparse
from typing import Literal

from pydantic import BaseModel, Field


class VAEConfig(BaseModel):
    # PICK = Qwen-Image VAE (16-ch f8, text-tuned decoder, Apache-2.0, ungated). Validated in Stage 0
    # (preserves å ä ö). It's Wan/video-derived (5D) -> vae.py wraps images with a T=1 frame dim.
    pretrained: str = "Qwen/Qwen-Image"
    subfolder: str = "vae"
    video_vae: bool = True
    latent_channels: int = 16
    downscale_factor: int = 8
    # decoded-vs-real CER must stay within +0.05 (5pp) of raw; measured +0.008 (passes)
    recon_cer_gate: float = 0.05


class BackboneConfig(BaseModel):
    # Backbone = diffusers SD3 MMDiT (SD3Transformer2DModel, model/backbone.py) — not hand-rolled — with its
    # absolute pos_embed table replaced by 2D RoPE on the image Q/K (model/rope.py): relative, so variable-width
    # lines have no width cap.
    dim: int = 1024
    num_layers: int = 24
    heads: int = 16
    patch: int = 2
    context_dim: int = 1024  # joint_attention_dim = the glyph token dim (also the pooled AdaLN dim)
    rope_theta: float = 10000.0  # RoPE base frequency (diffusers/Flux default)
    # SD3.5 (vs SD3.0) attention upgrades — diffusers builds them when set.
    #   qk_norm="rms_norm": RMSNorm on Q/K for training stability — the defining SD3.5 feature.
    #   dual_attention_layers: the heavier SD3.5-medium / MMDiT-X extra image-only self-attention in the
    #     listed blocks (e.g. tuple(range(13)) ≈ SD3.5-medium); () = off (SD3.5-large style; no CLI flag).
    # RoPEJointAttnProcessor rotates both the joint attn and the dual attn2.
    qk_norm: str | None = "rms_norm"
    dual_attention_layers: tuple[int, ...] = ()
    fill_ratio_cond: bool = True  # add a text-extent/canvas-width scalar to the pooled AdaLN vector


class ConditioningConfig(BaseModel):
    # Content = the Unifont glyph-LINE encoder: the whole line rendered as one image -> CNN feature map -> one
    # content token per image patch-column, with shared-column 2D-RoPE matching the image columns (Qwen-Image
    # MSRoPE), so the image attends to the glyph at its OWN column. Style = DINOv3 or the canvas reference.
    # Max characters per line rendered into the glyph strip. NOTE: the deeper limit on very long lines is
    # canvas width (data.max_line_width) + training data at that width — not this cap alone.
    max_chars: int = 128
    unifont_path: str | None = None  # GNU-Unifont TTF (covers all BMP glyphs incl. å ä ö + historical)
    glyph_size: int = 32
    glyph_cnn_pretrained: bool = (
        True  # ImageNet-init the glyph-line ResNet18 trunk (warm start); False = scratch
    )
    style_encoder: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    style_tokens: int = 64  # Perceiver latents the DINO features are resampled to before attention-pooling
    # glyph_pad: place each line's glyph strip over ITS OWN extent (the real line width at train time, an
    # estimate at inference) and leave the remaining columns blank, instead of stretching the strip over the
    # whole canvas. Glyph column j then lands on the image column where that letter is actually written, and
    # the glyph extent doubles as a length control (blank columns = "write nothing here").
    glyph_pad: bool = False
    # style_mode: how the writer reference reaches the model.
    #   "dino"   — frozen DINOv3 features of a 224x224-squashed crop -> resampler -> pooled AdaLN vector (every
    #              model trained so far).
    #   "canvas" — in-context (F5-TTS / DiffInk / Eruku style): the reference line's clean VAE latent sits
    #              LEFT of the target on one canvas, fed as extra conditioning channels, with its own
    #              transcription in the glyph tokens. The model continues the line in the same hand; the loss
    #              covers only the target columns. No style encoder.
    style_mode: Literal["dino", "canvas"] = "dino"


class FlowConfig(BaseModel):
    # Rectified flow / flow matching; v = eps - x0. Timesteps are logit-normal (flow.sample_timesteps).
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0
    sample_steps: int = 24  # denoising steps for eval/sample generation (inference scheduler)


class AuxLossConfig(BaseModel):
    # REPA token-wise representation alignment (ICLR'25): align the DiT's per-token hidden states to
    # DINOv3 per-patch features of the target line (at its native size: DINO's 16-px patches ARE the DiT
    # token grid). ON by default — faster convergence + finer detail early on. Disable with --no-repa.
    # With gradient checkpointing it must stay NON-REENTRANT (diffusers' default) — reentrant mode detaches
    # the hook capture; Diffu's runtime guard covers that case.
    repa: bool = True
    repa_weight: float = 0.5  # weight of the REPA cosine-alignment term
    repa_layer: int = 8  # which SD3 transformer block's hidden state to align (0-indexed)
    repa_stop_frac: float = 0.4  # terminate REPA after this fraction of training (HASTE); 1.0 = never.
    # NOTE: when <1.0 under multi-GPU DDP, repa_proj goes unused at the stop step — the Accelerator's
    # find_unused_parameters=True (train.py) is what lets that early-stop happen without crashing DDP.
    # ink-focal flow loss: up-weight the sparse ink cells so the ~90% paper background doesn't dominate the
    # velocity MSE. On in every production run so far (exp_base_weekend, the sd35 runs); --no-ink-focal.
    ink_focal: bool = True
    ink_weight: float = 3.0  # loss weight on a full-ink latent cell (paper = 1)
    ink_threshold: float = 0.3  # [-1,1] grayscale below this = ink (else tan/white paper)
    # The legibility gauge for the live gen_CER and the offline eval/VAE gate: the char-level historical-
    # Swedish TrOCR (a char vocab generalizes out-of-domain, and generated lines ARE out-of-domain).
    htr_recognizer_eval: str = "Riksarkivet/trocr-large-handwritten-hist-swe-3-char"


class DataConfig(BaseModel):
    line_height: int = (
        128  # 128px (2x native detail; native lines ~217px). 64px converged too (exp_sd35_fast).
    )
    max_line_width: int = 1024
    # A line wider than max_line_width is squashed to it (aspect lost) unless drop_wide_lines skips it. The
    # model has no width limit of its own (2D RoPE + column-aligned glyph tokens); this cap bounds memory.
    drop_wide_lines: bool = False
    # White-pad augmentation (train split ONLY, off by default): right-pad a line with extra white
    # before batch-width bucketing, so the model sees short text on a wider canvas at train time and
    # learns to leave the tail blank (paired with the fill_ratio scalar that explains the blank space).
    white_pad_prob: float = 0.0  # per-line probability of applying the right-white-pad augmentation
    white_pad_max_frac: float = 0.5  # max extra width as a fraction of the line's natural width
    # style_mode="canvas": widest reference line placed left of the target. A wider reference is cropped and
    # its transcription dropped (the text would no longer match the pixels). The canvas is ref + target wide.
    max_ref_width: int = 1024


class Config(BaseModel):
    vae: VAEConfig = Field(default_factory=VAEConfig)
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    cond: ConditioningConfig = Field(default_factory=ConditioningConfig)
    flow: FlowConfig = Field(default_factory=FlowConfig)
    aux: AuxLossConfig = Field(default_factory=AuxLossConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    seed: int = 42


def add_arch_args(ap: argparse.ArgumentParser) -> None:
    """The architecture flags shared by ``diffu-train`` and the checkpoint loaders. Each defaults to unset:
    training then keeps the Config default, a loader the checkpoint's ``config.json``."""
    g = ap.add_argument_group("architecture (loaders read it from the checkpoint's config.json)")
    g.add_argument("--line-height", type=int, default=None, help="line crop height in px (default 128)")
    g.add_argument(
        "--style-canvas",
        action="store_true",
        default=None,
        help="in-context style: the reference line's clean latent sits left of the target on one canvas",
    )
    g.add_argument(
        "--glyph-pad",
        action="store_true",
        default=None,
        help="glyph tokens over each line's own ink extent, blank past it (a length control)",
    )
    g.add_argument(
        "--no-fill-ratio",
        dest="fill_ratio",
        action="store_false",
        default=None,
        help="no text-extent/canvas-width scalar in the pooled AdaLN vector",
    )
    g.add_argument(
        "--max-line-width",
        type=int,
        default=None,
        help="cap on the line width in px (default 1024); wider lines are squashed unless --drop-wide-lines",
    )
    g.add_argument(
        "--max-ref-width", type=int, default=None, help="--style-canvas: cap on the reference width (px)"
    )
    g.add_argument("--max-chars", type=int, default=None, help="max characters in the glyph strip (128)")


def apply_arch_args(cfg: Config, args: argparse.Namespace) -> Config:
    """Write the architecture flags that were given (see :func:`add_arch_args`) onto ``cfg``; returns it."""
    if args.line_height is not None:
        cfg.data.line_height = args.line_height
    if args.style_canvas:
        cfg.cond.style_mode = "canvas"
    if args.glyph_pad:
        cfg.cond.glyph_pad = True
    if args.fill_ratio is not None:
        cfg.backbone.fill_ratio_cond = args.fill_ratio
    if args.max_line_width is not None:
        cfg.data.max_line_width = args.max_line_width
    if args.max_ref_width is not None:
        cfg.data.max_ref_width = args.max_ref_width
    if args.max_chars is not None:
        cfg.cond.max_chars = args.max_chars
    return cfg
