"""Config for Diffu (pydantic models)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VAEConfig(BaseModel):
    # PICK = Qwen-Image VAE (16-ch f8, text-tuned decoder, Apache-2.0, ungated). Validated in Stage 0
    # (preserves å ä ö). It's Wan/video-derived (5D) -> vae.py wraps images with a T=1 frame dim.
    pretrained: str = "Qwen/Qwen-Image"
    subfolder: str = "vae"
    video_vae: bool = True
    latent_channels: int = 16
    downscale_factor: int = 8
    finetune_decoder_only: bool = True  # encoder frozen; decoder fine-tuned on Swedish ink
    recon_cer_gate: float = (
        0.05  # decoded-vs-real CER must stay within +0.05 (5pp) of raw; measured +0.008 (passes)
    )


class BackboneConfig(BaseModel):
    # Backbone = diffusers SD3 MMDiT (SD3Transformer2DModel, model/backbone.py) — not hand-rolled.
    dim: int = 1024
    num_layers: int = 24
    heads: int = 16
    patch: int = 2
    context_dim: int = 1024  # joint_attention_dim = our (content + style) token dim
    sample_size: int = 128
    pos_embed_max_size: int = 192  # ONLY used when rope=False; the rope path drops this width cap
    # With rope=True the SD3 absolute (sincos) pos_embed table is removed and 2D RoPE is applied to the
    # image/latent token Q/K instead (model/rope.py). RoPE is relative, so variable-width lines are no
    # longer capped at pos_embed_max_size tokens — the right fit for our variable-width handwriting and
    # the root fix for the short-text-on-wide-canvas overflow (no absolute table to extrapolate off).
    rope: bool = True
    rope_theta: float = 10000.0  # RoPE base frequency (diffusers/Flux default)
    # SD3.5 (vs SD3.0) attention upgrades — diffusers builds them when set.
    #   qk_norm="rms_norm": RMSNorm on Q/K for training stability — the defining SD3.5 feature, ON by
    #     default so we train SD3.5, not vanilla SD3.0. (It adds params, so an SD3.0 checkpoint won't
    #     load into this config — fine, the next run is from scratch.)
    #   dual_attention_layers: the heavier SD3.5-medium / MMDiT-X extra image-only self-attention in the
    #     listed blocks (e.g. tuple(range(13)) ≈ SD3.5-medium); () = off (SD3.5-large style) — left off by
    #     default (extra params/compute, overfit risk for from-scratch limited data; opt in per run).
    # RoPEJointAttnProcessor already rotates both the joint attn and the dual attn2 (tested).
    qk_norm: str | None = "rms_norm"
    dual_attention_layers: tuple[int, ...] = ()
    fill_ratio_cond: bool = True  # add a text-extent/canvas-width scalar to the pooled AdaLN vector


class ConditioningConfig(BaseModel):
    # Content = Unifont glyph encoder. Style = DINOv3 (+DINOv2 fallback).
    # Max characters per line. The glyph encoder pads only to the batch's longest text (capped here),
    # so raising this to 128/256 for long lines is cheap for short lines. NOTE: the deeper limit on
    # very long lines is canvas width (data.max_line_width) + having training data at that width +
    # 2D-RoPE extrapolating past trained widths — not this cap alone.
    max_chars: int = 128
    unifont_path: str | None = None  # GNU-Unifont TTF (covers all BMP glyphs incl. å ä ö + historical)
    # Render content glyphs with GNU-Unifont (full BMP). False = PIL's default font — REQUIRED to match a
    # model trained before Unifont was wired (e.g. exp_base_weekend); the stencil font must equal training's.
    use_unifont: bool = True
    glyph_size: int = 32
    glyph_cnn_pretrained: bool = True  # ImageNet-init the per-glyph ResNet18 (warm start); False = scratch
    # glyph_line: render the WHOLE line as one image -> CNN feature map -> w_t column-aligned content
    # tokens (line-level content), with shared-column 2D-RoPE matching the image columns
    # (Qwen-Image MSRoPE) so the image attends to the glyph at its OWN column — text<->position
    # alignment for free. OFF by default (the per-char token path stays the default until validated).
    glyph_line: bool = False
    # glyph_concat: render the line -> a small conv "glyph block" -> a latent [B, Cg, h, w] that is
    # CHANNEL-CONCATENATED onto the noisy latent (DiffInk "fuse content into the input" — the
    # strongest coupling: content is present at every cell, not just attended, and inherits the image
    # RoPE for free). Backbone in_channels grows by glyph_concat_channels; out stays = latent_channels.
    # OFF by default (experimental, untested). Pair with --cond-dropout 0.1 for the DiffInk concat+CFG recipe.
    glyph_concat: bool = False
    glyph_concat_channels: int = 16
    style_encoder: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    style_encoder_fallback: str = "facebook/dinov2-with-registers-large"  # Apache-2.0, ungated
    style_tokens: int = 64
    # Inject style ONLY as the global pooled AdaLN vector, NOT as the K attendable
    # style tokens in joint attention. The K spatial tokens can carry the reference image's GLYPH SHAPES,
    # so the model copies them and ignores the text (hold the style ref, swap the text -> output unchanged).
    # Pooled-only forces content to come from the text/glyph path. True = legacy (style tokens in context).
    style_in_context: bool = True


class FlowConfig(BaseModel):
    # Rectified flow / flow matching; v = eps - x0. Timesteps are logit-normal (flow.sample_timesteps).
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0
    sample_steps: int = 24  # denoising steps for eval/sample generation (inference scheduler)


class AuxLossConfig(BaseModel):
    # REPA token-wise representation alignment (ICLR'25): align the DiT's per-token hidden states to
    # DINO per-patch features of the target. ON by default — its whole point is ~faster convergence +
    # finer detail (diacritics), exactly what a near-from-scratch DiT needs. Disable with --no-repa.
    # Without grad checkpointing (--no-grad-checkpoint) the hook-captured block output carries plain
    # autograd and the alignment gradient reaches the backbone directly. IF checkpointing is enabled it
    # must be NON-REENTRANT (Backbone.enable_optimizations) — reentrant mode detaches hook captures;
    # Diffu's runtime guard covers that case.
    repa: bool = True
    repa_weight: float = 0.5  # weight of the REPA cosine-alignment term
    repa_layer: int = 8  # which SD3 transformer block's hidden state to align (0-indexed)
    repa_stop_frac: float = 0.4  # terminate REPA after this fraction of training (HASTE); 1.0 = never.
    # NOTE: when <1.0 under multi-GPU DDP, repa_proj goes unused at the stop step — the Accelerator's
    # find_unused_parameters=True (train.py) is what lets that early-stop happen without crashing DDP.
    diacritic_class_weight: float = 3.0  # loss up-weight on ink cells when ink-focal flow is on
    ink_threshold: float = 0.3  # [-1,1] grayscale below this = ink (else tan/white paper)
    diacritic_focal_flow: bool = False  # ink-focal flow loss (up-weight sparse ink so bg doesn't dominate)
    # 2026 Swedish Lion, char-based: SOTA historical-Swedish HTR — the live gen_CER / offline eval
    # legibility gauge (read_lines + char_error_rate). Char vocab generalizes out-of-domain (our
    # generated lines ARE out-of-domain). Needs interpolate_pos_encoding=True at generate (1024x192).
    htr_recognizer_eval: str = "Riksarkivet/trocr-base-handwritten-hist-swe-3"


class DataConfig(BaseModel):
    line_height: int = (
        128  # 128px run (2x native detail; native lines ~217px). 64px converged (exp_sd35_fast).
    )
    max_line_width: int = 1024
    # White-pad augmentation (train split ONLY, off by default): right-pad a line with extra white
    # before batch-width bucketing, so the model sees short text on a wider canvas at train time and
    # learns to leave the tail blank (paired with the fill_ratio scalar that explains the blank space).
    # NOTE: the CLI flag is the actual control — train.py passes --white-pad-prob straight to
    # build_dataloader (train loader only) → collate. This FIELD is record-keeping: main() writes the
    # flag back here so the run-config log captures it. build_dataloader does read
    # cfg.data.white_pad_max_frac from config, but takes white_pad_prob as a parameter.
    white_pad_prob: float = 0.0  # per-line probability of applying the right-white-pad augmentation
    white_pad_max_frac: float = 0.5  # max extra width as a fraction of the line's natural width


class Config(BaseModel):
    vae: VAEConfig = Field(default_factory=VAEConfig)
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    cond: ConditioningConfig = Field(default_factory=ConditioningConfig)
    flow: FlowConfig = Field(default_factory=FlowConfig)
    aux: AuxLossConfig = Field(default_factory=AuxLossConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    seed: int = 42
