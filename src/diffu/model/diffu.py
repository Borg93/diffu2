"""Diffu — diffusers-first model assembly.

Diffu is a 2026 model for Swedish handwritten line-image generation: MMDiT backbone +
rectified flow + 16-ch VAE + glyph/DINOv3 conditioning + classic CFG.

  Backbone : diffusers SD3 MMDiT (model/backbone.py) or, with cfg.backbone.arch="flux2", the
             FLUX.2-style Flux2Transformer2DModel (model/backbone_flux2.py)   [not hand-rolled]
  VAE      : Qwen-Image 16-ch (model/vae.py, diffusers)          [validated in Stage 0]
  Scheduler: diffusers FlowMatchEulerDiscreteScheduler           [sampling]
  Content  : Unifont glyph encoder (model/content_encoder.py)    [custom — no diffusers equivalent]
  Style    : cfg.cond.style_mode
               "dino"   — DINOv3 + Perceiver resampler (model/conditioning.py)        [legacy]
               "canvas" — the reference line's clean latent LEFT of the target on one canvas,
                          as an extra input channel; the model continues the line   [in-context]
  Guidance : classifier-free guidance — text scale (+ a separate style scale in canvas mode)

Conditioning fed to SD3:
  dino   : encoder_hidden_states = concat(content tokens, style tokens)   [B, L+K, context_dim]
           pooled_projections    = pooled style vector                    [B, context_dim]
  canvas : hidden_states         = concat(noisy canvas, clean-ref cond)    [B, 2C, h, w_ref+w]
           encoder_hidden_states = glyph tokens of (ref text | target text), column-aligned to the canvas
           pooled_projections    = fill-ratio embedding (no style vector)

The diffusers wiring (SD3 timestep scale + FlowMatch step) is verified by the `diffu-smoke` synthetic forward/backward check
(one training forward + one generate on a GPU).
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from pydantic import BaseModel, ConfigDict

from ..config import Config
from ..flow import flow_loss, interpolate, repa_loss, sample_timesteps
from .backbone import Backbone
from .backbone_flux2 import Flux2Backbone
from .conditioning import GlyphLineRenderer, StyleEncoder
from .content_encoder import GlyphContentEncoder, GlyphLatentEncoder, GlyphLineContentEncoder
from .vae import VAEWrapper

# DINO expects ImageNet-normalized input; REPA aligns DiT hidden states to DINO features of the
# target image, so we re-normalize the [-1,1] target into the DINO convention before encoding it.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class CanvasRef(BaseModel):
    """The style reference for ``style_mode="canvas"``: its clean latent + what it says.

    ``latent`` is the normalized VAE latent ``[B, C, h, w_ref]`` placed left of the target. ``texts`` are
    the reference transcriptions ("" = unknown: its glyph tokens are zeroed). ``widths_px`` is each
    reference's real ink width in pixels (for ``glyph_pad``); ``None`` stretches the glyphs over the region.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    latent: torch.Tensor
    texts: list[str]
    widths_px: list[int] | None = None


class Diffu(nn.Module):
    """Full model: VAE + Unifont content + DINOv3 style + SD3 backbone."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.C = cfg.vae.latent_channels
        ctx = cfg.backbone.context_dim
        self.vae = VAEWrapper(cfg.vae)
        # Content path: per-char tokens (default) OR a whole-line glyph image -> w_t column-aligned tokens
        # with shared-column RoPE (cfg.cond.glyph_line; line-level column tokens + Qwen MSRoPE).
        self.glyph_line = cfg.cond.glyph_line
        self.glyph_content: nn.Module = (
            GlyphLineContentEncoder(cfg.cond, out_dim=ctx)
            if cfg.cond.glyph_line
            else GlyphContentEncoder(cfg.cond, out_dim=ctx)
        )
        self.canvas = cfg.cond.style_mode == "canvas"
        self.glyph_pad = cfg.cond.glyph_pad
        if (self.canvas or self.glyph_pad) and not cfg.cond.glyph_line:
            raise ValueError(
                "style_mode='canvas' and glyph_pad need the column-aligned glyph_line content path"
            )
        if self.canvas and cfg.cond.glyph_concat:
            raise ValueError(
                "style_mode='canvas' does not support glyph_concat (its glyph latent spans one text)"
            )
        # DINOv3: the style encoder in "dino" mode; in "canvas" mode only REPA's frozen target features need it.
        self.style: StyleEncoder | None = (
            StyleEncoder(cfg.cond, dim=ctx) if not self.canvas or cfg.aux.repa else None
        )
        self.style_in_context = cfg.cond.style_in_context  # False = pooled-only (no copyable style tokens)
        # glyph_concat: a spatial glyph latent channel-concatenated onto the noisy latent (fuse content
        # into the input). Backbone reads C+Cg channels, still predicts C velocity channels.
        self.glyph_latent = (
            GlyphLatentEncoder(cfg.cond, out_ch=cfg.cond.glyph_concat_channels)
            if cfg.cond.glyph_concat
            else None
        )
        extra = cfg.cond.glyph_concat_channels if cfg.cond.glyph_concat else 0
        # Canvas reference: on flux2 it is its own clean tokens with cached K/V (no extra channels); on sd3 it
        # shares the image grid and rides in conditioning channels (zeros over the target columns).
        self.ref_tokens = self.canvas and cfg.backbone.arch == "flux2"
        if self.canvas and not self.ref_tokens:
            extra += self.C
        backbone_cls = Flux2Backbone if cfg.backbone.arch == "flux2" else Backbone
        self.backbone: Backbone | Flux2Backbone = backbone_cls(
            cfg.backbone, in_channels=self.C + extra, context_dim=ctx, pooled_dim=ctx, out_channels=self.C
        )
        self.guidance_renderer = GlyphLineRenderer(
            cfg.cond, height=cfg.data.line_height
        )  # fill-ratio renderer

        # Fill-ratio conditioning: a scalar (text-extent / canvas-width) in [0,1] is embedded by a small
        # MLP and ADDED to the pooled AdaLN vector, so the model is told how much of a (possibly wide)
        # canvas the text actually fills — and learns to leave the rest blank instead of hallucinating a
        # tail. Train signal = ink extent of the target image; inference signal = natural text width /
        # canvas width (same glyph-renderer formula). Trained params are picked up in trainable_parameters.
        self.fill_ratio_mlp = (
            nn.Sequential(nn.Linear(1, ctx), nn.SiLU(), nn.Linear(ctx, ctx))
            if cfg.backbone.fill_ratio_cond
            else None
        )

        # REPA (optional): align a mid-block DiT hidden state to DINO features of the target image.
        # 3-layer SiLU MLP projector (REPA best-practice; a single Linear under-fits the DiT->DINO map).
        self.repa_proj = (
            nn.Sequential(
                nn.Linear(cfg.backbone.dim, cfg.backbone.dim),
                nn.SiLU(),
                nn.Linear(cfg.backbone.dim, cfg.backbone.dim),
                nn.SiLU(),
                nn.Linear(cfg.backbone.dim, self.style.enc.config.hidden_size),
            )
            if cfg.aux.repa and self.style is not None
            else None
        )

    def _token_extents(self, widths_px: list[int] | torch.Tensor | None) -> list[int] | None:
        """Pixel ink widths -> patch-column extents for ``glyph_pad`` (``None`` = stretch, the legacy path)."""
        if not self.glyph_pad or widths_px is None:
            return None
        unit = self.cfg.vae.downscale_factor * self.backbone.patch
        return [math.ceil(int(w) / unit) for w in widths_px]

    def _content(
        self,
        texts: list[str],
        device: torch.device,
        w_tokens: int | None,
        widths_px: list[int] | torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.glyph_line:
            return self.glyph_content(texts, device)  # [B, L, ctx]
        if w_tokens is None:
            raise ValueError("glyph_line content path needs w_tokens (the image patch-column count)")
        return self.glyph_content(
            texts, device, w_tokens, self._token_extents(widths_px)
        )  # [B, w_tokens, ctx]

    def _conditioning(
        self,
        texts: list[str],
        style_pixel_values: torch.Tensor | None,
        device: torch.device,
        fill_ratio: torch.Tensor | None = None,
        w_tokens: int | None = None,
        text_widths: list[int] | torch.Tensor | None = None,
        ref: CanvasRef | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Build joint context + pooled AdaLN vector. Returns ``(context, pooled, n_content)``.

        ``context`` is ``concat(content, style)`` along the token axis (content FIRST), so the first
        ``n_content`` tokens are the content tokens — the split point CFG drops (content only). If a
        ``fill_ratio`` is given and the MLP is enabled, its embedding is added to ``pooled``. In
        ``glyph_line`` mode ``w_tokens`` (the image patch-column count) is required and the content path
        emits exactly ``w_tokens`` column-aligned tokens.

        Canvas mode: the content is ``concat(ref glyph tokens, target glyph tokens)`` — one token per canvas
        column — and there are no style tokens; ``pooled`` carries only the fill-ratio embedding.
        """
        content = self._content(texts, device, w_tokens, text_widths)
        if self.canvas:
            if ref is None:
                raise ValueError("style_mode='canvas' needs a reference line (ref_images or ref_latents)")
            ref_tokens = ref.latent.shape[-1] // self.backbone.patch
            ref_content = self._content(ref.texts, device, ref_tokens, ref.widths_px)
            has_text = torch.tensor([bool(t) for t in ref.texts], device=device, dtype=ref_content.dtype)
            content = torch.cat([ref_content * has_text.view(-1, 1, 1), content], dim=1)
            pooled = content.new_zeros(content.shape[0], self.cfg.backbone.context_dim)
            style_tokens = None
        else:
            if self.style is None or style_pixel_values is None:
                raise ValueError("style_mode='dino' needs style_pixel_values")
            style_tokens, pooled = self.style(style_pixel_values)  # [B, K, ctx], [B, ctx]
        if self.fill_ratio_mlp is not None and fill_ratio is not None:
            fr = fill_ratio.to(pooled.dtype).view(-1, 1)  # [B, 1]
            pooled = pooled + self.fill_ratio_mlp(fr)
        # Keep the copyable spatial style tokens OUT of joint attention when style_in_context=False, so
        # content must come from the text/glyph path instead of being copied off the reference image.
        # (With the tokens in context the model copies the ref's glyphs and ignores the text instruction.)
        if style_tokens is not None and self.style_in_context:
            return torch.cat([content, style_tokens], dim=1), pooled, content.shape[1]
        return content, pooled, content.shape[1]

    def _canvas_ref(
        self,
        ref_images: torch.Tensor | None,
        ref_latents: torch.Tensor | None,
        ref_texts: list[str] | None,
        ref_widths: list[int] | torch.Tensor | None,
        batch: int,
    ) -> CanvasRef | None:
        """Bundle the canvas-mode reference (encoding ``ref_images`` unless a cached latent is given)."""
        if not self.canvas:
            return None
        if ref_latents is None:
            if ref_images is None:
                raise ValueError("style_mode='canvas' needs ref_images (or cached ref_latents)")
            ref_latents = self.vae.encode(ref_images)
        widths = None if ref_widths is None else [int(w) for w in ref_widths]
        return CanvasRef(
            latent=ref_latents, texts=list(ref_texts) if ref_texts else [""] * batch, widths_px=widths
        )

    # --- training ---------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        texts: list[str],
        style_pixel_values: torch.Tensor | None = None,
        *,
        latents: torch.Tensor | None = None,
        return_losses: bool = False,
        cond_dropout: float = 0.0,
        text_dropout: float = 0.0,
        ref_dropout: float = 0.0,
        repa_weight: float | None = None,
        text_widths: list[int] | torch.Tensor | None = None,
        ref_images: torch.Tensor | None = None,
        ref_latents: torch.Tensor | None = None,
        ref_texts: list[str] | None = None,
        ref_widths: list[int] | torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Stage-1 flow loss (+ optional REPA).

        ``return_losses`` returns ``{"loss", "flow"[, "repa"]}`` (detached components) so the training
        loop can log the **flow** term separately — that, not the REPA-inclusive total, is what is
        comparable to the eval-time val loss (REPA is training-only), keeping the overfitting check
        apples-to-apples.

        ``latents`` (optional) is a precomputed normalized VAE latent ``[B, C, h, w]`` from the
        ``--cache-latents`` path (data/latent_cache.py) — supplying it SKIPS the frozen f8 encoder
        (the single biggest removable op, ~23% of the step; see docs/PERF_AUDIT.md §0). ``None`` (the
        default) encodes ``images`` live, exactly as before. REPA / ink-focal / fill-ratio still read
        the raw ``images``, so the caller passes both.

        images ``[B,3,H,W]`` in ``[-1,1]``; texts ``list[str]``; style_pixel_values DINO-preprocessed
        (``style_mode="dino"`` only). ``text_widths`` are the target lines' real ink widths in px (for
        ``glyph_pad``). Canvas mode reads ``ref_images`` ``[B,3,H,W_ref]`` (or cached ``ref_latents``) +
        ``ref_texts`` / ``ref_widths``; ``ref_dropout`` drops the reference alone (the style-CFG mode).
        """
        x0 = self.vae.encode(images) if latents is None else latents  # [B, C, h, w] (normalized)
        b = x0.shape[0]
        ref = self._canvas_ref(ref_images, ref_latents, ref_texts, ref_widths, b)
        # Canvas mode, sd3: noise the whole [ref | target] canvas; the clean reference rides in a conditioning
        # channel and the loss is taken on the target columns only (F5-TTS / DiffInk infilling). flux2: only
        # the target is noised; the clean reference goes in as separate (cacheable) tokens.
        in_grid = ref is not None and not self.ref_tokens
        ref_cols = ref.latent.shape[-1] if in_grid else 0
        canvas_x0 = torch.cat([ref.latent.to(x0.dtype), x0], dim=-1) if in_grid else x0
        t = sample_timesteps(b, x0.device, self.cfg.flow.logit_normal_mean, self.cfg.flow.logit_normal_std)
        eps = torch.randn_like(canvas_x0)
        x_t, target_v = interpolate(canvas_x0, eps, t)
        # Fill-ratio (train signal): computed from the SAME natural-text-width / canvas-width formula
        # used at inference (_fill_ratio_from_texts), so train and inference feed the model an
        # identically-distributed scalar (no train/inference skew). canvas width = the target image's
        # pixel width. Computed before _conditioning so it is folded into pooled BEFORE cond-dropout —
        # the whole pooled (incl. the fill-ratio add) is then dropped together on CFG-dropped samples.
        w_t = x0.shape[-1] // self.backbone.patch  # image patch-column count (line-glyph token count)
        fill_ratio = self._fill_ratio_from_texts(texts, images.shape[-1])
        context, pooled, n_content = self._conditioning(
            texts, style_pixel_values, x0.device, fill_ratio, w_tokens=w_t, text_widths=text_widths, ref=ref
        )
        nc = n_content if self.glyph_line else None  # content-token RoPE only in line-glyph mode
        glyph_latent = (
            self.glyph_latent(texts, x0.device, (x0.shape[-2], x0.shape[-1])) if self.glyph_latent else None
        )
        ref_cond = torch.cat([ref.latent.to(x0.dtype), torch.zeros_like(x0)], dim=-1) if in_grid else None
        ref_latent = ref.latent.to(x0.dtype) if ref is not None and self.ref_tokens else None
        if cond_dropout > 0.0 and self.training:  # CFG: learn an unconditional mode on dropped samples
            keep = (torch.rand(b, device=x0.device) >= cond_dropout).to(context.dtype)
            context = context * keep.view(b, 1, 1)
            pooled = pooled * keep.view(b, 1)
            if glyph_latent is not None:  # drop the concat content together with the rest (true uncond)
                glyph_latent = glyph_latent * keep.to(glyph_latent.dtype).view(b, 1, 1, 1)
            if ref_cond is not None:
                ref_cond = ref_cond * keep.to(ref_cond.dtype).view(b, 1, 1, 1)
            if ref_latent is not None:
                ref_latent = ref_latent * keep.to(ref_latent.dtype).view(b, 1, 1, 1)
        if (
            text_dropout > 0.0 and self.training
        ):  # TEXT-only CFG (DiffInk drop_text): drop CONTENT, KEEP style
            keep_t = (torch.rand(b, device=x0.device) >= text_dropout).to(context.dtype)
            content_part = context[:, :n_content] * keep_t.view(b, 1, 1)  # zero content tokens only
            context = torch.cat([content_part, context[:, n_content:]], dim=1)  # style tokens untouched
            if glyph_latent is not None:  # the concat glyph IS content -> drop it too
                glyph_latent = glyph_latent * keep_t.to(glyph_latent.dtype).view(b, 1, 1, 1)
        if ref is not None and ref_dropout > 0.0 and self.training:
            # STYLE-only drop (DiffInk drop_cond): the reference pixels AND its transcription go, the target
            # text stays -> the "text, no style" mode that style guidance contrasts against at sampling.
            keep_r = (torch.rand(b, device=x0.device) >= ref_dropout).to(context.dtype)
            n_ref = ref.latent.shape[-1] // self.backbone.patch  # the reference's glyph-token columns
            context = torch.cat([context[:, :n_ref] * keep_r.view(b, 1, 1), context[:, n_ref:]], dim=1)
            if ref_cond is not None:
                ref_cond = ref_cond * keep_r.to(ref_cond.dtype).view(b, 1, 1, 1)
            if ref_latent is not None:
                ref_latent = ref_latent * keep_r.to(ref_latent.dtype).view(b, 1, 1, 1)
        # NOTE: combine --text-dropout with --cond-dropout for the DiffInk-style mixture — each sample
        # randomly ends up {nothing, text-only, both} dropped (style is NEVER dropped alone, which is
        # what we want: the failure is "text ignored," not "style ignored"). Canvas mode adds --ref-dropout.
        # Channel-concat the glyph latent onto the noisy latent (content present at every cell).
        model_in = torch.cat([x_t, *(c for c in (glyph_latent, ref_cond) if c is not None)], dim=1)

        # ink-focal weighting: up-weight the loss on sparse ink cells so the ~90% white-paper
        # background doesn't dominate the velocity MSE and starve letter learning.
        ink_weight = self._ink_weight(images, x0.shape[-2:]) if self.cfg.aux.diacritic_focal_flow else None

        # REPA early-stop: the train loop passes repa_weight=0 after cfg.aux.repa_stop_frac of training
        # (HASTE: REPA helps early but its late gradient fights the denoiser). 0 -> skip the REPA path.
        repa_w = self.cfg.aux.repa_weight if repa_weight is None else repa_weight
        if self.repa_proj is not None and self.training and repa_w > 0.0:
            # REPA (training only): align the DiT's per-token hidden states to DINO's per-patch features
            # of the target image. Token-wise (NOT pooled): we resample DINO's patch grid onto the DiT
            # latent grid so token i aligns with the same spatial location — the spatial correspondence
            # is the whole point, and mean-pooling both sides (the old code) threw it away.
            v_pred, hidden = self._backbone_with_hidden(
                model_in, t * 1000.0, context, pooled, n_content=nc, ref_latent=ref_latent
            )  # [B,N,dim] — target image tokens (flux2 reference tokens already removed)
            if not hidden.requires_grad:
                raise RuntimeError(
                    "REPA: the captured DiT hidden state is detached from the autograd graph, so the "
                    "alignment loss would train only the projection head, not the backbone. This is the "
                    "REENTRANT gradient-checkpointing failure mode; Backbone.enable_optimizations uses "
                    "non-reentrant checkpointing — check your diffusers/torch version (or run --no-repa)."
                )
            h_t, w_t = x0.shape[-2] // self.backbone.patch, x0.shape[-1] // self.backbone.patch
            if ref_cols:  # keep only the target columns of the row-major canvas tokens
                cw_t = x_t.shape[-1] // self.backbone.patch
                hidden = hidden.reshape(b, h_t, cw_t, -1)[:, :, cw_t - w_t :, :].reshape(b, h_t * w_t, -1)
            proj = self.repa_proj(hidden)  # [B, N, dino_dim]
            target = self._dino_target(images, (h_t, w_t))  # [B, N, dino_dim]
            flow = flow_loss(v_pred[..., ref_cols:], target_v[..., ref_cols:], weight=ink_weight)
            repa: torch.Tensor | None = repa_w * repa_loss(proj, target)
            loss = flow + repa
        else:
            v_pred = self._run_backbone(
                model_in, t * 1000.0, context, pooled, nc, ref_latent
            )  # SD3 timestep scale ~[0,1000]
            flow = flow_loss(v_pred[..., ref_cols:], target_v[..., ref_cols:], weight=ink_weight)
            repa = None
            loss = flow

        if return_losses:  # detached components for logging (flow is the val-comparable term)
            out = {"loss": loss, "flow": flow.detach()}
            if repa is not None:
                out["repa"] = repa.detach()
            return out
        return loss

    def _ink_weight(self, images: torch.Tensor, latent_hw: torch.Size) -> torch.Tensor:
        """Per-latent-cell loss weight up-weighting sparse ink (dark) regions vs the bright paper.

        Returns ``[B, 1, h, w]`` = ``1`` on background, up to ``diacritic_class_weight`` on full-ink cells.
        """
        gray = images.mean(1, keepdim=True)  # [B,1,H,W], ink = dark
        ink = (gray < self.cfg.aux.ink_threshold).to(images.dtype)
        ink_latent = F.interpolate(ink, size=tuple(latent_hw), mode="area")  # [B,1,h,w] ink fraction
        return 1.0 + (self.cfg.aux.diacritic_class_weight - 1.0) * ink_latent

    def _run_backbone(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        pooled: torch.Tensor,
        n_content: int | None,
        ref_latent: torch.Tensor | None,
    ) -> torch.Tensor:
        """One backbone pass; a flux2 reference goes in as clean tokens (its K/V cache is discarded)."""
        if ref_latent is None:
            return self.backbone(latent, timestep, context, pooled, n_content=n_content)
        if not isinstance(self.backbone, Flux2Backbone):
            raise TypeError("reference tokens need the flux2 backbone")
        return self.backbone.forward_with_ref(
            latent, timestep, context, pooled, ref_latent, n_content=n_content
        )[0]

    def _backbone_with_hidden(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        pooled: torch.Tensor,
        n_content: int | None = None,
        ref_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the backbone, tapping the REPA-layer block's image hidden states via a forward hook."""
        captured: dict[str, torch.Tensor] = {}
        p = self.backbone.patch
        n_ref = (ref_latent.shape[-2] // p) * (ref_latent.shape[-1] // p) if ref_latent is not None else 0
        n_text = context.shape[1] + (
            1 if isinstance(self.backbone, Flux2Backbone) else 0
        )  # + the pooled token
        block, extract = self.backbone.repa_target(self.cfg.aux.repa_layer, n_text, n_ref)

        def _hook(_module, _inputs, output):
            captured["h"] = extract(output)

        handle = block.register_forward_hook(_hook)
        try:
            v_pred = self._run_backbone(latent, timestep, context, pooled, n_content, ref_latent)
        finally:
            handle.remove()
        return v_pred, captured["h"]  # [B, C, h, w], [B, N, dim]

    @torch.no_grad()
    def _dino_target(self, images: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        """Per-token DINO features of the clean target image, resampled onto the DiT latent grid.

        Returns ``[B, h_t*w_t, dino_dim]`` (frozen REPA target), flattened ROW-MAJOR (h then w) to match
        the DiT image-token order (PatchEmbed's ``flatten(2).transpose(1, 2)``), so target token ``i``
        corresponds to the DiT hidden token at the same spatial patch.
        """
        h_t, w_t = grid
        x = (images.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mean = x.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        if self.style is None:
            raise RuntimeError("REPA needs the DINO encoder (self.style), which was not built")
        feats = self.style._features((x - mean) / std)  # [B, P_total, D] (prefix CLS/registers + patches)
        patch = getattr(self.style.enc.config, "patch_size", 16)
        side = 224 // patch
        feats = feats[:, -side * side :, :]  # patch tokens are the LAST P=side*side (drop CLS+registers)
        b, _, d = feats.shape
        fmap = feats.transpose(1, 2).reshape(b, d, side, side)  # [B, D, side, side] row-major patches
        fmap = F.interpolate(fmap, size=(h_t, w_t), mode="bilinear", align_corners=False)
        return fmap.flatten(2).transpose(1, 2)  # [B, h_t*w_t, D]

    # --- generation -------------------------------------------------------
    def _fill_ratio_from_texts(self, texts: list[str], canvas_px: int) -> torch.Tensor:
        """Per-text natural-text-width / canvas-width in ``[0, 1]`` — the inference fill-ratio signal.

        Uses the same glyph-renderer natural-width formula as auto-width (generate.py / app.py), so the
        model is told "this text only fills X% of this wide canvas" and won't hallucinate a tail.
        """
        ratios = [self.guidance_renderer.natural_width(t) / max(canvas_px, 1) for t in texts]
        dev = next(self.parameters()).device
        return torch.tensor(ratios, device=dev).clamp(0.0, 1.0)  # [B]

    @torch.no_grad()
    def _denoise_latents(
        self,
        texts: list[str],
        style_pixel_values: torch.Tensor | None,
        latent_hw: tuple[int, int],
        num_steps: int,
        cfg_scale: float = 0.0,
        *,
        style_cfg_scale: float = 0.0,
        text_widths: list[int] | torch.Tensor | None = None,
        ref: CanvasRef | None = None,
    ) -> Iterator[torch.Tensor]:
        """Flow sampling with optional guidance. Yields the latent x0 estimate ``x_t - sigma·v`` each step,
        then the final latent. ``cfg_scale>0`` = classic classifier-free guidance (text-following): a second
        backbone pass with the content tokens + concat glyph dropped (style/pooled kept), combined as
        ``v = v_uncond + cfg_scale·(v_cond − v_uncond)`` — the lever that uses the trained cond/text-dropout.

        Canvas mode samples the whole ``[ref | target]`` canvas with the clean reference in the conditioning
        channel and yields only the TARGET columns. ``style_cfg_scale>0`` adds style guidance (needs a model
        trained with ``ref_dropout``): ``v = v_null + cfg·(v_text − v_null) + style·(v_full − v_text)`` where
        ``v_text`` keeps the text but drops the reference and ``v_null`` drops both."""
        h, w = latent_hw
        b = len(texts)
        dev = next(self.parameters()).device
        w_t = w // self.backbone.patch  # image patch-column count (line-glyph token count)
        # Inference fill-ratio: natural text width / canvas width (canvas px = latent w × VAE downscale).
        fill_ratio = self._fill_ratio_from_texts(texts, w * self.cfg.vae.downscale_factor)
        context, pooled, n_content = self._conditioning(
            texts, style_pixel_values, dev, fill_ratio, w_tokens=w_t, text_widths=text_widths, ref=ref
        )
        nc = n_content if self.glyph_line else None  # content-token RoPE only in line-glyph mode
        # Glyph latent is constant across denoising steps — compute once, concat each step.
        glyph_latent = self.glyph_latent(texts, dev, (h, w)) if self.glyph_latent else None
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(num_steps, device=dev)
        scheduler.set_begin_index(0)  # known start: skips the step-index lookup's device->host sync
        # Sample in the MODEL dtype (bf16 inference works end-to-end); the scheduler upcasts to fp32
        # internally for precision and returns fp32, so restore the dtype at each model/VAE boundary.
        dtype = next(self.parameters()).dtype
        in_grid = ref is not None and not self.ref_tokens  # sd3 canvas: the reference shares the image grid
        ref_cols = ref.latent.shape[-1] if in_grid else 0
        x = torch.randn(b, self.C, h, ref_cols + w, device=dev, dtype=dtype)
        ref_cond = (
            torch.cat([ref.latent.to(dtype), torch.zeros(b, self.C, h, w, device=dev, dtype=dtype)], dim=-1)
            if in_grid
            else None
        )
        ref_tok = ref.latent.to(dtype) if ref is not None and self.ref_tokens else None
        # Guidance variants, each (context, glyph_latent, ref_cond, ref_tok); index 0 is the fully conditioned
        # pass. All are step-invariant and run as ONE k·B backbone forward per step: RoPE and SDPA are
        # per-sample, so concatenating along the batch is numerically identical and better fills the GPU.
        no_text = context.clone()
        no_text[:, :n_content] = 0.0
        no_ref = context.clone()
        if ref is not None:
            no_ref[:, : ref.latent.shape[-1] // self.backbone.patch] = 0.0
        zero_glyph = torch.zeros_like(glyph_latent) if glyph_latent is not None else None
        zero_ref = torch.zeros_like(ref_cond) if ref_cond is not None else None
        zero_tok = torch.zeros_like(ref_tok) if ref_tok is not None else None
        variants = [(context, glyph_latent, ref_cond, ref_tok)]
        if style_cfg_scale > 0.0 and ref is not None:  # text-only, null
            variants += [
                (no_ref, glyph_latent, zero_ref, zero_tok),
                (no_text, zero_glyph, zero_ref, zero_tok),
            ]
        elif cfg_scale > 0.0:
            variants.append((no_text, zero_glyph, ref_cond, ref_tok))  # classic: drop text, keep style
        k = len(variants)
        ctx_k = torch.cat([v[0] for v in variants], dim=0)
        pooled_k = pooled.repeat(k, 1)
        extras_k = [torch.cat([v[i] for v in variants], dim=0) for i in (1, 2) if variants[0][i] is not None]
        ref_tok_k = torch.cat([v[3] for v in variants], dim=0) if ref_tok is not None else None
        cache = None  # flux2: the reference K/V, computed on the first step and reused unchanged afterwards
        for i, t in enumerate(scheduler.timesteps):
            model_in = torch.cat([x.repeat(k, 1, 1, 1), *extras_k], dim=1)
            timestep = t.expand(k * b)
            if ref_tok_k is None:
                out = self.backbone(model_in, timestep, ctx_k, pooled_k, n_content=nc)
            elif not isinstance(self.backbone, Flux2Backbone):
                raise TypeError("reference tokens need the flux2 backbone")
            elif cache is None:
                out, cache = self.backbone.forward_with_ref(
                    model_in, timestep, ctx_k, pooled_k, ref_tok_k, n_content=nc
                )
            else:
                ref_patch_cols = ref_tok_k.shape[-1] // self.backbone.patch
                out = self.backbone.forward_cached(
                    model_in, timestep, ctx_k, pooled_k, cache, ref_patch_cols, n_content=nc
                )
            chunks = out.chunk(k, dim=0)
            if k == 3:
                v_full, v_text, v_null = chunks
                v = v_null + cfg_scale * (v_text - v_null) + style_cfg_scale * (v_full - v_text)
            elif k == 2:
                v = chunks[1] + cfg_scale * (chunks[0] - chunks[1])
            else:
                v = chunks[0]
            yield (x - scheduler.sigmas[i] * v)[..., ref_cols:].to(dtype)  # target x0 estimate at this step
            x = scheduler.step(v, t, x).prev_sample.to(dtype)
        yield x[..., ref_cols:]  # the final denoised target latent

    def _prepare_ref(
        self,
        ref_images: torch.Tensor | None,
        ref_texts: list[str] | None,
        ref_widths: list[int] | torch.Tensor | None,
        batch: int,
    ) -> CanvasRef | None:
        if not self.canvas:
            return None
        if ref_images is not None and ref_images.shape[0] == 1 and batch > 1:
            ref_images = ref_images.expand(batch, -1, -1, -1)
        if ref_texts is not None and len(ref_texts) == 1 and batch > 1:
            ref_texts = ref_texts * batch
        if ref_widths is not None and len(ref_widths) == 1 and batch > 1:
            ref_widths = [int(ref_widths[0])] * batch
        return self._canvas_ref(ref_images, None, ref_texts, ref_widths, batch)

    @torch.inference_mode()
    def generate(
        self,
        texts: list[str],
        style_pixel_values: torch.Tensor | None,
        latent_hw: tuple[int, int],
        num_steps: int = 24,
        cfg_scale: float = 0.0,
        *,
        style_cfg_scale: float = 0.0,
        text_widths: list[int] | torch.Tensor | None = None,
        ref_images: torch.Tensor | None = None,
        ref_texts: list[str] | None = None,
        ref_widths: list[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """latent_hw = (h, w) of the target latent grid (w derived from the glyph-line width).

        Canvas mode: ``ref_images`` ``[B or 1, 3, H, W_ref]`` in ``[-1, 1]`` (W_ref a multiple of 16) +
        ``ref_texts``; ``style_pixel_values`` is unused. Returns only the generated target line."""
        ref = self._prepare_ref(ref_images, ref_texts, ref_widths, len(texts))
        *_, latent = self._denoise_latents(
            texts,
            style_pixel_values,
            latent_hw,
            num_steps,
            cfg_scale,
            style_cfg_scale=style_cfg_scale,
            text_widths=text_widths,
            ref=ref,
        )
        return self.vae.decode(latent)  # [B, 3, H, W] in [-1, 1] — only the final latent is decoded

    @torch.inference_mode()
    def denoise_steps(
        self,
        texts: list[str],
        style_pixel_values: torch.Tensor | None,
        latent_hw: tuple[int, int],
        num_steps: int = 24,
        cfg_scale: float = 0.0,
        *,
        style_cfg_scale: float = 0.0,
        text_widths: list[int] | torch.Tensor | None = None,
        ref_images: torch.Tensor | None = None,
        ref_texts: list[str] | None = None,
        ref_widths: list[int] | torch.Tensor | None = None,
    ) -> Iterator[torch.Tensor]:
        """Streaming variant for the demo: yields the decoded image ``[B, 3, H, W]`` at each denoising step
        (the x0 estimate, blurry → sharp) and finally the true final decode. One VAE decode per yield —
        that's the cost of watching it denoise live."""
        ref = self._prepare_ref(ref_images, ref_texts, ref_widths, len(texts))
        for latent in self._denoise_latents(
            texts,
            style_pixel_values,
            latent_hw,
            num_steps,
            cfg_scale,
            style_cfg_scale=style_cfg_scale,
            text_widths=text_widths,
            ref=ref,
        ):
            yield self.vae.decode(latent)
