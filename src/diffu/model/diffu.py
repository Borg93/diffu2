"""Diffu — diffusers-first model assembly.

Diffu is a 2026 model for Swedish handwritten line-image generation: MMDiT backbone +
rectified flow + 16-ch VAE + glyph/DINOv3 conditioning + classic CFG.

  Backbone : diffusers SD3 MMDiT (model/backbone.py)             [not hand-rolled]
  VAE      : Qwen-Image 16-ch (model/vae.py, diffusers)          [validated in Stage 0]
  Scheduler: diffusers FlowMatchEulerDiscreteScheduler           [sampling]
  Content  : Unifont glyph encoder (model/content_encoder.py)    [custom — no diffusers equivalent]
  Style    : DINOv3 + Perceiver resampler (model/conditioning.py)
  Guidance : classic classifier-free guidance (CFG) — text-following at sampling

Conditioning fed to SD3:
  encoder_hidden_states = concat(content tokens, style tokens)   [B, L+K, context_dim]
  pooled_projections    = pooled style vector                    [B, context_dim]

The diffusers wiring (SD3 timestep scale + FlowMatch step) is verified by the `diffu-smoke` synthetic forward/backward check
(one training forward + one generate on a GPU).
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler

from ..config import Config
from ..flow import flow_loss, interpolate, repa_loss, sample_timesteps
from .backbone import Backbone
from .conditioning import GlyphLineRenderer, StyleEncoder
from .content_encoder import GlyphContentEncoder, GlyphLatentEncoder, GlyphLineContentEncoder
from .vae import VAEWrapper

# DINO expects ImageNet-normalized input; REPA aligns DiT hidden states to DINO features of the
# target image, so we re-normalize the [-1,1] target into the DINO convention before encoding it.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


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
        self.style = StyleEncoder(cfg.cond, dim=ctx)  # DINOv3 -> tokens + pooled
        self.style_in_context = cfg.cond.style_in_context  # False = pooled-only (no copyable style tokens)
        # glyph_concat: a spatial glyph latent channel-concatenated onto the noisy latent (fuse content
        # into the input). Backbone reads C+Cg channels, still predicts C velocity channels.
        self.glyph_latent = (
            GlyphLatentEncoder(cfg.cond, out_ch=cfg.cond.glyph_concat_channels)
            if cfg.cond.glyph_concat
            else None
        )
        extra = cfg.cond.glyph_concat_channels if cfg.cond.glyph_concat else 0
        self.backbone = Backbone(
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
        dino_dim = self.style.enc.config.hidden_size
        self.repa_proj = (
            nn.Sequential(
                nn.Linear(cfg.backbone.dim, cfg.backbone.dim),
                nn.SiLU(),
                nn.Linear(cfg.backbone.dim, cfg.backbone.dim),
                nn.SiLU(),
                nn.Linear(cfg.backbone.dim, dino_dim),
            )
            if cfg.aux.repa
            else None
        )

    def _conditioning(
        self,
        texts: list[str],
        style_pixel_values: torch.Tensor,
        device: torch.device,
        fill_ratio: torch.Tensor | None = None,
        w_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Build joint context + pooled AdaLN vector. Returns ``(context, pooled, n_content)``.

        ``context`` is ``concat(content, style)`` along the token axis (content FIRST), so the first
        ``n_content`` tokens are the content tokens — the split point CFG drops (content only). If a
        ``fill_ratio`` is given and the MLP is enabled, its embedding is added to ``pooled``. In
        ``glyph_line`` mode ``w_tokens`` (the image patch-column count) is required and the content path
        emits exactly ``w_tokens`` column-aligned tokens.
        """
        if self.glyph_line:
            if w_tokens is None:
                raise ValueError("glyph_line content path needs w_tokens (the image patch-column count)")
            content = self.glyph_content(texts, device, w_tokens)  # [B, w_tokens, ctx]
        else:
            content = self.glyph_content(texts, device)  # [B, L, ctx]
        style_tokens, style_pooled = self.style(style_pixel_values)  # [B, K, ctx], [B, ctx]
        if self.fill_ratio_mlp is not None and fill_ratio is not None:
            fr = fill_ratio.to(style_pooled.dtype).view(-1, 1)  # [B, 1]
            style_pooled = style_pooled + self.fill_ratio_mlp(fr)
        # Keep the copyable spatial style tokens OUT of joint attention when style_in_context=False, so
        # content must come from the text/glyph path instead of being copied off the reference image.
        # (With the tokens in context the model copies the ref's glyphs and ignores the text instruction.)
        context = torch.cat([content, style_tokens], dim=1) if self.style_in_context else content
        return context, style_pooled, content.shape[1]

    # --- training ---------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        texts: list[str],
        style_pixel_values: torch.Tensor,
        *,
        latents: torch.Tensor | None = None,
        return_losses: bool = False,
        cond_dropout: float = 0.0,
        text_dropout: float = 0.0,
        repa_weight: float | None = None,
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

        images ``[B,3,H,W]`` in ``[-1,1]``; texts ``list[str]``; style_pixel_values DINO-preprocessed.
        """
        x0 = self.vae.encode(images) if latents is None else latents  # [B, C, h, w] (normalized)
        b = x0.shape[0]
        t = sample_timesteps(b, x0.device, self.cfg.flow.logit_normal_mean, self.cfg.flow.logit_normal_std)
        eps = torch.randn_like(x0)
        x_t, target_v = interpolate(x0, eps, t)
        # Fill-ratio (train signal): computed from the SAME natural-text-width / canvas-width formula
        # used at inference (_fill_ratio_from_texts), so train and inference feed the model an
        # identically-distributed scalar (no train/inference skew). canvas width = the target image's
        # pixel width. Computed before _conditioning so it is folded into pooled BEFORE cond-dropout —
        # the whole pooled (incl. the fill-ratio add) is then dropped together on CFG-dropped samples.
        w_t = x0.shape[-1] // self.backbone.patch  # image patch-column count (line-glyph token count)
        fill_ratio = self._fill_ratio_from_texts(texts, images.shape[-1])
        context, pooled, n_content = self._conditioning(
            texts, style_pixel_values, x0.device, fill_ratio, w_tokens=w_t
        )
        nc = n_content if self.glyph_line else None  # content-token RoPE only in line-glyph mode
        glyph_latent = (
            self.glyph_latent(texts, x0.device, (x0.shape[-2], x0.shape[-1])) if self.glyph_latent else None
        )
        if cond_dropout > 0.0 and self.training:  # CFG: learn an unconditional mode on dropped samples
            keep = (torch.rand(b, device=x0.device) >= cond_dropout).to(context.dtype)
            context = context * keep.view(b, 1, 1)
            pooled = pooled * keep.view(b, 1)
            if glyph_latent is not None:  # drop the concat content together with the rest (true uncond)
                glyph_latent = glyph_latent * keep.to(glyph_latent.dtype).view(b, 1, 1, 1)
        if (
            text_dropout > 0.0 and self.training
        ):  # TEXT-only CFG (DiffInk drop_text): drop CONTENT, KEEP style
            keep_t = (torch.rand(b, device=x0.device) >= text_dropout).to(context.dtype)
            content_part = context[:, :n_content] * keep_t.view(b, 1, 1)  # zero content tokens only
            context = torch.cat([content_part, context[:, n_content:]], dim=1)  # style tokens untouched
            if glyph_latent is not None:  # the concat glyph IS content -> drop it too
                glyph_latent = glyph_latent * keep_t.to(glyph_latent.dtype).view(b, 1, 1, 1)
        # NOTE: combine --text-dropout with --cond-dropout for the DiffInk-style mixture — each sample
        # randomly ends up {nothing, text-only, both} dropped (style is NEVER dropped alone, which is
        # what we want: the failure is "text ignored," not "style ignored").
        # Channel-concat the glyph latent onto the noisy latent (content present at every cell).
        model_in = torch.cat([x_t, glyph_latent], dim=1) if glyph_latent is not None else x_t

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
                model_in, t * 1000.0, context, pooled, n_content=nc
            )  # [B,N,dim]
            if not hidden.requires_grad:
                raise RuntimeError(
                    "REPA: the captured DiT hidden state is detached from the autograd graph, so the "
                    "alignment loss would train only the projection head, not the backbone. This is the "
                    "REENTRANT gradient-checkpointing failure mode; Backbone.enable_optimizations uses "
                    "non-reentrant checkpointing — check your diffusers/torch version (or run --no-repa)."
                )
            h_t, w_t = x0.shape[-2] // self.backbone.patch, x0.shape[-1] // self.backbone.patch
            proj = self.repa_proj(hidden)  # [B, N, dino_dim]
            target = self._dino_target(images, (h_t, w_t))  # [B, N, dino_dim]
            flow = flow_loss(v_pred, target_v, weight=ink_weight)
            repa: torch.Tensor | None = repa_w * repa_loss(proj, target)
            loss = flow + repa
        else:
            v_pred = self.backbone(
                model_in, t * 1000.0, context, pooled, n_content=nc
            )  # SD3 timestep scale ~[0,1000]
            flow = flow_loss(v_pred, target_v, weight=ink_weight)
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

    def _backbone_with_hidden(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        pooled: torch.Tensor,
        n_content: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the backbone, tapping the REPA-layer block's image hidden states via a forward hook."""
        captured: dict[str, torch.Tensor] = {}

        def _hook(_module, _inputs, output):
            captured["h"] = output[1] if isinstance(output, tuple | list) and len(output) > 1 else output

        blocks = self.backbone.transformer.transformer_blocks
        layer = max(0, min(self.cfg.aux.repa_layer, len(blocks) - 1))
        handle = blocks[layer].register_forward_hook(_hook)
        try:
            v_pred = self.backbone(latent, timestep, context, pooled, n_content=n_content)
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
        style_pixel_values: torch.Tensor,
        latent_hw: tuple[int, int],
        num_steps: int,
        cfg_scale: float = 0.0,
    ) -> Iterator[torch.Tensor]:
        """Flow sampling with optional guidance. Yields the latent x0 estimate ``x_t - sigma·v`` each step,
        then the final latent. ``cfg_scale>0`` = classic classifier-free guidance (text-following): a second
        backbone pass with the content tokens + concat glyph dropped (style/pooled kept), combined as
        ``v = v_uncond + cfg_scale·(v_cond − v_uncond)`` — the lever that uses the trained cond/text-dropout."""
        h, w = latent_hw
        b = len(texts)
        dev = next(self.parameters()).device
        w_t = w // self.backbone.patch  # image patch-column count (line-glyph token count)
        # Inference fill-ratio: natural text width / canvas width (canvas px = latent w × VAE downscale).
        fill_ratio = self._fill_ratio_from_texts(texts, w * self.cfg.vae.downscale_factor)
        context, pooled, n_content = self._conditioning(
            texts, style_pixel_values, dev, fill_ratio, w_tokens=w_t
        )
        nc = n_content if self.glyph_line else None  # content-token RoPE only in line-glyph mode
        # Glyph latent is constant across denoising steps — compute once, concat each step.
        glyph_latent = self.glyph_latent(texts, dev, (h, w)) if self.glyph_latent else None
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(num_steps, device=dev)
        # Sample in the MODEL dtype (bf16 inference works end-to-end); the scheduler upcasts to fp32
        # internally for precision and returns fp32, so restore the dtype at each model/VAE boundary.
        dtype = next(self.parameters()).dtype
        x = torch.randn(b, self.C, h, w, device=dev, dtype=dtype)
        # classic CFG (text-following): drop content tokens + concat glyph, keep style. context/null_ctx/
        # pooled are step-invariant, so build the (cond ++ uncond) context batch ONCE outside the loop.
        cfg_ctx: tuple[torch.Tensor, torch.Tensor] | None = None
        if cfg_scale > 0.0:
            null_ctx = context.clone()
            null_ctx[:, :n_content] = 0.0
            cfg_ctx = (torch.cat([context, null_ctx], dim=0), torch.cat([pooled, pooled], dim=0))
        for i, t in enumerate(scheduler.timesteps):
            model_in = torch.cat([x, glyph_latent], dim=1) if glyph_latent is not None else x
            if cfg_ctx is not None:
                # ONE 2B forward (cond ++ uncond) instead of two sequential backbone passes: RoPE and SDPA are
                # per-sample, so concatenating along the batch is numerically identical and better fills the GPU.
                ctx2, pooled2 = cfg_ctx
                null_in = (
                    torch.cat([x, torch.zeros_like(glyph_latent)], dim=1) if glyph_latent is not None else x
                )
                cat_in = torch.cat([model_in, null_in], dim=0)
                out = self.backbone(cat_in, t.expand(2 * b), ctx2, pooled2, n_content=nc)
                v_cond, v_uncond = out.chunk(2, dim=0)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = self.backbone(model_in, t.expand(b), context, pooled, n_content=nc)
            yield (x - scheduler.sigmas[i] * v).to(dtype)  # x0 estimate (clean-image guess) at this step
            x = scheduler.step(v, t, x).prev_sample.to(dtype)
        yield x  # the final denoised latent

    @torch.inference_mode()
    def generate(
        self,
        texts: list[str],
        style_pixel_values: torch.Tensor,
        latent_hw: tuple[int, int],
        num_steps: int = 24,
        cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        """latent_hw = (h, w) of the target latent grid (w derived from the glyph-line width)."""
        *_, latent = self._denoise_latents(texts, style_pixel_values, latent_hw, num_steps, cfg_scale)
        return self.vae.decode(latent)  # [B, 3, H, W] in [-1, 1] — only the final latent is decoded

    @torch.inference_mode()
    def denoise_steps(
        self,
        texts: list[str],
        style_pixel_values: torch.Tensor,
        latent_hw: tuple[int, int],
        num_steps: int = 24,
        cfg_scale: float = 0.0,
    ) -> Iterator[torch.Tensor]:
        """Streaming variant for the demo: yields the decoded image ``[B, 3, H, W]`` at each denoising step
        (the x0 estimate, blurry → sharp) and finally the true final decode. One VAE decode per yield —
        that's the cost of watching it denoise live."""
        for latent in self._denoise_latents(texts, style_pixel_values, latent_hw, num_steps, cfg_scale):
            yield self.vae.decode(latent)
