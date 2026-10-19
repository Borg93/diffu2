"""Diffu — diffusers-first model assembly.

Diffu is a 2026 model for Swedish handwritten line-image generation: MMDiT backbone + rectified flow +
16-ch VAE + Unifont glyph-line content + a writer reference + classifier-free guidance.

  Backbone : diffusers SD3 MMDiT with 2D RoPE (model/backbone.py)    [not hand-rolled]
  VAE      : Qwen-Image 16-ch (model/vae.py, diffusers)              [validated in Stage 0]
  Scheduler: diffusers FlowMatchEulerDiscreteScheduler               [sampling]
  Content  : Unifont glyph-line encoder (model/content_encoder.py): one token per image column
  Style    : cfg.cond.style_mode
               "dino"   — frozen DINOv3 of a reference crop -> resampler -> attention-pooled AdaLN vector
               "canvas" — the reference line's clean latent LEFT of the target on one canvas, as extra
                          input channels; the model continues the line in the same hand
  Guidance : classifier-free guidance — text scale (+ a separate style scale in canvas mode)

Conditioning fed to SD3:
  dino   : encoder_hidden_states = glyph tokens of the target text            [B, w_t, context_dim]
           pooled_projections    = pooled style vector (+ fill-ratio)         [B, context_dim]
  canvas : hidden_states         = concat(noisy canvas, clean-ref cond)       [B, 2C, h, w_ref+w]
           encoder_hidden_states = glyph tokens of (ref text | target text), column-aligned to the canvas
           pooled_projections    = fill-ratio embedding (no style vector)
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
from ..data.dataset import IMAGENET_MEAN, IMAGENET_STD
from ..flow import flow_loss, interpolate, repa_loss, sample_timesteps
from .backbone import Backbone
from .conditioning import GlyphLineRenderer, StyleEncoder
from .content_encoder import GlyphLineContentEncoder
from .vae import VAEWrapper

_SD3_TIMESTEP_SCALE = 1000.0  # SD3 takes timesteps in [0, 1000]; the flow time t is in [0, 1]


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
    """Full model: VAE + Unifont glyph-line content + writer reference (DINOv3 style or canvas) + SD3 backbone."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        token_px = cfg.vae.downscale_factor * cfg.backbone.patch  # one DiT token = 16 px
        if cfg.data.line_height % token_px:
            raise ValueError(f"line_height must be a multiple of {token_px} px, got {cfg.data.line_height}")
        self.cfg = cfg
        self.C = cfg.vae.latent_channels
        ctx = cfg.backbone.context_dim
        self.vae = VAEWrapper(cfg.vae)
        # Content: the whole line as one glyph image -> w_t column-aligned tokens, with shared-column RoPE so
        # image column j attends to the glyphs of column j (Qwen-Image MSRoPE).
        self.glyph_content = GlyphLineContentEncoder(cfg.cond, out_dim=ctx)
        self.canvas = cfg.cond.style_mode == "canvas"
        self.glyph_pad = cfg.cond.glyph_pad
        # DINOv3: the style encoder in "dino" mode; in "canvas" mode only REPA's frozen target features.
        self.style: StyleEncoder | None = None
        if not self.canvas:
            self.style = StyleEncoder(cfg.cond, dim=ctx)
        elif cfg.aux.repa:
            self.style = StyleEncoder(cfg.cond, dim=ctx, features_only=True)
        # Canvas: + the clean-reference conditioning channels (zeros over the target columns); the backbone
        # still predicts only the C velocity channels.
        in_channels = 2 * self.C if self.canvas else self.C
        self.backbone = Backbone(cfg.backbone, in_channels=in_channels, out_channels=self.C)
        self.guidance_renderer = GlyphLineRenderer(
            cfg.cond, height=cfg.data.line_height
        )  # natural text width

        # Fill-ratio conditioning: a scalar (natural text width / canvas width) in [0,1] is embedded by a small
        # MLP and ADDED to the pooled AdaLN vector, so the model is told how much of a (possibly wide) canvas
        # the text fills — and learns to leave the rest blank instead of hallucinating a tail. Train and
        # inference use the same glyph-renderer formula (_fill_ratio_from_texts). Trained params are picked
        # up in trainable_parameters.
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

    def _token_extents(self, widths_px: list[int] | None) -> list[int] | None:
        """Pixel ink widths -> patch-column extents for ``glyph_pad`` (``None`` = stretch over the canvas)."""
        if not self.glyph_pad or widths_px is None:
            return None
        unit = self.cfg.vae.downscale_factor * self.backbone.patch
        return [math.ceil(int(w) / unit) for w in widths_px]

    def _content(
        self,
        texts: list[str],
        device: torch.device,
        w_tokens: int,
        widths_px: list[int] | None,
    ) -> torch.Tensor:
        return self.glyph_content(
            texts, device, w_tokens, self._token_extents(widths_px)
        )  # [B, w_tokens, ctx]

    def _conditioning(
        self,
        texts: list[str],
        style_pixel_values: torch.Tensor | None,
        device: torch.device,
        w_tokens: int,
        fill_ratio: torch.Tensor | None = None,
        text_widths: list[int] | None = None,
        ref: CanvasRef | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The context tokens + pooled AdaLN vector.

        ``context`` = the target text's ``w_tokens`` column-aligned glyph tokens, preceded in canvas mode by the
        reference text's (one per canvas column; zeroed when the reference text is unknown). ``pooled`` = the
        attention-pooled DINO style vector (dino mode) or zeros (canvas), plus the fill-ratio embedding. Style
        reaches the model ONLY through ``pooled``: spatial style tokens in joint attention carry the
        reference's glyph shapes, and the model copies them instead of writing the requested text.
        """
        context = self._content(texts, device, w_tokens, text_widths)
        if self.canvas:
            if ref is None:
                raise ValueError("style_mode='canvas' needs a reference line (ref_images)")
            ref_tokens = ref.latent.shape[-1] // self.backbone.patch
            ref_content = self._content(ref.texts, device, ref_tokens, ref.widths_px)
            has_text = torch.tensor([bool(t) for t in ref.texts], device=device, dtype=ref_content.dtype)
            context = torch.cat([ref_content * has_text.view(-1, 1, 1), context], dim=1)
            pooled = context.new_zeros(context.shape[0], self.cfg.backbone.context_dim)
        else:
            if self.style is None or style_pixel_values is None:
                raise ValueError("style_mode='dino' needs style_pixel_values")
            pooled = self.style(style_pixel_values)  # [B, ctx]
        if self.fill_ratio_mlp is not None and fill_ratio is not None:
            pooled = pooled + self.fill_ratio_mlp(fill_ratio.to(pooled.dtype).view(-1, 1))
        return context, pooled

    def _canvas_ref(
        self,
        ref_images: torch.Tensor | None,
        ref_texts: list[str] | None,
        ref_widths: list[int] | None,
        batch: int,
        *,
        sample: bool = True,
    ) -> CanvasRef | None:
        """Bundle the canvas-mode reference: its VAE latent (a posterior sample like the training target, or
        the mode with ``sample=False``) and what it says. One reference for a batch is encoded once."""
        if not self.canvas:
            return None
        if ref_images is None:
            raise ValueError("style_mode='canvas' needs ref_images")
        latent = self.vae.encode(ref_images, sample=sample)
        if latent.shape[0] == 1 and batch > 1:
            latent = latent.expand(batch, -1, -1, -1)
        texts = list(ref_texts) if ref_texts else [""] * batch
        widths = None if ref_widths is None else [int(w) for w in ref_widths]
        if len(texts) == 1 and batch > 1:
            texts, widths = texts * batch, None if widths is None else widths * batch
        return CanvasRef(latent=latent, texts=texts, widths_px=widths)

    # --- training ---------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        texts: list[str],
        style_pixel_values: torch.Tensor | None = None,
        *,
        return_losses: bool = False,
        cond_dropout: float = 0.0,
        text_dropout: float = 0.0,
        ref_dropout: float = 0.0,
        repa_weight: float | None = None,
        text_widths: list[int] | None = None,
        ref_images: torch.Tensor | None = None,
        ref_texts: list[str] | None = None,
        ref_widths: list[int] | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Stage-1 flow loss (+ optional REPA).

        ``return_losses`` returns ``{"loss", "flow"[, "repa"]}`` (detached components) so the training
        loop can log the **flow** term separately — that, not the REPA-inclusive total, is what is
        comparable to the eval-time val loss (REPA is training-only), keeping the overfitting check
        apples-to-apples.

        images ``[B,3,H,W]`` in ``[-1,1]``; texts ``list[str]``; style_pixel_values DINO-preprocessed
        (``style_mode="dino"`` only). ``text_widths`` are the target lines' real ink widths in px (for
        ``glyph_pad``). Canvas mode reads ``ref_images`` ``[B,3,H,W_ref]`` + ``ref_texts`` / ``ref_widths``;
        ``ref_dropout`` drops the reference alone (the style-CFG mode).
        """
        x0 = self.vae.encode(images)  # [B, C, h, w] (normalized)
        b = x0.shape[0]
        ref = self._canvas_ref(ref_images, ref_texts, ref_widths, b)
        # Canvas mode: noise the whole [ref | target] canvas; the clean reference rides in a conditioning
        # channel and the loss is taken on the target columns only (F5-TTS / DiffInk infilling).
        ref_cols = ref.latent.shape[-1] if ref is not None else 0
        canvas_x0 = torch.cat([ref.latent.to(x0.dtype), x0], dim=-1) if ref is not None else x0
        t = sample_timesteps(b, x0.device, self.cfg.flow.logit_normal_mean, self.cfg.flow.logit_normal_std)
        eps = torch.randn_like(canvas_x0)
        x_t, target_v = interpolate(canvas_x0, eps, t)
        # Fill-ratio (train signal): the SAME natural-text-width / canvas-width formula as inference, with the
        # canvas = the target image's pixel width. Folded into pooled BEFORE cond-dropout, so the whole pooled
        # vector (incl. the fill-ratio add) is dropped together on CFG-dropped samples.
        w_t = x0.shape[-1] // self.backbone.patch  # image patch-column count (glyph token count)
        fill_ratio = self._fill_ratio_from_texts(texts, images.shape[-1])
        context, pooled = self._conditioning(
            texts, style_pixel_values, x0.device, w_t, fill_ratio, text_widths=text_widths, ref=ref
        )
        ref_cond = (
            torch.cat([ref.latent.to(x0.dtype), torch.zeros_like(x0)], dim=-1) if ref is not None else None
        )
        # Dropout modes for guidance (each sample independently): --cond-dropout drops everything (the null),
        # --text-dropout the glyph tokens (style kept), --ref-dropout the canvas reference (text kept).
        ref_kept = torch.ones(
            b, dtype=torch.bool, device=x0.device
        )  # canvas: whose reference survives dropout
        if cond_dropout > 0.0 and self.training:
            keep = (torch.rand(b, device=x0.device) >= cond_dropout).to(context.dtype)
            context = context * keep.view(b, 1, 1)
            pooled = pooled * keep.view(b, 1)
            if ref_cond is not None:
                ref_cond = ref_cond * keep.to(ref_cond.dtype).view(b, 1, 1, 1)
                ref_kept &= keep.bool()
        if text_dropout > 0.0 and self.training:  # TEXT-only CFG (DiffInk drop_text)
            keep_t = (torch.rand(b, device=x0.device) >= text_dropout).to(context.dtype)
            context = context * keep_t.view(b, 1, 1)
        if ref_cond is not None and ref_dropout > 0.0 and self.training:
            # STYLE-only drop (DiffInk drop_cond): the reference pixels AND its transcription go, the target
            # text stays -> the "text, no style" mode that style guidance contrasts against at sampling.
            keep_r = (torch.rand(b, device=x0.device) >= ref_dropout).to(context.dtype)
            n_ref = ref_cols // self.backbone.patch
            context = torch.cat([context[:, :n_ref] * keep_r.view(b, 1, 1), context[:, n_ref:]], dim=1)
            ref_cond = ref_cond * keep_r.to(ref_cond.dtype).view(b, 1, 1, 1)
            ref_kept &= keep_r.bool()
        if ref_cond is not None and self.training and (cond_dropout > 0.0 or ref_dropout > 0.0):
            # x_t also carries the reference: its canvas columns hold it noised to level t. A dropped reference
            # leaves those columns pure noise, else the "no reference" mode still sees the hand (half-noised on a
            # typical step) and style guidance contrasts against a mode that has the style. Masked, no sync.
            kept_cols = torch.where(ref_kept.view(b, 1, 1, 1), x_t[..., :ref_cols], eps[..., :ref_cols])
            x_t = torch.cat([kept_cols, x_t[..., ref_cols:]], dim=-1)
        model_in = torch.cat([x_t, ref_cond], dim=1) if ref_cond is not None else x_t

        # ink-focal weighting: up-weight the loss on sparse ink cells so the ~90% white-paper
        # background doesn't dominate the velocity MSE and starve letter learning.
        ink_weight = self._ink_weight(images, x0.shape[-2:]) if self.cfg.aux.ink_focal else None
        timestep = t * _SD3_TIMESTEP_SCALE

        # REPA early-stop: the train loop passes repa_weight=0 after cfg.aux.repa_stop_frac of training
        # (HASTE: REPA helps early but its late gradient fights the denoiser). 0 -> skip the REPA path.
        repa_w = self.cfg.aux.repa_weight if repa_weight is None else repa_weight
        if self.repa_proj is not None and self.training and repa_w > 0.0:
            # REPA (training only): align the DiT's per-token hidden states to DINO's per-patch features of
            # the target image, token i to the DINO patch at the same spatial location.
            v_pred, hidden = self._backbone_with_hidden(model_in, timestep, context, pooled)
            if not hidden.requires_grad:
                raise RuntimeError(
                    "REPA: the captured DiT hidden state is detached from the autograd graph (reentrant "
                    "gradient checkpointing?), so the alignment loss would train only the projection head."
                )
            h_t, w_t = x0.shape[-2] // self.backbone.patch, x0.shape[-1] // self.backbone.patch
            if ref_cols:  # keep only the target columns of the row-major canvas tokens
                cw_t = x_t.shape[-1] // self.backbone.patch
                hidden = hidden.reshape(b, h_t, cw_t, -1)[:, :, cw_t - w_t :, :].reshape(b, h_t * w_t, -1)
            proj = self.repa_proj(hidden)  # [B, N, dino_dim]
            target = self._dino_target(images, (h_t, w_t))  # [B, N, dino_dim]
            repa: torch.Tensor | None = repa_w * repa_loss(proj, target)
        else:
            v_pred = self.backbone(model_in, timestep, context, pooled)
            repa = None
        flow = flow_loss(v_pred[..., ref_cols:], target_v[..., ref_cols:], weight=ink_weight)
        loss = flow if repa is None else flow + repa

        if return_losses:  # detached components for logging (flow is the val-comparable term)
            out = {"loss": loss, "flow": flow.detach()}
            if repa is not None:
                out["repa"] = repa.detach()
            return out
        return loss

    def _ink_weight(self, images: torch.Tensor, latent_hw: torch.Size) -> torch.Tensor:
        """Per-latent-cell loss weight up-weighting sparse ink (dark) regions vs the bright paper.

        Returns ``[B, 1, h, w]`` = ``1`` on background, up to ``cfg.aux.ink_weight`` on full-ink cells.
        """
        gray = images.mean(1, keepdim=True)  # [B,1,H,W], ink = dark
        ink = (gray < self.cfg.aux.ink_threshold).to(images.dtype)
        ink_latent = F.interpolate(ink, size=tuple(latent_hw), mode="area")  # [B,1,h,w] ink fraction
        return 1.0 + (self.cfg.aux.ink_weight - 1.0) * ink_latent

    def _backbone_with_hidden(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        pooled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the backbone, tapping the REPA-layer block's image hidden states via a forward hook."""
        captured: dict[str, torch.Tensor] = {}

        def _hook(_module, _inputs, output):
            captured["h"] = output[1]  # JointTransformerBlock returns (encoder_hidden_states, hidden_states)

        blocks = self.backbone.transformer.transformer_blocks
        layer = max(0, min(self.cfg.aux.repa_layer, len(blocks) - 1))
        handle = blocks[layer].register_forward_hook(_hook)
        try:
            v_pred = self.backbone(latent, timestep, context, pooled)
        finally:
            handle.remove()
        return v_pred, captured["h"]  # [B, C, h, w], [B, N, dim]

    @torch.no_grad()
    def _dino_target(self, images: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        """Per-token DINOv3 features of the clean target line on the DiT latent grid ``[B, h_t*w_t, dino_dim]``.

        The line goes in at its own size, not squashed to a square: DINOv3's 16-px patches (RoPE, any aspect) on
        a line whose height and width are multiples of 16 form exactly the DiT's token grid (VAE f8 × patch 2
        = 16 px), so target token ``i`` describes the same patch as hidden token ``i``. Flattened row-major to
        match PatchEmbed's token order.
        """
        if self.style is None:
            raise RuntimeError("REPA needs the DINO encoder (self.style), which was not built")
        h_t, w_t = grid
        x = (images.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
        mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        feats = self.style._features((x - mean) / std)  # [B, prefix + P, D] (CLS + registers first)
        patch = self.style.enc.config.patch_size
        gh, gw = x.shape[-2] // patch, x.shape[-1] // patch
        fmap = feats[:, -gh * gw :].transpose(1, 2).reshape(x.shape[0], -1, gh, gw)  # [B, D, gh, gw]
        if (gh, gw) != (h_t, w_t):  # another encoder patch size or VAE geometry: resample onto the DiT grid
            fmap = F.interpolate(fmap, size=(h_t, w_t), mode="bilinear", align_corners=False)
        return fmap.flatten(2).transpose(1, 2)  # [B, h_t*w_t, D]

    # --- generation -------------------------------------------------------
    def _fill_ratio_from_texts(self, texts: list[str], canvas_px: int) -> torch.Tensor:
        """Per-text natural-text-width / canvas-width in ``[0, 1]`` — the fill-ratio signal.

        The raw glyph-renderer width estimate over the canvas width, at train and inference alike, so the
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
        text_widths: list[int] | None = None,
        ref: CanvasRef | None = None,
    ) -> Iterator[torch.Tensor]:
        """Flow sampling with optional guidance. Yields the latent x0 estimate ``x_t - sigma·v`` each step,
        then the final latent. ``cfg_scale>0`` = classic classifier-free guidance (text-following): a second
        backbone pass with the glyph tokens dropped (style/pooled kept), combined as
        ``v = v_uncond + cfg_scale·(v_cond − v_uncond)`` — the lever that uses the trained cond/text-dropout.

        Canvas mode denoises the TARGET columns of a ``[ref | target]`` canvas, with the clean reference in the
        conditioning channel and its canvas columns set each step to the reference noised to the current level
        (as in training). ``style_cfg_scale>0`` adds style guidance (needs a model trained with ``ref_dropout``):
        ``v = v_null + cfg·(v_text − v_null) + style·(v_full − v_text)`` where ``v_text`` keeps the text but
        drops the reference and ``v_null`` drops both (and the pooled vector: the ``cond_dropout`` mode); with
        ``cfg_scale`` 0 (text guidance off) it is ``v_text + style·(v_full − v_text)``. A dropped reference
        leaves its canvas columns pure noise, matching how ``forward`` drops it."""
        if style_cfg_scale > 0.0 and not self.canvas:
            raise ValueError(
                "style_cfg_scale needs a style_mode='canvas' model: a dino model has no reference to drop"
            )
        h, w = latent_hw
        b = len(texts)
        dev = next(self.parameters()).device
        w_t = w // self.backbone.patch  # image patch-column count (glyph token count)
        # Inference fill-ratio: natural text width / canvas width (canvas px = latent w × VAE downscale).
        fill_ratio = self._fill_ratio_from_texts(texts, w * self.cfg.vae.downscale_factor)
        context, pooled = self._conditioning(
            texts, style_pixel_values, dev, w_t, fill_ratio, text_widths=text_widths, ref=ref
        )
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(num_steps, device=dev)
        scheduler.set_begin_index(0)  # known start: skips the step-index lookup's device->host sync
        # Sample in the MODEL dtype (bf16 inference works end-to-end); the scheduler steps in fp32 internally
        # and casts back to the velocity's dtype.
        dtype = next(self.parameters()).dtype
        ref_cols = ref.latent.shape[-1] if ref is not None else 0
        noise = torch.randn(b, self.C, h, ref_cols + w, device=dev, dtype=dtype)
        # The ODE state is the TARGET only. The reference columns are rebuilt every step as training feeds them:
        # the reference noised to the current level, or pure noise in the variants that drop the reference.
        # (The loss never trains the velocity on those columns, so integrating them would drift them off it.)
        eps_ref, x = noise[..., :ref_cols], noise[..., ref_cols:]
        ref_latent = ref.latent.to(dtype) if ref is not None else None
        ref_cond = (
            torch.cat([ref_latent, torch.zeros(b, self.C, h, w, device=dev, dtype=dtype)], dim=-1)
            if ref_latent is not None
            else None
        )
        # Guidance variants, each (context, pooled, ref_cond, reference kept); index 0 is the fully conditioned
        # pass. All are step-invariant and run as ONE k·B backbone forward per step: RoPE and SDPA are
        # per-sample, so concatenating along the batch is numerically identical and better fills the GPU.
        no_text = torch.zeros_like(context)
        no_ref = context.clone()
        no_ref[:, : ref_cols // self.backbone.patch] = 0.0
        zero_ref = torch.zeros_like(ref_cond) if ref_cond is not None else None
        style_guided = style_cfg_scale > 0.0 and ref is not None
        variants = [(context, pooled, ref_cond, True)]
        if style_guided:
            variants.append((no_ref, pooled, zero_ref, False))  # text, no reference
            if cfg_scale > 0.0:  # the null: the --cond-dropout mode, which drops pooled too
                variants.append((no_text, torch.zeros_like(pooled), zero_ref, False))
        elif cfg_scale > 0.0:
            variants.append((no_text, pooled, ref_cond, True))  # classic: drop text, keep style
        k = len(variants)
        ctx_k = torch.cat([v[0] for v in variants], dim=0)
        pooled_k = torch.cat([v[1] for v in variants], dim=0)
        ref_cond_k = torch.cat([v[2] for v in variants], dim=0) if ref_cond is not None else None
        for i, t in enumerate(scheduler.timesteps):
            sigma = scheduler.sigmas[i]
            canvas_x = x.repeat(k, 1, 1, 1)
            if ref_latent is not None and ref_cond_k is not None:
                noised_ref = ((1 - sigma) * ref_latent + sigma * eps_ref).to(dtype)
                ref_k = torch.cat([noised_ref if kept else eps_ref for *_, kept in variants], dim=0)
                canvas_x = torch.cat([torch.cat([ref_k, canvas_x], dim=-1), ref_cond_k], dim=1)
            out = self.backbone(canvas_x, t.expand(k * b), ctx_k, pooled_k)
            out = out[..., ref_cols:].chunk(k, dim=0)  # the target columns of each variant
            if style_guided:
                v_full, v_text, *v_null = out
                text_v = v_text if not v_null else v_null[0] + cfg_scale * (v_text - v_null[0])
                v = text_v + style_cfg_scale * (v_full - v_text)
            elif k == 2:
                v = out[1] + cfg_scale * (out[0] - out[1])
            else:
                v = out[0]
            yield (x - sigma * v).to(dtype)  # target x0 estimate at this step
            x = scheduler.step(v, t, x).prev_sample.to(dtype)
        yield x  # the final denoised target latent

    @torch.inference_mode()
    def generate(
        self,
        texts: list[str],
        style_pixel_values: torch.Tensor | None,
        latent_hw: tuple[int, int],
        num_steps: int | None = None,
        cfg_scale: float = 0.0,
        *,
        style_cfg_scale: float = 0.0,
        text_widths: list[int] | None = None,
        ref_images: torch.Tensor | None = None,
        ref_texts: list[str] | None = None,
        ref_widths: list[int] | None = None,
    ) -> torch.Tensor:
        """latent_hw = (h, w) of the target latent grid (w derived from the glyph-line width).

        ``num_steps`` defaults to ``cfg.flow.sample_steps``. Canvas mode: ``ref_images``
        ``[B or 1, 3, H, W_ref]`` in ``[-1, 1]`` (W_ref a multiple of 16) + ``ref_texts``;
        ``style_pixel_values`` is unused. Returns only the generated target line."""
        ref = self._canvas_ref(ref_images, ref_texts, ref_widths, len(texts), sample=False)
        *_, latent = self._denoise_latents(
            texts,
            style_pixel_values,
            latent_hw,
            num_steps or self.cfg.flow.sample_steps,
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
        num_steps: int | None = None,
        cfg_scale: float = 0.0,
        *,
        style_cfg_scale: float = 0.0,
        text_widths: list[int] | None = None,
        ref_images: torch.Tensor | None = None,
        ref_texts: list[str] | None = None,
        ref_widths: list[int] | None = None,
    ) -> Iterator[torch.Tensor]:
        """Streaming variant for the demo: yields the decoded image ``[B, 3, H, W]`` at each denoising step
        (the x0 estimate, blurry → sharp) and finally the true final decode. One VAE decode per yield —
        that's the cost of watching it denoise live."""
        ref = self._canvas_ref(ref_images, ref_texts, ref_widths, len(texts), sample=False)
        for latent in self._denoise_latents(
            texts,
            style_pixel_values,
            latent_hw,
            num_steps or self.cfg.flow.sample_steps,
            cfg_scale,
            style_cfg_scale=style_cfg_scale,
            text_widths=text_widths,
            ref=ref,
        ):
            yield self.vae.decode(latent)
