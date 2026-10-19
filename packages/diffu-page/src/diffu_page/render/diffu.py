"""Diffu-backed renderers — wrap the project's line generator, per-line and batched.

Heavy deps (torch, the model, the checkpoint) are imported and built lazily on first ``render``,
so importing :mod:`diffu_page` and running the layout/compositor needs only OpenCV + numpy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import LineJob
from .matte import ink_from_pil

if TYPE_CHECKING:
    import torch

    from diffu.config import Config
    from diffu.model import Diffu


def bucket_render_batches(
    widths: list[int], *, bucket_px: int, max_batch: int
) -> list[tuple[int, list[int]]]:
    """Plan the batched render — pure arithmetic, unit-testable without torch.

    Each natural ``width`` is quantised UP to a multiple of ``bucket_px``, job indices sharing a bucket
    width are grouped (so one padded canvas serves the whole group), and each group is split into chunks
    of at most ``max_batch``. Returns ``(bucket_width, job_indices)`` batches covering every job exactly
    once, in ascending bucket-width order.
    """
    if bucket_px < 1 or max_batch < 1:
        raise ValueError(f"bucket_px and max_batch must be >= 1, got {bucket_px} and {max_batch}")
    bucket = [((w + bucket_px - 1) // bucket_px) * bucket_px for w in widths]
    order = sorted(range(len(widths)), key=lambda i: bucket[i])  # group equal bucket widths together
    batches: list[tuple[int, list[int]]] = []
    i = 0
    while i < len(order):
        width = bucket[order[i]]
        group: list[int] = []
        while i < len(order) and bucket[order[i]] == width:
            group.append(order[i])
            i += 1
        for start in range(0, len(group), max_batch):
            batches.append((width, group[start : start + max_batch]))
    return batches


class DiffuRenderer:
    """Wrap Diffu's line generator: (text) → BGRA ink, one fixed writer style for the whole page.

    Heavy deps (torch, the model, the checkpoint) are imported and built lazily on first ``render``,
    so importing :mod:`diffu_page` and running the layout/compositor needs only OpenCV + numpy.
    """

    def __init__(
        self,
        ckpt: str,
        style_path: str,
        *,
        cfg: Config | None = None,
        num_steps: int = 24,
        cfg_scale: float = 3.0,
        device: str | None = None,
        compile_model: bool = False,
        lora: str | None = None,
        lora_scale: float = 1.0,
    ) -> None:
        self.ckpt = ckpt
        self.style_path = style_path
        self.cfg = cfg
        self.num_steps = num_steps
        self.cfg_scale = cfg_scale
        self.device = device
        self.compile_model = compile_model
        self.lora = lora
        self.lora_scale = lora_scale
        self._model: Diffu | None = None
        self._style: torch.Tensor | None = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch  # lazy

        from diffu.generate import build_model, load_config, load_style

        if self.cfg is None:
            # The checkpoint's own config.json (diffu-train saves it next to the weights). Without one,
            # config.py's defaults ARE the released model's arch (exp_sd35_128: glyph-line content, pooled-only
            # style, 128px); pass cfg for other archs.
            self.cfg = load_config(self.ckpt or None)
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # build_model: weights + the conditioner guard (fail fast: a mismatched arch generates gibberish), the
        # Stage-3 LoRA adapter (dir or Hub id) BEFORE compile (peft mutates modules), then the regional compile
        # (a page = many lines → compile cold-start amortizes well)
        self._model = build_model(
            self.cfg,
            self.ckpt,
            device=self.device,
            lora=self.lora,
            lora_scale=self.lora_scale,
            compile_backbone=self.compile_model,
        )
        self._style = load_style(self.style_path).unsqueeze(0).to(self.device)  # [1,3,S,S]

    def render(self, text: str, height: int) -> np.ndarray:
        import torch  # lazy

        from diffu.generate import generate_line

        self._ensure()
        if self._model is None or self._style is None:
            raise RuntimeError("DiffuRenderer not initialised")  # _ensure guarantees these
        with torch.no_grad():
            pil = generate_line(
                self._model,
                text or " ",
                self._style,
                num_steps=self.num_steps,
                cfg_scale=self.cfg_scale,
            )
        return ink_from_pil(pil, height)

    def measure_width(self, text: str, height: int) -> int:
        """Predicted width at ``height`` from the model's content-aware natural width (no denoise)."""
        from diffu.generate import auto_canvas_width

        self._ensure()
        if self._model is None or self.cfg is None:
            raise RuntimeError("DiffuRenderer not initialised")
        w = auto_canvas_width(self._model, text or " ")
        return int(w * height / self.cfg.data.line_height)


class BatchedDiffuRenderer:
    """Render every line of a whole job in one batched GPU pass — the page pipeline's main speedup.

    Per-line generation feeds the GPU one ~64px line at a time (it sits ~half-idle). Here the whole run's
    lines are width-bucketed and each bucket pushed through the model's batched ``generate`` in chunks of
    ``max_batch`` (optionally with fused/compiled blocks, which amortise over the few bucket widths), then
    ink-cropped. The model is built once; each line keeps its own writer style.
    """

    def __init__(
        self,
        ckpt: str,
        *,
        cfg: Config | None = None,
        num_steps: int = 24,
        cfg_scale: float = 3.0,
        device: str | None = None,
        max_batch: int = 48,
        bucket_px: int = 16,  # near-natural width: a coarse bucket makes the model spread/soften strokes
        compile_blocks: bool = True,
        dtype: str = "bfloat16",  # inference dtype ON CUDA only; CPU stays fp32 (see _ensure)
        lora: str | None = None,
        lora_scale: float = 1.0,
    ) -> None:
        self.ckpt = ckpt
        self.cfg = cfg
        self.num_steps = num_steps
        self.cfg_scale = cfg_scale
        self.device = device
        self.max_batch = max_batch
        self.bucket_px = bucket_px  # width quantisation: coarser -> fewer compiles, more padding waste
        self.compile_blocks = compile_blocks
        self.dtype = dtype
        self.lora = lora
        self.lora_scale = lora_scale
        self._model: Diffu | None = None
        self._styles: dict[str, torch.Tensor] = {}

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch  # lazy

        from diffu.generate import build_model, load_config

        if self.cfg is None:
            # The checkpoint's own config.json (diffu-train saves it next to the weights). Without one,
            # config.py's defaults ARE the released model's arch (exp_sd35_128: glyph-line content, pooled-only
            # style, 128px); pass cfg for other archs.
            self.cfg = load_config(self.ckpt or None)
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # bf16 end-to-end is the validated shipping config (docs/PERF_AUDIT.md §I) — CUDA only; a CPU
        # fallback (tests / no GPU) stays fp32 since bf16-on-CPU is unvalidated here and not the target path.
        torch_dtype = getattr(torch, self.dtype) if self.device.startswith("cuda") else torch.float32
        # build_model: TF32 matmuls, weights + the conditioner guard (fail fast: a mismatched arch generates
        # gibberish), the Stage-3 LoRA adapter (dir or Hub id) BEFORE compile (peft mutates modules), then the
        # regional compile (fuses per-block glue; amortised over the few bucket widths)
        self._model = build_model(
            self.cfg,
            self.ckpt,
            device=self.device,
            dtype=torch_dtype,
            lora=self.lora,
            lora_scale=self.lora_scale,
            compile_backbone=self.compile_blocks,
        )

    def _style(self, path: str) -> torch.Tensor:
        if path not in self._styles:
            from diffu.generate import load_style

            # Match the model's dtype (bf16 by default) — the frozen DINO style encoder errors on a
            # dtype mismatch between its (now bf16) weights and an fp32 input.
            model_dtype = next(self._model.parameters()).dtype if self._model is not None else None
            style = load_style(path).unsqueeze(0).to(self.device)  # [1, 3, S, S]
            self._styles[path] = style.to(dtype=model_dtype) if model_dtype is not None else style
        return self._styles[path]

    def measure_width(self, text: str, height: int) -> int:
        """Predicted rendered width of ``text`` at ``height`` from the model's content-aware natural width
        (no denoise) — lets the layout wrap by real width before the batched render."""
        from diffu.generate import auto_canvas_width

        self._ensure()
        if self._model is None or self.cfg is None:
            raise RuntimeError("renderer not initialised")  # _ensure guarantees these
        w = auto_canvas_width(self._model, text or " ")
        return int(w * height / self.cfg.data.line_height)

    def render_jobs(self, jobs: list[LineJob]) -> list[np.ndarray]:
        """Render each job to a BGRA ink raster, width-bucketed and batched. Output aligns with ``jobs``."""
        import torch  # lazy

        from diffu.generate import auto_canvas_width, ink_crop, to_pil

        self._ensure()
        if self._model is None or self.cfg is None:
            raise RuntimeError("renderer not initialised")  # _ensure guarantees these
        ds = self.cfg.vae.downscale_factor
        latent_h = self.cfg.data.line_height // ds

        widths = [auto_canvas_width(self._model, j.text or " ") for j in jobs]
        inks: list[np.ndarray | None] = [None] * len(jobs)
        for width, chunk in bucket_render_batches(widths, bucket_px=self.bucket_px, max_batch=self.max_batch):
            latent_w = width // ds
            texts = [jobs[k].text or " " for k in chunk]
            style = torch.cat([self._style(jobs[k].style_path) for k in chunk], dim=0)
            with torch.no_grad():
                imgs = self._model.generate(
                    texts,
                    style,
                    latent_hw=(latent_h, latent_w),
                    num_steps=self.num_steps,
                    cfg_scale=self.cfg_scale,
                )
            for k, img in zip(chunk, imgs, strict=True):
                inks[k] = ink_from_pil(ink_crop(to_pil(img)), jobs[k].height)
        # Every slot must be filled — silently dropping a None would shift the list and misalign
        # inks with their boxes downstream (labels on the wrong lines).
        filled = [ink for ink in inks if ink is not None]
        if len(filled) != len(jobs):
            raise RuntimeError(f"render_jobs left {len(jobs) - len(filled)} unfilled slots")
        return filled
