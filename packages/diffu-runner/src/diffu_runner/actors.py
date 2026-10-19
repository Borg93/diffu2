"""The two stateful pipeline stages, as Ray-Data actor classes.

``DiffuActor`` generates a batch of line images from (text, style) pairs; ``TrOCRActor`` reads each back and
scores CER. Each loads its model ONCE per worker (that's the point of a stateful actor — no reload per
batch) and is called by Ray Data as ``actor(batch) -> batch``. Both wrap the ``diffu`` package; neither
imports Ray, so they're unit-testable off-cluster.

Batches are plain dicts of column-lists (Ray Data's ``map_batches`` "numpy"/"pandas" batch is adapted in
:mod:`diffu_runner.pipeline`); here we take/return ``dict[str, list]``.
"""

from __future__ import annotations

from pathlib import Path

from PIL.Image import Image


def bucket_plan(widths: list[int], *, bucket_px: int, max_width: int) -> list[tuple[int, list[int]]]:
    """Plan a width-bucketed generate pass — pure arithmetic, unit-testable without torch.

    Deliberately DUPLICATES the logic of ``diffu_page.render.diffu.bucket_render_batches`` instead of
    importing it: diffu-runner's base deps exclude diffu-page (the compositor is only the ``[page]``
    extra), so the line actor must stay importable without it. Keep the two in sync by idea, not code.

    Each natural ``width`` is quantised UP to a multiple of ``bucket_px`` (coarse buckets → few distinct
    canvas shapes), clamped to ``[bucket_px, max_width]``, then rounded UP to a multiple of 16 px — the
    same ``((w + 15) // 16) * 16`` canvas invariant the ctor applies, so every bucket stays divisible by
    the VAE downscale factor. Indices sharing a bucket width are grouped (one padded canvas serves the
    whole group). Returns ``(bucket_width_px, batch_indices)`` covering every index exactly once, in
    ascending bucket-width order. No ``max_batch`` split here: Ray Data's ``gen_batch`` already caps the
    incoming batch, and each bucket is a subset of it — never bigger than what the old fixed-width path
    pushed through ``generate`` in one call.
    """
    if bucket_px < 1:
        raise ValueError(f"bucket_px must be >= 1, got {bucket_px}")

    def quantise(w: int) -> int:
        q = ((w + bucket_px - 1) // bucket_px) * bucket_px  # quantise UP — never crop a line's canvas
        q = min(max(q, bucket_px), max_width)
        return ((q + 15) // 16) * 16  # ctor's canvas invariant, per bucket

    bucket = [quantise(w) for w in widths]
    plan: list[tuple[int, list[int]]] = []
    for i in sorted(range(len(widths)), key=lambda i: bucket[i]):  # stable → indices ascend within a group
        if plan and plan[-1][0] == bucket[i]:
            plan[-1][1].append(i)
        else:
            plan.append((bucket[i], [i]))
    return plan


class DiffuActor:
    """Generate one line image per (text, style_path). Model + one config loaded once per worker.

    Runs the same bf16 + regional-``torch.compile`` recipe as the released inference path
    (``DiffuPipeline`` / ``generate.py`` — see ``docs/PERF_AUDIT.md`` §0e/§I): bf16 end-to-end is validated
    numerically safe on this model, and ``Backbone.compile_blocks()`` fuses the repeated MMDiT block
    (``dynamic=True``, so the pipeline's width-bucketed batches amortize into a handful of compiles instead
    of recompiling every distinct width). Both are ON by default; disable per-run via the constructor args
    (threaded through ``RunConfig``/the ``diffu-runner`` CLI) if a checkpoint or box misbehaves.
    """

    def __init__(self, ckpt: str, *, line_height: int = 128, num_steps: int = 24, cfg_scale: float = 5.0,
                 width: int = 1024, bucket_px: int | None = None, compile_model: bool = True,
                 dtype: str = "bfloat16", lora: str | None = None, lora_scale: float = 1.0) -> None:  # fmt: skip
        import torch

        from diffu.generate import build_model, load_config

        cfg = load_config(ckpt)  # the config.json diffu-train saves next to the weights
        if not (Path(ckpt).parent / "config.json").is_file():  # an older run: no saved config
            cfg.data.line_height = line_height

        self._steps, self._scale = num_steps, cfg_scale
        self._cfg = cfg  # kept on self: __call__ needs its VAE geometry per batch
        self._latent_h = cfg.data.line_height // cfg.vae.downscale_factor
        # ``bucket_px=None`` (the default, and the only mode validated on THIS checkpoint) renders every
        # line on the fixed ``width`` canvas — the glyph-line conditioning aligns each text's glyph strip
        # against the BATCH-max strip width, and the released model was trained only on batches whose
        # longest line ~fills the canvas. Width-bucketing (``bucket_px`` set) would save the 30-45% of
        # compute spent denoising padding, but width-homogeneous groups put the conditioning out of its
        # trained regime and the model STUTTERS (same 8 short lines: CER ≈ 1.0 alone vs ≈ 0.00 mixed with
        # long batchmates — measured 2026-07-13). Re-enable only for a checkpoint whose conditioning pads
        # glyph strips canvas-proportionally at train time. The pipeline additionally SHUFFLES work-lists
        # so fixed-canvas batches always mix lengths (see _shuffle_for_mixed_length_batches).
        self._max_width = ((width + 15) // 16) * 16
        self._bucket_px = bucket_px
        self._device = "cuda"  # single knob the unit tests stub to "cpu" (actors run off-cluster there)
        self._dtype = getattr(torch, dtype)
        # build_model: TF32 matmuls, weights + the conditioner guard, the Stage-3 LoRA adapter (dir or Hub id)
        # BEFORE compile (peft mutates the modules), then the regional compile — see Backbone.compile_blocks
        self._model = build_model(
            cfg, ckpt, device=self._device, dtype=self._dtype, lora=lora, lora_scale=lora_scale,
            compile_backbone=compile_model,
        )  # fmt: skip

    def __call__(self, batch: dict[str, list]) -> dict[str, list]:
        import torch

        from diffu.generate import auto_canvas_width, load_style, to_pil

        texts, styles = list(batch["text"]), list(batch["style_path"])
        if self._bucket_px is None:  # fixed canvas — the validated mode; see the ctor comment
            plan = [(self._max_width, list(range(len(texts))))]
        else:
            widths = [auto_canvas_width(self._model, t or " ") for t in texts]
            plan = bucket_plan(widths, bucket_px=self._bucket_px, max_width=self._max_width)
        pngs: list[bytes | None] = [None] * len(texts)
        for bucket_w, idxs in plan:
            # dtype must match the (bf16-by-default) model — the frozen DINO style encoder errors on a
            # weight/input dtype mismatch otherwise.
            style = torch.stack([load_style(styles[i]) for i in idxs]).to(
                self._device, dtype=self._dtype
            )  # [b,3,S,S]
            with torch.inference_mode():
                imgs = self._model.generate(
                    [texts[i] for i in idxs], style,
                    latent_hw=(self._latent_h, bucket_w // self._cfg.vae.downscale_factor),
                    num_steps=self._steps, cfg_scale=self._scale,
                )  # fmt: skip
            # PNG bytes, not raw arrays: a clean variable-width object column for Ray Data / parquet.
            # Scatter back to the ORIGINAL batch positions — a shifted slot would pair a png with the
            # wrong text downstream, i.e. mislabeled training data.
            for i, img in zip(idxs, imgs, strict=True):
                pngs[i] = _png_bytes(to_pil(img))
        if any(p is None for p in pngs):  # bucket_plan covers every index exactly once; fail loud if not
            raise RuntimeError("width-bucketed generate left unfilled slots")
        # pass every input column through (style_path, upcoming doc_id/line_idx, ...) so downstream
        # gate/write stages see whole rows — mirrors TrOCRActor.
        return {**{k: list(v) for k, v in batch.items()}, "png": pngs}


class TrOCRActor:
    """Read each line image back to text and score CER vs the intended transcription (the quality gauge)."""

    def __init__(self, recognizer: str = "Riksarkivet/trocr-base-handwritten-hist-swe-3") -> None:
        from diffu.recognizer import load_recognizer

        self._proc, self._model = load_recognizer(recognizer, "cuda")

    def __call__(self, batch: dict[str, list]) -> dict[str, list]:
        from diffu.recognizer import char_error_rate, read_lines

        images = [_from_png(b) for b in batch["png"]]
        preds = read_lines(self._proc, self._model, images)
        cers = [char_error_rate(p, t) for p, t in zip(preds, list(batch["text"]), strict=True)]
        # pass every input column through (e.g. the ink_ok flag) so driver-side accounting sees whole rows
        return {**{k: list(v) for k, v in batch.items()}, "pred": preds, "cer": cers}


def _png_bytes(img: Image) -> bytes:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _from_png(data: bytes) -> Image:
    import io

    from PIL import Image as PILImage

    return PILImage.open(io.BytesIO(data)).convert("RGB")
