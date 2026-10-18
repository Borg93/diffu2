"""Gradio-thin services behind the studio events — each takes a typed form, not a wall of positional knobs.

The view modules build the widgets and wire ``.click``/``.release``/``.change``; each event's adapter reads its
control-bag into a form (see :mod:`diffu_studio.controls`) and calls one of these. Streaming services yield
lists that map positionally onto the event's ``outputs`` list; the cheap re-composite/re-seat services return
one list. The GPU-bound streams (line, page) are wrapped in ``gpu_task`` at the wiring site.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr
from PIL import Image

from diffu_studio.checkpoints import CheckpointMismatch
from diffu_studio.device import torch_dtype
from diffu_studio.line import stream_line, style_from_pil
from diffu_studio.page import live_update_page, recompose_page, stream_page
from diffu_studio.viz import render_label_overlay

if TYPE_CHECKING:
    from collections.abc import Iterator

    from diffu_page.labels import PageLabels

    from diffu_studio.engine import StudioEngine
    from diffu_studio.forms import AugmentForm, LineForm, PageForm
    from diffu_studio.steps import RenderCache

_DEFAULT_FORMATS = ["page", "alto", "json"]


def write_labels(labels: dict[str, str]) -> list[str]:
    """Serialize the emitted label formats to temp files for download; return the paths."""
    ext = {"page": ".page.xml", "alto": ".alto.xml", "json": ".json"}
    tmp = Path(tempfile.gettempdir())
    out: list[str] = []
    for fmt, text in labels.items():
        path = tmp / f"diffu_studio_page{ext.get(fmt, '.' + fmt)}"
        path.write_text(text, encoding="utf-8")
        out.append(str(path))
    return out


def safe_overlay(page: Image.Image, page_labels: PageLabels) -> Image.Image | None:
    """Render the GT-label overlay, but never let a drawing failure blank the page — log and fall back to None.

    The overlay is an eyeball aid layered on the final frame; a bad polygon must not abort the whole generate.
    """
    try:
        return render_label_overlay(page, page_labels.lines)
    except Exception as exc:  # noqa: BLE001 — debug overlay must degrade gracefully, not crash the stream
        print(
            f"[overlay] render failed ({type(exc).__name__}: {exc}) — showing the page without labels",
            file=sys.stderr,
        )
        return None


def _style(engine: StudioEngine, ref: Image.Image):
    return style_from_pil(ref, engine.device.torch_device, torch_dtype(engine.device))


def stream_line_view(
    engine: StudioEngine,
    form: LineForm,
    *,
    ref: Image.Image | None,
    ckpt: str,
    lora: str | None = None,
    lora_scale: float = 1.0,
) -> Iterator[list]:
    """Stream single-line denoise → [image, readback, status]. One model, no A/B compare."""
    if ref is None or not form.text.strip():
        yield [None, "", "Pick a style reference and enter text."]
        return
    try:
        gen = stream_line(
            engine.model(ckpt, lora=lora, lora_scale=lora_scale),
            _style(engine, ref),
            form.text,
            form.settings,
            recognizer=engine.recognizer,
        )
    except CheckpointMismatch as exc:
        yield [None, "", f"⚠ {exc}"]
        return
    image: Image.Image | None = None
    read = ""
    for step in gen:
        image, read = step.image, (step.read or read)
        yield [image, read, "⟳ generating…"]
    yield [image, read, "✓ done"]


def stream_page_view(
    engine: StudioEngine,
    form: PageForm,
    *,
    ref: Image.Image | None,
    ckpt: str,
    lora: str | None = None,
    lora_scale: float = 1.0,
) -> Iterator[list]:
    """Stream whole-page build → [page, active_line, overlay, status, label_files, render_cache]."""
    if ref is None:
        yield [None, None, None, "Pick a style reference.", None, None]
        return
    if not form.text.strip() and not form.xml_layout:
        yield [None, None, None, "Enter text, or upload an ALTO/PAGE XML layout.", None, None]
        return
    request = form.to_request(animate=engine.device.interactive)
    try:
        style = _style(engine, ref)
        for step in stream_page(engine.model(ckpt, lora=lora, lora_scale=lora_scale), style, request):
            if step.phase == "final" and step.page_labels is not None and step.page is not None:
                overlay = safe_overlay(step.page, step.page_labels)
                yield [step.page, None, overlay, step.status, write_labels(step.labels or {}), step.render]
            else:
                yield [step.page, step.active, None, step.status, None, None]
    except CheckpointMismatch as exc:
        yield [None, None, None, f"⚠ {exc}", None, None]


def recompose_view(cache: RenderCache | None, form: AugmentForm, formats: list[str], *, seed: int) -> list:
    """Re-composite the CACHED page with new scan-look augmentation — no model, no GPU. ``seed`` re-rolls the look."""
    if cache is None:
        return [
            gr.update(),
            gr.update(),
            gr.update(),
            "Generate a page first — nothing cached to re-composite.",
        ]
    step = recompose_page(cache, form.to_augment(), formats or _DEFAULT_FORMATS, seed=seed)
    overlay = safe_overlay(step.page, step.page_labels) if step.page and step.page_labels else None
    return [step.page, overlay, write_labels(step.labels or {}), step.status]


def reseat_view(
    cache: RenderCache | None, form: AugmentForm, line_gap: float, para_gap: float, *, seed: int
) -> list:
    """Re-seat the CACHED inks at new line/paragraph spacing — no model. Falls back to Generate on a re-wrap."""
    if cache is None:
        return [
            gr.update(),
            gr.update(),
            gr.update(),
            "Generate a page first — nothing cached to re-seat.",
            gr.update(),
        ]
    gen = cache.gen.model_copy(
        update={"line_gap": line_gap, "para_gap": para_gap, "augment": form.to_augment()}
    )
    step = live_update_page(cache, gen, seed=seed)
    if step.phase == "error" or step.page is None:
        return [gr.update(), gr.update(), gr.update(), step.status, gr.update()]
    overlay = safe_overlay(step.page, step.page_labels) if step.page_labels else None
    return [step.page, overlay, write_labels(step.labels or {}), step.status, step.render]


def upscale_view(page: Image.Image | None) -> list:
    """4× super-resolve the CURRENT page with PiD → [(before, after), status]. Fallback-safe (button-only)."""
    if page is None:
        return [None, "Generate a page first, then upscale it."]
    from diffu_studio.upscale import pid_available, pid_upscale

    before = page if isinstance(page, Image.Image) else Image.fromarray(page)
    if not pid_available():
        return [(before, before), "⚠ PiD upscaler not installed on this device — showing the original."]
    hires = pid_upscale(before, prompt="")
    if hires is not None and hires is not before:
        note = f"✓ 4× upscaled (PiD) — {before.size[0]}×{before.size[1]} → {hires.size[0]}×{hires.size[1]}"
    else:
        note = (
            "⚠ upscale failed — most likely not enough free VRAM next to the other jobs on this GPU "
            "(details in the server log); showing the original."
        )
    return [(before, hires or before), note]
