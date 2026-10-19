"""Single-line generation — stream the denoise (noise → handwriting), then read it back with TrOCR.

``denoise_frames`` is the shared primitive (page mode reuses it per line): it drives the model's
``denoise_steps`` and yields each decoded x0 estimate as a PIL image. ``stream_line`` wraps that with the
ink-crop of the final frame and an optional recogniser read-back + CER.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import TYPE_CHECKING

from diffu_studio.configs import SampleSettings
from diffu_studio.steps import LineStep

if TYPE_CHECKING:
    import torch
    from PIL.Image import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    from diffu_studio.checkpoints import LoadedModel

type Recognizer = tuple[TrOCRProcessor, VisionEncoderDecoderModel]  # (processor, model) for TrOCR read-back


def style_from_pil(img: Image, device: str, dtype: torch.dtype) -> torch.Tensor:
    """A PIL style-reference line → an ImageNet-normalised ``[1,3,S,S]`` tensor (matches training)."""
    from diffu.generate import load_style

    return load_style(img).unsqueeze(0).to(device, dtype=dtype)


def sample_gallery(manifest: str, n: int = 24, seed: int = 0) -> list[str]:
    """A diverse sample of real line-crop paths for the style picker.

    Deduped by ``writer_id`` — one crop per distinct hand — so the gallery spans many writers instead of
    over-representing a single writer/volume (a blind shuffle clusters same-series crops). Tops up with any
    remaining crops if there are fewer distinct writers than ``n``.
    """
    from diffu.data.dataset import load_manifest

    rows = load_manifest(manifest)
    random.Random(seed).shuffle(rows)
    picked: list[str] = []
    seen_writers: set[object] = set()
    for row in rows:
        writer = row.get("writer_id")
        if writer is not None and writer in seen_writers:
            continue
        seen_writers.add(writer)
        picked.append(row["image"])
        if len(picked) >= n:
            return picked
    have = set(picked)  # fewer distinct writers than n → top up with any remaining crops
    picked.extend(row["image"] for row in rows if row["image"] not in have)
    return picked[:n]


def denoise_frames(
    loaded: LoadedModel, style: torch.Tensor, text: str, settings: SampleSettings
) -> Iterator[Image]:
    """Yield each denoise step (decoded x0: blurry static → sharp ink) for ``text`` in ``style``'s hand."""
    import torch

    from diffu.generate import auto_canvas_width, to_pil

    model, cfg = loaded.model, loaded.cfg
    text = text or " "
    canvas_w = auto_canvas_width(model, text) if settings.auto_width else _round16(settings.width)
    ds = cfg.vae.downscale_factor
    latent_hw = (cfg.data.line_height // ds, max(1, canvas_w // ds))
    with torch.no_grad():
        for img in model.denoise_steps(
            [text], style, latent_hw=latent_hw, num_steps=settings.steps, cfg_scale=settings.cfg_scale
        ):
            yield to_pil(img[0])


def stream_line(
    loaded: LoadedModel,
    style: torch.Tensor,
    text: str,
    settings: SampleSettings,
    *,
    recognizer: Recognizer | None = None,
) -> Iterator[LineStep]:
    """Stream the denoise of one line, then emit the ink-cropped result with an optional TrOCR read-back."""
    total = settings.steps
    if not text.strip():
        yield LineStep(image=None, status="⌨ enter text to render", index=0, total=total, done=True)
        return
    frame: Image | None = None
    for i, fr in enumerate(denoise_frames(loaded, style, text, settings)):
        frame = fr
        yield LineStep(image=fr, status=f"⟳ denoising {i + 1}/{total}", index=i + 1, total=total)
    final = _ink_crop(frame) if (settings.auto_width and frame is not None) else frame
    read = _read_back(recognizer, final, text)
    yield LineStep(image=final, status="✓ done", index=total, total=total, done=True, read=read)


def load_recognizer(name: str, device: str) -> Recognizer | None:
    """Load the TrOCR read-back recogniser (or None if unavailable — the app stays usable without it)."""
    try:
        from diffu.recognizer import load_recognizer as _load

        return _load(name, device)
    except Exception as exc:  # noqa: BLE001 — read-back is optional; never block generation on it
        print(f"[studio] HTR read-back disabled ({type(exc).__name__}: {exc})")
        return None


def _round16(width: int) -> int:
    return ((int(width) + 15) // 16) * 16


def _ink_crop(img: Image) -> Image:
    from diffu.generate import ink_crop

    return ink_crop(img)


def _read_back(recognizer: Recognizer | None, img: Image | None, text: str) -> str | None:
    """TrOCR read-back + CER for a finished line — the same recogniser that gauges gen_CER. None if disabled."""
    if img is None or recognizer is None:
        return None
    import torch

    from diffu.recognizer import char_error_rate, read_lines

    proc, model = recognizer
    with torch.no_grad():
        pred = read_lines(proc, model, [img])[0]
    cer = char_error_rate(pred, text.strip())
    return f"“{pred or '(blank)'}”  ·  CER {cer:.2f}"
