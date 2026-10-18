"""Frozen Swedish HTR recognizer — the legibility gauge (CER) for generated lines.

The recognizer reads a generated line back to text; CER vs the requested text = how legible we are.
Used by scripts/eval_cer.py (offline report) and train.py (live gen_CER logged to trackio).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def char_error_rate(pred: str, gt: str) -> float:
    """Character-level Levenshtein(pred, gt) / len(gt). 0 = perfect, ~1+ = illegible."""
    if not gt:
        return 1.0 if pred else 0.0
    prev = list(range(len(gt) + 1))
    for i, pc in enumerate(pred, 1):
        cur = [i]
        for j, gc in enumerate(gt, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (pc != gc)))
        prev = cur
    return prev[-1] / len(gt)


def load_recognizer(
    name: str, device: torch.device | str
) -> tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    """Load the frozen TrOCR recognizer from local cache.

    local_files_only dodges an online chat-template probe that 404s for this repo; device_map places
    weights straight on the GPU (a plain .to() can trip a meta-tensor copy error after Diffu builds).
    """
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    proc = TrOCRProcessor.from_pretrained(name, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(name, local_files_only=True, device_map=device).eval()
    return proc, model


def trim_to_ink(img: PILImage, margin: int = 40, pad: int = 6) -> PILImage:
    """Crop to the inked span so TrOCR reads only what was written, not blank paper.

    Our lines sit on tan/aged paper (not white), so 'ink' = columns containing pixels clearly darker
    than the paper median. Without this the recognizer (a language model) hallucinates plausible
    Swedish over the empty right-hand canvas, inflating CER — measuring overflow, not legibility.
    """
    gray = np.asarray(img.convert("L"), dtype=np.int16)
    paper = int(np.median(gray))
    ink_cols = np.where((gray < paper - margin).any(axis=0))[0]
    if ink_cols.size == 0:
        return img
    left = max(int(ink_cols[0]) - pad, 0)
    right = min(int(ink_cols[-1]) + pad + 1, img.width)
    return img.crop((left, 0, right, img.height))


@torch.no_grad()
def read_lines(
    proc: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    images: list[PILImage],
    *,
    trim: bool = True,
) -> list[str]:
    """Recognize each line to text. ``trim`` crops blank paper first so CER reflects legibility, not
    overflow (interpolate_pos_encoding handles the resulting variable widths)."""
    out: list[str] = []
    for img in images:
        crop = trim_to_ink(img) if trim else img
        pv = proc(images=crop.convert("RGB"), return_tensors="pt").pixel_values.to(model.device)
        ids = model.generate(pv, max_new_tokens=64, interpolate_pos_encoding=True)
        out.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return out
