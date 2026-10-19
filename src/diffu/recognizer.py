"""Frozen Swedish HTR recognizer — the legibility gauge (CER) for generated lines.

The recognizer reads a generated line back to text; CER vs the requested text = how legible we are.
Used by train.py (the live gen_CER), eval.py, the VAE gate, the eval scripts and the page/runner packages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .model.metrics import char_error_rate as char_error_rate  # re-exported: callers read CER from here

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel


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
def read_lines(proc: TrOCRProcessor, model: VisionEncoderDecoderModel, images: list[PILImage]) -> list[str]:
    """Recognize each line to text, cropped to its ink first so CER reflects legibility, not overflow
    (interpolate_pos_encoding handles the resulting variable widths)."""
    out: list[str] = []
    for img in images:
        pv = proc(images=trim_to_ink(img).convert("RGB"), return_tensors="pt").pixel_values.to(model.device)
        ids = model.generate(pv, max_new_tokens=64, interpolate_pos_encoding=True)
        out.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return out
