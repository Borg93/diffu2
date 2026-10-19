"""Evaluation: HTR Character Error Rate over a manifest of (image, transcription) lines.

The success metric for Diffu is downstream HTR CER on real Swedish — this module is the reusable
core: run a handwriting recognizer over a manifest and report CER. The recognizer is injectable so
the CER logic is testable without loading a model, and ``HTRRecognizer`` is reused by the Stage-0
VAE recon-CER gate (``model/vae.py``).

    diffu-eval --manifest data_out/val.jsonl --limit 200
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

import torch
from PIL import Image
from pydantic import BaseModel

from .config import AuxLossConfig
from .data.dataset import load_manifest
from .model.metrics import cer
from .recognizer import trim_to_ink

STANDARD_HTR = AuxLossConfig().htr_recognizer_eval  # the char-level historical-Swedish TrOCR

Recognizer = Callable[[Sequence[Image.Image]], list[str]]


class EvalResult(BaseModel):
    cer: float
    n: int
    samples: list[tuple[str, str]]  # a few (transcription, prediction) pairs to eyeball


def _materialize_trocr_positions(model: torch.nn.Module, device: str) -> None:
    """Work around a transformers-5.x TrOCR bug.

    ``TrOCRSinusoidalPositionalEmbedding.weights`` is a plain attribute (not a registered buffer) built
    under the meta-init context, so it stays on the ``meta`` device, ``.to()`` can't move it, and the
    first forward crashes with "Cannot copy out of meta tensor". Rebuild the (deterministic) sinusoidal
    table on the real device.
    """
    from transformers.models.trocr.modeling_trocr import TrOCRSinusoidalPositionalEmbedding

    for module in model.modules():
        if isinstance(module, TrOCRSinusoidalPositionalEmbedding):
            num = module.weights.shape[0]
            module.weights = module.get_embedding(num, module.embedding_dim, module.padding_idx).to(device)


class HTRRecognizer:
    """Riksarkivet Swedish TrOCR wrapper. Callable on PIL images or a ``[-1, 1]`` image tensor."""

    def __init__(
        self,
        model_id: str = STANDARD_HTR,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        # local_files_only dodges an online chat-template probe that 404s for the Swedish-Lion repo.
        self.processor = TrOCRProcessor.from_pretrained(model_id, local_files_only=True)
        self.model = (
            VisionEncoderDecoderModel.from_pretrained(model_id, local_files_only=True).to(self.device).eval()
        )
        _materialize_trocr_positions(self.model, self.device)

    @torch.no_grad()
    def __call__(self, images: Sequence[Image.Image] | torch.Tensor) -> list[str]:
        pils = _tensor_to_pils(images) if isinstance(images, torch.Tensor) else list(images)
        preds: list[str] = []
        for start in range(0, len(pils), self.batch_size):
            # trim_to_ink crops blank canvas so the recognizer reads the writing, not padding — without
            # it a short line on a wide (batch-padded) canvas inflates CER ~25x (proven: 0.42 -> 0.016).
            batch = [trim_to_ink(im.convert("RGB")) for im in pils[start : start + self.batch_size]]
            pv = self.processor(images=batch, return_tensors="pt").pixel_values.to(self.device)
            # interpolate_pos_encoding: the Swedish-Lion was trained at a wide aspect; this lets its
            # ViT encoder accept our variable-width line crops instead of assuming a square input.
            ids = self.model.generate(pv, max_new_tokens=64, interpolate_pos_encoding=True)
            preds.extend(p.strip() for p in self.processor.batch_decode(ids, skip_special_tokens=True))
        return preds


def _tensor_to_pils(images: torch.Tensor) -> list[Image.Image]:
    """``[B, 3, H, W]`` in ``[-1, 1]`` -> list of PIL images (``.float()``: torchvision can't convert bf16)."""
    from torchvision.transforms import functional as TF

    return [TF.to_pil_image(((img.clamp(-1, 1) + 1) / 2).float().cpu()) for img in images]


def evaluate_manifest(manifest_path: str, recognizer: Recognizer, *, limit: int | None = None) -> EvalResult:
    """Read a manifest, recognize each line image, and return CER vs the transcriptions."""
    rows = load_manifest(manifest_path)
    if limit is not None:
        rows = rows[:limit]
    gts = [r["text"] for r in rows]
    images = [_open(r["image"]) for r in rows]
    preds = recognizer(images)
    return EvalResult(
        cer=cer(preds, gts),
        n=len(rows),
        samples=list(zip(gts[:5], preds[:5], strict=False)),
    )


def _open(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description="HTR CER over a line manifest (the Diffu success metric).")
    ap.add_argument("--manifest", required=True, help="jsonl with {image, text} rows (e.g. val.jsonl)")
    ap.add_argument("--htr", default=STANDARD_HTR, help="HF TrOCR id (default: the standard recognizer)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    recognizer = HTRRecognizer(args.htr, batch_size=args.batch_size)
    result = evaluate_manifest(args.manifest, recognizer, limit=args.limit)
    print(f"CER {result.cer:.4f} over {result.n} lines")
    for gt, pred in result.samples:
        print(f"  gt={gt!r}  pred={pred!r}")


if __name__ == "__main__":
    main()
