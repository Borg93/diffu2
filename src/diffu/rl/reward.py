"""Black-box legibility reward for recognizer-reward fine-tuning.

``reward = 1 − CER(read(image), requested_text)`` — the recognizer is run FORWARD only; there is **no
backprop through it** (that is the whole point vs the removed differentiable recog-loss, which collapsed
because the recognizer's gradient is adversarial). See docs/RL_PLAN.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..recognizer import char_error_rate, read_lines

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def htr_cer_reward(
    images: list[PILImage],
    texts: list[str],
    recognizer: tuple[TrOCRProcessor, VisionEncoderDecoderModel],
    *,
    clamp: bool = True,
) -> list[float]:
    """Per-image reward = ``1 − CER`` (1.0 = perfectly legible, ≤0 = gibberish).

    ``recognizer`` MUST be **held out** from ``cfg.aux.htr_recognizer_eval`` (Goodhart: a model trained
    against its own judge games the judge). ``clamp`` floors the reward at 0 so a wildly-wrong read (CER > 1)
    doesn't dominate the group advantage.
    """
    if len(images) != len(texts):
        raise ValueError(f"images ({len(images)}) and texts ({len(texts)}) length mismatch")
    preds = read_lines(recognizer[0], recognizer[1], list(images))
    rewards = [1.0 - char_error_rate(pred, gt) for pred, gt in zip(preds, texts, strict=True)]
    return [max(0.0, r) for r in rewards] if clamp else rewards
