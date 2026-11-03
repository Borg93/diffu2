"""MiniCPM-OCR-Jev's prompt: the bytes its checkpoint was trained on, which a fine-tune keeps (a changed byte changes
what its scores mean). In a training sample (:func:`sample_prompt`) the two images are ms-swift's ``<image>``
placeholders, each followed by a line break; the model reads the turn as :func:`model_prompt` gives it, whose digest
with ``{text}`` for the text is the one OpenBMB publishes (``scripts/minicpm-ocr-jev/infer_vllm.py``: original image,
the candidate text, its rendering, the instruction, then the empty assistant turn whose last token is pooled)."""

from __future__ import annotations

import hashlib

IMAGE = "<image>"  # ms-swift's placeholder; the qwen3_5 template shows it as QWEN_IMAGE
QWEN_IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
TEXT_LABEL = "OCR文本："
INSTRUCTION = (
    "第一张图是原图，第二张图是上述 OCR 文本渲染得到的图片。\n"
    "请判断该 OCR 文本及其对应渲染图是否与原图内容一致。\n"
    "这是一个二分类任务：一致标为 1，不一致标为 0。"
)
ASSISTANT_TURN = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PROMPT_MD5 = "9aab68e8ba8ab4da17ba9752bddb199c"  # OpenBMB's, of model_prompt("{text}")


def _turn(text: str, image: str) -> str:
    return f"{image}\n{TEXT_LABEL}\n{text}\n{image}\n{INSTRUCTION}"


def sample_prompt(text: str) -> str:
    """The user turn of a Jev training sample for the candidate ``text``, as given: its crop's placeholder, the text,
    its rendering's placeholder and the instruction."""
    return _turn(text, IMAGE)


def model_prompt(text: str) -> str:
    """The whole prompt Jev reads for the candidate ``text``: the user turn with Qwen3.5's image tokens and the empty
    assistant turn."""
    return f"<|im_start|>user\n{_turn(text, QWEN_IMAGE)}{ASSISTANT_TURN}"


def prompt_digest() -> str:
    """The md5 of :func:`model_prompt` with ``{text}`` for the text, to compare with :data:`PROMPT_MD5`."""
    return hashlib.md5(model_prompt("{text}").encode("utf-8")).hexdigest()
