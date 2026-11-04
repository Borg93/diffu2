"""MiniCPM-OCR-Jev's prompt, ours in English: the crop, the candidate text, its rendering, then the instruction to
judge whether they match. Jev is fine-tuned on pairs in this prompt; its fingerprint (:func:`prompt_digest`) is in
every pair's meta, and a Jev run refuses pairs built in another wording, which would change what the scores mean.

In a training sample (:func:`sample_prompt`) the two images are ms-swift's ``<image>`` placeholders, each followed by a
line break; the model reads the turn as :func:`model_prompt` gives it (original image, the candidate text, its
rendering, the instruction, then the empty assistant turn whose last token is pooled)."""

from __future__ import annotations

import hashlib

IMAGE = "<image>"  # ms-swift's placeholder; the qwen3_5 template shows it as QWEN_IMAGE
QWEN_IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
ASSISTANT_TURN = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
TEXT_LABEL = "OCR text:"
INSTRUCTION = (
    "The first image is the original; the second image is the OCR text above rendered as an image.\n"
    "Judge whether the OCR text and its rendering match the content of the original exactly.\n"
    "This is a binary classification: label 1 if they match, 0 if they do not."
)


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
    """The prompt's fingerprint: the first 12 hex digits of the md5 of :func:`model_prompt` with ``{text}`` for the
    text."""
    return hashlib.md5(model_prompt("{text}").encode("utf-8")).hexdigest()[:12]
