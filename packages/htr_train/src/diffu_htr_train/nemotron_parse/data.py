"""A Nemotron-Parse set as the trainer reads it, and its pages batched as the model reads them.

Each sample (``diffu-reader-set --model nemotron-parse``) is checked to be a page asked the task prompt
(:data:`diffu_htr_train.nemotron_parse.USER_TURN`) and answered; its answer is tokenized once, up front, as the model's
own inference tokenizes text (``add_special_tokens=False``, its ``inference.py``), and ended by ``</s>``, where
generation stops; a page whose answer is longer than the trainer's limit is left out and counted. A batch
(:class:`PageCollator`) is its pages as the model's processor makes them (packed 16 x 16 patches, each page's padded
and content sizes) and, a row a page, the prompt's tokens and the answer's, the answer alone labelled, padded on the
right (the model has no positions, so padding only needs masking).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict
from torch.utils.data import Dataset

from diffu_htr_train.nemotron_parse import USER_TURN
from diffu_htr_train.records import Sample, read_jsonl

IGNORE = -100  # a label the loss skips: the prompt and the padding


class PageSample(BaseModel):
    """One page to learn: its image and its answer, the sample's assistant turn."""

    model_config = ConfigDict(frozen=True)

    image: Path
    answer: str


def read_pages(path: Path) -> list[PageSample]:
    """A set's pages, each sample checked to be Nemotron-Parse's: a user turn asking :data:`USER_TURN` of one image,
    then the assistant's answer.

    Raises:
        ValueError: If a line is not a sample, or not such a one.
    """
    pages: list[PageSample] = []
    for number, row in enumerate(read_jsonl(path, Sample), start=1):
        roles = [turn.role for turn in row.messages]
        if roles != ["user", "assistant"] or row.messages[0].content != USER_TURN or len(row.images) != 1:
            raise ValueError(
                f"{path}: sample {number} is no Nemotron-Parse page: one image asked {USER_TURN!r} and answered "
                "(build the set with diffu-reader-set --model nemotron-parse)"
            )
        pages.append(PageSample(image=Path(row.images[0]), answer=row.messages[1].content))
    return pages


class Tokenizer(Protocol):
    """The part of the model's tokenizer the sets use."""

    eos_token_id: int
    pad_token_id: int

    def __call__(self, text: list[str], *, add_special_tokens: bool) -> Any: ...


class PageDataset(Dataset[dict[str, Any]]):
    """Pages to learn, each its image (read when it is asked for, as RGB) and its answer's tokens."""

    def __init__(self, pages: list[PageSample], answers: list[np.ndarray]) -> None:
        self.pages = pages
        self.answers = answers

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with Image.open(self.pages[index].image) as image:
            return {"image": image.convert("RGB"), "answer": self.answers[index]}


class LoadedSets(BaseModel):
    """What reading sets came to: the pages kept and those left out for an answer past the limit."""

    model_config = ConfigDict(frozen=True)

    kept: int
    too_long: int


TOKENIZED_AT_ONCE = 1024  # answers tokenized in one call: each call's ids are kept as int32 arrays before the next


def page_dataset(paths: Sequence[Path], tokenizer: Tokenizer, *, max_tokens: int) -> tuple[PageDataset, LoadedSets]:
    """The pages of the sets at ``paths``, in order, each answer tokenized as the model's inference tokenizes text and
    ended by ``</s>``; a page whose answer then holds more than ``max_tokens`` left out.

    Raises:
        ValueError: If a set holds a sample that is not a Nemotron-Parse page (:func:`read_pages`).
    """
    pages = [page for path in paths for page in read_pages(path)]
    kept_pages: list[PageSample] = []
    answers: list[np.ndarray] = []
    for start in range(0, len(pages), TOKENIZED_AT_ONCE):
        chunk = pages[start : start + TOKENIZED_AT_ONCE]
        encoded = tokenizer([page.answer for page in chunk], add_special_tokens=False)["input_ids"]
        for page, ids in zip(chunk, encoded, strict=True):
            if len(ids) + 1 > max_tokens:
                continue
            kept_pages.append(page)
            answers.append(np.asarray([*ids, tokenizer.eos_token_id], dtype=np.int32))
    return PageDataset(kept_pages, answers), LoadedSets(kept=len(kept_pages), too_long=len(pages) - len(kept_pages))


class ImageProcessor(Protocol):
    """The model's image processor: pages to packed patches with their padded and content sizes."""

    def __call__(self, images: list[Image.Image]) -> dict[str, torch.Tensor]: ...


class PageCollator:
    """Pages batched as the model reads them: their images through its processor (``pixel_values``, ``image_sizes``,
    ``content_sizes``), then a row a page of the task prompt's tokens (``prompt``) and the answer's, padded on the right
    with ``pad``, the answer's alone labelled; the cache off, as nothing is generated."""

    def __init__(self, image_processor: ImageProcessor, prompt: list[int], pad: int) -> None:
        self.image_processor = image_processor
        self.prompt = torch.tensor(prompt, dtype=torch.long)
        self.pad = pad

    def __call__(self, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        vision = self.image_processor([item["image"] for item in items])
        answers = [torch.as_tensor(item["answer"], dtype=torch.long) for item in items]
        start = len(self.prompt)
        width = start + max(len(answer) for answer in answers)
        input_ids = torch.full((len(items), width), self.pad, dtype=torch.long)
        attention_mask = torch.zeros((len(items), width), dtype=torch.long)
        labels = torch.full((len(items), width), IGNORE, dtype=torch.long)
        for row, answer in enumerate(answers):
            end = start + len(answer)
            input_ids[row, :start] = self.prompt
            input_ids[row, start:end] = answer
            attention_mask[row, :end] = 1
            labels[row, start:end] = answer
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": vision["pixel_values"],
            "image_sizes": vision["image_sizes"],
            "content_sizes": vision["content_sizes"],
            "use_cache": False,
        }
