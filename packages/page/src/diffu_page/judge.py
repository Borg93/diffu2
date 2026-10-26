"""Judges: what a drawn image shows in each line box, against the text its label will carry.

A judge is its own stage after drawing: it loads its model once (an actor's ``__init__`` on a cluster) and maps a drawn
image and its lines to one :class:`LineVerdict` per line. :class:`TrOCRJudge` reads each box with a Swedish HTR model;
an OCR, VLM or layout model judges the same way behind the :class:`Judge` protocol. The labels then tag every line
whose verdict fails (:meth:`LineVerdict.passes`), so no label claims a text its image contradicts.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import Protocol

import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from diffu_page.captions.prompt import LineRecord
from diffu_page.readback import cer, materialize_position_tables, numbers_differ, read_lines
from diffu_page.settings import settings

MAX_LINE_CER = 0.25  # a line that reads back with a higher CER is labelled unclear


class LineVerdict(BaseModel):
    """What a judge read in one line box, its CER against the line's text, and whether a number in it reads otherwise."""

    model_config = ConfigDict(frozen=True)

    read: str
    cer: float
    numbers_differ: bool

    @classmethod
    def of(cls, read: str, text: str) -> LineVerdict:
        """The verdict on ``read`` as a reading of ``text``."""
        return cls(read=read, cer=cer(read, text), numbers_differ=numbers_differ(read, text))

    def passes(self, max_line_cer: float) -> bool:
        """Whether the line's label can carry its text: a CER of at most ``max_line_cer`` and every number read right."""
        return self.cer <= max_line_cer and not self.numbers_differ


class Judge(Protocol):
    """Reads the line boxes of a drawn image: one verdict per line, in the lines' order."""

    def __call__(self, image: Image.Image, lines: Sequence[LineRecord]) -> list[LineVerdict]: ...


class TrOCRJudge:
    """A TrOCR HTR model reading every line box (by default the Swedish model ``DIFFU_HTR_MODEL`` names, pinned)."""

    def __init__(
        self, model: str = settings.htr_model, *, revision: str | None = settings.htr_revision, device: str = "cuda"
    ) -> None:
        self.model = VisionEncoderDecoderModel.from_pretrained(model, revision=revision, dtype=torch.bfloat16)
        self.model.to(device).eval()
        materialize_position_tables(self.model)
        self.processor = TrOCRProcessor.from_pretrained(model, revision=revision)

    def __call__(self, image: Image.Image, lines: Sequence[LineRecord]) -> list[LineVerdict]:
        reads, _ = read_lines(image, list(lines), model=self.model, processor=self.processor)
        return [LineVerdict.of(read, line.text) for read, line in zip(reads, lines, strict=True)]


@cache
def default_judge(device: str) -> TrOCRJudge:
    """The default TrOCR judge on ``device``, loaded once per process for callers that bring no judge of their own."""
    return TrOCRJudge(device=device)
