"""The read side of the sets htr-train trains on: each line of a ``diffu-reader-set`` or ``htr-train jev-set``
meta file (:class:`SampleMeta`, the fields the leak checks and the dev split read), and a set's samples as ms-swift
reads them (:class:`Sample`). htr-train does not import diffu-page (its torch and transformers pins would collide in
the tool environments), so it reads the files, not the classes that wrote them."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from diffu_htr_train.output import validation_problems


class SampleMeta(BaseModel):
    """One sample's meta: its id, whether its page is ``real`` or ``synthetic`` (drawn), its collection and volume (""
    where its input gives none), its split (``train``, ``test``, or None for an unsplit set) and, for a Jev pair, the
    renderer its candidate was rendered with and the prompt it was built in. Other fields (side, kind, production, a pair's candidate) are kept as
    written."""

    model_config = ConfigDict(frozen=True, extra="allow")

    id: str
    source: Literal["real", "synthetic"]
    collection: str = ""
    volume: str = ""
    split: Literal["train", "test"] | None = None
    renderer: str = ""
    prompt: str = ""

    def is_of(self, side: str) -> bool:
        """Whether the sample may be used as one of ``side``: a drawn one always; a real one only when its set was split
        by the GT's test lists and it is on that side (a real sample of an unsplit set may be of a test page)."""
        return self.source == "synthetic" or self.split == side


class Turn(BaseModel):
    """One turn of a sample's conversation."""

    model_config = ConfigDict(frozen=True)

    role: str
    content: str


class Sample(BaseModel):
    """One line of a set as ms-swift reads it: the conversation, the images its ``<image>`` placeholders show, and a
    sequence classifier's label (Jev's 0 or 1)."""

    model_config = ConfigDict(frozen=True)

    messages: list[Turn]
    images: list[str]
    label: int | None = None


def read_jsonl[ModelT: BaseModel](path: Path, model: type[ModelT]) -> list[ModelT]:
    """Each non-blank line of the JSON-lines file at ``path`` as ``model``.

    Raises:
        ValueError: If a line is not one, naming the file and line.
    """
    rows: list[ModelT] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(model.model_validate_json(line))
        except ValidationError as error:
            raise ValueError(f"{path}:{number}: {validation_problems(error, 'line')}") from error
    return rows


def read_meta(path: Path) -> list[SampleMeta]:
    """A set's meta, a line a sample.

    Raises:
        ValueError: If a line is not a sample's meta.
    """
    return read_jsonl(path, SampleMeta)


def meta_path(path: Path) -> Path:
    """A set's meta file: ``<stem>.meta.jsonl`` beside it."""
    return path.with_name(f"{path.stem}.meta.jsonl")


def skipped_path(path: Path) -> Path:
    """A set's record of what it left out: ``<stem>.skipped.jsonl`` beside it."""
    return path.with_name(f"{path.stem}.skipped.jsonl")


def target_record_path(path: Path) -> Path:
    """A response set's record of the target that answered it: ``<stem>.target.json`` beside it."""
    return path.with_name(f"{path.stem}.target.json")


def options_path(path: Path) -> Path:
    """A set's record of how it was built: ``<stem>.options.json`` beside it."""
    return path.with_name(f"{path.stem}.options.json")
