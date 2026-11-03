"""Jev's pairs from a line set's exact labels (``diffu-reader-set --task lines``): each line's crop with its GT text
(label 1) and, about as many, wrong candidates for it (label 0, ``negatives`` a line), drawn from rule perturbations
of the GT text (:mod:`diffu_htr_train.sets.perturb`) and another line of the same page (the most alike in length);
and real readers' reads of the line, a positive where it is the GT after normalisation (OpenBMB's positives are real
OCR outputs annotated as matching) and a negative where it is not. Each candidate is rendered by our renderer
(:mod:`diffu_htr_train.jev.render`) at the crop's size, and each pair written in OpenBMB's seq_cls form (the crop, the
candidate and its rendering, :func:`diffu_htr_train.jev.prompt.sample_prompt`, and the label), its meta beside it: the
line, its page and volume, the candidate's origin and CER to the GT, and the renderer's version.

A training pair set takes training lines only and a test pair set (for Jev's evaluation) test lines only: a line of
the other side, or a real line of an unsplit set, is refused. Jev's own dev split is cut by volume after
(``htr-train split-dev``). A line whose crop cannot be read is left out and recorded.
"""

from __future__ import annotations

import random
import unicodedata
from collections import Counter, defaultdict
from contextlib import ExitStack
from enum import StrEnum
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from diffu_htr_train.jev.prompt import sample_prompt
from diffu_htr_train.jev.render import render_text, renderer_version
from diffu_htr_train.records import (
    Sample,
    SampleMeta,
    Turn,
    meta_path,
    options_path,
    read_jsonl,
    read_meta,
    skipped_path,
)
from diffu_htr_train.sets import SetRefused, read_options
from diffu_htr_train.sets.perturb import Perturbation, perturbed

JPEG_QUALITY = 95


class Side(StrEnum):
    """Which lines a pair set is of: the GT's training lines (to train on) or its test lines (to evaluate on)."""

    TRAIN = "train"
    TEST = "test"


class Read(BaseModel):
    """A reader's reading of a line (``htr-train jev-set --reads``, one a line): the line's sample id, the reader and
    its text."""

    model_config = ConfigDict(frozen=True)

    id: str
    reader: str
    text: str


class Candidate(BaseModel):
    """A text to judge against a line's crop: where it came from (``gt``, ``perturb:<rule>``, ``swap``,
    ``reader:<name>``), the text and its label (1: the crop shows it)."""

    model_config = ConfigDict(frozen=True)

    origin: str
    text: str
    label: Literal[0, 1]


class PairMeta(BaseModel):
    """A pair's meta: its id, its line, page, collection and volume, its page's source and split, its candidate's
    origin and CER to the GT text, and the renderer's version."""

    model_config = ConfigDict(frozen=True)

    id: str
    line: str
    page: str
    source: str
    collection: str
    volume: str
    split: Literal["train", "test"] | None
    candidate: str
    cer: float
    renderer: str


class PairSettings(BaseModel):
    """How a pair set is built: its side (training or test lines), wrong candidates a line, the height crops are
    scaled to (None: as cut), the seed and the renderer's font."""

    model_config = ConfigDict(frozen=True)

    side: Side = Side.TRAIN
    negatives: int = Field(default=1, ge=0)
    height: int | None = Field(default=48, gt=0)
    seed: int = 0
    font: Path


class LineSkipped(BaseModel):
    """A line left out of a pair set, and why."""

    model_config = ConfigDict(frozen=True)

    id: str
    error: str


class Built(BaseModel):
    """What building a pair set came to: its pairs by label and by origin, and the lines left out."""

    model_config = ConfigDict(frozen=True)

    positives: int
    negatives: int
    origins: dict[str, int]
    skipped: int


class PairOptions(BaseModel):
    """``<out stem>.options.json`` of a pair set: the line set it was built from and that set's own options, the reads
    used, the settings, the renderer's version and what it came to."""

    model_config = ConfigDict(frozen=True)

    lines: str
    lines_options: JsonValue
    reads: list[str]
    settings: PairSettings
    renderer: str
    built: Built


class _Line(BaseModel):
    """A line to pair: its crop, GT text and meta."""

    model_config = ConfigDict(frozen=True)

    image: Path
    text: str
    meta: SampleMeta

    @property
    def page(self) -> str:
        """The line's page id: its sample id less its element's and its own place (``<page id>_<k>_<j>``)."""
        return self.meta.id.rsplit("_", 2)[0]


def normalised(text: str) -> str:
    """``text`` in NFC with its whitespace collapsed to single spaces and stripped: what a read must equal to be the
    GT."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def edit_distance(a: str, b: str) -> int:
    """The Levenshtein distance between ``a`` and ``b``, in characters."""
    previous = list(range(len(b) + 1))
    for i, letter in enumerate(a, start=1):
        current = [i]
        for j, other in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (letter != other)))
        previous = current
    return previous[-1]


def cer(candidate: str, truth: str) -> float:
    """The candidate's character error rate against ``truth``, rounded to 4 places."""
    return round(edit_distance(candidate, truth) / max(len(truth), 1), 4)


def read_lines(path: Path, side: Side) -> list[_Line]:
    """A line set's lines of ``side``: each sample's crop and answer with its meta.

    Raises:
        SetRefused: If the set or its meta is missing or unreadable, they differ in length, or a line is of the other
            side (a real line of an unsplit set is of neither).
    """
    beside = meta_path(path)
    try:
        samples, metas = read_jsonl(path, Sample), read_meta(beside)
    except (OSError, ValueError) as error:
        raise SetRefused(f"{path} and its meta {beside.name} cannot be read: {error}") from error
    if len(samples) != len(metas):
        raise SetRefused(f"{beside} has {len(metas)} lines for {len(samples)} samples in {path.name}")
    lines: list[_Line] = []
    for sample, meta in zip(samples, metas, strict=True):
        if not meta.is_of(side):
            raise SetRefused(
                f"{path} holds {meta.id}, a real line of the {meta.split or 'unsplit'} side: a {side} pair set takes "
                f"{side} lines only (build line sets with --split-from-gt-test-lists)"
            )
        lines.append(_Line(image=Path(sample.images[0]), text=sample.messages[-1].content, meta=meta))
    return lines


def read_reads(paths: list[Path], known: set[str]) -> dict[str, list[Read]]:
    """The reads of ``paths`` by line id.

    Raises:
        SetRefused: If a file cannot be read or a read names a line the set does not hold.
    """
    reads: dict[str, list[Read]] = defaultdict(list)
    for path in paths:
        try:
            rows = read_jsonl(path, Read)
        except (OSError, ValueError) as error:
            raise SetRefused(f"{path} is not a file of reads: {error}") from error
        for row in rows:
            if row.id not in known:
                raise SetRefused(f"{path} reads {row.id}, a line the set does not hold")
            reads[row.id].append(row)
    return reads


def _swap(line: _Line, page_lines: list[_Line]) -> str | None:
    """The text of the page's other line closest to ``line``'s in length (the first such), differing from it."""
    others = [other for other in page_lines if normalised(other.text) != normalised(line.text)]
    if not others:
        return None
    return min(others, key=lambda other: abs(len(other.text) - len(line.text))).text


def candidates(line: _Line, page_lines: list[_Line], *, reads: list[Read], settings: PairSettings) -> list[Candidate]:
    """A line's candidates: its GT text; ``negatives`` wrong ones drawn (by the line's seed) from every rule's
    perturbation and the swapped line; and each read, labelled 1 when it is the GT after normalisation. Each text
    once, the first of its origins kept."""
    rng = random.Random(f"{settings.seed}:{line.meta.id}")
    wrong = [
        Candidate(origin=f"perturb:{kind}", text=text, label=0)
        for kind, text in perturbed(line.text, len(Perturbation), rng)
    ]
    swapped = _swap(line, page_lines)
    if swapped is not None:
        wrong.append(Candidate(origin="swap", text=swapped, label=0))
    rng.shuffle(wrong)
    made = [Candidate(origin="gt", text=line.text, label=1), *wrong[: settings.negatives]]
    made += [
        Candidate(
            origin=f"reader:{read.reader}",
            text=read.text,
            label=1 if normalised(read.text) == normalised(line.text) else 0,
        )
        for read in reads
    ]
    seen: set[str] = set()
    kept: list[Candidate] = []
    for candidate in made:
        if candidate.text not in seen:
            seen.add(candidate.text)
            kept.append(candidate)
    return kept


def _crop(line: _Line, height: int | None, crops: Path) -> Path:
    """The line's crop scaled to ``height`` (its aspect kept; None: as cut), saved as ``<line id>.jpg``."""
    with Image.open(line.image) as source:
        image = source.convert("RGB")
    if height is not None and image.height != height:
        image = image.resize((max(1, round(image.width * height / image.height)), height), Image.Resampling.LANCZOS)
    path = crops / f"{line.meta.id}.jpg"
    image.save(path, quality=JPEG_QUALITY)
    return path


def _line_pairs(
    line: _Line, crop: Path, *, made: list[Candidate], renders: Path, font: Path, version: str
) -> list[tuple[Sample, PairMeta]]:
    """Each of a line's candidates ``made`` paired with its crop: rendered at the crop's size in ``font`` into
    ``renders`` (``<line id>_<n>.png``), as its sample and its meta (``version`` the renderer's)."""
    with Image.open(crop) as image:
        size = image.size
    pairs: list[tuple[Sample, PairMeta]] = []
    for n, candidate in enumerate(made):
        render = renders / f"{line.meta.id}_{n}.png"
        render_text(candidate.text, font, size).save(render)
        sample = Sample(
            messages=[Turn(role="user", content=sample_prompt(candidate.text))],
            images=[str(crop), str(render)],
            label=candidate.label,
        )
        meta = PairMeta(
            id=f"{line.meta.id}_{n}",
            line=line.meta.id,
            page=line.page,
            source=line.meta.source,
            collection=line.meta.collection,
            volume=line.meta.volume,
            split=line.meta.split,
            candidate=candidate.origin,
            cer=cer(candidate.text, line.text),
            renderer=version,
        )
        pairs.append((sample, meta))
    return pairs


def build_pairs(lines_set: Path, out: Path, *, settings: PairSettings, reads: list[Path]) -> Built:
    """Jev's pairs for every line of ``lines_set`` written to ``out`` and ``<out stem>.meta.jsonl`` (crops in
    ``<out stem>_crops/``, renders in ``<out stem>_renders/``), lines left out in ``<out stem>.skipped.jsonl``, how it
    was built in ``<out stem>.options.json``.

    Raises:
        SetRefused: If the line set or the reads cannot be used (:func:`read_lines`, :func:`read_reads`).
    """
    lines = read_lines(lines_set, settings.side)
    by_line = read_reads(reads, {line.meta.id for line in lines})
    by_page: dict[str, list[_Line]] = defaultdict(list)
    for line in lines:
        by_page[line.page].append(line)
    version = renderer_version(settings.font)
    crops, renders = out.with_name(f"{out.stem}_crops").resolve(), out.with_name(f"{out.stem}_renders").resolve()
    crops.mkdir(parents=True, exist_ok=True)
    renders.mkdir(parents=True, exist_ok=True)
    labels: Counter[int] = Counter()
    origins: Counter[str] = Counter()
    skipped = 0
    with ExitStack() as stack:
        pairs, metas, left_out = (
            stack.enter_context(path.open("w", encoding="utf-8")) for path in (out, meta_path(out), skipped_path(out))
        )
        for line in lines:
            try:
                crop = _crop(line, settings.height, crops)
            except OSError as error:
                left_out.write(LineSkipped(id=line.meta.id, error=f"{type(error).__name__}: {error}").model_dump_json())
                left_out.write("\n")
                skipped += 1
                continue
            made = candidates(line, by_page[line.page], reads=by_line.get(line.meta.id, []), settings=settings)
            for sample, meta in _line_pairs(
                line, crop, made=made, renders=renders, font=settings.font, version=version
            ):
                pairs.write(sample.model_dump_json() + "\n")
                metas.write(meta.model_dump_json() + "\n")
            labels.update(candidate.label for candidate in made)
            origins.update(candidate.origin.split(":")[0] for candidate in made)
    built = Built(positives=labels[1], negatives=labels[0], origins=dict(origins), skipped=skipped)
    record = PairOptions(
        lines=str(lines_set.resolve()),
        lines_options=read_options(lines_set),
        reads=[str(path.resolve()) for path in reads],
        settings=settings,
        renderer=version,
        built=built,
    )
    options_path(out).write_text(record.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return built
