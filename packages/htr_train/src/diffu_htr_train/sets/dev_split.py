"""A dev split cut from a training set by whole volumes (ms-swift's own split is by sample, which puts lines of one page
on both sides): a seeded share of the set's volumes to ``<stem>.dev.jsonl``, the rest to ``<stem>.train.jsonl``, each
with its meta beside it, in the set's order, and its options (the set's own and the split's)."""

from __future__ import annotations

import random
from contextlib import ExitStack
from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue

from diffu_htr_train.records import SampleMeta, meta_path, options_path, read_meta
from diffu_htr_train.sets import SetRefused, read_options

MIN_VOLUMES = 3  # fewer, and a dev split of whole volumes leaves too little on one side


class DevSplit(BaseModel):
    """A dev split's files and what went where: the volumes held out, and the samples on each side."""

    model_config = ConfigDict(frozen=True)

    train: Path
    dev: Path
    dev_volumes: list[str]
    train_samples: int
    dev_samples: int


class DevOptions(BaseModel):
    """Each side's ``<side stem>.options.json``: the set it was cut from and that set's own options (None where it
    records none), the share and seed its volumes were drawn by (None when they were another split's), and the
    split."""

    model_config = ConfigDict(frozen=True)

    cut_from: str
    options: JsonValue
    share: float | None
    seed: int | None
    split: DevSplit


type Row = tuple[str, str, SampleMeta]  # a sample's line, its meta's line, and that meta read


def _volume(meta: SampleMeta) -> str:
    """A sample's volume as the split names it: ``<collection>/<volume>``."""
    return f"{meta.collection}/{meta.volume}"


def volumes_of(path: Path) -> set[str]:
    """The volumes (``<collection>/<volume>``) a set's meta names.

    Raises:
        SetRefused: If its meta cannot be read.
    """
    try:
        return {_volume(meta) for meta in read_meta(meta_path(path))}
    except (OSError, ValueError) as error:
        raise SetRefused(f"the meta of {path} cannot be read: {error}") from error


def _read_rows(path: Path) -> list[Row]:
    """A set's samples with their meta, each checked to name a volume and to be of the GT's training side.

    Raises:
        SetRefused: If the set or its meta is missing or mismatched, a sample has no volume, or a real sample is not of
            the GT's training side.
    """
    beside = meta_path(path)
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        meta_lines = [line for line in beside.read_text(encoding="utf-8").splitlines() if line.strip()]
        metas = read_meta(beside)
    except (OSError, ValueError) as error:
        raise SetRefused(f"{path} and its meta {beside.name} cannot be read: {error}") from error
    if len(lines) != len(metas):
        raise SetRefused(f"{beside} has {len(metas)} lines for {len(lines)} samples in {path.name}")
    for meta in metas:
        if not meta.volume:
            raise SetRefused(f"{path}: {meta.id} names no volume, so the set cannot be split by volume")
        if not meta.is_of("train"):
            raise SetRefused(
                f"{path}: {meta.id} is a real sample of the {meta.split or 'unsplit'} side; a dev split is cut from "
                "the GT's training samples (--split-from-gt-test-lists)"
            )
    return list(zip(lines, meta_lines, metas, strict=True))


def _held_volumes(path: Path, volumes: list[str], *, share: float, seed: int, like: set[str] | None) -> set[str]:
    """The volumes of ``volumes`` (sorted) held out: ``max(1, round(share x volumes))`` drawn by ``seed``, or those of
    ``like``.

    Raises:
        SetRefused: If there are fewer than :data:`MIN_VOLUMES`, or none is of ``like``.
    """
    if len(volumes) < MIN_VOLUMES:
        raise SetRefused(f"{path} has {len(volumes)} volumes: a dev split by volume needs at least {MIN_VOLUMES}")
    if like is None:
        return set(random.Random(seed).sample(volumes, max(1, round(share * len(volumes)))))
    held = like & set(volumes)
    if not held:
        raise SetRefused(f"{path} shares no volume with the dev split it is to be cut like")
    return held


def _write_sides(path: Path, rows: list[Row], held: set[str]) -> DevSplit:
    """Each row written, in the set's order, to the dev side's files where its volume is ``held``, else the training
    side's (``<stem>.<side>.jsonl`` and ``<stem>.<side>.meta.jsonl``): what went where."""
    stem = path.with_name(path.stem)
    files = {
        side: (stem.with_name(f"{stem.name}.{side}.jsonl"), stem.with_name(f"{stem.name}.{side}.meta.jsonl"))
        for side in ("train", "dev")
    }
    counts = {"train": 0, "dev": 0}
    with ExitStack() as stack:
        handles = {
            side: [stack.enter_context(file.open("w", encoding="utf-8")) for file in pair]
            for side, pair in files.items()
        }
        for line, meta_line, meta in rows:
            side = "dev" if _volume(meta) in held else "train"
            set_file, meta_file = handles[side]
            set_file.write(line + "\n")
            meta_file.write(meta_line + "\n")
            counts[side] += 1
    return DevSplit(
        train=files["train"][0],
        dev=files["dev"][0],
        dev_volumes=sorted(held),
        train_samples=counts["train"],
        dev_samples=counts["dev"],
    )


def split_dev(path: Path, *, share: float, seed: int, like: set[str] | None = None) -> DevSplit:
    """``path``'s samples split by volume: ``max(1, round(share x volumes))`` volumes, drawn by ``seed``, to the dev
    side; or, given ``like`` (another dev split's volumes, :func:`volumes_of`), those of its volumes, so the sets one
    run mixes hold out the same volumes.

    Raises:
        SetRefused: If the set or its meta is missing or mismatched, a sample has no volume, a real sample is not of
            the GT's training side, the set has fewer than :data:`MIN_VOLUMES` volumes, or it shares no volume with
            ``like``.
    """
    rows = _read_rows(path)
    volumes = sorted({_volume(meta) for _, _, meta in rows})
    held = _held_volumes(path, volumes, share=share, seed=seed, like=like)
    split = _write_sides(path, rows, held)
    record = DevOptions(
        cut_from=str(path.resolve()),
        options=read_options(path),
        share=None if like is not None else share,
        seed=None if like is not None else seed,
        split=split,
    )
    for side in (split.train, split.dev):
        options_path(side).write_text(record.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return split
