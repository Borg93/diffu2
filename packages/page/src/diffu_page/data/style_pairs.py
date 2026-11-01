"""Style pairs: a crop's text and layout drawn in the hand of a reference crop of the same hand, for FLUX.3 Klein's
style-reference LoRA (``configs/flux3/flux3_style_ref_lora_v3d.yaml``) and MOSS-Image's reference training
(``diffu_moss.train``).

Every training crop of ``diffu_page.data.build_crops`` becomes a target; its reference is a crop of its own page sharing
none of its lines, else of its facing page (the same scribe as far as the pages tell), shrunk to the reference size
(``diffu_page.crops.grid.as_reference``). The caption is the target's caption rewritten as a FLUX.3 image-edit record
that takes the hand, ink and paper from ``<ref_image_0>`` and describes the reference's measured hand, ink and paper
in words (``diffu_page.captions.restyle``'s ``described`` form, as a crop drawn from a reference is captioned;
MOSS-Image's training drops the words where a prompt form leaves the look to the image, ``diffu_moss.sample_caption``).
Writes, under ``--out`` (a new or empty dir):

- ``train.jsonl``: the manifest NeMo's FLUX.3 image-edit encoder caches (``diffu_train.cache_pairs``) and
  ``diffu-moss-cache-pairs`` reads, its media under ``targets/`` (hard links to the crops: the encoder only reads files
  inside the manifest's dir) and ``refs/``;
- ``val_eval/``: the held-out crops (the crops' ``val_eval`` split, ``diffu_page.data.pick_eval``), each with a
  reference picked as the training ones are: a ``diffu_page.cli.evaluate`` dir whose rows carry their reference's line
  boxes, so evaluate can measure how close a drawing's hand comes to its reference's;
- with ``--val-pairs N``, ``val/val.jsonl``: N pairs from the val split's pages, captioned as the training pairs are,
  the held-out set NeMo scores ``val_loss`` on during training;
- ``build.json``, last: how the set was built (``BuildRecord``).

The references are picked first (training, then held-out, from one seeded stream; the val pairs from their own), then
measured (in parallel), then every set is captioned, and only then is any file written, so a caption that fails leaves
no half-written set. A crop whose record keeps no caption template (built before records kept one) stops the build
before any file is written.

    python -m diffu_page.data.style_pairs --val-pairs 64 --crops data_flux3/layout_crops_v2b \\
        --out data_flux3/style_pairs_v3d
"""

from __future__ import annotations

import os
import random
import re
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image
from pydantic import BaseModel

from diffu_page.captions.hand import ReferenceLook, measure_reference
from diffu_page.captions.records import CropRecord, read_records
from diffu_page.captions.restyle import CaptionParts, ReferenceCaption, reference_caption
from diffu_page.crops.grid import as_reference
from diffu_page.evalset import ReferencedCrop, write_jsonl
from diffu_page.output import JPEG_QUALITY, console

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

type Volume = tuple[str, str]  # (collection dir, volume dir)


class Scan(BaseModel):
    """A crop record and where its scan is."""

    record: CropRecord
    path: Path


class Pair(BaseModel):
    """A crop to draw and the reference crop to draw it in the hand of."""

    target: Scan
    reference: Scan


class Media(BaseModel):
    """One image of a manifest row: the target to draw or the context (reference) to draw it from."""

    role: str  # "target" or "context"
    file_name: str  # relative to the manifest's dir


class PairMetadata(BaseModel):
    """What NeMo's manifest carries of a pair besides its media: the row index it requires, and the reference."""

    row_index: int
    reference: str


class ManifestRow(BaseModel):
    """One row of NeMo's image-edit manifest (``nemo_automodel.components.models.flux3.preprocessing``)."""

    id: str
    prompt: str
    media: list[Media]
    metadata: PairMetadata


class Placement(BaseModel):
    """A file a set's dir needs: a target hard-linked there, or a reference shrunk to the reference size."""

    source: Path
    destination: Path  # relative to the set's dir
    shrunk: bool

    def place(self, directory: Path) -> None:
        """Write the file into ``directory``."""
        destination = directory / self.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        (shrink if self.shrunk else link)(self.source, destination)


class TrainingSet(BaseModel):
    """The manifest's pairs and the files it needs."""

    rows: list[ManifestRow]
    placements: list[Placement]

    def spread(self, count: int) -> TrainingSet:
        """``count`` pairs spread over the set (in its order, sorted by collection), numbered from 0, with only the
        files they name."""
        kept = [renumbered(row, k) for k, row in enumerate(self.rows[:: max(1, len(self.rows) // count)][:count])]
        named = {media.file_name for row in kept for media in row.media}
        return TrainingSet(rows=kept, placements=[p for p in self.placements if p.destination.as_posix() in named])


class EvalSet(BaseModel):
    """An eval dir's rows and the files they need."""

    rows: list[ReferencedCrop]
    placements: list[Placement]


class BuildRecord(BaseModel):
    """``build.json``: what a set of pairs was built from and how."""

    crops: str
    val_pairs: int
    seed: int


def volume_of(record: CropRecord) -> Volume:
    """The volume of a crop, from its PAGE path ``<collection>/<volume>/<volume>/page/<file>.xml``."""
    collection, volume = record.page.split("/")[:2]
    return collection, volume


def link(source: Path, destination: Path) -> None:
    """Hard-link ``source`` to ``destination`` unless it is there (NeMo reads only files inside the manifest's dir)."""
    if not destination.exists():
        destination.hardlink_to(source)


def shrink(source: Path, destination: Path) -> None:
    """Save ``source`` at the reference size (``as_reference``), so training and generation read the same pixels."""
    if destination.exists():
        return
    with Image.open(source) as image:
        as_reference(image).save(destination, quality=JPEG_QUALITY)


def load_split(crops: Path, split: str) -> list[Scan]:
    """A split's crop records with where their scans are."""
    return [
        Scan(record=record, path=crops / split / record.file_name)
        for record in read_records(crops / split / "metadata.jsonl")
    ]


def by_volume(scans: list[Scan]) -> dict[Volume, list[Scan]]:
    """The scans grouped by their volume."""
    volumes: dict[Volume, list[Scan]] = defaultdict(list)
    for scan in scans:
        volumes[volume_of(scan.record)].append(scan)
    return volumes


def page_number(record: CropRecord) -> int | None:
    """The scan number a page file ends in (``.../40004026_00128.xml`` -> 128), or None."""
    match = re.search(r"(\d+)$", Path(record.page).stem)
    return int(match.group(1)) if match else None


def pick_same_hand(scan: Scan, volumes: dict[Volume, list[Scan]], rng: random.Random) -> Scan | None:
    """A crop of ``scan``'s own page, else of the facing page (scan number one off), that shares none of its lines;
    else None. Another page of the volume may be another scribe or another kind of document; the same page or its
    facing page almost never is. Consecutive scans can overlap (one sheet photographed in two parts), so a facing crop
    is checked for shared lines too: a reference holding the target's own lines teaches copying."""
    volume = volumes[volume_of(scan.record)]
    texts = {line.text for line in scan.record.lines}
    apart = [other for other in volume if not texts & {line.text for line in other.record.lines}]
    same_page = [other for other in apart if other.record.page == scan.record.page]
    if same_page:
        return rng.choice(same_page)
    number = page_number(scan.record)
    facing = [
        other
        for other in apart
        if number is not None and (n := page_number(other.record)) is not None and abs(n - number) == 1
    ]
    return rng.choice(facing) if facing else None


def pick_pairs(targets: list[Scan], volumes: dict[Volume, list[Scan]], *, rng: random.Random) -> list[Pair]:
    """Each target with the same-hand reference :func:`pick_same_hand` finds for it among ``volumes``; a target without
    one is left out."""
    pairs: list[Pair] = []
    for target in targets:
        reference = pick_same_hand(target, volumes, rng)
        if reference is not None:
            pairs.append(Pair(target=target, reference=reference))
    return pairs


def describe(scan: Scan) -> ReferenceLook:
    """What ``scan`` looks like as a reference (``measure_reference``), its hand inside its line boxes."""
    return measure_reference(scan.path, [line.bbox for line in scan.record.lines])


def describe_all(scans: Iterable[Scan], workers: int) -> dict[Path, ReferenceLook]:
    """Every distinct reference described (``describe``), in parallel on ``workers`` processes."""
    unique = list({scan.path: scan for scan in scans}.values())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        looks = list(pool.map(describe, unique, chunksize=16))
    return {scan.path: look for scan, look in zip(unique, looks, strict=True)}


def pair_caption(pair: Pair, looks: dict[Path, ReferenceLook]) -> str:
    """The pair's caption: its target's caption drawn from its reference, the reference's measured look described, by
    the call that captions a planned crop drawn from a reference (``reference_caption``, ``described``)."""
    parts = CaptionParts.of(pair.target.record)
    return reference_caption(parts, ReferenceCaption.DESCRIBED, looks[pair.reference.path])


def eval_row(pair: Pair, caption: str) -> ReferencedCrop:
    """An eval crop of the pair's target drawn in its reference's hand (its files: ``eval_placements``)."""
    target, reference = pair.target, pair.reference
    return ReferencedCrop(
        file_name=target.record.file_name,
        caption=caption,
        lines=target.record.lines,
        width=target.record.width,
        height=target.record.height,
        reference=f"refs/{reference.path.name}",
        reference_lines=[line.bbox for line in reference.record.lines],
    )


def eval_placements(pair: Pair) -> list[Placement]:
    """The files an eval row needs in its dir: the target scan linked, the reference shrunk into ``refs/``."""
    return [
        Placement(source=pair.target.path, destination=Path(pair.target.record.file_name), shrunk=False),
        Placement(source=pair.reference.path, destination=Path("refs") / pair.reference.path.name, shrunk=True),
    ]


def renumbered(row: ManifestRow, index: int) -> ManifestRow:
    """``row`` at ``index`` of a manifest (NeMo requires the row indices to count from 0)."""
    return row.model_copy(update={"metadata": row.metadata.model_copy(update={"row_index": index})})


def training_pairs(pairs: list[Pair], looks: dict[Path, ReferenceLook]) -> TrainingSet:
    """The manifest of ``pairs`` and the files it needs (targets linked, references shrunk)."""
    manifest: list[ManifestRow] = []
    placements: list[Placement] = []
    for pair in pairs:
        target, reference = pair.target, pair.reference
        placements += [
            Placement(source=target.path, destination=Path("targets") / target.record.file_name, shrunk=False),
            Placement(source=reference.path, destination=Path("refs") / reference.record.file_name, shrunk=True),
        ]
        manifest.append(
            ManifestRow(
                id=Path(target.record.file_name).stem,
                prompt=pair_caption(pair, looks),
                media=[
                    Media(role="target", file_name=f"targets/{target.record.file_name}"),
                    Media(role="context", file_name=f"refs/{reference.record.file_name}"),
                ],
                metadata=PairMetadata(row_index=len(manifest), reference=reference.record.file_name),
            )
        )
    return TrainingSet(rows=manifest, placements=placements)


def eval_set(pairs: list[Pair], looks: dict[Path, ReferenceLook]) -> EvalSet:
    """Each pair as an eval crop, with the files the rows need."""
    rows = [eval_row(pair, pair_caption(pair, looks)) for pair in pairs]
    placements = [placement for pair in pairs for placement in eval_placements(pair)]
    return EvalSet(rows=rows, placements=placements)


def write_set(directory: Path, built: TrainingSet | EvalSet, *, manifest: str) -> None:
    """A built set's files placed in ``directory`` and its rows written to ``directory / manifest``."""
    for placement in built.placements:
        placement.place(directory)
    write_jsonl(directory / manifest, built.rows)


@app.command()
def main(
    crops: Annotated[Path, typer.Option(help="the crops (diffu_page.data.build_crops output, with its val_eval)")],
    out: Annotated[Path, typer.Option(help="where the pairs go: a new or empty dir")],
    val_pairs: Annotated[int, typer.Option(min=0, help="pairs from the val split for NeMo's val_loss")] = 0,
    workers: Annotated[
        int, typer.Option(min=1, help="processes measuring the references: fewer beside a training")
    ] = max(1, (os.cpu_count() or 2) // 2),
    seed: int = 0,
) -> None:
    """Build the style pairs and the held-out set."""
    if out.exists() and any(out.iterdir()):
        raise typer.BadParameter(
            f"{out} is not empty: an earlier build's files would stay beside this one's", param_hint="--out"
        )
    train, val, held_out = (load_split(crops, split) for split in ("train", "val", "val_eval"))
    rng = random.Random(seed)
    train_pairs = pick_pairs(train, by_volume(train), rng=rng)
    val_set = pick_pairs(val, by_volume(val), rng=random.Random(seed + 2)) if val_pairs else []
    held_pairs = pick_pairs(held_out, by_volume(val + train), rng=rng)
    looks = describe_all((pair.reference for pair in [*train_pairs, *val_set, *held_pairs]), workers)

    # every set is captioned before any file is written
    pairs = training_pairs(train_pairs, looks)
    validation = training_pairs(val_set, looks)
    held = eval_set(held_pairs, looks)

    write_set(out, pairs, manifest="train.jsonl")
    if val_pairs:
        write_set(out / "val", validation.spread(val_pairs), manifest="val.jsonl")
    write_set(out / "val_eval", held, manifest="metadata.jsonl")
    build = BuildRecord(crops=str(crops), val_pairs=val_pairs, seed=seed)
    (out / "build.json").write_text(build.model_dump_json(indent=1), encoding="utf-8")
    console.print(f"train: {len(pairs.rows)} pairs -> {out} ({len(train) - len(train_pairs)} crops with no reference)")
    console.print(f"val_eval: {len(held.rows)} crops")


if __name__ == "__main__":
    app()
