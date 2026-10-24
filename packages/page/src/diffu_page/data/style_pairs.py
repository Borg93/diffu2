"""Reference pairs for the v3 LoRAs: draw a crop's text and layout in the hand of a reference crop.

Every v2 training crop becomes a target; its reference is another crop (``--reference``: from another page of the
same volume, or from the same page or its facing page, the same scribe as far as the pages tell), shrunk to the
reference size (``diffu_page.crops.grid.as_reference``). The caption is the target's v2 caption rewritten as a FLUX.3
image-edit record (``diffu_page.captions.restyle``): it takes the hand, ink and paper from ``<ref_image_0>``.
``--caption described`` also describes the reference's measured hand, ink and paper in words
(``described_edit_caption``); the default names none of them, so the hand can only come from the reference.
Writes, under ``--out`` (a new or empty dir):

- ``train.jsonl``: the manifest NeMo's FLUX.3 image-edit encoder caches (``diffu_train.cache_pairs``), its media under
  ``targets/`` (hard links to the v2 crops: the encoder only reads files inside the manifest's dir) and ``refs/``;
- ``val_eval/``: the 24 held-out v2 crops, each with a reference picked as the training ones are (a
  ``diffu_page.cli.evaluate`` dir);
- ``by_example/``: the steering test's templates drawn with each archive's example crop as the reference, the
  v3 counterpart of ``diffu_page.data.steer_set`` (a ``diffu_page.cli.evaluate`` dir that ``diffu_page.cli.sheets`` lays
  out).

Every eval row carries its reference's line boxes, so ``diffu_page.cli.evaluate`` can measure how close a drawing's
hand comes to its reference's.

With ``--named-fraction`` (v3b), that share of the training captions also names the target's archive, years, scan,
paper and ink (``named_edit_caption``), so the model gets the style in words and by example; the rest stay unnamed so
the reference stays necessary. The held-out and by-example captions then all name the template's own archive: a
by-example crop whose hand follows its reference shows the picture wins over the words.

With ``--val-pairs N``, ``val/val.jsonl`` holds N pairs from the val split's pages, captioned as the training pairs
are: the held-out set NeMo scores ``val_loss`` on during training.

The references are picked first (training, then held-out, then by-example, from one seeded stream), then measured
(``--caption described`` only, in parallel), then every set is captioned, and only then is any file written, so a
caption that fails leaves no half-written set; the named captions draw from a stream of their own. So the references do
not depend on the caption options. ``stale.json`` names, per set, the crops left out because their caption predates the
caption grammar, and ``build.json`` how the set was built (``BuildRecord``).

    python -m diffu_page.data.style_pairs --crops data_flux3/layout_crops_v2b --out data_flux3/style_pairs_v3
    python -m diffu_page.data.style_pairs --named-fraction 0.5 --val-pairs 64 \
        --crops data_flux3/layout_crops_v2b --out data_flux3/style_pairs_v3b
    python -m diffu_page.data.style_pairs --reference page --caption described --val-pairs 64 \\
        --crops data_flux3/layout_crops_v4 --out data_flux3/style_pairs_v3d
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image
from pydantic import BaseModel

from diffu_page.captions.hand import ReferenceLook, measure_reference
from diffu_page.captions.provenance import collection_key
from diffu_page.captions.records import CropRecord, read_records
from diffu_page.captions.restyle import (
    CaptionParts,
    ReferenceCaption,
    StaleCaptionError,
    named_edit_caption,
    reference_caption,
)
from diffu_page.crops.grid import as_reference
from diffu_page.data.steer_set import ARCHIVES, DEFAULT_TEMPLATES, SEED, archive_style
from diffu_page.evalset import ReferencedCrop, write_jsonl
from diffu_page.output import JPEG_QUALITY, console

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

type Volume = tuple[str, str]  # (collection dir, volume dir)
type Picker = Callable[[Scan, dict[Volume, list[Scan]], random.Random], Scan | None]


class Scan(BaseModel):
    """A crop record and where its scan is."""

    record: CropRecord
    path: Path


class Pair(BaseModel):
    """A crop to draw and the reference crop to draw it in the hand of."""

    target: Scan
    reference: Scan


type CaptionFor = Callable[[Pair], str]


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
    """The manifest's pairs, the targets left out because their caption predates the caption grammar, and the files
    the manifest needs."""

    rows: list[ManifestRow]
    stale: list[str]
    placements: list[Placement]


class EvalSet(BaseModel):
    """An eval dir's rows, the crops left out because their caption predates the caption grammar, and the files the
    rows need."""

    rows: list[ReferencedCrop]
    stale: list[str]
    placements: list[Placement]


class EvalPick(BaseModel):
    """A pair to evaluate, and how its row is seeded and labelled (by-example rows; held-out rows take none)."""

    pair: Pair
    seed: int | None = None
    template: int | None = None
    variant: str | None = None


class ReferenceSource(StrEnum):
    """Where a pair's reference comes from: another page of the volume (v3), or the same page or its facing page."""

    VOLUME = "volume"
    PAGE = "page"


class BuildRecord(BaseModel):
    """``build.json``: what a set of pairs was built from and how. The caption form must be the one generation is told
    the LoRA was trained on (``--reference-caption``)."""

    crops: str
    caption: ReferenceCaption
    reference: ReferenceSource
    named_fraction: float
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


def pick_reference(scan: Scan, volumes: dict[Volume, list[Scan]], rng: random.Random) -> Scan | None:
    """A crop from another page of ``scan``'s volume, or None when its volume has no other page."""
    others = [other for other in volumes[volume_of(scan.record)] if other.record.page != scan.record.page]
    return rng.choice(others) if others else None


def page_number(record: CropRecord) -> int | None:
    """The scan number a page file ends in (``.../40004026_00128.xml`` -> 128), or None."""
    match = re.search(r"(\d+)$", Path(record.page).stem)
    return int(match.group(1)) if match else None


def pick_same_hand(scan: Scan, volumes: dict[Volume, list[Scan]], rng: random.Random) -> Scan | None:
    """A crop of ``scan``'s own page, else of the facing page (scan number one off), that shares none of its lines;
    else None. Another page of the volume may be another scribe or another kind of document (v3's pairs); the same
    page or its facing page almost never is. Consecutive scans can overlap (one sheet photographed in two parts),
    so a facing crop is checked for shared lines too: a reference holding the target's own lines teaches copying."""
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


PICKERS: dict[ReferenceSource, Picker] = {ReferenceSource.VOLUME: pick_reference, ReferenceSource.PAGE: pick_same_hand}


def pick_pairs(
    targets: list[Scan], volumes: dict[Volume, list[Scan]], *, rng: random.Random, pick: Picker
) -> list[Pair]:
    """Each target with the reference ``pick`` finds for it among ``volumes``; a target without one is left out."""
    pairs: list[Pair] = []
    for target in targets:
        reference = pick(target, volumes, rng)
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


def eval_row(pick: EvalPick, caption: CaptionFor) -> ReferencedCrop:
    """An eval crop of the pick's target drawn in its reference's hand (its files: ``eval_placements``)."""
    target, reference = pick.pair.target, pick.pair.reference
    return ReferencedCrop(
        file_name=target.record.file_name,
        caption=caption(pick.pair),
        lines=target.record.lines,
        width=target.record.width,
        height=target.record.height,
        reference=f"refs/{reference.path.name}",
        reference_lines=[line.bbox for line in reference.record.lines],
        seed=pick.seed,
        template=pick.template,
        variant=pick.variant,
        example=f"refs/{reference.path.name}" if pick.variant else None,
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


def training_pairs(pairs: list[Pair], caption: CaptionFor) -> TrainingSet:
    """The manifest of ``pairs`` and the files it needs (targets linked, references shrunk); a pair whose target's
    caption predates the caption grammar is left out."""
    manifest: list[ManifestRow] = []
    stale: list[str] = []
    placements: list[Placement] = []
    for pair in pairs:
        target, reference = pair.target, pair.reference
        try:
            prompt = caption(pair)
        except StaleCaptionError:
            stale.append(target.record.file_name)
            continue
        placements += [
            Placement(source=target.path, destination=Path("targets") / target.record.file_name, shrunk=False),
            Placement(source=reference.path, destination=Path("refs") / reference.record.file_name, shrunk=True),
        ]
        manifest.append(
            ManifestRow(
                id=Path(target.record.file_name).stem,
                prompt=prompt,
                media=[
                    Media(role="target", file_name=f"targets/{target.record.file_name}"),
                    Media(role="context", file_name=f"refs/{reference.record.file_name}"),
                ],
                metadata=PairMetadata(row_index=len(manifest), reference=reference.record.file_name),
            )
        )
    return TrainingSet(rows=manifest, stale=stale, placements=placements)


def eval_set(picks: list[EvalPick], caption: CaptionFor) -> EvalSet:
    """Each pick as an eval crop, with the files the rows need; a crop whose caption predates the caption grammar is
    left out, as in ``training_pairs``."""
    rows: list[ReferencedCrop] = []
    stale: list[str] = []
    placements: list[Placement] = []
    for pick in picks:
        try:
            rows.append(eval_row(pick, caption))
        except StaleCaptionError:
            stale.append(pick.pair.target.record.file_name)
            continue
        placements += eval_placements(pick.pair)
    return EvalSet(rows=rows, stale=stale, placements=placements)


class ExamplePair(BaseModel):
    """A by-example crop: a steering template, the reference to draw it from, and its label."""

    pair: Pair
    template: int
    label: str


def pick_examples(
    held_out: list[Scan],
    templates: list[int],
    examples: dict[str, Scan],
    *,
    volumes: dict[Volume, list[Scan]],
    rng: random.Random,
    pick: Picker,
) -> list[ExamplePair]:
    """The steering templates by example: in their own volume's hand, then in each steering archive's."""
    picked: list[ExamplePair] = []
    for t in templates:
        template = held_out[t]
        own = pick(template, volumes, rng)
        references = [(f"own volume ({collection_key(template.record.collection)})", own)] if own else []
        references += list(examples.items())
        picked += [
            ExamplePair(pair=Pair(target=template, reference=reference), template=t, label=label)
            for label, reference in references
        ]
    return picked


def by_example_picks(picked: list[ExamplePair]) -> list[EvalPick]:
    """The by-example crops as eval picks, each seeded by its template as the v2 steering set is."""
    return [
        EvalPick(
            pair=example.pair,
            seed=SEED + example.template,
            template=example.template,
            variant=f"reference: {example.label}",
        )
        for example in picked
    ]


def write_set(directory: Path, placements: list[Placement], manifest: str, rows: Sequence[BaseModel]) -> None:
    """A set's files placed in ``directory`` and its rows written to ``directory / manifest``."""
    for placement in placements:
        placement.place(directory)
    write_jsonl(directory / manifest, rows)


def captions(
    form: ReferenceCaption, *, named_fraction: float, seed: int, looks: dict[Path, ReferenceLook]
) -> tuple[CaptionFor, CaptionFor]:
    """The training and the eval caption of a pair in ``form`` (``reference_caption``, as a planned crop drawn from a
    reference is captioned), the described form from the reference's ``looks``; with ``named_fraction`` (plain form
    only) that share of the training captions, drawn from their own stream, and every eval caption name the archive
    and look."""

    def drawn_from_reference(pair: Pair) -> str:
        return reference_caption(CaptionParts.of(pair.target.record), form, looks.get(pair.reference.path))

    if not named_fraction:
        return drawn_from_reference, drawn_from_reference
    words = random.Random(seed + 1)  # its own stream, so the references are the ones v3 picked

    def named(pair: Pair) -> str:
        return named_edit_caption(CaptionParts.of(pair.target.record))

    def training(pair: Pair) -> str:
        return named(pair) if words.random() < named_fraction else drawn_from_reference(pair)

    return training, named


@app.command()
def main(
    crops: Annotated[Path, typer.Option(help="the v2 crops (diffu_page.data.build_crops output)")],
    out: Annotated[Path, typer.Option(help="where the v3 pairs go: a new or empty dir")],
    templates: Annotated[
        list[int] | None, typer.Option("--template", help="a val_eval row for by_example/ (repeat it)")
    ] = None,
    named_fraction: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="share of training captions that also name the archive and look")
    ] = 0.0,
    val_pairs: Annotated[int, typer.Option(min=0, help="pairs from the val split for NeMo's val_loss")] = 0,
    reference: Annotated[
        ReferenceSource, typer.Option(help="volume: another page of the volume; page: the same or the facing page")
    ] = ReferenceSource.VOLUME,
    caption: Annotated[
        ReferenceCaption, typer.Option(help="plain: the reference only by <ref_image_0>; described: its look in words")
    ] = ReferenceCaption.PLAIN,
    workers: Annotated[
        int, typer.Option(min=1, help="processes measuring the references (described): fewer beside a training")
    ] = max(1, (os.cpu_count() or 2) // 2),
    seed: int = 0,
) -> None:
    """Build the v3 reference pairs, the held-out set and the by-example steering set."""
    if out.exists() and any(out.iterdir()):
        raise typer.BadParameter(
            f"{out} is not empty: an earlier build's files would stay beside this one's", param_hint="--out"
        )
    if named_fraction and caption is ReferenceCaption.DESCRIBED:
        raise typer.BadParameter("--named-fraction names the archive in the plain captions; leave it 0 with described")
    train, val, held_out = (load_split(crops, split) for split in ("train", "val", "val_eval"))
    rng = random.Random(seed)
    pick = PICKERS[reference]
    train_pairs = pick_pairs(train, by_volume(train), rng=rng, pick=pick)
    val_set = pick_pairs(val, by_volume(val), rng=random.Random(seed + 2), pick=pick) if val_pairs else []
    volumes = by_volume(val + train)
    held_pairs = pick_pairs(held_out, volumes, rng=rng, pick=pick)
    train_by_name = {scan.record.file_name: scan for scan in train}
    examples = {
        key: train_by_name[archive_style([s.record for s in train], key, crops).example.name] for key in ARCHIVES
    }
    example_pairs = pick_examples(
        held_out, templates or list(DEFAULT_TEMPLATES), examples, volumes=volumes, rng=rng, pick=pick
    )

    every_pair = [*train_pairs, *val_set, *held_pairs, *(example.pair for example in example_pairs)]
    described = caption is ReferenceCaption.DESCRIBED
    looks = describe_all((pair.reference for pair in every_pair), workers) if described else {}
    training_caption, eval_caption = captions(caption, named_fraction=named_fraction, seed=seed, looks=looks)

    # every set is captioned, in this order (the named captions' stream), before any file is written
    pairs = training_pairs(train_pairs, training_caption)
    validation = training_pairs(val_set, training_caption)
    held = eval_set([EvalPick(pair=pair) for pair in held_pairs], eval_caption)
    examples = eval_set(by_example_picks(example_pairs), eval_caption)

    write_set(out, pairs.placements, "train.jsonl", pairs.rows)
    if val_pairs:
        spread = validation.rows[:: max(1, len(validation.rows) // val_pairs)][:val_pairs]  # sorted by collection
        write_set(out / "val", validation.placements, "val.jsonl", [renumbered(row, k) for k, row in enumerate(spread)])
    write_set(out / "val_eval", held.placements, "metadata.jsonl", held.rows)
    write_set(out / "by_example", examples.placements, "metadata.jsonl", examples.rows)
    stale = {"train": pairs.stale, "val": validation.stale, "val_eval": held.stale, "by_example": examples.stale}
    (out / "stale.json").write_text(json.dumps(stale, indent=1), encoding="utf-8")
    build = BuildRecord(
        crops=str(crops),
        caption=caption,
        reference=reference,
        named_fraction=named_fraction,
        val_pairs=val_pairs,
        seed=seed,
    )
    (out / "build.json").write_text(build.model_dump_json(indent=1), encoding="utf-8")
    alone = len(train) - len(train_pairs)
    console.print(
        f"train: {len(pairs.rows)} pairs -> {out} ({alone} crops skipped: no reference to pick ({reference.value}); "
        f"{len(pairs.stale)} skipped: caption older than the caption grammar, see stale.json)"
    )
    console.print(
        f"val_eval: {len(held.rows)} crops ({len(held.stale)} stale), "
        f"by_example: {len(examples.rows)} prompts ({len(examples.stale)} stale)"
    )


if __name__ == "__main__":
    app()
