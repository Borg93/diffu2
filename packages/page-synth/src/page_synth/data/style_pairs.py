"""Reference pairs for the v3 LoRA: draw a crop's text and layout in the hand of a reference crop.

Every v2 training crop becomes a target; its reference is a crop from another page of the same volume (the same
scribe, as far as a volume tells), shrunk to ``REF_PIXELS``. The caption is the target's v2 caption rewritten as a
FLUX.3 image-edit record (``page_synth.captions.restyle.edit_caption``): it names no archive, paper or ink, so the hand can only
come from the reference. Writes, under ``data_flux3/style_pairs_v3``:

- ``train.jsonl``: the manifest NeMo's FLUX.3 image-edit encoder caches (``page_synth.data.cache_pairs``), its media under
  ``targets/`` (hard links to the v2 crops: the encoder only reads files inside the manifest's dir) and ``refs/``;
- ``val_eval/``: the 24 held-out v2 crops, each with a reference from its own volume (an ``page_synth.cli.evaluate`` dir);
- ``by_example/``: the steering test's templates drawn with each archive's example crop as the reference, the
  v3 counterpart of ``page_synth.data.steer_set`` (an ``page_synth.cli.evaluate`` dir that ``page_synth.cli.sheets`` lays out).

    python -m page_synth.data.style_pairs
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel

from page_synth.captions.prompt import LineRecord
from page_synth.captions.provenance import collection_key
from page_synth.captions.records import CropRecord, read_records
from page_synth.captions.restyle import edit_caption
from page_synth.data.steer_set import ARCHIVES, SEED, archive_style

REF_PIXELS = 512 * 1024  # a reference shows the hand smaller than the ~1 MP target, at half its tokens

type Volume = tuple[str, str]  # (collection dir, volume dir)


class Scan(BaseModel):
    """A crop record and where its scan is."""

    record: CropRecord
    path: Path


class Media(BaseModel):
    role: str  # "target" or "context"
    file_name: str  # relative to the manifest's dir


class ManifestRow(BaseModel):
    """One row of NeMo's image-edit manifest (``nemo_automodel.components.models.flux3.preprocessing``)."""

    id: str
    prompt: str
    media: list[Media]
    metadata: dict[str, Any]


class EvalRow(BaseModel):
    """One crop of an ``page_synth.cli.evaluate`` dir with a reference image."""

    file_name: str
    caption: str
    lines: list[LineRecord]
    width: int
    height: int
    reference: str  # relative to the eval dir
    seed: int | None = None
    template: int | None = None
    variant: str | None = None
    example: str | None = None  # page_synth.cli.sheets shows it beside the generated crop


def volume_of(record: CropRecord) -> Volume:
    """The volume of a crop, from its PAGE path ``<collection>/<volume>/<volume>/page/<file>.xml``."""
    collection, volume = record.page.split("/")[:2]
    return collection, volume


def link(source: Path, destination: Path) -> None:
    if not destination.exists():
        os.link(source, destination)


def shrink(source: Path, destination: Path) -> None:
    """Save ``source`` scaled to at most ``REF_PIXELS``, keeping its aspect ratio."""
    if destination.exists():
        return
    image = Image.open(source).convert("RGB")
    scale = min(1.0, (REF_PIXELS / (image.width * image.height)) ** 0.5)
    size = (round(image.width * scale), round(image.height * scale))
    image.resize(size, Image.Resampling.LANCZOS).save(destination, quality=92)


def load_split(crops: Path, split: str) -> list[Scan]:
    return [
        Scan(record=record, path=crops / split / record.file_name)
        for record in read_records(crops / split / "metadata.jsonl")
    ]


def by_volume(scans: list[Scan]) -> dict[Volume, list[Scan]]:
    volumes: dict[Volume, list[Scan]] = defaultdict(list)
    for scan in scans:
        volumes[volume_of(scan.record)].append(scan)
    return volumes


def pick_reference(scan: Scan, volumes: dict[Volume, list[Scan]], rng: random.Random) -> Scan | None:
    """A crop from another page of ``scan``'s volume, or None when its volume has no other page."""
    others = [other for other in volumes[volume_of(scan.record)] if other.record.page != scan.record.page]
    return rng.choice(others) if others else None


def eval_row(
    target: Scan,
    reference: Path,
    out: Path,
    *,
    seed: int | None = None,
    template: int | None = None,
    variant: str | None = None,
) -> EvalRow:
    """An eval crop of ``target`` drawn in ``reference``'s hand; the target scan and the shrunk reference go to
    ``out``."""
    (out / "refs").mkdir(parents=True, exist_ok=True)
    link(target.path, out / target.record.file_name)
    shrink(reference, out / "refs" / reference.name)
    return EvalRow(
        file_name=target.record.file_name,
        caption=edit_caption(target.record),
        lines=target.record.lines,
        width=target.record.width,
        height=target.record.height,
        reference=f"refs/{reference.name}",
        seed=seed,
        template=template,
        variant=variant,
        example=f"refs/{reference.name}" if variant else None,
    )


def write_rows(path: Path, rows: Sequence[BaseModel]) -> None:
    path.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")


def training_pairs(train: list[Scan], out: Path, rng: random.Random) -> list[ManifestRow]:
    """Every training crop that has another page in its volume, with a reference crop from that page."""
    for sub in ("targets", "refs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    volumes = by_volume(train)
    manifest: list[ManifestRow] = []
    for scan in train:
        reference = pick_reference(scan, volumes, rng)
        if reference is None:
            continue
        link(scan.path, out / "targets" / scan.record.file_name)
        shrink(reference.path, out / "refs" / reference.record.file_name)
        manifest.append(
            ManifestRow(
                id=Path(scan.record.file_name).stem,
                prompt=edit_caption(scan.record),
                media=[
                    Media(role="target", file_name=f"targets/{scan.record.file_name}"),
                    Media(role="context", file_name=f"refs/{reference.record.file_name}"),
                ],
                metadata={"row_index": len(manifest), "reference": reference.record.file_name},
            )
        )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, default=Path("data_flux3/layout_crops_v2"))
    ap.add_argument("--out", type=Path, default=Path("data_flux3/style_pairs_v3"))
    ap.add_argument("--templates", type=int, nargs="+", default=[1, 17], help="val_eval rows for by_example/")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train, val, held_out = (load_split(args.crops, split) for split in ("train", "val", "val_eval"))
    rng = random.Random(args.seed)
    manifest = training_pairs(train, args.out, rng)
    write_rows(args.out / "train.jsonl", manifest)

    # held-out crops, each in the hand of another page of its volume (an unseen page when there is one)
    eval_volumes = by_volume(val + train)
    val_rows = [
        eval_row(scan, reference.path, args.out / "val_eval")
        for scan in held_out
        if (reference := pick_reference(scan, eval_volumes, rng)) is not None
    ]
    write_rows(args.out / "val_eval/metadata.jsonl", val_rows)

    # the steering templates, by example: in their own volume's hand, then in each steering archive's
    examples = {key: archive_style([s.record for s in train], key, args.crops).example for key in ARCHIVES}
    example_rows: list[EvalRow] = []
    for t in args.templates:
        template = held_out[t]
        references: list[tuple[str, Path]] = []
        own = pick_reference(template, eval_volumes, rng)
        if own is not None:
            references.append((f"own volume ({collection_key(template.record.collection)})", own.path))
        references += list(examples.items())
        example_rows += [
            eval_row(
                template,
                path,
                args.out / "by_example",
                seed=SEED + t,
                template=t,
                variant=f"reference: {label}",
            )
            for label, path in references
        ]
    write_rows(args.out / "by_example/metadata.jsonl", example_rows)

    skipped = len(train) - len(manifest)
    print(f"train: {len(manifest)} pairs ({skipped} crops skipped: no other page in their volume) -> {args.out}")
    print(f"val_eval: {len(val_rows)} crops, by_example: {len(example_rows)} prompts")


if __name__ == "__main__":
    main()
