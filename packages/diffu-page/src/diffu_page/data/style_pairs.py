"""Reference pairs for the v3 LoRA: draw a crop's text and layout in the hand of a reference crop.

Every v2 training crop becomes a target; its reference is a crop from another page of the same volume (the same
scribe, as far as a volume tells), shrunk to ``REF_PIXELS``. The caption is the target's v2 caption rewritten as a
FLUX.3 image-edit record (``diffu_page.captions.restyle.edit_caption``): it names no archive, paper or ink, so the hand can only
come from the reference. Writes, under ``data_flux3/style_pairs_v3``:

- ``train.jsonl``: the manifest NeMo's FLUX.3 image-edit encoder caches (``diffu_train.cache_pairs``), its media under
  ``targets/`` (hard links to the v2 crops: the encoder only reads files inside the manifest's dir) and ``refs/``;
- ``val_eval/``: the 24 held-out v2 crops, each with a reference from its own volume (an ``diffu_gen.cli.evaluate`` dir);
- ``by_example/``: the steering test's templates drawn with each archive's example crop as the reference, the
  v3 counterpart of ``diffu_page.data.steer_set`` (an ``diffu_gen.cli.evaluate`` dir that ``diffu_gen.cli.sheets`` lays out).

    python -m diffu_page.data.style_pairs
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image
from pydantic import BaseModel

from diffu_page.captions.prompt import LineRecord
from diffu_page.captions.provenance import collection_key
from diffu_page.captions.records import CropRecord, read_records
from diffu_page.captions.restyle import edit_caption
from diffu_page.crops.grid import fit_to_grid
from diffu_page.data.steer_set import ARCHIVES, DEFAULT_TEMPLATES, SEED, archive_style
from diffu_page.output import console

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

REF_PIXELS = 512 * 1024  # a reference shows the hand smaller than the ~1 MP target, at half its tokens

type Volume = tuple[str, str]  # (collection dir, volume dir)


class Scan(BaseModel):
    """A crop record and where its scan is."""

    record: CropRecord
    path: Path


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


class EvalRow(BaseModel):
    """One crop of an ``diffu_gen.cli.evaluate`` dir with a reference image."""

    file_name: str
    caption: str
    lines: list[LineRecord]
    width: int
    height: int
    reference: str  # relative to the eval dir
    seed: int | None = None
    template: int | None = None
    variant: str | None = None
    example: str | None = None  # diffu_gen.cli.sheets shows it beside the generated crop


def volume_of(record: CropRecord) -> Volume:
    """The volume of a crop, from its PAGE path ``<collection>/<volume>/<volume>/page/<file>.xml``."""
    collection, volume = record.page.split("/")[:2]
    return collection, volume


def link(source: Path, destination: Path) -> None:
    """Hard-link ``source`` to ``destination`` unless it is there (NeMo reads only files inside the manifest's dir)."""
    if not destination.exists():
        destination.hardlink_to(source)


def shrink(source: Path, destination: Path) -> None:
    """Save ``source`` on the latent grid under ``REF_PIXELS`` (see ``fit_to_grid``), so training and generation
    read the same pixels."""
    if destination.exists():
        return
    with Image.open(source) as image:
        fit_to_grid(image, REF_PIXELS).save(destination, quality=92)


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
    """Write ``rows`` as JSON lines."""
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
                metadata=PairMetadata(row_index=len(manifest), reference=reference.record.file_name),
            )
        )
    return manifest


def held_out_rows(
    held_out: list[Scan], volumes: dict[Volume, list[Scan]], out: Path, rng: random.Random
) -> list[EvalRow]:
    """Each held-out crop in the hand of another page of its volume (an unseen page when there is one)."""
    return [
        eval_row(scan, reference.path, out)
        for scan in held_out
        if (reference := pick_reference(scan, volumes, rng)) is not None
    ]


def by_example_rows(
    held_out: list[Scan],
    templates: list[int],
    examples: dict[str, Path],
    volumes: dict[Volume, list[Scan]],
    out: Path,
    rng: random.Random,
) -> list[EvalRow]:
    """The steering templates by example: in their own volume's hand, then in each steering archive's."""
    rows: list[EvalRow] = []
    for t in templates:
        template = held_out[t]
        own = pick_reference(template, volumes, rng)
        references = [(f"own volume ({collection_key(template.record.collection)})", own.path)] if own else []
        references += list(examples.items())
        rows += [
            eval_row(template, path, out, seed=SEED + t, template=t, variant=f"reference: {label}")
            for label, path in references
        ]
    return rows


@app.command()
def main(
    crops: Annotated[Path, typer.Option(help="the v2 crops (diffu_page.data.build_crops output)")] = Path(
        "data_flux3/layout_crops_v2"
    ),
    out: Annotated[Path, typer.Option(help="where the v3 pairs go")] = Path("data_flux3/style_pairs_v3"),
    templates: Annotated[
        list[int] | None, typer.Option("--template", help="a val_eval row for by_example/ (repeat it)")
    ] = None,
    seed: int = 0,
) -> None:
    """Build the v3 reference pairs, the held-out set and the by-example steering set."""
    train, val, held_out = (load_split(crops, split) for split in ("train", "val", "val_eval"))
    rng = random.Random(seed)
    manifest = training_pairs(train, out, rng)
    write_rows(out / "train.jsonl", manifest)
    volumes = by_volume(val + train)
    val_rows = held_out_rows(held_out, volumes, out / "val_eval", rng)
    write_rows(out / "val_eval/metadata.jsonl", val_rows)
    examples = {key: archive_style([s.record for s in train], key, crops).example for key in ARCHIVES}
    example_rows = by_example_rows(
        held_out, templates or list(DEFAULT_TEMPLATES), examples, volumes, out / "by_example", rng
    )
    write_rows(out / "by_example/metadata.jsonl", example_rows)
    skipped = len(train) - len(manifest)
    console.print(f"train: {len(manifest)} pairs ({skipped} crops skipped: no other page in their volume) -> {out}")
    console.print(f"val_eval: {len(val_rows)} crops, by_example: {len(example_rows)} prompts")


if __name__ == "__main__":
    app()
