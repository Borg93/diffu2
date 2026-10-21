"""The v2 steering test: does naming an archive in the prompt steer the hand, and naming a look steer the scan?

Takes held-out v2 crops as templates (their text and layout) and writes one prompt per variant, all of one template
on the same seed so only the words differ:

- as written: the template's own caption;
- archive: the caption restyled as each of six archives, with that archive's commonest scan look and years;
- look: the template's own archive with three scan looks (colour cream / greyscale / yellowed).

The output is a crops dir for ``diffu_gen.cli.evaluate`` (``metadata.jsonl`` + the template scans), plus a real crop of
each archive under ``examples/`` that ``diffu_gen.cli.sheets`` shows beside the generated one.

    python -m diffu_page.data.steer_set   # -> data_flux3/steer_v2
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel

from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import LineRecord
from diffu_page.captions.provenance import Provenance, collection_key
from diffu_page.captions.records import CropRecord, read_records
from diffu_page.captions.restyle import Style, restyle
from diffu_page.output import console

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

ARCHIVES = (
    "JLF_jamtlands_domsaga",
    "RA_bergskollegium_stora_kopparbergs_bergslag",
    "RA_bergmastare_i_nora",
    "RA-X_gbg_1790-1794",
    "SSA_stockholms_domkapitel",
    "FNA_finnish_letter_recipes",
)
LOOKS = (  # a colour scan, a microfilm, an old yellowed colour scan
    ScanLook(greyscale=False, paper="cream", ink="dark brown"),
    ScanLook(greyscale=True, paper="light grey", ink="black"),
    ScanLook(greyscale=False, paper="yellowed", ink="faded brown"),
)
SEED = 1000  # template t's variants all use SEED + t
DEFAULT_TEMPLATES = (1, 17)  # the val_eval rows the steering test uses


class SteerRow(BaseModel):
    """One prompt of the steering test, in the shape ``diffu_gen.cli.evaluate`` reads."""

    file_name: str  # the template's scan (the eval's "real" side)
    caption: str
    lines: list[LineRecord]
    width: int
    height: int
    seed: int
    template: int
    variant: str
    example: str | None  # a real crop in the variant's archive style, relative to the output dir


class ArchiveStyle(BaseModel):
    """An archive's commonest look in the v2 training crops, with one crop showing it."""

    style: Style
    example: Path


class Variant(BaseModel):
    """One way to draw a template: its label, its style, and a real crop showing that style."""

    label: str
    style: Style
    example: Path | None = None


def archive_style(train: list[CropRecord], key: str, crops_dir: Path) -> ArchiveStyle:
    """The commonest scan look of an archive's training crops, and the commonest years among those crops.

    Raises:
        ValueError: If no training crop is from the archive.
    """
    records = [record for record in train if collection_key(record.collection) == key]
    if not records:
        raise ValueError(f"no v2 training crop from {key}")
    look = Counter(record.look for record in records).most_common(1)[0][0]
    typical = [record for record in records if record.look == look]
    years = Counter(record.provenance.years for record in typical).most_common(1)[0][0]
    provenance = Provenance(title=typical[0].provenance.title, years=years)
    return ArchiveStyle(
        style=Style(provenance=provenance, look=look), example=crops_dir / "train" / typical[0].file_name
    )


def look_label(look: ScanLook) -> str:
    """A variant's label for a scan look."""
    return f"look: {'greyscale' if look.greyscale else 'colour'}, {look.paper} paper, {look.ink} ink"


def variants(template: CropRecord, archives: dict[str, ArchiveStyle]) -> list[Variant]:
    """Every distinct variant of a template, its own style first."""
    own = Style.of(template)
    candidates = [
        Variant(label="as written", style=own),
        *(Variant(label=f"archive: {key}", style=a.style, example=a.example) for key, a in archives.items()),
        *(Variant(label=look_label(look), style=own.model_copy(update={"look": look})) for look in LOOKS),
    ]
    distinct: list[Variant] = []
    for candidate in candidates:
        if all(candidate.style != kept.style for kept in distinct):
            distinct.append(candidate)
    return distinct


def steer_row(template: CropRecord, variant: Variant, index: int) -> SteerRow:
    """The steering prompt drawing ``template`` as ``variant``, from template ``index``'s noise."""
    return SteerRow(
        file_name=template.file_name,
        caption=restyle(template, variant.style),
        lines=template.lines,
        width=template.width,
        height=template.height,
        seed=SEED + index,
        template=index,
        variant=variant.label,
        example=f"examples/{variant.example.name}" if variant.example else None,
    )


@app.command()
def main(
    crops: Annotated[Path, typer.Option(help="the v2 crops (diffu_page.data.build_crops output)")] = Path(
        "data_flux3/layout_crops_v2"
    ),
    out: Annotated[Path, typer.Option(help="where the steering set goes")] = Path("data_flux3/steer_v2"),
    templates: Annotated[
        list[int] | None, typer.Option("--template", help="a val_eval row to use as a template (repeat it)")
    ] = None,
) -> None:
    """Build the v2 steering set: held-out templates restyled as other archives and scan looks."""
    train = read_records(crops / "train/metadata.jsonl")
    held_out = read_records(crops / "val_eval/metadata.jsonl")
    archives = {key: archive_style(train, key, crops) for key in ARCHIVES}
    (out / "examples").mkdir(parents=True, exist_ok=True)
    for archive in archives.values():
        shutil.copy2(archive.example, out / "examples" / archive.example.name)
    rows: list[SteerRow] = []
    for t in templates or DEFAULT_TEMPLATES:
        template = held_out[t]
        shutil.copy2(crops / "val_eval" / template.file_name, out / template.file_name)
        rows += [steer_row(template, variant, t) for variant in variants(template, archives)]
    (out / "metadata.jsonl").write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
    console.print(f"{len(rows)} prompts -> {out}")
    for archive_name, archive in archives.items():
        console.print(f"  {archive_name}: {archive.style.model_dump()}")


if __name__ == "__main__":
    app()
