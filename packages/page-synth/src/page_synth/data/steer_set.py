"""The v2 steering test: does naming an archive in the prompt steer the hand, and naming a look steer the scan?

Takes held-out v2 crops as templates (their text and layout) and writes one prompt per variant, all of one template
on the same seed so only the words differ:

- as written: the template's own caption;
- archive: the caption restyled as each of six archives, with that archive's commonest scan look and years;
- look: the template's own archive with three scan looks (colour cream / greyscale / yellowed).

The output is a crops dir for ``page_synth.evaluate`` (``metadata.jsonl`` + the template scans), plus a real crop of
each archive under ``examples/`` that ``page_synth.sheets`` shows beside the generated one.

    python -m page_synth.data.steer_set   # -> data_flux3/steer_v2
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from page_synth.caption import LineRecord
from page_synth.look import ScanLook
from page_synth.provenance import Provenance
from page_synth.records import CropRecord, read_records
from page_synth.restyle import Style, restyle

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


class SteerRow(BaseModel):
    """One prompt of the steering test, in the shape ``page_synth.evaluate`` reads."""

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
    label: str
    style: Style
    example: Path | None = None


def archive_key(record: CropRecord) -> str:
    """``GT-JLF_jamtlands_domsaga_20260220`` -> ``JLF_jamtlands_domsaga``."""
    return re.sub(r"^GT-|_\d{8}$", "", record.collection)


def archive_style(train: list[CropRecord], key: str, crops_dir: Path) -> ArchiveStyle:
    """The commonest scan look of an archive's training crops, and the commonest years among those crops.

    Raises:
        ValueError: If no training crop is from the archive.
    """
    records = [record for record in train if archive_key(record) == key]
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, default=Path("data_flux3/layout_crops_v2"))
    ap.add_argument("--out", type=Path, default=Path("data_flux3/steer_v2"))
    ap.add_argument(
        "--templates", type=int, nargs="+", default=[1, 17], help="val_eval rows to use as templates"
    )
    args = ap.parse_args()

    train = read_records(args.crops / "train/metadata.jsonl")
    held_out = read_records(args.crops / "val_eval/metadata.jsonl")
    archives = {key: archive_style(train, key, args.crops) for key in ARCHIVES}

    (args.out / "examples").mkdir(parents=True, exist_ok=True)
    for archive in archives.values():
        shutil.copy2(archive.example, args.out / "examples" / archive.example.name)
    rows: list[SteerRow] = []
    for t in args.templates:
        template = held_out[t]
        shutil.copy2(args.crops / "val_eval" / template.file_name, args.out / template.file_name)
        rows += [
            SteerRow(
                file_name=template.file_name,
                caption=restyle(template, variant.style),
                lines=template.lines,
                width=template.width,
                height=template.height,
                seed=SEED + t,
                template=t,
                variant=variant.label,
                example=f"examples/{variant.example.name}" if variant.example else None,
            )
            for variant in variants(template, archives)
        ]
    (args.out / "metadata.jsonl").write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )
    print(f"{len(rows)} prompts from {len(args.templates)} templates -> {args.out}")
    for archive_name, archive in archives.items():
        print(f"  {archive_name}: {archive.style.model_dump()}")


if __name__ == "__main__":
    main()
