"""The v2 steering test: does naming an archive in the prompt steer the hand, and naming a look steer the scan?

Takes held-out v2 crops as templates (their text and layout) and writes one prompt per variant, all of one template
on the same seed so only the words differ:

- as written: the template's own caption;
- archive: the caption restyled as each of six archives, with that archive's commonest scan look and years;
- look: the template's own archive with three scan looks (colour cream / greyscale / yellowed).

The output is a crops dir for ``eval_layout.py`` (``metadata.jsonl`` + the template scans), plus a real crop of
each archive under ``examples/`` that ``steer_grid.py`` shows beside the generated one.

    .venv/bin/python experiments/flux3_klein/build_steer_set.py   # -> data_flux3/steer_v2
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from caption_style import Style, restyle
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[2]
ARCHIVES = (
    "JLF_jamtlands_domsaga",
    "RA_bergskollegium_stora_kopparbergs_bergslag",
    "RA_bergmastare_i_nora",
    "RA-X_gbg_1790-1794",
    "SSA_stockholms_domkapitel",
    "FNA_finnish_letter_recipes",
)
LOOKS = (  # (greyscale, paper, ink): a colour scan, a microfilm, an old yellowed colour scan
    (False, "cream", "dark brown"),
    (True, "light grey", "black"),
    (False, "yellowed", "faded brown"),
)
SEED = 1000  # template t's variants all use SEED + t


class SteerRow(BaseModel):
    """One prompt of the steering test, in the shape ``eval_layout.py`` reads."""

    file_name: str  # the template's scan (the eval's "real" side)
    caption: str
    lines: list[dict[str, Any]]
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


def archive_key(record: dict[str, Any]) -> str:
    """``GT-JLF_jamtlands_domsaga_20260220`` -> ``JLF_jamtlands_domsaga``."""
    return re.sub(r"^GT-|_\d{8}$", "", record["collection"])


def archive_style(train: list[dict[str, Any]], key: str, crops_dir: Path) -> ArchiveStyle:
    """The commonest (greyscale, paper, ink) of an archive's training crops, and its commonest years.

    Raises:
        ValueError: If no training crop is from the archive.
    """
    rows = [r for r in train if archive_key(r) == key]
    if not rows:
        raise ValueError(f"no v2 training crop from {key}")
    look = Counter((r["greyscale"], r["paper"], r["ink"]) for r in rows).most_common(1)[0][0]
    typical = [r for r in rows if (r["greyscale"], r["paper"], r["ink"]) == look]
    styles = [Style.of(r) for r in typical]
    years = Counter(s.years for s in styles).most_common(1)[0][0]
    style = styles[0].model_copy(update={"years": years})
    return ArchiveStyle(style=style, example=crops_dir / "train" / typical[0]["file_name"])


def variants(
    template: dict[str, Any], archives: dict[str, ArchiveStyle]
) -> list[tuple[str, Style, Path | None]]:
    """(label, style, example crop) of every distinct variant of a template, its own style first."""
    own = Style.of(template)
    candidates: list[tuple[str, Style, Path | None]] = [("as written", own, None)]
    candidates += [(f"archive: {key}", archive.style, archive.example) for key, archive in archives.items()]
    for greyscale, paper, ink in LOOKS:
        look = own.model_copy(update={"greyscale": greyscale, "paper": paper, "ink": ink})
        candidates.append(
            (f"look: {'greyscale' if greyscale else 'colour'}, {paper} paper, {ink} ink", look, None)
        )
    distinct: list[tuple[str, Style, Path | None]] = []
    for candidate in candidates:
        if all(candidate[1] != kept[1] for kept in distinct):
            distinct.append(candidate)
    return distinct


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, default=REPO / "data_flux3/layout_crops_v2")
    ap.add_argument("--out", type=Path, default=REPO / "data_flux3/steer_v2")
    ap.add_argument(
        "--templates", type=int, nargs="+", default=[1, 17], help="val_eval rows to use as templates"
    )
    args = ap.parse_args()

    with (args.crops / "train/metadata.jsonl").open(encoding="utf-8") as metadata:
        train = [json.loads(line) for line in metadata]
    with (args.crops / "val_eval/metadata.jsonl").open(encoding="utf-8") as metadata:
        held_out = [json.loads(line) for line in metadata]
    archives = {key: archive_style(train, key, args.crops) for key in ARCHIVES}

    (args.out / "examples").mkdir(parents=True, exist_ok=True)
    for archive in archives.values():
        shutil.copy2(archive.example, args.out / "examples" / archive.example.name)
    rows: list[SteerRow] = []
    for t in args.templates:
        template = held_out[t]
        shutil.copy2(args.crops / "val_eval" / template["file_name"], args.out / template["file_name"])
        own = Style.of(template)
        for label, style, example in variants(template, archives):
            rows.append(
                SteerRow(
                    file_name=template["file_name"],
                    caption=restyle(template["caption"], own, style),
                    lines=template["lines"],
                    width=template["width"],
                    height=template["height"],
                    seed=SEED + t,
                    template=t,
                    variant=label,
                    example=f"examples/{example.name}" if example else None,
                )
            )
    (args.out / "metadata.jsonl").write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )
    print(f"{len(rows)} prompts from {len(args.templates)} templates -> {args.out}")
    for archive_name, archive in archives.items():
        print(f"  {archive_name}: {archive.style.model_dump()}")


if __name__ == "__main__":
    main()
