"""The control probe for described reference captions: does FLUX.3 draw what the words say about a reference's hand?

BFL's guidance (flux-3-archival-formats: run a control probe before trusting a style vocabulary) is to hold the
subject and change only the style words. Here the subject is a held-out eval set with its references
(``diffu_page.data.style_pairs``' ``val_eval``), drawn three ways from the same references:

- ``plain/``: the caption cites ``<ref_image_0>`` and names no style (``edit_caption``, the v3 and v3c captions);
- ``described/``: it also describes the reference's measured hand, ink and paper (``described_edit_caption``, v3d);
- ``contradicting/``: it describes another reference of the set, the one whose words differ most from its own.

Each is an evaluate dir (``diffu_page.cli.evaluate``, which measures every drawn hand against its reference); the
contradicting rows also name the reference their words describe (``words_from``, ``words_lines``). The words work
when the described drawings come closer to their reference's hand than the plain ones, and the contradicting
drawings move towards their words; if neither moves, the words are inert and v3d's captions would not help.

    python -m diffu_page.data.probe_set --pairs data_flux3/style_pairs_v3c/val_eval --out data_flux3/probe_described
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from diffu_page.captions.hand import (
    DENSITY_WORDS,
    LETTER_WORDS,
    STROKE_WORDS,
    ReferenceLook,
    measure_reference,
    slant_words,
)
from diffu_page.captions.look import band_name
from diffu_page.captions.records import CropRecord, read_records
from diffu_page.captions.restyle import CaptionParts, described_edit_caption, edit_caption
from diffu_page.data.style_pairs import EvalRow, link, write_rows
from diffu_page.output import console

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

FORMS = ("plain", "described", "contradicting")


class ProbeRow(EvalRow):
    """An eval row of the probe: for ``contradicting/``, the reference its words describe instead of its own."""

    words_from: str | None = None
    words_lines: list[list[int]] | None = None


def words_of(look: ReferenceLook) -> tuple[str, ...]:
    """The words a described caption names ``look`` by: its slant, pen, letter and spacing bands, and its scan's,
    paper's and ink's names."""
    hand, scan = look.hand, look.scan
    return (
        slant_words(hand.slant),
        band_name(hand.stroke, STROKE_WORDS),
        band_name(hand.letters, LETTER_WORDS),
        band_name(hand.density, DENSITY_WORDS),
        "greyscale" if scan.greyscale else "colour",
        scan.paper,
        scan.ink,
    )


def furthest(look: ReferenceLook, others: list[ReferenceLook]) -> int:
    """The index of the look in ``others`` named by the most words unlike ``look``'s, the first of a tie."""
    own = words_of(look)
    return max(range(len(others)), key=lambda k: sum(a != b for a, b in zip(own, words_of(others[k]), strict=True)))


def probe_rows(
    rows: list[EvalRow], records: dict[str, CropRecord], looks: list[ReferenceLook]
) -> dict[str, list[ProbeRow]]:
    """``rows`` captioned three ways (``FORMS``) from their target records and their references' ``looks``."""
    probe: dict[str, list[ProbeRow]] = {form: [] for form in FORMS}
    for row, look in zip(rows, looks, strict=True):
        parts = CaptionParts.of(records[row.file_name])
        other = furthest(look, looks)
        base = row.model_dump()
        probe["plain"].append(ProbeRow.model_validate({**base, "caption": edit_caption(parts)}))
        probe["described"].append(ProbeRow.model_validate({**base, "caption": described_edit_caption(parts, look)}))
        probe["contradicting"].append(
            ProbeRow.model_validate(
                {
                    **base,
                    "caption": described_edit_caption(parts, looks[other]),
                    "words_from": rows[other].reference,
                    "words_lines": rows[other].reference_lines,
                }
            )
        )
    return probe


@app.command()
def main(
    pairs: Annotated[Path, typer.Option(help="a style_pairs eval dir whose rows list their reference_lines")],
    out: Annotated[Path, typer.Option(help="where plain/, described/ and contradicting/ go")],
    crops: Annotated[Path, typer.Option(help="the crops the pairs were built from")] = Path(
        "data_flux3/layout_crops_v2"
    ),
) -> None:
    """Write the probe's three eval dirs."""
    rows = [EvalRow.model_validate_json(line) for line in (pairs / "metadata.jsonl").open(encoding="utf-8")]
    records = {
        record.file_name: record
        for split in ("train", "val", "val_eval")
        for record in read_records(crops / split / "metadata.jsonl")
    }
    looks = [measure_reference(pairs / row.reference, row.reference_lines) for row in rows]
    for form, probe in probe_rows(rows, records, looks).items():
        (out / form / "refs").mkdir(parents=True, exist_ok=True)
        for row in probe:
            link(pairs / row.file_name, out / form / row.file_name)
            link(pairs / row.reference, out / form / row.reference)
        write_rows(out / form / "metadata.jsonl", probe)
    (out / "looks.json").write_text(
        json.dumps({row.reference: look.model_dump() for row, look in zip(rows, looks, strict=True)}, indent=1),
        encoding="utf-8",
    )
    console.print(f"{len(rows)} crops x {', '.join(FORMS)} -> {out}")


if __name__ == "__main__":
    app()
