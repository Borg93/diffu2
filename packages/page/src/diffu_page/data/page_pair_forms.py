"""Every prompt form for the rows of a page-pair manifest, written as a new manifest beside it: a set built before a
form was added gets that form, its images and the rest of its rows as they are.

A page-pair set (``diffu_page.data.page_pairs``) gives each page and window its Layout prompt in every form the
reference training mixes (``diffu_moss.train.pairs.Variant``), with its length in the text encoder's tokens. Here each
row of a manifest in the set's dir (its ``train.jsonl`` or ``val.jsonl``, or one made of their rows, which may repeat a
row under another id) is planned again as the build planned it, by the set's own settings (its ``scan.json``): its
page, or the page its window is of, read from the PAGE file its scan's record names (``scanned/``), cut from its scan
at its whole-page size, captioned in its archive's name and the look measured on it, and every form prompted and
counted (``diffu_page.data.page_pairs.page_record``); no image is written. A form the row holds must come out as it
holds it, prompt and tokens, or the row is refused: its set was planned otherwise than a build plans it now. The forms
it lacks are added; a row with a form past the text encoder's 4,096 tokens is refused, as a build would skip it. The
forms named by ``--replace`` (repeat it) are replaced by the ones planned now, where a form's caption changed after its
set was built (the sample form, its paper row citing the sample's paper): every other form is still checked. The rows
are written to ``--out``, a new file, in their order, each as it was but for the forms added and replaced.

    python -m diffu_page.data.page_pair_forms --pairs data_flux3/page_pairs_eight/train_bal.jsonl \\
        --out data_flux3/page_pairs_eight/train_bal_sample.jsonl --workers 4
    python -m diffu_page.data.page_pair_forms --pairs data_flux3/page_pairs_eight/train_bal_sample.jsonl \\
        --out data_flux3/page_pairs_eight/train_bal_sample2.jsonl --replace sample --workers 4
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict

from diffu_moss.modular_pipelines.moss.encoders import MAX_PROMPT_TOKENS
from diffu_moss.train.pairs import PagePairRow, Variant
from diffu_page.captions.provenance import provenance_of
from diffu_page.data.page_pairs import (
    SCAN_RECORD,
    SCANNED,
    ScanResult,
    ScanSettings,
    in_processes,
    load_scan,
    page_id,
    page_record,
    recorded_settings,
    whole_page,
)
from diffu_page.evalset import read_jsonl, write_jsonl
from diffu_page.layout import load_xml_layout, scan_pages
from diffu_page.layout.page_xml import find_scan
from diffu_page.output import console

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


class Forms(BaseModel):
    """A page's or a window's Layout prompt in every form, with its length in the text encoder's tokens."""

    model_config = ConfigDict(frozen=True)

    prompts: dict[Variant, str]
    tokens: dict[Variant, int]


class ScanTargets(BaseModel):
    """A scan of a page-pair set, by its PAGE file (relative to the GT root), the targets a manifest's rows draw of
    its pages (by page id: the page's own id, its windows' or both) and the set's settings."""

    xml: str
    targets: dict[str, list[str]]
    settings: ScanSettings


class FormsAdded(BaseModel):
    """What a manifest given every form holds: its rows, how many of them each form was added to and how many it was
    replaced in (planned otherwise now), and its longest prompt in the text encoder's tokens."""

    rows: int
    added: dict[Variant, int]
    replaced: dict[Variant, int]
    longest: int


def planned_forms(task: ScanTargets) -> dict[str, Forms]:
    """The targets of ``task``'s scan in every form, by id: each page with a target read from its PAGE file and its
    scan and planned as its build planned it (:func:`diffu_page.data.page_pairs.page_record`).

    Raises:
        ValueError: If the PAGE file names no scan beside it, the scan is not the size it gives, or a target is no page
            of it nor a window of one.
    """
    settings = task.settings
    xml = Path(settings.gt_root) / task.xml
    scan_path = find_scan(xml)
    if scan_path is None:
        raise ValueError(f"{task.xml} names no scan beside it")
    layout = load_xml_layout(xml)
    scan = load_scan(scan_path, layout)
    provenance = provenance_of(Path(task.xml).parts[0], xml)
    planned: dict[str, Forms] = {}
    for page in scan_pages(layout):
        identity = page_id(Path(task.xml), page)
        if identity not in task.targets:
            continue
        image = whole_page(page, scan.crop(page.rect), settings.pixels)
        record = page_record(page, image, identity, provenance=provenance, settings=settings)
        planned |= {item.id: Forms(prompts=item.prompts, tokens=item.tokens) for item in (record, *record.windows)}
    targets = [target for page_targets in task.targets.values() for target in page_targets]
    missing = [target for target in targets if target not in planned]
    if missing:
        raise ValueError(f"{task.xml} gives no page or window {', '.join(missing)}")
    return {target: planned[target] for target in targets}


def planned_otherwise(row: PagePairRow, forms: Forms) -> list[Variant]:
    """The forms ``row`` holds that ``forms`` (its target planned now) give otherwise, their prompt or their tokens."""
    return [
        variant
        for variant, prompt in row.prompts.items()
        if (prompt, row.tokens.get(variant)) != (forms.prompts[variant], forms.tokens[variant])
    ]


def with_every_form(row: PagePairRow, forms: Forms, *, replace: frozenset[Variant] = frozenset()) -> PagePairRow:
    """``row`` with its target's prompt in every form, as ``forms`` (its target planned now) give them: the forms it
    lacks added, the forms of ``replace`` replaced.

    Raises:
        ValueError: If a form the row holds, not one of ``replace``, is planned otherwise now, its prompt or its
            tokens, or a form passes the text encoder's tokens.
    """
    otherwise = [variant for variant in planned_otherwise(row, forms) if variant not in replace]
    if otherwise:
        raise ValueError(
            f"row {row.id}: its {', '.join(otherwise)} prompt is planned otherwise now than its set was built"
        )
    if max(forms.tokens.values()) > MAX_PROMPT_TOKENS:
        raise ValueError(f"row {row.id}: a form passes the text encoder's {MAX_PROMPT_TOKENS} tokens: {forms.tokens}")
    return row.model_copy(update={"prompts": forms.prompts, "tokens": forms.tokens})


def scan_xml(directory: Path, digest: str) -> str:
    """The PAGE file (relative to the GT root) of the scan of the set in ``directory`` whose PAGE file's digest is
    ``digest``, as the scan's record names it.

    Raises:
        ValueError: If the set holds no record of the scan.
    """
    record = directory / SCANNED / f"{digest}.json"
    if not record.is_file():
        raise ValueError(f"{directory} holds no record of the scan {SCANNED}/{record.name}")
    return ScanResult.model_validate_json(record.read_text(encoding="utf-8")).xml


def write_every_form(
    manifest: Path, out: Path, *, workers: int, replace: frozenset[Variant] = frozenset()
) -> FormsAdded:
    """The rows of ``manifest``, a manifest in a page-pair set's dir, in every form, the forms of ``replace`` planned
    again (the module's docstring), written to ``out`` on ``workers`` processes.

    Raises:
        ValueError: If ``out`` exists (the manifest itself among others), ``manifest`` is in no page-pair set's dir,
            the set holds no record of a row's scan, or a row is refused (:func:`with_every_form`).
    """
    if out.exists():
        raise ValueError(f"{out} exists: give a new file, the manifest is kept as it is")
    directory = manifest.parent
    if not (directory / SCAN_RECORD).is_file():
        raise ValueError(f"{manifest} is in no page-pair set's dir: no {SCAN_RECORD} beside it")
    settings = recorded_settings(directory / SCAN_RECORD)
    rows = read_jsonl(manifest, PagePairRow)
    targets: dict[str, dict[str, list[str]]] = {}  # by the digest of a scan's PAGE file, by page id, the targets drawn
    for row in rows:
        target = Path(row.target).stem
        page = target if row.window is None else row.window.page
        digest = page.split("_", 1)[0]  # a page's id begins with its PAGE file's digest (page_pairs.page_id)
        page_targets = targets.setdefault(digest, {}).setdefault(page, [])
        if target not in page_targets:
            page_targets.append(target)
    tasks = [
        ScanTargets(xml=scan_xml(directory, digest), targets=pages, settings=settings)
        for digest, pages in targets.items()
    ]
    forms: dict[str, Forms] = {}
    for k, planned in enumerate(in_processes(planned_forms, tasks, workers), start=1):
        forms |= planned
        if k % 100 == 0:
            console.print(f"{k} of {len(tasks)} scans planned")
    planned = [forms[Path(row.target).stem] for row in rows]
    written = [with_every_form(row, row_forms, replace=replace) for row, row_forms in zip(rows, planned, strict=True)]
    partial = out.with_name(f"{out.name}.partial")
    write_jsonl(partial, written)
    partial.rename(out)
    added = Counter(
        variant for old, new in zip(rows, written, strict=True) for variant in new.prompts.keys() - old.prompts.keys()
    )
    replaced = Counter(
        variant for row, row_forms in zip(rows, planned, strict=True) for variant in planned_otherwise(row, row_forms)
    )
    return FormsAdded(
        rows=len(written),
        added={variant: added[variant] for variant in Variant if added[variant]},
        replaced={variant: replaced[variant] for variant in Variant if replaced[variant]},
        longest=max((max(row.tokens.values()) for row in written), default=0),
    )


@app.command()
def main(
    pairs: Annotated[
        Path,
        typer.Option(
            help="a manifest in a page-pair set's dir: its train.jsonl or val.jsonl, or one made of their rows"
        ),
    ],
    out: Annotated[Path, typer.Option(help="the manifest to write, a new file: its rows in every form")],
    workers: Annotated[int, typer.Option(min=1, help="processes planning the scans: fewer beside a training")] = max(
        1, (os.cpu_count() or 2) // 2
    ),
    replace: Annotated[
        list[Variant] | None,
        typer.Option(
            "--replace",
            help="a form to plan again and replace in every row (repeat it), as a caption changed since the set was "
            "built; every other form must still come out as the row holds it",
        ),
    ] = None,
) -> None:
    """Give the rows of a page-pair manifest every prompt form, planned as their set was built: a new manifest."""
    try:
        added = write_every_form(pairs, out, workers=workers, replace=frozenset(replace or ()))
    except ValueError as error:  # a used --out, a manifest of no set, or a row planned otherwise: the arguments'
        raise typer.BadParameter(str(error)) from error
    console.print(f"{added.model_dump_json()} -> {out}")


if __name__ == "__main__":
    app()
