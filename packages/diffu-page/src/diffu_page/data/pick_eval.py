"""Pick a fixed evaluation set from a split of diffu_page.data.build_crops: one crop per collection, spread over them.

Each collection contributes its first crop with a margin note, heading or page number (else its first crop),
and when there are more collections than ``-n`` the picks are spread evenly over the alphabetical list, so
every archive group is represented. The picked images are hard-linked next to their metadata.

    python -m diffu_page.data.pick_eval data_flux3/layout_crops_v2/val \
        data_flux3/layout_crops_v2/val_eval -n 24
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from diffu_page.captions.records import CropRecord, read_records
from diffu_page.output import console

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


def has_role(record: CropRecord) -> bool:
    """Whether the crop has a line with a role (a margin note, heading, page number, ...)."""
    return any(line.role for line in record.lines)


def pick(records: list[CropRecord], count: int) -> list[CropRecord]:
    """One record per collection (preferring one with a role line), at most ``count``, spread over them.

    Raises:
        ValueError: If ``count`` is below 1.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    chosen: dict[str, CropRecord] = {}
    for record in records:
        current = chosen.get(record.collection)
        if current is None or (has_role(record) and not has_role(current)):
            chosen[record.collection] = record
    names = sorted(chosen)
    if len(names) > count:
        step = (len(names) - 1) / max(count - 1, 1)
        names = list(dict.fromkeys(names[round(i * step)] for i in range(count)))
    return [chosen[name] for name in names]


@app.command()
def main(
    split: Annotated[Path, typer.Argument(help="a split folder: metadata.jsonl and the crops")],
    out: Annotated[Path, typer.Argument(help="where the picked crops and their metadata.jsonl go")],
    n: Annotated[int, typer.Option("-n", help="crops to pick")] = 24,
) -> None:
    """Pick one held-out crop per collection, spread over the collections."""
    picked = pick(read_records(split / "metadata.jsonl"), n)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metadata.jsonl").open("w", encoding="utf-8") as metadata:
        for record in picked:
            image = out / record.file_name
            if not image.exists():
                image.hardlink_to(split / record.file_name)
            metadata.write(f"{record.model_dump_json()}\n")
    console.print(f"{len(picked)} crops, {sum(has_role(r) for r in picked)} with a margin note, heading or page number")


if __name__ == "__main__":
    app()
