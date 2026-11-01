"""A run's drawn pages out of its table as files, to look at or to hand to tools that read files: ``<id>.jpg`` beside
``<id>.alto.xml`` and ``<id>.page.xml`` (both name the image ``<id>.jpg``).

    diffu-run-export --out /outputs/run --to /outputs/run/files --limit 20
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import lance
import typer

from diffu_page.output import console, plain_file_name
from diffu_runner.table import PAGES, quoted

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

COLUMNS = ["id", "image", "alto_xml", "page_xml"]


def export_pages(table: Path, to: Path, *, ids: Sequence[str] = (), limit: int | None = None) -> list[str]:
    """Write the pages ``table`` holds drawn into ``to``: those of ``ids`` (every page when empty), at most ``limit``.

    Returns:
        The ids written, in the table's order.
    """
    selected = "error IS NULL"
    if ids:
        selected += f" AND id IN ({', '.join(quoted(page_id) for page_id in ids)})"
    to.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    batches = lance.dataset(str(table)).to_batches(
        columns=COLUMNS, filter=selected, limit=limit, batch_size=16, blob_handling="all_binary"
    )
    for batch in batches:
        for row in batch.to_pylist():
            page_id = plain_file_name(row["id"])
            (to / f"{page_id}.jpg").write_bytes(row["image"])
            (to / f"{page_id}.alto.xml").write_text(row["alto_xml"], encoding="utf-8")
            (to / f"{page_id}.page.xml").write_text(row["page_xml"], encoding="utf-8")
            written.append(page_id)
    return written


@app.command()
def main(
    out: Annotated[Path, typer.Option(help="the run's directory (diffu-run --out), which holds pages.lance")],
    to: Annotated[Path, typer.Option(help="where the files go")],
    page_id: Annotated[list[str] | None, typer.Option("--id", help="a page to write (repeat it; default: all)")] = None,
    limit: Annotated[int | None, typer.Option(min=1, help="write at most this many pages")] = None,
) -> None:
    """Write a run's drawn pages as image, ALTO and PAGE files."""
    table = out / PAGES
    if not table.exists():
        raise typer.BadParameter(f"{out} holds no page table ({PAGES})", param_hint="--out")
    written = export_pages(table, to, ids=page_id or [], limit=limit)
    console.print(f"{len(written)} pages -> {to}")


if __name__ == "__main__":
    app()
