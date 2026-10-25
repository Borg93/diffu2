"""diffu's E2 held-out sets: each crop of an eval split scaled as E2 trains (``--scale``, 0.5 for the 512-px stage),
drawn two ways by ``diffu_page.cli.evaluate``:

- ``text/``: from the caption alone, as E2's text-only arm trains;
- ``glyphs/``: with the crop's glyph map (``diffu_page.glyphs``) as its reference image, as the glyph arm trains.

Both sets share the scaled crops (``images/``); ``glyph_maps/`` holds the maps. A row's paths are relative to its set.

    python -m diffu_page.data.glyph_set --crops data_flux3/layout_crops_v2/val_eval --out data_flux3/e2_val_eval
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from PIL import Image

from diffu_page.captions.records import read_records
from diffu_page.crops.grid import scaled
from diffu_page.evalset import EvalCrop, write_jsonl
from diffu_page.glyphs import glyph_map
from diffu_page.output import console

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


def build(crops: Path, out: Path, *, scale: float) -> tuple[list[EvalCrop], list[EvalCrop]]:
    """The text and glyph rows of every crop in ``crops``, its scaled image and glyph map written under ``out``."""
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "glyph_maps").mkdir(exist_ok=True)
    text_rows: list[EvalCrop] = []
    glyph_rows: list[EvalCrop] = []
    for record in read_records(crops / "metadata.jsonl"):
        with Image.open(crops / record.file_name) as image:
            window = scaled(image, scale)
        name = Path(record.file_name).stem
        window.save(out / "images" / f"{name}.png")
        glyph_map(record.lines, window.size).save(out / "glyph_maps" / f"{name}.png")
        row = EvalCrop(
            file_name=f"../images/{name}.png",
            caption=record.caption,
            width=window.width,
            height=window.height,
            lines=record.lines,
        )
        text_rows.append(row)
        glyph_rows.append(
            row.model_copy(
                update={
                    "reference": f"../glyph_maps/{name}.png",
                    "reference_lines": [line.bbox for line in record.lines],
                }
            )
        )
    return text_rows, glyph_rows


@app.command()
def main(
    crops: Annotated[Path, typer.Option(help="an eval split: metadata.jsonl and its crops")],
    out: Annotated[Path, typer.Option(help="where text/, glyphs/, images/ and glyph_maps/ go")],
    scale: Annotated[float, typer.Option(min=0.1, max=1.0, help="scale the crops as E2 trains (0.5: 512 px)")] = 0.5,
) -> None:
    """Write E2's held-out sets: the scaled crops, drawn from the caption alone and with their glyph maps."""
    text_rows, glyph_rows = build(crops, out, scale=scale)
    write_jsonl(out / "text" / "metadata.jsonl", text_rows)
    write_jsonl(out / "glyphs" / "metadata.jsonl", glyph_rows)
    console.print(f"{len(text_rows)} crops at scale {scale:g} -> {out}/text, {out}/glyphs")


if __name__ == "__main__":
    app()
