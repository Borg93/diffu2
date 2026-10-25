"""The glyph map diffu traces in E2: each line's text set into its own box, on its baseline, at the target's size."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from diffu_page.captions.prompt import LineRecord
from diffu_page.captions.records import CropRecord
from diffu_page.crops.geometry import SLOT_SIDE_MARGIN
from diffu_page.data.glyph_set import app as glyph_set_app
from diffu_page.evalset import read_crops
from diffu_page.glyphs import glyph_map

SIZE = (512, 256)


def _line(text: str, bbox: tuple[int, int, int, int], baseline: int | None = None) -> LineRecord:
    return LineRecord(id="Sv_Text_1", text=text, role="", bbox=bbox, baseline=baseline)


def _ink(lines: list[LineRecord]) -> np.ndarray:
    return np.asarray(glyph_map(lines, SIZE).convert("L")) < 128


def test_glyph_map_sets_each_line_across_its_own_box_and_nowhere_else() -> None:
    lines = [_line("Stockholm den 3 Junii", (100, 100, 350, 900)), _line("Wälborne Herr", (500, 200, 750, 700))]

    ink = _ink(lines)

    for line in lines:
        y0, x0, y1, x1 = (round(v * side / 1000) for v, side in zip(line.bbox, (256, 512, 256, 512), strict=True))
        margin = round(SLOT_SIDE_MARGIN * (y1 - y0))
        columns = np.flatnonzero(ink[y0:y1].any(axis=0))
        # within a pixel: the stretched glyphs' first and last columns are anti-aliased
        assert (columns[0], columns[-1]) == (pytest.approx(x0 + margin, abs=1), pytest.approx(x1 - margin - 1, abs=1))
    outside = ink.copy()
    for line in lines:
        y0, x0, y1, x1 = (round(v * side / 1000) for v, side in zip(line.bbox, (256, 512, 256, 512), strict=True))
        outside[y0:y1, x0:x1] = False
    assert not outside.any()


def test_glyph_map_sits_the_glyphs_on_the_line_baseline() -> None:
    ink = _ink([_line("mnmnmnmn", (100, 100, 400, 900), baseline=320)])

    assert np.flatnonzero(ink.any(axis=1))[-1] == round(320 * 256 / 1000) - 1


def test_glyph_map_leaves_out_a_character_the_font_lacks() -> None:
    with_missing = glyph_map([_line("aꝕb", (100, 100, 400, 900))], SIZE)

    assert with_missing.tobytes() == glyph_map([_line("ab", (100, 100, 400, 900))], SIZE).tobytes()


def test_glyph_map_refuses_a_size_off_the_latent_grid() -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        glyph_map([_line("a", (100, 100, 400, 900))], (500, 256))


def test_glyph_set_writes_each_scaled_crop_once_drawn_from_its_caption_and_once_from_its_glyph_map(
    tmp_path: Path,
) -> None:
    crops = tmp_path / "crops"
    crops.mkdir()
    Image.linear_gradient("L").resize((128, 64)).convert("RGB").save(crops / "a.jpg")
    line = _line("Wälborne Herr", (100, 100, 400, 900))
    record = CropRecord(
        file_name="a.jpg",
        caption="A page.",
        collection="GT-A_20260101",
        page="GT-A_20260101/v/v/page/1.xml",
        width=128,
        height=64,
        greyscale=False,
        paper="cream",
        ink="brown",
        lines=[line],
    )
    (crops / "metadata.jsonl").write_text(record.model_dump_json() + "\n", encoding="utf-8")
    out = tmp_path / "e2"

    result = CliRunner().invoke(glyph_set_app, ["--crops", str(crops), "--out", str(out), "--scale", "0.5"])

    assert result.exit_code == 0, result.output
    (text,), (glyphs,) = read_crops(out / "text"), read_crops(out / "glyphs")
    assert (text.reference, text.width, text.height) == (None, 64, 32)
    assert text.real_image(out / "text", (64, 32)).size == (64, 32)
    reference = glyphs.reference_image(out / "glyphs")
    assert reference is not None
    assert reference.tobytes() == glyph_map([line], (64, 32)).tobytes()
