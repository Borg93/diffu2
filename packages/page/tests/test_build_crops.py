"""The training-crop builder on a GT folder laid out as the Transkribus export is: a page it cannot read is recorded,
not the end of the build, and no tile is saved without its record."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from diffu_page.crops.geometry import CropSettings
from diffu_page.data.build_crops import build

COLLECTION = "GT-RA_arjeplogs_tingslag_20260220"
VOLUME = "Arjeplogs tingslag (1690-1700)"


def test_build_records_a_page_it_cannot_read_and_writes_the_crops_of_the_rest(
    body_and_notes_page: Path, tmp_path: Path
) -> None:
    volume = tmp_path / "gt" / COLLECTION / VOLUME / VOLUME
    (volume / "page").mkdir(parents=True)
    shutil.copy(body_and_notes_page, volume / "page" / "0001.xml")
    (volume / "page" / "0002.xml").write_text('<?xml version="1.0"?><PcGts', encoding="utf-8")  # cut off mid-write
    for scan in ("0001.jpg", "0002.jpg"):
        Image.new("RGB", (2000, 1900), "white").save(volume / scan)
    out = tmp_path / "out"

    failed = build(tmp_path / "gt", [COLLECTION], out, max_pages=2, val_frac=0.0, seed=0, settings=CropSettings())

    records = [json.loads(line) for line in (out / "train" / "metadata.jsonl").read_text().splitlines()]
    assert list(failed) == [f"{COLLECTION}/{VOLUME}/{VOLUME}/page/0002.xml"]
    assert json.loads((out / "failed_pages.json").read_text()) == failed
    assert sorted(record["file_name"] for record in records) == sorted(tile.name for tile in out.glob("train/*.jpg"))
    assert records
