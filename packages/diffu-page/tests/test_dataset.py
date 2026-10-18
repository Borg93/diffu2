"""Dataset assembly: line crops cover their line and carry the verbatim text, splits are deterministic and
page-coherent, and the manifest round-trips. These guard a reproducible, leak-free HTR corpus build.
"""

from __future__ import annotations

import json

import numpy as np
from diffu_page.dataset import (
    CropRecord,
    PageRecord,
    assign_split,
    export_line_crops,
    line_crops,
    write_manifest,
)
from diffu_page.labels import LineLabel


def _page_and_lines() -> tuple[np.ndarray, list[LineLabel]]:
    page = np.full((200, 300, 3), 240, np.uint8)
    lines = [
        LineLabel(text="alpha", polygon=[(10, 10), (90, 10), (90, 40), (10, 40)], region="body", order=0),
        LineLabel(
            text="beta gamma", polygon=[(10, 60), (180, 60), (180, 95), (10, 95)], region="body", order=1
        ),
    ]
    return page, lines


def test_line_crops_cover_each_line_in_reading_order():
    page, lines = _page_and_lines()
    crops = line_crops(page, lines, pad=4)
    assert [ln.text for _, ln in crops] == ["alpha", "beta gamma"], "crops out of reading order or text lost"
    for img, ln in crops:
        xs, ys = [p[0] for p in ln.polygon], [p[1] for p in ln.polygon]
        assert img.size > 0
        assert img.shape[0] >= max(ys) - min(ys) and img.shape[1] >= max(xs) - min(xs), "crop too small"


def test_export_line_crops_writes_images_and_records(tmp_path):
    page, lines = _page_and_lines()
    records = export_line_crops(page, lines, out_dir=tmp_path, page_name="page_0000", split="train")
    assert len(records) == 2
    for rec in records:
        assert (tmp_path / rec.image).exists(), "crop image not written"
        assert rec.split == "train" and rec.page == "page_0000"


def test_assign_split_is_deterministic_and_honors_fractions():
    assert assign_split(7, val_frac=0.1, test_frac=0.1) == assign_split(7, val_frac=0.1, test_frac=0.1)
    assert all(assign_split(i, val_frac=0.0, test_frac=0.0) == "train" for i in range(50)), (
        "no-split != train"
    )
    splits = [assign_split(i, val_frac=0.2, test_frac=0.2) for i in range(2000)]
    assert 0.12 < splits.count("test") / len(splits) < 0.28, "test fraction off"
    assert 0.12 < splits.count("val") / len(splits) < 0.28, "val fraction off"


def test_write_manifest_round_trips(tmp_path):
    pages = [
        PageRecord(
            image="page_0000.png",
            width=300,
            height=200,
            template="single_column",
            seed=0,
            binding=False,
            recto=True,
            n_lines=2,
            split="train",
            styles=["a.png"],
        )
    ]
    crops = [
        CropRecord(
            image="lines/page_0000_l000.png",
            text="alpha",
            page="page_0000",
            region="body",
            order=0,
            split="train",
        )
    ]
    write_manifest(tmp_path, pages, crops)
    manifest = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert manifest[0]["n_lines"] == 2 and manifest[0]["styles"] == ["a.png"]
    lines = [json.loads(line) for line in (tmp_path / "lines.jsonl").read_text().splitlines()]
    assert lines[0]["text"] == "alpha"
