"""Picking the evaluation set: one crop per collection, spread over the collections."""

from __future__ import annotations

import pytest

from diffu_page.captions.prompt import LineRecord
from diffu_page.captions.records import CropRecord
from diffu_page.data.pick_eval import pick


def _record(collection: str, name: str = "crop", lines: tuple[LineRecord, ...] = ()) -> CropRecord:
    return CropRecord(
        file_name=f"{collection}_{name}.jpg",
        caption="",
        collection=collection,
        page=f"{collection}/v/v/page/1.xml",
        width=1024,
        height=1024,
        greyscale=False,
        paper="cream",
        ink="brown",
        lines=list(lines),
    )


def test_pick_one_of_several_collections_takes_the_first() -> None:
    records = [_record(name) for name in ("GT-A_20260101", "GT-B_20260101", "GT-C_20260101")]

    picked = pick(records, 1)

    assert [record.collection for record in picked] == ["GT-A_20260101"]


def test_pick_prefers_a_collections_crop_with_a_margin_note() -> None:
    note = LineRecord(id="r1", text="Not.", role="marginalia", bbox=(0, 0, 100, 100))
    records = [_record("GT-A_20260101", "plain"), _record("GT-A_20260101", "noted", lines=(note,))]

    picked = pick(records, 1)

    assert [record.file_name for record in picked] == ["GT-A_20260101_noted.jpg"]


def test_pick_refuses_a_count_below_one() -> None:
    with pytest.raises(ValueError, match="count must be at least 1, got 0"):
        pick([_record("GT-A_20260101")], 0)
