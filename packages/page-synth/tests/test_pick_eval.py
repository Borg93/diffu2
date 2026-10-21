"""Picking the evaluation set: one crop per collection, spread over the collections."""

from __future__ import annotations

import pytest

from page_synth.captions.records import CropRecord
from page_synth.data.pick_eval import pick


def _record(collection: str) -> CropRecord:
    return CropRecord(
        file_name=f"{collection}.jpg",
        caption="",
        collection=collection,
        page=f"{collection}/v/v/page/1.xml",
        width=1024,
        height=1024,
        greyscale=False,
        paper="cream",
        ink="brown",
        layout=[],
        lines=[],
    )


def test_pick_one_of_several_collections_takes_the_first() -> None:
    records = [_record(name) for name in ("GT-A_20260101", "GT-B_20260101", "GT-C_20260101")]

    picked = pick(records, 1)

    assert [record.collection for record in picked] == ["GT-A_20260101"]


def test_pick_refuses_a_count_below_one() -> None:
    with pytest.raises(ValueError, match="count must be at least 1, got 0"):
        pick([_record("GT-A_20260101")], 0)
