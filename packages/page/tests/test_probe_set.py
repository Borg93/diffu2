"""The control probe: each crop captioned plain, described by its own reference and described by the reference
whose words differ most from it, all drawn from its own reference image."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from diffu_page.captions.hand import HandLook, ReferenceLook
from diffu_page.captions.look import ScanLook
from diffu_page.captions.records import CropRecord
from diffu_page.captions.restyle import CaptionParts, described_edit_caption, edit_caption
from diffu_page.crops.geometry import Crop
from diffu_page.data.probe_set import probe_rows
from diffu_page.evalset import ReferencedCrop

if TYPE_CHECKING:
    from conftest import RecordMaker

SLANTED = ReferenceLook(
    scan=ScanLook(greyscale=False, paper="cream", ink="brown"),
    hand=HandLook(slant=33.0, stroke=0.05, letters=0.2, density=0.09, paper="#E7D4B9", ink="#836148"),
)
SLANTED_BROADER = SLANTED.model_copy(update={"hand": SLANTED.hand.model_copy(update={"stroke": 0.09})})
UPRIGHT = ReferenceLook(
    scan=ScanLook(greyscale=True, paper="light grey", ink="black"),
    hand=HandLook(slant=2.0, stroke=0.09, letters=0.45, density=0.2, paper="#C8C8C8", ink="#202020"),
)


@pytest.fixture
def records(body_and_notes_crop: Crop, make_record: RecordMaker) -> dict[str, CropRecord]:
    """Three crops of one page, as build_crops records them."""
    return {name: make_record(body_and_notes_crop, name) for name in ("a.jpg", "b.jpg", "c.jpg")}


def _rows(records: dict[str, CropRecord]) -> list[ReferencedCrop]:
    """The records as an eval set's rows, each drawn from its own reference image."""
    return [
        ReferencedCrop(
            file_name=name,
            caption=record.caption,
            lines=record.lines,
            width=record.width,
            height=record.height,
            reference=f"refs/{k}.jpg",
            reference_lines=[[100 * k, 0, 100 * k + 90, 1000]],
        )
        for k, (name, record) in enumerate(records.items())
    ]


def test_probe_captions_each_crop_plain_described_and_described_as_the_most_unlike_reference(
    records: dict[str, CropRecord],
) -> None:
    rows = _rows(records)
    looks = [SLANTED, SLANTED_BROADER, UPRIGHT]

    probe = probe_rows(rows, records, looks)

    parts = [CaptionParts.of(record) for record in records.values()]
    assert [row.caption for row in probe["plain"]] == [edit_caption(p) for p in parts]
    assert [row.caption for row in probe["described"]] == [
        described_edit_caption(p, look) for p, look in zip(parts, looks, strict=True)
    ]
    assert [(row.caption, row.words_from, row.reference) for row in probe["contradicting"]] == [
        (described_edit_caption(parts[0], UPRIGHT), "refs/2.jpg", "refs/0.jpg"),
        (described_edit_caption(parts[1], UPRIGHT), "refs/2.jpg", "refs/1.jpg"),
        (described_edit_caption(parts[2], SLANTED), "refs/0.jpg", "refs/2.jpg"),
    ]


def test_probe_refuses_references_whose_words_never_differ(records: dict[str, CropRecord]) -> None:
    looks = [SLANTED, SLANTED, SLANTED]

    with pytest.raises(ValueError, match="no reference is named by other words than this one"):
        probe_rows(_rows(records), records, looks)
