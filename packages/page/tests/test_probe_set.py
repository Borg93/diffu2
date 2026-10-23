"""The control probe: each crop captioned plain, described by its own reference and described by the reference
whose words differ most from it, all drawn from its own reference image."""

from __future__ import annotations

from pathlib import Path

import pytest

from diffu_page.captions.hand import HandLook, ReferenceLook
from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import caption_for
from diffu_page.captions.provenance import provenance_of
from diffu_page.captions.records import CropRecord
from diffu_page.captions.restyle import CaptionParts, described_edit_caption, edit_caption
from diffu_page.crops.geometry import CropSettings, crops_for_page
from diffu_page.data.probe_set import probe_rows
from diffu_page.data.style_pairs import EvalRow
from diffu_page.layout import load_xml_layout
from diffu_page.plan import regions_of

COLLECTION = "GT-RA_arjeplogs_tingslag_20260220"
PAGE = f"{COLLECTION}/Arjeplogs tingslag (1690-1700)/Arjeplogs tingslag (1690-1700)/page/0001.xml"
LOOK = ScanLook(greyscale=False, paper="yellowed", ink="dark brown")
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
def records(body_and_notes_page: Path) -> dict[str, CropRecord]:
    """Three crops of one page, as build_crops records them."""
    layout = load_xml_layout(body_and_notes_page)
    crop = crops_for_page(regions_of(layout), (layout.width, layout.height), CropSettings())[0]
    caption = caption_for(crop, provenance_of(COLLECTION, Path(PAGE)), LOOK)
    return {
        name: CropRecord(
            file_name=name,
            caption=caption.text,
            collection=COLLECTION,
            page=PAGE,
            width=crop.size[0],
            height=crop.size[1],
            **LOOK.model_dump(),
            layout=caption.layout,
            lines=caption.lines,
        )
        for name in ("a.jpg", "b.jpg", "c.jpg")
    }


def test_probe_captions_each_crop_plain_described_and_described_as_the_most_unlike_reference(
    records: dict[str, CropRecord],
) -> None:
    rows = [
        EvalRow(
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
