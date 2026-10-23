"""The v3 reference pairs: a crop whose caption predates the caption grammar is left out without moving any other
crop's reference."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from PIL import Image

from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import LineRecord, caption_for
from diffu_page.captions.provenance import Provenance, provenance_of
from diffu_page.captions.records import CropRecord
from diffu_page.captions.restyle import CaptionParts, edit_caption
from diffu_page.crops.geometry import Crop, CropSettings, crops_for_page
from diffu_page.data.style_pairs import Scan, by_volume, pick_pairs, pick_reference, pick_same_hand, training_pairs
from diffu_page.layout import load_xml_layout
from diffu_page.plan import regions_of

COLLECTION = "GT-RA_arjeplogs_tingslag_20260220"
VOLUME = f"{COLLECTION}/Arjeplogs tingslag (1690-1700)/Arjeplogs tingslag (1690-1700)/page"
LOOK = ScanLook(greyscale=False, paper="yellowed", ink="dark brown")
RAW_FOLDER_NAME = Provenance(title="RA arjeplogs tingslag 20260220", years="1690–1700")  # before collection_key


@pytest.fixture
def crop(body_and_notes_page: Path) -> Crop:
    layout = load_xml_layout(body_and_notes_page)
    return crops_for_page(regions_of(layout), (layout.width, layout.height), CropSettings())[0]


def _scans(crop: Crop, directory: Path, stale: str) -> list[Scan]:
    """Six crops on three pages of one volume, as build_crops writes them; ``stale`` was captioned before
    collection_key."""
    directory.mkdir()
    scans: list[Scan] = []
    for k in range(6):
        name = f"{k}.jpg"
        page = f"{VOLUME}/{k // 2:04d}.xml"
        Image.new("RGB", crop.size, "white").save(directory / name)
        provenance = RAW_FOLDER_NAME if name == stale else provenance_of(COLLECTION, Path(page))
        caption = caption_for(crop, provenance, LOOK)
        record = CropRecord(
            file_name=name,
            caption=caption.text,
            collection=COLLECTION,
            page=page,
            width=crop.size[0],
            height=crop.size[1],
            **LOOK.model_dump(),
            layout=caption.layout,
            lines=caption.lines,
        )
        scans.append(Scan(record=record, path=directory / name))
    return scans


def test_training_pairs_leave_out_a_stale_caption_and_keep_every_other_pair(crop: Crop, tmp_path: Path) -> None:
    scans = _scans(crop, tmp_path / "crops", stale="3.jpg")
    pairs = pick_pairs(scans, by_volume(scans), rng=random.Random(0), pick=pick_reference)

    built = training_pairs(pairs, tmp_path / "pairs", lambda pair: edit_caption(CaptionParts.of(pair.target.record)))

    assert built.stale == ["3.jpg"]
    assert [(row.media[0].file_name, row.metadata.reference) for row in built.rows] == [
        (f"targets/{pair.target.record.file_name}", pair.reference.record.file_name)
        for pair in pairs
        if pair.target.record.file_name != "3.jpg"
    ]


def _scan(page: int, texts: list[str]) -> Scan:
    """A crop of scan ``page`` of one volume, holding lines with ``texts``."""
    lines = [LineRecord(id=f"Sv_Text_{k}", text=text, role="", bbox=[0, 0, 100, 1000]) for k, text in enumerate(texts)]
    record = CropRecord(
        file_name=f"{page}_{texts[0]}.jpg",
        caption="",
        collection=COLLECTION,
        page=f"{VOLUME}/40004026_{page:05d}.xml",
        width=64,
        height=64,
        **LOOK.model_dump(),
        layout=[],
        lines=lines,
    )
    return Scan(record=record, path=Path(record.file_name))


def test_same_hand_reference_is_a_crop_of_the_page_sharing_no_line() -> None:
    target, overlapping, apart, facing = (
        _scan(10, ["a", "b"]),
        _scan(10, ["b", "c"]),
        _scan(10, ["d"]),
        _scan(11, ["e"]),
    )

    reference = pick_same_hand(target, by_volume([target, overlapping, apart, facing]), random.Random(0))

    assert reference is apart


def test_same_hand_reference_falls_back_to_a_facing_page_crop_sharing_no_line() -> None:
    """Consecutive scans can overlap (one sheet photographed in two parts), so a crop of the facing scan may hold the
    target's own lines."""
    target, overlapped, facing, farther = (
        _scan(20, ["a", "b"]),
        _scan(21, ["b", "c"]),
        _scan(21, ["d"]),
        _scan(23, ["e"]),
    )
    volumes = by_volume([target, overlapped, facing, farther])

    references = [pick_same_hand(target, volumes, random.Random(seed)) for seed in range(8)]

    assert references == [facing] * 8
