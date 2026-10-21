"""Restyling a crop's caption: another archive's style (the steering test), or a reference image's (v3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import Caption, caption_for
from diffu_page.captions.provenance import Provenance, provenance_of
from diffu_page.captions.records import CropRecord
from diffu_page.captions.restyle import StaleCaptionError, Style, edit_caption, restyle
from diffu_page.crops.geometry import Crop, CropSettings, crops_for_page
from diffu_page.layout import load_xml_layout
from diffu_page.plan import regions_of

COLLECTION = "GT-RA_arjeplogs_tingslag_20260220"
PAGE = f"{COLLECTION}/Arjeplogs tingslag (1690-1700)/Arjeplogs tingslag (1690-1700)/page/0001.xml"
LOOK = ScanLook(greyscale=False, paper="yellowed", ink="dark brown")


@pytest.fixture
def crop(body_and_notes_page: Path) -> Crop:
    layout = load_xml_layout(body_and_notes_page)
    return crops_for_page(regions_of(layout), (layout.width, layout.height), CropSettings())[0]


def _record(crop: Crop, caption: Caption) -> CropRecord:
    return CropRecord(
        file_name="crop.jpg",
        caption=caption.text,
        collection=COLLECTION,
        page=PAGE,
        width=crop.size[0],
        height=crop.size[1],
        **LOOK.model_dump(),
        layout=caption.layout,
        lines=caption.lines,
    )


def test_restyle_writes_the_caption_the_crop_gets_in_the_target_style(crop: Crop) -> None:
    record = _record(crop, caption_for(crop, provenance_of(COLLECTION, Path(PAGE)), LOOK))
    target = Style(
        provenance=Provenance(title="the court records of Jämtlands domsaga", years="1650"),
        look=ScanLook(greyscale=True, paper="light grey", ink="black"),
    )

    restyled = restyle(record, target)

    assert restyled == caption_for(crop, target.provenance, target.look).text


def test_edit_caption_takes_only_the_paper_from_the_reference_and_names_no_style(crop: Crop) -> None:
    record = _record(crop, caption_for(crop, provenance_of(COLLECTION, Path(PAGE)), LOOK))

    caption = edit_caption(record)

    rows = json.loads(caption[caption.index(" [{") + 1 :])
    assert [(row["id"], row["from"]) for row in rows] == [
        (row.id, "ref_image_0" if row.id == "paper_1" else "target") for row in record.layout
    ]
    assert "<ref_image_0>" in caption
    assert "Arjeplogs" not in caption
    assert "dark brown" not in caption
    assert "yellowed" not in caption


def test_restyle_refuses_a_caption_an_older_provenance_rule_wrote(crop: Crop) -> None:
    raw_folder_name = Provenance(title="RA arjeplogs tingslag 20260220", years="1690–1700")  # before collection_key
    record = _record(crop, caption_for(crop, raw_folder_name, LOOK))

    with pytest.raises(StaleCaptionError, match="crop.jpg: its caption was not written by the current caption grammar"):
        restyle(record, Style.of(record))
