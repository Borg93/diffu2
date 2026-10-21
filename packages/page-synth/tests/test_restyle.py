"""Restyling a crop's caption: another archive's style (the steering test), or a reference image's (v3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from page_synth.captions.look import ScanLook
from page_synth.captions.prompt import Caption, caption_for
from page_synth.captions.provenance import Provenance, provenance_of
from page_synth.captions.records import CropRecord
from page_synth.captions.restyle import Style, edit_caption, restyle
from page_synth.crops.geometry import Crop, CropSettings, crops_for_page
from page_synth.crops.plan import regions_of
from page_synth.layout import load_xml_layout

COLLECTION = "GT-RA_arjeplogs_tingslag_20260220"
PAGE = f"{COLLECTION}/Arjeplogs tingslag (1690-1700)/Arjeplogs tingslag (1690-1700)/page/0001.xml"
LOOK = ScanLook(greyscale=False, paper="yellowed", ink="dark brown")


@pytest.fixture
def crop(body_and_notes_page: Path) -> Crop:
    layout = load_xml_layout(str(body_and_notes_page), single_page=False)
    return crops_for_page(regions_of(layout), (layout.spec.width, layout.spec.height), CropSettings())[0]


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
