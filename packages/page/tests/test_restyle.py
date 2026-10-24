"""Restyling a crop's caption: another archive's style (the steering test), or a reference image's, named or described
(v3, v3b, v3d)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from diffu_page.captions.hand import HandLook, ReferenceLook
from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import caption_for
from diffu_page.captions.provenance import Provenance
from diffu_page.captions.records import CropRecord
from diffu_page.captions.restyle import (
    CaptionParts,
    StaleCaptionError,
    Style,
    described_edit_caption,
    edit_caption,
    named_edit_caption,
    restyle,
)
from diffu_page.crops.geometry import Crop, CropSettings, crops_for_page, regions_of
from diffu_page.layout import load_xml_layout

CITED_LINES = (  # how the paragraph cites body_and_notes_crop's lines, after their ink
    ", including two marginal note lines; the main text runs from top to bottom: <Sv_Text_1> <Sv_Text_2> <Sv_Text_3> "
    "<Sv_Text_5> <Sv_Text_7> <Sv_Text_8> <Sv_Text_9> <Sv_Text_10> <Sv_Text_11> <Sv_Text_12> <Sv_Text_13> <Sv_Text_14>. "
    "Two marginal note lines <Sv_Text_4> <Sv_Text_6> sit in the left margin."
)


def _paragraph_and_rows(caption: str) -> tuple[str, list[dict[str, Any]]]:
    split = caption.index(" [{")
    return caption[:split], json.loads(caption[split + 1 :])


def test_restyle_writes_the_caption_the_crop_gets_in_the_target_style(
    body_and_notes_crop: Crop, make_record: Callable[..., CropRecord]
) -> None:
    record = make_record(body_and_notes_crop)
    target = Style(
        provenance=Provenance(title="the court records of Jämtlands domsaga", years="1650"),
        look=ScanLook(greyscale=True, paper="light grey", ink="black"),
    )

    restyled = restyle(CaptionParts.of(record), target)

    assert restyled == caption_for(body_and_notes_crop, target.provenance, target.look).text


@pytest.fixture
def certificate_crop(certificate_page: Path) -> Crop:
    """The crop of a printed certificate filled in by hand: printed lines, handwritten ones and ruled lines."""
    layout = load_xml_layout(certificate_page)
    rules = [(rule.x0, rule.y0, rule.x1, rule.y1) for rule in layout.rules]
    return crops_for_page(regions_of(layout), (layout.width, layout.height), CropSettings(), rules=rules)[0]


def test_restyle_writes_the_caption_a_form_crop_gets_in_the_target_style(
    certificate_crop: Crop, make_record: Callable[..., CropRecord]
) -> None:
    record = make_record(certificate_crop)
    target = Style(
        provenance=Provenance(title="the court records of Jämtlands domsaga", years="1650"),
        look=ScanLook(greyscale=True, paper="light grey", ink="black"),
    )

    restyled = restyle(CaptionParts.of(record), target)

    assert restyled == caption_for(certificate_crop, target.provenance, target.look).text


def test_edit_caption_takes_only_the_paper_from_the_reference_and_names_no_style(
    body_and_notes_crop: Crop, make_record: Callable[..., CropRecord]
) -> None:
    record = make_record(body_and_notes_crop)

    caption = edit_caption(CaptionParts.of(record))

    paragraph, rows = _paragraph_and_rows(caption)
    assert paragraph == (
        "Create a flat, evenly lit archival scan of a new handwritten page in the hand, ink and paper of <ref_image_0>. "
        f"The paper <paper_1> carries 14 lines of cursive handwriting{CITED_LINES}"
    )
    assert rows == [
        {
            "id": "paper_1",
            "from": "ref_image_0",
            "src_bbox": [0, 0, 1000, 1000],
            "tgt_bbox": [0, 0, 1000, 1000],
            "desc": "Aged paper of a bound archival volume.",
        },
        *(
            {
                "id": row.id,
                "from": "target",
                "src_bbox": None,
                "tgt_bbox": row.bbox,
                "desc": row.desc.replace("in dark brown ink.", "in the reference's hand and ink."),
            }
            for row in record.layout[1:]
        ),
    ]


def test_named_edit_caption_names_the_archive_and_look_beside_the_reference(
    body_and_notes_crop: Crop, make_record: Callable[..., CropRecord]
) -> None:
    record = make_record(body_and_notes_crop)

    caption = named_edit_caption(CaptionParts.of(record))

    paragraph, rows = _paragraph_and_rows(caption)
    assert paragraph == (
        "Create a flat, evenly lit archival scan of a new handwritten page from the court minutes of Arjeplogs "
        "tingslag, dated 1690–1700, in the hand, ink and paper of <ref_image_0>. The yellowed paper <paper_1> carries "
        f"14 lines of cursive handwriting in dark brown ink{CITED_LINES}"
    )
    assert [(row["id"], row["from"]) for row in rows] == [
        (row.id, "ref_image_0" if row.id == "paper_1" else "target") for row in record.layout
    ]
    assert {row["desc"].rsplit(", cursive script ", 1)[-1] for row in rows if row["id"].startswith("Sv_Text_")} == {
        "in the reference's hand, in dark brown ink."
    }


def test_described_edit_caption_describes_the_reference_in_place_of_the_crop_style(
    body_and_notes_crop: Crop, make_record: Callable[..., CropRecord]
) -> None:
    record = make_record(body_and_notes_crop)
    reference = ReferenceLook(
        scan=ScanLook(greyscale=True, paper="light grey", ink="black"),
        hand=HandLook(slant=33.0, stroke=0.05, letters=0.3, density=0.2, paper="#C8C8C8", ink="#202020"),
    )

    caption = described_edit_caption(CaptionParts.of(record), reference)

    paragraph, rows = _paragraph_and_rows(caption)
    assert paragraph == (
        "Create a flat, evenly lit greyscale archival scan of a new handwritten page in the hand, ink and paper of "
        "<ref_image_0>: right-slanted cursive, about 35 degrees, with fine pen strokes and medium-sized, closely "
        "packed letters, in black ink (#202020) on light grey paper (#C8C8C8). The paper <paper_1> carries 14 lines "
        f"of cursive handwriting in black ink{CITED_LINES}"
    )
    assert [(row["id"], row["from"], row["desc"]) for row in rows if row["id"] == "paper_1"] == [
        ("paper_1", "ref_image_0", "Aged light grey paper (#C8C8C8) of a bound archival volume, in a greyscale scan.")
    ]
    assert {row["desc"].rsplit('", ', 1)[-1] for row in rows if row["id"].startswith("Sv_Text_")} == {
        "right-slanted cursive in the reference's hand, in black ink."
    }
    assert [word for word in ("Arjeplogs", "yellowed", "dark brown") if word in caption] == []


@pytest.mark.parametrize(
    "hand",
    [
        pytest.param(
            HandLook(slant=None, stroke=None, letters=None, density=0.0, paper="#C8C8C8", ink="#202020"),
            id="no-ink-in-its-lines",
        ),
        pytest.param(
            HandLook(slant=33.0, stroke=0.05, letters=0.3, density=0.2, paper="#C8C8C8", ink=None), id="no-ink-colour"
        ),
    ],
)
def test_described_edit_caption_refuses_a_reference_whose_hand_was_not_measured(
    body_and_notes_crop: Crop, make_record: Callable[..., CropRecord], hand: HandLook
) -> None:
    record = make_record(body_and_notes_crop)
    reference = ReferenceLook(scan=ScanLook(greyscale=True, paper="light grey", ink="black"), hand=hand)

    with pytest.raises(ValueError, match="cannot be described"):
        described_edit_caption(CaptionParts.of(record), reference)


def test_restyle_refuses_a_caption_an_older_provenance_rule_wrote(
    body_and_notes_crop: Crop, make_record: Callable[..., CropRecord]
) -> None:
    record = make_record(body_and_notes_crop)
    raw_folder_name = Provenance(title="RA arjeplogs tingslag 20260220", years="1690–1700")  # before collection_key
    stale = record.model_copy(update={"caption": caption_for(body_and_notes_crop, raw_folder_name, record.look).text})

    with pytest.raises(StaleCaptionError, match="crop.jpg: its caption was not written by the current caption grammar"):
        restyle(CaptionParts.of(stale), Style.of(stale))
