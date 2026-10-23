"""Crop planning: a real page is cut where the training builder cut it, its lines read block by block, and a page
drawn from a reference captioned exactly as the reference pairs were captioned for training."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import LineRecord, caption_for, caption_order
from diffu_page.captions.provenance import Provenance, provenance_of
from diffu_page.captions.records import CropRecord
from diffu_page.captions.restyle import ReferenceCaption
from diffu_page.crops.geometry import Crop, CropSettings, crops_for_page
from diffu_page.data.build_crops import parse_page
from diffu_page.data.style_pairs import Pair, Scan, captions, describe
from diffu_page.layout import load_xml_layout
from diffu_page.plan import PageStyle, Reference, plan_page, regions_of

STYLE = PageStyle(
    provenance=Provenance(title="the archive", years="1700"),
    look=ScanLook(greyscale=False, paper="cream", ink="brown"),
)


def test_plan_reads_the_body_block_before_its_margin_notes(body_and_notes_page: Path) -> None:
    layout = load_xml_layout(body_and_notes_page)

    plan = plan_page(layout, STYLE)[0]

    numbered = [line.text for line in plan.lines]  # the caption's numbering: top to bottom, notes interleaved
    body = [text for text in numbered if text.startswith("body")]
    assert [plan.lines[i].text for i in plan.reading_order] == [*body, "note 0", "note 1"]


def test_plan_cuts_a_page_where_the_training_builder_cut_it(body_and_notes_page: Path) -> None:
    trained = crops_for_page(parse_page(body_and_notes_page), (2000, 1900), CropSettings())

    planned = plan_page(load_xml_layout(body_and_notes_page), STYLE)

    assert [(plan.rect, [line.bbox for line in plan.lines]) for plan in planned] == [
        (crop.rect, [crop.slot(line) for line in caption_order(crop.lines)]) for crop in trained
    ]


COLLECTION = "GT-RA_arjeplogs_tingslag_20260220"
PAGE = f"{COLLECTION}/Arjeplogs tingslag (1690-1700)/Arjeplogs tingslag (1690-1700)/page/0001.xml"
LOOK = ScanLook(greyscale=False, paper="yellowed", ink="dark brown")
REFERENCE_LINES = [[100, 50, 300, 950], [500, 50, 700, 950]]


def _record(crop: Crop, name: str) -> CropRecord:
    """The crop as diffu_page.data.build_crops records it for training."""
    caption = caption_for(crop, provenance_of(COLLECTION, Path(PAGE)), LOOK)
    return CropRecord(
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


def test_plan_captions_a_page_drawn_from_a_reference_as_its_training_pairs_were_captioned(
    body_and_notes_page: Path, tmp_path: Path
) -> None:
    layout = load_xml_layout(body_and_notes_page)
    image = tmp_path / "reference.png"
    Image.linear_gradient("L").rotate(30).convert("RGB").save(image)
    reference = Reference(image=image, caption=ReferenceCaption.DESCRIBED, lines=REFERENCE_LINES)
    style = PageStyle(provenance=provenance_of(COLLECTION, Path(PAGE)), look=LOOK)

    planned = plan_page(layout, style, reference=reference)

    crops = crops_for_page(regions_of(layout), (layout.width, layout.height), CropSettings())
    lines = [LineRecord(id=f"Sv_Text_{k}", text="", role="", bbox=box) for k, box in enumerate(REFERENCE_LINES, 1)]
    reference_crop = Scan(record=_record(crops[0], "reference.png").model_copy(update={"lines": lines}), path=image)
    training, _ = captions(
        ReferenceCaption.DESCRIBED, named_fraction=0.0, seed=0, looks={image: describe(reference_crop)}
    )
    assert [plan.caption for plan in planned] == [
        training(
            Pair(target=Scan(record=_record(crop, f"{k}.jpg"), path=tmp_path / f"{k}.jpg"), reference=reference_crop)
        )
        for k, crop in enumerate(crops)
    ]
