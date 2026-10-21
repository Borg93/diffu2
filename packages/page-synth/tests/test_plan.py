"""Crop planning: a real page is cut where the training builder cut it, and its lines read block by block."""

from __future__ import annotations

from pathlib import Path

from page_synth.captions.look import ScanLook
from page_synth.captions.prompt import caption_order
from page_synth.captions.provenance import Provenance
from page_synth.crops.geometry import CropSettings, crops_for_page
from page_synth.data.build_crops import parse_page
from page_synth.layout import load_xml_layout
from page_synth.plan import PageStyle, plan_page

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
