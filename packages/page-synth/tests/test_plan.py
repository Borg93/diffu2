"""Crop planning: a page's crops number their lines top to bottom but read them block by block."""

from __future__ import annotations

from pathlib import Path

from page_synth.captions.look import ScanLook
from page_synth.captions.provenance import Provenance
from page_synth.crops.plan import PageStyle, plan_page
from page_synth.layout import load_xml_layout

STYLE = PageStyle(
    provenance=Provenance(title="the archive", years="1700"),
    look=ScanLook(greyscale=False, paper="cream", ink="brown"),
)


def test_plan_reads_the_body_block_before_its_margin_notes(body_and_notes_page: Path) -> None:
    layout = load_xml_layout(str(body_and_notes_page), single_page=False)

    plan = plan_page(layout, STYLE)[0]

    numbered = [line.text for line in plan.lines]  # the caption's numbering: top to bottom, notes interleaved
    body = [text for text in numbered if text.startswith("body")]
    assert [plan.lines[i].text for i in plan.reading_order] == [*body, "note 0", "note 1"]
