"""Tests for refill: a real page's line geometry written through with new text at the page's own writing density,
its other regions (page number, marginalia) left as they were."""

from __future__ import annotations

from pathlib import Path

import pytest

from diffu_page.layout import Layout, LineBox, load_xml_layout, refill, running_text

_BODY_TEXTS = [  # 40 characters on each 680 px line: every refilled line gets a 40-character budget
    "den fjarde maj artonhundra attiotva fors",
    "hordes arbetskarlen Anders Persson i Hag",
    "a forsamling om natten och vittnen insta",
    "mde och saken hanskots till rattens vida",
    "re provning i staden efter lagen sa somt",
]


def _line(i: int, x0: int, y0: int, x1: int, text: str) -> str:
    points = f"{x0},{y0} {x1},{y0} {x1},{y0 + 40} {x0},{y0 + 40}"
    return f'<TextLine id="l{i}"><Coords points="{points}"/><TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>'


@pytest.fixture
def page(tmp_path: Path) -> Layout:
    """A replayed PAGE file: a page number, a marginal note and five running-text lines."""
    body = "".join(_line(i, 220, 200 + 60 * i, 900, text) for i, text in enumerate(_BODY_TEXTS))
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Page imageFilename="p.jpg" imageWidth="1000" imageHeight="1400">
    <TextRegion id="n" type="page-number">{_line(90, 850, 80, 900, "61.")}</TextRegion>
    <TextRegion id="m" type="marginalia">{_line(91, 20, 200, 180, "No 12")}</TextRegion>
    <TextRegion id="b" type="paragraph">{body}</TextRegion>
  </Page>
</PcGts>"""
    path = tmp_path / "page.xml"
    path.write_text(xml, encoding="utf-8")
    return load_xml_layout(path)


def _body_texts(layout: Layout) -> list[str]:
    return [box.text for box in layout.boxes if box.region == "body"]


def test_refill_flows_the_text_through_the_lines_at_the_pages_density_dropping_the_rest(page: Layout):
    paragraph = "Anno sextonhundra och tjugo den femte dagen i maj kom for ratten bonden Olof Jonsson ifran Rodon"
    assert _body_texts(refill(page, [paragraph])) == [
        "Anno sextonhundra och tjugo den femte",
        "dagen i maj kom for ratten bonden Olof",
        "Jonsson ifran Rodon",
    ]


def test_refill_starts_each_paragraph_on_a_fresh_line(page: Layout):
    assert _body_texts(refill(page, ["Kort stycke.", "Nasta stycke borjar har"])) == [
        "Kort stycke.",
        "Nasta stycke borjar har",
    ]


def test_refill_splits_a_word_that_overruns_its_line_with_the_break_mark(page: Layout):
    paragraph = "Anno 1620 kom landskapshandlingssammanstallningen fram"
    assert _body_texts(refill(page, [paragraph])) == [
        "Anno 1620 kom landskapshandlingssammans¬",
        "tallningen fram",
    ]


def test_refill_keeps_the_other_regions_text(page: Layout):
    others = {box.region: box.text for box in refill(page, ["Ny text."]).boxes if box.region != "body"}
    assert others == {"page-number": "61.", "marginalia": "No 12"}


def test_refill_refuses_a_page_without_a_line_to_measure(page: Layout):
    short = page.model_copy(update={"boxes": [box.model_copy(update={"text": "kort"}) for box in page.boxes]})
    with pytest.raises(ValueError, match="writing density"):
        refill(short, ["Ny text."])


def test_running_text_joins_the_words_a_transcription_breaks_across_lines():
    lines = ["nastkomne mandag up¬", "gifwa fortekning pa de som", "krogar halla wilja"]
    boxes = [LineBox(text=text, x=0, y=60 * i, w=600, h=40, baseline_y=60 * i + 30) for i, text in enumerate(lines)]
    page = Layout(
        width=600,
        height=400,
        boxes=[*boxes, LineBox(text="97.", x=0, y=0, w=40, h=40, baseline_y=30, region="page-number")],
    )
    assert running_text(page) == "nastkomne mandag upgifwa fortekning pa de som krogar halla wilja"
