"""Reading a real page's layout from PAGE or ALTO XML: lines with their own outline, baseline, region role, block and
reading order."""

from __future__ import annotations

from pathlib import Path

import pytest

from page_synth.layout import load_xml_layout

_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"


def _page(tmp_path: Path, regions: str) -> Path:
    path = tmp_path / "page.xml"
    path.write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>\n<PcGts xmlns="{_NS}">'
        f'<Page imageFilename="p.jpg" imageWidth="1000" imageHeight="1400">{regions}</Page></PcGts>',
        encoding="utf-8",
    )
    return path


def _line(line_id: str, points: str, text: str, inner: str = "") -> str:
    return (
        f'<TextLine id="{line_id}"><Coords points="{points}"/>{inner}'
        f"<TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>"
    )


@pytest.fixture
def transkribus_page(tmp_path: Path) -> Path:
    """Region types and reading order as Transkribus writes them into ``@custom``, regions out of file order."""
    return _page(
        tmp_path,
        '<TextRegion id="r3" custom="readingOrder {index:2;}">'
        + _line("b1", "220,200 900,200 900,250 220,250", "den fjarde maj")
        + '</TextRegion><TextRegion id="r1" custom="readingOrder {index:0;} structure {type:page-number;}">'
        + _line("n1", "850,60 900,60 900,100 850,100", "97.")
        + '</TextRegion><TextRegion id="r2" custom="readingOrder {index:1;} structure {type:marginalia;}">'
        + _line("m1", "20,200 180,200 180,240 20,240", "No 12")
        + '</TextRegion><TableRegion id="t1"><TableCell id="c1">'
        + _line("c1l", "220,300 400,300 400,340 220,340", "Anders Persson")
        + "</TableCell></TableRegion>",
    )


def test_replay_reads_regions_in_transkribus_reading_order_with_their_roles(transkribus_page: Path) -> None:
    layout = load_xml_layout(transkribus_page)

    assert [(box.text, box.region, box.block) for box in layout.boxes] == [
        ("97.", "page-number", "r1"),
        ("No 12", "marginalia", "r2"),
        ("den fjarde maj", "body", "r3"),
        ("Anders Persson", "table", "c1"),
    ]


def test_replay_takes_a_lines_own_outline_not_its_last_words(tmp_path: Path) -> None:
    words = (
        '<Word id="w1"><Coords points="100,200 300,200 300,250 100,250"/></Word>'
        '<Word id="w2"><Coords points="600,200 900,200 900,250 600,250"/></Word>'
    )
    page = _page(
        tmp_path,
        '<TextRegion id="r">' + _line("l", "100,190 900,190 900,260 100,260", "ett tva", words) + "</TextRegion>",
    )

    box = load_xml_layout(page).boxes[0]

    assert (box.x, box.y, box.w, box.h) == (100.0, 190.0, 800.0, 70.0)


def test_replay_takes_the_baseline_from_the_lines_baseline_points(tmp_path: Path) -> None:
    baseline = '<Baseline points="100,240 900,244"/>'
    page = _page(
        tmp_path,
        '<TextRegion id="r">' + _line("l", "100,190 900,190 900,260 100,260", "ett", baseline) + "</TextRegion>",
    )

    assert load_xml_layout(page).boxes[0].baseline_y == 242.0


def test_replay_reads_an_alto_page(tmp_path: Path) -> None:
    alto = tmp_path / "page.alto.xml"
    alto.write_text(
        '<?xml version="1.0"?><alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"><Layout>'
        '<Page WIDTH="6000" HEIGHT="5000"><PrintSpace><TextBlock ID="b1">'
        '<TextLine HPOS="100" VPOS="200" WIDTH="300" HEIGHT="50" BASELINE="100,240 400,240">'
        '<String CONTENT="Hej"/><String CONTENT="du"/></TextLine>'
        "</TextBlock></PrintSpace></Page></Layout></alto>",
        encoding="utf-8",
    )

    layout = load_xml_layout(alto)

    assert (layout.width, layout.height) == (6000, 5000)
    assert [(b.text, b.x, b.y, b.w, b.h, b.baseline_y, b.block) for b in layout.boxes] == [
        ("Hej du", 100.0, 200.0, 300.0, 50.0, 240.0, "b1")
    ]
