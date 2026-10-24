"""Reading a real page's layout from PAGE or ALTO XML: lines with their own outline, baseline, region role, block and
reading order."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from diffu_page.crops.geometry import regions_of
from diffu_page.layout import CellRef, Production, Rule, load_xml_layout

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
        + '</TextRegion><TableRegion id="t1"><TableCell id="c1" row="0" col="0">'
        + _line("c1l", "220,300 400,300 400,340 220,340", "Anders Persson")
        + "</TableCell></TableRegion>",
    )


def test_replay_reads_regions_in_transkribus_reading_order_with_their_roles(transkribus_page: Path) -> None:
    layout = load_xml_layout(transkribus_page)

    assert [(line.text, line.kind, line.block) for line in layout.lines] == [
        ("97.", "page-number", "r1"),
        ("No 12", "marginalia", "r2"),
        ("den fjarde maj", "body", "r3"),
        ("Anders Persson", "table", "c1"),
    ]


@pytest.fixture
def form_page(tmp_path: Path) -> Path:
    """A printed form filled in by hand: a printed heading, a printed region with one handwritten line, a table in
    Transkribus' TableCell form and one in PAGE 2019's TableCellRole form, and two ruled lines."""
    return _page(
        tmp_path,
        '<TextRegion id="title" production="printed">'
        + _line("t1", "300,40 700,40 700,90 300,90", "Flyttningsbetyg")
        + '</TextRegion><TextRegion id="field" production="printed">'
        + _line("f1", "100,120 200,120 200,160 100,160", "Att")
        + '<TextLine id="f2" production="handwritten-cursive"><Coords points="220,120 600,120 600,160 220,160"/>'
        "<TextEquiv><Unicode>Anna Lisa</Unicode></TextEquiv></TextLine>"
        '</TextRegion><TableRegion id="list" production="printed"><TableCell id="c00" row="0" col="0">'
        + _line("h1", "100,300 300,300 300,340 100,340", "Namn")
        + '</TableCell><TableCell id="c10" row="1" col="0" production="handwritten-cursive">'
        + _line("n1", "100,360 300,360 300,400 100,400", "Anders Persson")
        + '</TableCell></TableRegion><TableRegion id="ledger"><TextRegion id="r1c2">'
        '<Coords points="400,500 600,500 600,560 400,560"/><Roles><TableCellRole rowIndex="1" columnIndex="2"/></Roles>'
        + _line("s1", "420,510 580,510 580,550 420,550", "12 dr")
        + "</TextRegion></TableRegion>"
        '<SeparatorRegion id="s1"><Coords points="100,350 600,350 600,352 100,352"/></SeparatorRegion>'
        '<SeparatorRegion id="s2"><Coords points="310,300 312,300 312,410 310,410"/></SeparatorRegion>',
    )


def test_replay_keeps_each_lines_production_and_table_cell(form_page: Path) -> None:
    layout = load_xml_layout(form_page)

    assert [(line.text, line.production, line.cell) for line in layout.lines] == [
        ("Flyttningsbetyg", Production.PRINTED, None),
        ("Att", Production.PRINTED, None),
        ("Anna Lisa", Production.HANDWRITTEN, None),
        ("Namn", Production.PRINTED, CellRef(table="list", row=0, column=0)),
        ("Anders Persson", Production.HANDWRITTEN, CellRef(table="list", row=1, column=0)),
        ("12 dr", Production.HANDWRITTEN, CellRef(table="ledger", row=1, column=2)),
    ]


def test_replay_reads_ruled_lines_from_separator_regions(form_page: Path) -> None:
    assert load_xml_layout(form_page).rules == [
        Rule(x0=100.0, y0=350.0, x1=600.0, y1=352.0),
        Rule(x0=310.0, y0=300.0, x1=312.0, y1=410.0),
    ]


def test_replay_refuses_a_production_page_does_not_define(tmp_path: Path) -> None:
    page = _page(
        tmp_path,
        '<TextRegion id="r" production="engraved">' + _line("l", "1,1 9,1 9,9 1,9", "ett") + "</TextRegion>",
    )

    with pytest.raises(ValueError, match="'engraved'"):
        load_xml_layout(page)


def test_replay_takes_a_lines_own_outline_not_its_last_words(tmp_path: Path) -> None:
    words = (
        '<Word id="w1"><Coords points="100,200 300,200 300,250 100,250"/></Word>'
        '<Word id="w2"><Coords points="600,200 900,200 900,250 600,250"/></Word>'
    )
    page = _page(
        tmp_path,
        '<TextRegion id="r">' + _line("l", "100,190 900,190 900,260 100,260", "ett tva", words) + "</TextRegion>",
    )

    line = load_xml_layout(page).lines[0]

    assert line.box == (100.0, 190.0, 900.0, 260.0)


def test_replay_takes_the_baseline_from_the_lines_baseline_points(tmp_path: Path) -> None:
    baseline = '<Baseline points="100,240 900,244"/>'
    page = _page(
        tmp_path,
        '<TextRegion id="r">' + _line("l", "100,190 900,190 900,260 100,260", "ett", baseline) + "</TextRegion>",
    )

    assert load_xml_layout(page).lines[0].baseline_y == 242.0


def _alto(tmp_path: Path, blocks: str, description: str = "") -> Path:
    path = tmp_path / "page.alto.xml"
    path.write_text(
        f'<?xml version="1.0"?><alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">{description}<Layout>'
        f'<Page WIDTH="6000" HEIGHT="5000"><PrintSpace>{blocks}</PrintSpace></Page></Layout></alto>',
        encoding="utf-8",
    )
    return path


def test_replay_reads_an_alto_page(tmp_path: Path) -> None:
    alto = _alto(
        tmp_path,
        '<TextBlock ID="b1"><TextLine HPOS="100" VPOS="200" WIDTH="300" HEIGHT="50" BASELINE="100,240 400,240">'
        '<String CONTENT="Hej"/><String CONTENT="du"/></TextLine></TextBlock>',
    )

    layout = load_xml_layout(alto)

    assert (layout.width, layout.height) == (6000, 5000)
    assert [(line.text, line.box, line.baseline_y, line.block) for line in layout.lines] == [
        ("Hej du", (100.0, 200.0, 400.0, 250.0), 240.0, "b1")
    ]


def _id_less_alto(tmp_path: Path) -> Path:
    line = '<TextLine HPOS="{x}" VPOS="200" WIDTH="300" HEIGHT="50"><String CONTENT="{text}"/></TextLine>'
    blocks = (f"<TextBlock>{line.format(x=x, text=text)}</TextBlock>" for x, text in ((100, "left"), (3000, "right")))
    return _alto(tmp_path, "".join(blocks))


def _id_less_page(tmp_path: Path) -> Path:
    return _page(
        tmp_path,
        "<TextRegion>" + _line("l", "100,200 400,200 400,250 100,250", "left") + "</TextRegion>"
        "<TextRegion>" + _line("r", "600,200 900,200 900,250 600,250", "right") + "</TextRegion>",
    )


@pytest.mark.parametrize("page", [pytest.param(_id_less_alto, id="alto"), pytest.param(_id_less_page, id="page")])
def test_replay_keeps_blocks_without_an_id_apart(tmp_path: Path, page: Callable[[Path], Path]) -> None:
    layout = load_xml_layout(page(tmp_path))

    regions = regions_of(layout)

    assert [[line.text for line in region.lines] for region in regions] == [["left"], ["right"]]


def test_replay_refuses_an_alto_page_measured_in_other_units_than_pixels(tmp_path: Path) -> None:
    alto = _alto(
        tmp_path,
        '<TextBlock ID="b1"><TextLine HPOS="10" VPOS="20" WIDTH="30" HEIGHT="5"><String CONTENT="Hej"/>'
        "</TextLine></TextBlock>",
        description="<Description><MeasurementUnit>mm10</MeasurementUnit></Description>",
    )

    with pytest.raises(ValueError, match="measured in mm10, not pixels"):
        load_xml_layout(alto)


def test_replay_names_the_file_it_cannot_parse(tmp_path: Path) -> None:
    page = tmp_path / "0007.xml"
    page.write_text('<?xml version="1.0"?><PcGts', encoding="utf-8")  # cut off mid-write

    with pytest.raises(ValueError, match="0007.xml"):
        load_xml_layout(page)
