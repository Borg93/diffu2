"""Label export: PAGE XML that validates against PRImA's schema, ALTO in the htrflow shape, both carrying the same
regions, reading order, baselines and tags."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import xmlschema
from pydantic import ValidationError

from diffu_page.blocks.ground_truth import line_labels, rule_rects
from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import LineRecord
from diffu_page.captions.provenance import Provenance
from diffu_page.labels import LineLabel, PageLabels, export, to_alto, to_page_xml
from diffu_page.layout import CellRef, Production, load_xml_layout
from diffu_page.plan import CropPlan, PageStyle, plan_page

# PRImA's PAGE 2019-07-15 schema (Apache-2.0, github.com/PRImA-Research-Lab/PAGE-XML,
# PAGE-release/gts/pagecontent/2019-07-15/pagecontent.xsd): the version the exports declare
_PAGE_SCHEMA = Path(__file__).parent / "schemas" / "page2019.xsd"
# The Library of Congress's ALTO 4.4 schema (CC BY-SA 4.0, loc.gov/standards/alto/v4/alto-4-4.xsd, sha256 5564fd29) and
# the METS XLink schema it imports (loc.gov/standards/xlink/xlink.xsd, sha256 f1f5bb60), read from here, not the network
_ALTO_SCHEMA = Path(__file__).parent / "schemas" / "alto-4-4.xsd"
_XLINK_SCHEMA = Path(__file__).parent / "schemas" / "xlink.xsd"


@pytest.fixture(scope="module")
def prima_schema() -> xmlschema.XMLSchema:
    """The PRImA PAGE XSD, compiled once for the module (compilation is the expensive part)."""
    return xmlschema.XMLSchema(str(_PAGE_SCHEMA))


@pytest.fixture(scope="module")
def alto_schema() -> xmlschema.XMLSchema:
    """The ALTO 4.4 XSD with its XLink import, compiled once for the module."""
    return xmlschema.XMLSchema(str(_ALTO_SCHEMA), locations=[("http://www.w3.org/1999/xlink", str(_XLINK_SCHEMA))])


def _labels() -> PageLabels:
    """A page number, a running-text line with a baseline that read back badly, and a margin note."""
    return PageLabels(
        width=800,
        height=1000,
        image="crop.jpg",
        lines=[
            LineLabel(text="2.", polygon=[(700, 10), (740, 10), (740, 40), (700, 40)], region="page-number", order=0),
            LineLabel(
                text="Hej varlden",
                polygon=[(10, 100), (110, 100), (110, 130), (10, 130)],
                baseline=[(10, 125), (110, 125)],
                order=1,
                tag="unclear",
            ),
            LineLabel(text="12 dr", polygon=[(10, 300), (90, 300), (90, 330), (10, 330)], region="marginalia", order=2),
        ],
    )


def _form_labels() -> PageLabels:
    """A printed heading, a table of a printed header cell and a handwritten cell, and two ruled lines."""
    return PageLabels(
        width=800,
        height=1000,
        image="form.jpg",
        lines=[
            LineLabel(
                text="Förteckning",
                polygon=[(100, 20), (700, 20), (700, 60), (100, 60)],
                region="heading",
                order=0,
                production=Production.PRINTED,
            ),
            LineLabel(
                text="Namn",
                polygon=[(100, 100), (300, 100), (300, 140), (100, 140)],
                region="table",
                block="table|c00",
                order=1,
                production=Production.PRINTED,
                cell=CellRef(table="register", row=0, column=0),
            ),
            LineLabel(
                text="Anders Persson",
                polygon=[(100, 160), (300, 160), (300, 200), (100, 200)],
                region="table",
                block="table|c10",
                order=2,
                cell=CellRef(table="register", row=1, column=0),
            ),
        ],
        rules=[(90, 150, 710, 152), (310, 90, 312, 210)],
    )


def _local(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def test_line_labels_put_each_line_on_its_baseline_in_reading_order_tagging_lines_over_the_cer_limit() -> None:
    plan = CropPlan(
        rect=(0, 0, 1000, 200),
        width=1000,
        height=200,
        caption="",
        lines=[
            # its slot cut off by the crop's top edge: the baseline is not 0.7 of the way down what is left of it
            LineRecord(id="Sv_Text_1", text="Anno 1715", role="", bbox=[0, 100, 105, 900], baseline=60),
            # a record written before baselines were kept: 0.7 of the way down its slot
            LineRecord(id="Sv_Text_2", text="den 4 maj", role="", bbox=[300, 100, 450, 900]),
            LineRecord(id="Sv_Text_3", text="Stockholm", role="marginalia", bbox=[500, 0, 650, 90], baseline=605),
        ],
        reading_order=[2, 0, 1],
    )

    labels = line_labels(plan, [0.25, 0.26, 0.0], max_line_cer=0.25)

    assert labels == [
        LineLabel(
            text="Anno 1715",
            polygon=[(100, 0), (900, 0), (900, 21), (100, 21)],
            baseline=[(100, 12), (900, 12)],
            order=1,
        ),
        LineLabel(
            text="den 4 maj",
            polygon=[(100, 60), (900, 60), (900, 90), (100, 90)],
            baseline=[(100, 81), (900, 81)],
            order=2,
            tag="unclear",
        ),
        LineLabel(
            text="Stockholm",
            polygon=[(0, 100), (90, 100), (90, 130), (0, 130)],
            baseline=[(0, 121), (90, 121)],
            region="marginalia",
            order=0,
        ),
    ]


@pytest.mark.parametrize("labels", [_labels(), _form_labels()], ids=["text", "form"])
def test_page_xml_validates_against_the_prima_schema(prima_schema: xmlschema.XMLSchema, labels: PageLabels) -> None:
    errors = list(prima_schema.iter_errors(to_page_xml(labels)))

    assert errors == []


@pytest.mark.parametrize("labels", [_labels(), _form_labels()], ids=["text", "form"])
def test_alto_validates_against_the_alto_4_4_schema(alto_schema: xmlschema.XMLSchema, labels: PageLabels) -> None:
    errors = list(alto_schema.iter_errors(to_alto(labels)))

    assert errors == []


type ReadLine = tuple[str, str, str, str, str | None]  # text, region kind, production, tag, baseline points


def _read_page(xml: str) -> list[ReadLine]:
    """Each line of a PAGE export, in document order, as its labels said it."""
    lines: list[ReadLine] = []
    for region in (e for e in ET.fromstring(xml).iter() if _local(e) == "TextRegion"):
        in_cell = any(_local(e) == "TableCellRole" for e in region.iter())
        kind = "table" if in_cell else {"paragraph": "body"}.get(region.get("type", ""), region.get("type", ""))
        for line in (e for e in region if _local(e) == "TextLine"):
            baseline = next((e.get("points") for e in line if _local(e) == "Baseline"), None)
            text = next(e.text or "" for e in line.iter() if _local(e) == "Unicode")
            production = "printed" if line.get("production") == "printed" else "handwritten"
            tag = "unclear" if "unclear {" in line.get("custom", "") else ""
            lines.append((text, kind, production, tag, baseline))
    return lines


def _read_alto(xml: str) -> list[ReadLine]:
    """Each line of an ALTO export, in document order, as its labels said it: kinds and tags read through TAGREFS."""
    root = ET.fromstring(xml)
    tags = {e.get("ID", ""): (e.get("TYPE", ""), e.get("LABEL", "")) for e in root.iter() if _local(e).endswith("Tag")}
    lines: list[ReadLine] = []
    for block in (e for e in root.iter() if _local(e) == "TextBlock"):
        (_, kind), *_ = (tags[ref] for ref in block.get("TAGREFS", "").split())
        for line in (e for e in block if _local(e) == "TextLine"):
            refs = dict(tags[ref] for ref in line.get("TAGREFS", "").split())
            text = next(e.get("CONTENT", "") for e in line if _local(e) == "String")
            lines.append((text, kind, refs.get("production", ""), refs.get("readback", ""), line.get("BASELINE")))
    return lines


@pytest.mark.parametrize("labels", [_labels(), _form_labels()], ids=["text", "form"])
@pytest.mark.parametrize(
    ("export_as", "read"), [(to_page_xml, _read_page), (to_alto, _read_alto)], ids=["page", "alto"]
)
def test_both_formats_carry_each_lines_text_kind_production_tag_and_baseline_in_reading_order(
    labels: PageLabels, export_as: Callable[[PageLabels], str], read: Callable[[str], list[ReadLine]]
) -> None:
    exported = export_as(labels)

    assert read(exported) == [
        (
            line.text,
            line.region,
            line.production.value,
            line.tag,
            " ".join(f"{x},{y}" for x, y in line.baseline) or None,
        )
        for line in sorted(labels.lines, key=lambda line: line.order)
    ]


def test_page_xml_of_a_planned_table_crop_reads_back_as_the_same_table(table_page: Path, tmp_path: Path) -> None:
    style = PageStyle(
        provenance=Provenance(title="the archive", years=""), look=ScanLook(greyscale=False, paper="cream", ink="brown")
    )
    plan = plan_page(load_xml_layout(table_page), style)[0]
    labels = PageLabels(
        width=plan.width,
        height=plan.height,
        lines=line_labels(plan, [0.0] * len(plan.lines), 1.0),
        rules=rule_rects(plan),
    )
    exported = tmp_path / "labels.xml"
    exported.write_text(to_page_xml(labels), encoding="utf-8")

    read = load_xml_layout(exported)

    def cells(lines: list) -> list[tuple]:
        return sorted((line.text, line.production, line.cell and (line.cell.row, line.cell.column)) for line in lines)

    assert (cells(read.lines), [(r.x0, r.y0, r.x1, r.y1) for r in read.rules]) == (
        cells(plan.lines),
        [tuple(float(v) for v in rect) for rect in labels.rules],
    )


def test_page_xml_lists_the_regions_in_reading_order_with_their_page_types() -> None:
    root = ET.fromstring(to_page_xml(_labels()))

    order = [e.get("regionRef") for e in root.iter() if _local(e) == "RegionRefIndexed"]
    types = {e.get("id"): e.get("type") for e in root.iter() if _local(e) == "TextRegion"}

    assert order == ["page-number", "body", "marginalia"]
    assert types == {"page-number": "page-number", "body": "paragraph", "marginalia": "marginalia"}


def test_page_xml_of_a_planned_crop_keeps_each_body_block_its_own_region(wide_and_narrow_body_page: Path) -> None:
    style = PageStyle(
        provenance=Provenance(title="the archive", years=""), look=ScanLook(greyscale=False, paper="cream", ink="brown")
    )
    plan = plan_page(load_xml_layout(wide_and_narrow_body_page), style)[0]
    labels = PageLabels(width=plan.width, height=plan.height, lines=line_labels(plan, [0.0] * len(plan.lines), 1.0))

    root = ET.fromstring(to_page_xml(labels))

    regions = {
        region.get("id"): [line.get("id") for line in region if _local(line) == "TextLine"]
        for region in root.iter()
        if _local(region) == "TextRegion"
    }
    assert regions == {
        "body_1": ["body_1_l0", "body_1_l1", "body_1_l2"],
        "body_2": ["body_2_l3", "body_2_l4", "body_2_l5"],
    }


def test_page_xml_tags_an_unclear_line_the_way_transkribus_does() -> None:
    root = ET.fromstring(to_page_xml(_labels()))

    customs = {e.get("id"): e.get("custom") for e in root.iter() if _local(e) == "TextLine"}

    assert customs == {  # Transkribus: each line's place in its region, and an unclear text range over the whole line
        "page-number_l0": "readingOrder {index:0;}",
        "body_l1": "readingOrder {index:0;} unclear {offset:0; length:11;}",
        "marginalia_l2": "readingOrder {index:0;}",
    }


def test_alto_writes_each_ruled_line_as_a_graphical_element() -> None:
    root = ET.fromstring(to_alto(_form_labels()))

    rules = [
        (e.get("HPOS"), e.get("VPOS"), e.get("WIDTH"), e.get("HEIGHT"))
        for e in root.iter()
        if _local(e) == "GraphicalElement"
    ]

    assert rules == [("90", "150", "620", "2"), ("310", "90", "2", "120")]


def test_alto_matches_the_htrflow_shape() -> None:
    root = ET.fromstring(to_alto(_labels()))
    names = {_local(e) for e in root.iter()}
    # the elements real Riksarkivet/htrflow ALTO carries that a bbox-only export would miss
    assert {"Processing", "ReadingOrder", "OrderedGroup", "ElementRef", "Shape", "Polygon"} <= names
    page = next(e for e in root.iter() if _local(e) == "Page")
    assert page.get("PC") == "1.0"
    refs = [e.get("REF") for e in root.iter() if _local(e) == "ElementRef"]
    assert refs == [e.get("ID") for e in root.iter() if _local(e) == "TextLine"]


@pytest.mark.parametrize(
    ("region", "cell"),
    [
        pytest.param("table", None, id="table-without-cell"),
        pytest.param("body", CellRef(table="t", row=0, column=0), id="cell-outside-table"),
    ],
)
def test_line_label_refuses_a_table_line_without_its_cell_and_a_cell_outside_a_table(
    region: str, cell: CellRef | None
) -> None:
    with pytest.raises(ValidationError, match="in the table region exactly when it has a table cell"):
        LineLabel(text="12 dr", polygon=[(10, 300), (90, 300), (90, 330), (10, 330)], region=region, order=0, cell=cell)


def test_export_refuses_an_unknown_format() -> None:
    with pytest.raises(ValueError, match="unknown format 'json'"):
        unknown: Any = "json"  # not a LabelFormat: the point of the test
        export(_labels(), unknown)
