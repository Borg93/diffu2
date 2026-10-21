"""Label export: PAGE XML that validates against PRImA's schema, ALTO in the htrflow shape, both carrying the same
regions, reading order, baselines and tags."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import xmlschema

from diffu_page.labels import LineLabel, PageLabels, export, to_alto, to_page_xml

_PAGE_SCHEMA = Path(__file__).parent / "schemas" / "page2013.xsd"


@pytest.fixture(scope="module")
def prima_schema() -> xmlschema.XMLSchema:
    """The PRImA PAGE XSD, compiled once for the module (compilation is the expensive part)."""
    return xmlschema.XMLSchema(str(_PAGE_SCHEMA))


def _labels() -> PageLabels:
    """A page number, a running-text line with a baseline that read back badly, and a table cell."""
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
            LineLabel(text="12 dr", polygon=[(10, 300), (90, 300), (90, 330), (10, 330)], region="table", order=2),
        ],
    )


def _local(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def test_page_xml_validates_against_the_prima_schema(prima_schema: xmlschema.XMLSchema) -> None:
    xml = to_page_xml(_labels()).replace("/2019-07-15", "/2013-07-15")  # the vendored XSD is 2013's (same model)

    errors = list(prima_schema.iter_errors(xml))

    assert errors == []


def test_page_xml_lists_the_regions_in_reading_order_with_their_page_types() -> None:
    root = ET.fromstring(to_page_xml(_labels()))

    order = [e.get("regionRef") for e in root.iter() if _local(e) == "RegionRefIndexed"]
    types = {e.get("id"): e.get("type") for e in root.iter() if _local(e) == "TextRegion"}

    assert order == ["page-number", "body", "table"]
    assert types == {"page-number": "page-number", "body": "paragraph", "table": "other"}


def test_page_xml_tags_an_unclear_line_the_way_transkribus_does() -> None:
    root = ET.fromstring(to_page_xml(_labels()))

    customs = {e.get("id"): e.get("custom") for e in root.iter() if _local(e) == "TextLine"}

    assert customs == {"page-number_l0": None, "body_l1": "structure {type:unclear;}", "table_l2": None}


def test_alto_writes_each_line_as_one_string_with_its_text() -> None:
    root = ET.fromstring(to_alto(_labels()))

    contents = [e.get("CONTENT") for e in root.iter() if _local(e) == "String"]

    assert contents == ["2.", "Hej varlden", "12 dr"]


def test_alto_matches_the_htrflow_shape() -> None:
    root = ET.fromstring(to_alto(_labels()))
    names = {_local(e) for e in root.iter()}
    # the elements real Riksarkivet/htrflow ALTO carries that a bbox-only export would miss
    assert {"Processing", "ReadingOrder", "OrderedGroup", "ElementRef", "Shape", "Polygon"} <= names
    page = next(e for e in root.iter() if _local(e) == "Page")
    assert page.get("PC") == "1.0"
    refs = [e.get("REF") for e in root.iter() if _local(e) == "ElementRef"]
    assert refs == [e.get("ID") for e in root.iter() if _local(e) == "TextLine"]


def test_page_and_alto_carry_baselines() -> None:
    labels = _labels()
    page = ET.fromstring(to_page_xml(labels))
    alto = ET.fromstring(to_alto(labels))
    assert [e.get("points") for e in page.iter() if _local(e) == "Baseline"] == ["10,125 110,125"]
    assert [e.get("BASELINE") for e in alto.iter() if _local(e) == "TextLine"] == [None, "10,125 110,125", None]


def test_export_refuses_an_unknown_format() -> None:
    with pytest.raises(ValueError, match="unknown format 'json'"):
        export(_labels(), "json")  # pyrefly: ignore[bad-argument-type]
