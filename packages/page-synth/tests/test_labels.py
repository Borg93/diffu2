"""Labels: every export format round-trips; ALTO carries word boxes; unknown format raises."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import xmlschema

from page_synth.labels import EXPORTERS, LineLabel, PageLabels, WordLabel, export, to_alto, to_page_xml
from page_synth.layout import LineBox, PageSpec, build_layout

_PAGE_SCHEMA = Path(__file__).parent / "schemas" / "page2013.xsd"


@pytest.fixture(scope="module")
def prima_schema() -> xmlschema.XMLSchema:
    """The PRImA PAGE XSD, compiled once for the module (compilation is the expensive part)."""
    return xmlschema.XMLSchema(str(_PAGE_SCHEMA))


def _page_xml_for(template: str) -> str:
    """A fully-furnished page (header/marginalia/page-number + printed cells + edits + words) as PAGE-XML,
    in the 2013 namespace so it validates against the vendored 2013 XSD (identical content model to 2019)."""
    spec = PageSpec(
        width=1000,
        height=1300,
        template=template,
        header_lines=["Protokoll"],
        page_number="3.",
        marginalia=["N:o"],
        insertion_prob=0.3,
        deletion_prob=0.3,
    )
    layout = build_layout(spec, ["alpha beta gamma", "ett tva tre fyra"] * 9, seed=2)
    lines = [_slot_label(box, order) for order, box in enumerate(layout.boxes)]
    xml = to_page_xml(PageLabels(width=1000, height=1300, image="p.png", lines=lines))
    return xml.replace("/2019-07-15", "/2013-07-15")


def _slot_label(box: LineBox, order: int) -> LineLabel:
    """A layout box as a label: its slot as the polygon, its baseline across the slot."""
    x0, y0, x1, y1 = box.x, box.y, box.x + box.w, box.y + box.h
    return LineLabel(
        text=box.text,
        polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        baseline=[(x0, box.baseline_y), (x1, box.baseline_y)],
        region=box.region,
        order=order,
        tag=box.tag,
        printed=box.printed,
    )


def _labels() -> PageLabels:
    return PageLabels(
        width=800,
        height=1000,
        image="p.png",
        lines=[
            LineLabel(
                text="Hej varlden",
                polygon=[(10, 10), (110, 10), (110, 40), (10, 40)],
                region="body",
                order=0,
                words=[
                    WordLabel(text="Hej", polygon=[(10, 10), (50, 10), (50, 40), (10, 40)]),
                    WordLabel(text="varlden", polygon=[(55, 10), (110, 10), (110, 40), (55, 40)]),
                ],
            )
        ],
    )


@pytest.mark.parametrize("fmt", list(EXPORTERS))
def test_export_round_trips(fmt: str) -> None:
    out = export(_labels(), fmt)
    if fmt == "json":
        assert json.loads(out)["lines"][0]["text"] == "Hej varlden"
    else:
        ET.fromstring(out)  # well-formed XML


def test_alto_has_word_strings() -> None:
    root = ET.fromstring(to_alto(_labels()))
    strings = [e for e in root.iter() if e.tag.endswith("String")]
    assert len(strings) == 2
    assert strings[0].get("CONTENT") == "Hej"


def test_export_unknown_format_raises() -> None:
    with pytest.raises(ValueError):
        export(_labels(), "nope")


# every template + furniture/printed/edits/words must validate against the real PRImA content model
@pytest.mark.parametrize("template", ["single_column", "multi_column", "semi_table", "columns", "key_value", "spread"])
def test_page_xml_validates_against_the_prima_schema(template: str, prima_schema: xmlschema.XMLSchema) -> None:
    errors = list(prima_schema.iter_errors(_page_xml_for(template)))
    assert not errors, f"{template} PAGE-XML fails the PRImA schema: {errors[0] if errors else ''}"


def test_page_xml_marks_production_and_edit_styles() -> None:
    labels = PageLabels(
        width=400,
        height=300,
        lines=[
            LineLabel(
                text="x",
                polygon=[(10, 10), (40, 10), (40, 30), (10, 30)],
                region="row_number",
                order=0,
                printed=True,
            ),
            LineLabel(text="y", polygon=[(10, 40), (40, 40), (40, 60), (10, 60)], region="body", order=1, tag="del"),
        ],
    )
    xml = to_page_xml(labels)
    assert 'production="printed"' in xml and 'production="handwritten-cursive"' in xml
    assert '<TextStyle strikethrough="true"/>' in xml


def test_alto_matches_the_htrflow_shape() -> None:
    root = ET.fromstring(to_alto(_labels()))
    names = {e.tag.rsplit("}", 1)[-1] for e in root.iter()}
    # the elements real Riksarkivet/htrflow ALTO carries that a bbox-only export would miss
    assert {"Processing", "ReadingOrder", "OrderedGroup", "ElementRef", "Shape", "Polygon"} <= names
    assert root.get("{http://www.w3.org/2001/XMLSchema-instance}schemaLocation"), "missing schemaLocation"
    page = next(e for e in root.iter() if e.tag.endswith("Page"))
    assert page.get("PC") == "1.0", "ground-truth page confidence"
    poly = next(e for e in root.iter() if e.tag.endswith("Polygon"))
    assert poly.get("POINTS") == "10,10 110,10 110,40 10,40", "line polygon must be the tight outline"
    ref = next(e for e in root.iter() if e.tag.endswith("ElementRef"))
    line_ids = {e.get("ID") for e in root.iter() if e.tag.endswith("TextLine")}
    assert ref.get("REF") in line_ids, "reading-order ElementRef must point at a real TextLine"


def _multi_region() -> PageLabels:
    return PageLabels(
        width=800,
        height=1000,
        lines=[
            LineLabel(text="2.", polygon=[(700, 10), (740, 10), (740, 40), (700, 40)], region="page_number", order=0),
            LineLabel(
                text="Hej varlden",
                polygon=[(10, 100), (110, 100), (110, 130), (10, 130)],
                baseline=[(10, 125), (110, 125)],
                region="body",
                order=1,
            ),
        ],
    )


def test_page_xml_emits_reading_order_and_typed_regions() -> None:
    root = ET.fromstring(to_page_xml(_multi_region()))
    refs = [e for e in root.iter() if e.tag.endswith("RegionRefIndexed")]
    assert [r.get("regionRef") for r in refs] == ["page_number", "body"], "reading order missing/scrambled"
    types = {e.get("id"): e.get("type") for e in root.iter() if e.tag.endswith("TextRegion")}
    assert types == {"page_number": "page-number", "body": "paragraph"}


def test_page_xml_tags_edits_with_a_custom_structure_attr() -> None:
    labels = PageLabels(
        width=400,
        height=300,
        lines=[
            LineLabel(
                text="struck",
                polygon=[(10, 10), (40, 10), (40, 30), (10, 30)],
                region="body",
                order=0,
                tag="del",
            ),
            LineLabel(
                text="added",
                polygon=[(10, 5), (30, 5), (30, 15), (10, 15)],
                region="insertion",
                order=1,
                tag="add",
            ),
        ],
    )
    xml = to_page_xml(labels)
    assert 'custom="structure {type:del;}"' in xml
    assert 'custom="structure {type:add;}"' in xml


def test_page_and_alto_carry_baselines() -> None:
    labels = _multi_region()
    page = ET.fromstring(to_page_xml(labels))
    assert any(e.tag.endswith("Baseline") and e.get("points") == "10,125 110,125" for e in page.iter())
    alto = ET.fromstring(to_alto(labels))
    body_line = next(e for e in alto.iter() if e.tag.endswith("TextLine") and e.get("ID") == "body_l1")
    assert body_line.get("BASELINE") == "10,125 110,125"
