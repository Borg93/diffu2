"""Layout: templates validate clean, size invariants hold, geometry is relative, ALTO replays."""

import pytest
from pydantic import ValidationError

from page_synth.layout import Column, PageSpec, build_layout, load_xml_layout, validate_layout

TEMPLATES = ["single_column", "multi_column", "semi_table", "columns", "key_value"]
BODY = ["Goteborg poliskammare den fjarde maj om natten i staden", "Anhallen for fylleri", "kort"] * 8
KV = ["Namn: Anna Andersson", "Datum: 5 Juli 1744", "Ort: Stockholm"] * 4


@pytest.mark.parametrize("template", TEMPLATES)
def test_templates_validate_clean(template: str) -> None:
    text = KV if template == "key_value" else BODY
    cols = [Column(x_frac=0.0, w_frac=0.45), Column(x_frac=0.55, w_frac=0.45)] if template == "columns" else []
    spec = PageSpec(template=template, columns=cols)
    assert validate_layout(build_layout(spec, text, seed=1)) == []


@pytest.mark.parametrize("hs,ms", [(1.0, 1.0), (1.8, 0.6), (1.3, 0.85)])
def test_size_invariants(hs: float, ms: float) -> None:
    spec = PageSpec(
        template="single_column",
        header_lines=["H"],
        header_scale=hs,
        marginalia=["m"],
        marg_scale=ms,
        line_height=48,
    )
    boxes = build_layout(spec, BODY, seed=1).boxes
    body = next(b.h for b in boxes if b.region == "body")
    assert all(b.h >= body for b in boxes if b.region == "header")  # header never smaller than body
    assert all(b.h <= body for b in boxes if b.region == "marginalia")  # marginalia never bigger


def test_pydantic_rejects_invalid_scales() -> None:
    with pytest.raises(ValidationError):
        PageSpec(header_scale=0.5)  # header < body forbidden
    with pytest.raises(ValidationError):
        PageSpec(marg_scale=1.5)  # marginalia > body forbidden


def test_relative_geometry_scales_with_page_size() -> None:
    assert PageSpec(width=2000, height=2800).margin > PageSpec(width=1000, height=1400).margin
    assert PageSpec(width=1000, height=1400).margin == int(0.06 * 1000)


def test_load_xml_layout_reads_alto(tmp_path) -> None:
    alto = (
        '<?xml version="1.0"?><alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"><Layout>'
        '<Page WIDTH="6000" HEIGHT="5000"><PrintSpace><TextBlock>'
        '<TextLine HPOS="100" VPOS="200" WIDTH="300" HEIGHT="50">'
        '<String HPOS="100" VPOS="200" WIDTH="300" HEIGHT="50" CONTENT="Hej"/></TextLine>'
        "</TextBlock></PrintSpace></Page></Layout></alto>"
    )
    p = tmp_path / "x.alto.xml"
    p.write_text(alto)
    lay = load_xml_layout(str(p))
    assert (lay.spec.width, lay.spec.height) == (6000, 5000)  # real page dims used
    assert len(lay.boxes) == 1 and lay.boxes[0].text == "Hej"
