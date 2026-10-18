"""Tests for faithful ALTO/PAGE replay — load_xml_layout carries real region semantics (not a flat "xml"
region) and turns PAGE SeparatorRegions into furniture rules, and the single-leaf split re-origins rules."""

from __future__ import annotations

from pathlib import Path

from diffu_page.layout import load_xml_layout

_PAGE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Page imageFilename="p.jpg" imageWidth="1000" imageHeight="1400">
    <TextRegion id="m1" type="marginalia">
      <TextLine id="ml1"><Coords points="20,100 180,100 180,140 20,140"/>
        <TextEquiv><Unicode>No 12</Unicode></TextEquiv></TextLine>
    </TextRegion>
    <TextRegion id="b1" type="paragraph">
      <TextLine id="bl1"><Coords points="220,100 900,100 900,150 220,150"/>
        <TextEquiv><Unicode>den fjarde maj artonhundra attiotva</Unicode></TextEquiv></TextLine>
    </TextRegion>
    <TableRegion id="t1">
      <TableCell id="c1"><TextLine id="cl1"><Coords points="220,300 400,300 400,340 220,340"/>
        <TextEquiv><Unicode>Anders Persson</Unicode></TextEquiv></TextLine></TableCell>
    </TableRegion>
    <SeparatorRegion id="s1"><Coords points="210,90 214,90 214,360 210,360"/></SeparatorRegion>
    <SeparatorRegion id="s2"><Coords points="220,280 900,280 900,284 220,284"/></SeparatorRegion>
  </Page>
</PcGts>
"""


def test_replay_carries_region_semantics_and_rules(tmp_path: Path):
    xml = tmp_path / "p.xml"
    xml.write_text(_PAGE_XML, encoding="utf-8")
    layout = load_xml_layout(str(xml), single_page=False)

    regions = {b.region for b in layout.boxes}
    assert "xml" not in regions, "replay must map real region types, not flatten to 'xml'"
    assert regions == {"marginalia", "body", "table"}, regions
    # both separators become rules; one vertical (s1) and one horizontal (s2)
    assert len(layout.rules) == 2, [f"{r.x1},{r.y1}->{r.x2},{r.y2}" for r in layout.rules]
    vertical = [r for r in layout.rules if r.x1 == r.x2]
    horizontal = [r for r in layout.rules if r.y1 == r.y2]
    assert len(vertical) == 1 and len(horizontal) == 1, layout.rules


def test_single_leaf_split_reorigins_rules(tmp_path: Path):
    xml = tmp_path / "p.xml"
    xml.write_text(_PAGE_XML, encoding="utf-8")
    leaf = load_xml_layout(str(xml), single_page=True)  # splits (left cluster vs right body) + re-origins
    # every carried rule is re-origined onto the (narrowed) leaf canvas — none left off-page by the split
    assert leaf.rules, "the kept leaf should still carry its in-range separator rule(s)"
    assert all(0 <= r.x1 <= leaf.spec.width and 0 <= r.x2 <= leaf.spec.width for r in leaf.rules), (
        [f"{r.x1}->{r.x2} on w={leaf.spec.width}" for r in leaf.rules]
    )
