"""PAGE/ALTO XML parsing -> LineRecords (no torch)."""

from __future__ import annotations

from PIL import Image

from diffu.data.dataset import build_manifest, load_manifest
from diffu.data.parse import parse_file

_PAGE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="p1.jpg" imageWidth="100" imageHeight="50">
    <TextRegion>
      <TextLine>
        <Coords points="5,5 50,5 50,20 5,20"/>
        <TextEquiv><Unicode>Göteborg</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>"""

_ALTO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v2#">
  <Description><sourceImageInformation><fileName>scan1.tif</fileName></sourceImageInformation></Description>
  <Layout><Page><PrintSpace>
    <TextLine HPOS="10" VPOS="20" WIDTH="40" HEIGHT="15">
      <String CONTENT="Smörgås"/>
      <String CONTENT="bord"/>
    </TextLine>
  </PrintSpace></Page></Layout>
</alto>"""


def test_parse_page_xml(tmp_path):
    f = tmp_path / "page.xml"
    f.write_text(_PAGE_XML, encoding="utf-8")
    [rec] = parse_file(str(f))  # exactly one TextLine
    assert rec.text == "Göteborg"
    assert rec.bbox == (5.0, 5.0, 50.0, 20.0)
    assert rec.source_id == "p1"  # derived from imageFilename


def test_parse_alto_xml(tmp_path):
    f = tmp_path / "alto.xml"
    f.write_text(_ALTO_XML, encoding="utf-8")
    [rec] = parse_file(str(f))
    assert rec.text == "Smörgås bord"  # words joined with a space
    assert rec.bbox == (10.0, 20.0, 50.0, 35.0)  # HPOS,VPOS -> +WIDTH,+HEIGHT
    assert rec.source_id == "scan1"


def test_parse_file_autodetects_format(tmp_path):
    (tmp_path / "a.xml").write_text(_PAGE_XML, encoding="utf-8")
    (tmp_path / "b.xml").write_text(_ALTO_XML, encoding="utf-8")
    assert parse_file(str(tmp_path / "a.xml"))[0].source_id == "p1"
    assert parse_file(str(tmp_path / "b.xml"))[0].source_id == "scan1"


def test_build_manifest_end_to_end(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (100, 50), "white").save(img_dir / "p1.jpg")
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    (xml_dir / "p1.xml").write_text(_PAGE_XML, encoding="utf-8")
    out = tmp_path / "out"

    n = build_manifest(str(xml_dir), str(img_dir), str(out), height=32)

    assert n == 1
    rows = load_manifest(str(out / "manifest.jsonl"))
    assert rows[0]["text"] == "Göteborg"
    assert rows[0]["writer_id"] == "p1"
    assert (out / "lines" / "0000000.png").exists()
