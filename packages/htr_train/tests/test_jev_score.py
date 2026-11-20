"""Jev's scoring of a scan's labels: the lines read from the ALTO file, cut as its training lines were, gathered."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from diffu_htr_train.jev.score import HEIGHT, Line, alto_lines, line_crop, match_score, scan_score

ALTO_FILE = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"><Layout><Page WIDTH="400" HEIGHT="300"><PrintSpace>
<TextBlock ID="b">
<TextLine ID="a" HPOS="20" VPOS="20" WIDTH="200" HEIGHT="40">
  <Shape><Polygon POINTS="20,20 220,20 220,60 20,60"/></Shape>
  <String CONTENT="Anno" HPOS="20" VPOS="20" WIDTH="90" HEIGHT="40"/><String CONTENT="1689" HPOS="120" VPOS="20" WIDTH="100" HEIGHT="40"/>
</TextLine>
<TextLine ID="empty" HPOS="20" VPOS="80" WIDTH="200" HEIGHT="40"><String CONTENT=""/></TextLine>
<TextLine ID="speck" HPOS="20" VPOS="140" WIDTH="200" HEIGHT="4"><String CONTENT="x"/></TextLine>
<TextLine ID="boxed" HPOS="20" VPOS="200" WIDTH="100" HEIGHT="40"><String CONTENT="Datum"/></TextLine>
</TextBlock></PrintSpace></Page></Layout></alto>
"""


def test_alto_lines_keeps_the_lines_that_hold_text_and_room_for_writing(tmp_path: Path) -> None:
    labels = tmp_path / "scan.xml"
    labels.write_text(ALTO_FILE, "utf-8")

    lines = alto_lines(labels)

    assert [line.text for line in lines] == ["Anno 1689", "Datum"]
    assert lines[0].box == (20, 20, 220, 60)
    assert lines[1].box == (20, 200, 120, 240)  # no polygon in the file: the line's own box


def test_line_crop_is_white_outside_the_outline_and_as_high_as_the_training_crops() -> None:
    scan = Image.new("RGB", (400, 300), (0, 0, 0))
    triangle = Line(text="x", outline=((100, 100), (300, 100), (100, 200)))

    crop = line_crop(scan, triangle)

    assert crop.size == (96, HEIGHT)
    assert crop.getpixel((5, 5)) == (0, 0, 0)  # inside the outline: the scan's own black
    assert crop.getpixel((90, 44)) == (255, 255, 255)  # outside it


def test_scan_score_gathers_the_lines_and_a_scan_without_lines_has_no_score() -> None:
    scored = scan_score("scan.jpg", [0.9, 0.1, 0.8, 0.6])

    assert (scored.lines, scored.low, scored.worst) == (4, 0.25, 0.1)
    assert scored.mean is not None
    assert abs(scored.mean - 0.6) < 1e-9
    assert scan_score("blank.jpg", []).mean is None


def test_match_score_is_the_probability_of_the_match_logit() -> None:
    assert match_score(0.0, 0.0) == 0.5
    assert match_score(2.0, -2.0) < 0.02
    assert match_score(-2.0, 2.0) > 0.98
