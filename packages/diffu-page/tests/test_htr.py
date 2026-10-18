"""GT extraction for the HTR readability check.

Pure-XML parsing only (no htrflow), so it runs in the normal test env. Guards the bug where adding
``<Word>`` elements to PAGE-XML made ``ground_truth_lines`` return only each line's FIRST WORD (the
first nested ``<Unicode>``), which silently inflated every CER number.
"""

from __future__ import annotations

from diffu_page.htr import ground_truth_lines, ground_truth_lines_with_boxes
from diffu_page.labels import LineLabel, PageLabels, WordLabel, to_page_xml


def test_ground_truth_lines_reads_the_full_line_not_the_first_word(tmp_path):
    labels = PageLabels(
        width=400,
        height=200,
        lines=[
            LineLabel(
                text="Göteborgs poliskammare den fjärde",
                polygon=[(10, 10), (300, 10), (300, 40), (10, 40)],
                region="body",
                order=0,
                words=[
                    WordLabel(text="Göteborgs", polygon=[(10, 10), (90, 10), (90, 40), (10, 40)]),
                    WordLabel(text="poliskammare", polygon=[(95, 10), (200, 10), (200, 40), (95, 40)]),
                ],
            )
        ],
    )
    path = tmp_path / "p.page.xml"
    path.write_text(to_page_xml(labels), encoding="utf-8")
    assert ground_truth_lines(str(path)) == ["Göteborgs poliskammare den fjärde"]


def test_ground_truth_lines_with_boxes_returns_full_line_text_and_its_bbox(tmp_path):
    labels = PageLabels(
        width=400,
        height=200,
        lines=[
            LineLabel(
                text="Anhållen för fylleri",
                polygon=[(10, 20), (300, 20), (300, 60), (10, 60)],
                region="body",
                order=0,
            )
        ],
    )
    path = tmp_path / "p.page.xml"
    path.write_text(to_page_xml(labels), encoding="utf-8")
    assert ground_truth_lines_with_boxes(str(path)) == [("Anhållen för fylleri", (10, 20, 300, 60))]
