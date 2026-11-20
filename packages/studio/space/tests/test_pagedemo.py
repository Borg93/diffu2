"""The demo's own code on CPU: a document as the layout the pipeline draws, a sample's line pitch, and the page
planned from both as the model was trained."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from diffu_moss.train.pairs import SampleSize
from diffu_page.draw.sample import sized_sample
from diffu_page.page.layout import LineKind, Production
from pagedemo.document import Box, BoxKind, Document, page_layout, page_size
from pagedemo.planning import planned
from pagedemo.sample import line_pitch, sample_lines
from pagedemo.templates import TEMPLATES

LETTER = Document(
    aspect=0.7,
    lines=25,
    boxes=[
        Box(id="heading", kind=BoxKind.HEADING, rect=(300, 80, 700, 120), text="Protocoll"),
        Box(
            id="body",
            rect=(150, 200, 850, 400),
            text="Anno 1748 den 12 Martii hölls ordinarie Ting med Allmogen\nNärvarande Nämnden",
        ),
        Box(id="note", kind=BoxKind.NOTE, rect=(20, 200, 130, 280), text="N:o 4"),
        Box(id="number", kind=BoxKind.NUMBER, rect=(850, 30, 950, 70), text="17"),
        Box(id="form", rect=(150, 600, 850, 640), text="Underskrift", printed=True),
        Box(id="line", kind=BoxKind.RULE, rect=(150, 660, 850, 664)),
    ],
)


def _striped(size: tuple[int, int], pitch: int, *, margin: int = 40) -> Image.Image:
    """A page of dark strokes at one line pitch on light paper: each line a band of broken strokes, a third of a pitch
    tall."""
    width, height = size
    page = np.full((height, width), 225, dtype=np.uint8)
    generator = np.random.default_rng(0)
    for top in range(margin, height - margin - pitch, pitch):
        strokes = generator.random(width - 2 * margin) < 0.6
        page[top + pitch // 3 : top + 2 * pitch // 3, margin : width - margin][:, strokes] = 40
    return Image.fromarray(page).convert("RGB")


def test_a_documents_boxes_become_line_slots_of_one_pitch_holding_their_text() -> None:
    page = page_layout(LETTER, 2.0)

    layout = page.layout
    width, height = page_size(0.7, 2.0)
    pitch = height / 25
    assert (layout.width, layout.height) == (width, height) and width % 16 == 0 and height % 16 == 0
    assert 1.9e6 < width * height < 2.1e6
    assert all(line.height == pytest.approx(pitch) for line in layout.lines)
    assert all(line.baseline_y == pytest.approx(line.box[1] + 0.7 * pitch) for line in layout.lines)
    body = [line for line in layout.lines if line.block == "body"]
    assert [line.text for line in body][-1] == "Närvarande Nämnden"  # the author's line break starts a new line
    assert " ".join(line.text for line in body[:-1]) == "Anno 1748 den 12 Martii hölls ordinarie Ting med Allmogen"
    assert body[0].box[1] == pytest.approx(200 * height / 1000)  # stacked from the box's top edge
    kinds = {line.block: line.kind for line in layout.lines}
    assert kinds == {
        "number": LineKind.PAGE_NUMBER,
        "heading": LineKind.HEADING,
        "body": LineKind.BODY,
        "note": LineKind.MARGINALIA,
        "form": LineKind.BODY,
    }
    assert [line.production for line in layout.lines if line.block == "form"] == [Production.PRINTED]
    assert [line.block for line in layout.lines][:2] == ["number", "heading"]  # read from the top of the page down
    (rule,) = layout.rules
    assert rule.y1 - rule.y0 == pytest.approx(2.0) and rule.x1 - rule.x0 == pytest.approx(0.7 * width)
    assert page.notes == []


def test_a_box_without_room_or_without_text_is_told_of_and_a_page_without_text_refused() -> None:
    crowded = Document(
        lines=30,
        boxes=[
            Box(id="small", rect=(100, 100, 400, 133), text="ett två tre fyra fem sex sju åtta nio tio elva tolv"),
            Box(id="empty", rect=(100, 300, 900, 400)),
        ],
    )

    page = page_layout(crowded, 2.0)

    (line,) = page.layout.lines
    kept = len(line.text.split())
    assert 0 < kept < 12
    assert page.notes == [f"small had no room for its last {12 - kept} words", "empty holds no text and was left out"]
    with pytest.raises(ValueError, match="no box holds a text"):
        page_layout(Document(boxes=[Box(id="line", kind=BoxKind.RULE, rect=(0, 10, 900, 14))]), 2.0)


def test_an_editors_value_that_is_no_document_is_refused_in_words() -> None:
    assert Document.parse(LETTER.model_dump_json()) == LETTER
    assert Document.parse(None) == Document()
    with pytest.raises(ValueError, match=r"boxes\.0\.rect: Value error, the box .* does not lie on the page"):
        Document.parse(json.dumps({"boxes": [{"id": "a", "rect": [10, 10, 5, 2000]}]}))
    with pytest.raises(ValueError, match="two boxes share a name"):
        Document.parse(json.dumps({"boxes": [{"id": "a", "rect": [0, 0, 9, 9]}, {"id": "a", "rect": [0, 0, 9, 9]}]}))


def test_a_pages_line_pitch_is_read_from_its_rows_and_a_blank_page_has_none() -> None:
    assert line_pitch(_striped((900, 1300), 52)) == pytest.approx(52, rel=0.05)
    assert line_pitch(_striped((1600, 1100), 90)) == pytest.approx(90, rel=0.05)
    assert line_pitch(Image.new("RGB", (800, 1200), (225, 220, 205))) is None


def test_a_page_is_planned_from_its_sample_as_the_model_was_trained() -> None:
    sample = _striped((1000, 1400), 70)

    page = planned(LETTER, sample, megapixels=2.0, zoom=1.0)

    plan = page.plan
    prompt = json.loads(page.prompt)
    assert prompt["summary"]["caption"].startswith(
        "Create a flat, evenly lit archival scan of a new page printed and filled in by hand in the hand, ink and "
        "paper of the sample image."
    )
    assert "archive" not in page.prompt and "cream" not in page.prompt  # the look is the sample's alone
    paper, *items = prompt["items"]
    assert paper == {
        "label": "object",
        "caption": "The paper of the sample image.",
        "bbox": [-plan.height // 2, -plan.width // 2, plan.height // 2, plan.width // 2],
    }
    texts = [item for item in items if item["label"] == "text"]
    assert {item["text"] for item in texts} >= {"Protocoll", "Närvarande Nämnden", "N:o 4", "17", "Underskrift"}
    heading = next(item for item in texts if item["text"] == "Protocoll")
    pitch = plan.height / 25
    top, left, bottom, right = heading["bbox"]  # the box drawn: one pitch tall from its top edge, its own width
    assert top + plan.height // 2 == pytest.approx(80 * plan.height / 1000, abs=1.5)
    assert bottom - top == pytest.approx(pitch, abs=2)
    assert (left + plan.width // 2, right + plan.width // 2) == pytest.approx(
        (300 * plan.width / 1000, 700 * plan.width / 1000), abs=2
    )
    assert heading["caption"].startswith("Heading")
    assert any(item["caption"].startswith("Printed text") for item in texts)
    assert [item["caption"] for item in items if item["label"] == "object"] == ["A straight horizontal ruled line."]
    # the sample is cut at the page's line pitch: its lines as tall as the page's slots
    assert not plan.whole_page and page.sample_lines
    sized = sized_sample(sample, SampleSize.LEGIBLE, plan=plan, lines=page.sample_lines)
    assert line_pitch(sized) == pytest.approx(pitch, rel=0.08)
    assert sized.width <= plan.width and sized.height <= plan.height
    assert sized.width % 16 == 0 and sized.height % 16 == 0


def test_a_zoomed_sample_shows_its_writing_larger_and_an_unmeasured_one_is_sized_as_a_page() -> None:
    sample = _striped((1000, 1400), 70)
    blank = Image.new("RGB", (900, 1300), (225, 220, 205))

    zoomed = planned(LETTER, sample, megapixels=2.0, zoom=2.0)
    unmeasured = planned(LETTER, blank, megapixels=2.0, zoom=1.0)

    pitch = zoomed.plan.height / 25
    sized = sized_sample(sample, SampleSize.LEGIBLE, plan=zoomed.plan, lines=zoomed.sample_lines)
    assert line_pitch(sized) == pytest.approx(2 * pitch, rel=0.08)
    assert unmeasured.plan.whole_page and unmeasured.sample_lines == []
    assert unmeasured.notes == [
        "the sample's line height could not be measured: it is shown to the model as a whole page"
    ]
    whole = sized_sample(blank, SampleSize.LEGIBLE, plan=unmeasured.plan, lines=[])
    assert whole.width * whole.height == pytest.approx(unmeasured.plan.width * unmeasured.plan.height, rel=0.05)
    with pytest.raises(ValueError, match="positive"):
        sample_lines(sample, 70, zoom=0)


def test_every_template_fits_its_boxes_and_fills_them() -> None:
    assert list(TEMPLATES)[-1] == "Blank page" and TEMPLATES["Blank page"].boxes == []
    for name, document in TEMPLATES.items():
        if not document.boxes:
            continue
        page = page_layout(document, 2.0)
        assert page.notes == [], name
        pitch = page.layout.height / document.lines
        for box in document.boxes:
            if box.kind is BoxKind.RULE:
                continue
            taken = sum(line.block == box.id for line in page.layout.lines)
            assert taken == round((box.rect[3] - box.rect[1]) * page.layout.height / 1000 / pitch), (name, box.id)
