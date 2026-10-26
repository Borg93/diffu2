"""Crop planning: a real page is cut where the training builder cut it, its lines read block by block, and a page
drawn from a reference captioned exactly as the reference pairs were captioned for training."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import LineRecord, PageStyle
from diffu_page.captions.provenance import Provenance
from diffu_page.captions.records import CropRecord
from diffu_page.captions.restyle import ReferenceCaption
from diffu_page.crops.geometry import CropSettings, crops_for_page, regions_of
from diffu_page.data.style_pairs import Pair, Scan, captions, describe
from diffu_page.layout import Layout, TextLine, load_xml_layout
from diffu_page.plan import CropPlan, Reference, plan_page, plan_whole_page

STYLE = PageStyle(
    provenance=Provenance(title="the archive", years="1700"),
    look=ScanLook(greyscale=False, paper="cream", ink="brown"),
)


def test_plan_reads_the_body_block_before_its_margin_notes(body_and_notes_page: Path) -> None:
    layout = load_xml_layout(body_and_notes_page)

    plan = plan_page(layout, STYLE)[0]

    numbered = [line.text for line in plan.lines]  # the caption's numbering: top to bottom, notes interleaved
    body = [text for text in numbered if text.startswith("body")]
    assert [plan.lines[i].text for i in plan.reading_order] == [*body, "note 0", "note 1"]


def _rows(plan: CropPlan) -> dict[str, str]:
    """A plan's layout rows, id -> desc."""
    return {row["id"]: row["desc"] for row in json.loads(plan.caption[plan.caption.index(" [{") + 1 :])}


def test_plan_whole_page_draws_every_line_of_the_page_at_the_pixels_asked_for(body_and_notes_page: Path) -> None:
    layout = load_xml_layout(body_and_notes_page)

    plan = plan_whole_page(layout, STYLE, pixels=1_000_000)

    assert (plan.width, plan.height) == (1024, 976)  # 2000 x 1900 px at 0.513, each side on the 16-px grid
    assert sorted(line.text for line in plan.lines) == sorted(line.text for line in layout.lines)
    assert [plan.lines[i].text for i in plan.reading_order][-2:] == ["note 0", "note 1"]


def test_plan_whole_page_keeps_every_ruled_line_of_the_page(table_page: Path) -> None:
    layout = load_xml_layout(table_page)

    plan = plan_whole_page(layout, STYLE, pixels=1_000_000)

    assert len(plan.rules) == len(layout.rules) == 5


def test_plan_whole_page_refuses_a_page_with_no_line_of_text() -> None:
    with pytest.raises(ValueError, match="no line of text"):
        plan_whole_page(Layout(width=1000, height=1000), STYLE, pixels=1_000_000)


def test_plan_keeps_every_row_of_a_table_whole_in_its_windows(table_page: Path) -> None:
    layout = load_xml_layout(table_page)
    cells_per_row = Counter(line.cell.row for line in layout.lines if line.cell is not None)

    windows = [
        Counter(line.cell.row for line in plan.lines if line.cell is not None) for plan in plan_page(layout, STYLE)
    ]

    assert set().union(*windows) == set(cells_per_row)
    assert [row for window in windows for row, cells in window.items() if cells != cells_per_row[row]] == []


def test_plan_captions_a_table_cell_as_printed_or_handwritten(table_page: Path) -> None:
    plan = plan_page(load_xml_layout(table_page), STYLE)[0]

    row_of = {line.text: line.id for line in plan.lines}
    rows = _rows(plan)
    assert (rows[row_of["Namn"]], rows[row_of["r1c1"]]) == (
        'Printed text reading "Namn" in a table cell, in a serif typeface in black printer\'s ink.',
        'Handwritten line of text reading "r1c1" in a table cell, cursive script in brown ink.',
    )


def test_plan_gives_every_ruled_line_crossing_a_window_a_row(table_page: Path) -> None:
    plans = plan_page(load_xml_layout(table_page), STYLE)

    assert plans
    assert [[desc for row_id, desc in _rows(plan).items() if row_id.startswith("rule_")] for plan in plans] == [
        ["A straight vertical ruled line."] * 5 for _ in plans
    ]


def test_plan_plans_a_form_whose_every_line_is_its_own_region(certificate_page: Path) -> None:
    layout = load_xml_layout(certificate_page)

    plans = plan_page(layout, STYLE)

    assert {line.text for plan in plans for line in plan.lines} == {line.text for line in layout.lines}


def test_plan_captions_a_form_drawn_from_a_reference_keeping_its_print_and_rules(
    certificate_page: Path, tmp_path: Path
) -> None:
    image = tmp_path / "reference.png"
    Image.linear_gradient("L").convert("RGB").save(image)
    layout = load_xml_layout(certificate_page)
    named = plan_page(layout, STYLE)[0].caption

    caption = plan_page(layout, STYLE, reference=Reference(image=image, caption=ReferenceCaption.PLAIN))[0].caption

    paragraph, rows = caption[: caption.index(" [{")], json.loads(caption[caption.index(" [{") + 1 :])
    unstyled = {
        row["id"]: row["desc"]
        for row in json.loads(named[named.index(" [{") + 1 :])
        if row["desc"].startswith(("Printed", "A straight"))
    }
    assert paragraph == (
        "Create a flat, evenly lit archival scan of a new page printed and filled in by hand in the hand, ink and "
        "paper of <ref_image_0>. The paper <paper_1> carries 5 lines of cursive handwriting and 6 printed lines in "
        "black printer's ink <Sv_Text_1> <Sv_Text_2> <Sv_Text_4> <Sv_Text_6> <Sv_Text_8> <Sv_Text_10>; the main text "
        "runs from top to bottom: <Sv_Text_3> <Sv_Text_5> <Sv_Text_7> <Sv_Text_9> <Sv_Text_11>. A heading <Sv_Text_1> "
        "sits at the top center. Five straight ruled lines cross the page: <rule_1> <rule_2> <rule_3> <rule_4> "
        "<rule_5>."
    )
    assert {row["id"]: row["desc"] for row in rows if row["id"] in unstyled} == unstyled
    assert {
        row["desc"].rsplit(", ", 1)[-1]
        for row in rows
        if row["id"].startswith("Sv_Text_") and row["id"] not in unstyled
    } == {"cursive script in the reference's hand and ink."}


def test_plan_records_where_each_lines_baseline_lies_on_its_crop(body_and_notes_page: Path) -> None:
    layout = load_xml_layout(body_and_notes_page)
    on_page = {line.text: line.baseline_y for line in layout.lines}

    planned = plan_page(layout, STYLE)

    assert [[line.baseline for line in plan.lines] for plan in planned] == [
        [round(1000 * (on_page[line.text] - plan.rect[1]) / (plan.rect[3] - plan.rect[1])) for line in plan.lines]
        for plan in planned
    ]


def test_crop_plan_refuses_a_reading_order_that_is_not_an_order_of_its_lines() -> None:
    lines = [LineRecord(id=f"Sv_Text_{k}", text="", role="", bbox=[0, 0, 1000, 1000]) for k in (1, 2)]

    with pytest.raises(ValidationError, match="reading order"):
        CropPlan(rect=(0, 0, 10, 10), width=10, height=10, caption="", lines=lines, reading_order=[1, 1])


@pytest.mark.parametrize("role", [pytest.param("marginal", id="no-such-kind"), pytest.param("table", id="not-a-role")])
def test_crop_plan_refuses_a_line_whose_role_is_no_role_on_the_page(role: str) -> None:
    line = {"id": "Sv_Text_1", "text": "", "role": role, "bbox": [0, 0, 1000, 1000]}
    plan = {"rect": (0, 0, 10, 10), "width": 10, "height": 10, "caption": "", "lines": [line], "reading_order": [0]}

    with pytest.raises(ValidationError, match=f"role '{role}' is not one of"):
        CropPlan.model_validate(plan)


def _block(name: str, x0: float, x1: float) -> list[TextLine]:
    return [
        TextLine(text=f"{name} line {k}", box=(x0, 60 + 64 * k, x1, 110 + 64 * k), baseline_y=100 + 64 * k, block=name)
        for k in range(12)
    ]


def test_crops_keep_a_region_against_the_page_edge() -> None:
    # 600 px wide plus half a pitch a side counts as a column (664 >= 640), but the page edge leaves 632
    layout = Layout(width=1600, height=1200, lines=_block("A", 0, 600) + _block("B", 740, 1540))

    crops = crops_for_page(regions_of(layout), (layout.width, layout.height), CropSettings())

    assert {line.block for crop in crops for line in crop.lines} == {"A", "B"}


REFERENCE_LINES = [[100, 50, 300, 950], [500, 50, 700, 950]]


def test_plan_names_a_reference_whose_hand_cannot_be_described(body_and_notes_page: Path, tmp_path: Path) -> None:
    blank = tmp_path / "blank.png"
    Image.new("RGB", (64, 64), "white").save(blank)
    reference = Reference(image=blank, caption=ReferenceCaption.DESCRIBED, lines=REFERENCE_LINES)

    with pytest.raises(ValueError, match=f"^{blank}: .*cannot be described"):
        plan_page(load_xml_layout(body_and_notes_page), STYLE, reference=reference)


def test_plan_captions_a_page_drawn_from_a_reference_as_its_training_pairs_were_captioned(
    body_and_notes_page: Path, make_record: Callable[..., CropRecord], tmp_path: Path
) -> None:
    layout = load_xml_layout(body_and_notes_page)
    image = tmp_path / "reference.png"
    Image.linear_gradient("L").rotate(30).convert("RGB").save(image)
    reference = Reference(image=image, caption=ReferenceCaption.DESCRIBED, lines=REFERENCE_LINES)
    crops = crops_for_page(regions_of(layout), (layout.width, layout.height), CropSettings())
    records = [make_record(crop, f"{k}.jpg") for k, crop in enumerate(crops)]  # the page's crops as trained on
    style = records[0].style

    planned = plan_page(layout, style, reference=reference)

    lines = [LineRecord(id=f"Sv_Text_{k}", text="", role="", bbox=box) for k, box in enumerate(REFERENCE_LINES, 1)]
    reference_crop = Scan(record=make_record(crops[0], "reference.png").model_copy(update={"lines": lines}), path=image)
    training, _ = captions(
        ReferenceCaption.DESCRIBED, named_fraction=0.0, seed=0, looks={image: describe(reference_crop)}
    )
    assert [plan.caption for plan in planned] == [
        training(Pair(target=Scan(record=record, path=tmp_path / record.file_name), reference=reference_crop))
        for record in records
    ]
