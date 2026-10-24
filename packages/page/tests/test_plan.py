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
from diffu_page.captions.prompt import LineRecord
from diffu_page.captions.provenance import Provenance
from diffu_page.captions.records import CropRecord
from diffu_page.captions.restyle import ReferenceCaption
from diffu_page.crops.geometry import CropSettings, crops_for_page, regions_of
from diffu_page.data.style_pairs import Pair, Scan, captions, describe
from diffu_page.layout import load_xml_layout
from diffu_page.plan import CropPlan, PageStyle, Reference, plan_page

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


def test_plan_keeps_every_row_of_a_table_whole_in_its_windows(table_page: Path) -> None:
    layout = load_xml_layout(table_page)
    cells_per_row = Counter(box.cell.row for box in layout.boxes if box.cell is not None)

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

    assert {line.text for plan in plans for line in plan.lines} == {box.text for box in layout.boxes}


def test_plan_records_where_each_lines_baseline_lies_on_its_crop(body_and_notes_page: Path) -> None:
    layout = load_xml_layout(body_and_notes_page)
    on_page = {box.text: box.baseline_y for box in layout.boxes}

    planned = plan_page(layout, STYLE)

    assert [[line.baseline for line in plan.lines] for plan in planned] == [
        [round(1000 * (on_page[line.text] - plan.rect[1]) / (plan.rect[3] - plan.rect[1])) for line in plan.lines]
        for plan in planned
    ]


def test_crop_plan_refuses_a_reading_order_that_is_not_an_order_of_its_lines() -> None:
    lines = [LineRecord(id=f"Sv_Text_{k}", text="", role="", bbox=[0, 0, 1000, 1000]) for k in (1, 2)]

    with pytest.raises(ValidationError, match="reading order"):
        CropPlan(rect=(0, 0, 10, 10), width=10, height=10, caption="", lines=lines, reading_order=[1, 1])


REFERENCE_LINES = [[100, 50, 300, 950], [500, 50, 700, 950]]


def test_plan_captions_a_page_drawn_from_a_reference_as_its_training_pairs_were_captioned(
    body_and_notes_page: Path, make_record: Callable[..., CropRecord], tmp_path: Path
) -> None:
    layout = load_xml_layout(body_and_notes_page)
    image = tmp_path / "reference.png"
    Image.linear_gradient("L").rotate(30).convert("RGB").save(image)
    reference = Reference(image=image, caption=ReferenceCaption.DESCRIBED, lines=REFERENCE_LINES)
    crops = crops_for_page(regions_of(layout), (layout.width, layout.height), CropSettings())
    records = [make_record(crop, f"{k}.jpg") for k, crop in enumerate(crops)]  # the page's crops as trained on
    style = PageStyle(provenance=records[0].provenance, look=records[0].look)

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
