"""A ground-truth template marked from its scan (diffu_page.marking): the lines its PAGE file says nothing of read as
printed or handwritten, its own ruled lines kept, and a page planned from it captioned with its print and its rules."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from diffu_page.captions.prompt import PageStyle
from diffu_page.jobs import PageJob, crop_jobs
from diffu_page.layout import Production, Rule, load_xml_layout
from diffu_page.marking import PrintHandClassifier, marked, template_scan

if TYPE_CHECKING:
    from conftest import ScannedFormMaker

PRINTED, HANDWRITTEN = Production.PRINTED, Production.HANDWRITTEN


def test_marked_reads_the_lines_a_template_says_nothing_of_as_printed_when_sure_and_keeps_what_it_says(
    make_scanned_form: ScannedFormMaker, ink_meter: PrintHandClassifier
) -> None:
    form = make_scanned_form()

    lines = marked(load_xml_layout(form), template_scan(form), classifier=ink_meter).lines

    assert [(line.text, line.production) for line in lines] == [
        ("Mantalslängd 1765", PRINTED),  # inked as densely as type is
        *((f"entry {k} of the roll", HANDWRITTEN) for k in range(6)),  # scribbled
        ("Summa", PRINTED),  # scribbled, but printed as its PAGE file says
        ("Anders Persson", HANDWRITTEN),  # inked densely, but handwritten as its PAGE file says
        ("Anno", HANDWRITTEN),  # printed faintly: 70 % sure is not sure enough
    ]


def test_marked_keeps_the_ruled_lines_a_template_has_and_finds_none(
    make_scanned_form: ScannedFormMaker, ink_meter: PrintHandClassifier
) -> None:
    form = make_scanned_form(
        '<SeparatorRegion id="s"><Coords points="100,139 1100,139 1100,141 100,141"/></SeparatorRegion>'
    )

    rules = marked(load_xml_layout(form), template_scan(form), classifier=ink_meter).rules

    assert rules == [Rule(x0=100, y0=139, x1=1100, y1=141)]  # the scan's other three ruled lines are not added


def _rows(caption: str) -> dict[str, str]:
    """A caption's layout rows, id -> desc."""
    return {row["id"]: row["desc"] for row in json.loads(caption[caption.index(" [{") + 1 :])}


def test_crop_jobs_of_a_template_marked_from_its_scan_caption_its_print_as_print_and_its_rules(
    make_scanned_form: ScannedFormMaker, ink_meter: PrintHandClassifier, page_style: PageStyle, page_text: str
) -> None:
    page = PageJob(template=make_scanned_form(), text=page_text, style=page_style, page_pixels=250_000)

    (job,) = crop_jobs(page, 0, print_hand=ink_meter)

    rows = _rows(job.plan.caption)
    heading = next(line.id for line in job.plan.lines if line.text == "Mantalslängd 1765")
    assert rows[heading] == 'Printed heading reading "Mantalslängd 1765", in a serif typeface in black printer\'s ink.'
    assert [desc for row_id, desc in rows.items() if row_id.startswith("rule_")] == [
        "A straight horizontal ruled line.",
        "A straight horizontal ruled line.",
        "A straight vertical ruled line.",
        "A straight vertical ruled line.",
    ]
