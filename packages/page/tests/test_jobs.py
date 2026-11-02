"""A run's pages planned into crop jobs: cut by the plan steps exactly as the page is planned (captioned for its
reference when it has one, the reference's line boxes kept beside it), each named and seeded by its page's place in
the run or by its id. A page that plans to
nothing is refused, not dropped from the run: test_pipeline's generate test reads the refusal from the summary. A job
of a two-page opening becomes a job per page, named apart. A page's layout is its template's or given inline, and its
text new or its own."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from diffu_page.captions.prompt import PageStyle
from diffu_page.draw.jobs import PageJob, crop_jobs, read_page_jobs, single_pages
from diffu_page.draw.pipeline import PagePlanBlocks
from diffu_page.draw.plan import Reference, plan_page
from diffu_page.page.layout import Layout, LineKind, PageSide, TextLine, load_xml_layout, refill
from diffu_page.page.marking import PrintHandClassifier

ONE_SOURCE = "give the page's template or its layout: exactly one of them"
NO_TEXT_TO_KEEP = "lines[0] holds no text, and no text is given that would be written into it"
BOXES = [(600.0, 240.0 + 100 * i, 1900.0, 320.0 + 100 * i) for i in range(14)]  # body_and_notes_page's body lines
BOXES_LAYOUT = {"width": 2000, "height": 2400, "lines": [{"box": box} for box in BOXES]}  # as a run file gives it


def _boxes_layout(texts: list[str] | None = None) -> Layout:
    """BOXES_LAYOUT, as boxes alone or each holding its text of ``texts``."""
    if texts is None:
        return Layout.model_validate(BOXES_LAYOUT)
    lines = [{"box": box, "text": text} for box, text in zip(BOXES, texts, strict=True)]
    return Layout.model_validate({**BOXES_LAYOUT, "lines": lines})


@pytest.mark.parametrize(
    "reference",
    [
        pytest.param(None, id="named-style"),
        pytest.param(Reference(image=Path("ref.jpg"), lines=[(100, 80, 160, 920)]), id="reference-and-its-lines"),
    ],
)
def test_crop_jobs_plan_page_two_of_a_run_into_named_and_seeded_crops(
    body_and_notes_page: Path, reference: Reference | None, page_style: PageStyle, page_text: str
) -> None:
    page = PageJob(template=body_and_notes_page, text=page_text, style=page_style, seed=5, reference=reference)

    jobs = crop_jobs(page, 2)

    plans = plan_page(refill(load_xml_layout(body_and_notes_page), page_text), page_style, reference=reference)
    image, lines = (reference.image, reference.lines) if reference else (None, [])
    assert [(job.name, job.seed, job.plan, job.reference, job.reference_lines) for job in jobs] == [
        (f"002_page_{k}", 2005 + k, plan, image, lines) for k, plan in enumerate(plans)
    ]


def test_crop_jobs_plan_a_page_with_page_pixels_whole_as_one_named_and_seeded_crop(
    body_and_notes_page: Path, page_style: PageStyle, page_text: str
) -> None:
    page = PageJob(template=body_and_notes_page, text=page_text, style=page_style, seed=5, page_pixels=1_000_000)

    jobs = crop_jobs(page, 2)

    assert [(job.name, job.seed, job.plan.width, job.plan.height) for job in jobs] == [("002_page_0", 2005, 912, 1088)]


def test_crop_jobs_name_and_seed_a_page_with_an_id_alike_wherever_it_stands_in_a_run(
    body_and_notes_page: Path, page_style: PageStyle, page_text: str
) -> None:
    page = PageJob(template=body_and_notes_page, text=page_text, style=page_style, seed=5, id="vol7_p12")

    early, late = crop_jobs(page, 2), crop_jobs(page, 7)

    assert [(job.name, job.seed) for job in late] == [(job.name, job.seed) for job in early]
    assert [job.name for job in early] == [f"vol7_p12_{k}" for k in range(len(early))]


def test_read_page_jobs_names_the_line_of_a_job_file_that_is_not_a_page_job(
    body_and_notes_page: Path, tmp_path: Path, page_style: PageStyle, page_text: str
) -> None:
    job_file = tmp_path / "run.jsonl"
    page = PageJob(template=body_and_notes_page, text=page_text, style=page_style)
    job_file.write_text(
        page.model_dump_json() + "\n\n" + '{"template": "page.xml", "text": "den 4 maj"}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match=r"run\.jsonl:3 is not a page job: style: Field required$"):
        read_page_jobs(job_file, presets={}, loras={})


def test_single_pages_of_an_opening_are_a_job_for_each_of_its_pages_its_id_suffixed_with_its_side(
    opening_page: Path, page_style: PageStyle, page_text: str
) -> None:
    page = PageJob(template=opening_page, text=page_text, style=page_style, seed=5, page_pixels=250_000, id="vol7_p12")

    pages = single_pages(page)

    assert pages == [
        PageJob(
            template=opening_page, text=page_text, style=page_style, seed=5, page_pixels=250_000, id=page_id, side=side
        )
        for page_id, side in (("vol7_p12_left", PageSide.LEFT), ("vol7_p12_right", PageSide.RIGHT))
    ]


def test_single_pages_keep_a_job_already_given_a_side_as_it_is(
    opening_page: Path, page_style: PageStyle, page_text: str
) -> None:
    page = PageJob(template=opening_page, text=page_text, style=page_style, side=PageSide.RIGHT)

    assert single_pages(page) == [page]


def test_read_page_jobs_keep_the_page_of_an_opening_a_line_asks_for(
    opening_page: Path, tmp_path: Path, page_style: PageStyle, page_text: str
) -> None:
    page = PageJob(template=opening_page, text=page_text, style=page_style, side=PageSide.RIGHT)
    job_file = tmp_path / "run.jsonl"
    job_file.write_text(page.model_dump_json() + "\n", encoding="utf-8")

    pages = read_page_jobs(job_file, presets={}, loras={})

    assert [resolved.page for resolved in pages] == [page]


def test_crop_jobs_refuse_a_side_of_a_single_page_naming_its_template(
    body_and_notes_page: Path, page_style: PageStyle, page_text: str
) -> None:
    page = PageJob(template=body_and_notes_page, text=page_text, style=page_style, side=PageSide.LEFT)  # 2000 x 2400 px

    with pytest.raises(ValueError, match=f"^{re.escape(str(body_and_notes_page))} has no left page to draw: "):
        crop_jobs(page, 0)


def test_a_layout_line_given_as_a_box_is_running_text_without_text_its_baseline_at_the_box_bottom() -> None:
    line = TextLine.model_validate({"box": [600, 240, 1900, 320]})

    assert line == TextLine(text="", box=(600, 240, 1900, 320), baseline_y=320, kind=LineKind.BODY)


def test_crop_jobs_plan_a_template_without_text_as_its_lines_own_text(
    body_and_notes_page: Path, page_style: PageStyle
) -> None:
    page = PageJob(template=body_and_notes_page, style=page_style, seed=5)

    jobs = crop_jobs(page, 2)

    plans = plan_page(load_xml_layout(body_and_notes_page), page_style)
    assert [(job.name, job.seed, job.plan) for job in jobs] == [
        (f"002_page_{k}", 2005 + k, plan) for k, plan in enumerate(plans)
    ]


def test_crop_jobs_plan_a_layout_without_text_as_its_lines_own_text(page_style: PageStyle) -> None:
    layout = _boxes_layout([f"line {i} of the running text of the page" for i in range(len(BOXES))])
    page = PageJob(layout=layout, style=page_style, seed=5)

    jobs = crop_jobs(page, 2)

    plans = plan_page(layout, page_style)
    assert [(job.name, job.seed, job.plan) for job in jobs] == [
        (f"002_layout_{k}", 2005 + k, plan) for k, plan in enumerate(plans)
    ]


def test_crop_jobs_plan_a_layout_given_as_boxes_with_the_text_written_into_it(
    page_style: PageStyle, page_text: str
) -> None:
    layout = _boxes_layout()
    page = PageJob(layout=layout, text=page_text, style=page_style, seed=5)

    jobs = crop_jobs(page, 2)

    plans = plan_page(refill(layout, page_text), page_style)
    assert [(job.name, job.seed, job.template, job.plan) for job in jobs] == [
        (f"002_layout_{k}", 2005 + k, None, plan) for k, plan in enumerate(plans)
    ]


def test_crop_jobs_plan_a_layout_as_the_single_page_it_is_neither_marked_from_a_scan_nor_split(
    ink_meter: PrintHandClassifier, page_style: PageStyle, page_text: str
) -> None:
    across = [{"box": (100.0, 240.0 + 100 * i, 2400.0, 320.0 + 100 * i)} for i in range(4)]  # over the middle
    layout = Layout.model_validate({"width": 2500, "height": 1200, "lines": across})  # a template's would be a spread
    page = PageJob(layout=layout, text=page_text, style=page_style, seed=5)

    jobs = crop_jobs(page, 2, print_hand=ink_meter)

    plans = plan_page(refill(layout, page_text), page_style)
    assert [(job.name, job.plan) for job in jobs] == [(f"002_layout_{k}", plan) for k, plan in enumerate(plans)]


@pytest.mark.parametrize("both", [pytest.param(False, id="neither"), pytest.param(True, id="both")])
def test_page_plan_blocks_refuse_a_page_given_by_both_or_neither_of_a_template_and_a_layout(
    body_and_notes_page: Path, both: bool, page_style: PageStyle, page_text: str
) -> None:
    planner = PagePlanBlocks().init_pipeline()
    template, layout = (body_and_notes_page, _boxes_layout()) if both else (None, None)

    with pytest.raises(ValueError, match=f"^{re.escape(ONE_SOURCE)}$"):
        planner(template=template, layout=layout, text=page_text, style=page_style, output="crop_plans")


@pytest.mark.parametrize(
    ("fields", "problem"),
    [
        pytest.param({"text": "den 4 maj"}, ONE_SOURCE, id="neither"),
        pytest.param({"template": "page.xml", "layout": BOXES_LAYOUT, "text": "den 4 maj"}, ONE_SOURCE, id="both"),
        pytest.param(
            {"layout": BOXES_LAYOUT, "text": "den 4 maj", "side": "left"},
            "a layout is drawn as the single page it is: side picks a page of a template's opening",
            id="side",
        ),
        pytest.param(
            {"layout": {**BOXES_LAYOUT, "lines": []}, "text": "den 4 maj"},
            "the layout has no lines to draw",
            id="no-lines",
        ),
        pytest.param(
            {"layout": {**BOXES_LAYOUT, "height": 320}, "text": "den 4 maj"},
            "lines[1]'s box (600.0, 340.0, 1900.0, 420.0) does not lie on the 2000 x 320 px page with a width and a "
            "height",
            id="off-the-page",
        ),
        pytest.param({"layout": BOXES_LAYOUT}, NO_TEXT_TO_KEEP, id="no-text"),
        pytest.param(
            {"layout": {**BOXES_LAYOUT, "lines": [{"box": BOXES[0], "kind": "heading"}]}, "text": "den 4 maj"},
            NO_TEXT_TO_KEEP,
            id="a-line-the-text-keeps-without-text",
        ),
    ],
)
def test_page_job_refuses_a_page_it_cannot_take_a_layout_and_text_from(
    fields: dict[str, object], problem: str, page_style: PageStyle
) -> None:
    with pytest.raises(ValidationError, match=f"Value error, {re.escape(problem)} "):
        PageJob.model_validate({**fields, "style": page_style})
