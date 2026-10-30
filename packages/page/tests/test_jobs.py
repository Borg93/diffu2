"""A run's pages planned into crop jobs: cut by the plan steps exactly as the page is planned (captioned for its
reference when it has one), each named and seeded by its page's place in the run or by its id. A page that plans to
nothing is refused, not dropped from the run: test_pipeline's generate test reads the refusal from the summary. A job
of a two-page opening becomes a job per page, named apart."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from diffu_page.captions.prompt import PageStyle
from diffu_page.jobs import PageJob, crop_jobs, read_page_jobs, single_pages
from diffu_page.layout import PageSide, load_xml_layout, refill
from diffu_page.plan import Reference, plan_page


@pytest.mark.parametrize(
    "reference",
    [pytest.param(None, id="named-style"), pytest.param(Reference(image=Path("ref.jpg")), id="reference")],
)
def test_crop_jobs_plan_page_two_of_a_run_into_named_and_seeded_crops(
    body_and_notes_page: Path, reference: Reference | None, page_style: PageStyle, page_text: str
) -> None:
    page = PageJob(template=body_and_notes_page, text=page_text, style=page_style, seed=5, reference=reference)

    jobs = crop_jobs(page, 2)

    plans = plan_page(refill(load_xml_layout(body_and_notes_page), page_text), page_style, reference=reference)
    image = reference.image if reference else None
    assert [(job.name, job.seed, job.plan, job.reference) for job in jobs] == [
        (f"002_page_{k}", 2005 + k, plan, image) for k, plan in enumerate(plans)
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
