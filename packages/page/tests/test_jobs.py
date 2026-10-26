"""A run's pages planned into crop jobs: cut by the plan steps exactly as the page is planned (captioned for its
reference when it has one), each named and seeded by its page's place in the run or by its id. A page that plans to
nothing is refused, not dropped from the run: test_pipeline's generate test reads the refusal from the summary."""

from __future__ import annotations

from pathlib import Path

import pytest

from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import PageStyle
from diffu_page.captions.provenance import Provenance
from diffu_page.jobs import PageJob, crop_jobs
from diffu_page.layout import load_xml_layout, refill
from diffu_page.plan import Reference, plan_page, plan_whole_page

STYLE = PageStyle(
    provenance=Provenance(title="the archive", years="1700"), look=ScanLook(greyscale=False, paper="cream", ink="brown")
)
TEXT = "Anno sextonhundra och tjugo den femte dagen i maj kom for ratten bonden Olof Jonsson ifran Rodon " * 4


@pytest.mark.parametrize(
    ("reference", "page_pixels"),
    [
        pytest.param(None, None, id="named-style"),
        pytest.param(Reference(image=Path("ref.jpg")), None, id="reference"),
        pytest.param(None, 1_000_000, id="whole-page"),
    ],
)
def test_crop_jobs_plan_page_two_of_a_run_into_named_and_seeded_crops(
    body_and_notes_page: Path, reference: Reference | None, page_pixels: int | None
) -> None:
    page = PageJob(
        template=body_and_notes_page, text=TEXT, style=STYLE, seed=5, reference=reference, page_pixels=page_pixels
    )

    jobs = crop_jobs(page, 2)

    layout = refill(load_xml_layout(body_and_notes_page), TEXT)
    plans = (
        plan_page(layout, STYLE, reference=reference)
        if page_pixels is None
        else [plan_whole_page(layout, STYLE, pixels=page_pixels)]
    )
    image = reference.image if reference else None
    assert [(job.name, job.seed, job.plan, job.reference) for job in jobs] == [
        (f"002_page_{k}", 2005 + k, plan, image) for k, plan in enumerate(plans)
    ]


def test_crop_jobs_name_and_seed_a_page_with_an_id_alike_wherever_it_stands_in_a_run(body_and_notes_page: Path) -> None:
    page = PageJob(template=body_and_notes_page, text=TEXT, style=STYLE, seed=5, id="vol7_p12")

    early, late = crop_jobs(page, 2), crop_jobs(page, 7)

    assert [(job.name, job.seed) for job in late] == [(job.name, job.seed) for job in early]
    assert [job.name for job in early] == [f"vol7_p12_{k}" for k in range(len(early))]
