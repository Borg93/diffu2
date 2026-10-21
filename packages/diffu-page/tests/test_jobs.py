"""A run's pages planned into crop jobs: cut by the plan steps exactly as the page is planned, named and seeded as the
page pipeline names and seeds its crops."""

from __future__ import annotations

from pathlib import Path

from diffu_page.captions.look import ScanLook
from diffu_page.captions.provenance import Provenance
from diffu_page.jobs import PageJob, crop_jobs
from diffu_page.layout import load_xml_layout, refill
from diffu_page.plan import PageStyle, plan_page

STYLE = PageStyle(
    provenance=Provenance(title="the archive", years="1700"), look=ScanLook(greyscale=False, paper="cream", ink="brown")
)
TEXT = "Anno sextonhundra och tjugo den femte dagen i maj kom for ratten bonden Olof Jonsson ifran Rodon " * 4


def test_crop_jobs_plan_page_two_of_a_run_into_named_and_seeded_crops(body_and_notes_page: Path) -> None:
    page = PageJob(template=body_and_notes_page, text=TEXT, style=STYLE, seed=5)

    jobs = crop_jobs(page, 2)

    plans = plan_page(refill(load_xml_layout(body_and_notes_page), [TEXT]), STYLE)
    assert [(job.name, job.seed, job.plan) for job in jobs] == [
        (f"002_page_{k}", 2005 + k, plan) for k, plan in enumerate(plans)
    ]
