"""A run's pages planned into crop jobs: cut by the plan steps exactly as the page is planned (captioned for its
reference when it has one), each named and seeded by its page's place in the run; a page that plans to nothing is
refused, not silently dropped from the run."""

from __future__ import annotations

from pathlib import Path

import pytest

from diffu_page.captions.look import ScanLook
from diffu_page.captions.provenance import Provenance
from diffu_page.jobs import PageJob, crop_jobs
from diffu_page.layout import load_xml_layout, refill
from diffu_page.plan import PageStyle, Reference, plan_page

STYLE = PageStyle(
    provenance=Provenance(title="the archive", years="1700"), look=ScanLook(greyscale=False, paper="cream", ink="brown")
)
TEXT = "Anno sextonhundra och tjugo den femte dagen i maj kom for ratten bonden Olof Jonsson ifran Rodon " * 4


@pytest.mark.parametrize(
    "reference",
    [pytest.param(None, id="named-style"), pytest.param(Reference(image=Path("ref.jpg")), id="reference")],
)
def test_crop_jobs_plan_page_two_of_a_run_into_named_and_seeded_crops(
    body_and_notes_page: Path, reference: Reference | None
) -> None:
    page = PageJob(template=body_and_notes_page, text=TEXT, style=STYLE, seed=5, reference=reference)

    jobs = crop_jobs(page, 2)

    plans = plan_page(refill(load_xml_layout(body_and_notes_page), [TEXT]), STYLE, reference=reference)
    image = reference.image if reference else None
    assert [(job.name, job.seed, job.plan, job.reference) for job in jobs] == [
        (f"002_page_{k}", 2005 + k, plan, image) for k, plan in enumerate(plans)
    ]


def test_crop_jobs_refuse_a_template_that_plans_to_no_crops(tmp_path: Path) -> None:
    one_line = tmp_path / "one_line.xml"  # a single line: no pitch to measure, so no window
    one_line.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        '<Page imageFilename="one_line.jpg" imageWidth="1000" imageHeight="1000">'
        '<TextRegion id="r" type="paragraph"><TextLine id="l"><Coords points="100,440 900,440 900,520 100,520"/>'
        '<Baseline points="100,500 900,500"/><TextEquiv><Unicode>den fjerde dagen i maj</Unicode></TextEquiv>'
        "</TextLine></TextRegion></Page></PcGts>",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="plans to no crops"):
        crop_jobs(PageJob(template=one_line, text=TEXT, style=STYLE), 0)
