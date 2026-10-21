"""The work units of a generation run: a page to draw, and the crops it is planned into.

A :class:`PageJob` is a real template page, the text to write into it and the style to draw it in (one per line of a
run file). The page pipeline draws a page whole; diffu-runner plans it on CPU into :class:`CropJob` s
(:func:`crop_jobs`) that GPU actors draw one by one with the crop pipeline. Both name and seed a page's crops with
``crop_identity``, so a page comes out the same either way.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from diffu_page.blocks import crop_identity
from diffu_page.pipeline import PagePlanBlocks
from diffu_page.plan import CropPlan, PageStyle

SEED_STRIDE = 1000  # page i of a run draws crop k from noise seed  seed + SEED_STRIDE * i + k


class PageJob(BaseModel):
    """A template page to redraw: its PAGE/ALTO file, the text to write into it, the style and the noise seed.

    ``reference`` is a crop whose hand, ink and paper to draw in (style by example) instead of the style's archive.
    """

    template: Path
    text: str
    style: PageStyle
    seed: int = 0
    reference: Path | None = None

    def name_at(self, index: int) -> str:
        """The page's name as page ``index`` of a run."""
        return f"{index:03d}_{self.template.stem}"

    def seed_at(self, index: int) -> int:
        """The noise seed of the page's first crop as page ``index`` of a run."""
        return self.seed + SEED_STRIDE * index


class CropJob(BaseModel):
    """One crop to draw: its name, its page's template, its plan, its noise seed and its reference."""

    name: str
    template: Path
    plan: CropPlan
    seed: int
    reference: Path | None = None


def crop_jobs(page: PageJob, index: int) -> list[CropJob]:
    """Page ``index`` of a run, planned into its crops by :class:`PagePlanBlocks` and named and seeded as the page
    pipeline names and seeds them."""
    planner = PagePlanBlocks().init_pipeline()
    plans = planner(template=page.template, text=page.text, style=page.style, output="crop_plans")
    jobs: list[CropJob] = []
    for k, plan in enumerate(plans):
        name, seed = crop_identity(page.name_at(index), page.seed_at(index), k)
        jobs.append(CropJob(name=name, template=page.template, plan=plan, seed=seed, reference=page.reference))
    return jobs
