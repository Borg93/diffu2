"""The work units of a generation run: a page to draw, and the crops it is planned into.

A :class:`PageJob` is a real template page, the text to write into it and the style to draw it in; :func:`crop_jobs`
refills and plans it into :class:`CropJob` s, the unit a generator draws (``diffu-generate`` one by one, diffu-runner
across a Ray cluster). Planning is CPU only.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from diffu_page.layout import load_xml_layout, refill
from diffu_page.plan import CropPlan, PageStyle, plan_page

SEED_STRIDE = 1000  # page i's crop k draws from noise seed  seed + SEED_STRIDE * i + k


class PageJob(BaseModel):
    """A template page to redraw: its PAGE/ALTO file, the text to write into it, the style and the noise seed.

    ``reference`` is a crop whose hand, ink and paper to draw in (style by example) instead of the style's archive.
    """

    template: Path
    text: str
    style: PageStyle
    seed: int = 0
    reference: Path | None = None


class CropJob(BaseModel):
    """One crop to draw: its name, its page's template, its plan, its noise seed and its reference."""

    name: str
    template: Path
    plan: CropPlan
    seed: int
    reference: Path | None = None


def crop_jobs(page: PageJob, index: int) -> list[CropJob]:
    """Page ``index`` of a run, refilled with its text and planned into crops; crop ``k`` draws from noise seed
    ``page.seed + SEED_STRIDE * index + k``."""
    plans = plan_page(refill(load_xml_layout(page.template), [page.text]), page.style)
    return [
        CropJob(
            name=f"{index:03d}_{page.template.stem}_{k}",
            template=page.template,
            plan=plan,
            seed=page.seed + SEED_STRIDE * index + k,
            reference=page.reference,
        )
        for k, plan in enumerate(plans)
    ]
