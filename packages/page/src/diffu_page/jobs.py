"""The work units of a generation run: a page to draw, and the crops it is planned into.

A :class:`PageJob` is a real template page, the text to write into it and the style to draw it in (one per line of a
run file). Each page is planned on CPU into :class:`CropJob` s (:func:`crop_jobs`), and each crop job is drawn on its
own with the crop pipeline: by diffu-generate one after another, by diffu-runner's GPU actors in parallel. Crops are
named and seeded by ``crop_identity``, so a page comes out the same either way.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, field_validator

from diffu_page.blocks import crop_identity
from diffu_page.captions.prompt import PageStyle
from diffu_page.output import plain_file_name
from diffu_page.pipeline import PagePlanBlocks
from diffu_page.plan import CropPlan, Reference

SEED_STRIDE = 1000  # page i of a run draws crop k from noise seed  seed + SEED_STRIDE * i + k


class PageJob(BaseModel):
    """A template page to redraw: its PAGE/ALTO file, the text to write into it (a blank line ends a paragraph), the
    style and the noise seed.

    ``reference`` is a crop whose hand, ink and paper to draw in (style by example) instead of the style's archive;
    the crops are then captioned as its LoRA was trained. ``id``, a plain file name such as a content hash, names and
    seeds the page's crops in place of the page's place in the run, so adding or removing pages moves no other page.
    """

    template: Path
    text: str
    style: PageStyle
    seed: int = 0
    reference: Reference | None = None
    id: str | None = None

    @field_validator("id")
    @classmethod
    def _plain_file_name(cls, page_id: str | None) -> str | None:
        return None if page_id is None else plain_file_name(page_id)

    def name_at(self, index: int) -> str:
        """The page's name as page ``index`` of a run: its id if it has one."""
        return self.id if self.id is not None else f"{index:03d}_{self.template.stem}"

    def seed_at(self, index: int) -> int:
        """The noise seed of the page's first crop as page ``index`` of a run: from its id if it has one."""
        if self.id is None:
            return self.seed + SEED_STRIDE * index
        return self.seed + SEED_STRIDE * int.from_bytes(hashlib.sha256(self.id.encode()).digest()[:4], "big")


class CropJob(BaseModel):
    """One crop to draw: its name, its page's template, its plan, its noise seed and its reference image."""

    name: str
    template: Path
    plan: CropPlan
    seed: int
    reference: Path | None = None

    def reference_image(self) -> Image.Image | None:
        """The image the crop is drawn from, as RGB; None without a reference."""
        if self.reference is None:
            return None
        with Image.open(self.reference) as image:
            return image.convert("RGB")


def crop_jobs(page: PageJob, index: int) -> list[CropJob]:
    """Page ``index`` of a run, planned into its crops by :class:`PagePlanBlocks`, each named and seeded by
    ``crop_identity``.

    Raises:
        ValueError: If the page plans to no crops: it would silently drop out of the run.
    """
    planner = PagePlanBlocks().init_pipeline()
    plans = planner(
        template=page.template, text=page.text, style=page.style, reference=page.reference, output="crop_plans"
    )
    if not plans:
        raise ValueError(f"{page.template} plans to no crops: no window of it holds enough whole lines")
    image = page.reference.image if page.reference else None
    jobs: list[CropJob] = []
    for k, plan in enumerate(plans):
        name, seed = crop_identity(page.name_at(index), page.seed_at(index), k)
        jobs.append(CropJob(name=name, template=page.template, plan=plan, seed=seed, reference=image))
    return jobs
