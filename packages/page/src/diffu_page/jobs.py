"""The work units of a generation run: a page to draw, and the crops it is planned into.

A :class:`PageJob` is a real template page, the text to write into it and the style to draw it in (one per line of a
run file, where a line may name its style instead: a :class:`PageRequest`, resolved with the style presets of
:mod:`diffu_page.styles`). A job of a two-page opening becomes a job per page (:func:`single_pages`), so a run draws
single pages, never a spread; a scan wider than tall whose pages its lines cannot split is refused when it is planned
(:func:`diffu_page.layout.is_unsplit_spread`), its error recorded as any page's that cannot be planned. Each page is
planned on CPU into :class:`CropJob` s (:func:`crop_jobs`), and each crop
job is drawn on its own with the crop pipeline: by diffu-generate one after another, by diffu-runner's GPU actors in
parallel. Crops are named and seeded by ``crop_identity``, so a page comes out the same either way.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, Field, ValidationError, field_validator

from diffu_page.blocks import crop_identity
from diffu_page.captions.prompt import PageStyle
from diffu_page.layout import PageSide, load_xml_layout, scan_pages
from diffu_page.marking import PrintHandClassifier
from diffu_page.output import load_rgb, plain_file_name, validation_problems
from diffu_page.pipeline import PagePlanBlocks
from diffu_page.plan import CropPlan, Reference
from diffu_page.styles import StylePreset

SEED_STRIDE = 1000  # page i of a run draws crop k from noise seed  seed + SEED_STRIDE * i + k


class PageJob(BaseModel):
    """A template page to redraw: its PAGE/ALTO file, the text to write into it (a blank line ends a paragraph), the
    style and the noise seed.

    ``reference`` is a crop whose hand, ink and paper to draw in (style by example) instead of the style's archive;
    the crops are then captioned as its LoRA was trained. ``page_pixels`` plans the page whole, as one crop of about
    that many pixels, instead of as training-scale windows. ``id``, a plain file name such as a content hash, names and
    seeds the page's crops in place of the page's place in the run, so adding or removing pages moves no other page.
    ``side`` draws one page of a two-page opening as a single page, in its own coordinates (:func:`single_pages`).
    """

    template: Path
    text: str
    style: PageStyle
    seed: int = 0
    reference: Reference | None = None
    page_pixels: int | None = Field(default=None, gt=0)
    id: str | None = None
    side: PageSide | None = None

    @field_validator("id")
    @classmethod
    def _plain_file_name(cls, page_id: str | None) -> str | None:
        return None if page_id is None else plain_file_name(page_id)

    def name_at(self, index: int) -> str:
        """The page's name as page ``index`` of a run: its id if it has one, else its place and template, then the
        side of the opening it is a page of."""
        if self.id is not None:
            return self.id
        name = f"{index:03d}_{self.template.stem}"
        return name if self.side is None else f"{name}_{self.side}"

    def seed_at(self, index: int) -> int:
        """The noise seed of the page's first crop as page ``index`` of a run: from its id if it has one."""
        if self.id is None:
            return self.seed + SEED_STRIDE * index
        return self.seed + SEED_STRIDE * int.from_bytes(hashlib.sha256(self.id.encode()).digest()[:4], "big")


class ResolvedPage(BaseModel):
    """A page as a run draws it: its job, its style spelled out, and the LoRAs that draw it (NeMo ``model/`` dirs by
    adapter name)."""

    page: PageJob
    loras: dict[str, Path]


class PageRequest(BaseModel):
    """A line of a job file: a :class:`PageJob` whose style is spelled out or named (a preset of
    :mod:`diffu_page.styles`)."""

    template: Path
    text: str
    style: PageStyle | str
    seed: int = 0
    reference: Reference | None = None
    page_pixels: int | None = Field(default=None, gt=0)
    id: str | None = None
    side: PageSide | None = None

    def resolve(self, presets: Mapping[str, StylePreset], loras: dict[str, Path]) -> ResolvedPage:
        """The page asked for and the LoRAs that draw it: a style spelled out by ``loras`` (the run's); a named one in
        its preset's style, from the preset's reference unless the request gives its own, by the preset's hand.

        Raises:
            ValueError: If no preset has the style's name, or the page is not a page job (a ``ValidationError``).
        """
        if isinstance(self.style, PageStyle):
            return ResolvedPage(page=self._page(self.style, self.reference), loras=loras)
        preset = presets.get(self.style)
        if preset is None:
            raise ValueError(f"no style preset is named {self.style!r} (the presets: {', '.join(presets) or 'none'})")
        reference = self.reference if self.reference is not None else preset.reference
        return ResolvedPage(page=self._page(preset.style, reference), loras=preset.loras)

    def _page(self, style: PageStyle, reference: Reference | None) -> PageJob:
        return PageJob(
            template=self.template,
            text=self.text,
            style=style,
            seed=self.seed,
            reference=reference,
            page_pixels=self.page_pixels,
            id=self.id,
            side=self.side,
        )


def single_pages(page: PageJob) -> list[PageJob]:
    """The single pages a page job draws: a job for each page of a two-page opening that holds a line
    (:func:`diffu_page.layout.scan_pages`), its side set and the job's id, if it has one, suffixed with the side. Any
    other job is kept as it is: a single page's job, one already given a side, one whose template cannot be read, or one
    of a spread its lines cannot split, which planning refuses (:func:`crop_jobs`)."""
    if page.side is not None:
        return [page]
    try:
        layout = load_xml_layout(page.template)
    except (OSError, ValueError):
        return [page]  # planning reads the template too, and records why it cannot
    sides = [scan_page.side for scan_page in scan_pages(layout) if scan_page.side is not None]
    return [
        page.model_copy(update={"side": side, "id": None if page.id is None else f"{page.id}_{side}"}) for side in sides
    ] or [page]


def read_page_jobs(path: Path, *, presets: Mapping[str, StylePreset], loras: dict[str, Path]) -> list[ResolvedPage]:
    """The pages of a job file: one :class:`PageRequest` per line (JSON Lines, blank lines skipped), each resolved with
    ``presets`` and the run's ``loras`` (:meth:`PageRequest.resolve`).

    Raises:
        ValueError: If a line is not a page job or names no preset, naming the file, the line and what is wrong with it.
    """
    pages: list[ResolvedPage] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            pages.append(PageRequest.model_validate_json(line).resolve(presets, loras))
        except ValidationError as error:
            raise ValueError(f"{path}:{number} is not a page job: {validation_problems(error, 'line')}") from error
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    return pages


class CropJob(BaseModel):
    """One crop to draw: its name, its page's template, its plan, its noise seed and its reference image."""

    name: str
    template: Path
    plan: CropPlan
    seed: int
    reference: Path | None = None

    def reference_image(self) -> Image.Image | None:
        """The image the crop is drawn from, as RGB; None without a reference."""
        return None if self.reference is None else load_rgb(self.reference)


def crop_jobs(page: PageJob, index: int, *, print_hand: PrintHandClassifier | None = None) -> list[CropJob]:
    """Page ``index`` of a run, planned into its crops by :class:`PagePlanBlocks`, its template first marked from its
    scan with ``print_hand`` if one is given (:func:`diffu_page.marking.marked`), each crop named and seeded by
    ``crop_identity``.

    Raises:
        ValueError: If the page plans to no crops: it would silently drop out of the run.
    """
    planner = PagePlanBlocks().init_pipeline()
    plans = planner(
        template=page.template,
        print_hand=print_hand,
        side=page.side,
        text=page.text,
        style=page.style,
        reference=page.reference,
        page_pixels=page.page_pixels,
        output="crop_plans",
    )
    if not plans:
        raise ValueError(f"{page.template} plans to no crops: no window of it holds enough whole lines")
    image = page.reference.image if page.reference else None
    jobs: list[CropJob] = []
    for k, plan in enumerate(plans):
        name, seed = crop_identity(page.name_at(index), page.seed_at(index), k)
        jobs.append(CropJob(name=name, template=page.template, plan=plan, seed=seed, reference=image))
    return jobs
