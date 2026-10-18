"""Whole-page generation — stream a page being built line by line, then emit the ground-truth labels.

Each handwriting line denoises (reusing :func:`diffu_studio.line.denoise_frames`), gets matted with the
new median+pad-flatten matte, and is seated into its slot via :class:`diffu_page.compose.PageComposer`, so
the page visibly grows top→bottom; printed furniture is typeset. A final scan/aging pass and the PAGE/ALTO/
JSON labels close it out. On a non-interactive device (CPU) the per-line denoise frames are skipped — only
the growing page is yielded — so a slow box doesn't stall the stream.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import cv2
import numpy as np
from diffu_page.augment import AugmentConfig
from diffu_page.config import GenConfig, Template
from diffu_page.labels import PageLabels, export
from diffu_page.layout import Layout, LineBox, PageSpec, build_layout, load_xml_layout
from diffu_page.sampler import LayoutMix, sample_page_spec
from pydantic import BaseModel, ConfigDict, Field

from diffu_studio.configs import SampleSettings
from diffu_studio.line import denoise_frames
from diffu_studio.steps import PageStep, RenderCache

if TYPE_CHECKING:
    import torch
    from PIL.Image import Image

    from diffu_studio.checkpoints import LoadedModel


class LayoutDetail(BaseModel):
    """The :class:`PageSpec`-only layout knobs the studio exposes beyond :class:`GenConfig` — furniture
    scales, sectioning, edit probabilities and the table/kv geometry. Threaded verbatim into
    :func:`_page_spec`; defaults mirror :class:`PageSpec` so an untouched panel changes nothing."""

    model_config = {"frozen": True}

    header_scale: float = Field(default=1.6, ge=1.0)  # header size relative to body (never smaller)
    marg_scale: float = Field(default=0.95, gt=0.0, le=1.0)  # marginalia size relative to body (never bigger)
    margin_ratio: float = Field(default=0.06, ge=0.0, le=0.5)  # page margin, fraction of min(W,H)
    px_per_char: float = Field(default=15.0, gt=0.0)  # wrap-density estimate (chars → slot width)
    drop_cap: bool = False  # enlarge the opening initial into a versal (region "initial")
    catchword: str | None = None  # next leaf's first word at the bottom-outer corner; None = off
    section_headings: list[str] = Field(default_factory=list)  # centered rubrics between paragraphs
    enum_prob: float = Field(default=0.0, ge=0.0, le=1.0)  # per paragraph: prefix an article enumerator
    enum_marker: str = "§"  # the enumerator glyph
    underline_prob: float = Field(default=0.0, ge=0.0, le=1.0)  # per body line: underline it (labelled GT)
    margin_keys: list[str] = Field(default_factory=list)  # keyed margin notes, one per body paragraph
    table_rows: int = Field(default=14, ge=1)  # table templates (semi_table / ledger / register)
    table_cols: int = Field(default=3, ge=1)
    kv_label_frac: float = Field(default=0.34, gt=0.0, lt=1.0)  # key_value: label-column body fraction


class PageRequest(BaseModel):
    """Everything that defines one page to generate, apart from the model + writer style (the engine)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    paragraphs: list[str]
    gen: GenConfig = Field(default_factory=GenConfig)
    settings: SampleSettings = Field(default_factory=SampleSettings)
    seed: int = 0
    header_lines: list[str] = Field(default_factory=list)
    marginalia: list[str] = Field(default_factory=list)
    page_number: str | None = None
    detail: LayoutDetail = Field(default_factory=LayoutDetail)  # PageSpec-only knobs (see LayoutDetail)
    # "Surprise me": draw the WHOLE spec from the corpus-proportion sampler instead of the hand-set
    # template/detail fields — deterministic given ``seed`` (reroll the seed to reroll the layout).
    surprise: bool = False
    spread_frac: float = Field(default=0.35, ge=0.0, le=1.0)  # surprise only: two-page-opening fraction
    xml_layout: str | None = (
        None  # a real ALTO/PAGE XML: replay its geometry; own text reflows into its boxes
    )
    animate: bool = True  # yield per-line denoise frames (interactive GPU); off for CPU → grow-only


def stream_page(loaded: LoadedModel, style: torch.Tensor, request: PageRequest) -> Iterator[PageStep]:
    """Stream a page as it is built line by line, ending with the final scanned page + serialized labels."""
    from diffu_page.compose import PageComposer
    from diffu_page.render import PlaceholderRenderer, ink_from_pil  # ink_from_pil: the exact proven matte

    from diffu.generate import ink_crop

    cfg = request.gen
    spec = _page_spec(request)
    measure = lambda t, h: _measure(loaded, t, h)  # noqa: E731 — small local closure, reads fine inline
    own_text = [p for p in request.paragraphs if p.strip()]
    if request.xml_layout:
        layout = load_xml_layout(request.xml_layout, spec)  # replay the real document's line geometry
        if own_text:  # geometry stencil: pour the user's own words into the real boxes
            layout = _reflow_text(layout, own_text, measure)
    else:
        layout = build_layout(spec, request.paragraphs, seed=request.seed, measure=measure)
    composer = PageComposer(layout, seed=request.seed, aug=cfg.augment)
    typeset = PlaceholderRenderer()
    total = len(layout.boxes)
    inks: list[np.ndarray] = []  # kept aligned to layout.boxes for the re-composite cache

    for i, box in enumerate(layout.boxes):
        if box.printed:
            ink = typeset.render(box.text, box.h)
        else:
            frame: Image | None = None
            for fr in denoise_frames(loaded, style, box.text, request.settings):
                frame = fr
                if request.animate:
                    yield PageStep(
                        page=_to_rgb(composer.page),
                        active=fr,
                        status=f"⟳ line {i + 1}/{total}",
                        index=i,
                        total=total,
                        phase="denoise",
                    )
            ink = (
                ink_from_pil(ink_crop(frame), box.h) if frame is not None else typeset.render(box.text, box.h)
            )
        inks.append(ink)
        composer.seat(box, ink)
        yield PageStep(
            page=_to_rgb(composer.page),
            active=None,
            status=f"seated {i + 1}/{total}",
            index=i + 1,
            total=total,
            phase="seated",
        )

    page, lines = composer.finish()
    labels = PageLabels(width=page.shape[1], height=page.shape[0], image="page.png", lines=lines)
    serialized = {fmt.value: export(labels, fmt.value) for fmt in cfg.formats}
    yield PageStep(
        page=_to_rgb(page),
        active=None,
        status=f"✓ {total} lines",
        index=total,
        total=total,
        phase="final",
        labels=serialized,
        page_labels=labels,
        render=RenderCache(
            layout=layout,
            inks=inks,
            seed=request.seed,
            gen=cfg,
            paragraphs=request.paragraphs,
            xml_layout=request.xml_layout,
            measure=measure,
        ),
    )


def recompose_page(
    cache: RenderCache, aug: AugmentConfig, formats: list[str], *, seed: int | None = None
) -> PageStep:
    """Re-seat the CACHED inks with a new augmentation — the CPU-only path for augmentation/scan control changes.

    The handwriting isn't touched (no model, no GPU): only paper, ink colour and the scan/aging pass change.
    ``seed`` re-rolls WHERE the scan look lands (fold/dust/vignette placement) on the same ink; ``None`` keeps
    the page's original seed. Returns a ``final`` PageStep so the UI updates page + overlay + downloads.
    """
    from diffu_page.compose import compose_page_from_inks

    use_seed = cache.seed if seed is None else seed
    page, lines = compose_page_from_inks(cache.layout, cache.inks, seed=use_seed, aug=aug)
    labels = PageLabels(width=page.shape[1], height=page.shape[0], image="page.png", lines=lines)
    serialized = {fmt: export(labels, fmt) for fmt in formats}
    return PageStep(
        page=_to_rgb(page),
        status=f"✓ recomposited ({len(lines)} lines) — augmentation only, no re-generation",
        index=len(lines),
        total=len(lines),
        phase="final",
        labels=serialized,
        page_labels=labels,
        render=cache.model_copy(update={"seed": use_seed}),
    )


def live_update_page(cache: RenderCache, gen: GenConfig, *, seed: int | None = None) -> PageStep:
    """Apply a CHEAP layout change (line/paragraph gap) + augmentation from cached inks — no model, no GPU.

    Rebuilds the layout at the new geometry with the SAME wrapping (``cache.measure``); if the new boxes' texts
    still match the cached inks (a prefix match — a wider gap just drops trailing lines that overflowed), the
    cached inks re-seat 1:1 into the new positions. If the change RE-WRAPS the text (different words per line),
    the inks can't map, so this returns an ``error`` step telling the user to Generate — never a mis-labelled page.
    """
    from diffu_page.compose import PageComposer

    # Layout wrapping stays on the ORIGINAL seed so a spacing tweak never spuriously re-wraps; only the scan
    # look (the composer) re-rolls with the override seed. The spec is the CACHED layout's own spec with just
    # the gaps swapped — not a rebuild from the request — so every generation-time knob (template, layout
    # detail, page number, a surprise-sampled spec) survives the spacing tweak instead of resetting.
    use_seed = cache.seed if seed is None else seed
    spec = cache.layout.spec.model_copy(update={"line_gap": gen.line_gap, "para_gap": gen.para_gap})
    if cache.xml_layout:
        layout = load_xml_layout(cache.xml_layout, spec)
        own = [p for p in cache.paragraphs if p.strip()]
        if own:
            layout = _reflow_text(layout, own, cache.measure or (lambda t, h: len(t) * h // 2))
    else:
        layout = build_layout(spec, cache.paragraphs, seed=cache.seed, measure=cache.measure)

    new_texts = [b.text for b in layout.boxes]
    old_texts = [b.text for b in cache.layout.boxes]
    keep = len(new_texts)
    if keep > len(cache.inks) or new_texts != old_texts[:keep]:
        return PageStep(
            page=None,
            status="⚠ this change re-wraps the text — click Generate to re-render",
            index=0,
            total=keep,
            phase="error",
        )

    composer = PageComposer(layout, seed=use_seed, aug=gen.augment)
    for box, ink in zip(layout.boxes, cache.inks[:keep], strict=True):
        composer.seat(box, ink)
    page, lines = composer.finish()
    labels = PageLabels(width=page.shape[1], height=page.shape[0], image="page.png", lines=lines)
    serialized = {fmt.value: export(labels, fmt.value) for fmt in gen.formats}
    return PageStep(
        page=_to_rgb(page),
        status=f"✓ re-seated {len(lines)} lines (layout only, no re-generation)",
        index=len(lines),
        total=len(lines),
        phase="final",
        labels=serialized,
        page_labels=labels,
        render=cache.model_copy(
            update={"layout": layout, "inks": cache.inks[:keep], "gen": gen, "seed": use_seed}
        ),
    )


def _reflow_text(layout: Layout, paragraphs: list[str], measure: Callable[[str, int], int]) -> Layout:
    """Pour the user's words into an XML layout's boxes (geometry stencil): each handwriting box takes as many
    words as fit its width; printed furniture and any boxes past the text keep their original transcription."""
    words = " ".join(paragraphs).split()
    cursor = 0
    boxes: list[LineBox] = []
    for box in layout.boxes:
        if box.printed or cursor >= len(words):
            boxes.append(box)
            continue
        taken = [words[cursor]]
        cursor += 1
        while cursor < len(words) and measure(" ".join([*taken, words[cursor]]), box.h) <= box.w:
            taken.append(words[cursor])
            cursor += 1
        boxes.append(box.model_copy(update={"text": " ".join(taken)}))
    return layout.model_copy(update={"boxes": boxes})


def _page_spec(request: PageRequest) -> PageSpec:
    """Build the page geometry from the request, mirroring the corpus CLI's per-page spec construction.

    With ``surprise`` on, the WHOLE spec comes from the corpus-proportion sampler instead of the hand-set
    form fields (template/detail/size are ignored) — deterministic given ``seed``, so rerolling the seed
    rerolls the layout kind too.
    """
    cfg = request.gen
    rng = random.Random(request.seed)
    if request.surprise:
        _, spec = sample_page_spec(rng, LayoutMix(spread_frac=request.spread_frac))
        return spec
    detail = request.detail
    return PageSpec(
        width=cfg.width,
        height=cfg.height,
        template=cfg.template.value,
        n_columns=cfg.n_columns,
        typeset_furniture=cfg.typeset_furniture,
        line_height=cfg.line_height,
        line_gap=cfg.line_gap,
        para_gap=cfg.para_gap,
        binding=cfg.template is not Template.spread and rng.random() < cfg.binding_prob,
        recto=rng.random() < 0.5,
        header_lines=request.header_lines,
        marginalia=request.marginalia,
        page_number=request.page_number,
        insertion_prob=cfg.insertion_prob,
        deletion_prob=cfg.deletion_prob,
        margin_ratio=detail.margin_ratio,
        px_per_char=detail.px_per_char,
        header_scale=detail.header_scale,
        marg_scale=detail.marg_scale,
        margin_keys=detail.margin_keys,
        table_rows=detail.table_rows,
        table_cols=detail.table_cols,
        kv_label_frac=detail.kv_label_frac,
        underline_prob=detail.underline_prob,
        catchword=detail.catchword,
        section_headings=detail.section_headings,
        enum_prob=detail.enum_prob,
        enum_marker=detail.enum_marker,
        drop_cap=detail.drop_cap,
    )


def _measure(loaded: LoadedModel, text: str, height: int) -> int:
    """Predicted rendered width of ``text`` at ``height`` — the model's content-aware natural width."""
    from diffu.generate import natural_width

    width = natural_width(loaded.model.guidance_renderer, text or " ", loaded.cfg)
    return int(width * height / loaded.cfg.data.line_height)


def _to_rgb(page_bgr: np.ndarray) -> Image:
    """A BGR page (OpenCV) → an RGB PIL image for display/streaming."""
    from PIL import Image as PILImage

    return PILImage.fromarray(cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB))
