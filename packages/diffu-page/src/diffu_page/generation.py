"""Generation orchestration — the run logic behind the ``diffu-page`` CLI.

Everything a run needs between "a validated :class:`~diffu_page.config.GenConfig`" and "rendered
pages + labels on disk" lives here, CLI-free and importable: sample text + furniture, apply flag
overrides, build the renderer, lay out all pages, assign writer styles, render/compose, and write
the outputs (:func:`generate_pages` is the whole run). The thin typer command in
:mod:`diffu_page.cli` just parses flags and drives these.
"""

import json
import random
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import typer
from rich.console import Console

from .compose import compose_page, compose_page_from_inks
from .config import GenConfig, Renderer, Template
from .dataset import CropRecord, PageRecord, assign_split, export_line_crops, write_manifest
from .labels import EXTENSIONS, LineLabel, PageLabels, export
from .layout import Layout, PageSpec, build_layout, load_xml_layout
from .render import (
    BatchedDiffuRenderer,
    DiffuRenderer,
    ErukuRenderer,
    LineRenderer,
    PlaceholderRenderer,
    render_layouts,
)

console = Console()

# A little public-domain-ish Swedish filler so placeholder pages look like text. Swap via --text-file.
_SAMPLE = [
    "Göteborgs poliskammare den fjärde maj artonhundra åttiotvå.",
    "Anhållen för fylleri och förargelseväckande beteende å allmän gata.",
    "Mannen uppgav sig vara arbetare och boende i Haga församling.",
    "Förd till häktet och förhörd av vakthavande konstapel om natten.",
    "Smörgåsbord, kärlek och ärlighet varar längst sade den gamle mannen.",
]


def load_paragraphs(text_file: Path | None) -> list[str]:
    """Paragraphs for a run: one per non-blank line of ``text_file``, or the built-in Swedish filler."""
    if text_file:
        return [ln for ln in text_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return _SAMPLE * 12  # repeat so columns/tables fill


# Plausible archival furniture, auto-generated per page when GenConfig.furniture_prob fires. The exact
# wording is irrelevant to the labels — what matters is that the page carries a header / marginalia /
# page number, like a real protocol, instead of being a bare body.
_HEADERS = [
    ["Göteborgs Poliskammare.", "Protokoll d. {d} Maj 1882."],
    ["Rådhusrätten i Göteborg.", "d. {d} Mars 1884."],
    ["Kongl. Maj:ts Befallningshafvande."],
    ["Till Magistraten.", "Inkommet d. {d} Juni."],
]
_MARGINALIA = [
    "Fylleri.",
    "Erkännande.",
    "Vittnen.",
    "Till domstol.",
    "Anhållan.",
    "Utslag.",
    "Käranden.",
    "Svaranden.",
    "Afslag.",
    "Bifall.",
    "Böter.",
    "Frikänd.",
]


def sample_furniture(rng: random.Random) -> tuple[list[str], list[str], str]:
    """Auto-generate a header block, a few marginalia notes and a page number for one page."""
    header = [h.replace("{d}", str(rng.randint(1, 28))) for h in rng.choice(_HEADERS)]
    marginalia = rng.sample(_MARGINALIA, k=rng.randint(2, 4))
    return header, marginalia, f"{rng.randint(1, 400)}."


def with_overrides(cfg: GenConfig, **flags: object) -> GenConfig:
    """Apply the CLI flags that were actually set (non-None) over ``cfg`` — CLI > YAML > default."""
    aug = {k: flags.pop(k) for k in ("bed_prob", "clasp_prob") if flags.get(k) is not None}
    top = {k: v for k, v in flags.items() if v is not None}
    if aug:
        top["augment"] = cfg.augment.model_copy(update=aug)
    return cfg.model_copy(update=top)


def build_renderer(cfg: GenConfig) -> LineRenderer | BatchedDiffuRenderer:
    """The renderer for ``cfg.renderer``: a batched Diffu renderer (one GPU pass for the whole run), Eruku,
    or the no-model placeholder."""
    if cfg.renderer is Renderer.diffu:
        if not (cfg.ckpt and (cfg.style or cfg.styles)):
            raise typer.BadParameter(
                "renderer=diffu requires ckpt and a style/styles (in the config or via --ckpt/--style)"
            )
        if cfg.fast:  # batched: faster for big runs, but a touch softer (wider canvas spreads strokes)
            return BatchedDiffuRenderer(
                str(cfg.ckpt), num_steps=cfg.steps, cfg_scale=cfg.cfg_scale,
                lora=cfg.lora, lora_scale=cfg.lora_scale,
            )
        style = cfg.style or cfg.styles[0]  # per-line path is single-style; the batched path mixes hands
        return DiffuRenderer(
            str(cfg.ckpt), str(style), num_steps=cfg.steps, cfg_scale=cfg.cfg_scale,
            lora=cfg.lora, lora_scale=cfg.lora_scale,
        )
    if cfg.renderer is Renderer.eruku:
        if not cfg.style:
            raise typer.BadParameter("renderer=eruku requires style")
        return ErukuRenderer(str(cfg.style), cfg_scale=cfg.eruku_cfg)
    return PlaceholderRenderer()


def page_box_styles(layouts: list[Layout], cfg: GenConfig, *, seed: int) -> list[list[str]]:
    """Per-box writer-style path for each page. A page samples one MAIN hand from the style pool; when the
    pool has more than one, marginalia get a second hand so a page can mix writers (main scribe + annotator).
    Pure + deterministic given ``seed`` so a corpus is reproducible. Empty for the no-style renderers."""
    pool = [str(p) for p in cfg.styles] or ([str(cfg.style)] if cfg.style else [])
    if not pool:
        return [["" for _ in layout.boxes] for layout in layouts]
    out: list[list[str]] = []
    for i, layout in enumerate(layouts):
        rng = random.Random(seed + i)
        main = rng.choice(pool)
        alt = rng.choice(pool) if len(pool) > 1 else main
        out.append([alt if box.region == "marginalia" else main for box in layout.boxes])
    return out


def load_specs(path: Path) -> list[PageSpec]:
    """Load the FULL control surface from a JSON file — one ``PageSpec`` (object) or many (array), so an
    agent or user can drive every knob (margin_keys, drop_cap, catchword, table columns, ...) without code.
    Pydantic validates each; an out-of-range or unknown field fails fast with a clear error."""
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else [data]
    return [PageSpec.model_validate(item) for item in items]


def build_layouts(
    cfg: GenConfig,
    *,
    n: int,
    seed: int,
    paragraphs: list[str],
    xml: Path | None,
    measure: Callable[[str, int], int] | None,
    specs: list[PageSpec] | None = None,
) -> list[Layout]:
    """Lay out all ``n`` pages up-front (so the Diffu path can batch every line of every page at once).

    With ``specs`` (from ``--spec``), each page uses a full caller-supplied :class:`PageSpec` verbatim
    (cycled if fewer than ``n``); otherwise a spec is built from ``cfg`` and a ``furniture_prob`` fraction of
    pages get an auto-generated header + marginalia + page number.
    """
    layouts: list[Layout] = []
    for i in range(n):
        if specs is not None:  # full user/agent-authored spec — respected verbatim, no cfg furniture override
            spec = specs[i % len(specs)]
            layouts.append(
                load_xml_layout(str(xml), spec)
                if xml
                else build_layout(spec, paragraphs, seed=seed + i, measure=measure)
            )
            continue
        page_rng = random.Random(seed + i)
        header: list[str] = []
        marginalia: list[str] = []
        page_number: str | None = None
        if page_rng.random() < cfg.furniture_prob:
            header, marginalia, page_number = sample_furniture(page_rng)
        spec = PageSpec(
            width=cfg.width,
            height=cfg.height,
            template=cfg.template.value,
            n_columns=cfg.n_columns,
            line_height=cfg.line_height,
            line_gap=cfg.line_gap,
            para_gap=cfg.para_gap,
            # mix bound/unbound across the corpus; a spread has its own central gutter, not a single spine
            binding=cfg.template is not Template.spread and page_rng.random() < cfg.binding_prob,
            recto=page_rng.random() < 0.5,  # random page side -> spine + page-no on the matching edge
            binding_ratio=round(page_rng.uniform(0.025, 0.075), 3),  # vary how close text runs to the gutter
            header_lines=header,
            marginalia=marginalia,
            page_number=page_number,
            insertion_prob=cfg.insertion_prob,
            deletion_prob=cfg.deletion_prob,
        )
        layouts.append(
            load_xml_layout(str(xml), spec)
            if xml
            else build_layout(spec, paragraphs, seed=seed + i, measure=measure)
        )
    return layouts


def render_pages(
    renderer: LineRenderer | BatchedDiffuRenderer,
    layouts: list[Layout],
    cfg: GenConfig,
    *,
    seed: int,
    box_styles: list[list[str]],
) -> list[tuple[np.ndarray, list[LineLabel]]]:
    """Render + compose every page. The batched Diffu renderer goes through one pass over all lines (mixing
    hands per ``box_styles``); the placeholder / Eruku renderers compose page-by-page."""
    paper = str(cfg.paper_dir) if cfg.paper_dir else None
    if isinstance(renderer, BatchedDiffuRenderer):
        inks_per_page = render_layouts(renderer, layouts, box_styles)
        return [
            compose_page_from_inks(lay, inks, paper_dir=paper, seed=seed + i, aug=cfg.augment)
            for i, (lay, inks) in enumerate(zip(layouts, inks_per_page, strict=True))
        ]
    return [
        compose_page(lay, renderer, paper_dir=paper, seed=seed + i, aug=cfg.augment)
        for i, lay in enumerate(layouts)
    ]


def _export_page(
    page: np.ndarray,
    lines: list[LineLabel],
    layout: Layout,
    cfg: GenConfig,
    *,
    out: Path,
    name: str,
    seed: int,
    split: str,
    styles: list[str],
) -> tuple[PageRecord, list[CropRecord]]:
    """Write one page's image + label files (+ optional line crops); return its manifest records."""
    ph, pw = page.shape[:2]  # real shape (xml replay / bed mount can differ from the requested size)
    cv2.imwrite(str(out / f"{name}.png"), page)
    labels = PageLabels(width=pw, height=ph, image=f"{name}.png", lines=lines)
    for f in cfg.formats:
        (out / f"{name}{EXTENSIONS[f.value]}").write_text(export(labels, f.value), encoding="utf-8")
    crops = export_line_crops(page, lines, out_dir=out, page_name=name, split=split) if cfg.crop_lines else []
    record = PageRecord(
        image=f"{name}.png",
        width=pw,
        height=ph,
        template=layout.spec.template,  # the page's real template (may come from --spec, not cfg)
        seed=seed,
        binding=layout.spec.binding,
        recto=layout.spec.recto,
        n_lines=len(lines),
        split=split,
        styles=styles,
    )
    return record, crops


def generate_pages(
    cfg: GenConfig,
    *,
    out: Path,
    n: int,
    seed: int,
    text_file: Path | None = None,
    xml: Path | None = None,
    spec: Path | None = None,
) -> None:
    """Run a whole generation into ``out``: layout, render, compose and write every page + the manifest.

    ``cfg`` is the validated run config (CLI overrides already applied via :func:`with_overrides`); page
    ``i`` derives everything from ``seed + i``. ``text_file`` supplies one paragraph per line, ``xml``
    replays a real ALTO/PAGE layout, and ``spec`` is a full PageSpec JSON (one object or an array, cycled).
    """
    out.mkdir(parents=True, exist_ok=True)
    line_renderer = build_renderer(cfg)
    measure = getattr(line_renderer, "measure_width", None)  # wrap by real width when available
    specs = load_specs(spec) if spec else None  # --spec: the full control surface, verbatim per page
    layouts = build_layouts(
        cfg, n=n, seed=seed, paragraphs=load_paragraphs(text_file), xml=xml, measure=measure, specs=specs
    )
    box_styles = page_box_styles(layouts, cfg, seed=seed)
    pages: list[PageRecord] = []
    crop_records: list[CropRecord] = []
    rendered = render_pages(line_renderer, layouts, cfg, seed=seed, box_styles=box_styles)
    for i, (page, lines) in enumerate(rendered):
        name = f"page_{i:04d}"
        split = assign_split(i, val_frac=cfg.val_frac, test_frac=cfg.test_frac)
        record, crops = _export_page(
            page,
            lines,
            layouts[i],
            cfg,
            out=out,
            name=name,
            seed=seed + i,
            split=split,
            styles=sorted({s for s in box_styles[i] if s}),
        )
        pages.append(record)
        crop_records.extend(crops)
        console.print(
            f"[green]✓[/green] {name}.png  ({len(lines)} lines, {split})  "
            f"+ {', '.join(f.value for f in cfg.formats)}"
        )
    write_manifest(out, pages, crop_records)
    tail = f" + lines.jsonl ({len(crop_records)} crops)" if crop_records else ""
    console.print(f"[green]✓[/green] manifest.jsonl ({len(pages)} pages){tail}")
