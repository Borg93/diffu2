"""Audit the diffu_page augmentation against the GT-Lejonet single-page corpus.

Generates a batch of synthetic single pages spanning the conditions in
``packages/diffu-page/AUGMENTATION_SPEC.md`` (C1..C15), measures each one, and prints a PASS/FAIL line per
condition. It also writes a **contact sheet** of the batch and a **reference montage** of real corpus pages
(``gt_study_samples/``) so the result can be looked at side by side — the loop is not done on numbers alone.

It uses the CPU :class:`~diffu_page.render.PlaceholderRenderer` on purpose: every condition here is a
layout / scan-appearance property (density, tone, damage, fill, bounds) that is independent of handwriting
stroke quality, so the audit stays fast, deterministic and needs no GPU or checkpoint. Stroke realism is a
separate axis, validated with the real model elsewhere.

    uv run python scripts/audit_augmentation.py           # 16 pages -> audit_out/
    uv run python scripts/audit_augmentation.py --n 20 --out /tmp/aud
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from diffu_page.augment import AugmentConfig
from diffu_page.compose import compose_page
from diffu_page.labels import LineLabel, PageLabels
from diffu_page.layout import PageSpec, build_layout, validate_layout
from diffu_page.overlay import draw_label_overlay
from diffu_page.profiles import PROFILE_NAMES, profile
from diffu_page.render import PlaceholderRenderer
from diffu_page.sampler import DEFAULT_MIX, SPREAD_KINDS, LayoutMix, sample_kind, sample_page_spec
from diffu_page.validate import validate_labels, validate_table, validate_xml_roundtrip
from pydantic import BaseModel

# ---------------------------------------------------------------------------------------------------------
# Text corpus — Swedish-historical flavour, multiplied to fill dense pages. A real val.jsonl is used if given.
# ---------------------------------------------------------------------------------------------------------
_POOL = [
    "Goteborgs poliskammare den fjarde maj artonhundra attiotva",
    "anhallen for fylleri och forargelsevackande beteende a allman plats",
    "vittnen instamde och saken hanskots till rattens vidare provning",
    "han erkande att han varit drucken men nekade till oforratten",
    "hustrun uppgaf att mannen sedan langre tid forsummat sitt arbete",
    "arbetskarlen Anders Persson fran Landvetter socken forhordes",
    "protokollet upplastes och vederborande fingo del af innehallet",
    "malet uppskots till nasta sammantrade for hörande af flera vittnen",
    "en summa af tolf riksdaler femtio ore utdomdes i boter och kostnad",
    "den tilltalade forpliktades att gälda rattegangskostnaderna",
    "handlanden i staden intygade att varorna blifvit riktigt levererade",
    "kvinnan bekände att hon tillgripit brodet af hungersnod",
    "poliskonstapeln redogjorde for hvad vid tillfallet forekommit",
    "domaren erinrade om edens vigt och foljder af falsk utsago",
    "sedan parterna hörts afkunnades foljande utslag i saken",
    "till protokollet antecknades att svaranden uteblifvit utan laga forfall",
    "nämndemannen omnämnde att vägen varit svår att befara om vintern",
    "kronolänsmannen anhöll om biträde vid den fortsatta undersökningen",
    "sockenstämman beslöt att en ny skolgård skulle uppföras vid kyrkan",
    "församlingens fattigkassa befanns vara i behöfligt godt tillstånd",
    "af handlingarna framgår att köpet afslutats den nittonde februari",
    "de närvarande enades om att uppskjuta frågan till kommande år",
    "en anteckning gjordes rörande den aflidnes efterlemnade egendom",
    "bouppteckningen förrättades i vittnens närvaro å stället",
    "tvenne grannar tillkallades för att bestyrka riktigheten häraf",
]


def _load_pool(jsonl: str | None) -> list[str]:
    """Return the text pool, preferring lines from a real ``val.jsonl`` if one is given/found."""
    for path in [jsonl, "data_out_lejonet_128/val.jsonl", "data_out_lejonet/val.jsonl"]:
        if path and Path(path).exists():
            with Path(path).open() as fh:
                rows = [json.loads(line)["text"].replace("¬", "").strip() for line in fh]
            rows = [t for t in rows if 12 <= len(t) <= 90]
            if len(rows) >= 40:
                return rows
    return _POOL


# ---------------------------------------------------------------------------------------------------------
# Batch plan
# ---------------------------------------------------------------------------------------------------------
class DensityClass(BaseModel):
    """Line-metric preset for one density band + the text volume needed to fill the page at it."""

    model_config = {"frozen": True}

    name: str
    line_height: int
    line_gap: float
    para_gap: float
    target_lines: int


_DENSITY = {
    "letter": DensityClass(name="letter", line_height=58, line_gap=0.35, para_gap=0.6, target_lines=14),
    "medium": DensityClass(name="medium", line_height=36, line_gap=0.40, para_gap=0.5, target_lines=48),
    "dense": DensityClass(name="dense", line_height=24, line_gap=0.20, para_gap=0.25, target_lines=88),
    "very_dense": DensityClass(
        name="very_dense", line_height=17, line_gap=0.12, para_gap=0.12, target_lines=130
    ),
    "blank": DensityClass(name="blank", line_height=40, line_gap=0.4, para_gap=0.5, target_lines=4),
}

# 16-page density cycle: 3 letter (sparse tail), 7 medium, 4 dense, 2 very_dense -> median lands mid-band
# and the batch spans <=15 to >=100 lines.
_DENSITY_CYCLE = [
    "letter",
    "medium",
    "dense",
    "medium",
    "very_dense",
    "medium",
    "letter",
    "dense",
    "medium",
    "very_dense",
    "medium",
    "dense",
    "letter",
    "blank",
    "dense",
    "medium",
]


class PagePlan(BaseModel):
    """The deterministic recipe for one audited page: density, tone profile, and which physical features to
    force on so the batch covers every condition. A feature flag set to True forces its probability to 1.0;
    the others are forced to 0.0, so per-batch feature fractions are exact and reproducible."""

    model_config = {"frozen": True}

    index: int
    density: str
    profile_name: str
    template: str
    binding: bool
    glove: bool
    hole: bool
    curl: bool
    fold: bool


def _plan(n: int) -> list[PagePlan]:
    """Build the batch plan: density from the cycle, all 4 tone profiles round-robin (so C2's >=4 profiles
    always hold), and physical features assigned by simple modular rules to hit the target fractions."""
    plans: list[PagePlan] = []
    for i in range(n):
        density = _DENSITY_CYCLE[i % len(_DENSITY_CYCLE)]
        plans.append(
            PagePlan(
                index=i,
                density=density,
                profile_name=PROFILE_NAMES[i % len(PROFILE_NAMES)],
                template={"letter": "letter", "blank": "blank_leaf"}.get(density, "single_column"),
                binding=(i % 5 == 0),  # ~20% single-page inner gutter (C7)
                glove=(i % 5 == 2),  # ~20% gloved fingers (C6, in [0.15, 0.40])
                hole=(i % 8 == 4),  # holes present (C8)
                curl=(i % 5 == 3),  # curled corners (C8)
                fold=(i % 4 == 1),  # folds/creases (C8)
            )
        )
    return plans


def _page_text(pool: list[str], rng: np.random.Generator, target_lines: int) -> list[str]:
    """Enough Swedish-historical paragraphs to fill a page to ~``target_lines`` (oversupplied ~1.5x)."""
    n_sent = max(6, int(target_lines * 1.6))
    idx = rng.integers(0, len(pool), size=n_sent)
    sents = [pool[k] for k in idx]
    paras: list[str] = []
    i = 0
    while i < len(sents):
        step = int(rng.integers(4, 9))
        paras.append(" ".join(sents[i : i + step]))
        i += step
    return paras


def _aug_for(pl: PagePlan) -> AugmentConfig:
    """The page's aging config: the profile's tone knobs, with the planned physical features forced on (1.0)
    and the unplanned ones forced off (0.0) for exact per-batch fractions."""
    return profile(pl.profile_name).aug.model_copy(
        update={
            "glove_prob": 1.0 if pl.glove else 0.0,
            "hole_prob": 1.0 if pl.hole else 0.0,
            "curl_prob": 1.0 if pl.curl else 0.0,
            "fold_prob": 1.0 if pl.fold else 0.0,
        }
    )


def _spec_for(pl: PagePlan) -> PageSpec:
    dc = _DENSITY[pl.density]
    prose = pl.template == "single_column"
    return PageSpec(
        width=1654,
        height=2339,
        template=pl.template,
        line_height=dc.line_height,
        line_gap=dc.line_gap,
        para_gap=dc.para_gap,
        binding=pl.binding,
        recto=(pl.index % 2 == 0),
        page_number=str(pl.index * 2 + 3) if pl.index % 2 == 0 else None,
        marginalia=["N:o " + str(pl.index), "vide"]
        if (pl.index % 4 == 1 and pl.template != "letter")
        else [],
        # grounded text furniture, spread across the prose pages so the contact sheet shows them
        catchword="Anno" if pl.index % 3 == 0 else None,
        underline_prob=0.3 if (prose and pl.index % 3 == 1) else 0.0,
        section_headings=["Resolutio", "Utslag:"] if (prose and pl.index % 3 == 2) else [],
        enum_prob=0.6 if (prose and pl.index % 3 == 2) else 0.0,
        drop_cap=(
            pl.template in ("single_column", "letter") and pl.index % 4 == 0
        ),  # enlarged opening versal
        enum_marker="N:o",  # the CPU placeholder has no § glyph; N:o renders + is equally authentic
    )


# ---------------------------------------------------------------------------------------------------------
# Per-page metrics
# ---------------------------------------------------------------------------------------------------------
class PageResult(BaseModel):
    """Everything measured for one composed page."""

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    index: int
    profile_name: str
    density: str
    template: str
    n_lines: int
    luma_mean: float  # central-crop mean luminance (paper tone), bed/frame excluded
    contrast: float  # central-crop luminance stdev
    oob: int  # label points outside the page
    min_poly_pts: int  # smallest polygon vertex count (label validity)
    is_letter: bool  # a signature region present -> letter format (C9)
    features: dict[str, float]  # scan-feature trace (C6-C10)
    portrait: bool


def _central(page: np.ndarray) -> np.ndarray:
    h, w = page.shape[:2]
    return page[int(h * 0.2) : int(h * 0.8), int(w * 0.2) : int(w * 0.8)]


def _measure(page: np.ndarray, lines: list[LineLabel], pl: PagePlan, trace: dict[str, float]) -> PageResult:
    h, w = page.shape[:2]
    gray = cv2.cvtColor(_central(page), cv2.COLOR_BGR2GRAY).astype(np.float32)
    oob = sum(
        1
        for ln in lines
        for x, y in [*ln.polygon, *(p for word in ln.words for p in word.polygon)]
        if not (0 <= x < w and 0 <= y < h)
    )
    min_pts = min((len(ln.polygon) for ln in lines), default=0)
    return PageResult(
        index=pl.index,
        profile_name=pl.profile_name,
        density=pl.density,
        template=pl.template,
        n_lines=len(lines),
        luma_mean=float(gray.mean()),
        contrast=float(gray.std()),
        oob=oob,
        min_poly_pts=min_pts,
        is_letter=any(ln.region == "signature" for ln in lines),
        features=dict(trace),
        portrait=(h > w and 1.1 <= h / max(1, w) <= 2.2),
    )


# ---------------------------------------------------------------------------------------------------------
# Tabular / register pages (C12) — the corpus's non-prose layouts, checked separately from prose density
# ---------------------------------------------------------------------------------------------------------
_REG_NAMES = [
    "Anders Persson",
    "Erik Nilsson",
    "Johan Bergstrom",
    "Per Olsson",
    "Sven Larsson",
    "Nils Andersson",
    "Lars Svensson",
    "Olof Jonsson",
    "Carl Magnusson",
    "Brita Nilsdotter",
]

# Geometry-free aging for the structural validation compose: cell/rule positions stay in one frame.
_STRUCT_AUG = AugmentConfig(
    bed_prob=0.0,
    skew=0.0,
    rotate=(0.0, 0.0),
    microfilm_prob=0.0,
    dust_prob=0.0,
    correction_prob=0.0,
    stamp_prob=0.0,
    fold_prob=0.0,
    glove_prob=0.0,
    hole_prob=0.0,
    curl_prob=0.0,
)


class TableResult(BaseModel):
    """What a single tabular/register page carries: template, furniture rules, regions, bounds, and the
    structural problems ``validate_table`` found (empty == a correct table)."""

    model_config = {"frozen": True}

    template: str
    n_rules: int
    regions: list[str]
    n_lines: int
    oob: int
    problems: list[str] = []


def _register_text() -> list[str]:
    rows: list[str] = []
    for header, k in [("Vreta Kloster Bonder", 7), ("Fralse Bonder", 6), ("Skatte Bonder", 9)]:
        rows.append(header)
        rows += [f"{_REG_NAMES[i % len(_REG_NAMES)]}: {(i * 3) % 24 + 1}" for i in range(k)]
    return rows


def _table_pages(seed: int) -> list[tuple[str, np.ndarray, TableResult, list[LineLabel]]]:
    """Compose one of each non-prose layout (ruled grid / leader-dot register / numeric ledger) so the
    audit can prove the corpus's tabular structures are reproducible, each with furniture + valid labels.
    Returns the composed ``lines`` too, so the label-correctness check + overlay see the real table geometry."""
    tokens = [" ".join(n.split()[0] for n in _REG_NAMES) for _ in range(40)]
    specs = [
        (
            "semi_table",
            PageSpec(
                width=1654, height=2339, template="semi_table", line_height=44, table_rows=18, table_cols=7
            ),
            [" ".join(_REG_NAMES)] * 4,
            "faded_gray",
        ),
        (
            "register",
            PageSpec(width=1654, height=2339, template="register", line_height=44, line_gap=0.5),
            _register_text(),
            "warm_sepia",
        ),
        (
            "ledger",
            PageSpec(width=1654, height=2339, template="ledger", line_height=40, line_gap=0.35, table_cols=5),
            tokens,
            "bitonal",
        ),
    ]
    renderer = PlaceholderRenderer()
    out: list[tuple[str, np.ndarray, TableResult, list[LineLabel]]] = []
    for name, spec, text, pname in specs:
        layout = build_layout(spec, text, seed=seed)
        prof = profile(pname)
        page, lines = compose_page(layout, renderer, seed=seed, aug=prof.aug, paper_tone=prof.paper_tone)
        h, w = page.shape[:2]
        oob = sum(1 for ln in lines for x, y in ln.polygon if not (0 <= x < w and 0 <= y < h))
        # Validate STRUCTURE on a geometry-free compose: the scan warps (bed/rotate/skew) move cells and
        # furniture rules together, so structure is warp-invariant — but validate_table compares post-warp
        # cell polygons against the pre-warp rule coords in `layout.rules`, so it must see the unwarped frame.
        _, clean = compose_page(layout, renderer, seed=seed, aug=_STRUCT_AUG)
        res = TableResult(
            template=name,
            n_rules=len(layout.rules),
            regions=sorted({b.region for b in layout.boxes}),
            n_lines=len(lines),
            oob=oob,
            problems=validate_table(layout, clean),
        )
        out.append((f"{name} ({pname}) rules={res.n_rules}", page, res, lines))
    return out


# ---------------------------------------------------------------------------------------------------------
# Two-page spreads (C15) — the sampled fraction of prose-like pages emitted as an open-book opening
# ---------------------------------------------------------------------------------------------------------
class SpreadResult(BaseModel):
    """What one sampled spread carries: landscape geometry, two-leaf structure around a text-free central
    gutter, and label/layout validity — the spread counterpart of the single-leaf checks."""

    model_config = {"frozen": True}

    kind: str
    landscape: bool  # composed page is landscape-ish (an open-book opening, not a leaf)
    aspect: float
    n_boxes: int
    leaves: int  # populated leaves (2, or 1 when _spread blanks one side)
    gutter_clear: bool  # no layout box intrudes into the central gutter band
    oob: int  # composed label points outside the page
    problems: list[str] = []  # validate_layout defects (in-bounds boxes, no cross-region overlap)


def _spread_pages(
    seed: int, pool: list[str], renderer: PlaceholderRenderer, k: int = 3
) -> list[tuple[str, np.ndarray, SpreadResult]]:
    """Draw ``k`` spreads from the corpus-proportion sampler (spread_frac=1 over the spread-eligible kinds)
    and measure the spread invariants: landscape canvas, both leaves laid out around a clear central gutter,
    a clean ``validate_layout``, and in-bounds composed labels."""
    # sorted(): SPREAD_KINDS is a set — a fixed weight order keeps the draw reproducible across runs
    mix = LayoutMix(
        weights={kind: DEFAULT_MIX.weights[kind] for kind in sorted(SPREAD_KINDS)}, spread_frac=1.0
    )
    rng = random.Random(seed + 5150)
    out: list[tuple[str, np.ndarray, SpreadResult]] = []
    for i in range(k):
        kind, spec = sample_page_spec(rng, mix)
        spec = spec.model_copy(update={"enum_marker": "N:o"})  # the CPU placeholder has no § glyph
        text = _page_text(pool, np.random.default_rng(seed + 700 + i), 40)
        layout = build_layout(spec, text, seed=seed + 700 + i, measure=renderer.measure_width)
        cx = spec.width // 2
        g = max(spec.line_height, int(0.03 * spec.width))  # templates._spread's half-gutter
        left = [b for b in layout.boxes if b.x + b.w / 2 < cx]
        right = [b for b in layout.boxes if b.x + b.w / 2 >= cx]
        prof = profile(PROFILE_NAMES[i % len(PROFILE_NAMES)])
        page, lines = compose_page(
            layout, renderer, seed=seed + 700 + i, aug=prof.aug, paper_tone=prof.paper_tone
        )
        h, w = page.shape[:2]
        res = SpreadResult(
            kind=kind,
            landscape=(w > h and 1.15 <= w / max(1, h) <= 2.2),
            aspect=round(w / max(1, h), 2),
            n_boxes=len(layout.boxes),
            leaves=int(bool(left)) + int(bool(right)),
            gutter_clear=all(b.x + b.w <= cx - g or b.x >= cx + g for b in layout.boxes),
            oob=sum(1 for ln in lines for x, y in ln.polygon if not (0 <= x < w and 0 <= y < h)),
            problems=validate_layout(layout),
        )
        out.append((f"spread:{kind} leaves={res.leaves}", page, res))
    return out


# ---------------------------------------------------------------------------------------------------------
# C3 fill probe — a clean, geometry-free page so ink-fill of the slot can be measured directly
# ---------------------------------------------------------------------------------------------------------
def _fill_probe(pool: list[str], renderer: PlaceholderRenderer) -> tuple[float, float, float]:
    """Compose one page with all geometric aging disabled and measure how much of each body slot the ink
    fills. Returns (p75 fill of body lines, median fill, min fill) — C3 wants full lines >= 0.90."""
    spec = PageSpec(width=1654, height=2339, template="single_column", line_height=40, line_gap=0.4)
    flat = AugmentConfig(
        bed_prob=0.0,
        skew=0.0,
        rotate=(0.0, 0.0),
        microfilm_prob=0.0,
        dust_prob=0.0,
        correction_prob=0.0,
        stamp_prob=0.0,
        fold_prob=0.0,
        glove_prob=0.0,
        hole_prob=0.0,
        curl_prob=0.0,
    )
    text = [" ".join(pool[k % len(pool)] for k in range(j, j + 7)) for j in range(0, 56, 7)]
    layout = build_layout(spec, text, seed=0, measure=renderer.measure_width)
    body_w = max((b.w for b in layout.boxes if b.region == "body"), default=1)
    _, lines = compose_page(layout, renderer, seed=0, aug=flat)
    fills = sorted(
        (max(p[0] for p in ln.polygon) - min(p[0] for p in ln.polygon)) / body_w
        for ln in lines
        if ln.region == "body"
    )
    if not fills:
        return 0.0, 0.0, 0.0
    p75 = fills[int(len(fills) * 0.75)] if len(fills) > 1 else fills[0]
    return p75, statistics.median(fills), fills[0]


# ---------------------------------------------------------------------------------------------------------
# Contact sheets
# ---------------------------------------------------------------------------------------------------------
def _montage(images: list[np.ndarray], labels: list[str], cols: int, cell_w: int) -> np.ndarray:
    """A labelled grid of pages, each scaled to ``cell_w`` with a caption strip above it."""
    cells: list[np.ndarray] = []
    strip = 26
    max_h = 0
    scaled: list[np.ndarray] = []
    for img in images:
        h, w = img.shape[:2]
        cell = cv2.resize(img, (cell_w, int(h * cell_w / w)), interpolation=cv2.INTER_AREA)
        scaled.append(cell)
        max_h = max(max_h, cell.shape[0])
    for cell, label in zip(scaled, labels, strict=True):
        canvas = np.full((max_h + strip, cell_w, 3), 245, np.uint8)
        cv2.putText(canvas, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        canvas[strip : strip + cell.shape[0], : cell.shape[1]] = cell
        cells.append(canvas)
    rows = []
    for r in range(0, len(cells), cols):
        row = cells[r : r + cols]
        while len(row) < cols:
            row.append(np.full_like(cells[0], 245))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def _gt_reference(gt_dir: str, cols: int, cell_w: int, limit: int = 12) -> np.ndarray | None:
    """A montage of real corpus pages to eyeball the synthetic batch against."""
    d = Path(gt_dir)
    if not d.exists():
        return None
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    files = [p for p in files if "montage" not in p.name][:limit]
    if not files:
        return None
    imgs = [cv2.imread(str(p)) for p in files]
    imgs = [im for im in imgs if im is not None]
    return _montage(imgs, [p.name[:22] for p in files[: len(imgs)]], cols, cell_w) if imgs else None


# ---------------------------------------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------------------------------------
class Check(BaseModel):
    """One condition's verdict + the measured evidence behind it."""

    cid: str
    name: str
    ok: bool
    detail: str


def _evaluate(
    results: list[PageResult],
    fill: tuple[float, float, float],
    tables: list[TableResult],
    spreads: list[SpreadResult],
) -> list[Check]:
    counts = sorted(r.n_lines for r in results)
    median = statistics.median(counts)
    profiles_used = {r.profile_name for r in results}
    luma = [r.luma_mean for r in results]
    luma_spread = statistics.pstdev(luma) if len(luma) > 1 else 0.0
    contrasts = [r.contrast for r in results]
    feats: dict[str, int] = {}
    for r in results:
        for k in r.features:
            feats[k] = feats.get(k, 0) + 1
    n = len(results)
    glove_frac = feats.get("gloved_finger", 0) / n
    letter_frac = sum(r.is_letter for r in results) / n
    water = feats.get("foxing", 0) + feats.get("wash", 0) + feats.get("water_tide", 0)
    bleeds = [r.features["bleed"] for r in results if "bleed" in r.features]

    p75, med_fill, min_fill = fill
    return [
        Check(
            cid="C1",
            name="density",
            ok=(40 <= median <= 70 and counts[0] <= 15 and counts[-1] >= 100),
            detail=f"median={median:.0f} lines, span {counts[0]}..{counts[-1]} (want 40-70, <=15, >=100)",
        ),
        Check(
            cid="C2",
            name="scan-profile variety",
            ok=(len(profiles_used) >= 4 and luma_spread >= 22.0),
            detail=f"{len(profiles_used)} profiles, luma stdev={luma_spread:.1f} "
            f"(range {min(luma):.0f}..{max(luma):.0f}), contrast {min(contrasts):.0f}..{max(contrasts):.0f}",
        ),
        Check(
            cid="C3",
            name="horizontal fill",
            ok=(p75 >= 0.90),
            detail=f"body-line fill p75={p75:.2f}, median={med_fill:.2f}, shortest={min_fill:.2f} (want p75>=0.90)",
        ),
        Check(
            cid="C4",
            name="in-bounds",
            ok=(sum(r.oob for r in results) == 0),
            detail=f"{sum(r.oob for r in results)} label points outside the page across {n} pages",
        ),
        Check(
            cid="C5",
            name="label validity",
            ok=(min((r.min_poly_pts for r in results if r.n_lines), default=3) >= 3),
            detail=f"min polygon vertices={min((r.min_poly_pts for r in results if r.n_lines), default=0)} (want >=3)",
        ),
        Check(
            cid="C6",
            name="gloved fingers",
            ok=(0.15 <= glove_frac <= 0.40),
            detail=f"{feats.get('gloved_finger', 0)}/{n} pages = {glove_frac:.0%} (want 15-40%)",
        ),
        Check(
            cid="C7",
            name="single-page inner gutter",
            ok=(feats.get("gutter", 0) >= 2),
            detail=f"{feats.get('gutter', 0)}/{n} pages show a spine-side gutter",
        ),
        Check(
            cid="C8",
            name="damage variety",
            ok=(
                water >= 1
                and feats.get("hole", 0) >= 1
                and feats.get("fold", 0) >= 1
                and feats.get("curled_corner", 0) >= 1
            ),
            detail=f"water_stain={water}, hole={feats.get('hole', 0)}, fold={feats.get('fold', 0)}, "
            f"curled_corner={feats.get('curled_corner', 0)} (each want >=1)",
        ),
        Check(
            cid="C9",
            name="letter/sparse layout",
            ok=(0.12 <= letter_frac <= 0.30),
            detail=f"{sum(r.is_letter for r in results)}/{n} = {letter_frac:.0%} letter-format (want 12-30%)",
        ),
        Check(
            cid="C10",
            name="bleed variance",
            ok=(bool(bleeds) and min(bleeds) < 0.04 and max(bleeds) > 0.10),
            detail=f"bleed strengths {min(bleeds):.3f}..{max(bleeds):.3f} over {len(bleeds)} pages"
            if bleeds
            else "no bleed observed",
        ),
        Check(
            cid="C11",
            name="leaf/spread geometry",
            ok=(all(r.portrait for r in results) and all(s.landscape for s in spreads)),
            detail=f"{sum(r.portrait for r in results)}/{n} leaves portrait; "
            f"{sum(s.landscape for s in spreads)}/{len(spreads)} spreads landscape "
            f"(aspects {', '.join(f'{s.aspect:.2f}' for s in spreads) or 'none'})",
        ),
        Check(
            cid="C12",
            name="tabular/register variety",
            ok=(
                {"semi_table", "register", "ledger"} <= {t.template for t in tables}
                and all(t.n_rules > 0 for t in tables)
                and sum(t.oob for t in tables) == 0
                and all(not t.problems for t in tables)  # structurally correct, not just rules>0
                and any("reg_value" in t.regions for t in tables)
                and any("table_header" in t.regions for t in tables)
            ),
            detail="; ".join(
                f"{t.template}:{t.n_rules}rules,{t.oob}oob,{len(t.problems)}defects" for t in tables
            )
            or "no table pages",
        ),
        Check(
            cid="C15",
            name="two-page spreads",
            ok=(
                bool(spreads)
                and all(s.landscape and s.gutter_clear and s.n_boxes > 0 for s in spreads)
                and all(s.oob == 0 and not s.problems for s in spreads)
                and any(s.leaves == 2 for s in spreads)  # both leaves reachable (one may be blank ~15%)
            ),
            detail="; ".join(
                f"{s.kind}:asp={s.aspect:.2f},leaves={s.leaves},"
                f"gutter={'clear' if s.gutter_clear else 'HIT'},{s.oob}oob,{len(s.problems)}defects"
                for s in spreads
            )
            or "no spread pages",
        ),
    ]


def _sampler_check(seed: int, n: int = 4000, tol: float = 0.04) -> Check:
    """C13: the corpus-proportion sampler's empirical draw matches the measured LayoutMix within ``tol``."""
    rng = random.Random(seed)
    counts = Counter(sample_kind(rng) for _ in range(n))
    worst = max(abs(counts[k] / n - w) for k, w in DEFAULT_MIX.weights.items())
    hist = ", ".join(f"{k}={counts[k] / n:.2f}" for k in DEFAULT_MIX.weights)
    return Check(
        cid="C13",
        name="corpus-proportion sampler",
        ok=(worst <= tol),
        detail=f"max|emp-target|={worst:.3f}<={tol} over {n} draws ({hist})",
    )


def _sampler_pages(
    seed: int, pool: list[str], renderer: PlaceholderRenderer, k: int = 4
) -> list[tuple[str, np.ndarray]]:
    """A few pages drawn straight from the corpus-proportion sampler, for the contact sheet."""
    rng = random.Random(seed + 4242)
    out: list[tuple[str, np.ndarray]] = []
    for i in range(k):
        kind, spec = sample_page_spec(rng)
        spec = spec.model_copy(update={"enum_marker": "N:o"})  # the CPU placeholder has no § glyph
        text = _page_text(pool, np.random.default_rng(seed + 900 + i), 60)
        layout = build_layout(spec, text, seed=seed + 900 + i, measure=renderer.measure_width)
        prof = profile(PROFILE_NAMES[i % len(PROFILE_NAMES)])
        page, _ = compose_page(
            layout, renderer, seed=seed + 900 + i, aug=prof.aug, paper_tone=prof.paper_tone
        )
        out.append((f"sample:{kind}", page))
    return out


def _label_check(page_labels: list[PageLabels]) -> Check:
    """C14: every composed page's ground-truth labels are geometrically valid (polygons in bounds, baselines
    inside their line, word boxes inside their line, reading order a contiguous permutation) AND round-trip
    losslessly to PAGE / ALTO / JSON — proving the GT the training stack consumes is correct, not just visible."""
    problems: list[str] = []
    for labels in page_labels:
        problems += validate_labels(labels)
        problems += validate_xml_roundtrip(labels)
    n = len(page_labels)
    detail = (
        f"{n} pages: labels valid + PAGE/ALTO/JSON round-trip lossless"
        if not problems
        else f"{len(problems)} problem(s) over {n} pages; first: {problems[0]}"
    )
    return Check(cid="C14", name="label/XML correctness", ok=(not problems), detail=detail)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit diffu_page augmentation vs the GT-Lejonet corpus.")
    ap.add_argument("--n", type=int, default=16, help="pages in the batch (>=12)")
    ap.add_argument("--out", default="audit_out", help="output directory for pages + contact sheets")
    ap.add_argument("--seed", type=int, default=7, help="base seed (each page = seed + index)")
    ap.add_argument("--text-jsonl", default=None, help="optional val.jsonl to draw real line text from")
    ap.add_argument(
        "--gt-dir", default="gt_study_samples", help="real corpus pages for the reference montage"
    )
    ap.add_argument("--cols", type=int, default=4, help="contact-sheet columns")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "pages").mkdir(parents=True, exist_ok=True)
    pool = _load_pool(args.text_jsonl)
    renderer = PlaceholderRenderer()
    plans = _plan(args.n)

    pages: list[np.ndarray] = []
    labels: list[str] = []
    results: list[PageResult] = []
    page_labels: list[PageLabels] = []  # ground truth per page, checked for validity + XML round-trip (C14)
    overlays: list[tuple[str, np.ndarray, list[LineLabel]]] = []  # (caption, page, lines) to draw GT onto
    for pl in plans:
        spec = _spec_for(pl)
        text = _page_text(
            pool, np.random.default_rng(args.seed + pl.index), _DENSITY[pl.density].target_lines
        )
        layout = build_layout(spec, text, seed=args.seed + pl.index, measure=renderer.measure_width)
        trace: dict[str, float] = {}
        page, lines = compose_page(
            layout,
            renderer,
            seed=args.seed + pl.index,
            aug=_aug_for(pl),
            paper_tone=profile(pl.profile_name).paper_tone,
            trace=trace,
        )
        res = _measure(page, lines, pl, trace)
        results.append(res)
        h, w = page.shape[:2]
        page_labels.append(PageLabels(width=w, height=h, image=f"page_{pl.index:02d}.png", lines=lines))
        cv2.imwrite(str(out / "pages" / f"page_{pl.index:02d}_{pl.profile_name}_{pl.density}.png"), page)
        pages.append(page)
        labels.append(f"{pl.index} {pl.profile_name}/{pl.density} n={res.n_lines}")
        if pl.index == 5:  # a medium prose page carrying rubric + enumerator furniture — clear GT to overlay
            overlays.append((f"prose/{pl.profile_name}", page, lines))

    tables: list[TableResult] = []
    for label, page, res, tlines in _table_pages(args.seed):
        cv2.imwrite(str(out / "pages" / f"table_{res.template}.png"), page)
        pages.append(page)
        labels.append(label)
        tables.append(res)
        h, w = page.shape[:2]
        page_labels.append(PageLabels(width=w, height=h, image=f"table_{res.template}.png", lines=tlines))
        if res.template == "semi_table":  # the densest cell geometry — overlay it to eyeball row-major order
            overlays.append(("semi_table", page, tlines))

    for label, page in _sampler_pages(
        args.seed, pool, renderer
    ):  # pages drawn from the corpus-proportion mix
        pages.append(page)
        labels.append(label)

    spreads: list[SpreadResult] = []
    for label, page, sres in _spread_pages(args.seed, pool, renderer):  # two-page openings (C15)
        cv2.imwrite(str(out / "pages" / f"spread_{len(spreads)}_{sres.kind}.png"), page)
        pages.append(page)
        labels.append(label)
        spreads.append(sres)

    for name, page, lines in overlays:  # GT drawn on the ink: polygon / baseline / word box / reading order
        pages.append(draw_label_overlay(page, lines))
        labels.append(f"overlay:{name}")

    fill = _fill_probe(pool, renderer)
    checks = [
        *_evaluate(results, fill, tables, spreads),
        _sampler_check(args.seed),
        _label_check(page_labels),
    ]

    contact = _montage(pages, labels, args.cols, cell_w=360)
    cv2.imwrite(str(out / "contact_sheet.png"), contact)
    gt = _gt_reference(args.gt_dir, args.cols, cell_w=360)
    if gt is not None:
        cv2.imwrite(str(out / "gt_reference.png"), gt)

    print("=" * 78)
    print(f"diffu_page augmentation audit — {args.n} pages, seed {args.seed}")
    print("=" * 78)
    n_fail = 0
    for c in checks:
        mark = "PASS" if c.ok else "FAIL"
        if not c.ok:
            n_fail += 1
        print(f"  [{mark}] {c.cid} {c.name:<26} {c.detail}")
    print("-" * 78)
    print(
        f"  {len(checks) - n_fail}/{len(checks)} conditions PASS"
        + (" — ALL PASS" if n_fail == 0 else f" — {n_fail} FAIL")
    )
    print(f"  contact sheet : {out / 'contact_sheet.png'}")
    if gt is not None:
        print(f"  gt reference  : {out / 'gt_reference.png'}")
    print(f"  pages         : {out / 'pages'}/")
    print("=" * 78)


if __name__ == "__main__":
    main()
