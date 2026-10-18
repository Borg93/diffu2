"""Ray page pipeline — coherent, quality-gated document pages.

A page is COHERENT: its text is a consecutive passage from ONE real document (train.jsonl line names are
``DOC_LINE``, so grouping by ``DOC`` recovers the passage) written in ONE writer's hand (a style crop from
that same document). Templates come from the corpus-proportion sampler (prose / landskap / semi_table /
register / ledger / letter / blank). Each composed page is GATED: a sample of its lines is read back by
TrOCR and the page is kept only if the FRACTION of bad lines (line CER above a deliberately lenient
threshold) stays under a cap — low quality is fine, we train on it. Uses Ray actors (stateful: model +
recognizer loaded once per worker).

BANK MODE (``PageConfig.bank``): render + read-back are replaced by a pre-generated Lance line-bank (see
:mod:`diffu_runner.bank`) — lines were rendered and TrOCR-gated ONCE by the line pipeline, so pages become
pure CPU compose over already-gated pixels and the actors need no GPU at all.
"""

from __future__ import annotations

import io
import json
import logging
import random
from pathlib import Path
from typing import TextIO

import numpy as np
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PageConfig(BaseModel):
    """Knobs for a page-generation run (frozen)."""

    model_config = {"frozen": True}

    ckpt: str
    train_jsonl: str  # TRAIN split only (coherent docs + writer styles); val.jsonl stays held-out
    out: str
    # Lance line-bank path. When set, every handwriting line comes PRE-RENDERED (and pre-CER-gated) from
    # this bank instead of being re-rendered by the model — line-bank reuse is the 40× lever that turns
    # the 1M-page run from ~79 GPU-days into ~1 CPU-day (ckpt/recognizer are never loaded; see run_pages).
    bank: str | None = None
    paper_dir: str = "assets/paper_gtl"
    recognizer: str = "Riksarkivet/trocr-base-handwritten-hist-swe-3"
    line_cer_max: float = 0.35  # a body line is "bad" if its CER exceeds this (pure scribble reads ~1.0)
    max_bad_frac: float = 0.12  # drop the whole page if more than this fraction of its lines are bad
    gate_max_lines: int = 48  # cap recognizer calls/page: read every line up to this, sample beyond (tables)
    num_steps: int = 24
    cfg_scale: float = 5.0
    # Optional Stage-3 LoRA adapter (dir or Hub id) applied on top of ``ckpt`` in the render-path
    # BatchedDiffuRenderer (``diffu.lora.load_lora``). Ignored in bank mode — that path loads no model.
    lora: str | None = None
    lora_scale: float = 1.0  # adapter strength in [0, 1+] (1.0 = full)
    upscale: bool = False  # opt-in FINAL stage: PiD 4× super-resolve each KEPT page (needs its own GPU actor)
    upscale_max_side: int = 1024  # PiD works the page at this long side → ~4× output (demo parity, ~24 GiB)
    # Structured GT written per kept page to <out>/page/page_NNNNNN<ext> (the GT-Lejonet img/ + page/
    # layout): "page" (PAGE-XML, PRImA-valid), "alto" (ALTO v4), "json". Without these the training
    # corpus would be images with no line geometry — the compositor's polygons/baselines/words would be
    # computed for the gate and then thrown away.
    formats: list[str] = ["page"]


def coherent_docs(train_jsonl: str, min_lines: int = 6) -> list[dict]:
    """Group train lines into coherent documents: ``{"texts": [consecutive lines], "style": crop}``. Text
    and writer both come from ONE document, so a page built from a doc is coherent in content AND hand."""
    rows = [json.loads(x) for x in Path(train_jsonl).read_text(encoding="utf-8").splitlines()]
    by_doc: dict[str, list[dict]] = {}
    for r in rows:
        doc = Path(r["image"]).stem.rsplit("_", 1)[0]  # 003814_007 -> 003814
        by_doc.setdefault(doc, []).append(r)
    docs: list[dict] = []
    for lines in by_doc.values():
        lines.sort(key=lambda r: r["image"])
        texts = [t for t in (ln["text"].replace("¬", "").strip() for ln in lines) if 2 <= len(t) <= 90]
        if len(texts) >= min_lines:
            docs.append({"texts": texts, "style": lines[len(lines) // 2]["image"]})
    return docs


def _crop(page: np.ndarray, poly: list[tuple[int, int]]) -> Image.Image:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, y0, x1, y1 = (
        max(0, min(xs)),
        max(0, min(ys)),
        min(page.shape[1], max(xs)),
        min(page.shape[0], max(ys)),
    )
    return Image.fromarray(page[y0:y1, x0:x1][:, :, ::-1])  # BGR->RGB


class PageActor:
    """Render + compose + CER-gate a batch of coherent pages. Model + recognizer loaded once per worker.

    With ``cfg.bank`` set, the render + read-back stages are replaced by bank sampling (:meth:`make`
    routes to :meth:`_make_from_bank`) — no model, no recognizer, no GPU."""

    def __init__(self, cfg: PageConfig) -> None:
        self.cfg = cfg
        if cfg.bank is not None:
            # BANK mode: lines were rendered + TrOCR-gated ONCE at bank build (the line pipeline), so this
            # actor loads NO model and NO recognizer — composition is pure CPU and run_pages places it
            # with num_gpus=0. This is what turns the 1M-page run from GPU-days into a CPU-day.
            from .bank import LineBank

            self.bank = LineBank(cfg.bank)
            return
        import torch
        from diffu_page.render import BatchedDiffuRenderer
        from safetensors import safe_open

        from diffu.config import Config
        from diffu.recognizer import load_recognizer

        dcfg = Config()
        dcfg.cond.glyph_line = True
        dcfg.cond.style_in_context = False
        dcfg.data.line_height = 128
        with safe_open(cfg.ckpt, framework="pt") as h:
            dcfg.backbone.qk_norm = "rms_norm" if any(".norm_q." in k for k in h.keys()) else None  # noqa: SIM118
        dcfg.cond.use_unifont = dcfg.backbone.qk_norm is not None
        torch.set_float32_matmul_precision("high")
        self.renderer = BatchedDiffuRenderer(
            cfg.ckpt, cfg=dcfg, num_steps=cfg.num_steps, cfg_scale=cfg.cfg_scale, device="cuda",
            lora=cfg.lora, lora_scale=cfg.lora_scale,
        )
        self.proc, self.rec = load_recognizer(cfg.recognizer, "cuda")

    def make(self, jobs: list[dict]) -> list[dict]:
        """``jobs``: list of ``{"doc": {texts, style}, "kind", "spec", "seed", "profile"}``. Returns the
        KEPT (gated) pages as ``{"png": bytes, "template", "page_cer", "n_lines"}``. One bad job is logged
        (with its seed, so it can be reproduced) and skipped — it never kills the rest of the batch."""
        if self.cfg.bank is not None:
            return self._make_from_bank(jobs)
        from diffu_page.compose import compose_page_from_inks
        from diffu_page.layout import PageSpec, build_layout
        from diffu_page.profiles import profile

        from diffu.recognizer import char_error_rate, read_lines

        layouts, styles_per_page, metas = [], [], []
        for jb in jobs:
            try:
                spec = PageSpec.model_validate(jb["spec"])
                # coherent PARAGRAPHS: join runs of consecutive doc lines so wrapping fills the FULL column
                # width (not one short line per paragraph). Oversupply so dense pages fill; tables ignore it.
                prng = random.Random(jb["seed"])
                texts, paras, i = jb["doc"]["texts"], [], 0
                while i < len(texts):
                    k = prng.randint(3, 8)
                    paras.append(" ".join(texts[i : i + k]))
                    i += k
                page_text = (paras * (40 // max(1, len(paras)) + 1))[:40]
                layout = build_layout(spec, page_text, seed=jb["seed"], measure=self.renderer.measure_width)
            except Exception:
                logger.exception("page job seed=%s failed at layout — skipped", jb["seed"])
                continue
            layouts.append(layout)
            styles_per_page.append([jb["doc"]["style"]] * len(layout.boxes))
            metas.append(jb)

        from diffu_page.render import render_layouts

        inks = render_layouts(self.renderer, layouts, styles_per_page)
        kept: list[dict] = []
        for layout, ink, jb in zip(layouts, inks, metas, strict=True):
            try:
                prof = profile(jb["profile"])
                page, lines = compose_page_from_inks(
                    layout,
                    ink,
                    paper_dir=self.cfg.paper_dir,
                    seed=jb["seed"],
                    aug=prof.aug,
                    paper_tone=prof.paper_tone,
                )
                body = [ln for ln in lines if ln.text.strip() and len(ln.polygon) >= 3]
                if not body:
                    continue
                sample = (  # job-seeded sampling: the gate verdict is reproducible from the job seed
                    body
                    if len(body) <= self.cfg.gate_max_lines
                    else random.Random(jb["seed"]).sample(body, self.cfg.gate_max_lines)
                )
                preds = read_lines(self.proc, self.rec, [_crop(page, ln.polygon) for ln in sample])
                cers = [char_error_rate(p, ln.text) for p, ln in zip(preds, sample, strict=True)]
                bad_frac = sum(c > self.cfg.line_cer_max for c in cers) / len(cers)
                cer = float(np.mean(cers))
                if bad_frac <= self.cfg.max_bad_frac:  # reject scribble-heavy pages, not just high-mean ones
                    buf = io.BytesIO()
                    Image.fromarray(page[:, :, ::-1]).save(buf, format="PNG")
                    kept.append(
                        {
                            "png": buf.getvalue(),
                            "template": jb["kind"],
                            "page_cer": round(cer, 3),
                            "n_lines": len(lines),
                            # the STRUCTURED GT (polygons/baselines/words/reading order) — serialized so
                            # the driver can export PAGE/ALTO next to the numbered PNG; dropping it here
                            # would leave a training corpus of images with no line geometry
                            "labels": _labels_json(page, lines),
                            # provenance: the page is reproducible/traceable from (seed, profile, style)
                            # + the run-level run_meta.json (ckpt fingerprint, code version, PageConfig)
                            "seed": jb["seed"],
                            "profile": jb["profile"],
                            "style": jb["doc"]["style"],  # ONE writer crop styles the whole page
                        }
                    )
            except Exception:
                logger.exception("page job seed=%s failed at compose/gate — skipped", jb["seed"])
        return kept

    def _make_from_bank(self, jobs: list[dict]) -> list[dict]:
        """Bank-mode :meth:`make`: same jobs in, same kept-page payload out — but every handwriting line
        is a PRE-RENDERED bank row (no model) and the gate is pure math over the rows' STORED CERs (no
        recognizer). Layout geometry still comes from the job's spec/seed; each handwriting box is then
        filled with the next bank line IN RUN ORDER and takes that line's TEXT, so the label always
        matches the seated strokes exactly — bank pixels are fixed, and a re-wrapped fragment of joined
        bank texts would lie about its ink. Printed form furniture is typeset as in the render path."""
        from diffu_page.compose import compose_page_from_inks
        from diffu_page.layout import PageSpec, build_layout
        from diffu_page.profiles import profile
        from diffu_page.render import PlaceholderRenderer
        from diffu_page.render.matte import ink_from_pil

        from diffu.generate import ink_crop

        from .bank import empirical_measure, gate_stored_cers

        typeset = PlaceholderRenderer()
        kept: list[dict] = []
        for jb in jobs:
            try:
                spec = PageSpec.model_validate(jb["spec"])
                prng = random.Random(jb["seed"])
                # one page's worth of consecutive-run (or one-hand) rows; 40 mirrors the render path's
                # page_text oversupply cap below. Keep the DATASET row ids — they are the page's line
                # provenance (exactly which bank rows were seated, manifest "bank_rows").
                row_ids = self.bank.sample_run(prng, n_lines=40)
                fetched = self.bank.fetch(row_ids)
                # ink_crop ONCE per row — the same post-generate step BatchedDiffuRenderer.render_jobs
                # applies; the cropped pixels feed BOTH the measure callback and the per-box matting
                fetched = [(text, cer, ink_crop(pil)) for text, cer, pil in fetched]
                # coherent PARAGRAPHS, exactly like the render path: join runs of consecutive bank lines
                # so wrapping fills the FULL column width; oversupply to 40 so dense pages fill.
                texts, paras, i = [t for t, _, _ in fetched], [], 0
                while i < len(texts):
                    k = prng.randint(3, 8)
                    paras.append(" ".join(texts[i : i + k]))
                    i += k
                page_text = (paras * (40 // max(1, len(paras)) + 1))[:40]
                # wrap by the run's own measured ink aspect ratios (the ink IS the truth in bank mode),
                # standing in for the render path's model width predictor
                layout = build_layout(spec, page_text, seed=jb["seed"], measure=empirical_measure(fetched))
                # belt-and-braces behind prose_only: a handwriting slot too short to hold a legible bank
                # sentence (a table cell that slipped through) would seat unreadable microtext — reject
                # the whole page rather than emit it with confident full-sentence labels
                micro = [b for b in layout.boxes if not b.printed and b.h < 24]
                if micro:
                    logger.warning(
                        "bank page seed=%s has %d sub-24px handwriting slots (template %s) — skipped",
                        jb["seed"], len(micro), jb["kind"],
                    )  # fmt: skip
                    continue
                boxes, inks = [], []
                used: set[int] = set()  # bank rows actually seated — the gate scores THESE, once each
                hand_i = 0
                for box in layout.boxes:
                    if box.printed:  # typeset form furniture — same split as render_layouts
                        boxes.append(box)
                        inks.append(typeset.render(box.text, box.h))
                        continue
                    # dense pages cycle the run, mirroring the render path's repeated-paragraph oversupply
                    text, _, pil = fetched[hand_i % len(fetched)]
                    used.add(hand_i % len(fetched))
                    hand_i += 1
                    boxes.append(box.model_copy(update={"text": text}))  # label == the seated bank line
                    inks.append(ink_from_pil(pil, box.h))  # same matting as the render path post-generate
                layout = layout.model_copy(update={"boxes": boxes})
                prof = profile(jb["profile"])
                page, lines = compose_page_from_inks(
                    layout,
                    inks,
                    paper_dir=self.cfg.paper_dir,
                    seed=jb["seed"],
                    aug=prof.aug,
                    paper_tone=prof.paper_tone,
                )
                if not any(ln.text.strip() and len(ln.polygon) >= 3 for ln in lines):
                    continue  # nothing legible seated — mirror the render path's empty-body skip
                keep, cer = gate_stored_cers(
                    [fetched[k][1] for k in sorted(used)],
                    line_cer_max=self.cfg.line_cer_max,
                    max_bad_frac=self.cfg.max_bad_frac,
                )
                if keep:
                    buf = io.BytesIO()
                    Image.fromarray(page[:, :, ::-1]).save(buf, format="PNG")
                    kept.append(
                        {
                            "png": buf.getvalue(),
                            "template": jb["kind"],
                            "page_cer": round(cer, 3),
                            "n_lines": len(lines),
                            "labels": _labels_json(page, lines),  # structured GT — see make()
                            # provenance: exactly which bank rows were seated (dataset row ids — join
                            # back to the bank for each line's text/style/doc/cer) + the job's knobs
                            "seed": jb["seed"],
                            "profile": jb["profile"],
                            "bank_rows": [row_ids[k] for k in sorted(used)],
                        }
                    )
            except Exception:
                logger.exception("bank page job seed=%s failed — skipped", jb["seed"])
        return kept


class PidActor:
    """FINAL page stage: PiD 4× super-resolution. Its own GPU actor because PiD (~9 GiB weights + a large
    activation footprint) can't share a device with the render+recognizer :class:`PageActor` without OOM —
    keeping it a separate actor lets Ray place it on a free GPU and keeps that heavy model out of the
    generation loop. Loaded once per worker; fallback-safe (a failed upscale returns the page unchanged)."""

    def __init__(self, cfg: PageConfig) -> None:
        import torch
        from diffu_page.upscale import load_pid

        torch.set_float32_matmul_precision("high")
        self.cfg = cfg
        load_pid()  # build + cache PiD now (inside this GPU actor), so make() pays only for decodes

    def upscale(self, pages: list[dict]) -> list[dict]:
        """Replace each kept page's ``png`` bytes with the 4×-upscaled PNG (unchanged on any failure)."""
        import io

        from diffu_page.upscale import pid_upscale
        from PIL import Image

        out: list[dict] = []
        for p in pages:
            before = Image.open(io.BytesIO(p["png"])).convert("RGB")
            hires = pid_upscale(before, max_side=self.cfg.upscale_max_side)
            if hires is not None and hires is not before:
                buf = io.BytesIO()
                hires.save(buf, format="PNG")
                out.append({**p, "png": buf.getvalue(), "upscaled": [hires.width, hires.height]})
            else:
                out.append({**p, "upscaled": None})  # kept as-is; label geometry stays valid (never scaled)
        return out


_HEADERS = [
    ["Kongl. Maj:ts Befallningshafvande."], ["Rådhusrätten i Göteborg.", "Anno 1788."],
    ["Protocoll", "hållet vid Härads Tinget."], ["Till Magistraten."], ["Anno 1723 d. 14 Junii."],
]  # fmt: skip
_MARGS = ["Fylleri.", "Resolution.", "Vittnen.", "Utslag.", "Böter.", "Käranden.", "Svaranden.", "Afslag.", "Vide.", "Bifall."]  # fmt: skip
_CATCH = ["Anno", "och", "att", "den", "som", "för", "Vidare"]
_KEYS = ["N:o 1", "N:o 2", "N:o 3", "Resol.", "vide", "N:B."]
_SECTIONS = [["Resolutio", "Utslag:"], ["Företedt.", "Beslut."], ["Frågan.", "Svaret."]]


def varied_job(rng: random.Random, docs: list[dict], seed: int, *, prose_only: bool = False) -> dict:
    """A HIGH-VARIANCE page job: a corpus-weighted template + a coherent doc (text+writer) + heavily
    randomized furniture (binding, headers, marginalia, keyed notes, catchword, drop-cap, enumerators,
    density) so the 10k pages span the corpus's real diversity, not one look.

    ``prose_only`` (bank mode) zeroes the tabular kinds out of the mix: a bank row is a whole 8-65-char
    SENTENCE, and seating one into a narrow register/ledger number cell shrinks it ~10x into unreadable
    microtext with a full-sentence label. Tabular pages need the render path (which writes short typed
    cell values) or a future keyed cell-bank."""
    from diffu_page.layout import PageSpec
    from diffu_page.profiles import PROFILE_NAMES
    from diffu_page.sampler import DEFAULT_MIX, LayoutMix, sample_page_spec

    mix = None
    if prose_only:
        tabular = {"register", "landskap", "ledger", "semi_table"}
        mix = LayoutMix(weights={k: w for k, w in DEFAULT_MIX.weights.items() if k not in tabular})
    kind, spec = sample_page_spec(random.Random(seed), mix)
    prose = spec.template in ("single_column", "multi_column")
    u: dict = {"recto": rng.random() < 0.5}
    # in-band now the spine reads — but never on a spread: its spine is the CENTRAL text-free gutter,
    # and a single-edge binding would warp/clamp one leaf's outer margin instead.
    u["binding"] = not spec.spread and rng.random() < 0.22
    if rng.random() < 0.7:
        u["page_number"] = str(rng.randint(1, 480))
    if rng.random() < 0.6:
        u["header_lines"] = rng.choice(_HEADERS)
    if prose and rng.random() < 0.6:
        u["marginalia"] = rng.sample(_MARGS, k=rng.randint(1, 3))
    if prose and rng.random() < 0.45:
        u["margin_keys"] = rng.sample(_KEYS, k=rng.randint(2, 4))
    if rng.random() < 0.3:
        u["catchword"] = rng.choice(_CATCH)
    if prose and rng.random() < 0.5:
        u["drop_cap"] = True
    if prose and rng.random() < 0.5:
        u["section_headings"] = rng.choice(_SECTIONS)
        u["enum_prob"] = rng.choice([0.0, 0.5, 0.8])
    if prose and rng.random() < 0.25:
        u["underline_prob"] = 0.3
    if prose and rng.random() < 0.5:  # vary density at GT scale: airy ~13-line pages down to denser ~24-line
        u["line_height"] = rng.randint(84, 118)
        u["line_gap"] = round(rng.uniform(0.25, 0.55), 2)
    u["enum_marker"] = "§"
    spec = spec.model_copy(update=u)
    return {
        "doc": rng.choice(docs),
        "kind": kind,
        "spec": PageSpec.model_validate(spec).model_dump(),
        "seed": seed,
        "profile": rng.choice(PROFILE_NAMES),
    }


def _upscale_written_pages(cfg: PageConfig, out: Path, *, gpus: int, batch: int = 8) -> int:
    """FINAL stage: 4×-upscale every written page PNG in place through a pool of PiD GPU actors.

    Runs AFTER generation so it can reuse the freed GPUs — PiD needs a whole device and can't co-reside with
    a :class:`PageActor`. Overwrites ``img/page_*.png`` with the hi-res version; the upscaled COUNT goes into
    the returned stats (the manifest is already written and untouched here). The manifest carries no
    coordinate labels (pages store only the PNG + CER metadata), so rescaling the image is label-safe; if
    per-line labels are ever emitted here, scale their polygons to match (note: pid_upscale resizes the long
    side to ``upscale_max_side`` BEFORE the 4×, so the output is not an integer multiple of the original).
    """
    import ray

    pngs = sorted((out / "img").glob("page_*.png"))
    if not pngs:
        return 0
    # In bank mode ``gpus`` is a plain CPU-worker count, so it can exceed the physical GPUs — a pool of
    # unschedulable num_gpus=1 actors would make the ray.get below block forever. Cap at what the cluster
    # actually has; with no GPU at all, upscaling is impossible — skip loudly instead of hanging.
    cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
    if cluster_gpus == 0:
        logger.warning("upscale requested but the Ray cluster has no GPU — skipping the upscale stage")
        return 0
    pid_actor = ray.remote(num_gpus=1)(PidActor)
    # fault tolerance: restart a dead actor + retry its tasks (mirrors the line pipeline's actor kwargs)
    workers = [
        pid_actor.options(max_restarts=3, max_task_retries=3).remote(cfg)
        for _ in range(max(1, min(gpus, cluster_gpus)))
    ]
    done = 0
    i = 0
    try:
        while i < len(pngs):
            futs, assigned = [], []
            for w in workers:
                chunk = pngs[i : i + batch]
                if not chunk:
                    break
                i += batch
                jobs = [{"png": p.read_bytes(), "path": str(p)} for p in chunk]
                assigned.append([j["path"] for j in jobs])
                futs.append(w.upscale.remote(jobs))
            for paths, results in zip(assigned, ray.get(futs), strict=True):
                for path, r in zip(paths, results, strict=True):
                    Path(path).write_bytes(r["png"])
                    done += r["upscaled"] is not None
    finally:
        for w in workers:  # release the PiD GPUs even on failure
            ray.kill(w)
    return done


def _existing_pages(img_dir: Path) -> int:
    """Pages a previous run already wrote into this out dir. Reruns RESUME from this count so the numbering
    and the append-mode manifest stay consistent — restarting at 0 would rewrite ``page_000000.png`` with
    new content while the manifest kept the stale row describing the old one."""
    return len(list(img_dir.glob("page_*.png")))


def _labels_json(page, lines) -> str:
    """Serialize the compositor's structured GT for the driver — the actor→driver labels hand-off."""
    from diffu_page.labels import PageLabels

    return PageLabels(
        width=page.shape[1], height=page.shape[0], image="page.png", lines=lines
    ).model_dump_json()


def _write_kept(
    pages: list[dict], out: Path, manifest: TextIO, kept: int, target: int, formats: list[str] | None = None
) -> int:
    """Write one batch's kept pages (PNG + structured-GT exports + manifest row), stopping EXACTLY at
    ``target``; returns the new kept count. Pure driver logic, factored out so the no-overshoot
    accounting is unit-testable. Labels land in ``<out>/page/page_NNNNNN<ext>`` (the GT-Lejonet
    ``img/`` + ``page/`` corpus layout) with the image filename stamped in, so HTR tooling's
    ``<stem>.page.xml`` auto-matching works."""
    from diffu_page.labels import EXTENSIONS, PageLabels, export

    for p in pages:
        if kept >= target:
            break
        stem = f"page_{kept:06d}"
        (out / "img" / f"{stem}.png").write_bytes(p["png"])
        label_paths: dict[str, str] = {}
        if formats and "labels" in p:
            labels = PageLabels.model_validate_json(p["labels"]).model_copy(update={"image": f"{stem}.png"})
            (out / "page").mkdir(exist_ok=True)
            for fmt in formats:
                path = out / "page" / f"{stem}{EXTENSIONS[fmt]}"
                path.write_text(export(labels, fmt), encoding="utf-8")
                label_paths[fmt] = str(path)
        row = {
            "image": f"{out}/img/{stem}.png",
            "template": p["template"],
            "page_cer": p["page_cer"],
            "n_lines": p["n_lines"],
            **({"labels": label_paths} if label_paths else {}),
            # per-page provenance (mode-dependent): render pages carry the writer crop ("style");
            # bank pages carry the exact bank rows seated ("bank_rows"); both carry seed + profile
            **{k: p[k] for k in ("seed", "profile", "style", "bank_rows") if k in p},
        }
        manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
        kept += 1
    return kept


def run_pages(
    cfg: PageConfig, target: int, *, gpus: int = 1, batch: int = 16, seed: int = 0, max_rounds: int = 200
) -> tuple[str, dict]:
    """Generate ``target`` gated coherent pages via a pool of Ray ``PageActor``s (one per GPU). Each round
    dispatches ``batch`` varied jobs per worker, collects the KEPT (gated) pages, and repeats until target —
    or until ``max_rounds``, in which case the shortfall is logged and recorded in stats (no infinite loop
    on a pathological reject rate). Writes page PNGs + a manifest; a rerun into the same ``cfg.out`` resumes
    from the pages already on disk. When ``cfg.upscale`` is set, a FINAL PiD 4× stage runs over every kept
    page once generation is done (the render actors are released first, so PiD reuses the GPUs).
    With ``cfg.bank`` set the actors compose on CPU (``num_gpus=0``) and ``gpus`` is read as a plain
    worker count. Returns ``(path, stats)``."""
    if gpus < 1:
        raise ValueError(f"gpus must be >= 1 (an empty actor pool would loop forever), got {gpus}")
    if cfg.upscale:  # fail fast BEFORE burning GPU-hours on pages the final stage can't upscale
        from diffu_page.upscale import pid_available

        if not pid_available():
            raise ValueError("cfg.upscale is set but PiD is not installed — install diffu-page[upscale]")

    import ray

    if not ray.is_initialized():
        ray.init(include_dashboard=False)
    # bank mode: page text + hand come from the bank rows inside the actor, so the coherent-doc pool (and
    # its train.jsonl read) is skipped — the job's "doc" is unused filler keeping varied_job one-shaped.
    docs = [{"texts": [], "style": ""}] if cfg.bank is not None else coherent_docs(cfg.train_jsonl)
    # bank mode composes on CPU (no model, no recognizer) — num_gpus=0, and `gpus` degrades to a plain
    # worker count of parallel CPU composers; the render path still pins one whole GPU per actor.
    actor = ray.remote(num_gpus=0 if cfg.bank is not None else 1)(PageActor)
    # fault tolerance: restart a dead actor + retry its tasks (mirrors the line pipeline's actor kwargs)
    workers = [actor.options(max_restarts=3, max_task_retries=3).remote(cfg) for _ in range(gpus)]

    out = Path(cfg.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    kept = _existing_pages(out / "img")  # rerun-resume: numbering + append-mode manifest stay consistent
    if kept:
        logger.info("resuming: %d pages already in %s", kept, out / "img")
    rng = random.Random(seed)
    attempted = job_i = rounds = 0
    try:
        with (out / "manifest.jsonl").open("a", encoding="utf-8") as manifest:
            while kept < target and rounds < max_rounds:
                futs = []
                for w in workers:
                    jobs = [
                        varied_job(rng, docs, seed * 10_000_000 + job_i + k, prose_only=bool(cfg.bank))
                        for k in range(batch)
                    ]
                    job_i += batch
                    futs.append(w.make.remote(jobs))
                for pages in ray.get(futs):
                    attempted += batch
                    kept = _write_kept(pages, out, manifest, kept, target, formats=cfg.formats)
                    if kept >= target:
                        break
                manifest.flush()
                rounds += 1
    finally:
        for w in workers:  # release the render GPUs even on failure (PiD — or the next run — needs them)
            ray.kill(w)

    if kept < target:
        logger.warning("run_pages stopped at max_rounds=%d with %d/%d pages kept", max_rounds, kept, target)
    stats = {"kept": kept, "attempted": attempted, "rounds": rounds, "shortfall": max(0, target - kept)}
    if cfg.upscale:
        stats["upscaled"] = _upscale_written_pages(cfg, out, gpus=gpus)
    # run-level provenance next to the corpus: which model/bank, which code, which knobs made these
    # pages (per-page provenance lives in the manifest rows: seed/profile/style|bank_rows/labels)
    from .pipeline import ckpt_fingerprint, code_version, write_run_meta

    source = (
        {"bank": cfg.bank, "bank_rows_total": _bank_len(cfg.bank)}
        if cfg.bank is not None
        else {"ckpt": ckpt_fingerprint(cfg.ckpt)}
    )
    write_run_meta(
        str(out),
        {
            "kind": "pages",
            "page_config": cfg.model_dump(),
            "args": {"target": target, "gpus": gpus, "batch": batch, "seed": seed, "max_rounds": max_rounds},
            "source": source,
            "code": code_version(),
            "stats": stats,
        },
    )
    return str(out), stats


def _bank_len(path: str) -> int:
    """Row count of the source bank — a cheap identity check for 'which bank built these pages'."""
    import lance

    return lance.dataset(path).count_rows()
