"""The Ray-Data pipeline: forge a synthetic HTR line dataset — generate, gate, write — in one stream.

    work-list (train-only text × style)
        → DiffuActor  (GPU, generate)          concurrency = generator GPUs
        → ink flag    (CPU, free)              rows KEEP an ``ink_ok`` flag, so rejects are counted/retried
        → TrOCRActor  (fractional GPU, score)  many light readers per GPU
        → ink + CER filter
        → Lance dataset (only clean lines)

LEAKAGE-SAFE: the work-list draws text AND style crops only from the TRAIN split; the held-out real test
set (``val.jsonl``) is never read. Ray Data streams the stages with backpressure — the cheap TrOCR gate
never bottlenecks the slow generator — and the stateful actors keep each model loaded once per worker.

FLEET SHAPE (benchmarked 2026-07-13, 2× Blackwell, n=768): co-locate readers WITH generators — every GPU
at ``gen_gpu_frac=0.75`` + one TrOCR reader at ``trocr_gpu_frac=0.25`` beats dedicating a whole GPU to
gating by +28–39% throughput per GPU (the readers barely tax the generator). ``gen_batch=48`` already
saturates a Blackwell at 128px lines; 96/128 measured SLOWER at steady state (8.57 → 8.31 → 7.99 lines/s).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import NotRequired, TypedDict

from pydantic import BaseModel

from .actors import DiffuActor, TrOCRActor
from .gate import GateConfig, cer_ok, ink_ok

logger = logging.getLogger(__name__)


class RunConfig(BaseModel):
    """One generation run's knobs (frozen)."""

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    ckpt: str
    train_jsonl: str  # text + style source — TRAIN split ONLY (never val.jsonl, the held-out real test set)
    out: str  # output dir for the Lance dataset of clean (image, text, cer) rows
    n: int = 1_000_000  # target line count (before gating)
    gen_gpus: int = 2  # DiffuActor replicas (one per generation GPU)
    gen_gpu_frac: float = (
        1.0  # GPU fraction per DiffuActor — prod should use the benchmarked co-located FLEET SHAPE
        #      above (0.75 + a 0.25 reader per GPU); 1.0 dedicates the GPU and parks readers elsewhere
    )
    trocr_replicas: int = 4  # TrOCRActor replicas (light — several share a GPU)
    trocr_gpu_frac: float = 0.25  # GPU fraction per TrOCRActor (co-locate ~4 per GPU)
    gen_batch: int = 48
    trocr_batch: int = 32
    seed: int = 0
    gate: GateConfig = GateConfig()
    # DiffuActor perf knobs — same bf16 + regional-compile recipe as the released inference path
    # (docs/PERF_AUDIT.md §0e/§I). Both ON by default; flip off only if a checkpoint/box misbehaves.
    compile_model: bool = True
    gen_dtype: str = "bfloat16"
    # Optional Stage-3 LoRA adapter (dir or Hub id) applied on top of ``ckpt`` in every DiffuActor, before
    # compile — adapts the base hand/collection at generation time (``diffu.lora.load_lora``). None = base only.
    lora: str | None = None
    lora_scale: float = 1.0  # adapter strength in [0, 1+] (set_adapters weight); 1.0 = full
    # 24 is the historical default; 12 is benchmark-validated equal quality on the released 128px model
    # (paired ΔCER −0.0008 ± 0.0063 over 96 common-noise val lines, all under the production gate) for
    # ~1.5× on the generation stage. 8 shows real letterform degradation — don't.
    num_steps: int = 24
    # Width bucketing is DISABLED (None) until the glyph conditioning is fixed at train time: the current
    # model aligns glyph strips against the BATCH-max width, so near-natural canvases (and any
    # width-homogeneous batch) put it out of its trained regime and it stutters — measured 2026-07-13,
    # kept-rate 92% → 64-73%. The knob stays for a future conditioning-fixed checkpoint.
    bucket_px: int | None = None
    coherent: bool = False  # doc-coherent work-list (whole documents, (doc_id, line_idx)-keyed) instead
    #                         of random text × style — for the line bank behind prose-page composition
    # Post-write compaction (lance-ray, distributed): the writers fragment the dataset — run_target
    # appends one fragment per ~1024-row slice (~880 fragments per 900k-line round) and the streaming
    # write lands one per task — which taxes every later scan/take. Compaction rewrites to
    # ~32k-row fragments (~1.6 GB at ~50 KB/row of PNG). Best-effort: a compaction failure never
    # invalidates the (already-correct) dataset.
    compact: bool = True


class WorkRow(TypedDict):
    """One work-list row = one line to generate. Plain dict (Ray Data row), typed for the builders."""

    text: str
    style_path: str  # writer-style crop the generator conditions on
    doc_id: NotRequired[str]  # coherent work-list only: source document of the line
    line_idx: NotRequired[int]  # coherent work-list only: pre-filter position within the document


def _pools(cfg: RunConfig) -> tuple[list[str], list[str]]:
    """The TRAIN-split text + writer-style pools (leakage-safe — never val.jsonl)."""
    rows = [json.loads(line) for line in Path(cfg.train_jsonl).read_text(encoding="utf-8").splitlines()]
    texts = [t for t in (r["text"].replace("¬", "").strip() for r in rows) if 8 <= len(t) <= 65]
    return texts, [r["image"] for r in rows]


def build_worklist(cfg: RunConfig) -> list[WorkRow]:
    """A random (text, style_path) work-list from the TRAIN split — independent text and writer, so the
    dataset is combinatorially diverse and leakage-free."""
    texts, styles = _pools(cfg)
    rng = random.Random(cfg.seed)
    work: list[WorkRow] = [
        {"text": rng.choice(texts), "style_path": rng.choice(styles)} for _ in range(cfg.n)
    ]
    _shuffle_for_mixed_length_batches(work, cfg.seed)
    return work


def build_worklist_coherent(cfg: RunConfig) -> list[WorkRow]:
    """A doc-coherent work-list: WHOLE documents' lines, each keyed by ``(doc_id, line_idx)`` — pages
    composed from the bank replay CONTIGUOUS runs of one document, so prose pages read coherently; the
    random :func:`build_worklist` stays for register/table-style pages. One doc is one writer in this
    corpus, so each line keeps its OWN crop as the style reference (hand stays consistent per doc).
    Takes whole documents until ``cfg.n`` rows are reached, then truncates to exactly ``cfg.n``."""
    rows = [json.loads(line) for line in Path(cfg.train_jsonl).read_text(encoding="utf-8").splitlines()]
    by_doc: dict[str, list[dict]] = {}
    for r in rows:  # same grouping convention as diffu_runner.pages.coherent_docs: 003814_007 -> 003814
        by_doc.setdefault(Path(r["image"]).stem.rsplit("_", 1)[0], []).append(r)
    work: list[WorkRow] = []
    for doc_id, doc_rows in by_doc.items():
        if len(work) >= cfg.n:
            break
        # line_idx = the TRUE page position from the crop filename suffix (prepare_gt_lejonet names crops
        # {doc}_{position:03d} with position counted BEFORE its own drops), not enumerate(): a prep-dropped
        # line leaves no manifest row, so enumerate would compress the gap and bank.contiguous_runs would
        # silently bridge a genuinely missing prose line. Filename order also makes us robust to a
        # shuffled manifest (coherent_docs sorts for the same reason). Non-numeric suffixes (foreign
        # corpora) fall back to manifest position.
        doc_rows.sort(key=lambda r: Path(r["image"]).stem)
        for fallback_idx, r in enumerate(doc_rows):
            text = r["text"].replace("¬", "").strip()
            if not (8 <= len(text) <= 65):  # same text gate as _pools
                continue
            suffix = Path(r["image"]).stem.rsplit("_", 1)[-1]
            line_idx = int(suffix) if suffix.isdigit() else fallback_idx
            work.append({"text": text, "style_path": r["image"], "doc_id": doc_id, "line_idx": line_idx})
    del work[cfg.n :]  # the last doc may exceed the target — trim to exactly n
    # Doc order would feed the generator RUNS of similar-length lines (a register doc is dozens of short
    # lines in a row) — exactly the homogeneous batches the conditioning can't handle; shuffle instead.
    _shuffle_for_mixed_length_batches(work, cfg.seed)
    return work


def _shuffle_for_mixed_length_batches(work: list[WorkRow], seed: int) -> None:
    """Shuffle so every generator batch MIXES text lengths — a LOAD-BEARING quality invariant, not a nicety.

    The glyph-line conditioning pads each text's glyph strip to the BATCH-max strip width before
    resampling to the canvas's patch columns (``GlyphLineContentEncoder.forward``), and training only ever
    saw batches whose longest line roughly filled the canvas. A batch of ONLY short lines on the fixed
    canvas therefore stretches every glyph strip across the whole width and the model writes stuttered,
    duplicated strokes — measured 2026-07-13: the same 8 short lines read at CER ≈ 1.0 generated alone vs
    CER ≈ 0.00 mixed with long lines (see the fanout commit message). Length-sorting the work-list (an
    earlier padding-efficiency idea) or consuming documents in order both create such batches; a seeded
    shuffle guarantees the mixed-length regime. Bank consumers random-access by (doc_id, line_idx), so
    dataset order is free to exploit."""
    random.Random(seed ^ 0x5B5B5B).shuffle(work)  # decouple from the draw stream, stay reproducible


def ckpt_fingerprint(path: str) -> dict:
    """A fast, stable identity for the generating checkpoint: size + sha256 of the FIRST 64 MiB.

    Full-file hashing a 5.5 GB safetensors per run is ~20 s for no extra discrimination — the header +
    first shard already differ between any two real checkpoints. Enough to answer 'which weights made
    this dataset' months later."""
    import hashlib

    try:
        p = Path(path)
        h = hashlib.sha256()
        with p.open("rb") as f:
            h.update(f.read(64 * 1024 * 1024))
        return {"path": str(p), "bytes": p.stat().st_size, "sha256_first64MiB": h.hexdigest()}
    except OSError:  # provenance is best-effort — an unreadable path must never fail the run itself
        return {"path": str(path), "error": "unreadable"}


def code_version() -> dict:
    """Best-effort code identity: package version + the git commit of the repo checkout (if we are one)."""
    import importlib.metadata
    import subprocess

    version = {
        "diffu": importlib.metadata.version("diffu"),
        "diffu_runner": importlib.metadata.version("diffu-runner"),
    }
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, timeout=5
        ).stdout.strip()
        version["git"] = sha or "unknown"
    except Exception:  # noqa: BLE001 — provenance is best-effort; a missing git never fails a run
        version["git"] = "unknown"
    return version


def write_run_meta(out: str, payload: dict) -> None:
    """Drop ``run_meta.json`` next to the dataset — the run-level provenance sidecar.

    Per-ROW provenance lives in the rows themselves (style_path / doc_id / line_idx / cer); this file
    records everything shared by the whole run: the exact RunConfig/PageConfig, the checkpoint
    fingerprint, the code version, and when it ran — so any bank or page corpus can answer 'which model,
    which code, which knobs made you'. A plain JSON sidecar (not Lance metadata) so every tool can read
    it. Lance ignores unknown files in the dataset dir."""
    import datetime

    meta = {"generated_at": datetime.datetime.now(datetime.UTC).isoformat(), **payload}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _compact(cfg: RunConfig) -> None:
    """Distributed post-write compaction of the Lance dataset at ``cfg.out`` (see RunConfig.compact)."""
    if not cfg.compact:
        return
    try:
        from typing import cast

        import lance
        from lance.optimize import CompactionOptions
        from lance_ray import compact_files

        before = len(lance.dataset(cfg.out).get_fragments())
        # cast: lance's stubs declare CompactionOptions a TOTAL TypedDict, but the runtime accepts a
        # partial dict and fills defaults — passing every key would hardcode their defaults here instead
        opts = cast("CompactionOptions", {"target_rows_per_fragment": 32_768})
        compact_files(uri=cfg.out, compaction_options=opts)
        after = len(lance.dataset(cfg.out).get_fragments())
        logger.info("compacted %s: %d -> %d fragments", cfg.out, before, after)
    except Exception:  # noqa: BLE001 — the uncompacted dataset is already valid; never fail the run here
        logger.exception("post-write compaction failed — dataset at %s is valid but fragmented", cfg.out)


def _scored_dataset(cfg: RunConfig, worklist: list[WorkRow]):
    """The generate → ink-flag → TrOCR-score DAG for a work-list; rows carry ``ink_ok`` + ``cer`` (NOT yet
    filtered — flag-and-keep, so every dispatched row reaches the driver's accounting and can be retried).
    Ray retries infra faults (``max_restarts``/``max_task_retries``); the actors are stateful (model once)."""
    import ray

    if not ray.is_initialized():  # local convenience — a real cluster inits Ray itself before calling
        ray.init(include_dashboard=False)  # keep the working_dir package small via the repo's .rayignore

    # one read block per generator batch: the default 200-block minimum would chop a smoke-sized worklist
    # into ~1-row blocks (pure per-block overhead) and misalign blocks with gen_batch at scale
    ds = ray.data.from_items(worklist, override_num_blocks=max(1, -(-len(worklist) // cfg.gen_batch)))
    # fault tolerance: restart a dead actor + retry its tasks (actor-map kwargs, not ray_remote_args)
    ds = ds.map_batches(
        DiffuActor,
        concurrency=cfg.gen_gpus,
        num_gpus=cfg.gen_gpu_frac,
        batch_size=cfg.gen_batch,
        fn_constructor_args=(cfg.ckpt,),
        fn_constructor_kwargs={
            "compile_model": cfg.compile_model,
            "dtype": cfg.gen_dtype,
            "num_steps": cfg.num_steps,
            "bucket_px": cfg.bucket_px,
            "lora": cfg.lora,
            "lora_scale": cfg.lora_scale,
        },
        max_restarts=3,
        max_task_retries=3,
    )
    # flag, don't filter: a filtered row would silently vanish from the counts (and never be regenerated)
    ds = ds.map(lambda row: {**row, "ink_ok": ink_ok(_pil(row["png"]), cfg.gate)})
    return ds.map_batches(
        TrOCRActor,
        concurrency=cfg.trocr_replicas,
        num_gpus=cfg.trocr_gpu_frac,
        batch_size=cfg.trocr_batch,
        max_restarts=3,
        max_task_retries=3,
    )


def run(cfg: RunConfig) -> str:
    """One streaming pass: generate ``cfg.n`` → gate → write the clean rows as a Lance dataset. Best for
    huge runs (over-provision ``n`` a few % to absorb the reject rate). Returns the output path.

    Writes via the dedicated ``lance-ray`` package's ``write_lance`` (NOT Ray Data's OWN ``Dataset.write_lance``):
    ``ray[data]==2.56.0``'s bundled ``LanceDatasink`` calls ``write_fragments(..., storage_options_provider=...)``,
    a kwarg no released ``pylance`` ships (8.0.0 is current and lacks it — confirmed via ``inspect.signature``),
    so ``ds.write_lance()`` reliably ``TypeError``s at the write stage after silently burning the whole
    generate/gate/score pass first (the failure surfaces late because Ray Data pipelines stages; upstream keeps
    running until the write task errors). ``lance-ray``'s ``write_lance`` is a separately-versioned package (not
    bundled in ``ray[data]``) and writes correctly against the same installed pylance (verified: wrote + read
    back a real Ray Dataset with no error) — and distributes the write across Ray tasks instead of funneling
    every batch through the driver process.
    """
    from lance_ray import write_lance as lance_ray_write_lance

    builder = build_worklist_coherent if cfg.coherent else build_worklist
    ds = _scored_dataset(cfg, builder(cfg))
    ds = ds.filter(lambda row: row["ink_ok"] and cer_ok(row["cer"], cfg.gate))
    # ink_ok/pred are gate-internal — dropping them keeps this writer's schema consistent with
    # run_target's whitelist (pred alone is tens of MB of strings at 1M rows)
    ds = ds.map_batches(lambda b: {k: v for k, v in b.items() if k not in ("ink_ok", "pred")})
    lance_ray_write_lance(ds, cfg.out)
    _compact(cfg)
    write_run_meta(
        cfg.out,
        {
            "kind": "lines-stream",
            "run_config": cfg.model_dump(),
            "ckpt": ckpt_fingerprint(cfg.ckpt),
            "code": code_version(),
        },
    )
    return cfg.out


def _plan_round(
    retry: list[str], texts: list[str], rng: random.Random, need: int
) -> tuple[list[str], list[str], int]:
    """The texts to dispatch this round: pending retries first (capped at ``need`` — the overflow CARRIES
    to the next round instead of being silently dropped), topped up with fresh random picks. Returns
    ``(batch, remaining_retry, n_retries_dispatched)``. Pure, so the retry accounting is unit-testable."""
    dispatched = retry[:need]
    fresh = [rng.choice(texts) for _ in range(need - len(dispatched))]
    return dispatched + fresh, retry[need:], len(dispatched)


def run_target(cfg: RunConfig, target: int, *, over: float = 1.3, max_rounds: int = 6) -> tuple[str, dict]:
    """Generate until ``target`` clean lines exist, REGENERATING rejects (ink or CER) — a flagged line's
    text is retried with a FRESH random writer style (a misfire is usually style-specific, so a new hand
    fixes it). Fault-tolerant: Ray restarts dead actors + retries tasks; this loop absorbs the quality
    reject rate. Clean rows go STRAIGHT into the Lance dataset at ``cfg.out``, one consumed slice at a
    time (only counts and the retry texts stay in driver memory — driver RSS is O(one slice), never the
    round). Always uses the RANDOM work-list even when ``cfg.coherent`` is set: retry is keyed by TEXT
    with a fresh style, which would break a (doc_id, line_idx)-keyed bank — coherent banking goes through
    the streaming :func:`run` path. Writes up to ``target`` clean lines — never more; if ``max_rounds``
    runs out first, the shortfall is logged and recorded in stats. Returns ``(path, stats)``."""
    import lance
    import pyarrow as pa

    # every persistable column a scored row may carry (which arrive depends on actor passthrough — the
    # slice schema is built from what is actually present); ``ink_ok``/``pred`` are gate-internal, never written
    col_types = {
        "text": pa.string(),
        "png": pa.binary(),
        "cer": pa.float64(),
        "doc_id": pa.string(),
        "line_idx": pa.int64(),
        "style_path": pa.string(),
    }
    texts, styles = _pools(cfg)
    rng = random.Random(cfg.seed)
    retry: list[str] = []  # rejected texts awaiting a fresh-style regeneration
    kept = 0
    stats = {"rounds": 0, "attempted": 0, "regenerated": 0}
    mode = "overwrite"  # first write replaces any stale dataset; every later slice appends to it
    while kept < target and stats["rounds"] < max_rounds:
        need = int((target - kept) * over)
        batch, retry, dispatched = _plan_round(retry, texts, rng, need)
        stats["regenerated"] += dispatched
        worklist: list[WorkRow] = [{"text": t, "style_path": rng.choice(styles)} for t in batch]
        # retry rounds cluster similar (often short) rejects together — re-mix the lengths, same
        # load-bearing invariant as the builders (see _shuffle_for_mixed_length_batches)
        _shuffle_for_mixed_length_batches(worklist, cfg.seed + stats["rounds"])
        # STREAM the scored rows — take_all() would materialize the whole round's png bytes in the driver
        # (~40GB at 900k lines × ~50KB); iter_batches yields one slice at a time and each slice is flushed
        # to Lance before the next is pulled. 1024 rows ≈ 50MB of png per slice — bounded driver memory,
        # and few enough fragments per round that Lance compaction stays trivial.
        scored = _scored_dataset(cfg, worklist).iter_batches(batch_size=1024, batch_format="pyarrow")
        for slice_table in scored:
            assert isinstance(slice_table, pa.Table)  # batch_format="pyarrow" guarantees Tables
            keep: list[dict] = []
            for r in slice_table.to_pylist():
                stats["attempted"] += 1
                if not (r["ink_ok"] and cer_ok(r["cer"], cfg.gate)):
                    retry.append(r["text"])  # regenerate this text with a new writer next round
                elif kept < target:  # clean rows past target are over-provision surplus — dropped
                    keep.append(r)
                    kept += 1
            if not keep:  # all-reject / past-target slice — nothing to write
                continue
            schema = pa.schema([(c, t) for c, t in col_types.items() if c in keep[0]])
            lance.write_dataset(
                pa.table({c: [r[c] for r in keep] for c in schema.names}, schema=schema),
                cfg.out,
                mode=mode,
            )
            mode = "append"
        stats["rounds"] += 1
    if mode == "overwrite":  # nothing written (degenerate target / all rejects) — leave a valid empty dataset
        schema = pa.schema([(c, col_types[c]) for c in ("text", "png", "cer")])
        lance.write_dataset(pa.table({c: [] for c in schema.names}, schema=schema), cfg.out, mode=mode)
    if kept < target:
        logger.warning("run_target stopped at max_rounds=%d with %d/%d clean lines", max_rounds, kept, target)
    final_stats = {**stats, "kept": kept, "shortfall": max(0, target - kept)}
    _compact(cfg)  # the per-slice appends above fragment heavily (~one fragment per 1024-row slice)
    write_run_meta(
        cfg.out,
        {
            "kind": "lines-exact-target",
            "run_config": cfg.model_dump(),
            "args": {"target": target, "over": over, "max_rounds": max_rounds},
            "ckpt": ckpt_fingerprint(cfg.ckpt),
            "code": code_version(),
            "stats": final_stats,
        },
    )
    return cfg.out, final_stats


def _pil(data: bytes):
    import io

    from PIL import Image

    return Image.open(io.BytesIO(data)).convert("RGB")
