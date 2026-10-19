"""Source-specific recipe: prepare the local GT-Lejonet HTR corpus into line crops + manifest.

GT-Lejonet ships ~227 volumes across many Riksarkivet collections, each laid out as::

    GT-<collection>_<date>/<volume>/<volume>/
        <id>.jpg|.tif            # page image (sibling of page/)
        page/<id>.xml            # PAGE-XML  (line polygons + transcription)
        alto/<id>.xml            # ALTO mirror (ignored — same lines)

This is the per-source GLUE only: it walks the nested ``page/`` dirs, resolves each page image as
its sibling, and feeds the lines through the GENERIC pipeline (``crop_line`` / ``normalize_text`` /
``write_splits`` from ``diffu.data``) that every source shares. Unlike the goteborgs recipe there is
nothing to download — the corpus is already on disk; the layout is just nested, so the flat
``build_manifest`` can't be pointed at it directly (it would also slurp ``alto/`` + ``mets`` and lose
the real volume as the split key).

    uv run python scripts/prepare_gt_lejonet.py --base <GT-Lejonet root> --out-dir data_out_lejonet
    uv run python scripts/prepare_gt_lejonet.py --base <GT-Lejonet root> --collections finnish_letter \
        --limit-pages 30  # quick check
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

from PIL import Image

from diffu.data.dataset import IMAGE_EXTS, LINE_HEIGHT, char_coverage, crop_line, normalize_text, write_splits
from diffu.data.parse import parse_file

Image.MAX_IMAGE_PIXELS = None  # trusted local scans; some pages exceed PIL's decompression-bomb limit


def _sibling_image(xml_path: Path, image_field: str) -> Path | None:
    """Resolve a page image as a sibling of the ``page/`` dir: ``<volume>/<imageFilename>``.

    Falls back to any same-stem image in the volume dir (handles a stale ``imageFilename`` or a
    .jpg/.tif mismatch), so resolution never silently crosses into another volume.
    """
    vol_dir = xml_path.parent.parent  # <volume>/page/<id>.xml -> <volume>
    if image_field:
        cand = vol_dir / Path(image_field).name
        if cand.is_file():
            return cand
    stem = xml_path.stem
    for ext in IMAGE_EXTS:
        cand = vol_dir / f"{stem}{ext}"
        if cand.is_file():
            return cand
    return None


def _volume_of(xml_path: Path, base: Path) -> tuple[str, str]:
    """Return ``(collection, volume)`` for a ``…/page/<id>.xml`` path — collection is the top-level
    ``GT-*`` folder, volume is the document folder that holds ``page/`` (the split group)."""
    rel = xml_path.relative_to(base).parts
    collection = rel[0]
    volume = xml_path.parent.parent.name
    return collection, volume


def _process_page(task: tuple[str, str, str, int, int, bool]) -> tuple[list[dict[str, str]], int, int]:
    """Crop every line of one page. Worker for the process pool — args are plain strings/ints/bools so
    they pickle. Crops are named ``<page_idx>_<line_idx>.png`` (no shared counter across workers).
    With ``mask`` set, each line is masked to its PAGE polygon (white outside); else plain bbox.

    Returns ``(manifest_rows, missing_image, unreadable_page)``.
    """
    xml_str, base_str, out_str, height, idx, mask = task
    xml_path, base, out_dir = Path(xml_str), Path(base_str), Path(out_str)
    collection, volume = _volume_of(xml_path, base)
    records = parse_file(xml_str)
    img_path = _sibling_image(xml_path, records[0].image) if records else None
    if img_path is None:
        return [], (1 if records else 0), 0
    rows: list[dict[str, str]] = []
    try:
        with Image.open(img_path) as page:
            page.load()
            for j, rec in enumerate(records):
                crop = crop_line(page, rec.bbox, height, polygon=rec.polygon if mask else None)
                text = normalize_text(rec.text)
                if crop is None or not text:
                    continue
                out_png = out_dir / "lines" / f"{idx:06d}_{j:03d}.png"
                crop.save(out_png)
                # writer_id is globally unique (collection/volume/page) so a page stem reused across
                # volumes can never merge two scribes for style-pairing or leak across the split.
                rows.append(
                    {
                        "image": str(out_png),
                        "text": text,
                        "writer_id": f"{collection}/{volume}/{rec.source_id}",
                        "volume": volume,
                        "collection": collection,
                    }
                )
    except Exception:  # noqa: BLE001 — one unreadable/corrupt page must never kill a 30k-page batch
        return [], 0, 1
    return rows, 0, 0


def build(
    base: Path,
    out_dir: Path,
    height: int,
    collections: str | None,
    limit_pages: int,
    log_every: int,
    workers: int,
    mask: bool,
) -> int:
    pages = sorted(p for p in base.glob("*/**/page/*.xml") if not p.name.startswith("._"))
    if collections:
        wanted = [c.strip().lower() for c in collections.split(",") if c.strip()]
        pages = [p for p in pages if any(w in str(p).lower() for w in wanted)]
    if limit_pages:
        pages = pages[:limit_pages]
    if not pages:
        raise SystemExit(f"no page/*.xml found under {base} (filters: {collections!r})")

    (out_dir / "lines").mkdir(parents=True, exist_ok=True)
    tasks = [(str(p), str(base), str(out_dir), height, i, mask) for i, p in enumerate(pages)]
    n, missing, bad, done = 0, 0, 0, 0
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as mf, mp.Pool(workers) as pool:
        for rows, m, b in pool.imap_unordered(_process_page, tasks, chunksize=8):
            for r in rows:
                mf.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += len(rows)
            missing += m
            bad += b
            done += 1
            if done % log_every == 0 or done == len(tasks):
                print(
                    f"  [{done}/{len(tasks)} pages] {n} lines  "
                    f"({missing} no-image, {bad} unreadable)  ×{workers} workers",
                    flush=True,
                )
    print(f"wrote {n} line crops from {len(pages)} pages -> {out_dir / 'manifest.jsonl'}", flush=True)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare the local GT-Lejonet HTR corpus into line crops for Diffu."
    )
    ap.add_argument(
        "--base",
        type=Path,
        required=True,
        help="GT-Lejonet root directory, e.g. .../GT-Lejonet_HTR_202602 (holds GT-<collection>_<date>/)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data_out_lejonet"),
        help="output dir for crops + manifest + splits",
    )
    ap.add_argument(
        "--height",
        type=int,
        default=LINE_HEIGHT,
        help=f"line crop height px (default {LINE_HEIGHT}, the training line height)",
    )
    ap.add_argument(
        "--collections", default=None, help="comma-separated substrings to include (default: all)"
    )
    ap.add_argument(
        "--limit-pages", type=int, default=0, help="cap pages processed (0 = all) — for quick validation"
    )
    ap.add_argument(
        "--split-by",
        choices=["writer_id", "volume"],
        default="volume",
        help="split granularity (default volume: a scribe-stint stays on one side)",
    )
    ap.add_argument("--log-every", type=int, default=200, help="progress log cadence in pages")
    ap.add_argument(
        "--workers",
        type=int,
        default=min(32, (os.cpu_count() or 4) - 2),
        help="parallel page workers (default min(32, cpus-2))",
    )
    ap.add_argument(
        "--no-mask",
        action="store_true",
        help="disable polygon masking — crop the plain bbox (noisier; keeps neighbour ink)",
    )
    args = ap.parse_args()
    if not args.base.is_dir():
        ap.error(f"--base {args.base} is not a directory")

    n = build(
        args.base,
        args.out_dir,
        args.height,
        args.collections,
        args.limit_pages,
        args.log_every,
        args.workers,
        mask=not args.no_mask,
    )
    if n == 0:
        raise SystemExit("no line crops written")

    manifest = str(args.out_dir / "manifest.jsonl")
    cov = char_coverage(manifest)
    print("Swedish diacritic coverage:", {ch: cov.get(ch, 0) for ch in "åäöÅÄÖ"})
    for name, counts in write_splits(manifest, str(args.out_dir), group_by=args.split_by).items():
        print(f"  {name}: {counts['lines']} lines / {counts['groups']} {args.split_by}s")


if __name__ == "__main__":
    main()
