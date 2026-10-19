"""Ingest ALREADY-cropped (line image, transcription) data into the Diffu manifest + writer-disjoint
split. The pre-cropped counterpart to ``prepare.py`` (which crops from PAGE/ALTO XML); both produce the
same ``manifest.jsonl`` + ``train/val/test.jsonl`` via the shared ``dataset.write_splits``.

Two input shapes, both deriving the writer/source id automatically — no manual labelling:

  # A folder tree, one folder per page/volume, with sidecar text files (line_002.png + line_002.txt):
  diffu-ingest --dir crops/ --out-dir data_out               # writer id = the folder name

  # OR a CSV / JSONL you already have (image, text columns); writer from a column or the path:
  diffu-ingest --input lines.jsonl --out-dir data_out --writer-col scribe
  diffu-ingest --input lines.csv --out-dir data_out --writer-depth 1   # writer = image's parent dir

Pure stdlib + the dataset helpers — no torch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Iterator

from .dataset import IMAGE_EXTS, char_coverage, normalize_text, write_splits


def writer_from_path(image_path: str, depth: int = 1) -> str:
    """Writer/source proxy = the ``depth``-th parent directory of the image (1 = immediate folder)."""
    p = os.path.dirname(image_path)
    for _ in range(depth - 1):
        p = os.path.dirname(p)
    return os.path.basename(p) or "unknown"


def scan_folder(root: str, *, text_suffix: str = ".txt", writer_depth: int = 1) -> list[dict[str, str]]:
    """Walk ``root`` for line images, pair each with its sidecar text file, derive the writer from the
    folder. Images without a sidecar (or with empty text) are skipped.
    """
    records: list[dict[str, str]] = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTS:
                continue
            image = os.path.join(dirpath, name)
            sidecar = os.path.join(dirpath, stem + text_suffix)
            if not os.path.exists(sidecar):
                continue
            with open(sidecar, encoding="utf-8") as f:
                text = normalize_text(f.read())
            if text:
                records.append(
                    {"image": image, "text": text, "writer_id": writer_from_path(image, writer_depth)}
                )
    return records


def _read_rows(input_path: str) -> Iterator[dict[str, str]]:
    """Yield raw rows from a ``.jsonl`` or ``.csv`` file (auto-detected by extension)."""
    if input_path.endswith((".jsonl", ".ndjson")):
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif input_path.endswith(".csv"):
        with open(input_path, encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f)
    else:
        raise ValueError(f"unsupported input {input_path!r}: expected a .jsonl or .csv file")


def read_records(
    input_path: str,
    *,
    image_col: str = "image",
    text_col: str = "text",
    writer_col: str | None = "writer_id",
    writer_depth: int | None = None,
    image_root: str | None = None,
) -> list[dict[str, str]]:
    """Read CSV/JSONL records into the manifest schema ``{image, text, writer_id}``.

    The writer id comes from ``writer_col`` if given, else (when ``writer_depth`` is set) from the
    image's parent directory. Text is NFC-normalized; empty-text rows are dropped.

    Raises:
        KeyError: if the first row is missing a requested column (image/text/writer).
    """
    records: list[dict[str, str]] = []
    for i, row in enumerate(_read_rows(input_path)):
        if i == 0:
            needed = {image_col, text_col} | ({writer_col} if writer_col and writer_depth is None else set())
            missing = needed - set(row)
            if missing:
                raise KeyError(
                    f"input is missing column(s) {sorted(missing)}; found {sorted(row)}. "
                    f"Pass --image-col / --text-col / --writer-col, or use --writer-depth."
                )
        text = normalize_text(str(row[text_col]))
        if not text:
            continue
        image = str(row[image_col])
        if image_root:
            image = os.path.join(image_root, image)
        if writer_depth is not None:
            writer = writer_from_path(image, writer_depth)
        elif writer_col is not None:
            writer = str(row[writer_col])
        else:  # library callers can pass both None; fail fast (an assert would vanish under python -O)
            raise ValueError("read_records needs writer_col or writer_depth set; both are None")
        records.append({"image": image, "text": text, "writer_id": writer})
    return records


def write_manifest(records: list[dict[str, str]], out_dir: str) -> str:
    """Write records to ``out_dir/manifest.jsonl``; returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = os.path.join(out_dir, "manifest.jsonl")
    with open(manifest, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ingest pre-cropped (image, text) lines -> manifest + writer-disjoint split."
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--dir", help="root folder of line crops + sidecar text files (one folder per page/volume)"
    )
    src.add_argument("--input", help="a .jsonl or .csv of line records")
    ap.add_argument("--out-dir", required=True, help="output dir for manifest.jsonl + train/val/test.jsonl")
    ap.add_argument("--text-suffix", default=".txt", help="sidecar transcription extension (folder mode)")
    ap.add_argument(
        "--writer-depth",
        type=int,
        default=None,
        help="derive writer id from the N-th parent dir of the image (1 = its folder). "
        "Folder mode defaults to 1; in --input mode this overrides --writer-col.",
    )
    ap.add_argument("--image-col", default="image")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--writer-col", default="writer_id")
    ap.add_argument("--image-root", default=None, help="prefix for relative image paths (--input mode)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.dir:
        records = scan_folder(args.dir, text_suffix=args.text_suffix, writer_depth=args.writer_depth or 1)
    else:
        records = read_records(
            args.input,
            image_col=args.image_col,
            text_col=args.text_col,
            writer_col=args.writer_col,
            writer_depth=args.writer_depth,
            image_root=args.image_root,
        )
    if not records:
        raise SystemExit("no (image, text) records found — check --dir / --text-suffix or the CSV columns.")

    manifest = write_manifest(records, args.out_dir)
    n_writers = len({r["writer_id"] for r in records})
    print(f"wrote {len(records)} lines from {n_writers} writers/sources -> {manifest}")

    cov = char_coverage(manifest)
    print("Swedish diacritic coverage:", {ch: cov.get(ch, 0) for ch in "åäöÅÄÖ"})

    for name, counts in write_splits(
        manifest, args.out_dir, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    ).items():
        print(f"  {name}: {counts['lines']} lines / {counts['groups']} writers")


if __name__ == "__main__":
    main()
