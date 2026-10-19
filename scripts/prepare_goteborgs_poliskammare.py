"""Source-specific recipe: prepare the Riksarkivet "Göteborgs poliskammare före 1900" dataset.

This is the per-source GLUE — downloading the HF dataset (image + PAGE-XML tar.gz shards), extracting
them, and pointing the GENERIC pipeline at the result. The reusable parts (XML parsing, line cropping,
writer-disjoint split) live in ``diffu.data`` and are shared across sources; only this orchestration
is dataset-specific.

    uv run python scripts/prepare_goteborgs_poliskammare.py --out-dir data_out
    uv run python scripts/prepare_goteborgs_poliskammare.py --skip-download   # reuse data_raw/
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download

from diffu.data.dataset import LINE_HEIGHT, build_manifest, char_coverage, write_splits

REPO_ID = "Riksarkivet/goteborgs_poliskammare_fore_1900"


def download_and_extract(raw_dir: Path) -> tuple[Path, Path]:
    """Download the image + page-xml tar.gz shards and extract them. Returns (xml_dir, image_dir)."""
    dl = raw_dir / "_dl"
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=["data/images/*", "data/page_xmls/*"],
        local_dir=str(dl),
        max_workers=8,
    )
    images, xmls = raw_dir / "images", raw_dir / "page_xmls"
    images.mkdir(parents=True, exist_ok=True)
    xmls.mkdir(parents=True, exist_ok=True)
    for tgz in sorted((dl / "data" / "images").glob("*.tar.gz")):
        with tarfile.open(tgz) as t:
            t.extractall(images, filter="data")  # filter="data" guards against path traversal
    for tgz in sorted((dl / "data" / "page_xmls").glob("*.tar.gz")):
        with tarfile.open(tgz) as t:
            t.extractall(xmls, filter="data")
    return xmls, images


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare the Göteborgs poliskammare dataset for Diffu.")
    ap.add_argument("--raw-dir", default="data_raw", help="where to download/extract raw images + xml")
    ap.add_argument("--out-dir", default="data_out", help="output dir for manifest + train/val/test")
    ap.add_argument(
        "--height",
        type=int,
        default=LINE_HEIGHT,
        help=f"line crop height px (default {LINE_HEIGHT}, the training line height)",
    )
    ap.add_argument("--skip-download", action="store_true", help="reuse an already-extracted --raw-dir")
    ap.add_argument(
        "--split-by",
        choices=["writer_id", "volume"],
        default="volume",
        help="split granularity (default volume: one scribe-stint stays on one side)",
    )
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    if args.skip_download:
        xmls, images = raw / "page_xmls", raw / "images"
    else:
        xmls, images = download_and_extract(raw)

    n = build_manifest(str(xmls), str(images), args.out_dir, args.height)
    manifest = f"{args.out_dir}/manifest.jsonl"
    print(f"wrote {n} line crops -> {manifest}")

    cov = char_coverage(manifest)
    print("Swedish diacritic coverage:", {ch: cov.get(ch, 0) for ch in "åäöÅÄÖ"})
    for name, counts in write_splits(manifest, args.out_dir, group_by=args.split_by).items():
        print(f"  {name}: {counts['lines']} lines / {counts['groups']} {args.split_by}s")


if __name__ == "__main__":
    main()
