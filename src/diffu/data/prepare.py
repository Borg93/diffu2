"""CLI: PAGE/ALTO XML + page images -> training triples, writer-disjoint split, coverage report.

  python -m diffu.data.prepare --xml-dir XML/ --image-dir images/ --out-dir data_out/
  # or the installed console script:  diffu-data --xml-dir XML/ --image-dir images/ --out-dir data_out/

Works for both PAGE-XML and ALTO-XML (auto-detected per file). No torch needed.
"""

from __future__ import annotations

import argparse

from .dataset import LINE_HEIGHT, build_manifest, char_coverage, write_splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml-dir", required=True, help="dir with PAGE/ALTO .xml (searched recursively)")
    ap.add_argument("--image-dir", required=True, help="dir with the page images")
    ap.add_argument("--out-dir", required=True, help="output dir for line crops + manifests")
    ap.add_argument("--height", type=int, default=LINE_HEIGHT, help="crop height in px (the training height)")
    ap.add_argument(
        "--split-by",
        choices=["writer_id", "volume"],
        default="writer_id",
        help="split granularity: writer_id (page) or volume (coarser, more conservative eval)",
    )
    args = ap.parse_args()

    manifest = f"{args.out_dir}/manifest.jsonl"
    n = build_manifest(args.xml_dir, args.image_dir, args.out_dir, args.height)
    print(f"wrote {n} line crops -> {manifest}")

    cov = char_coverage(manifest)
    print("Swedish diacritic coverage:", {ch: cov.get(ch, 0) for ch in "åäöÅÄÖ"})

    for name, counts in write_splits(manifest, args.out_dir, group_by=args.split_by).items():
        print(f"  {name}: {counts['lines']} lines / {counts['groups']} {args.split_by}s")


if __name__ == "__main__":
    main()
