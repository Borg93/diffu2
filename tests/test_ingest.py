"""Pre-cropped ingest: folder scan + CSV/JSONL records -> manifest + writer-disjoint split."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from diffu.data.dataset import load_manifest, write_splits
from diffu.data.ingest import read_records, scan_folder, writer_from_path


def _img(path: Path) -> None:
    Image.new("RGB", (40, 32), "white").save(path)


def test_writer_from_path_depth():
    p = "/data/vol123/page045/line002.png"
    assert writer_from_path(p, 1) == "page045"  # immediate folder
    assert writer_from_path(p, 2) == "vol123"  # one up


def test_scan_folder_pairs_sidecars_and_derives_writer(tmp_path):
    for page in ("p1", "p2"):  # two "pages" = two writers/sources
        d = tmp_path / page
        d.mkdir()
        _img(d / "l0.png")
        (d / "l0.txt").write_text("Göteborg", encoding="utf-8")
    _img(tmp_path / "p1" / "extra.png")  # image with no sidecar -> skipped

    recs = scan_folder(str(tmp_path))

    assert len(recs) == 2
    assert {r["writer_id"] for r in recs} == {"p1", "p2"}
    assert all(r["text"] == "Göteborg" for r in recs)


def test_read_records_jsonl_writer_from_column(tmp_path):
    img = tmp_path / "a.png"
    _img(img)
    mf = tmp_path / "in.jsonl"
    mf.write_text(json.dumps({"image": str(img), "text": "öl", "writer_id": "w7"}), encoding="utf-8")
    assert read_records(str(mf)) == [{"image": str(img), "text": "öl", "writer_id": "w7"}]


def test_read_records_csv_writer_from_depth(tmp_path):
    d = tmp_path / "volA"
    d.mkdir()
    img = d / "x.png"
    _img(img)
    mf = tmp_path / "in.csv"
    with open(mf, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "text"])
        w.writeheader()
        w.writerow({"image": str(img), "text": "Åska"})
    recs = read_records(str(mf), writer_depth=1)  # no writer column -> use parent dir "volA"
    assert recs[0]["writer_id"] == "volA"


def test_write_splits_is_writer_disjoint_and_writes_files(tmp_path):
    rows = [{"image": f"{i}.png", "text": "x", "writer_id": f"w{i % 10}"} for i in range(100)]
    out = tmp_path / "out"
    out.mkdir()
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    summary = write_splits(str(out / "manifest.jsonl"), str(out), val_frac=0.2, test_frac=0.2, seed=0)

    assert sum(s["lines"] for s in summary.values()) == 100
    sets = {
        n: {r["writer_id"] for r in load_manifest(str(out / f"{n}.jsonl"))} for n in ("train", "val", "test")
    }
    assert sets["train"] & sets["val"] == set()
    assert sets["train"] & sets["test"] == set()
