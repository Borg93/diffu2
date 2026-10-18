"""Manifest helpers: text normalization, writer-disjoint split, char coverage (no torch)."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from diffu.data.dataset import char_coverage, normalize_text, writer_disjoint_split


def test_normalize_text_nfc_precomposes_diacritics():
    decomposed = "a" + unicodedata.lookup("COMBINING RING ABOVE")  # a + ̊  -> å
    out = normalize_text(decomposed)
    assert out == "å"
    assert len(out) == 1  # single precomposed codepoint


def test_normalize_text_strips_control_chars_keeps_tabs_newlines():
    assert normalize_text("Gö\x00te\x07borg") == "Göteborg"
    assert normalize_text("  spaced  ") == "spaced"  # stripped


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_writer_disjoint_split_no_writer_in_two_splits(tmp_path):
    rows = [{"image": f"{i}.png", "text": "x", "writer_id": f"w{i % 10}"} for i in range(100)]
    mf = tmp_path / "manifest.jsonl"
    _write_manifest(mf, rows)

    split = writer_disjoint_split(str(mf), val_frac=0.2, test_frac=0.2, seed=0)

    writers = {k: {r["writer_id"] for r in v} for k, v in split.items()}
    assert writers["train"] & writers["val"] == set()
    assert writers["train"] & writers["test"] == set()
    assert writers["val"] & writers["test"] == set()
    assert sum(len(v) for v in split.values()) == len(rows)  # nothing dropped


def test_writer_disjoint_split_is_deterministic(tmp_path):
    rows = [{"image": f"{i}.png", "text": "x", "writer_id": f"w{i % 8}"} for i in range(40)]
    mf = tmp_path / "manifest.jsonl"
    _write_manifest(mf, rows)
    a = writer_disjoint_split(str(mf), seed=42)
    b = writer_disjoint_split(str(mf), seed=42)
    assert [r["image"] for r in a["train"]] == [r["image"] for r in b["train"]]


def test_char_coverage_counts_diacritics(tmp_path):
    rows = [
        {"image": "0.png", "text": "Göteborg", "writer_id": "w0"},
        {"image": "1.png", "text": "öl", "writer_id": "w1"},
    ]
    mf = tmp_path / "manifest.jsonl"
    _write_manifest(mf, rows)
    cov = char_coverage(str(mf))
    assert cov["ö"] == 2
    assert cov["G"] == 1


def test_split_by_volume_groups_by_page_id_prefix(tmp_path):
    # page ids are "<vol>_<page>"; --split-by volume must group by the <vol> prefix (no volume field).
    rows = [
        {"image": f"{vol}_{pg}.png", "text": "x", "writer_id": f"{vol}_{pg:04d}"}
        for vol in ("100", "200", "300")
        for pg in range(4)
    ]
    mf = tmp_path / "manifest.jsonl"
    _write_manifest(mf, rows)

    split = writer_disjoint_split(str(mf), val_frac=0.34, test_frac=0.34, seed=0, group_by="volume")

    vols = {k: {r["writer_id"].rsplit("_", 1)[0] for r in v} for k, v in split.items()}
    assert vols["train"] & vols["val"] == set()  # volume-disjoint, not just page-disjoint
    assert vols["train"] & vols["test"] == set()
    assert vols["val"] & vols["test"] == set()
