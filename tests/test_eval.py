"""Eval harness CER logic, with an injected stub recognizer (no model download)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from diffu.eval import evaluate_manifest


def _manifest(tmp_path: Path, rows: list[dict[str, str]]) -> str:
    for r in rows:
        Image.new("RGB", (40, 32), "white").save(r["image"])
    mf = tmp_path / "val.jsonl"
    mf.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(mf)


def test_evaluate_manifest_computes_length_weighted_cer(tmp_path):
    rows = [
        {"image": str(tmp_path / "0.png"), "text": "abc", "writer_id": "w0"},
        {"image": str(tmp_path / "1.png"), "text": "abcd", "writer_id": "w0"},
    ]
    mf = _manifest(tmp_path, rows)

    def stub(images):
        assert len(images) == 2  # got one PIL per row
        return ["abc", "abce"]  # 0 edits + 1 edit over (3 + 4) chars

    res = evaluate_manifest(mf, stub)
    assert res.n == 2
    assert res.cer == 1 / 7
    assert res.samples[0] == ("abc", "abc")


def test_evaluate_manifest_respects_limit(tmp_path):
    rows = [{"image": str(tmp_path / f"{i}.png"), "text": "x", "writer_id": "w0"} for i in range(5)]
    mf = _manifest(tmp_path, rows)
    seen = {}

    def stub(images):
        seen["n"] = len(images)
        return ["x"] * len(images)

    res = evaluate_manifest(mf, stub, limit=2)
    assert res.n == 2
    assert seen["n"] == 2
