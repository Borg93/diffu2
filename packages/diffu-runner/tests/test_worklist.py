"""Work-list builder tests — doc-coherent grouping (whole docs, pre-filter line_idx, own-crop style),
the _pools text gate, the width-homogeneous length sort of BOTH builders, and RunConfig.coherent picking
the builder inside run(). All pure/filesystem-only: no Ray, no GPU, no models."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from diffu_runner import pipeline
from diffu_runner.pipeline import RunConfig, build_worklist, build_worklist_coherent

# three docs x three valid lines, distinct lengths — the "003814_007 -> 003814" grouping convention
_DOCS: dict[str, list[str]] = {
    "00A": [
        "anno domini adertonhundrafemtio",
        "protokollet hallet vid vintertinget",
        "narvarande namndeman och lansman",
    ],
    "00B": [
        "boskapslangd for hela socknen",
        "summa kreatur tio kor och en oxe",
        "upprattad av kronolansmannen sjalv",
    ],
    "00C": [
        "malet uppskjutes till hosttinget",
        "vittnena avlade sin ed infor ratten",
        "domboken justerades och undertecknades",
    ],
}


def _cfg(tmp_path: Path, rows: list[dict], n: int, seed: int = 0) -> RunConfig:
    train = tmp_path / "train.jsonl"
    train.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return RunConfig(
        ckpt="ckpt.safetensors", train_jsonl=str(train), out=str(tmp_path / "out"), n=n, seed=seed
    )


def _rows(docs: dict[str, list[str]]) -> list[dict]:
    """train.jsonl rows in document order, images following the pages.coherent_docs naming convention."""
    return [
        {"text": text, "image": f"pages/{doc}_{i:03d}.png"}
        for doc, texts in docs.items()
        for i, text in enumerate(texts)
    ]


def _by_doc(work: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """Rows regrouped by doc_id, in line order — the (doc_id, line_idx) random-access view a bank
    consumer uses (the returned list itself is length-sorted, so positional order is meaningless)."""
    docs: dict[str, list[Mapping[str, Any]]] = {}
    for r in sorted(work, key=lambda r: (r["doc_id"], r["line_idx"])):
        docs.setdefault(r["doc_id"], []).append(r)
    return docs


def test_coherent_keeps_whole_docs_with_own_style_crop(tmp_path: Path) -> None:
    work = build_worklist_coherent(_cfg(tmp_path, _rows(_DOCS), n=9))
    docs = _by_doc(work)
    assert set(docs) == {"00A", "00B", "00C"}
    for doc_id, lines in docs.items():
        assert [r["line_idx"] for r in lines] == [0, 1, 2], "unfiltered docs stay contiguous from 0"
        assert [r["text"] for r in lines] == _DOCS[doc_id]
        # one doc = one writer: each line's style reference is its OWN crop, never another doc's
        assert [r["style_path"] for r in lines] == [f"pages/{doc_id}_{i:03d}.png" for i in range(3)]


def test_coherent_line_idx_counts_prefilter_positions(tmp_path: Path) -> None:
    rows = _rows(
        {
            "00X": [
                "kort",  # < 8 chars -> filtered, must still consume line_idx 0
                "handlingen upplastes for ratten",
                "for¬handlingarna fortsattes",  # the "¬" strips away, leaving a valid line
                "x" * 70,  # > 65 chars -> filtered, must still consume line_idx 3
                "malet uppskjutes till hosttinget",
            ]
        }
    )
    work = build_worklist_coherent(_cfg(tmp_path, rows, n=10))
    lines = _by_doc(work)["00X"]
    assert [r["line_idx"] for r in lines] == [1, 2, 4], (
        "line_idx is the PRE-filter doc position, so downstream sees the contiguity gaps"
    )
    assert lines[1]["text"] == "forhandlingarna fortsattes", "the soft hyphen must be stripped"


def test_coherent_takes_whole_docs_then_truncates_to_n(tmp_path: Path) -> None:
    work = build_worklist_coherent(_cfg(tmp_path, _rows(_DOCS), n=4))
    assert len(work) == 4
    docs = _by_doc(work)
    assert [r["line_idx"] for r in docs["00A"]] == [0, 1, 2], "the first doc arrives whole"
    # doc B tipped the count past n and got truncated — but to a contiguous PREFIX, never a mid-doc gap
    assert [r["line_idx"] for r in docs["00B"]] == [0]
    assert "00C" not in docs, "docs are taken whole-or-prefix in order; later docs never leak in"


@pytest.mark.parametrize("builder", [build_worklist, build_worklist_coherent])
def test_worklists_are_shuffled_for_mixed_length_batches(builder, tmp_path: Path) -> None:
    # LOAD-BEARING: generator batches must MIX text lengths — the glyph conditioning pads strips to the
    # batch max, and a width-homogeneous batch makes the model stutter (see
    # _shuffle_for_mixed_length_batches). So: deterministic per seed, rows preserved, doc/draw order broken.
    work = builder(_cfg(tmp_path, _rows(_DOCS), n=9))
    again = builder(_cfg(tmp_path, _rows(_DOCS), n=9))
    assert work == again, "the shuffle must be seeded — reruns of the same config dispatch identically"
    assert len(work) == 9
    lengths = [len(r["text"]) for r in work]
    assert lengths != sorted(lengths, reverse=True), "length-sorting would recreate homogeneous batches"
    other_seed = builder(_cfg(tmp_path, _rows(_DOCS), n=9, seed=1))
    assert [r["text"] for r in other_seed] != [r["text"] for r in work], "a new seed reshuffles"


@pytest.mark.parametrize(("coherent", "expected"), [(True, "coherent"), (False, "random")])
def test_run_picks_the_builder_from_runconfig(
    monkeypatch, tmp_path: Path, coherent: bool, expected: str
) -> None:
    import lance_ray

    class _FakeDs:
        def filter(self, fn) -> _FakeDs:
            return self

        def map_batches(self, fn) -> _FakeDs:  # the gate-internal column drop before the write
            return self

    calls: list[str] = []
    monkeypatch.setattr(pipeline, "build_worklist", lambda cfg: calls.append("random") or [])
    monkeypatch.setattr(pipeline, "build_worklist_coherent", lambda cfg: calls.append("coherent") or [])
    monkeypatch.setattr(pipeline, "_scored_dataset", lambda cfg, worklist: _FakeDs())
    monkeypatch.setattr(lance_ray, "write_lance", lambda ds, out: None)
    cfg = RunConfig(ckpt="c", train_jsonl="t.jsonl", out=str(tmp_path / "o"), coherent=coherent)
    assert pipeline.run(cfg) == cfg.out
    assert calls == [expected]
