"""Tests for the pure driver-loop logic — no-overshoot page accounting, rerun-resume, retry round planning,
fail-fast validation, and the STREAMING run_target driver (fake scored dataset + captured Lance writes,
proving slices flush immediately and the whole round is never materialized). No Ray, no GPU, no models."""

from __future__ import annotations

import io
import json
import logging
import random
from collections.abc import Iterator
from pathlib import Path

import lance
import pyarrow as pa
import pytest
from diffu_runner import pipeline
from diffu_runner.pages import PageConfig, _existing_pages, _write_kept, run_pages
from diffu_runner.pipeline import RunConfig, _plan_round, run_target


def _pages(n: int) -> list[dict]:
    """Synthetic kept-page payloads, as PageActor.make would return them."""
    return [{"png": b"png%d" % i, "template": "prose", "page_cer": 0.1, "n_lines": 12} for i in range(n)]


def test_write_kept_stops_exactly_at_target(tmp_path: Path) -> None:
    (tmp_path / "img").mkdir()
    manifest = io.StringIO()
    kept = _write_kept(_pages(5), tmp_path, manifest, kept=0, target=3)
    assert kept == 3, "a surplus batch must not overshoot the target"
    assert len(list((tmp_path / "img").glob("page_*.png"))) == 3
    rows = [json.loads(x) for x in manifest.getvalue().splitlines()]
    assert len(rows) == 3
    assert rows[0]["image"].endswith("page_000000.png") and rows[0]["template"] == "prose"


def test_write_kept_continues_numbering_from_kept(tmp_path: Path) -> None:
    (tmp_path / "img").mkdir()
    manifest = io.StringIO()
    kept = _write_kept(_pages(2), tmp_path, manifest, kept=4, target=10)
    assert kept == 6
    names = sorted(p.name for p in (tmp_path / "img").glob("page_*.png"))
    assert names == ["page_000004.png", "page_000005.png"], "a resumed run must not overwrite earlier pages"


def test_write_kept_noop_when_target_already_met(tmp_path: Path) -> None:
    (tmp_path / "img").mkdir()
    manifest = io.StringIO()
    assert _write_kept(_pages(3), tmp_path, manifest, kept=2, target=2) == 2
    assert manifest.getvalue() == "" and not list((tmp_path / "img").glob("page_*.png"))


def test_existing_pages_counts_a_previous_run(tmp_path: Path) -> None:
    img = tmp_path / "img"
    img.mkdir()
    assert _existing_pages(img) == 0
    for i in range(4):
        (img / f"page_{i:06d}.png").write_bytes(b"x")
    assert _existing_pages(img) == 4


def test_run_pages_rejects_gpuless_pool(tmp_path: Path) -> None:
    cfg = PageConfig(ckpt="ckpt.safetensors", train_jsonl="train.jsonl", out=str(tmp_path))
    with pytest.raises(ValueError, match="gpus"):
        run_pages(cfg, target=1, gpus=0)


def test_plan_round_dispatches_retries_first_and_carries_overflow() -> None:
    batch, remaining, dispatched = _plan_round(["a", "b", "c"], ["fresh"], random.Random(0), need=2)
    assert batch == ["a", "b"]
    assert remaining == ["c"], "retry overflow carries to the next round, never silently dropped"
    assert dispatched == 2, "regenerated counts only the retries actually dispatched"


def test_plan_round_tops_up_with_fresh_texts() -> None:
    batch, remaining, dispatched = _plan_round(["a"], ["x", "y"], random.Random(0), need=3)
    assert len(batch) == 3 and batch[0] == "a"
    assert set(batch[1:]) <= {"x", "y"}
    assert remaining == [] and dispatched == 1


# --- run_target: streaming write, retry accounting, column passthrough ---------------------------------


def _cfg(tmp_path: Path) -> RunConfig:
    """A RunConfig over a tiny real train.jsonl (run_target reads it via _pools)."""
    rows = [
        {"text": f"radnummer {i} i det svenska protokollet", "image": f"doc{i}_000.png"} for i in range(4)
    ]
    train = tmp_path / "train.jsonl"
    train.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return RunConfig(ckpt="ckpt.safetensors", train_jsonl=str(train), out=str(tmp_path / "bank.lance"))


class _FakeScored:
    """Stands in for the scored Ray dataset: streams pyarrow slices; take_all is BANNED — it is the
    driver-memory funnel (a whole round's png bytes in one process) that run_target must never reopen."""

    def __init__(self, rows: list[dict], slice_rows: int) -> None:
        self._rows, self._slice_rows = rows, slice_rows

    def take_all(self) -> list[dict]:
        raise AssertionError("run_target must stream via iter_batches, never materialize with take_all")

    def iter_batches(self, *, batch_size: int, batch_format: str) -> Iterator[pa.Table]:
        assert batch_format == "pyarrow", "run_target consumes pyarrow slices"
        assert batch_size > 0
        for i in range(0, len(self._rows), self._slice_rows):
            yield pa.Table.from_pylist(self._rows[i : i + self._slice_rows])


def _scored_row(w: dict, *, ok: bool = True, **extra: object) -> dict:
    """A row as the gate stages emit it: work-list columns + png + ink flag + TrOCR read-back."""
    return {
        "text": w["text"],
        "style_path": w["style_path"],
        "png": b"png:" + str(w["text"]).encode(),
        "ink_ok": ok,
        "pred": w["text"] if ok else "",
        "cer": 0.05 if ok else 0.95,
        **extra,
    }


def _patch_scored(monkeypatch, score, *, slice_rows: int = 2) -> list[list[dict]]:
    """Swap pipeline._scored_dataset for a fake scoring each row with ``score(row, round_i)``; returns
    the captured per-round work-lists (their length is the round counter the scorer sees)."""
    worklists: list[list[dict]] = []

    def fake(cfg: RunConfig, worklist: list[dict]) -> _FakeScored:
        round_i = len(worklists)
        worklists.append(list(worklist))
        return _FakeScored([score(w, round_i) for w in worklist], slice_rows)

    monkeypatch.setattr(pipeline, "_scored_dataset", fake)
    return worklists


def _capture_writes(monkeypatch) -> list[tuple[pa.Table, str]]:
    """Capture lance.write_dataset calls as (table, mode) without touching disk."""
    writes: list[tuple[pa.Table, str]] = []
    monkeypatch.setattr(lance, "write_dataset", lambda table, uri, mode: writes.append((table, mode)))
    return writes


def test_run_target_streams_each_slice_and_never_takes_all(monkeypatch, tmp_path: Path) -> None:
    _patch_scored(monkeypatch, lambda w, i: _scored_row(w), slice_rows=2)  # all clean, 2-row slices
    writes = _capture_writes(monkeypatch)
    path, stats = run_target(_cfg(tmp_path), target=4)  # need = int(4 * 1.3) = 5 -> slices of 2, 2, 1
    assert path.endswith("bank.lance")
    # per-slice flushing: two 2-row writes; the 5th (past-target surplus) row's slice writes nothing
    assert [(t.num_rows, mode) for t, mode in writes] == [(2, "overwrite"), (2, "append")]
    assert stats == {"rounds": 1, "attempted": 5, "regenerated": 0, "kept": 4, "shortfall": 0}


def test_run_target_retries_rejected_texts_with_fresh_style_next_round(monkeypatch, tmp_path: Path) -> None:
    # round 0 rejects everything (ink), round 1 passes — the classic style-misfire-then-recover shape
    worklists = _patch_scored(monkeypatch, lambda w, i: _scored_row(w, ok=i > 0))
    writes = _capture_writes(monkeypatch)
    _, stats = run_target(_cfg(tmp_path), target=2)  # need = int(2 * 1.3) = 2 per round
    assert [w["text"] for w in worklists[1]] == [w["text"] for w in worklists[0]], (
        "a rejected TEXT must be re-dispatched (with a fresh random style) the next round"
    )
    assert all("style_path" in w for w in worklists[1])
    assert stats == {"rounds": 2, "attempted": 4, "regenerated": 2, "kept": 2, "shortfall": 0}
    # no clean rows in round 0 -> nothing written; round 1's first slice is the dataset-creating overwrite
    assert [(t.num_rows, mode) for t, mode in writes] == [(2, "overwrite")]


def test_run_target_degenerate_target_still_writes_empty_dataset(monkeypatch, tmp_path: Path) -> None:
    _patch_scored(monkeypatch, lambda w, i: _scored_row(w))
    writes = _capture_writes(monkeypatch)
    _, stats = run_target(_cfg(tmp_path), target=0)
    assert len(writes) == 1
    table, mode = writes[0]
    assert mode == "overwrite" and table.num_rows == 0
    assert table.schema.names == ["text", "png", "cer"]
    assert stats == {"rounds": 0, "attempted": 0, "regenerated": 0, "kept": 0, "shortfall": 0}


def test_run_target_passes_through_extra_columns(monkeypatch, tmp_path: Path) -> None:
    # once the actors pass style/coherence metadata through, the writer must persist it — but never the
    # gate-internal ink_ok/pred flags
    _patch_scored(monkeypatch, lambda w, i: _scored_row(w, doc_id="00A", line_idx=3))
    writes = _capture_writes(monkeypatch)
    run_target(_cfg(tmp_path), target=2)
    table = writes[0][0]
    assert table.schema.names == ["text", "png", "cer", "doc_id", "line_idx", "style_path"]
    assert table["doc_id"].to_pylist() == ["00A", "00A"] and table["line_idx"].to_pylist() == [3, 3]


def test_run_target_records_shortfall_when_max_rounds_run_out(monkeypatch, tmp_path: Path, caplog) -> None:
    _patch_scored(monkeypatch, lambda w, i: _scored_row(w, ok=False))  # nothing ever passes the gate
    writes = _capture_writes(monkeypatch)
    with caplog.at_level(logging.WARNING):
        _, stats = run_target(_cfg(tmp_path), target=2, max_rounds=2)
    assert stats == {"rounds": 2, "attempted": 4, "regenerated": 2, "kept": 0, "shortfall": 2}
    assert "max_rounds" in caplog.text
    # no clean slice ever written -> the fallback still leaves a valid empty dataset behind
    assert [(t.num_rows, mode) for t, mode in writes] == [(0, "overwrite")]


def test_varied_job_emits_spreads_without_a_spine_binding() -> None:
    from diffu_runner.pages import varied_job

    docs = [{"texts": ["Goteborgs poliskammare den fjarde maj"] * 8, "style": "s.png"}]
    rng = random.Random(0)
    specs = [varied_job(rng, docs, seed)["spec"] for seed in range(200)]
    spreads = [s for s in specs if s["template"] == "spread"]
    assert spreads, "the sampler's spread_frac must reach the production varied_job path"
    # a spread's spine is the CENTRAL gutter; the u["binding"] override must never re-enable a
    # single-edge binding on one (it would warp + clamp one leaf's outer margin)
    assert all(not s["binding"] for s in spreads)
    assert any(s["binding"] for s in specs), "single leaves still draw the spine binding"
