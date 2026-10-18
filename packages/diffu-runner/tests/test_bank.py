"""Tests for the Lance line-bank — contiguous-run detection, both schema-driven sampling modes, fetch
decoding, the stored-CER page gate, and a GPU-less end-to-end bank-mode page compose. No Ray, no GPU,
no models."""

from __future__ import annotations

import io
import random
from pathlib import Path

import lance
import pyarrow as pa
import pytest
from diffu_runner.bank import LineBank, contiguous_runs, empirical_measure, gate_stored_cers
from PIL import Image, ImageDraw


def _png(width: int = 96, height: int = 32) -> bytes:
    """A tiny line image with a few dark strokes — decodable, croppable, matte-able ink. The background is
    off-white PAPER tone, not pure white: ``ink_matte`` treats >=250 as a real crop's padding and flattens
    it to the body tone, so an all-white background would cancel the strokes entirely (a real generated
    line always carries the model's paper wash)."""
    img = Image.new("RGB", (width, height), (235, 232, 226))
    draw = ImageDraw.Draw(img)
    draw.line((4, height // 2, width - 8, height // 2), fill=(38, 34, 46), width=3)
    draw.line((width // 3, 4, width // 3, height - 4), fill=(38, 34, 46), width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_bank(path: Path, rows: list[dict]) -> str:
    table = pa.table({key: [row[key] for row in rows] for key in rows[0]})
    lance.write_dataset(table, str(path))
    return str(path)


def _coherent_rows() -> list[dict]:
    """A coherent-schema bank: doc A has a gap (line_idx 3-4 rejected at bank build), doc B is unbroken.
    Rows are shuffled so runs must be recovered from line_idx, not from the bank's write order."""
    rows = [
        {
            "text": f"doc {doc} rad {i} innehåll",
            "png": _png(),
            "cer": 0.05 * (i % 3),
            "doc_id": doc,
            "line_idx": i,
        }
        for doc, idxs in (("A", [0, 1, 2, 5, 6]), ("B", [3, 4, 5, 6, 7, 8]))
        for i in idxs
    ]
    random.Random(7).shuffle(rows)
    return rows


def test_contiguous_runs_respects_gaps() -> None:
    doc_ids = ["A", "A", "A", "A", "B", "B"]
    line_idxs = [0, 1, 5, 6, 2, 3]
    runs = contiguous_runs(doc_ids, line_idxs)
    # A's gap (2..4 rejected) must split its rows; B stays one run; no run bridges documents
    assert sorted(runs) == [[0, 1], [2, 3], [4, 5]]


def test_contiguous_runs_orders_rows_by_line_idx() -> None:
    runs = contiguous_runs(["A", "A", "A"], [2, 0, 1])
    assert runs == [[1, 2, 0]], "bank row order is write order — a run must restore the passage order"


def test_coherent_sample_is_consecutive_lines_of_one_doc(tmp_path: Path) -> None:
    bank = LineBank(_write_bank(tmp_path / "bank", _coherent_rows()))
    assert bank.coherent
    assert bank.doc_ids is not None and bank.line_idxs is not None
    rng = random.Random(0)
    for _ in range(50):
        rows = bank.sample_run(rng, n_lines=3)
        assert len(rows) in {2, 3}, "3 when the drawn run allows, the whole run when it is shorter"
        assert len({bank.doc_ids[r] for r in rows}) == 1, "one page = one document"
        idxs = [bank.line_idxs[r] for r in rows]
        assert idxs == list(range(idxs[0], idxs[0] + len(idxs))), "a page never bridges a rejected-line gap"


def test_coherent_sample_returns_short_runs_whole(tmp_path: Path) -> None:
    bank = LineBank(_write_bank(tmp_path / "bank", _coherent_rows()))
    assert bank.doc_ids is not None and bank.line_idxs is not None
    rng = random.Random(1)
    for _ in range(30):
        rows = bank.sample_run(rng, n_lines=40)  # longer than every run (max 6) — returns the run itself
        assert len(rows) in {2, 3, 6}
        idxs = [bank.line_idxs[r] for r in rows]
        assert idxs == list(range(idxs[0], idxs[0] + len(idxs)))


def test_random_bank_sampling_returns_one_writers_rows(tmp_path: Path) -> None:
    rows = [{"text": f"w1 rad {i}", "png": _png(), "cer": 0.02, "style_path": "w1.png"} for i in range(6)]
    rows += [{"text": f"w2 rad {i}", "png": _png(), "cer": 0.02, "style_path": "w2.png"} for i in range(3)]
    bank = LineBank(_write_bank(tmp_path / "bank", rows), min_writer_lines=4)
    assert not bank.coherent
    rng = random.Random(2)
    for _ in range(20):
        picked = bank.sample_run(rng, n_lines=4)
        assert len(picked) == len(set(picked)) == 4
        assert all(r < 6 for r in picked), "w2 has fewer than min_writer_lines rows and must never be drawn"


def test_plain_bank_without_style_falls_back_to_one_pool(tmp_path: Path) -> None:
    # the existing 344k-line bank's schema: text/png/pred/cer — no style_path, no doc_id/line_idx
    rows = [{"text": f"rad {i}", "png": _png(), "pred": f"rad {i}", "cer": 0.1} for i in range(5)]
    bank = LineBank(_write_bank(tmp_path / "bank", rows), min_writer_lines=3)
    assert not bank.coherent
    picked = bank.sample_run(random.Random(0), n_lines=4)
    assert len(picked) == len(set(picked)) == 4
    assert all(0 <= r < 5 for r in picked)


def test_fetch_decodes_pngs_in_index_order(tmp_path: Path) -> None:
    rows = [{"text": f"rad {i}", "png": _png(width=60 + 10 * i), "cer": 0.1 * i} for i in range(5)]
    bank = LineBank(_write_bank(tmp_path / "bank", rows))
    fetched = bank.fetch([3, 0, 2])
    assert [text for text, _, _ in fetched] == ["rad 3", "rad 0", "rad 2"]
    assert [cer for _, cer, _ in fetched] == pytest.approx([0.3, 0.0, 0.2])
    assert [pil.size for _, _, pil in fetched] == [(90, 32), (60, 32), (80, 32)]
    assert all(pil.mode == "RGB" for _, _, pil in fetched)


def test_bank_requires_the_core_columns(tmp_path: Path) -> None:
    path = _write_bank(tmp_path / "bank", [{"text": "rad", "png": _png()}])  # no cer column
    with pytest.raises(ValueError, match="cer"):
        LineBank(path)


def test_gate_stored_cers_accepts_and_rejects() -> None:
    keep, cer = gate_stored_cers([0.1, 0.2, 0.3], line_cer_max=0.35, max_bad_frac=0.12)
    assert keep and cer == pytest.approx(0.2)
    keep, cer = gate_stored_cers([0.1, 0.1, 0.9], line_cer_max=0.35, max_bad_frac=0.12)
    assert not keep, "1/3 bad lines exceeds the 0.12 cap"
    assert cer == pytest.approx((0.1 + 0.1 + 0.9) / 3), "page_cer reports even on a rejected page"
    # boundaries mirror the render-path gate: cer == line_cer_max is NOT bad; bad_frac == max_bad_frac keeps
    keep, _ = gate_stored_cers([0.35, 0.35], line_cer_max=0.35, max_bad_frac=0.0)
    assert keep
    keep, _ = gate_stored_cers([0.9] + [0.0] * 9, line_cer_max=0.35, max_bad_frac=0.1)
    assert keep
    keep, cer = gate_stored_cers([], line_cer_max=0.35, max_bad_frac=1.0)
    assert not keep and cer == 1.0, "an empty page (nothing seated) must never be kept"


def test_empirical_measure_scales_with_ink_aspect() -> None:
    pil = Image.new("RGB", (100, 25), "white")  # 10 chars over a 4:1 ink -> 0.4 px/char at height 1
    measure = empirical_measure([("abcdefghij", 0.0, pil)])
    assert measure("abcdefghij", 25) == 100, "the full line at native height measures its true ink width"
    assert measure("abcde", 50) == 100, "half the chars at double the height — same pixels"
    assert measure("", 25) == 1, "never returns a degenerate 0 width"


def test_page_actor_bank_mode_composes_and_gates_without_gpu(tmp_path: Path) -> None:
    from diffu_page.layout import PageSpec
    from diffu_runner.pages import PageActor, PageConfig

    bank_path = _write_bank(tmp_path / "bank", _coherent_rows())
    paper = tmp_path / "paper"
    paper.mkdir()  # empty dir -> procedural paper (no dependency on repo assets)
    cfg = PageConfig(
        ckpt="unused-in-bank-mode.safetensors",  # must never be opened — the paths don't exist
        train_jsonl="unused-in-bank-mode.jsonl",
        out=str(tmp_path / "out"),
        paper_dir=str(paper),
        bank=bank_path,
    )
    actor = PageActor(cfg)
    spec = PageSpec(template="single_column", width=640, height=480, line_height=40)
    jobs = [
        {
            "doc": {"texts": [], "style": ""},  # bank mode ignores the doc — run_pages passes filler
            "kind": "prose",
            "spec": spec.model_dump(),
            "seed": 5,
            "profile": "warm_sepia",
        }
    ]
    kept = actor.make(jobs)
    assert len(kept) == 1
    page = kept[0]
    assert set(page) == {
        "png",
        "template",
        "page_cer",
        "n_lines",
        "labels",
        "seed",
        "profile",
        "bank_rows",
    }, "the kept payload = image + structured GT + provenance (seed/profile/exact bank rows)"
    assert page["template"] == "prose" and page["n_lines"] >= 1
    assert 0.0 <= page["page_cer"] <= 0.35, "page_cer is the mean of the used lines' STORED cers"
    assert page["seed"] == 5 and page["profile"] == "warm_sepia"
    from diffu_page.labels import PageLabels

    labels = PageLabels.model_validate_json(page["labels"])  # structured GT survives the actor hand-off
    assert len(labels.lines) == page["n_lines"] and labels.width == 640 and labels.height == 480
    bank = actor.bank
    assert page["bank_rows"] and all(0 <= r < len(bank) for r in page["bank_rows"]), (
        "bank_rows are dataset row ids — the page's line provenance joins back to the bank"
    )
    seated_texts = {ln.text for ln in labels.lines if ln.text.strip()}
    assert seated_texts <= {bank.texts[r] for r in page["bank_rows"]}, (
        "every label text comes from a listed row"
    )
    img = Image.open(io.BytesIO(page["png"]))
    assert img.size == (640, 480)
    # the stored-CER gate wiring: an impossible threshold rejects the very same job (no recognizer runs)
    strict = cfg.model_copy(update={"line_cer_max": -1.0, "max_bad_frac": 0.0})
    assert PageActor(strict).make(jobs) == []
