"""The coherent column path END-TO-END through a real (CPU-only) Ray Data pipeline.

Every other test stubs the seam it sits next to; this one keeps the REAL from_items/override_num_blocks,
the real ``ink_ok`` map lambda, the real filter, the real column drop and the real ``lance_ray.write_lance``
— only the two model stages are dict-of-lists passthrough fakes shaped exactly like the actors. It is the
guard for the 1M-line bank: a column drop/reorder anywhere in the DAG would silently mislabel the corpus.
Slowest test in the suite (~20s: it boots a local Ray cluster) — that is the price of the seam.
"""

from __future__ import annotations

import io
from pathlib import Path

from diffu_runner.bank import LineBank
from diffu_runner.gate import GateConfig, cer_ok, ink_ok
from diffu_runner.pipeline import _pil, _shuffle_for_mixed_length_batches
from PIL import Image, ImageDraw


def test_coherent_columns_survive_to_the_written_bank(tmp_path: Path, monkeypatch) -> None:
    import ray
    from lance_ray import write_lance

    # NESTED fakes: cloudpickle ships local functions BY VALUE — module-level ones are pickled by
    # reference to this test module, which Ray workers cannot import.
    def _png_bytes(seed: int) -> bytes:
        img = Image.new("RGB", (96, 32), (235, 232, 226))
        d = ImageDraw.Draw(img)
        d.line((4, 16, 90, 16), fill=(30, 30, 40), width=3)
        d.line((30 + seed % 20, 4, 30 + seed % 20, 28), fill=(30, 30, 40), width=2)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _fake_diffu(batch: dict) -> dict:
        # mirrors DiffuActor.__call__'s return shape exactly (full passthrough + png)
        return {
            **{k: list(v) for k, v in batch.items()},
            "png": [_png_bytes(i) for i in range(len(batch["text"]))],
        }

    def _fake_trocr(batch: dict) -> dict:
        # reject doc B's line_idx==2 via CER, so the written bank carries a gate gap
        cers = [
            0.9 if (str(d) == "B" and int(li) == 2) else 0.05
            for d, li in zip(batch["doc_id"], batch["line_idx"], strict=True)
        ]
        return {**{k: list(v) for k, v in batch.items()}, "pred": list(batch["text"]), "cer": cers}

    # Ray's uv-aware runtime env resolves the project from the CWD — from packages/diffu-runner it points
    # workers at a different environment and the cluster fails to come up; pin to the repo root.
    monkeypatch.chdir(Path(__file__).resolve().parents[3])
    gate = GateConfig()
    worklist = [
        {
            "text": f"doc {d} rad {i} innehall har",
            "style_path": f"lines/{d}_{i:03d}.png",
            "doc_id": d,
            "line_idx": i,
        }
        for d in ("A", "B")
        for i in range(4)
    ]
    _shuffle_for_mixed_length_batches(worklist, seed=0)  # same shuffle as the builders
    out = str(tmp_path / "bank")
    try:
        ray.init(include_dashboard=False, num_cpus=2, logging_level="ERROR")
        ds = ray.data.from_items(worklist, override_num_blocks=max(1, -(-len(worklist) // 3)))
        ds = ds.map_batches(_fake_diffu, batch_size=3)
        ds = ds.map(lambda row: {**row, "ink_ok": ink_ok(_pil(row["png"]), gate)})  # the REAL flag stage
        ds = ds.map_batches(_fake_trocr, batch_size=3)
        ds = ds.filter(lambda row: row["ink_ok"] and cer_ok(row["cer"], gate))
        ds = ds.map_batches(lambda b: {k: v for k, v in b.items() if k not in ("ink_ok", "pred")})
        write_lance(ds, out)
    finally:
        ray.shutdown()

    import lance

    tbl = lance.dataset(out).to_table()
    assert set(tbl.schema.names) == {"text", "style_path", "doc_id", "line_idx", "png"} | {"cer"}
    rows = tbl.to_pylist()
    assert sorted((r["doc_id"], r["line_idx"]) for r in rows) == [
        ("A", 0), ("A", 1), ("A", 2), ("A", 3), ("B", 0), ("B", 1), ("B", 3),
    ]  # fmt: skip
    for r in rows:  # per-row alignment: any column shift would mispair these
        assert r["text"].startswith(f"doc {r['doc_id']} rad {r['line_idx']}"), f"COLUMN MISALIGNMENT: {r}"
        assert r["style_path"] == f"lines/{r['doc_id']}_{r['line_idx']:03d}.png"
        assert isinstance(r["png"], bytes) and Image.open(io.BytesIO(r["png"])).size == (96, 32)

    bank = LineBank(out)
    assert bank.coherent and len(bank) == 7
    assert bank._runs is not None and bank.doc_ids is not None and bank.line_idxs is not None
    run_keys = sorted(sorted((bank.doc_ids[r], bank.line_idxs[r]) for r in run) for run in bank._runs)
    # doc B's rejected line splits it into 2-line/1-line scraps, which the >=4 floor drops from the pool;
    # doc A's intact 4-line run is the only samplable passage
    assert run_keys == [[("A", 0), ("A", 1), ("A", 2), ("A", 3)]]
