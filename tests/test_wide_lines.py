"""Lines wider than the cap: skipped with drop_wide_lines (not squashed); canvas refs prefer ones that fit."""

from __future__ import annotations

import json

from PIL import Image

from diffu.config import Config
from diffu.data.dataset import HandwritingLineDataset


def _rows(tmp_path, widths: list[int]) -> list[dict[str, str]]:
    rows = []
    for i, w in enumerate(widths):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (w, 32), "white").save(path)
        rows.append({"image": str(path), "text": f"line {i}", "writer_id": "w0"})
    return rows


def test_drop_wide_lines_skips_instead_of_squashing(tmp_path):
    from diffu.train import build_dataloader

    with open(tmp_path / "train.jsonl", "w", encoding="utf-8") as f:
        for row in _rows(tmp_path, [40, 300, 90]):
            f.write(json.dumps(row) + "\n")
    cfg = Config()
    cfg.data.line_height, cfg.data.max_line_width = 32, 100
    squashing = build_dataloader(cfg, str(tmp_path), 1, 0, strict=False)
    cfg.data.drop_wide_lines = True
    dropping = build_dataloader(cfg, str(tmp_path), 1, 0, strict=False)
    assert len(squashing.dataset) == 3 and len(dropping.dataset) == 2


def test_canvas_refs_prefer_a_peer_that_fits_uncropped(tmp_path):
    rows = _rows(tmp_path, [100, 5000, 200])
    ds = HandwritingLineDataset(rows, height=32, ref_lines=True, widths=[100, 5000, 200], max_ref_width=1000)
    assert {ds._style_index("w0", 0) for _ in range(20)} == {2}  # never the 5000 px peer
    ds.max_ref_width = 100  # nothing else fits -> any other peer, as before
    assert {ds._style_index("w0", 0) for _ in range(40)} == {1, 2}


def test_resume_auto_picks_the_newest_complete_checkpoint(tmp_path):
    from diffu.train import _latest_checkpoint

    assert _latest_checkpoint(str(tmp_path / "missing")) is None
    for step, complete in ((1000, True), (3000, True), (4000, False)):  # step_4000: killed mid-save
        (tmp_path / f"step_{step}").mkdir()
        if complete:
            (tmp_path / f"step_{step}" / "config.json").write_text("{}")
    (tmp_path / "final").mkdir()
    assert _latest_checkpoint(str(tmp_path)) == str(tmp_path / "step_3000")
