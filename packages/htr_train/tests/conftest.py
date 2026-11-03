"""Fixtures for htr-train's tests: tiny line sets as diffu-reader-set writes them (each sample's crop, conversation and
meta line, every page real), a run's settings in a temporary folder, and the renderer's font."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from diffu_htr_train.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from conftest import MakeSet, SetLine


@pytest.fixture(scope="session")
def jev_font() -> Path:
    """The Jev renderer's font, as its settings name it (``HTR_TRAIN_JEV_FONT``; DejaVu Sans by default): a test that
    renders fails here, by name, on a machine without it."""
    font = Settings().jev_font
    if not font.is_file():
        pytest.fail(f"the Jev renderer's font {font} is missing: install fonts-dejavu-core or set HTR_TRAIN_JEV_FONT")
    return font


@pytest.fixture
def make_set(tmp_path: Path) -> MakeSet:
    """A builder of line sets in ``tmp_path/sets``: each line a grey crop of ``size`` (its shade from its place, so
    crops differ), asked ``<image>Text Recognition:`` and answered with its text, its meta line beside it."""

    def make(name: str, lines: Sequence[SetLine], size: tuple[int, int] = (200, 40)) -> Path:
        folder = tmp_path / "sets"
        crops = folder / f"{name}_lines"
        crops.mkdir(parents=True, exist_ok=True)
        rows, metas = [], []
        for k, (line_id, text, collection, volume, split) in enumerate(lines):
            crop = crops / f"{line_id}.jpg"
            Image.new("RGB", size, (40 * k % 256,) * 3).save(crop)
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": "<image>Text Recognition:"},
                        {"role": "assistant", "content": text},
                    ],
                    "images": [str(crop.resolve())],
                }
            )
            metas.append(
                {
                    "id": line_id,
                    "source": "real",
                    "collection": collection,
                    "volume": volume,
                    "scan": f"{line_id}.jpg",
                    "side": None,
                    "split": split,
                    "kind": "body",
                    "production": "handwritten",
                    "element": 0,
                    "line": k,
                }
            )
        path = folder / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        (folder / f"{name}.meta.jsonl").write_text(
            "".join(json.dumps(meta, ensure_ascii=False) + "\n" for meta in metas), encoding="utf-8"
        )
        return path

    return make


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """htr-train's settings with every machine path in ``tmp_path``: runs, envs, the queue, the guard's floor file
    (holding 15) and the guard's path (nothing there until a test puts a stand-in)."""
    floor = tmp_path / "queue" / "guard_floor_gb"
    floor.parent.mkdir(parents=True)
    floor.write_text("15\n", encoding="utf-8")
    return Settings(
        runs=tmp_path / "runs",
        envs=tmp_path / "envs",
        guard=tmp_path / "jobs" / "spark_guard.sh",
        floor_file=floor,
        queue=tmp_path / "queue",
        memory_gb=119,
    )
