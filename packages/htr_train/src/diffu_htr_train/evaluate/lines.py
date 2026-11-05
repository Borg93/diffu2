"""The lines a reader is scored on, one format for every source (``lines.jsonl``: an id, the subset it is scored in,
its image's path and its reference text): the Swedish Lion's out-of-domain set (``Riksarkivet/eval_htr_out_of_domain_lines``,
a parquet file a volume, its images written out) and the test side of one of our line sets (its collection the subset)."""

from __future__ import annotations

import io
import json
import random
from collections.abc import Iterator
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

OOD_REPO = "Riksarkivet/eval_htr_out_of_domain_lines"


class EvalLine(BaseModel):
    """A line to read: its id, the subset it is scored in, its image (an absolute path) and its reference text."""

    id: str
    subset: str
    image: str
    text: str


def read_lines(path: Path) -> list[EvalLine]:
    """The lines of a ``lines.jsonl``."""
    with path.open(encoding="utf-8") as rows:
        return [EvalLine.model_validate_json(row) for row in rows if row.strip()]


def write_lines(lines: list[EvalLine], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line.model_dump_json()}\n" for line in lines), encoding="utf-8")


def ood_lines(parquets: list[Path], images: Path) -> Iterator[EvalLine]:
    """The out-of-domain set's lines, each image written into ``images`` as PNG, the volume (its file's stem less
    ``_eval_lines_1``) the subset."""
    import pyarrow.parquet as pq

    images.mkdir(parents=True, exist_ok=True)
    for parquet in sorted(parquets):
        subset = parquet.stem.removesuffix("_eval_lines_1")
        for k, row in enumerate(pq.read_table(parquet).to_pylist()):
            path = images / f"{subset}_{k:04d}.png"
            if not path.exists():
                Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB").save(path)
            yield EvalLine(id=f"{subset}_{k:04d}", subset=subset, image=str(path.resolve()), text=row["transcription"])


def sampled(lines: list[EvalLine], count: int | None, seed: int) -> list[EvalLine]:
    """``count`` of ``lines`` drawn by ``seed``, in their order (all of them where ``count`` is None or no smaller)."""
    if count is None or count >= len(lines):
        return lines
    keep = set(random.Random(seed).sample(range(len(lines)), count))
    return [line for k, line in enumerate(lines) if k in keep]


def set_lines(samples: Path) -> Iterator[EvalLine]:
    """The lines of one of our line sets (``diffu-reader-set --task lines``: a chat sample a line and its meta file
    beside it), each in its collection's subset."""
    meta = samples.with_name(f"{samples.stem}.meta.jsonl")
    with samples.open(encoding="utf-8") as rows, meta.open(encoding="utf-8") as metas:
        for row, meta_row in zip(rows, metas, strict=True):
            sample, info = json.loads(row), json.loads(meta_row)
            answer = next(m["content"] for m in sample["messages"] if m["role"] == "assistant")
            yield EvalLine(id=info["id"], subset=info["collection"], image=sample["images"][0], text=answer)
