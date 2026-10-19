"""Train straight off a Hugging Face ``datasets`` dataset of (line image, transcription).

Wraps a Hub dataset (e.g. ``Teklia/IAM-line`` — columns ``image`` + ``text``) into the SAME item
contract as :class:`~diffu.data.dataset.HandwritingLineDataset`, so it drops into ``collate_lines`` and
the training loop unchanged. Column names are auto-detected (overridable). If the dataset has no writer
column, the style reference is a random OTHER line (at inference the style ref is always a foreign line,
so training on a random peer matches — never the target's own image, which would leak its content).

``datasets`` is an optional dependency: ``uv sync --extra hf`` (or ``pip install datasets``).
"""

from __future__ import annotations

import random
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from PIL import Image

from .dataset import normalize_text, style_tensor

if TYPE_CHECKING:
    import torch


_IMAGE_COLS = ["image", "img", "pixel_values"]
_TEXT_COLS = ["text", "transcription", "label", "sentence", "gt"]
_WRITER_COLS = ["writer", "writer_id", "author", "form"]


def _load(name: str, split: str, max_rows: int | None) -> Any:
    """``load_dataset`` with a concrete ``split`` returns a Dataset; its published type is a broad union
    (Dataset | DatasetDict | Iterable…) that the checker won't let us index by int — so type it Any."""
    from datasets import load_dataset

    ds: Any = load_dataset(name, split=split)
    if max_rows is not None and max_rows < len(ds):
        ds = ds.select(range(max_rows))
    return ds


def _pick(cols: list[str], candidates: list[str], *, required: bool = True) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise ValueError(f"could not find any of {candidates} in dataset columns {cols}")
    return None


class HFLineDataset:
    """A HF image+text dataset as diffu training triples ``{image, text, writer_id, style_pixel_values}``."""

    def __init__(
        self,
        name: str,
        split: str = "train",
        *,
        height: int = 128,
        max_width: int = 1024,
        image_col: str | None = None,
        text_col: str | None = None,
        writer_col: str | None = None,
        max_rows: int | None = None,
        ref_lines: bool = False,
    ) -> None:
        ds = _load(name, split, max_rows)
        self.ds: Any = ds
        self.height, self.max_width = height, max_width
        self.ref_lines = ref_lines  # also return the reference at line height (style_mode="canvas")
        cols = list(ds.column_names)
        self.image_col = image_col or _pick(cols, _IMAGE_COLS)
        self.text_col = text_col or _pick(cols, _TEXT_COLS)
        self.writer_col = writer_col or _pick(cols, _WRITER_COLS, required=False)
        if ref_lines and self.writer_col is None:
            warnings.warn(
                f"{name} has no writer column: each style reference is a RANDOM other line (usually another "
                "writer), so the model cannot learn to copy a hand from it. Use a dataset with writer ids.",
                stacklevel=2,
            )
        # Style-peer index: group by writer when the column exists, else draw a random other line. The ids
        # are read once here: indexing a row to get them would decode its image too.
        self.writer_ids = [str(w) for w in ds[self.writer_col]] if self.writer_col is not None else None
        self.by_writer: defaultdict[str, list[int]] = defaultdict(list)
        for i, w in enumerate(self.writer_ids or ()):
            self.by_writer[w].append(i)

    def __len__(self) -> int:
        return len(self.ds)

    def _to_line(self, pil: Image.Image) -> torch.Tensor:
        from torchvision.transforms import functional as TF

        img = pil.convert("RGB")
        w, h = img.size
        new_w = min(self.max_width, max(8, round(w * self.height / max(h, 1))))
        img = img.resize((new_w, self.height), Image.LANCZOS)
        return TF.normalize(TF.to_tensor(img), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [3,H,W] in [-1,1]

    def _style_index(self, i: int) -> int:
        if self.writer_ids is not None:
            peers = [p for p in self.by_writer[self.writer_ids[i]] if p != i]
            if peers:
                return random.choice(peers)
        if len(self.ds) == 1:
            return i
        j = random.randrange(len(self.ds) - 1)  # a uniformly-random OTHER line
        return j if j < i else j + 1

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.ds[i]
        si = self._style_index(i)
        srow = self.ds[si]
        item: dict[str, Any] = {
            "image": self._to_line(row[self.image_col]),
            "text": normalize_text(str(row[self.text_col])),
            "writer_id": self.writer_ids[i] if self.writer_ids is not None else str(i),
            "style_text": normalize_text(str(srow[self.text_col])),
        }
        if self.ref_lines:  # the reference at line height (style_mode="canvas"; no DINO crop)
            item["style_image"] = self._to_line(srow[self.image_col])
        else:
            item["style_pixel_values"] = style_tensor(srow[self.image_col])
        return item


def hf_to_manifest(
    name: str,
    split: str,
    out_dir: str,
    *,
    image_col: str | None = None,
    text_col: str | None = None,
    max_rows: int | None = None,
) -> str:
    """Dump a Hub image+text split to ``out_dir/lines/*.png`` + ``out_dir/{split}.jsonl`` (image, text, and
    writer_id: the dataset's writer column, else the row index).

    The REAL-data side of the augmentation A/B: materialize e.g. IAM train/test as a local manifest so
    ``diffu-htr-train`` / ``diffu-eval`` can consume it exactly like the synthetic manifest. Returns the
    manifest path. Skips rows with empty text.
    """
    import json
    import os

    ds = _load(name, split, max_rows)
    cols = list(ds.column_names)
    icol = image_col or _pick(cols, _IMAGE_COLS)
    tcol = text_col or _pick(cols, _TEXT_COLS)
    wcol = _pick(cols, _WRITER_COLS, required=False)
    lines_dir = os.path.join(out_dir, "lines")
    os.makedirs(lines_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{split}.jsonl")
    n = 0
    with open(path, "w", encoding="utf-8") as mf:
        for i in range(len(ds)):
            row = ds[i]
            text = normalize_text(str(row[tcol]))
            if not text:
                continue
            png = os.path.join(lines_dir, f"{i:07d}.png")
            row[icol].convert("RGB").save(png)
            writer = str(row[wcol]) if wcol is not None else str(i)
            mf.write(json.dumps({"image": png, "text": text, "writer_id": writer}, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} real lines -> {path}", flush=True)
    return path
