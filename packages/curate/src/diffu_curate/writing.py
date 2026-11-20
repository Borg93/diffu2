"""The writing layer: where a scan holds writing, by a small head on DINOv3's patch tokens, set against where its
labels say lines are.

The head (CURATION E10) was fitted on the ground truth's line outlines. A scan fitted to :data:`LONG` pixels gives a
map of "writing here"; the labels' line outlines, closed over the gaps between lines, give a map of "a label covers
this". :func:`coverage` compares the two: writing no label covers points at lines the old pipeline missed, and labels
over no writing at lines it invented or placed wrongly. Neither depends on how well the letters were read.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader, Dataset

from diffu_curate.alto import AltoLine, read_alto

LONG = 1024  # a scan's longer side as the network sees it (E10: no gain from 1,536)
PATCH = 16
MASK_STEP = 2  # the maps are compared at half the fitted scan's pixels
CLOSE = 9  # the label map is closed with a square of this many map pixels: the gaps between lines count as labelled
THRESHOLD = 0.5
MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
HEAD = "1024_dino_linear"  # the head of writing_eval.py's heads.pt that E10 chose
PART = 20_000  # scans a Parquet part
ROWS = pa.schema(
    [
        ("image", pa.utf8()),
        ("written", pa.float32()),
        ("labelled", pa.float32()),
        ("covered", pa.float32()),
        ("filled", pa.float32()),
    ]
)

Image.MAX_IMAGE_PIXELS = None


class Coverage(BaseModel):
    """A scan's two maps compared, each a share between 0 and 1.

    Attributes:
        written: The share of the scan the layer calls writing.
        labelled: The share of the scan under the closed labels.
        covered: The share of the writing that lies under the closed labels (1.0 where there is no writing).
        filled: The share of the closed labels that holds writing (1.0 where there is no label).
    """

    model_config = ConfigDict(frozen=True)

    written: float
    labelled: float
    covered: float
    filled: float


def fitted_size(width: int, height: int) -> tuple[int, int]:
    """The size a scan is resized to: its longer side :data:`LONG`, both sides a multiple of the patch."""
    scale = LONG / max(width, height)
    return max(PATCH, round(width * scale / PATCH) * PATCH), max(PATCH, round(height * scale / PATCH) * PATCH)


def label_mask(lines: Sequence[AltoLine], page: tuple[float, float], size: tuple[int, int]) -> np.ndarray:
    """The lines' outlines drawn as a mask of ``size`` (width, height) for a page of ``page`` (width, height) in the
    labels' own units."""
    mask = Image.new("L", size)
    pen = ImageDraw.Draw(mask)
    across, down = size[0] / page[0], size[1] / page[1]
    for line in lines:
        pen.polygon([(x * across, y * down) for x, y in line.outline], fill=255)
    return np.asarray(mask) > 0


def closed(masks: torch.Tensor) -> torch.Tensor:
    """Boolean masks (batch, height, width) closed with a :data:`CLOSE` square: grown, then shrunk back."""
    grown = F.max_pool2d(masks[:, None].float(), CLOSE, 1, CLOSE // 2)
    return (-F.max_pool2d(-grown, CLOSE, 1, CLOSE // 2))[:, 0] > 0.5


def coverage(writing: torch.Tensor, labels: torch.Tensor) -> list[Coverage]:
    """Each scan's writing map set against its label map (both boolean, batch x height x width)."""
    region = closed(labels)
    pixels = writing[0].numel()
    written, labelled = writing.flatten(1).sum(1), region.flatten(1).sum(1)
    both = (writing & region).flatten(1).sum(1)
    return [
        Coverage(
            written=w / pixels,
            labelled=r / pixels,
            covered=b / w if w else 1.0,
            filled=b / r if r else 1.0,
        )
        for w, r, b in zip(written.tolist(), labelled.tolist(), both.tolist(), strict=True)
    ]


class WritingLayer:
    """DINOv3 and the writing head, on one device."""

    def __init__(self, heads: Path, device: str = "cuda") -> None:
        from transformers import AutoModel  # heavy, and needed only where the network runs

        self.device = device
        half = device == "cuda"
        self.encoder = (
            AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16 if half else torch.float32).to(device).eval()
        )
        state, mean, spread = torch.load(heads, weights_only=True, map_location="cpu")[HEAD]
        self.weight = (state["weight"][0] / spread).to(device)
        self.bias = (state["bias"][0] - (state["weight"][0] * mean / spread).sum()).to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def writing(self, scans: torch.Tensor) -> torch.Tensor:
        """The writing maps (boolean, batch x height/:data:`MASK_STEP` x width/:data:`MASK_STEP`) of scans of one
        fitted size (uint8, batch x 3 x height x width)."""
        pixels = scans.to(self.device).float() / 255
        rows, columns = pixels.shape[2] // PATCH, pixels.shape[3] // PATCH
        tokens = self.encoder(pixel_values=((pixels - self.mean) / self.std).to(self.encoder.dtype)).last_hidden_state
        patches = tokens[:, -rows * columns :].float()
        logits = (patches @ self.weight + self.bias).reshape(-1, 1, rows, columns)
        size = (pixels.shape[2] // MASK_STEP, pixels.shape[3] // MASK_STEP)
        return torch.sigmoid(F.interpolate(logits, size=size, mode="bilinear", align_corners=False))[:, 0] > THRESHOLD


class _Scans(Dataset):
    """(image, label file) pairs as fitted pixels and label masks; an unreadable pair gives None."""

    def __init__(self, pairs: Sequence[tuple[str, str]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor, torch.Tensor] | tuple[int, None, None]:
        image, labels = self.pairs[index]
        try:
            page = read_alto(Path(labels))
            with Image.open(image) as scan:
                scan.draft("RGB", (LONG * 2, LONG * 2))
                size = fitted_size(*scan.size)
                fitted = scan.convert("RGB").resize(size, Image.Resampling.BICUBIC)
        except (OSError, ValueError, SyntaxError):  # an unreadable scan or label file: counted, not fatal
            return index, None, None
        mask = label_mask(page.lines, (page.width, page.height), (size[0] // MASK_STEP, size[1] // MASK_STEP))
        return index, torch.from_numpy(np.asarray(fitted)).permute(2, 0, 1), torch.from_numpy(mask)


def _batches(loader: DataLoader, batch: int) -> Iterator[list[tuple[int, torch.Tensor, torch.Tensor]] | None]:
    """The loader's scans grouped by fitted size, ``batch`` at a time; None for each unreadable scan."""
    waiting: dict[tuple[int, int], list[tuple[int, torch.Tensor, torch.Tensor]]] = {}
    for index, pixels, mask in loader:
        if pixels is None:
            yield None
            continue
        held = waiting.setdefault((pixels.shape[1], pixels.shape[2]), [])
        held.append((index, pixels, mask))
        if len(held) == batch:
            yield list(held)
            held.clear()
    yield from (held for held in waiting.values() if held)


def write_coverage(
    pairs: Sequence[tuple[str, str]],
    out: Path,
    layer: WritingLayer,
    *,
    workers: int = 20,
    batch: int = 16,
    label: str = "0",
) -> tuple[int, int]:
    """Score ``pairs`` of (scan image, its ALTO file) and write Parquet parts (:data:`ROWS`) in ``out``.

    A scan a part of ``out`` holds already is skipped, so a stopped run goes on where it was.

    Returns:
        How many scans were scored, and how many could not be read.
    """
    out.mkdir(parents=True, exist_ok=True)
    done = {
        image
        for part in out.glob("part-*.parquet")
        for image in pq.read_table(part, columns=["image"])["image"].to_pylist()
    }
    todo = [pair for pair in pairs if pair[0] not in done]
    loader = DataLoader(
        _Scans(todo), batch_size=1, num_workers=workers, collate_fn=lambda rows: rows[0], prefetch_factor=4
    )
    rows: list[tuple[str, Coverage]] = []
    scored = lost = 0
    start = time.time()

    def flush() -> None:
        table = pa.table(
            {
                "image": [image for image, _ in rows],
                **{name: [getattr(c, name) for _, c in rows] for name in ROWS.names[1:]},
            },
            schema=ROWS,
        )
        part = out / f"part-{label}-{time.time_ns()}.parquet"
        pq.write_table(table, part.with_suffix(".unfinished"))
        part.with_suffix(".unfinished").replace(part)
        rows.clear()

    for group in _batches(loader, batch):
        if group is None:
            lost += 1
            continue
        writing = layer.writing(torch.stack([pixels for _, pixels, _ in group]))
        labels = torch.stack([mask for _, _, mask in group]).to(writing.device)
        rows.extend(
            (todo[index][0], found) for (index, _, _), found in zip(group, coverage(writing, labels), strict=True)
        )
        before, scored = scored, scored + len(group)
        if len(rows) >= PART:
            flush()
        if scored // 2000 > before // 2000:
            print(
                f"{scored} of {len(todo)} scans, {scored / (time.time() - start):.1f} a second, {lost} unreadable",
                flush=True,
            )
    if rows:
        flush()
    return scored, lost


def quiet_threads() -> None:
    """Before torch loads its threads: a process that feeds a GPU needs one (CURATION E40: 47 to 70 scans a second)."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
