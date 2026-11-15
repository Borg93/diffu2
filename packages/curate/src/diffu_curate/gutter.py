"""An opening's gutter read from the scan itself, by a small network on a thumbnail.

The set builder cuts a scan where its labelled lines leave a band free, which fails where the labels do: a facing
page without a line puts the cut beyond the fold, lines close to the fold on both pages leave no band, and a scan no
wider than tall is never looked at. The fold is in the image. :class:`GutterNet` (0.2 M parameters) reads a greyscale
thumbnail and says, a column, how likely the gutter is there; a single sheet has none.

It is trained on what the page table already knows: openings whose two pages both hold labelled lines are cut between
their text blocks (:func:`training_scans`), and tall scans kept whole are single sheets. Those cuts are the middle of
a gap, not the fold, so the network ends nearer the fold than its labels.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from multiprocessing import Pool as ProcessPool
from pathlib import Path

import lance
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import BaseModel
from torch import nn

from diffu_curate.table import append_rows, compact, held_values

THUMBNAIL = (384, 256)  # width, height: every scan is read at this size, whatever its shape
COLUMNS = THUMBNAIL[0] // 4  # the columns the network answers on
SEARCH = (0.15, 0.85)  # a gutter is looked for in this part of the scan's width
FOUND = 0.5  # from this score the scan is cut at its gutter
LABEL_SPREAD = 0.012  # how wide, as a share of the width, a training cut is smeared
SCANS_A_WRITE = 20_000

Image.MAX_IMAGE_PIXELS = None  # the scans are large; they are our own


class Gutter(BaseModel):
    """What the network says of a scan: how sure it is that the scan has a gutter, and where (a share of its width)."""

    score: float
    x: float

    @property
    def found(self) -> bool:
        return self.score >= FOUND


class ScanGutter(BaseModel):
    """A scan's row in the table of gutters, by its image's path."""

    image: str
    pool: str
    volume: str
    scan: str
    gutter_score: float | None = None
    gutter_x: float | None = None


def scan_thumbnail(path: str) -> np.ndarray | None:
    """The scan at ``path`` as a greyscale thumbnail (rows, columns); None if it cannot be read."""
    try:
        with Image.open(path) as scan:
            scan.draft("L", (THUMBNAIL[0] * 2, THUMBNAIL[1] * 2))
            return np.asarray(scan.convert("L").resize(THUMBNAIL, Image.Resampling.BILINEAR), dtype=np.uint8)
    except (OSError, ValueError):
        return None


def _block(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1), nn.BatchNorm2d(outputs), nn.ReLU(),
        nn.Conv2d(outputs, outputs, 3, padding=1), nn.BatchNorm2d(outputs), nn.ReLU(),
    )  # fmt: skip


class GutterNet(nn.Module):
    """Thumbnails (batch, 1, rows, columns) to a logit a column (batch, :data:`COLUMNS`): four blocks of convolutions,
    the rows pooled away, a convolution along the columns."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_block(1, 16), _block(16, 32), _block(32, 64), _block(64, 96)])
        self.head = nn.Conv1d(96, 1, 5, padding=2)

    def forward(self, thumbnails: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(self.blocks[0](thumbnails), 2)
        x = F.max_pool2d(self.blocks[1](x), 2)
        x = self.blocks[2](x)
        x = self.blocks[3](F.max_pool2d(x, (2, 1)))
        return self.head(x.mean(2)).squeeze(1)


def _as_input(thumbnails: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(thumbnails).float().div(255).unsqueeze(1)


def find_gutters(net: GutterNet, thumbnails: np.ndarray, batch: int = 256) -> list[Gutter]:
    """What ``net`` says of each thumbnail (scans, rows, columns): the most likely column in :data:`SEARCH` and its
    score."""
    net.eval()
    first, last = int(SEARCH[0] * COLUMNS), int(SEARCH[1] * COLUMNS)
    found = []
    with torch.inference_mode():
        for start in range(0, len(thumbnails), batch):
            likely = torch.sigmoid(net(_as_input(thumbnails[start : start + batch])))[:, first:last]
            scores, columns = likely.max(1)
            found += [
                Gutter(score=float(score), x=(int(column) + first + 0.5) / COLUMNS)
                for score, column in zip(scores, columns, strict=True)
            ]
    return found


def train_gutter_net(
    thumbnails: np.ndarray, cuts: list[float | None], *, epochs: int = 8, seed: int = 0, batch: int = 64
) -> GutterNet:
    """A network trained on ``thumbnails`` and their ``cuts`` (a share of the width; None: a single sheet), each cut
    smeared over its neighbouring columns, the thumbnails' exposure varied."""
    torch.manual_seed(seed)
    inputs = _as_input(thumbnails)
    centres = (torch.arange(COLUMNS) + 0.5) / COLUMNS
    targets = torch.zeros(len(cuts), COLUMNS)
    for row, cut in enumerate(cuts):
        if cut is not None:
            targets[row] = torch.exp(-0.5 * ((centres - cut) / LABEL_SPREAD) ** 2)
    net = GutterNet()
    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * -(-len(inputs) // batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, 2e-3, total_steps=steps)
    for _ in range(epochs):
        net.train()
        order = torch.randperm(len(inputs))
        for start in range(0, len(order), batch):
            rows = order[start : start + batch]
            gain = 0.8 + 0.4 * torch.rand(len(rows), 1, 1, 1)
            offset = 0.1 * (torch.rand(len(rows), 1, 1, 1) - 0.5)
            loss = F.binary_cross_entropy_with_logits(
                net(inputs[rows] * gain + offset), targets[rows], pos_weight=torch.tensor(8.0)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
    return net


def save_gutter_net(net: GutterNet, path: Path) -> None:
    """Write the network's weights to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), path)


def load_gutter_net(path: Path) -> GutterNet:
    """The network whose weights :func:`save_gutter_net` wrote to ``path``."""
    net = GutterNet()
    net.load_state_dict(torch.load(path, weights_only=True))
    return net.eval()


def training_scans(pages: Path, openings: int, sheets: int, seed: int) -> list[tuple[str, float | None]]:
    """Scans to train on from the page table at ``pages``, as (image, cut): openings whose two pages both hold lines
    (their cut the left page's right edge), and tall scans kept whole with lines on them (single sheets, no cut), as
    many of each from every pool; the ground truth's test scans left out."""
    rows = (
        lance.dataset(str(pages))
        .to_table(
            columns=["pool", "side", "image", "scan_width", "scan_height", "x1", "lines", "alone", "split"],
            filter="error IS NULL AND lines > 0",
        )
        .to_pylist()
    )
    rows = [row for row in rows if row["split"] != "test"]
    chosen: list[tuple[str, float | None]] = []
    draw = random.Random(seed)
    for pool in sorted({row["pool"] for row in rows}):
        ours = [row for row in rows if row["pool"] == pool]
        cut = [(r["image"], r["x1"] / r["scan_width"]) for r in ours if r["side"] == "left" and r["alone"] is False]
        whole = [(r["image"], None) for r in ours if r["side"] == "whole" and r["scan_width"] < 0.8 * r["scan_height"]]
        chosen += draw.sample(cut, min(openings, len(cut))) + draw.sample(whole, min(sheets, len(whole)))
    draw.shuffle(chosen)
    return chosen


def read_thumbnails(images: list[str], workers: int) -> list[np.ndarray | None]:
    """The thumbnail of each scan in ``images``, read by ``workers`` processes."""
    if workers <= 1:
        return [scan_thumbnail(image) for image in images]
    with ProcessPool(workers) as processes:
        return processes.map(scan_thumbnail, images, chunksize=64)


def write_gutters(net: GutterNet, pages: Path, table: Path, workers: int) -> Iterator[tuple[int, int]]:
    """Append to ``table`` the gutter of every scan the page table at ``pages`` names, a scan ``table`` already holds
    left as it is (a scan that cannot be read is a row without a score). Yields (scans done, scans to do)."""
    held = held_values(table, "image", "image IS NOT NULL")
    named = lance.dataset(str(pages)).to_table(columns=["image", "pool", "volume", "scan"]).to_pylist()
    scans = sorted({row["image"]: row for row in named if row["image"] not in held}.values(), key=lambda r: r["image"])
    for start in range(0, len(scans), SCANS_A_WRITE):
        chunk = scans[start : start + SCANS_A_WRITE]
        thumbnails = read_thumbnails([row["image"] for row in chunk], workers)
        readable = [(i, thumbnail) for i, thumbnail in enumerate(thumbnails) if thumbnail is not None]
        gutters = find_gutters(net, np.stack([thumbnail for _, thumbnail in readable])) if readable else []
        found = {i: gutter for (i, _), gutter in zip(readable, gutters, strict=True)}
        rows = []
        for i, row in enumerate(chunk):
            gutter = found.get(i)
            rows.append(
                ScanGutter(
                    **row,
                    gutter_score=gutter.score if gutter else None,
                    gutter_x=gutter.x if gutter else None,
                )
            )
        append_rows(table, rows)
        yield start + len(chunk), len(scans)
    compact(table)


def read_gutters(table: Path, pool: str) -> dict[str, Gutter]:
    """The gutters ``table`` holds for ``pool``'s scans, by image path; a scan without a score is left out."""
    rows = (
        lance.dataset(str(table))
        .to_table(columns=["image", "gutter_score", "gutter_x"], filter=f"pool = '{pool}' AND gutter_score IS NOT NULL")
        .to_pylist()
    )
    return {row["image"]: Gutter(score=row["gutter_score"], x=row["gutter_x"]) for row in rows}
