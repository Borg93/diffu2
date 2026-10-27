"""Training the print/hand classifier (CPU) on its line set (:mod:`diffu_page.data.print_hand_set`): random windows of
the lines, the two kinds drawn as often and the collections of a kind about as often, each window jittered (the text's
size in it, its resolution, the ink's weight and contrast) so that neither a scan nor a ground truth's line boxes tell
print from hand.

It is validated by collection: the collections (related ones grouped, :func:`collection_groups`) are dealt into
``--folds`` folds and each fold's lines are read by a net trained on the other folds' lines. ``report.json`` gives every
collection's accuracy so held out and the confusion over all lines at each threshold, ``held_out.npz`` each line's
chance of being printed. The net saved, ``print_hand.pt`` (a state dict, loaded with ``weights_only``), is then trained
on every line.

    OMP_NUM_THREADS=4 nice -n 10 python -m diffu_page.marking.train --lines runs/print_hand/lines.npz \\
        --out runs/print_hand
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import torch
import typer
from pydantic import BaseModel, ConfigDict
from torch import nn

from diffu_page.data.print_hand_set import LineSet, read_line_set
from diffu_page.marking.lines import LINE_HEIGHT
from diffu_page.marking.print_hand import PRINTED_CHANCE, WINDOW, PrintHandClassifier, PrintHandNet, ink
from diffu_page.output import console

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

GROUPED = ("OCRmaskinskrift_", "OCRtidningar_Newseye_")  # the series whose collections share scans or newspapers
THRESHOLDS = (0.5, 0.8, PRINTED_CHANCE, 0.95)
_DATED = re.compile(r"_\d{8}$")  # a collection's export date: two exports of one series are one group


class Settings(BaseModel):
    """How a net is trained: ``epochs`` of ``samples`` windows each, in batches, with AdamW at a learning rate that
    warms up for the first epoch's tenth and then falls along a cosine."""

    model_config = ConfigDict(frozen=True)

    epochs: int = 6
    samples: int = 16_000
    batch: int = 128
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4


class CollectionScore(BaseModel):
    """How a held-out collection's lines were read: their kind, count, the share read right at ``PRINTED_CHANCE`` and
    their mean chance of being printed."""

    collection: str
    printed: bool
    lines: int
    accuracy: float
    mean_chance: float


class Confusion(BaseModel):
    """The held-out lines at a threshold: printed and handwritten lines, each read as printed or as handwritten."""

    threshold: float
    printed_as_printed: int
    printed_as_handwritten: int
    handwritten_as_printed: int
    handwritten_as_handwritten: int


class Report(BaseModel):
    """The validation: the collections each fold held out, every collection's score and the confusion at each
    threshold, and the training settings."""

    folds: list[list[str]]
    collections: list[CollectionScore]
    confusion: list[Confusion]
    settings: Settings


def collection_groups(collections: list[str], grouped: tuple[str, ...] = GROUPED) -> list[str]:
    """Each collection's group, held out together: the series it starts with (``grouped``), else its name without an
    export date (``_YYYYMMDD``)."""
    return [
        next((series for series in grouped if name.startswith(series)), _DATED.sub("", name)) for name in collections
    ]


def deal_folds(groups: list[str], printed: list[bool], *, folds: int, seed: int) -> list[int]:
    """Each collection's fold: its group's, the printed groups and the handwritten ones each dealt round the folds in a
    random (seeded) order, so every fold holds out both kinds."""
    rng = random.Random(seed)
    fold_of: dict[str, int] = {}
    for kind in (True, False):
        kind_groups = sorted({group for group, is_printed in zip(groups, printed, strict=True) if is_printed is kind})
        rng.shuffle(kind_groups)
        fold_of |= {group: k % folds for k, group in enumerate(kind_groups)}
    return [fold_of[group] for group in groups]


def sampling_weights(lines: LineSet, chosen: np.ndarray) -> np.ndarray:
    """How often each chosen line is drawn: each kind half the time, a collection of a kind in proportion to the root of
    its line count (a large one not drowning the rest, a small one not repeated into memorization)."""
    counts = Counter(lines.collection[chosen].tolist())
    weights = np.array([1 / math.sqrt(counts[collection]) for collection in lines.collection[chosen]])
    for kind in (True, False):
        of_kind = lines.printed[chosen] == kind
        weights[of_kind] *= 0.5 / weights[of_kind].sum()
    return weights


def jittered(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random :data:`WINDOW`-wide window of a line image as ink (:func:`diffu_page.marking.print_hand.ink`), the
    text 0.7-1.3 times its size in it, its resolution sometimes lowered, its strokes sometimes thickened or thinned, its
    contrast bent."""
    zoom = rng.uniform(0.7, 1.3)
    height = max(LINE_HEIGHT // 2, round(LINE_HEIGHT * zoom))
    scaled = cv2.resize(image, (max(1, round(image.shape[1] * zoom)), height), interpolation=cv2.INTER_AREA)
    canvas = np.full((LINE_HEIGHT, max(WINDOW, scaled.shape[1])), 255, np.uint8)
    if height >= LINE_HEIGHT:
        top = int(rng.integers(0, height - LINE_HEIGHT + 1))
        canvas[:, : scaled.shape[1]] = scaled[top : top + LINE_HEIGHT]
    else:
        top = int(rng.integers(0, LINE_HEIGHT - height + 1))
        canvas[top : top + height, : scaled.shape[1]] = scaled
    start = int(rng.integers(0, canvas.shape[1] - WINDOW + 1))
    window = canvas[:, start : start + WINDOW]
    if rng.random() < 0.3:
        factor = rng.uniform(0.35, 0.8)
        small = cv2.resize(window, (round(WINDOW * factor), round(LINE_HEIGHT * factor)), interpolation=cv2.INTER_AREA)
        window = cv2.resize(small, (WINDOW, LINE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3:
        stroke = cv2.erode if rng.random() < 0.5 else cv2.dilate  # on dark ink, erode thickens and dilate thins
        window = stroke(window, np.ones((2, 2), np.uint8))
    return ink(window) ** rng.uniform(0.6, 1.5)


def train_net(lines: LineSet, chosen: np.ndarray, settings: Settings, *, seed: int) -> PrintHandNet:
    """A net trained on the ``chosen`` lines."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = PrintHandNet()
    optimizer = torch.optim.AdamW(net.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    steps = settings.epochs * math.ceil(settings.samples / settings.batch)
    warmup = max(1, steps // (10 * settings.epochs))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * step / steps))
    )
    loss_of = nn.BCEWithLogitsLoss()
    weights = sampling_weights(lines, chosen)
    net.train()
    for epoch in range(settings.epochs):
        drawn = rng.choice(chosen, size=settings.samples, p=weights)
        total = 0.0
        for start in range(0, settings.samples, settings.batch):
            batch = drawn[start : start + settings.batch]
            windows = torch.from_numpy(np.stack([jittered(lines.images[i], rng) for i in batch]))[:, None]
            labels = torch.from_numpy(lines.printed[batch].astype(np.float32))
            loss = loss_of(net(windows), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            total += loss.item() * len(batch)
        console.print(f"  epoch {epoch + 1}/{settings.epochs}: loss {total / settings.samples:.4f}")
    return net.eval()


def scores(lines: LineSet, chances: np.ndarray) -> list[CollectionScore]:
    """Every collection's score from its lines' held-out chances of being printed."""
    read_printed = chances >= PRINTED_CHANCE
    return [
        CollectionScore(
            collection=name,
            printed=bool(lines.printed[of][0]),
            lines=int(of.sum()),
            accuracy=float((read_printed[of] == lines.printed[of]).mean()),
            mean_chance=float(chances[of].mean()),
        )
        for k, name in enumerate(lines.collections)
        if (of := lines.collection == k).any()
    ]


def confusion(lines: LineSet, chances: np.ndarray, threshold: float) -> Confusion:
    """The held-out lines' confusion at ``threshold``."""
    read_printed = chances >= threshold
    return Confusion(
        threshold=threshold,
        printed_as_printed=int((read_printed & lines.printed).sum()),
        printed_as_handwritten=int((~read_printed & lines.printed).sum()),
        handwritten_as_printed=int((read_printed & ~lines.printed).sum()),
        handwritten_as_handwritten=int((~read_printed & ~lines.printed).sum()),
    )


@app.command()
def main(
    lines_path: Annotated[Path, typer.Option("--lines", exists=True, dir_okay=False, help="the line set to train on")],
    out: Annotated[Path, typer.Option(help="where print_hand.pt, report.json and held_out.npz go")],
    folds: Annotated[int, typer.Option(min=2, help="folds of collections, each held out once")] = 4,
    epochs: int = Settings().epochs,
    samples: Annotated[int, typer.Option(help="windows drawn per epoch")] = Settings().samples,
    seed: int = 0,
) -> None:
    """Validate the print/hand classifier by held-out collections, then train it on every line."""
    settings = Settings(epochs=epochs, samples=samples)
    lines = read_line_set(lines_path)
    kinds = [bool(lines.printed[lines.collection == k][0]) for k in range(len(lines.collections))]
    fold_of = deal_folds(collection_groups(lines.collections), kinds, folds=folds, seed=seed)
    line_fold = np.array(fold_of)[lines.collection]
    chances = np.zeros(len(lines.images))
    for fold in range(folds):
        held_out = np.flatnonzero(line_fold == fold)
        console.print(f"fold {fold + 1}/{folds}: {len(held_out)} lines held out")
        net = train_net(lines, np.flatnonzero(line_fold != fold), settings, seed=seed + fold)
        chances[held_out] = PrintHandClassifier(net)([[lines.images[i]] for i in held_out])
    report = Report(
        folds=[
            [name for name, f in zip(lines.collections, fold_of, strict=True) if f == fold] for fold in range(folds)
        ],
        collections=scores(lines, chances),
        confusion=[confusion(lines, chances, threshold) for threshold in THRESHOLDS],
        settings=settings,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(report.model_dump_json(indent=1), encoding="utf-8")
    np.savez_compressed(out / "held_out.npz", chances=chances)
    console.print("the net on every line")
    net = train_net(lines, np.arange(len(lines.images)), settings, seed=seed + folds)
    torch.save(net.state_dict(), out / "print_hand.pt")
    console.print(f"-> {out}")


if __name__ == "__main__":
    app()
