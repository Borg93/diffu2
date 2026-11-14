"""A random sample of a pool's scans staged as ``diffu-reader-set`` reads a ground truth: per volume an ``alto/``
folder of links to the pool's label files and, beside it, links to their scans."""

from __future__ import annotations

import random
from pathlib import Path

from pydantic import BaseModel

from diffu_curate.alto import read_alto
from diffu_curate.calibration import OLD_PIPELINE
from diffu_curate.settings import Pool, volume_of


class Staged(BaseModel):
    """What a sample staged: the scans linked, of how many the pool offers, and the volumes left out."""

    scans: int
    offered: int
    volumes_left_out: int


def truth_volumes(ground_truth: Path, pool: Pool) -> set[str]:
    """The pool's volumes that hold a ground-truth scan: their hands are in the fine-tuning and test sets, so a
    pretraining sample leaves them out whole."""
    held = {volume.name for volume in pool.volumes()}
    return {volume_of(path.stem) for path in ground_truth.glob("*/*/*/alto/*.xml")} & held


def stage_sample(pool: Pool, out: Path, scans: int, seed: int, without: set[str]) -> Staged:
    """Link ``scans`` scans of ``pool``, drawn uniformly by ``seed`` from the volumes not in ``without``, below
    ``out``. A scan the old pipeline did not label, or without its image, is passed over."""
    files = sorted(path for volume in pool.volumes() if volume.name not in without for path in volume.glob("*.xml"))
    random.Random(seed).shuffle(files)
    staged = 0
    for path in files:
        if staged == scans:
            break
        image = pool.scan_file(path.stem)
        try:
            labelled = read_alto(path).writer == OLD_PIPELINE
        except ValueError:
            labelled = False
        if not labelled or not image.exists():
            continue
        folder = out / path.parent.name
        (folder / "alto").mkdir(parents=True, exist_ok=True)
        (folder / "alto" / path.name).symlink_to(path)
        (folder / image.name).symlink_to(image)
        staged += 1
    return Staged(scans=staged, offered=len(files), volumes_left_out=len(without))
