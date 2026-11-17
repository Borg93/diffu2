"""DINOv3 and TIPSv2 whole-scan vectors for every pool scan with labelled lines (CURATION E40).

Made as embed_pages.py made them for the 45,000 first described scans, so a head fitted on those reads these: the
scan fitted to 448 px on its longer side (both sides a multiple of 112), and of each model the class token and the
mean patch token, each normalised, 2,048 numbers a model. The scans are taken in an order shuffled once, so those
embedded at any moment are a random sample of the pools; scans of one shape go through the models together. Parquet
part files (image, dinov3, tipsv2) in OUT, 20,000 scans a part; a scan in a part is not embedded again.

    CUDA_VISIBLE_DEVICES=1 uv run python runs/htr/curate/prototypes/embed_pool.py OUT SHARD SHARDS [LIMIT]

IMAGES=<file of scan paths, one a line> embeds those instead of the pools; DEVICE=cpu runs without a GPU (a check).
"""
import os
import random
import sys
import time
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

Image.MAX_IMAGE_PIXELS = None
LONG, BATCH, PART, WORKERS = 448, 32, 20000, int(os.environ.get("WORKERS", "20"))
DEVICE = os.environ.get("DEVICE", "cuda"); HALF = DEVICE == "cuda"
ROWS = pa.schema([("image", pa.utf8()), ("dinov3", pa.list_(pa.float16(), 2048)), ("tipsv2", pa.list_(pa.float16(), 2048))])


class Scans(Dataset):
    def __init__(self, images: list[str]) -> None: self.images = images
    def __len__(self) -> int: return len(self.images)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor | None]:
        try:
            with Image.open(self.images[index]) as scan:
                scan.draft("RGB", (2400, 2400)); scan = scan.convert("RGB")
            scale = LONG / max(scan.size)
            size = (max(28, round(scan.width * scale / 112) * 112), max(28, round(scan.height * scale / 112) * 112))
            return index, torch.from_numpy(np.asarray(scan.resize(size, Image.Resampling.BICUBIC))).permute(2, 0, 1)
        except (OSError, ValueError):  # an unreadable or truncated scan; anything else is a bug and stops the run
            return index, None


def vectors(dino, tips, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """The two models' vectors of a batch of scans of one shape (uint8, B x 3 x H x W)."""
    x = x.to(DEVICE).float() / 255; patches = (x.shape[2] // 16) * (x.shape[3] // 16)
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    seen = dino(pixel_values=((x - mean) / std).to(dino.dtype)).last_hidden_state.float()
    d = torch.cat([F.normalize(seen[:, 0], dim=1), F.normalize(seen[:, -patches:].mean(1), dim=1)], dim=1)
    with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=HALF):
        token, _, patch = tips(x)
    t = torch.cat([F.normalize(token[:, 0].float(), dim=1), F.normalize(patch.float().mean(1), dim=1)], dim=1)
    return d.cpu().numpy().astype(np.float16), t.cpu().numpy().astype(np.float16)


def write(out: Path, shard: int, images: list[str], d: list[np.ndarray], t: list[np.ndarray]) -> None:
    def column(rows: list[np.ndarray]) -> pa.FixedSizeListArray:
        return pa.FixedSizeListArray.from_arrays(pa.array(np.concatenate(rows).ravel(), pa.float16()), 2048)
    pq.write_table(pa.table({"image": images, "dinov3": column(d), "tipsv2": column(t)}, schema=ROWS), out / f"part-{shard}-{time.time_ns()}.parquet")


if __name__ == "__main__":
    out, shard, shards = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    if os.environ.get("IMAGES"):
        images = [line.strip() for line in open(os.environ["IMAGES"], encoding="utf-8") if line.strip()]
    else:
        images = lance.dataset("runs/htr/curate/scans.lance").to_table(columns=["image"], filter="pool != 'gt' AND lines > 0").column("image").to_pylist()
        random.Random(0).shuffle(images)
    done = {image for part in out.glob("part-*.parquet") for image in pq.read_table(part, columns=["image"]).column("image").to_pylist()}
    todo = [image for image in images[shard::shards] if image not in done][:limit]
    print(len(images), "scans;", len(done), "embedded before;", len(todo), "to embed in this shard", flush=True)
    dino = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m", dtype=torch.bfloat16 if HALF else torch.float32).to(DEVICE).eval()
    tips = AutoModel.from_pretrained("google/tipsv2-l14", trust_remote_code=True).vision_encoder.to(DEVICE).eval()
    loader = DataLoader(Scans(todo), batch_size=1, num_workers=WORKERS, collate_fn=lambda batch: batch[0], prefetch_factor=8)
    waiting: dict[tuple[int, int], list[tuple[int, torch.Tensor]]] = {}
    names: list[str] = []; d_rows: list[np.ndarray] = []; t_rows: list[np.ndarray] = []; lost = 0; start = time.time()

    def run(batch: list[tuple[int, torch.Tensor]]) -> None:
        d, t = vectors(dino, tips, torch.stack([x for _, x in batch]))
        names.extend(todo[i] for i, _ in batch); d_rows.append(d); t_rows.append(t)

    with torch.inference_mode():
        for n, (i, x) in enumerate(loader):
            if x is None:
                lost += 1
            else:
                held = waiting.setdefault((x.shape[1], x.shape[2]), []); held.append((i, x))
                if len(held) == BATCH:
                    run(held); held.clear()
            if len(names) >= PART:
                write(out, shard, names, d_rows, t_rows); names, d_rows, t_rows = [], [], []
            if n % 2000 == 1999:
                print(f"{n + 1} scans, {(n + 1) / (time.time() - start):.1f} a second, {lost} unreadable", flush=True)
        for held in waiting.values():
            if held:
                run(held)
    if names:
        write(out, shard, names, d_rows, t_rows)
    print(f"done {len(todo) - lost} of {len(todo)} in {time.time() - start:.0f} s, {lost} unreadable", flush=True)
