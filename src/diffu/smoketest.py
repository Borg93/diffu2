"""Synthetic smoke test: run a few real training steps on random in-memory data.

No prepared dataset required — it fabricates (image, text, style) triples matching the dataset
contract and pushes them through the real model + collate + flow loss. Use it to verify the GPU(s)
and the multi-GPU path before you have data:

    make smoke-1gpu      # GPU 0, single process
    make smoke-2gpu      # GPU 0 + 1, accelerate --multi_gpu

First run downloads the VAE (Qwen-Image) and the style encoder (DINOv3, gated: accept its licence) from the
Hub; subsequent runs are cached.
"""

from __future__ import annotations

import argparse
import random
import tempfile
from functools import partial
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .data.collate import collate_lines
from .train import train

# A few Swedish words (with å ä ö) so the glyph content encoder renders the diacritic shapes.
_WORDS = ("Göteborg", "Smörgåsbord", "Råå", "Ängelholm", "förmiddag", "väder", "Åska", "öster")


class SyntheticLineDataset(Dataset):
    """Random triples matching ``HandwritingLineDataset`` items, for a dataset-free smoke run."""

    def __init__(self, n: int, height: int = 64, style_size: int = 224, seed: int = 0) -> None:
        self.height = height
        self.style_size = style_size
        rng = random.Random(seed)
        self._widths = [rng.randint(128, 512) for _ in range(n)]  # varied widths exercise bucketing
        self._texts = [rng.choice(_WORDS) for _ in range(n)]

    def __len__(self) -> int:
        return len(self._widths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = torch.rand(3, self.height, self._widths[index]) * 2.0 - 1.0  # [3, H, W] in [-1, 1]
        style = torch.rand(3, self.style_size, self.style_size)  # [3, S, S]
        return {
            "image": image,
            "text": self._texts[index],
            "writer_id": str(index % 4),
            "style_pixel_values": style,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Diffu synthetic smoke (a few training steps on random data).")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--samples", type=int, default=64, help="synthetic dataset size")
    ap.add_argument(
        "--no-repa", dest="repa", action="store_false", help="disable the REPA aux loss (on by default)"
    )
    ap.set_defaults(repa=True)
    ap.add_argument(
        "--glyph-line", action="store_true", help="exercise the line-glyph + shared-column RoPE content path"
    )
    ap.add_argument(
        "--glyph-concat", action="store_true", help="exercise the channel-concat glyph-latent content path"
    )
    ap.add_argument(
        "--compile", action="store_true", help="regional torch.compile the backbone (compile-path smoke)"
    )
    args = ap.parse_args()

    cfg = Config()
    cfg.aux.repa = args.repa  # REPA on by default — smoke validates the (non-reentrant) grad path
    cfg.cond.glyph_line = args.glyph_line  # opt-in line-glyph content path (experimental)
    cfg.cond.glyph_concat = args.glyph_concat  # opt-in channel-concat content path (experimental)
    ds = SyntheticLineDataset(args.samples, height=cfg.data.line_height)
    collate = partial(collate_lines, max_width=cfg.data.max_line_width)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)

    with tempfile.TemporaryDirectory() as tmp:
        train(
            cfg,
            loader,
            steps=args.steps,
            log_every=1,
            use_ema=False,
            out_dir=tmp,
            save_final=False,
            compile_model=args.compile,
        )
    print(f"smoke OK — {args.steps} step(s) on {args.batch_size}-line batches{' +repa' if args.repa else ''}")


if __name__ == "__main__":
    main()
