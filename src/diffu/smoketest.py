"""Synthetic smoke test: run a few real training steps on random in-memory data.

No prepared dataset required — it fabricates (image, text, style) items matching the dataset contract and
pushes them through the real model + collate + flow loss. Use it to verify the GPU(s), the multi-GPU path
and an architecture (e.g. ``--style-canvas --glyph-pad``) before you have data:

    make smoke-1gpu      # one GPU, single process
    make smoke-2gpu      # two GPUs, accelerate --multi_gpu

First run downloads the VAE (Qwen-Image) and DINOv3 (gated: accept its licence) from the Hub; subsequent
runs are cached.
"""

from __future__ import annotations

import argparse
import random
import tempfile
from functools import partial
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config, add_arch_args, apply_arch_args
from .data.collate import collate_lines
from .data.dataset import STYLE_SIZE
from .train import train

# A few Swedish words (with å ä ö) so the glyph content encoder renders the diacritic shapes.
_WORDS = ("Göteborg", "Smörgåsbord", "Råå", "Ängelholm", "förmiddag", "väder", "Åska", "öster")


class SyntheticLineDataset(Dataset):
    """Random items matching ``HandwritingLineDataset``'s, for a dataset-free smoke run: a DINO style crop,
    or with ``ref_lines`` (canvas mode) a reference line at line height."""

    def __init__(self, n: int, height: int, *, ref_lines: bool = False, seed: int = 0) -> None:
        self.height = height
        self.ref_lines = ref_lines
        rng = random.Random(seed)
        self._widths = [rng.randint(128, 512) for _ in range(n)]  # varied widths exercise bucketing
        self._texts = [rng.choice(_WORDS) for _ in range(n)]

    def __len__(self) -> int:
        return len(self._widths)

    def _line(self, index: int) -> torch.Tensor:
        return torch.rand(3, self.height, self._widths[index]) * 2.0 - 1.0  # [3, H, W] in [-1, 1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = (index + 1) % len(self)  # another line, as the style reference
        item: dict[str, Any] = {
            "image": self._line(index),
            "text": self._texts[index],
            "writer_id": str(index % 4),
            "style_text": self._texts[ref],
        }
        if self.ref_lines:
            item["style_image"] = self._line(ref)
        else:
            item["style_pixel_values"] = torch.rand(3, STYLE_SIZE, STYLE_SIZE)
        return item


def main() -> None:
    ap = argparse.ArgumentParser(description="Diffu synthetic smoke (a few training steps on random data).")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--samples", type=int, default=64, help="synthetic dataset size")
    ap.add_argument(
        "--no-repa", dest="repa", action="store_false", help="disable the REPA aux loss (on by default)"
    )
    ap.add_argument(
        "--compile", action="store_true", help="regional torch.compile the backbone (compile-path smoke)"
    )
    add_arch_args(ap)
    args = ap.parse_args()

    cfg = apply_arch_args(Config(), args)
    cfg.aux.repa = args.repa  # REPA on by default — smoke validates the (non-reentrant) grad path
    canvas = cfg.cond.style_mode == "canvas"
    ds = SyntheticLineDataset(args.samples, cfg.data.line_height, ref_lines=canvas)
    collate = partial(collate_lines, max_width=cfg.data.max_line_width, max_ref_width=cfg.data.max_ref_width)
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
            grad_checkpoint=not args.compile,  # checkpoint recompute + compiled blocks: see train()
            ref_dropout=0.2 if canvas else 0.0,
        )
    mode = "canvas" if canvas else "dino"
    print(
        f"smoke OK — {args.steps} step(s) on {args.batch_size}-line batches ({mode}{' +repa' if args.repa else ''})"
    )


if __name__ == "__main__":
    main()
