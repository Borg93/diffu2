"""Launch the gated PAGE pipeline. Each flag falls back to its environment variable: CKPT, TRAIN_JSONL,
PAPER_DIR (required), OUT, TARGET, GPUS, BATCH, SEED. For bank mode, set ``bank=`` in the ``PageConfig`` below.

    CKPT=runs/<run>/final/model.safetensors TRAIN_JSONL=data_out/train.jsonl PAPER_DIR=<paper scans> \
        OUT=synth_pages/pages TARGET=10000 GPUS=1 BATCH=24 SEED=1 uv run python scripts/launch_pages.py

Runs from /tmp so Ray's captured working_dir stays small (the repo root would ship checkpoints/data
to every worker despite .rayignore pruning most of it), so relative paths are resolved before that.
"""

import argparse
import os
from pathlib import Path

from diffu_runner.pages import PageConfig, run_pages


def main() -> None:
    env = os.environ.get
    ap = argparse.ArgumentParser(
        description="Launch the gated PAGE pipeline (flags default to their env vars)."
    )
    ap.add_argument("--ckpt", default=env("CKPT"), help="model weights (.safetensors) [$CKPT]")
    ap.add_argument(
        "--train-jsonl", default=env("TRAIN_JSONL"), help="the TRAIN split manifest [$TRAIN_JSONL]"
    )
    ap.add_argument("--paper-dir", default=env("PAPER_DIR"), help="dir of real paper scans [$PAPER_DIR]")
    ap.add_argument("--out", default=env("OUT", "synth_pages/pages"), help="output dir [$OUT]")
    ap.add_argument("--target", type=int, default=int(env("TARGET", "10000")), help="pages to keep [$TARGET]")
    ap.add_argument(
        "--gpus", type=int, default=int(env("GPUS", "1")), help="render workers, 1 GPU each [$GPUS]"
    )
    ap.add_argument(
        "--batch", type=int, default=int(env("BATCH", "24")), help="jobs per worker per round [$BATCH]"
    )
    ap.add_argument("--seed", type=int, default=int(env("SEED", "1")), help="[$SEED]")
    args = ap.parse_args()
    required = (
        (args.ckpt, "--ckpt", "CKPT"),
        (args.train_jsonl, "--train-jsonl", "TRAIN_JSONL"),
        (args.paper_dir, "--paper-dir", "PAPER_DIR"),
    )
    for value, flag, var in required:
        if not value:
            ap.error(f"{flag} (or ${var}) is required")
    ckpt, train_jsonl, paper_dir, out = (
        str(Path(p).resolve()) for p in (args.ckpt, args.train_jsonl, args.paper_dir, args.out)
    )

    os.makedirs("/tmp/ray_wd_pages", exist_ok=True)
    os.chdir("/tmp/ray_wd_pages")

    cfg = PageConfig(ckpt=ckpt, train_jsonl=train_jsonl, out=out, paper_dir=paper_dir)
    _, stats = run_pages(cfg, target=args.target, gpus=args.gpus, batch=args.batch, seed=args.seed)
    print("PAGES DONE:", stats)


if __name__ == "__main__":
    main()
