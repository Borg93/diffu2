"""Bulk-generate a SYNTHETIC line manifest from a trained Diffu checkpoint (+ optional LoRA).

The generation half of the augmentation A/B: draw texts + writer styles from a source dataset (e.g.
Teklia/IAM-line), generate a handwritten line per text, and write ``lines/*.png`` + ``train.jsonl``.
Then train a TrOCR recognizer on REAL vs REAL+SYNTH (``diffu-htr-train``) and compare CER on held-out
real test (``diffu-eval``) — the answer to "does the synthetic data work as HTR training data?".

    diffu-synth --ckpt <base.safetensors> --lora runs/iam_lora/step_1000 \
        --hf-dataset Teklia/IAM-line --split train --n 4000 --out runs/synth_lora
"""

from __future__ import annotations

import argparse
import json
import os
import random

from .config import Config, add_arch_args, apply_arch_args
from .data.hf_dataset import _IMAGE_COLS, _TEXT_COLS, _load, _pick
from .generate import build_model, generate_line, load_config, load_ref_line, load_style


def generate_synth_manifest(
    cfg: Config,
    ckpt: str,
    out_dir: str,
    *,
    hf_dataset: str,
    split: str = "train",
    n: int = 4000,
    lora: str | None = None,
    cfg_scale: float = 5.0,
    num_steps: int | None = None,
    compile_model: bool = True,
    seed: int = 0,
    texts: list[str] | None = None,
) -> str:
    """Generate ``n`` synthetic lines and write ``out_dir/{lines/*.png, train.jsonl}``. Returns the path.

    ``texts`` (optional): render this explicit list of transcriptions instead of sampling the dataset —
    the hook for TARGETED synthesis (error-driven augmentation), where the caller has already chosen which
    hard content to over-generate; ``n`` is then ``len(texts)``. Styles are still sampled from the dataset.
    Each line is rendered at the deploy geometry (``generate_line``: auto width + ink-crop).
    """
    model = build_model(cfg, ckpt, lora=lora, compile_backbone=compile_model)
    device = next(model.parameters()).device
    ds = _load(hf_dataset, split, None)
    cols = list(ds.column_names)
    text_col, img_col = _pick(cols, _TEXT_COLS), _pick(cols, _IMAGE_COLS)
    lines_dir = os.path.join(out_dir, "lines")
    os.makedirs(lines_dir, exist_ok=True)
    rng = random.Random(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)
    if texts is not None:  # targeted mode: render the caller's text list (styles still sampled from ds)
        n = len(texts)

    path = os.path.join(out_dir, "train.jsonl")
    written = 0
    with open(path, "w", encoding="utf-8") as mf:
        for k in range(n):
            text = texts[k].strip() if texts is not None else str(ds[order[k % len(order)]][text_col]).strip()
            if not text:
                continue
            row = ds[rng.randrange(len(ds))]  # a random writer's hand (foreign line)
            if model.canvas:
                ref = load_ref_line(row[img_col], model.cfg, text=str(row[text_col]))
                img = generate_line(model, text, None, num_steps=num_steps, cfg_scale=cfg_scale, ref=ref)
            else:
                style = load_style(row[img_col]).unsqueeze(0).to(device)
                img = generate_line(model, text, style, num_steps=num_steps, cfg_scale=cfg_scale)
            png = os.path.join(lines_dir, f"{k:07d}.png")
            img.save(png)
            mf.write(
                json.dumps({"image": png, "text": text, "writer_id": f"synth_{k}"}, ensure_ascii=False) + "\n"
            )
            written += 1
            if k % 200 == 0:
                print(f"  synth {k}/{n}", flush=True)
    print(f"wrote {written} synthetic lines -> {path}", flush=True)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bulk-generate a synthetic line manifest from a Diffu checkpoint."
    )
    ap.add_argument("--ckpt", required=True, help="base model weights (.safetensors)")
    ap.add_argument("--lora", default=None, help="optional LoRA adapter dir/Hub id to load onto the base")
    ap.add_argument(
        "--hf-dataset", required=True, help="source of texts + writer styles (e.g. Teklia/IAM-line)"
    )
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=4000, help="number of synthetic lines to generate")
    ap.add_argument("--out", required=True, help="output dir (writes lines/*.png + train.jsonl)")
    ap.add_argument("--cfg-scale", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=None, help="denoising steps (default: the config's 24)")
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="shuffles text/style order; vary to generate a fresh, disjoint batch",
    )
    ap.add_argument(
        "--texts",
        default=None,
        help="render this newline-delimited text file (targeted synth) instead of sampling --hf-dataset; "
        "--n is ignored",
    )
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    add_arch_args(ap)
    args = ap.parse_args()

    texts = None
    if args.texts:
        with open(args.texts, encoding="utf-8") as f:
            texts = [line.rstrip("\n") for line in f]
    generate_synth_manifest(
        apply_arch_args(load_config(args.ckpt), args),
        args.ckpt,
        args.out,
        hf_dataset=args.hf_dataset,
        split=args.split,
        n=args.n,
        lora=args.lora,
        cfg_scale=args.cfg_scale,
        num_steps=args.steps,
        seed=args.seed,
        compile_model=args.compile,
        texts=texts,
    )


if __name__ == "__main__":
    main()
