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

import torch

from .config import Config
from .generate import ink_crop, load_checkpoint, natural_width, to_pil
from .model import Diffu


def generate_synth_manifest(
    ckpt: str,
    out_dir: str,
    *,
    hf_dataset: str,
    split: str = "train",
    n: int = 4000,
    lora: str | None = None,
    line_height: int = 128,
    cfg_scale: float = 5.0,
    num_steps: int = 24,
    device: str | None = None,
    compile_model: bool = True,
    seed: int = 0,
    texts: list[str] | None = None,
) -> str:
    """Generate ``n`` synthetic lines and write ``out_dir/{lines/*.png, train.jsonl}``. Returns the path.

    ``texts`` (optional): render this explicit list of transcriptions instead of sampling the dataset —
    the hook for TARGETED synthesis (error-driven augmentation), where the caller has already chosen which
    hard content to over-generate. Styles are still sampled from the dataset. ``n`` defaults to ``len(texts)``.
    """
    from datasets import load_dataset
    from PIL import Image
    from torchvision.transforms import functional as TF

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    # Match the released base architecture (glyph-line content, pooled-only style).
    cfg = Config()
    cfg.cond.glyph_line = True
    cfg.cond.style_in_context = False
    cfg.data.line_height = line_height

    model = Diffu(cfg).to(device).eval()
    load_checkpoint(model, ckpt)
    if lora:
        from .lora import load_lora

        load_lora(model, lora)
    if compile_model:
        model.backbone.compile_blocks()

    ds = load_dataset(hf_dataset, split=split)
    text_col = "text" if "text" in ds.column_names else ds.column_names[-1]
    img_col = "image" if "image" in ds.column_names else ds.column_names[0]
    latent_h = line_height // cfg.vae.downscale_factor
    lines_dir = os.path.join(out_dir, "lines")
    os.makedirs(lines_dir, exist_ok=True)
    rng = random.Random(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)
    if texts is not None:  # targeted mode: render the caller's text list (styles still sampled from ds)
        n = len(texts)

    def style_of(i: int) -> torch.Tensor:
        im = ds[i][img_col].convert("RGB").resize((224, 224), Image.BILINEAR)
        t = TF.normalize(TF.to_tensor(im), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return t.unsqueeze(0).to(device)

    path = os.path.join(out_dir, "train.jsonl")
    written = 0
    with open(path, "w", encoding="utf-8") as mf:
        for k in range(n):
            if texts is not None:
                text = texts[k].strip()
            else:
                text = str(ds[order[k % len(order)]][text_col]).strip()
            if not text:
                continue
            style = style_of(rng.randrange(len(ds)))  # a random writer's hand (foreign line)
            # Use the SAME biased+capped width as the deploy path (generate.py:natural_width). The raw
            # guidance_renderer.natural_width is ~2x too wide at 128px -> the model fills by looping, which
            # is what garbled the earlier synth manifests and sank the HTR A/B. See _WIDTH_BIAS.
            latent_w = natural_width(model.guidance_renderer, text, cfg) // cfg.vae.downscale_factor
            with torch.inference_mode():
                out = model.generate([text], style, latent_hw=(latent_h, latent_w), num_steps=num_steps, cfg_scale=cfg_scale)
            png = os.path.join(lines_dir, f"{k:07d}.png")
            ink_crop(to_pil(out[0])).save(png)  # trim the blank tail, matching generate_line
            mf.write(json.dumps({"image": png, "text": text, "writer_id": f"synth_{k}"}, ensure_ascii=False) + "\n")
            written += 1
            if k % 200 == 0:
                print(f"  synth {k}/{n}", flush=True)
    print(f"wrote {written} synthetic lines -> {path}", flush=True)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Bulk-generate a synthetic line manifest from a Diffu checkpoint.")
    ap.add_argument("--ckpt", required=True, help="base model weights (.safetensors)")
    ap.add_argument("--lora", default=None, help="optional LoRA adapter dir/Hub id to load onto the base")
    ap.add_argument("--hf-dataset", required=True, help="source of texts + writer styles (e.g. Teklia/IAM-line)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=4000, help="number of synthetic lines to generate")
    ap.add_argument("--out", required=True, help="output dir (writes lines/*.png + train.jsonl)")
    ap.add_argument("--line-height", type=int, default=128)
    ap.add_argument("--cfg-scale", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0, help="shuffles text/style order; vary to generate a fresh, disjoint batch")
    ap.add_argument("--texts", default=None, help="render this newline-delimited text file (targeted synth) instead of sampling --hf-dataset; --n is ignored")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.set_defaults(compile=True)
    args = ap.parse_args()

    texts = [line.rstrip("\n") for line in open(args.texts, encoding="utf-8")] if args.texts else None
    generate_synth_manifest(
        args.ckpt,
        args.out,
        hf_dataset=args.hf_dataset,
        split=args.split,
        n=args.n,
        lora=args.lora,
        line_height=args.line_height,
        cfg_scale=args.cfg_scale,
        num_steps=args.steps,
        seed=args.seed,
        compile_model=args.compile,
        texts=texts,
    )


if __name__ == "__main__":
    main()
