"""Quantify legibility with a frozen Swedish HTR recognizer (CER), so 'smudge' becomes a number.

real_CER = recognizer error on the ground-truth scans     -> the achievable floor (sanity check).
gen_CER  = recognizer error on OUR output for the same text -> our quality.
A working generator drives gen_CER toward the real floor; a smudge stays near ~1.0.

The architecture comes from the checkpoint's ``config.json`` (arch flags override it — needed only for a
checkpoint saved without one). A dino-mode model takes each real line itself as its style reference. A
``--style-canvas`` model takes ANOTHER line of the same ``writer_id`` (the next one in the manifest) as its
in-canvas reference, with that line's transcription, as in training — the target itself would show the
model the answer — so rows whose writer has no other line in the manifest are skipped. style_sim needs the
DINO style encoder, so it is dino-mode only.

    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/eval_cer.py \
        --ckpt runs/<run>/step_<N>/ema_model.safetensors -n 8
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from diffu.config import add_arch_args, apply_arch_args
from diffu.data.dataset import load_manifest, style_tensor
from diffu.generate import build_model, generate_line, load_config, load_ref_line, load_style
from diffu.recognizer import char_error_rate


def _canvas_refs(rows: list[dict[str, str]]) -> list[int | None]:
    """Per row, the index of another row by the same writer (the next one, wrapping around) — a canvas
    model's reference line — or None when the writer has no other line (or the row no ``writer_id``)."""
    by_writer: defaultdict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if row.get("writer_id") is not None:
            by_writer[str(row["writer_id"])].append(i)
    refs: list[int | None] = [None] * len(rows)
    for peers in by_writer.values():
        if len(peers) > 1:
            for k, i in enumerate(peers):
                refs[i] = peers[(k + 1) % len(peers)]
    return refs


def main() -> None:
    ap = argparse.ArgumentParser(description="HTR-recognizer CER on real vs generated lines.")
    ap.add_argument("--ckpt", required=True, help="model weights (.safetensors)")
    ap.add_argument("--manifest", default="data_out/val.jsonl")
    ap.add_argument("-n", type=int, default=64, help="held-out lines to score (>=64 for a stable mean)")
    ap.add_argument("--steps", type=int, default=None, help="denoising steps (default: the config's 24)")
    ap.add_argument(
        "--cfg-scale", type=float, default=0.0, help="classifier-free guidance scale (deploy recipe ~3-4)"
    )
    ap.add_argument("--recognizer", default=None, help="override the HTR recognizer repo id")
    add_arch_args(ap)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = apply_arch_args(load_config(args.ckpt), args)
    recognizer = args.recognizer or cfg.aux.htr_recognizer_eval
    # Load the recognizer BEFORE Diffu: building Diffu can leave a meta/default-device state that
    # makes TrOCR's pos-embed land on meta (-> "cannot copy out of meta tensor" at generate time).
    proc = TrOCRProcessor.from_pretrained(recognizer)
    # device_map places weights straight on the GPU (no meta->cuda copy that crashes a plain .to()).
    htr = VisionEncoderDecoderModel.from_pretrained(recognizer, device_map=dev).eval()

    model = build_model(cfg, args.ckpt, device=dev)

    def read(img: Image.Image) -> str:
        pv = proc(images=img.convert("RGB"), return_tensors="pt").pixel_values.to(htr.device)
        with torch.no_grad():
            ids = htr.generate(pv, max_new_tokens=64, interpolate_pos_encoding=True)
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def style_sim(ref_style: torch.Tensor, gen: Image.Image) -> float:
        """Style fidelity: cosine between the style ref and the OUTPUT in the model's DINO style-pooled
        space. A self-consistent proxy (uses the model's own style encoder), not an independent judge."""
        g = style_tensor(gen).unsqueeze(0).to(dev)
        with torch.no_grad():
            p_ref = model.style(ref_style)
            p_gen = model.style(g)
        return F.cosine_similarity(p_ref, p_gen, dim=-1).item()

    all_rows = load_manifest(args.manifest)
    rows = all_rows[: args.n]
    refs = _canvas_refs(all_rows) if model.canvas else []
    real_cers: list[float] = []
    gen_cers: list[float] = []
    style_sims: list[float] = []
    skipped = 0
    for i, row in enumerate(rows):
        peer = refs[i] if model.canvas else None
        if model.canvas and peer is None:
            skipped += 1
            continue
        gt = row["text"]
        real = Image.open(row["image"]).convert("RGB")
        sim: float | None = None
        # SAME geometry as deploy (auto-width + ink-crop) so gen_CER measures what actually ships.
        if peer is not None:  # canvas: another line of this writer, with its transcription
            ref = load_ref_line(all_rows[peer]["image"], model.cfg, text=all_rows[peer]["text"])
            gen_img = generate_line(model, gt, None, num_steps=args.steps, cfg_scale=args.cfg_scale, ref=ref)
        else:
            style = load_style(real).unsqueeze(0).to(dev)
            gen_img = generate_line(model, gt, style, num_steps=args.steps, cfg_scale=args.cfg_scale)
            sim = style_sim(style, gen_img)
            style_sims.append(sim)
        real_cer = char_error_rate(read(real), gt)
        gen_cer = char_error_rate(read(gen_img), gt)
        real_cers.append(real_cer)
        gen_cers.append(gen_cer)
        style_col = f"  style={sim:.2f}" if sim is not None else ""
        print(f"gt={gt[:34]!r:36}  real_CER={real_cer:.2f}  gen_CER={gen_cer:.2f}{style_col}", flush=True)

    if skipped:
        print(
            f"\nskipped {skipped} line(s): their writer has no other line in the manifest to use as the reference"
        )
    n = max(len(gen_cers), 1)
    print(f"\nMEAN real_CER  = {sum(real_cers) / n:.3f}   (recognizer floor on true scans)")
    print(f"MEAN gen_CER   = {sum(gen_cers) / n:.3f}   (OUR model; ~1.0 = illegible smudge)")
    if style_sims:
        print(
            f"MEAN style_sim = {sum(style_sims) / n:.3f}   (output vs ref in DINO style space; 1.0 = identical)"
        )


if __name__ == "__main__":
    main()
