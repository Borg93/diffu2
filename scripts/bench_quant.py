"""Benchmark torchao backbone quantization: inference latency AND legibility (CER) vs bf16.

Builds the model once per variant (quantization is in-place/destructive), compiles it (torchao's
speedup comes from compiled low-bit kernels — eager int8 dequantizes and is often slower), warms up
EVERY text once (each distinct auto-width pays its compile outside the timed region), then times
``generate_line`` and reads every image back with the frozen TrOCR recognizer — so the report is
speed AND accuracy, not speed and an eyeball. INFERENCE-ONLY — never touches the training run.

    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 uv run --with torchao python scripts/bench_quant.py \\
        --ckpt checkpoints/exp_sd35_fast/step_280000 --style data_out_lejonet/lines/003424_000.png \\
        --manifest data_out_lejonet/val.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL.Image import Image

from diffu.config import Config
from diffu.generate import generate_line, load_checkpoint, load_style
from diffu.model import Diffu
from diffu.quantize import QuantMode, quantize_backbone
from diffu.recognizer import char_error_rate, load_recognizer, read_lines

FALLBACK_TEXTS = ["Stockholm den 12 mars", "ett litet prov", "kungliga biblioteket"]


def manifest_texts(path: str, n: int) -> list[str]:
    """First ``n`` unique val texts of sane length — in-domain prompts give a meaningful CER."""
    texts: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            text = json.loads(line)["text"].strip()
            if 10 <= len(text) <= 40 and text not in texts:
                texts.append(text)
            if len(texts) == n:
                break
    return texts or FALLBACK_TEXTS


def resolve_weights(ckpt: str) -> str:
    """Accept a ``step_*`` DIR or a weights file (EMA preferred for a dir) — matches the studio loader."""
    p = Path(ckpt)
    if p.is_dir():
        for name in ("ema_model.safetensors", "model.safetensors"):
            if (p / name).exists():
                return str(p / name)
        raise FileNotFoundError(f"no *.safetensors in {p}")
    return ckpt


def build(cfg: Config, ckpt: str, device: str, quant: QuantMode | None) -> Diffu:
    # bf16 end-to-end — the dtype torchao's low-bit kernels are tuned for (fp32 activations put the
    # quantized matmuls on a slow path and unfairly flatter the baseline); matches the studio's bf16 mode.
    model = Diffu(cfg).to(device, dtype=torch.bfloat16).eval()
    load_checkpoint(model, resolve_weights(ckpt))
    if quant is not None:
        quantize_backbone(model, quant)
    model.backbone.compile_blocks()  # torchao speedup needs compiled low-bit kernels
    return model


def bench(
    model: Diffu, style: torch.Tensor, texts: list[str], cfg: Config, cfg_scale: float
) -> tuple[float, list[Image]]:
    steps = cfg.flow.sample_steps
    for t in texts:  # warm every width so compile never lands in the timed region
        generate_line(model, t, style, cfg=cfg, num_steps=steps, cfg_scale=cfg_scale)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    imgs = [generate_line(model, t, style, cfg=cfg, num_steps=steps, cfg_scale=cfg_scale) for t in texts]
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / len(texts), imgs


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark torchao backbone quantization for inference.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--manifest", default=None, help="val.jsonl to draw in-domain texts from")
    ap.add_argument("--n-texts", type=int, default=6)
    ap.add_argument("--cfg-scale", type=float, default=5.0)
    ap.add_argument("--line-height", type=int, default=64, help="crop height the checkpoint trained at")
    ap.add_argument("--no-glyph-line", dest="glyph_line", action="store_false")
    ap.add_argument("--style-tokens", dest="style_in_context", action="store_true")
    ap.add_argument("--out", default="quant_bench")
    ap.set_defaults(glyph_line=True, style_in_context=False)
    args = ap.parse_args()

    device = "cuda"
    # One set of globals for ALL variants (torchao's configs no longer flip TF32/inductor flags at
    # quantize time — src/diffu/quantize.py pins set_inductor_config=False), so speedups measure
    # quantization, not global-toggle asymmetry between baseline and quant variants.
    torch.set_float32_matmul_precision("high")
    cfg = Config()  # then match the checkpoint's arch — Config() defaults track the CURRENT run, not this one
    cfg.cond.glyph_line = args.glyph_line
    cfg.cond.style_in_context = args.style_in_context
    cfg.data.line_height = args.line_height

    texts = manifest_texts(args.manifest, args.n_texts) if args.manifest else FALLBACK_TEXTS
    style = load_style(args.style).unsqueeze(0).to(device, dtype=torch.bfloat16)  # match the model dtype
    proc, recog = load_recognizer(cfg.aux.htr_recognizer_eval, device)
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    variants: list[tuple[str, QuantMode | None]] = [
        ("bf16", None),
        ("int8", "int8"),
        ("fp8dq", "fp8dq"),
    ]
    latency: dict[str, float] = {}
    cers: dict[str, float] = {}
    for name, quant in variants:
        try:
            torch.manual_seed(
                cfg.seed
            )  # identical noise per variant — Δ CER measures QUANTIZATION, not seeds
            model = build(cfg, args.ckpt, device, quant)
            lat, imgs = bench(model, style, texts, cfg, args.cfg_scale)
            reads = read_lines(proc, recog, imgs)
            cers[name] = sum(char_error_rate(p, t) for p, t in zip(reads, texts, strict=True)) / len(texts)
            latency[name] = lat  # record LAST — a late failure must not leave a latency-only entry (KeyError)
            for i, im in enumerate(imgs):
                im.save(out / f"{name}_{i}.png")
            print(f"{name:6s}: {lat * 1000:7.1f} ms/line   CER {cers[name]:.4f}", flush=True)
            del model
            torch.cuda.empty_cache()
        except Exception as e:  # one variant failing shouldn't sink the rest
            print(f"{name:6s}: FAILED — {type(e).__name__}: {str(e)[:120]}", flush=True)

    base_lat, base_cer = latency.get("bf16"), cers.get("bf16")
    print(f"\n=== latency + CER vs bf16 ({len(texts)} lines, cfg={args.cfg_scale}) ===")
    for name in latency:
        sp = f"{base_lat / latency[name]:.2f}x" if base_lat else "—"
        dc = f"{cers[name] - base_cer:+.4f}" if base_cer is not None else "—"
        print(f"  {name:6s} {latency[name] * 1000:7.1f} ms/line  {sp:>6s}   CER {cers[name]:.4f} (Δ {dc})")
    print(f"\nimages in {out}/ — CER is the shipping gate; eyeball only to confirm")


if __name__ == "__main__":
    main()
