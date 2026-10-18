"""Quantify legibility with a frozen Swedish HTR recognizer (CER), so 'smudge' becomes a number.

real_CER = recognizer error on the ground-truth scans     -> the achievable floor (sanity check).
gen_CER  = recognizer error on OUR output for the same text -> our quality.
A working generator drives gen_CER toward the real floor; a smudge stays near ~1.0.

    CUDA_VISIBLE_DEVICES=2 uv run --no-sync python scripts/eval_cer.py \
        --ckpt data_out/run3/step_35000/model.safetensors -n 8
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from diffu.config import Config
from diffu.data.dataset import load_manifest
from diffu.generate import _IMAGENET_MEAN, _IMAGENET_STD, generate_line, load_checkpoint
from diffu.model import Diffu
from diffu.recognizer import char_error_rate


def main() -> None:
    ap = argparse.ArgumentParser(description="HTR-recognizer CER on real vs generated lines.")
    ap.add_argument("--ckpt", default="data_out/run3/step_35000/model.safetensors")
    ap.add_argument("--manifest", default="data_out/val.jsonl")
    ap.add_argument("-n", type=int, default=64, help="held-out lines to score (>=64 for a stable mean)")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument(
        "--cfg-scale", type=float, default=0.0, help="classifier-free guidance scale (deploy recipe ~3-4)"
    )
    ap.add_argument("--recognizer", default=None, help="override the HTR recognizer repo id")
    # Arch flags — Config() defaults track the CURRENT training run, not the checkpoint being scored;
    # defaults here match the best released run (exp_sd35_fast: 64px, glyph-line, pooled style).
    ap.add_argument("--line-height", type=int, default=64, help="crop height the checkpoint trained at")
    ap.add_argument("--no-glyph-line", dest="glyph_line", action="store_false")
    ap.add_argument("--style-tokens", dest="style_in_context", action="store_true")
    ap.set_defaults(glyph_line=True, style_in_context=False)
    args = ap.parse_args()

    dev = "cuda"
    cfg = Config()
    cfg.cond.glyph_line = args.glyph_line
    cfg.cond.style_in_context = args.style_in_context
    cfg.data.line_height = args.line_height
    recognizer = args.recognizer or cfg.aux.htr_recognizer_eval
    # Load the recognizer BEFORE Diffu: building Diffu can leave a meta/default-device state that
    # makes TrOCR's pos-embed land on meta (-> "cannot copy out of meta tensor" at generate time).
    proc = TrOCRProcessor.from_pretrained(recognizer)
    # device_map places weights straight on the GPU (no meta->cuda copy that crashes a plain .to()).
    htr = VisionEncoderDecoderModel.from_pretrained(recognizer, device_map=dev).eval()

    model = Diffu(cfg).to(dev).eval()
    load_checkpoint(model, args.ckpt)

    def read(img: Image.Image) -> str:
        pv = proc(images=img.convert("RGB"), return_tensors="pt").pixel_values.to(htr.device)
        with torch.no_grad():
            ids = htr.generate(pv, max_new_tokens=64, interpolate_pos_encoding=True)
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def style_sim(ref_style: torch.Tensor, gen: Image.Image) -> float:
        """Style fidelity: cosine between the style ref and the OUTPUT in the model's DINO style-pooled
        space. A self-consistent proxy (uses the model's own style encoder), not an independent judge."""
        g = (
            TF.normalize(
                TF.to_tensor(gen.convert("RGB").resize((224, 224), Image.BILINEAR)),
                _IMAGENET_MEAN,
                _IMAGENET_STD,
            )
            .unsqueeze(0)
            .to(dev)
        )
        with torch.no_grad():
            _, p_ref = model.style(ref_style)
            _, p_gen = model.style(g)
        return F.cosine_similarity(p_ref, p_gen, dim=-1).item()

    rows = load_manifest(args.manifest)[: args.n]
    real_cers: list[float] = []
    gen_cers: list[float] = []
    style_sims: list[float] = []
    for row in rows:
        gt = row["text"]
        real = Image.open(row["image"]).convert("RGB")
        style = (
            TF.normalize(TF.to_tensor(real.resize((224, 224), Image.BILINEAR)), _IMAGENET_MEAN, _IMAGENET_STD)
            .unsqueeze(0)
            .to(dev)
        )
        # SAME geometry as deploy (auto-width + ink-crop) so gen_CER measures what actually ships.
        gen_img = generate_line(model, gt, style, cfg=cfg, num_steps=args.steps, cfg_scale=args.cfg_scale)
        real_cer = char_error_rate(read(real), gt)
        gen_cer = char_error_rate(read(gen_img), gt)
        sim = style_sim(style, gen_img)
        real_cers.append(real_cer)
        gen_cers.append(gen_cer)
        style_sims.append(sim)
        print(
            f"gt={gt[:34]!r:36}  real_CER={real_cer:.2f}  gen_CER={gen_cer:.2f}  style={sim:.2f}", flush=True
        )

    n = max(len(rows), 1)
    print(f"\nMEAN real_CER  = {sum(real_cers) / n:.3f}   (recognizer floor on true scans)")
    print(f"MEAN gen_CER   = {sum(gen_cers) / n:.3f}   (OUR model; ~1.0 = illegible smudge)")
    print(
        f"MEAN style_sim = {sum(style_sims) / n:.3f}   (output vs ref in DINO style space; 1.0 = identical)"
    )


if __name__ == "__main__":
    main()
