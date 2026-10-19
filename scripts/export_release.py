"""Assemble a Hugging Face release for a trained Diffu checkpoint: ``config.json`` + model card (+ upload).

The release repo holds only ARTIFACTS — ``model.safetensors`` + ``config.json`` + ``README.md``. The
runnable code is the ``diffu`` package (``from diffu.pipeline import DiffuPipeline``), which a user
installs; no loose code file travels with the model.

The release ``config.json`` is the checkpoint's own (``diffu-train`` saves one in every checkpoint dir), with
any arch flags applied on top. A checkpoint saved without one gets the ``Config`` defaults — pass the arch
flags it trained with (``--style-canvas``, ``--glyph-pad``, ``--line-height`` ...).

    uv run python scripts/export_release.py --ckpt runs/<run>/step_<N> --repo you/diffu-swedish-htr
    # add --push to actually upload (needs `hf auth login`); omit it to just print the commands.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from diffu.config import Config, add_arch_args, apply_arch_args
from diffu.generate import load_config


def _weights(ckpt: str) -> Path:
    p = Path(ckpt)
    if p.is_dir():
        for name in ("ema_model.safetensors", "model.safetensors"):
            if (p / name).exists():
                return p / name
        raise FileNotFoundError(f"no *.safetensors in {p}")
    return p


def _arch(cfg: Config) -> str:
    """The architecture fields a loader must get right, as one line."""
    return (
        f"style_mode={cfg.cond.style_mode!r}, glyph_pad={cfg.cond.glyph_pad}, "
        f"line_height={cfg.data.line_height}, fill_ratio_cond={cfg.backbone.fill_ratio_cond}, "
        f"max_line_width={cfg.data.max_line_width}, max_chars={cfg.cond.max_chars}, "
        f"qk_norm={cfg.backbone.qk_norm!r}"
    )


def _model_card(repo: str, cfg: Config) -> str:
    canvas = cfg.cond.style_mode == "canvas"
    style = (
        "in-context style (the reference line's latent sits left of the line it writes, on one canvas)"
        if canvas
        else "DINOv3 pooled style"
    )
    style_text = ', style_text="what that line says"' if canvas else ""
    reference = (
        " + the transcription of that crop (so the model can match its letters to strokes)" if canvas else ""
    )
    dino = (
        ""
        if canvas
        else f"- **DINOv3 access (gated):** trained with `{cfg.cond.style_encoder}` — accept its licence on "
        "the Hub\n  (free).\n"
    )
    return f"""---
license: apache-2.0
language: [sv]
tags: [handwriting-generation, diffusion, historical-documents, rectified-flow]
library_name: diffu
pipeline_tag: image-to-image
---

# Diffu — Swedish historical handwriting line generator

Renders a line of text in a target writer's hand. Give it a string + a single-line crop of the target
handwriting{reference}; it generates a matching handwritten line (trained on Swedish archive material).
Generated lines read back at ~0–5% CER through the real Riksarkivet HTR pipeline.

From-scratch diffusion: diffusers SD3 MMDiT backbone + custom 2D-RoPE joint-attention, frozen
Qwen-Image VAE (f8), glyph-line content conditioning (GNU-Unifont), {style}, rectified flow.

## Usage

```python
from diffu.pipeline import DiffuPipeline          # pip install git+https://github.com/Borg93/diffu2

pipe = DiffuPipeline.from_pretrained("{repo}")
img = pipe("Göteborgs poliskammare", style="a_line_crop_of_the_target_hand.png"{style_text}, cfg_scale=5.0)
img.save("out.png")
```

## Performance

```python
# one-off lines: default (eager) is fine — ~1 s/line on a modern GPU
pipe = DiffuPipeline.from_pretrained("{repo}")

# many lines / serving: compile the backbone once (regional torch.compile, dynamic —
# one compile covers variable line widths). One-time cold start, then ~0.4 s/line.
pipe = DiffuPipeline.from_pretrained("{repo}", compile_backbone=True)

# bf16: ~half the VRAM, same legibility
import torch
pipe = DiffuPipeline.from_pretrained("{repo}", dtype=torch.bfloat16, compile_backbone=True)
```

**Don't quantize by default.** Measured on this model (torchao int8/fp8, CER-gated): legibility is
unaffected but inference gets *slower* — the GEMMs are small and not memory-bound, so weight-quant
overhead dominates. bf16 + compile is the recommended serving config.

## Requirements

- The **`diffu` package** (the modeling code this pipeline wraps).
{dino}- **Qwen-Image VAE** (`{cfg.vae.pretrained}`, Apache-2.0) — pulled automatically.
- CUDA GPU recommended (~6 GB VRAM / line in float32, less in bf16).

## Files
`model.safetensors` (trained weights, single non-strict state_dict) · `config.json` (arch: {_arch(cfg)}).
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble + optionally push a Diffu HF release.")
    ap.add_argument("--ckpt", required=True, help="a checkpoint dir (step_*/final) or a .safetensors file")
    ap.add_argument("--repo", required=True, help="target HF repo id, e.g. you/diffu-swedish-htr")
    ap.add_argument("--out", default=None, help="local release dir (default: release/<repo-name>)")
    ap.add_argument("--push", action="store_true", help="actually upload (needs `hf auth login`)")
    add_arch_args(ap)
    args = ap.parse_args()

    weights = _weights(args.ckpt)
    if not (weights.parent / "config.json").is_file():
        # Every field not given as an arch flag then comes from Config()'s CURRENT defaults — if one differs
        # from what this checkpoint trained with, the exported config.json silently misdescribes the arch.
        print(f"WARNING: no config.json next to {weights}: using the Config defaults + the arch flags given")
    cfg = apply_arch_args(load_config(weights), args)

    out = Path(args.out or f"release/{args.repo.split('/')[-1]}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    (out / "README.md").write_text(_model_card(args.repo, cfg), encoding="utf-8")
    print(f"wrote {out}/config.json + {out}/README.md  (weights: {weights}; {_arch(cfg)})")

    uploads = [
        (str(out / "config.json"), "config.json"),
        (str(out / "README.md"), "README.md"),
        (str(weights), "model.safetensors"),
    ]
    if args.push:
        subprocess.run(["hf", "repo", "create", args.repo, "--repo-type", "model"], check=False)
        for local, remote in uploads:
            subprocess.run(["hf", "upload", args.repo, local, remote], check=True)
        print(f"pushed to https://huggingface.co/{args.repo}")
    else:
        print(f"\n# to publish (after `hf auth login`):\nhf repo create {args.repo} --repo-type model")
        for local, remote in uploads:
            print(f"hf upload {args.repo} {local} {remote}")


if __name__ == "__main__":
    main()
