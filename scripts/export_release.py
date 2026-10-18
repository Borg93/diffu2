"""Assemble a Hugging Face release for a trained Diffu checkpoint: ``config.json`` + model card (+ upload).

The release repo holds only ARTIFACTS — ``model.safetensors`` + ``config.json`` + ``README.md``. The
runnable code is the ``diffu`` package (``from diffu.pipeline import DiffuPipeline``), which a user
installs; no loose code file travels with the model.

    uv run python scripts/export_release.py --ckpt checkpoints/exp_sd35_fast/step_280000 --repo you/diffu-swedish-htr
    # add --push to actually upload (needs `hf auth login`); omit it to just print the commands.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from diffu.config import Config


def _weights(ckpt: str) -> Path:
    p = Path(ckpt)
    if p.is_dir():
        for name in ("ema_model.safetensors", "model.safetensors"):
            if (p / name).exists():
                return p / name
        raise FileNotFoundError(f"no *.safetensors in {p}")
    return p


def _infer_qk_norm(weights: Path) -> str | None:
    """``"rms_norm"`` if the checkpoint was trained with SD3.5 qk-norm (has ``...attn.norm_q.*``)."""
    from safetensors import safe_open

    with safe_open(str(weights), framework="pt") as handle:
        return "rms_norm" if any(".norm_q." in k for k in handle.keys()) else None  # noqa: SIM118


def _model_card(repo: str, cfg: Config) -> str:
    return f"""---
license: apache-2.0
language: [sv]
tags: [handwriting-generation, diffusion, historical-documents, rectified-flow]
library_name: diffu
pipeline_tag: image-to-image
---

# Diffu — Swedish historical handwriting line generator

Renders a line of text in a target writer's hand. Give it a string + a single-line crop of the target
handwriting; it generates a matching handwritten line (trained on Swedish archive material). Generated
lines read back at ~0–5% CER through the real Riksarkivet HTR pipeline.

From-scratch diffusion: diffusers SD3 MMDiT backbone + custom 2D-RoPE joint-attention, frozen
Qwen-Image VAE (f8), glyph-line content conditioning (GNU-Unifont), DINOv3 pooled style, rectified flow.

## Usage

```python
from diffu.pipeline import DiffuPipeline          # pip install git+https://github.com/AI-Riksarkivet/diffu

pipe = DiffuPipeline.from_pretrained("{repo}")
img = pipe("Göteborgs poliskammare", style="a_line_crop_of_the_target_hand.png", cfg_scale=5.0)
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
- **DINOv3 access (gated):** trained with `{cfg.cond.style_encoder}` — accept its licence on the Hub
  (free). Don't substitute DINOv2: different style embedding than the backbone was trained on.
- **Qwen-Image VAE** (`{cfg.vae.pretrained}`, Apache-2.0) — pulled automatically.
- CUDA GPU recommended (~6 GB VRAM / line in float32, less in bf16).

## Files
`model.safetensors` (trained weights, single non-strict state_dict) · `config.json` (arch:
line_height={cfg.data.line_height}, glyph_line={cfg.cond.glyph_line}, style_in_context={cfg.cond.style_in_context}, qk_norm={cfg.backbone.qk_norm!r}).
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble + optionally push a Diffu HF release.")
    ap.add_argument("--ckpt", required=True, help="step_* dir or a .safetensors file")
    ap.add_argument("--repo", required=True, help="target HF repo id, e.g. you/diffu-swedish-htr")
    ap.add_argument("--out", default=None, help="local release dir (default: release/<repo-name>)")
    ap.add_argument("--line-height", type=int, default=64, help="crop height the checkpoint trained at")
    ap.add_argument("--no-glyph-line", dest="glyph_line", action="store_false")
    ap.add_argument("--style-tokens", dest="style_in_context", action="store_true")
    ap.add_argument("--glyph-concat", action="store_true", help="checkpoint channel-concats a glyph latent")
    ap.add_argument("--push", action="store_true", help="actually upload (needs `hf auth login`)")
    ap.set_defaults(glyph_line=True, style_in_context=False)
    args = ap.parse_args()

    weights = _weights(args.ckpt)
    # CAUTION: every field NOT set below inherits Config()'s CURRENT defaults, which track the live
    # training run — if a default drifts from what this checkpoint trained with, the exported
    # config.json silently misdescribes the arch. Diff the exported config against the run's trackio
    # config before publishing a new run's checkpoint.
    cfg = Config()
    cfg.cond.glyph_line = args.glyph_line
    cfg.cond.style_in_context = args.style_in_context
    cfg.cond.glyph_concat = args.glyph_concat
    cfg.data.line_height = args.line_height
    cfg.backbone.qk_norm = _infer_qk_norm(weights)  # SD3.5 vs SD3.0, read from the weights
    cfg.cond.use_unifont = cfg.backbone.qk_norm is not None

    out = Path(args.out or f"release/{args.repo.split('/')[-1]}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    (out / "README.md").write_text(_model_card(args.repo, cfg), encoding="utf-8")
    print(
        f"wrote {out}/config.json + {out}/README.md  (weights: {weights}, qk_norm={cfg.backbone.qk_norm!r})"
    )

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
