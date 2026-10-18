"""Extract the EMA weights from an ``accelerate save_state`` checkpoint into a plain safetensors.

Training keeps an EMA shadow (diffusers ``EMAModel``) and *samples from it* (train.py store/copy_to/
restore), so the periodic sample grids — and any honest inference — should use the EMA weights, not the
raw model. ``accelerate`` saves the EMA as ``custom_checkpoint_0.pkl`` (a registered checkpointable)
alongside ``model.safetensors`` (the raw weights). This loads the raw model, overlays the EMA shadow onto
the trainable params (in the exact ``trainable_parameters`` order EMA was built with), and writes a clean
``ema_model.safetensors`` that ``diffu-studio``/``diffu-generate`` can load directly.

    uv run python scripts/extract_ema.py \
        --ckpt-dir checkpoints/exp_base_weekend/step_360000 --no-style-tokens --glyph-line

The architecture flags MUST match the training run (they decide the param set EMA shadows).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers.training_utils import EMAModel
from safetensors.torch import save_file

from diffu.config import Config
from diffu.generate import load_checkpoint
from diffu.model import Diffu
from diffu.train import trainable_parameters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True, help="a step_<N>/ (or final/) accelerate checkpoint dir")
    ap.add_argument("--out", default=None, help="output safetensors (default: <ckpt-dir>/ema_model.safetensors)")
    ap.add_argument("--no-style-tokens", dest="style_in_context", action="store_false")
    ap.add_argument("--glyph-line", action="store_true")
    ap.add_argument("--glyph-concat", action="store_true")
    ap.set_defaults(style_in_context=True)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    ema_pkl = ckpt_dir / "custom_checkpoint_0.pkl"
    raw = ckpt_dir / "model.safetensors"
    if not ema_pkl.exists():
        raise SystemExit(f"no EMA at {ema_pkl} (was the run trained with EMA?)")

    cfg = Config()
    cfg.cond.style_in_context = args.style_in_context
    cfg.cond.glyph_line = args.glyph_line
    cfg.cond.glyph_concat = args.glyph_concat

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Diffu(cfg).to(device).eval()
    load_checkpoint(model, str(raw))  # frozen encoders + raw trainable params

    params = trainable_parameters(model)
    ema = EMAModel(params)
    ema.load_state_dict(torch.load(ema_pkl, map_location=device, weights_only=False))
    ema.copy_to(params)  # overlay EMA shadow onto the trainable params in place
    print(f"overlaid EMA (optimization_step={ema.optimization_step}) onto {len(params)} param tensors")

    out = Path(args.out) if args.out else ckpt_dir / "ema_model.safetensors"
    save_file({k: v.contiguous().cpu() for k, v in model.state_dict().items()}, str(out))
    print(f"wrote {out}  ({out.stat().st_size // (1024 * 1024)} MB)")


if __name__ == "__main__":
    main()
