"""Extract the EMA weights from an OLD ``accelerate save_state`` checkpoint into a plain safetensors.

``diffu-train`` now writes ``ema_model.safetensors`` into every ``step_N/`` itself, so this script is only
for step checkpoints saved before it did.

Training keeps an EMA shadow (diffusers ``EMAModel``) and *samples from it* (train.py store/copy_to/
restore), so the periodic sample grids — and any honest inference — should use the EMA weights, not the
raw model. ``accelerate`` saves the EMA as ``custom_checkpoint_0.pkl`` (a registered checkpointable)
alongside ``model.safetensors`` (the raw weights). This loads the raw model, overlays the EMA shadow onto
the trainable params (in the exact ``trainable_parameters`` order EMA was built with), and writes a clean
``ema_model.safetensors`` that ``diffu-studio``/``diffu-generate`` can load directly.

    uv run python scripts/extract_ema.py --ckpt-dir runs/<run>/step_<N>
    uv run python scripts/extract_ema.py --ckpt-dir runs/<run>/step_<N> --style-canvas --glyph-pad

The architecture comes from the checkpoint's ``config.json``; a checkpoint saved without one gets the
``Config`` defaults, so pass the arch flags it trained with. They MUST match the training run (they decide
the param set EMA shadows): an EMA that doesn't fit the trainable params is refused.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers.training_utils import EMAModel
from safetensors.torch import save_file

from diffu.config import add_arch_args, apply_arch_args
from diffu.generate import load_checkpoint, load_config
from diffu.model import Diffu
from diffu.train import trainable_parameters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True, help="a step_<N>/ accelerate checkpoint dir")
    ap.add_argument(
        "--out", default=None, help="output safetensors (default: <ckpt-dir>/ema_model.safetensors)"
    )
    add_arch_args(ap)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    ema_pkl = ckpt_dir / "custom_checkpoint_0.pkl"
    raw = ckpt_dir / "model.safetensors"
    if not ema_pkl.exists():
        raise SystemExit(f"no EMA at {ema_pkl} (was the run trained with EMA?)")

    cfg = apply_arch_args(load_config(ckpt_dir), args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Diffu(cfg).to(device).eval()
    load_checkpoint(model, str(raw))  # frozen encoders + raw trainable params

    params = trainable_parameters(model)
    ema = EMAModel(params)
    ema.load_state_dict(torch.load(ema_pkl, map_location=device, weights_only=False))
    if [s.shape for s in ema.shadow_params] != [p.shape for p in params]:
        raise SystemExit(
            f"the EMA shadows {len(ema.shadow_params)} tensors that don't match this config's {len(params)} "
            "trainable params: pass the arch flags the run trained with"
        )
    ema.copy_to(params)  # overlay EMA shadow onto the trainable params in place
    print(f"overlaid EMA (optimization_step={ema.optimization_step}) onto {len(params)} param tensors")

    out = Path(args.out) if args.out else ckpt_dir / "ema_model.safetensors"
    save_file({k: v.contiguous().cpu() for k, v in model.state_dict().items()}, str(out))
    print(f"wrote {out}  ({out.stat().st_size // (1024 * 1024)} MB)")
    config = out.parent / "config.json"
    if not config.exists():  # so the loaders read the architecture instead of needing the flags again
        config.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
        print(f"wrote {config}")


if __name__ == "__main__":
    main()
