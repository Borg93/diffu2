"""Diffu: an MMDiT + rectified flow handwriting generator (16-ch VAE + glyph
conditioning + DINOv3 style) with classic CFG guidance, for Swedish HTR augmentation.

This package root is intentionally empty of re-exports — no import side effects, and one obvious
home per symbol. Import from the submodule that owns it:

    from diffu.config import Config          # lightweight: pydantic only, no torch
    from diffu.model import Diffu           # the full model (pulls in torch / diffusers)
    from diffu.data.dataset import HandwritingLineDataset

Keeping it bare also means ``python -m diffu.data.prepare`` (data prep) never imports torch.
"""
