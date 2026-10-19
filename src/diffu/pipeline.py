"""``DiffuPipeline`` — a one-call inference wrapper so a trained Diffu model is easy to load and run.

Diffu is a bespoke diffusion model (diffusers SD3 MMDiT backbone + a custom 2D-RoPE joint-attention
processor + a frozen Qwen-Image VAE + glyph-line content conditioning + a writer reference), loaded from a
single non-strict ``state_dict``. It is NOT decomposed into standard diffusers ``ModelMixin`` components,
so this pipeline wraps the whole :class:`~diffu.model.Diffu`, loaded by the same ``build_model`` as the CLI
and the studio, and renders through the same ``generate_line``.

    from diffu.pipeline import DiffuPipeline
    pipe = DiffuPipeline.from_pretrained("runs/my_run/final")   # a local dir or a Hub repo id
    img = pipe("Göteborgs poliskammare", style="a_line_crop_of_the_target_hand.png")
    img.save("out.png")

The release repo/dir holds ``config.json`` + ``model.safetensors`` (what ``diffu-train`` saves, or
``scripts/export_release.py``). ``style`` is any single-line crop of the target writer's hand (PIL image or
path). A dino-mode model reads it through the DINOv3 style encoder (gated — accept the licence on the Hub);
a canvas-mode model places it left of the line it writes (pass its transcription as ``style_text``).
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .config import Config
from .generate import build_model, generate_line, load_ref_line, load_style
from .model import Diffu


class DiffuPipeline:
    """A loaded Diffu model + its config, callable as ``pipe(text, style) -> PIL.Image``."""

    def __init__(self, model: Diffu, cfg: Config, *, device: str, dtype: torch.dtype) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        *,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        weights_name: str = "model.safetensors",
        compile_backbone: bool = False,
        lora: str | None = None,
        lora_scale: float = 1.0,
        attention_backend: str | None = None,
        compile_mode: str | None = None,
        compile_vae: bool = False,
        fuse_qkv: bool = True,
    ) -> DiffuPipeline:
        """Build from a local dir OR a Hub repo id, each holding ``config.json`` + ``weights_name``.

        ``Diffu(cfg)`` constructs the frozen VAE (+ the DINOv3 encoder for a dino-mode model) via
        ``from_pretrained`` (so the host needs those repos available/cached), then the trained weights load
        on top. ``compile_backbone`` regionally compiles the MMDiT blocks (``dynamic=True`` — one compile
        covers the variable line widths); its cold start pays off over many lines, so it is off by default.
        ``compile_mode`` / ``compile_vae``: a ``torch.compile`` mode (``"reduce-overhead"`` /
        ``"max-autotune"`` add CUDA graphs) and compiling the VAE decode. ``fuse_qkv`` (default) merges each
        stream's Q/K/V projections into one matmul — exact; skipped when a LoRA is attached.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg_path, weights_path = cls._resolve(path, weights_name)
        cfg = Config.model_validate_json(Path(cfg_path).read_text(encoding="utf-8"))
        model = build_model(
            cfg,
            weights_path,
            device=device,
            dtype=dtype,
            lora=lora,
            lora_scale=lora_scale,
            fuse_qkv=fuse_qkv,
            attention_backend=attention_backend,
            compile_mode=compile_mode,
            compile_vae=compile_vae,
            compile_backbone=compile_backbone,
        )
        return cls(model, model.cfg, device=device, dtype=dtype)

    @staticmethod
    def _resolve(path: str, weights_name: str) -> tuple[str, str]:
        local = Path(path)
        if local.is_dir():
            return str(local / "config.json"), str(local / weights_name)
        from huggingface_hub import hf_hub_download  # only needed for a Hub id

        return hf_hub_download(path, "config.json"), hf_hub_download(path, weights_name)

    @torch.inference_mode()
    def __call__(
        self,
        text: str,
        style: str | Path | Image.Image,
        *,
        num_steps: int | None = None,
        cfg_scale: float = 5.0,
        style_text: str = "",
        style_cfg_scale: float = 0.0,
    ) -> Image.Image:
        """Render ``text`` in the writer's ``style`` -> a tight line image (auto-width + ink-crop).

        ``num_steps`` defaults to the config's. ``style_text`` / ``style_cfg_scale`` apply to canvas-mode
        (``style_mode="canvas"``) models only: the reference's transcription and the style-guidance scale.
        """
        if self.model.canvas:
            ref = load_ref_line(style, self.cfg, text=style_text)
            return generate_line(
                self.model,
                text,
                None,
                num_steps=num_steps,
                cfg_scale=cfg_scale,
                ref=ref,
                style_cfg_scale=style_cfg_scale,
            )
        style_pixels = load_style(style).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        return generate_line(self.model, text, style_pixels, num_steps=num_steps, cfg_scale=cfg_scale)

    def to(self, device: str) -> DiffuPipeline:
        self.model.to(device)
        self.device = device
        return self
