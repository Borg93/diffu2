"""``DiffuPipeline`` — a one-call inference wrapper so a trained Diffu model is easy to load and run.

Diffu is a bespoke diffusion model (diffusers SD3 MMDiT backbone + a custom 2D-RoPE joint-attention
processor + a frozen Qwen-Image VAE + glyph-line content conditioning + DINOv3 style), loaded from a
single non-strict ``state_dict``. It is NOT decomposed into standard diffusers ``ModelMixin`` components,
so this pipeline wraps the whole :class:`~diffu.model.Diffu` and exposes the same ``load + generate`` path
the studio and eval ruler use.

    from diffu.pipeline import DiffuPipeline
    pipe = DiffuPipeline.from_pretrained("gabriel/diffu-swedish-htr")   # a Hub repo id or a local dir
    img = pipe("Göteborgs poliskammare", style="a_line_crop_of_the_target_hand.png")
    img.save("out.png")

The release repo/dir holds ``config.json`` + ``model.safetensors`` (see ``scripts/export_release.py``).
``style`` is any single-line crop of the target writer's hand (PIL image or path); it is ImageNet-
normalised to the DINO input the model was trained on. The model was trained with the DINOv3 style
encoder, so inference needs it too (gated — accept the licence on the Hub).
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .config import Config
from .generate import assert_conditioner_loaded, generate_line, load_checkpoint, load_ref_line
from .model import Diffu

# DINOv3 expects ImageNet-normalised 224x224 input (matches diffu.generate.load_style / training).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


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
    ) -> DiffuPipeline:
        """Build from a local dir OR a Hub repo id, each holding ``config.json`` + ``weights_name``.

        ``Diffu(cfg)`` constructs the frozen VAE + DINOv3 encoder via ``from_pretrained`` (so the host
        needs those repos available/cached), then ``load_checkpoint`` overwrites with the trained weights.

        ``compile_backbone`` fuses the repeated MMDiT blocks (the same regional ``torch.compile`` win used
        in training; ``dynamic=True`` — one compile covers the variable line widths). The cold-start cost
        pays off over many lines (batch/served), not a single one-off call — left off by default.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # TF32 matmuls for fp32 inference (train.py sets this for training; without it every GEMM —
        # compiled or not — runs the several-fold-slower strict-fp32 path).
        torch.set_float32_matmul_precision("high")
        cfg_path, weights_path = cls._resolve(path, weights_name)
        cfg = Config.model_validate_json(Path(cfg_path).read_text(encoding="utf-8"))
        model = Diffu(cfg).to(device=device, dtype=dtype).eval()
        load_checkpoint(model, weights_path)
        assert_conditioner_loaded(model, weights_path)  # a bad config/weights pair must not ship gibberish
        if lora:  # a Stage-3 LoRA adapter (dir or Hub id) on top of the base checkpoint
            from .lora import load_lora

            load_lora(model, lora, scale=lora_scale)
        if attention_backend:  # a diffusers kernel backend, e.g. "_native_cudnn" / "sage_blackwell_hub"
            model.backbone.set_attention_backend(attention_backend)
        if compile_backbone:
            model.backbone.compile_blocks()  # regional compile — same fusion as the training win
        return cls(model, cfg, device=device, dtype=dtype)

    @staticmethod
    def _resolve(path: str, weights_name: str) -> tuple[str, str]:
        local = Path(path)
        if local.is_dir():
            return str(local / "config.json"), str(local / weights_name)
        from huggingface_hub import hf_hub_download  # only needed for a Hub id

        return hf_hub_download(path, "config.json"), hf_hub_download(path, weights_name)

    def _preprocess_style(self, style: str | Path | Image.Image, size: int = 224) -> torch.Tensor:
        """Any writer-style line crop -> ImageNet-normalised ``[1, 3, size, size]`` on the model device."""
        if isinstance(style, (str, Path)):
            with Image.open(style) as handle:
                img = handle.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        else:
            img = style.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        tensor = TF.normalize(TF.to_tensor(img), _IMAGENET_MEAN, _IMAGENET_STD).unsqueeze(0)
        return tensor.to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def __call__(
        self,
        text: str,
        style: str | Path | Image.Image,
        *,
        num_steps: int = 24,
        cfg_scale: float = 5.0,
        style_text: str = "",
        style_cfg_scale: float = 0.0,
    ) -> Image.Image:
        """Render ``text`` in the writer's ``style`` -> a tight line image (auto-width + ink-crop).

        ``style_text`` / ``style_cfg_scale`` apply to canvas-mode (``style_mode="canvas"``) models only:
        the reference's transcription and the style-guidance scale.
        """
        if self.model.canvas:
            ref = load_ref_line(style, self.cfg, text=style_text)
            return generate_line(
                self.model,
                text,
                None,
                cfg=self.cfg,
                num_steps=num_steps,
                cfg_scale=cfg_scale,
                ref=ref,
                style_cfg_scale=style_cfg_scale,
            )
        style_tensor = self._preprocess_style(style)
        return generate_line(
            self.model, text, style_tensor, cfg=self.cfg, num_steps=num_steps, cfg_scale=cfg_scale
        )

    def to(self, device: str) -> DiffuPipeline:
        self.model.to(device)
        self.device = device
        return self
