"""NeMo AutoModel LoRA checkpoints loaded into a pipeline with the model's own (diffusers) LoRA loader."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers.modular_pipelines import ModularPipeline
from safetensors.torch import load_file

_NEMO_PREFIX = "base_model.model."


def nemo_lora_state_dict(lora_dir: Path) -> dict[str, torch.Tensor]:
    """A NeMo AutoModel LoRA checkpoint (PEFT keys under ``base_model.model.``) as a diffusers LoRA state dict.

    The ``lora_alpha / r`` scale is folded into ``lora_B``, so the adapter loads at its trained strength whatever
    alpha the loader assumes.
    """
    config = json.loads((lora_dir / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    weights = load_file(lora_dir / "adapter_model.safetensors")
    return {
        "transformer." + key.removeprefix(_NEMO_PREFIX): value * scale if ".lora_B." in key else value
        for key, value in weights.items()
    }


def load_nemo_lora(pipe: ModularPipeline, lora_dir: Path, adapter_name: str) -> None:
    """Load a NeMo AutoModel LoRA checkpoint into ``pipe``'s transformer as the adapter ``adapter_name``."""
    pipe.load_lora_weights(nemo_lora_state_dict(lora_dir), adapter_name=adapter_name)
