"""NeMo AutoModel LoRA checkpoints as diffusers LoRA state dicts, for the model's own (diffusers) LoRA loader."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

ADAPTER_CONFIG = "adapter_config.json"  # the two files of a NeMo LoRA checkpoint's model/ dir
ADAPTER_WEIGHTS = "adapter_model.safetensors"
_NEMO_PREFIX = "base_model.model."


def nemo_lora_state_dict(lora_dir: Path) -> dict[str, torch.Tensor]:
    """A NeMo AutoModel LoRA checkpoint (PEFT keys under ``base_model.model.``) as a diffusers LoRA state dict.

    The ``lora_alpha / r`` scale is folded into ``lora_B``, so the adapter loads at its trained strength whatever
    alpha the loader assumes.

    Raises:
        ValueError: If the checkpoint scales otherwise than by one ``lora_alpha / r`` (rsLoRA, per-module ranks or
            alphas), or holds keys outside NeMo's prefix: either would load at the wrong strength or place.
    """
    config = json.loads((lora_dir / ADAPTER_CONFIG).read_text())
    if config.get("use_rslora") or config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError(f"{lora_dir}: rsLoRA and per-module ranks or alphas are not supported, only lora_alpha / r")
    scale = config["lora_alpha"] / config["r"]
    weights = load_file(lora_dir / ADAPTER_WEIGHTS)
    foreign = [key for key in weights if not key.startswith(_NEMO_PREFIX)]
    if foreign:
        raise ValueError(f"{lora_dir}: {len(foreign)} keys lack NeMo's {_NEMO_PREFIX!r} prefix, e.g. {foreign[0]!r}")
    return {
        "transformer." + key.removeprefix(_NEMO_PREFIX): value * scale if ".lora_B." in key else value
        for key, value in weights.items()
    }
