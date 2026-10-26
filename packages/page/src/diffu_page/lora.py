"""NeMo AutoModel LoRA checkpoints as diffusers LoRA state dicts, for the model's own (diffusers) LoRA loader."""

from __future__ import annotations

from pathlib import Path

import torch
from pydantic import BaseModel, ValidationError
from safetensors.torch import load_file

ADAPTER_CONFIG = "adapter_config.json"  # the two files of a NeMo LoRA checkpoint's model/ dir
ADAPTER_WEIGHTS = "adapter_model.safetensors"
_NEMO_PREFIX = "base_model.model."


class AdapterConfig(BaseModel):
    """The part of a PEFT ``adapter_config.json`` that sets a LoRA's strength; its other keys are ignored."""

    r: int
    lora_alpha: float
    use_rslora: bool = False
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, float] = {}

    @classmethod
    def read(cls, lora_dir: Path) -> AdapterConfig:
        """``lora_dir``'s adapter config.

        Raises:
            ValueError: If it is not a LoRA config with a rank and an alpha, naming the directory.
        """
        try:
            return cls.model_validate_json((lora_dir / ADAPTER_CONFIG).read_text(encoding="utf-8"))
        except ValidationError as error:
            raise ValueError(f"{lora_dir}/{ADAPTER_CONFIG} is no LoRA config with r and lora_alpha: {error}") from error


def nemo_lora_state_dict(lora_dir: Path) -> dict[str, torch.Tensor]:
    """A NeMo AutoModel LoRA checkpoint (PEFT keys under ``base_model.model.``) as a diffusers LoRA state dict.

    The ``lora_alpha / r`` scale is folded into ``lora_B``, so the adapter loads at its trained strength whatever
    alpha the loader assumes.

    Raises:
        ValueError: If the checkpoint scales otherwise than by one ``lora_alpha / r`` (rsLoRA, per-module ranks or
            alphas), or holds keys outside NeMo's prefix: either would load at the wrong strength or place.
    """
    config = AdapterConfig.read(lora_dir)
    if config.use_rslora or config.rank_pattern or config.alpha_pattern:
        raise ValueError(f"{lora_dir}: rsLoRA and per-module ranks or alphas are not supported, only lora_alpha / r")
    scale = config.lora_alpha / config.r
    weights = load_file(lora_dir / ADAPTER_WEIGHTS)
    foreign = [key for key in weights if not key.startswith(_NEMO_PREFIX)]
    if foreign:
        raise ValueError(f"{lora_dir}: {len(foreign)} keys lack NeMo's {_NEMO_PREFIX!r} prefix, e.g. {foreign[0]!r}")
    return {
        "transformer." + key.removeprefix(_NEMO_PREFIX): value * scale if ".lora_B." in key else value
        for key, value in weights.items()
    }
