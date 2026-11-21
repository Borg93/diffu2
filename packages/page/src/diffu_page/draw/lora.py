"""LoRAs at draw time: the folder every LoRA option names, laid out as PEFT saves an adapter (a PEFT adapter of
MOSS-Image's DiT, ``diffu-moss-convert-lora``'s; read and loaded by :mod:`diffu_moss.adapters`)."""

from __future__ import annotations

from pathlib import Path

ADAPTER_CONFIG = "adapter_config.json"  # the two files of a LoRA adapter folder
ADAPTER_WEIGHTS = "adapter_model.safetensors"


def lora_model_dir(path: Path) -> Path:
    """``path``, checked to be a LoRA adapter folder (a MOSS-Image adapter of ``diffu-moss-convert-lora``): it holds
    the adapter's config and weights.

    Raises:
        ValueError: If it lacks either file.
    """
    missing = [file for file in (ADAPTER_CONFIG, ADAPTER_WEIGHTS) if not (path / file).is_file()]
    if missing:
        raise ValueError(f"{path} is not a LoRA adapter folder: it has no {' or '.join(missing)}")
    return path
