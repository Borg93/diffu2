"""The crop generator as two Modular Diffusers pipelines over one :class:`ComponentsManager`.

FLUX.3's own blocks (``Flux3AutoBlocks``) with the prompt upsampler popped (it would rewrite the quoted lines) and
split at the text encoder, the pattern of huggingface.co/blog/modular-diffusers:

- the **text pipeline**: ``CropPromptStep`` -> FLUX.3's text encoder (Qwen3-VL). A crop plan in, its prompt
  embeddings out. On a Ray cluster it is its own stage, encoding ahead of the GPU denoiser.
- the **crop pipeline**: ``CropPromptStep`` -> reference VAE encoder (only when an ``image`` is passed) -> denoise ->
  decode -> ``LineReadbackStep`` -> ``LabelExportStep``. The plan and the embeddings in, the image, the per-line
  read-back and the labels out. FLUX.3's auto blocks pick text-to-image or reference-conditioned denoising from the
  inputs, so one pipeline draws both the v2 (style by name) and the v3 (style by example) crops.

The base model is whatever ``model`` points at: FLUX.3 Klein for the reference runs, later diffu (FLUX.3's
architecture with our own weights) through the same blocks. LoRAs load with diffusers' own FLUX.3 LoRA loader, so
several adapters (v2, v3) can sit on one transformer and be switched with ``set_adapters``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from diffusers import ComponentsManager
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.modular_pipelines import Flux3AutoBlocks, ModularPipeline, SequentialPipelineBlocks
from pydantic import BaseModel, ConfigDict
from safetensors.torch import load_file

from page_synth.blocks import CropPromptStep, LabelExportStep, LineReadbackStep
from page_synth.readback import materialize_position_tables

FLUX3_KLEIN = Path(os.environ.get("FLUX3_MODEL", Path.home() / "Desktop/repos/flux3-models/flux-3-image-klein-base"))
_NEMO_PREFIX = "base_model.model."


def text_blocks() -> SequentialPipelineBlocks:
    """A crop plan in, FLUX.3's prompt embeddings out."""
    return SequentialPipelineBlocks.from_blocks_dict(
        {"crop_prompt": CropPromptStep(), "text_encoder": Flux3AutoBlocks().sub_blocks["text_encoder"]}
    )


def crop_blocks() -> SequentialPipelineBlocks:
    """A crop plan and its prompt embeddings in, the crop's image, read-back and labels out."""
    blocks = Flux3AutoBlocks()
    blocks.sub_blocks.pop("prompt_upsampler")
    blocks.sub_blocks.pop("text_encoder")
    blocks.sub_blocks.insert("crop_prompt", CropPromptStep(), 0)
    blocks.sub_blocks.insert("line_readback", LineReadbackStep(), len(blocks.sub_blocks))
    blocks.sub_blocks.insert("label_export", LabelExportStep(), len(blocks.sub_blocks))
    return blocks


class Pipelines(BaseModel):
    """The text and crop pipelines, loaded once, sharing one ComponentsManager."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: ModularPipeline
    crop: ModularPipeline
    components: ComponentsManager


def _require_loaded(pipe: ModularPipeline) -> None:
    """Fail fast: ``load_components`` logs a component that failed to load and leaves it ``None``."""
    missing = [name for name in pipe.pretrained_component_names if getattr(pipe, name, None) is None]
    if missing:
        raise RuntimeError(f"components failed to load (see the log above): {missing}")


def load_pipelines(model: Path, *, guidance_scale: float, device: str = "cuda") -> Pipelines:
    """Both pipelines with their components loaded (VAE in float32, the rest in bfloat16) and moved to ``device``.

    Raises:
        RuntimeError: If a component failed to load.
    """
    components = ComponentsManager()
    text = text_blocks().init_pipeline(str(model), components_manager=components, collection="text")
    text.load_components(dtype=torch.bfloat16)
    _require_loaded(text)
    crop = crop_blocks().init_pipeline(str(model), components_manager=components, collection="crop")
    crop.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
    _require_loaded(crop)
    crop.update_components(guider=ClassifierFreeGuidance(guidance_scale=guidance_scale))
    text.to(device)
    crop.to(device)
    materialize_position_tables(crop.htr_model)
    return Pipelines(text=text, crop=crop, components=components)


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
