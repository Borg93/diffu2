"""The crop generator as two Modular Diffusers pipelines over one :class:`ComponentsManager`.

FLUX.3's own steps (from ``Flux3AutoBlocks``) without the prompt upsampler (it would rewrite the quoted lines), split
at the text encoder (the pattern of huggingface.co/blog/modular-diffusers) and declared like diffusers declares
FLUX.3's own blocks, as :class:`CropTextBlocks` and :class:`CropBlocks`:

- the **text pipeline**: ``CropPromptStep`` -> FLUX.3's text encoder (Qwen3-VL). A crop plan in, its prompt
  embeddings out.
- the **crop pipeline**: ``CropPromptStep`` -> ``ReferenceGridStep`` (a reference on v3's training grid) -> reference
  VAE encoder (only when an ``image`` is passed) -> denoise -> decode -> ``LineReadbackStep`` -> ``LabelExportStep``. The plan and the embeddings in, the image, the per-line
  read-back and the labels out. FLUX.3's auto blocks pick text-to-image or reference-conditioned denoising from the
  inputs, so one pipeline draws both the v2 (style by name) and the v3 (style by example) crops.

The base model is whatever ``model`` points at: FLUX.3 Klein for the reference runs, later diffu (FLUX.3's
architecture with our own weights) through the same blocks. LoRAs load with diffusers' own FLUX.3 LoRA loader, so
several adapters (v2, v3) can sit on one transformer and be switched with ``set_adapters``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers import ComponentsManager
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.modular_pipelines import Flux3AutoBlocks, ModularPipeline, OutputParam, SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict
from pydantic import BaseModel, ConfigDict
from safetensors.torch import load_file

from page_synth.modular.blocks import CropPromptStep, LabelExportStep, LineReadbackStep, ReferenceGridStep
from page_synth.modular.readback import materialize_position_tables

_NEMO_PREFIX = "base_model.model."


_FLUX3 = Flux3AutoBlocks().sub_blocks  # FLUX.3's own steps, by name

CROP_TEXT_BLOCKS = InsertableDict([("crop_prompt", CropPromptStep()), ("text_encoder", _FLUX3["text_encoder"])])

CROP_BLOCKS = InsertableDict(
    [
        ("crop_prompt", CropPromptStep()),
        ("reference_grid", ReferenceGridStep()),
        ("vae_encoder", _FLUX3["vae_encoder"]),
        ("denoise", _FLUX3["denoise"]),
        ("decode", _FLUX3["decode"]),
        ("line_readback", LineReadbackStep()),
        ("label_export", LabelExportStep()),
    ]
)


class CropTextBlocks(SequentialPipelineBlocks):
    """A crop plan in, FLUX.3's prompt embeddings out: the text-encoder stage."""

    model_name = "flux3"
    block_classes = CROP_TEXT_BLOCKS.values()
    block_names = CROP_TEXT_BLOCKS.keys()

    @property
    def description(self) -> str:
        return "Encodes a crop plan's layout caption with FLUX.3's text encoder (Qwen3-VL), as given (no upsampler)."

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("prompt_embeds"), OutputParam.template("negative_prompt_embeds")]


class CropBlocks(SequentialPipelineBlocks):
    """A crop plan and its prompt embeddings in, the crop's image, read-back and labels out.

    FLUX.3's VAE encoder and denoise steps are auto blocks: an ``image`` input (a reference crop) switches them to
    reference-conditioned denoising, so one pipeline draws both the v2 (style by name) and v3 (style by example)
    crops.
    """

    model_name = "flux3"
    block_classes = CROP_BLOCKS.values()
    block_names = CROP_BLOCKS.keys()
    _workflow_map = {
        "text2image": {"crop_plan": True, "prompt_embeds": True},
        "image_conditioned": {"crop_plan": True, "prompt_embeds": True, "image": True},
    }

    @property
    def description(self) -> str:
        return (
            "Draws a planned crop with FLUX.3 from its prompt embeddings (reference-conditioned when an `image` is "
            "given), reads every line back with TrOCR and writes the crop's ALTO/PAGE labels."
        )

    @property
    def outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("images"),
            OutputParam("line_reads", type_hint=list, description="What the recognizer read in each line box."),
            OutputParam("line_cers", type_hint=list, description="Per-line CER against the line's text."),
            OutputParam("labels_xml", type_hint=str, description="The crop's labels."),
            OutputParam("unclear_lines", type_hint=int, description="Lines over max_line_cer."),
        ]


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
    text = CropTextBlocks().init_pipeline(str(model), components_manager=components, collection="text")
    text.load_components(dtype=torch.bfloat16)
    _require_loaded(text)
    crop = CropBlocks().init_pipeline(str(model), components_manager=components, collection="crop")
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
