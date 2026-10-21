"""The crop generator: one crop plan in, one image with its labels out. The unit a Ray Data actor wraps.

Loaded once (both pipelines of :mod:`page_synth.pipelines`, plus LoRAs), then called per crop. ``encode`` and
``draw`` are the two stages; ``__call__`` runs both, a Ray job can run them as separate actors.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict

from page_synth.blocks import LabelFormat
from page_synth.pipelines import load_nemo_lora, load_pipelines
from page_synth.plan import CropPlan
from page_synth.readback import read_lines


class PromptEmbeddings(BaseModel):
    """A crop's prompt embeddings: the conditional and, under classifier-free guidance, the empty prompt's."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None


class GeneratedCrop(BaseModel):
    """One drawn crop: its image, its labels and how well its lines read back."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Image.Image
    labels_xml: str
    line_reads: list[str]
    line_cers: list[float]
    unclear_lines: int
    seconds: float


class GeneratorSettings(BaseModel):
    """How every crop is drawn.

    Attributes:
        steps: Denoising steps per crop.
        guidance_scale: Classifier-free guidance scale; 1 or less samples without guidance.
        label_format: ``alto`` or ``page``.
        text_layers: The text-encoder hidden layers stacked into the prompt embeddings; ``None`` is the base
            model's own choice (FLUX.3: every 4th of Qwen3-VL's).
        device: Where the components run.
    """

    steps: int = 50
    guidance_scale: float = 4.0
    label_format: LabelFormat = "alto"
    text_layers: tuple[int, ...] | None = None
    device: str = "cuda"


class CropGenerator:
    """FLUX.3 (+ LoRAs) as a crop generator, loaded once.

    Args:
        model: The base model directory (a Modular Diffusers FLUX.3 repo).
        loras: NeMo LoRA checkpoint ``model/`` dirs by adapter name; all are active.
        settings: How every crop is drawn.
    """

    def __init__(self, model: Path, *, loras: dict[str, Path], settings: GeneratorSettings) -> None:
        self._pipes = load_pipelines(model, guidance_scale=settings.guidance_scale, device=settings.device)
        for name, lora_dir in loras.items():
            load_nemo_lora(self._pipes.crop, lora_dir, name)
        if loras:
            self._pipes.crop.set_adapters(list(loras))
        self._settings = settings

    def encode(self, plan: CropPlan) -> PromptEmbeddings:
        """The plan's prompt embeddings (the text-encoder stage)."""
        layers = {"text_encoder_out_layers": self._settings.text_layers} if self._settings.text_layers else {}
        state = self._pipes.text(
            crop_plan=plan,
            negative_prompt="" if self._settings.guidance_scale > 1 else None,
            output=["prompt_embeds", "negative_prompt_embeds"],
            **layers,
        )
        return PromptEmbeddings(**state)

    def draw(
        self, plan: CropPlan, embeddings: PromptEmbeddings, *, seed: int, image_name: str, reference: Image.Image | None
    ) -> GeneratedCrop:
        """The crop drawn from its embeddings, read back and labelled (the denoise stage)."""
        start = time.perf_counter()
        state = self._pipes.crop(
            crop_plan=plan,
            prompt_embeds=embeddings.prompt_embeds,
            negative_prompt_embeds=embeddings.negative_prompt_embeds,
            num_inference_steps=self._settings.steps,
            generator=torch.Generator().manual_seed(seed),
            label_format=self._settings.label_format,
            image_name=image_name,
            output=["images", "line_reads", "line_cers", "labels_xml", "unclear_lines"],
            **({"image": reference} if reference is not None else {}),
        )
        return GeneratedCrop(
            image=state["images"][0],
            labels_xml=state["labels_xml"],
            line_reads=state["line_reads"],
            line_cers=state["line_cers"],
            unclear_lines=state["unclear_lines"],
            seconds=time.perf_counter() - start,
        )

    def __call__(
        self, plan: CropPlan, *, seed: int, image_name: str, reference: Image.Image | None = None
    ) -> GeneratedCrop:
        return self.draw(plan, self.encode(plan), seed=seed, image_name=image_name, reference=reference)

    def read(self, image: Image.Image, plan: CropPlan) -> tuple[list[str], list[float]]:
        """Read ``image``'s line boxes with the generator's HTR model (a real crop, for comparison)."""
        crop = self._pipes.crop
        return read_lines(image, plan.lines, model=crop.htr_model, processor=crop.htr_processor)
