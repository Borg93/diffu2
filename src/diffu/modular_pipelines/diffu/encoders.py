"""diffu's encoder blocks: the prompt through Qwen3.5, reference images through the Qwen-Image VAE.

Their contracts are FLUX.3's (``Flux3TextEncoderStep``, ``Flux3VaeEncoderStep``), so FLUX.3's own denoise blocks run on
their outputs unchanged.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, override

import torch
from diffusers import AutoencoderKLQwenImage
from diffusers.modular_pipelines import ComponentSpec, InputParam, ModularPipelineBlocks, OutputParam, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import ConfigSpec
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Qwen3_5ForConditionalGeneration

from diffu.configs import QWEN35_LAYERS
from diffu.latents import encode_latents

if TYPE_CHECKING:
    from diffu.modular_pipelines.diffu.modular_pipeline import DiffuModularPipeline

logger = logging.getLogger(__name__)

# The prompt window is right-padded to a multiple of this many tokens. The transformer attends to the padding, so the
# window is part of the model recipe, as it is FLUX.3's.
TEXT_WINDOW = 80
MAX_SEQUENCE_LENGTH = 8192  # prompt tokens the text encoder reads at most; training caches with the same limit


def prompt_embeds(
    text_encoder: Qwen3_5ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    layers: Sequence[int],
    max_sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    """One prompt's embeddings ``(1, window, len(layers) * hidden)``.

    The prompt goes through the chat template with thinking off, is right-padded to a multiple of :data:`TEXT_WINDOW`
    tokens (at most ``max_sequence_length``), and the hidden states at ``layers`` are stacked on channels.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    real_length = len(tokenizer(text, truncation=True, max_length=max_sequence_length)["input_ids"])
    window = min(math.ceil(real_length / TEXT_WINDOW) * TEXT_WINDOW, max_sequence_length)
    inputs = tokenizer(
        text, return_tensors="pt", padding="max_length", padding_side="right", truncation=True, max_length=window
    )
    input_ids, attention_mask = inputs["input_ids"].to(device), inputs["attention_mask"].to(device)
    # explicit position ids, padding at 1 as in FLUX.3's encoder: the padding's states are conditioning, so they must
    # not change with a transformers version's default for padded positions
    position_ids = torch.arange(window, device=device)[None].masked_fill(attention_mask == 0, 1)
    output = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
        use_cache=False,
    )
    stacked = torch.stack([output.hidden_states[layer] for layer in layers], dim=2)  # (1, window, layers, hidden)
    return stacked.flatten(2).to(text_encoder.dtype)


def _check_prompts(prompt: object, negative_prompt: object) -> None:
    """One prompt per call: the window is sized per prompt and the transformer attends to its padding."""
    if not isinstance(prompt, str):
        raise ValueError(f"`prompt` must be a str (one prompt per call), got {type(prompt).__name__}")
    if negative_prompt is not None and not isinstance(negative_prompt, str):
        raise ValueError(f"`negative_prompt` must be a str or None, got {type(negative_prompt).__name__}")


def _encodes_negative(components: DiffuModularPipeline, negative_prompt: str | None) -> bool:
    """Whether the negative prompt is encoded: the guider needs unconditional embeddings (the empty prompt's unless one
    is given), or there is no guider and the caller passed one."""
    if components.requires_unconditional_embeds:
        return True
    if negative_prompt is not None and getattr(components, "guider", None) is not None:
        logger.warning("`negative_prompt` is given but the guider is disabled; it is ignored")
        return False
    return negative_prompt is not None


class DiffuTextEncoderStep(ModularPipelineBlocks):
    """The prompt, and under classifier-free guidance the negative (empty) prompt, as Qwen3.5 embeddings."""

    model_name = "diffu"

    @property
    @override
    def description(self) -> str:
        return (
            "Text encoder step that embeds the prompt with Qwen3.5 (chat template, thinking off), right-padded to a "
            f"multiple of {TEXT_WINDOW} tokens, its chosen hidden layers stacked on channels. One prompt per call."
        )

    @property
    @override
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3_5ForConditionalGeneration),
            ComponentSpec("tokenizer", AutoTokenizer),
        ]

    @property
    @override
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec(
                "text_layers",
                QWEN35_LAYERS,
                description="The text-encoder hidden layers the model was trained on (its modular_model_index.json).",
            )
        ]

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", description="The prompt to guide the generation (one prompt per call)"),
            InputParam.template("negative_prompt", description="The prompt not to guide the generation"),
            InputParam.template(
                "max_sequence_length",
                default=MAX_SEQUENCE_LENGTH,
                description="Maximum number of prompt tokens; longer prompts are truncated.",
            ),
            InputParam(
                "text_encoder_out_layers",
                type_hint=tuple[int, ...],
                required=False,
                description="Hidden layers to stack instead of the model's own `text_layers`",
            ),
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "prompt_embeds",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="Text embeddings (stacked Qwen3.5 hidden states) used to guide the image generation",
            ),
            OutputParam(
                "negative_prompt_embeds",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="Negative text embeddings used to guide the image generation",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: DiffuModularPipeline, state: PipelineState
    ) -> tuple[DiffuModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        _check_prompts(block_state.prompt, block_state.negative_prompt)
        layers = block_state.text_encoder_out_layers or components.config.text_layers

        def encode(prompt: str) -> torch.Tensor:
            return prompt_embeds(
                components.text_encoder,
                components.tokenizer,
                prompt,
                layers=layers,
                max_sequence_length=block_state.max_sequence_length,
                device=components._execution_device,
            )

        block_state.prompt_embeds = encode(block_state.prompt)
        negative = _encodes_negative(components, block_state.negative_prompt)
        block_state.negative_prompt_embeds = encode(block_state.negative_prompt or "") if negative else None
        self.set_block_state(state, block_state)
        return components, state


class DiffuVaeEncoderStep(ModularPipelineBlocks):
    """Reference images as diffu latents (:func:`diffu.latents.encode_latents`), one tensor per image."""

    model_name = "diffu"

    @property
    @override
    def description(self) -> str:
        return (
            "VAE encoder step that encodes the preprocessed reference images with the Qwen-Image VAE into normalized "
            "packed latents, one latent tensor per image."
        )

    @property
    @override
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKLQwenImage)]

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "condition_images",
                required=True,
                type_hint=list[torch.Tensor],
                description="The preprocessed reference images, one (1, 3, height_i, width_i) tensor in [-1, 1] each",
            ),
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "image_cond_latents",
                type_hint=list[torch.Tensor],
                description="Normalized packed latents of the reference images, one (1, 4 * z_dim, height_i // 16, "
                "width_i // 16) tensor each",
            )
        ]

    @torch.no_grad()
    def __call__(
        self, components: DiffuModularPipeline, state: PipelineState
    ) -> tuple[DiffuModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        vae = components.vae
        device = components._execution_device
        block_state.image_cond_latents = [
            encode_latents(vae, image.to(device=device, dtype=vae.dtype)) for image in block_state.condition_images
        ]
        self.set_block_state(state, block_state)
        return components, state
