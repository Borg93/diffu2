"""MOSS-Image's encoder blocks: a crop plan's layout caption compiled as MOSS's Layout prompt, the prompt encoded by the
frozen Qwen3.5-9B text encoder (its last hidden states over its chat template, thinking enabled, no system prompt), as
MOSS-Image was trained to read it, and a reference image encoded by the FLUX.2 VAE for a DiT trained to take one."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from diffusers import AutoencoderKLFlux2
from diffusers.modular_pipelines import ComponentSpec, InputParam, ModularPipelineBlocks, OutputParam, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import ConfigSpec
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Qwen3_5Model

from diffu_moss.latents import encode_latents, image_pixels
from diffu_moss.layout_prompt import layout_prompt
from diffu_moss.sample_caption import with_sample, without_sample

if TYPE_CHECKING:
    from diffu_moss.modular_pipelines.moss.modular_pipeline import MossModularPipeline

NEGATIVE_PROMPT = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"  # the official demo's default
MAX_PROMPT_TOKENS = 4096  # the text encoder's context in MOSS-Image's training: a longer prompt is refused, not cut
TAKES_REFERENCE = ConfigSpec(
    "takes_reference",
    False,
    description="Whether the DiT was trained to take a reference image (its modular_model_index.json; diffu-moss-init "
    "reads it from a reference training's export): the released models were not",
)


def chat_text(tokenizer: PreTrainedTokenizerBase, prompt: str) -> str:
    """The prompt in the text encoder's chat template, as MOSS-Image was trained to read it."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )


def prompt_embeds(
    text_encoder: Qwen3_5Model, tokenizer: PreTrainedTokenizerBase, prompt: str, *, device: torch.device
) -> torch.Tensor:
    """One prompt as the DiT reads it: the text encoder's last hidden states ``[1, tokens, hidden]`` over its chat
    template.

    Raises:
        ValueError: If the prompt is longer than the text encoder takes (:data:`MAX_PROMPT_TOKENS`).
    """
    inputs = tokenizer(chat_text(tokenizer, prompt), return_tensors="pt").to(device)
    length = inputs["input_ids"].shape[1]
    if length > MAX_PROMPT_TOKENS:
        raise ValueError(f"the prompt is {length} tokens, past the text encoder's {MAX_PROMPT_TOKENS}")
    return text_encoder(**inputs, use_cache=False, return_dict=True).last_hidden_state


class MossLayoutPromptStep(ModularPipelineBlocks):
    """A crop plan's layout caption as MOSS's Layout prompt (:func:`diffu_moss.layout_prompt.layout_prompt`), from what
    the page pipeline's crop prompt step gives: the caption, its lines and the crop's size. A caption that cites a
    reference image names "the sample image" when one is given (``image``) and drops the citation when not
    (:mod:`diffu_moss.sample_caption`), as a reference training's prompts were written. Without lines the prompt is
    drawn as given."""

    model_name = "moss"

    @property
    @override
    def description(self) -> str:
        return (
            "Compiles a crop's layout caption and its lines (page_lines) into MOSS-Image's Layout prompt: the "
            "caption's prose as the summary, a text item per line with its exact text in its pixel box; its citations "
            "of a reference name the sample image when one is given and are dropped when not. Without page_lines the "
            "prompt is kept as given."
        )

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", description="The prompt, or a crop plan's layout caption (one per call)"),
            InputParam(
                "page_lines",
                type_hint=list,
                description="The crop plan's lines (each with its layout row id and text), cited by the caption",
            ),
            InputParam("height", type_hint=int, description="The height in pixels of the image drawn"),
            InputParam("width", type_hint=int, description="The width in pixels of the image drawn"),
            InputParam("image", description="The reference image (PIL), if any: the sample the caption cites"),
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("prompt", type_hint=str, description="The prompt to encode: MOSS's Layout prompt")]

    def __call__(
        self, components: MossModularPipeline, state: PipelineState
    ) -> tuple[MossModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        if block_state.page_lines is not None:
            texts = {line.id: line.text for line in block_state.page_lines}
            caption = (
                without_sample(block_state.prompt) if block_state.image is None else with_sample(block_state.prompt)
            )
            block_state.prompt = layout_prompt(caption, texts, (block_state.width, block_state.height))
        self.set_block_state(state, block_state)
        return components, state


class MossTextEncoderStep(ModularPipelineBlocks):
    """The prompt, and under classifier-free guidance the negative prompt, as the Qwen3.5 text encoder's last hidden
    states."""

    model_name = "moss"

    @property
    @override
    def description(self) -> str:
        return (
            "Text encoder step that embeds the prompt with Qwen3.5 as MOSS-Image reads it: the last hidden states over "
            f"the chat template, thinking enabled. One prompt per call, at most {MAX_PROMPT_TOKENS} tokens."
        )

    @property
    @override
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("text_encoder", Qwen3_5Model), ComponentSpec("tokenizer", AutoTokenizer)]

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", description="The prompt to guide the generation (one prompt per call)"),
            InputParam.template(
                "negative_prompt",
                default=NEGATIVE_PROMPT,
                description="The prompt the unconditional pass is drawn from (the official demo's by default)",
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
                description="The prompt's last hidden states (1, tokens, hidden)",
            ),
            OutputParam(
                "negative_prompt_embeds",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="The negative prompt's last hidden states; None when the guider needs none",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: MossModularPipeline, state: PipelineState
    ) -> tuple[MossModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        if not isinstance(block_state.prompt, str):
            raise TypeError(f"`prompt` must be a str (one prompt per call), got {type(block_state.prompt).__name__}")
        device = components._execution_device

        def encode(prompt: str) -> torch.Tensor:
            return prompt_embeds(components.text_encoder, components.tokenizer, prompt, device=device)

        block_state.prompt_embeds = encode(block_state.prompt)
        negative = components.requires_unconditional_embeds
        block_state.negative_prompt_embeds = encode(block_state.negative_prompt or "") if negative else None
        self.set_block_state(state, block_state)
        return components, state


class MossReferenceEncodeStep(ModularPipelineBlocks):
    """A reference image (or several) as the clean latents the DiT reads beside the drawn image's: read on its own
    16-px grid, never resized (:func:`diffu_moss.latents.image_pixels`; the page pipeline's ``ReferenceGridStep``
    puts it there as training did), then encoded (:func:`diffu_moss.latents.encode_latents`). Only a DiT trained to
    take a reference (``takes_reference`` in its model index) is given one. Without an image it does nothing.

    Raises:
        ValueError: From the call, if a reference is given to a model that takes none, or is off the 16-px grid.
    """

    model_name = "moss"

    @property
    @override
    def description(self) -> str:
        return (
            "VAE encoder step that encodes the reference images, each on its own 16-px grid, into the DiT's clean "
            "latents, for a model trained to take them (takes_reference). Without an image it does nothing."
        )

    @property
    @override
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKLFlux2)]

    @property
    @override
    def expected_configs(self) -> list[ConfigSpec]:
        return [TAKES_REFERENCE]

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [InputParam("image", description="The reference image (PIL) or images, each on the 16-px grid")]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "reference_latents",
                type_hint=list,
                description="The references' clean latents, one (1, latent_channels, h, w) each; None without one",
            )
        ]

    @torch.no_grad()
    def __call__(
        self, components: MossModularPipeline, state: PipelineState
    ) -> tuple[MossModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        image = block_state.image
        block_state.reference_latents = None
        if image is not None:
            if not components.config.takes_reference:
                raise ValueError(
                    "this MOSS-Image model takes no reference image: only a DiT trained with references does (its "
                    "model index's takes_reference, which diffu-moss-init reads from the export of a reference training)"
                )
            images: list[Image.Image] = image if isinstance(image, list) else [image]
            block_state.reference_latents = [
                encode_latents(components.vae, image_pixels(reference)) for reference in images
            ]
        self.set_block_state(state, block_state)
        return components, state
