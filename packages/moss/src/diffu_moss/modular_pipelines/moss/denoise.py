"""MOSS-Image's denoising loop: at each noise level the DiT predicts the velocity of the latents from the prompt, and
under classifier-free guidance from the negative prompt too with some of its blocks skipped; the two are combined in
fp32 and the scheduler steps the latents with it (FLUX.3's own scheduler step).

The latents stay tokens through the loop, as FLUX.3's do, and the loop's steps carry FLUX.3's names (``denoiser``,
``after_denoiser``), so the page pipeline's line repair (``diffu_page.blocks.repair``) runs in it as in FLUX.3's. The
DiT itself takes the latents on their grid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import FrozenDict
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.modular_pipelines import (
    ComponentSpec,
    InputParam,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    OutputParam,
    PipelineState,
)
from diffusers.modular_pipelines.flux3.denoise import Flux3LoopAfterDenoiser
from diffusers.modular_pipelines.modular_pipeline import BlockState
from diffusers.modular_pipelines.modular_pipeline_utils import ConfigSpec

from diffu_moss.latents import PIXELS_PER_LATENT
from diffu_moss.transformer import MossTransformer

if TYPE_CHECKING:
    from diffu_moss.modular_pipelines.moss.modular_pipeline import MossModularPipeline

GUIDANCE_SCALE = 3.0  # the official demo's
DEMO_UNCOND_SKIP_LAYERS = (22,)  # the official demo's, of the 42 blocks of the released models
UNCOND_SKIP_LAYERS = ConfigSpec(
    "uncond_skip_layers",
    DEMO_UNCOND_SKIP_LAYERS,
    description="The DiT blocks the unconditional pass of classifier-free guidance skips (its modular_model_index.json)",
)


class MossLoopDenoiser(ModularPipelineBlocks):
    """The guided velocity of the latents at one noise level: the DiT's prediction from the prompt and, while the
    guider guides, from the negative prompt with the model's ``uncond_skip_layers`` skipped, combined by the guider in
    fp32."""

    model_name = "moss"

    @property
    @override
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", MossTransformer),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": GUIDANCE_SCALE}),
                default_creation_method="from_config",
            ),
        ]

    @property
    @override
    def expected_configs(self) -> list[ConfigSpec]:
        return [UNCOND_SKIP_LAYERS]

    @property
    @override
    def description(self) -> str:
        return (
            "Step within the denoising loop that predicts the guided velocity of the latents: the DiT on the prompt "
            "and, under guidance, on the negative prompt with the model's uncond_skip_layers skipped, combined in "
            "fp32. This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `MossDenoiseStep`)"
        )

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor, description="The latents as tokens."),
            InputParam.template("prompt_embeds"),
            InputParam.template("negative_prompt_embeds"),
            InputParam("height", required=True, type_hint=int, description="The height in pixels of the image drawn"),
            InputParam("width", required=True, type_hint=int, description="The width in pixels of the image drawn"),
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("noise_pred", type_hint=torch.Tensor, description="The guided velocity, as tokens, in fp32")
        ]

    @torch.no_grad()
    def __call__(
        self, components: MossModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> tuple[MossModularPipeline, BlockState]:
        tokens: torch.Tensor = block_state.latents
        grid = (block_state.height // PIXELS_PER_LATENT, block_state.width // PIXELS_PER_LATENT)
        latents = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], *grid)
        guider_inputs = {
            "text": (block_state.prompt_embeds, block_state.negative_prompt_embeds),
            "skip_layers": (frozenset(), frozenset(components.config.uncond_skip_layers)),
        }
        guider = components.guider
        guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        passes = guider.prepare_inputs(guider_inputs)
        for guider_pass in passes:
            guider.prepare_models(components.transformer)
            text: torch.Tensor = guider_pass.text
            mask = torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)
            velocity = components.transformer(latents, t, text, mask, guider_pass.skip_layers).float()
            guider_pass.noise_pred = velocity.flatten(2).transpose(1, 2)
            guider.cleanup_models(components.transformer)
        block_state.noise_pred = guider(passes)[0]
        return components, block_state


class MossDenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    """Runs its steps once per timestep, in order."""

    model_name = "moss"

    @property
    @override
    def description(self) -> str:
        return (
            "Pipeline block that iteratively denoises the latents over `timesteps`. "
            "The specific steps within each iteration can be customized with `sub_blocks` attribute"
        )

    @property
    @override
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [
            # config-less: the scheduler config is owned by `MossSetTimestepsStep`, whose spec is combined first
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler, default_creation_method="from_config"),
            ComponentSpec("transformer", MossTransformer),
        ]

    @property
    @override
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam("timesteps", required=True, type_hint=torch.Tensor, description="The timesteps to denoise at."),
            InputParam(
                "num_inference_steps", required=True, type_hint=int, description="The number of denoising steps."
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: MossModularPipeline, state: PipelineState
    ) -> tuple[MossModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        with self.progress_bar(total=block_state.num_inference_steps) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                progress_bar.update()
        self.set_block_state(state, block_state)
        return components, state


class MossDenoiseStep(MossDenoiseLoopWrapper):
    """The denoising loop: at each timestep :class:`MossLoopDenoiser`, then FLUX.3's scheduler step."""

    block_classes = [MossLoopDenoiser(), Flux3LoopAfterDenoiser()]
    block_names = ["denoiser", "after_denoiser"]

    @property
    @override
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoises the latents for MOSS-Image.\n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `MossLoopDenoiser`\n"
            " - `Flux3LoopAfterDenoiser`"
        )
