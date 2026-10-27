"""MOSS-Image's steps before the denoising loop: the initial noise, drawn in fp32 from the generator and packed into
tokens as FLUX.3 packs its latents (so the page pipeline's line repair runs on them), and the sampling grid: ``steps``
noise levels from 1 down, shifted, each moved to the nearest level of the 1,000-step training grid."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import FrozenDict
from diffusers.modular_pipelines import ComponentSpec, InputParam, ModularPipelineBlocks, OutputParam, PipelineState
from diffusers.utils.torch_utils import randn_tensor
from torch import Tensor

from diffu_moss.latents import PIXELS_PER_LATENT

if TYPE_CHECKING:
    from diffu_moss.modular_pipelines.moss.modular_pipeline import MossModularPipeline

TRAIN_STEPS = 1000
SAMPLING_SHIFT = 5.0  # the official demo's


def sampling_sigmas(steps: int, shift: float) -> list[float]:
    """The sampler's noise levels: ``steps`` levels from 1 down, shifted by ``shift``, each moved to the nearest level
    of the training grid (consecutive repeats merged)."""

    def shifted_grid(count: int, grid_shift: float) -> Tensor:
        values = torch.linspace(1, grid_shift / (TRAIN_STEPS + grid_shift - 1), count)
        return grid_shift * values / (1 + (grid_shift - 1) * values)

    sigmas = shifted_grid(steps, shift)
    training = shifted_grid(TRAIN_STEPS, 1.0).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=TRAIN_STEPS - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    return torch.unique_consecutive(training[nearest]).tolist()


class MossPrepareLatentsStep(ModularPipelineBlocks):
    """The initial noise of one image, drawn in fp32 from the generator as ``[1, latent_channels, H / 16, W / 16]``
    and packed into tokens ``[1, H / 16 * W / 16, latent_channels]``, row by row.

    Raises:
        ValueError: From the call, if a side is not a positive multiple of 16 px, or a reference image is given:
            MOSS-Image draws from its prompt alone.
    """

    model_name = "moss"

    @property
    @override
    def description(self) -> str:
        return (
            "Prepare latents step that draws the initial noise in fp32 from the generator and packs it into tokens, "
            "row by row, as FLUX.3 packs its latents."
        )

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", default=1024),
            InputParam.template("width", default=1024),
            InputParam.template("generator"),
            InputParam("image", description="Refused: MOSS-Image takes no reference image"),
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents", type_hint=Tensor, description="The initial noise as tokens (1, seq_len, latent_channels)"
            )
        ]

    @torch.no_grad()
    def __call__(
        self, components: MossModularPipeline, state: PipelineState
    ) -> tuple[MossModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        if block_state.image is not None:
            raise ValueError("MOSS-Image draws from its prompt alone: it takes no reference image")
        height, width = block_state.height, block_state.width
        if height <= 0 or width <= 0 or height % PIXELS_PER_LATENT or width % PIXELS_PER_LATENT:
            raise ValueError(f"the size {(width, height)} is not a positive multiple of {PIXELS_PER_LATENT} px")
        shape = (1, components.num_channels_latents, height // PIXELS_PER_LATENT, width // PIXELS_PER_LATENT)
        noise = randn_tensor(
            shape, generator=block_state.generator, device=components._execution_device, dtype=torch.float32
        )
        block_state.latents = noise.flatten(2).transpose(1, 2)
        self.set_block_state(state, block_state)
        return components, state


class MossSetTimestepsStep(ModularPipelineBlocks):
    """The scheduler set to MOSS-Image's sampling grid (:func:`sampling_sigmas`) at the shift its config names: the
    sampling shift, 5 by default, as MOSS-Image's own scheduler config names it. The grid is shifted and on the
    training grid already, so the scheduler steps it as given, at shift 1."""

    model_name = "moss"

    @property
    @override
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "scheduler",
                FlowMatchEulerDiscreteScheduler,
                config=FrozenDict({"num_train_timesteps": TRAIN_STEPS, "shift": SAMPLING_SHIFT}),
                default_creation_method="from_config",
            )
        ]

    @property
    @override
    def description(self) -> str:
        return (
            "Step that sets the scheduler's timesteps to MOSS-Image's sampling grid: the steps shifted by the "
            "scheduler's shift, then each moved to the nearest level of the 1,000-step training grid."
        )

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [InputParam.template("num_inference_steps")]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=Tensor, description="The timesteps to use for inference"),
            OutputParam(
                "num_inference_steps",
                type_hint=int,
                description="The number of denoising steps: fewer than asked when shifted levels share a grid level",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: MossModularPipeline, state: PipelineState
    ) -> tuple[MossModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        scheduler: FlowMatchEulerDiscreteScheduler = components.scheduler
        sigmas = sampling_sigmas(block_state.num_inference_steps, scheduler.config.shift)
        scheduler.set_shift(1.0)
        scheduler.set_timesteps(sigmas=sigmas, device=components._execution_device)
        scheduler.set_begin_index(0)
        block_state.timesteps = scheduler.timesteps
        block_state.num_inference_steps = len(scheduler.timesteps)
        self.set_block_state(state, block_state)
        return components, state
