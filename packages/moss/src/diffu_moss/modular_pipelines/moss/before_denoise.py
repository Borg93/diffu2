"""MOSS-Image's steps before the denoising loop: the initial noise, drawn in fp32 from the generator and packed into
tokens as FLUX.3 packs its latents (so the page pipeline's line repair runs on them), and the sampling grid: ``steps``
noise levels from 1 down, shifted, each moved to the nearest level of the 1,000-step training grid, stepped by
MOSS-Image's own Euler sampler or by UniPC (:class:`Sampler`)."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, override

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from diffusers.configuration_utils import FrozenDict
from diffusers.modular_pipelines import ComponentSpec, InputParam, ModularPipelineBlocks, OutputParam, PipelineState
from diffusers.utils.torch_utils import randn_tensor
from torch import Tensor

from diffu_moss.latents import PIXELS_PER_LATENT

if TYPE_CHECKING:
    from diffu_moss.modular_pipelines.moss.modular_pipeline import MossModularPipeline

TRAIN_STEPS = 1000
SAMPLING_SHIFT = 5.0  # the official demo's


class Sampler(StrEnum):
    """What steps the sampling grid, calling the DiT once a pass a step at the grid's levels either way."""

    EULER = "euler"  # MOSS-Image's own: diffusers' FlowMatchEulerDiscreteScheduler, the velocity a step
    UNIPC = "unipc"  # diffusers' UniPCMultistepScheduler: a predictor-corrector on the last steps' predictions too


def unipc_sampler(euler: FlowMatchEulerDiscreteScheduler) -> UniPCMultistepScheduler:
    """UniPC for the sampling grid ``euler`` draws: diffusers' multistep predictor-corrector (its order 2, bh2, the
    corrector at every step after the first, the last step of order 1 and onto sigma 0) on flow sigmas, the DiT's
    velocity (noise minus image) read as its ``flow_prediction`` (the image is ``sample - sigma * velocity``).

    UniPC shifts a grid it is given by its ``flow_shift``, and MOSS-Image's is shifted already (by the sampling shift,
    then snapped to the training grid): it steps it at ``flow_shift`` 1, and its config keeps ``euler``'s sampling shift
    as ``shift``, which :class:`MossSetTimestepsStep` draws the grid at, as from the Euler sampler's config."""
    unipc = UniPCMultistepScheduler(
        num_train_timesteps=euler.config.num_train_timesteps,
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=1.0,
    )
    unipc.register_to_config(shift=euler.config.shift)
    return unipc


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
        ValueError: From the call, if a side is not a positive multiple of 16 px.
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
    sampling shift, 5 by default, as MOSS-Image's own scheduler config names it (and a UniPC sampler's keeps it:
    :func:`unipc_sampler`). The grid is shifted and on the training grid already, so the scheduler steps it as given,
    at shift 1, and the DiT is called at its levels, sigma x 1,000, whichever the sampler: UniPC's own timesteps are
    those cut to whole steps (some a step below their level), and it starts a millionth below sigma 1, which keeps its
    log-SNR finite.

    Raises:
        ValueError: From the call, if a UniPC sampler would shift the grid again (its ``flow_shift`` is not 1).
    """

    model_name = "moss"

    @property
    @override
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "scheduler",
                FlowMatchEulerDiscreteScheduler,  # MOSS-Image's own; diffu_moss.inference swaps in UniPC when asked
                config=FrozenDict({"num_train_timesteps": TRAIN_STEPS, "shift": SAMPLING_SHIFT}),
                default_creation_method="from_config",
            )
        ]

    @property
    @override
    def description(self) -> str:
        return (
            "Step that sets the scheduler's timesteps to MOSS-Image's sampling grid: the steps shifted by the "
            "scheduler's shift, then each moved to the nearest level of the 1,000-step training grid. The Euler "
            "sampler or UniPC steps it as given; the DiT is called at its levels."
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
        scheduler: FlowMatchEulerDiscreteScheduler | UniPCMultistepScheduler = components.scheduler
        sigmas = sampling_sigmas(block_state.num_inference_steps, scheduler.config.shift)
        device = components._execution_device
        if isinstance(scheduler, UniPCMultistepScheduler):
            if scheduler.config.flow_shift != 1.0:
                raise ValueError(
                    f"a UniPC sampler at flow_shift {scheduler.config.flow_shift} would shift MOSS-Image's grid again: "
                    "give it flow_shift 1 (diffu_moss.modular_pipelines.moss.before_denoise.unipc_sampler)"
                )
            scheduler.set_timesteps(sigmas=np.array(sigmas), device=device)  # an array: it multiplies them by its shift
            levels = torch.tensor(sigmas, dtype=torch.float32, device=device)
            timesteps = levels * scheduler.config.num_train_timesteps  # as the Euler sampler's own
        else:
            scheduler.set_shift(1.0)
            scheduler.set_timesteps(sigmas=sigmas, device=device)
            timesteps = scheduler.timesteps
        scheduler.set_begin_index(0)
        block_state.timesteps = timesteps
        block_state.num_inference_steps = len(timesteps)
        self.set_block_state(state, block_state)
        return components, state
