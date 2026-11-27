"""Step caches for MOSS-Image's DiT, from diffusers' cache hooks (``diffusers.hooks``): from step to step of a page,
what the 42 joint blocks add changes less and less, so at some steps they are skipped and what earlier steps computed
stands in. ``DrawSettings.cache`` names one (:data:`StepCache`); :func:`enable_step_cache` sets it on the DiT.

- :class:`FirstBlockCache`: every step runs the first joint block; when its residual (output minus input) moved less
  than ``threshold`` from the previous step's (their mean absolute difference over its mean absolute value), the other
  blocks are skipped and the residual they added when they last ran is added again. As Cache-DiT bounds it, it may
  also skip none of a pass's first ``warmup_steps`` steps and compute the step after ``max_skipped_in_a_row`` skipped
  ones (:class:`_BoundedHeadBlockHook`); both off, it is diffusers' own.
- :class:`TaylorSeerCache`: the DiT runs whole at its first ``disable_cache_before_step`` steps, then every
  ``cache_interval`` steps, and at every step from ``disable_cache_after_step`` on; at the other steps its joint blocks
  are skipped and its output projection's output is extrapolated from its last whole runs (a Taylor series of order
  ``max_order``).
- :class:`MagCache`: past the first ``retention_ratio`` of the steps, the joint blocks are skipped while the error
  accumulated from how the residual's magnitude changes from step to step (``mag_ratios``) stays under ``threshold``,
  for at most ``max_skip_steps`` steps in a row, the residual they added last added again. The ratios belong to a
  model, a sampler and a number of steps: pages drawn with :class:`MagCacheCalibration` measure them.

Each pass of guidance keeps its own state (the denoising loop runs each in the DiT's ``cache_context``, named as the
guider names the pass), cleared as each page starts (:func:`reset_step_cache`), and a call draws one sample. The hooks
decide eagerly (``torch.compiler.disable``; FirstBlockCache reads its measure back from the GPU), around each joint
block's forward: the blocks run eagerly, from an AOTInductor package, or with their own forward compiled beneath the
hooks (:func:`compile_beneath_step_cache`), never with their whole call compiled, which would trace the hooks and break
its graph where they decide.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path
from typing import Annotated, Any, Literal, override

import torch
from diffusers.hooks import FirstBlockCacheConfig, HookRegistry, MagCacheConfig, ModelHook, TaylorSeerCacheConfig
from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from diffusers.hooks.first_block_cache import FBCHeadBlockHook
from diffusers.hooks.hooks import BaseState, StateManager
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from diffu_moss.transformer import JointBlock, MossTransformer

# How a joint block takes and gives its tokens, for diffusers' block caches: as hidden_states, returned alone
JOINT_BLOCK = TransformerBlockMetadata(return_hidden_states_index=0, return_encoder_hidden_states_index=None)
SKIPPED_BLOCKS = r"layers\.\d+"  # TaylorSeer: the joint blocks, skipped at a step it extrapolates
EXTRAPOLATED = r"final_linear_video"  # TaylorSeer: the output projection, whose output it extrapolates
MAG_EPSILON = 1e-8  # MagCache's: what a residual of no magnitude is divided by
FIRST_BLOCK_INPUT = "moss_mag_calibration_input"  # the calibration's hooks, by name in their blocks' registries
LAST_BLOCK_RATIO = "moss_mag_calibration_ratio"
FBC_HEAD_BLOCK = "fbc_leader_block_hook"  # diffusers' FirstBlockCache hook on the first block, by name in its registry


class FirstBlockCache(BaseModel):
    """diffusers' FirstBlockCache (the module's docstring).

    Attributes:
        kind: ``first_block``.
        threshold: How little the first block's residual must move from the previous step's (their mean absolute
            difference over its mean absolute value) for the other blocks to be skipped: higher skips more. diffusers'
            default, 0.05.
        warmup_steps: How many first steps of each pass of guidance are computed whole, whatever the measure says
            (Cache-DiT's ``max_warmup_steps``, 4 by its default); 0, diffusers' own, skips from the second step on.
        max_skipped_in_a_row: After this many skipped steps in a row a pass's next step is computed whole, whatever
            the measure says (Cache-DiT's ``max_continuous_cached_steps``, 3 by its default); ``None``, diffusers'
            own, sets no bound.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["first_block"] = "first_block"
    threshold: float = Field(default=0.05, gt=0.0)
    warmup_steps: int = Field(default=0, ge=0)
    max_skipped_in_a_row: int | None = Field(default=None, ge=1)

    @property
    def bounded(self) -> bool:
        """Whether warm-up steps or a bound on skips in a row are asked: diffusers' own cache has neither."""
        return self.warmup_steps > 0 or self.max_skipped_in_a_row is not None


class TaylorSeerCache(BaseModel):
    """diffusers' TaylorSeer cache (the module's docstring), its joint blocks skipped and its output projection
    extrapolated at the steps it does not compute (:data:`SKIPPED_BLOCKS`, :data:`EXTRAPOLATED`), in float32.

    Attributes:
        kind: ``taylorseer``.
        cache_interval: Every how many steps the DiT runs whole, past the first ones.
        disable_cache_before_step: How many first steps run whole (at least one: the first extrapolates from it).
        disable_cache_after_step: The step from which every step runs whole again; ``None``: none.
        max_order: The Taylor series' order: 0 repeats the last whole run's output, 1 extrapolates its change too.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["taylorseer"] = "taylorseer"
    cache_interval: int = Field(default=5, ge=1)
    disable_cache_before_step: int = Field(default=3, ge=1)
    disable_cache_after_step: int | None = Field(default=None, ge=0)
    max_order: int = Field(default=1, ge=0)


class MagCache(BaseModel):
    """diffusers' MagCache (the module's docstring).

    Attributes:
        kind: ``magcache``.
        mag_ratios: How the residual's magnitude changes from each step to the next, one ratio a step: measured for
            this model, sampler and number of steps by :class:`MagCacheCalibration` (:func:`calibrated_mag_ratios`),
            and spread over another number of steps if given one.
        threshold: The accumulated error under which the joint blocks are skipped: higher skips more.
        max_skip_steps: The most steps skipped in a row.
        retention_ratio: The share of the first steps that always run whole.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["magcache"] = "magcache"
    mag_ratios: tuple[Annotated[float, Field(gt=0.0)], ...] = Field(min_length=1)
    threshold: float = Field(default=0.06, gt=0.0)
    max_skip_steps: int = Field(default=3, ge=1)
    retention_ratio: float = Field(default=0.2, ge=0.0, le=1.0)


class MagCacheCalibration(BaseModel):
    """Pages drawn whole that measure MagCache's ratios as diffusers' MagCache calibration does (``calibrate``): at
    every step of every pass, the mean over the tokens of how much larger the residual of the joint blocks (the last
    block's output minus the first block's input) is than the step before's (1 at the first step). Each pass's ratios
    are printed, a JSON line ``{"pass": ..., "mag_ratios": [...]}`` once it has run every step of a page, and appended
    to ``ratios_file`` if one is named; :func:`calibrated_mag_ratios` reads them back.

    Attributes:
        kind: ``magcache_calibration``.
        ratios_file: The JSON-lines file each pass's ratios are appended to.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["magcache_calibration"] = "magcache_calibration"
    ratios_file: Path | None = None


type StepCache = Annotated[
    FirstBlockCache | TaylorSeerCache | MagCache | MagCacheCalibration, Field(discriminator="kind")
]


def enable_step_cache(
    transformer: MossTransformer, cache: StepCache, *, steps: int, skip_layers: Collection[int]
) -> None:
    """``cache`` set on ``transformer``'s joint blocks (after any AOTInductor package: its hooks wrap whatever forward
    a block has), for pages of ``steps`` denoising steps (what MagCache spreads its ratios over, and what a calibration
    reports after), whose passes of guidance skip the joint blocks ``skip_layers``.

    Raises:
        ValueError: If a pass skips the first or the last joint block, which FirstBlockCache and MagCache run in every
            pass, to decide and to measure.
    """
    if not isinstance(cache, TaylorSeerCache):
        _check_end_blocks(transformer, skip_layers)
    match cache:
        case FirstBlockCache():
            TransformerBlockRegistry.register(model_class=JointBlock, metadata=JOINT_BLOCK)
            transformer.enable_cache(FirstBlockCacheConfig(threshold=cache.threshold))
            if cache.bounded:
                _bound_first_block_cache(transformer, cache)
        case TaylorSeerCache():
            config = TaylorSeerCacheConfig(
                cache_interval=cache.cache_interval,
                disable_cache_before_step=cache.disable_cache_before_step,
                disable_cache_after_step=cache.disable_cache_after_step,
                max_order=cache.max_order,
                taylor_factors_dtype=torch.float32,  # the output projection's: a small tensor
                skip_predict_identifiers=[SKIPPED_BLOCKS],
                cache_identifiers=[EXTRAPOLATED],
            )
            transformer.enable_cache(config)
        case MagCache():
            TransformerBlockRegistry.register(model_class=JointBlock, metadata=JOINT_BLOCK)
            config = MagCacheConfig(
                threshold=cache.threshold,
                max_skip_steps=cache.max_skip_steps,
                retention_ratio=cache.retention_ratio,
                num_inference_steps=steps,
                mag_ratios=list(cache.mag_ratios),
            )
            transformer.enable_cache(config)
        case MagCacheCalibration():
            _record_mag_ratios(transformer, steps=steps, ratios_file=cache.ratios_file)


def compile_beneath_step_cache(transformer: MossTransformer) -> None:
    """Each of ``transformer``'s joint blocks with its own forward compiled (``torch.compile``, whole graph, dynamic
    shapes: page sizes and prompt lengths vary), for a step cache whose hooks are set after: they wrap the compiled
    forward and run eagerly around it. ``compile_repeated_blocks`` compiles each block's whole call instead, which
    would trace the hooks too and break at their eager decision (``torch.compiler.disable``). The blocks share one
    compiled graph, their weights its inputs."""
    for block in transformer.layers:
        block.forward = torch.compile(block.forward, fullgraph=True, dynamic=True)


def reset_step_cache(transformer: MossTransformer) -> None:
    """Every step cache state ``transformer`` holds cleared (each pass's, kept from step to step), as a page starts: a
    previous page's, of another size, would break the next. Without a step cache there is none."""
    HookRegistry.check_if_exists_or_initialize(transformer).reset_stateful_hooks()


def calibrated_mag_ratios(path: Path) -> tuple[float, ...]:
    """MagCache's ratios from a calibration's ``ratios_file`` (:class:`MagCacheCalibration`), to draw with: each
    step's mean over every pass and page it holds.

    Raises:
        ValueError: If it holds no ratios, or ratios of different numbers of steps.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    ratios = [json.loads(line)["mag_ratios"] for line in lines if line.strip()]
    if not ratios or len({len(steps) for steps in ratios}) != 1:
        raise ValueError(f"{path} holds no calibration's ratios, or ratios of different numbers of steps")
    return tuple(torch.tensor(ratios, dtype=torch.float64).mean(0).tolist())


def magnitude_ratio(residual: Tensor, previous: Tensor) -> float:
    """MagCache's ratio of a step's residual to the previous step's: the mean over the tokens of their norms' ratio,
    in float32, as diffusers' calibration takes it."""
    norms = torch.linalg.norm(residual.float(), dim=-1)
    return (norms / (torch.linalg.norm(previous.float(), dim=-1) + MAG_EPSILON)).mean().item()


def _check_end_blocks(transformer: MossTransformer, skip_layers: Collection[int]) -> None:
    ends = sorted({0, len(transformer.layers) - 1} & set(skip_layers))
    if ends:
        raise ValueError(
            f"a pass of guidance skips the joint blocks {ends}: FirstBlockCache and MagCache run the first and the last "
            "block in every pass"
        )


class _PassMagnitudes(BaseState):
    """One pass's calibration on the page being drawn: the first joint block's input at this step, the residual at
    the step before and the ratios so far."""

    def __init__(self) -> None:
        self.first_input: Tensor | None = None
        self.previous: Tensor | None = None
        self.ratios: list[float] = []

    @override
    def reset(self) -> None:
        self.first_input = self.previous = None
        self.ratios = []


class _FirstBlockInput(ModelHook):
    """Keeps the first joint block's input, the pass's; clears every pass's calibration as a page starts."""

    _is_stateful = True

    def __init__(self, passes: StateManager) -> None:
        super().__init__()
        self.passes = passes

    @override
    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        self.passes.get_state().first_input = args[0] if args else kwargs["hidden_states"]
        return args, kwargs

    @override
    def reset_state(self, module: nn.Module) -> nn.Module:
        self.passes.reset()
        return module


class _LastBlockRatio(ModelHook):
    """After the last joint block, the pass's ratio at this step (:func:`magnitude_ratio`); a pass's ratios printed,
    and appended to ``ratios_file``, once it has run ``steps`` steps."""

    def __init__(self, passes: StateManager, *, steps: int, ratios_file: Path | None) -> None:
        super().__init__()
        self.passes = passes
        self.steps = steps
        self.ratios_file = ratios_file

    @override
    def post_forward(self, module: nn.Module, output: Tensor) -> Tensor:
        state: _PassMagnitudes = self.passes.get_state()
        if state.first_input is None:
            raise RuntimeError("the last joint block ran before the first in this pass")
        residual = output - state.first_input
        state.ratios.append(1.0 if state.previous is None else magnitude_ratio(residual, state.previous))
        state.previous = residual
        if len(state.ratios) == self.steps:
            line = json.dumps({"pass": self.passes.context.name, "mag_ratios": state.ratios})
            print(line, flush=True)  # machine output: what diffusers' calibration prints, by pass
            if self.ratios_file is not None:
                with self.ratios_file.open("a", encoding="utf-8") as lines:
                    lines.write(line + "\n")
        return output


def _record_mag_ratios(transformer: MossTransformer, *, steps: int, ratios_file: Path | None) -> None:
    """The calibration's two hooks set on ``transformer``'s first and last joint blocks, sharing each pass's state."""
    passes = StateManager(_PassMagnitudes)
    first, last = transformer.layers[0], transformer.layers[-1]
    HookRegistry.check_if_exists_or_initialize(first).register_hook(_FirstBlockInput(passes), FIRST_BLOCK_INPUT)
    ratio = _LastBlockRatio(passes, steps=steps, ratios_file=ratios_file)
    HookRegistry.check_if_exists_or_initialize(last).register_hook(ratio, LAST_BLOCK_RATIO)
    HookRegistry.check_if_exists_or_initialize(
        transformer
    ).invalidate_child_registries_cache()  # cache_context finds them


class _PassSkips(BaseState):
    """One pass's count, on the page being drawn, of the steps its first joint block has run and of the steps skipped
    in a row since the last computed one."""

    def __init__(self) -> None:
        self.steps = 0
        self.skipped_in_a_row = 0

    @override
    def reset(self) -> None:
        self.steps = self.skipped_in_a_row = 0


class _BoundedHeadBlockHook(FBCHeadBlockHook):
    """diffusers' FirstBlockCache hook on the first joint block, its decision bounded as Cache-DiT bounds it: a pass's
    first ``warmup_steps`` steps are computed, and so is the step after ``max_skipped_in_a_row`` skipped ones, without
    measuring; otherwise the measure decides as in diffusers. Each pass counts its own steps (:class:`_PassSkips`, in a
    state manager of its own that ``cache_context`` names as it names the shared one), cleared as a page starts."""

    def __init__(
        self, state_manager: StateManager, threshold: float, *, warmup_steps: int, max_skipped_in_a_row: int | None
    ) -> None:
        super().__init__(state_manager, threshold)
        self.passes = StateManager(_PassSkips)
        self.warmup_steps = warmup_steps
        self.max_skipped_in_a_row = max_skipped_in_a_row

    @torch.compiler.disable
    @override
    def _should_compute_remaining_blocks(self, hidden_states_residual: Tensor) -> bool:
        counts: _PassSkips = self.passes.get_state()
        warming_up = counts.steps < self.warmup_steps
        capped = self.max_skipped_in_a_row is not None and counts.skipped_in_a_row >= self.max_skipped_in_a_row
        compute = warming_up or capped or super()._should_compute_remaining_blocks(hidden_states_residual)
        counts.steps += 1
        counts.skipped_in_a_row = 0 if compute else counts.skipped_in_a_row + 1
        return compute

    @override
    def reset_state(self, module: nn.Module) -> nn.Module:
        self.passes.reset()
        return super().reset_state(module)


def _bound_first_block_cache(transformer: MossTransformer, cache: FirstBlockCache) -> None:
    """diffusers' FirstBlockCache hook on ``transformer``'s first joint block replaced by :class:`_BoundedHeadBlockHook`,
    which shares its state with the other blocks' hooks; the other blocks' hooks, and ``disable_cache``, are left as
    diffusers set them."""
    registry = HookRegistry.check_if_exists_or_initialize(transformer.layers[0])
    head = registry.get_hook(FBC_HEAD_BLOCK)
    if head is None:
        raise RuntimeError("diffusers' FirstBlockCache set no hook on the first joint block")
    registry.remove_hook(FBC_HEAD_BLOCK, recurse=False)
    bounded = _BoundedHeadBlockHook(
        head.state_manager,
        cache.threshold,
        warmup_steps=cache.warmup_steps,
        max_skipped_in_a_row=cache.max_skipped_in_a_row,
    )
    registry.register_hook(bounded, FBC_HEAD_BLOCK)
    HookRegistry.check_if_exists_or_initialize(transformer).invalidate_child_registries_cache()
