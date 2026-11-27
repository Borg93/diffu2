"""The step cache's bounds and its compiled blocks (diffu_moss.step_cache) on CPU with conftest's tiny MOSS, drawn as
the benchmark draws (diffu_moss.bench_inference: prepare_inference, then the pipeline's own denoising loop, two passes
of guidance a step, the unconditional one skipping block 1 of 3): FirstBlockCache skips none of a pass's first
``warmup_steps`` steps and computes the step after ``max_skipped_in_a_row`` skipped ones, each pass counting its own;
both off, it is diffusers' own hook, and bounds that never bind draw exactly as it; and with ``compile_blocks`` each
block's own forward is compiled beneath the hooks, which draws as the eager blocks do under the same cache."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from diffusers.hooks import HookRegistry
from diffusers.hooks.first_block_cache import _FBC_BLOCK_HOOK as FBC_BLOCK
from diffusers.hooks.first_block_cache import FBCHeadBlockHook
from torch import nn
from torch._dynamo.utils import counters

from diffu_moss.bench_inference import BenchSettings, draw, load
from diffu_moss.inference import MossInference
from diffu_moss.modular_pipelines.moss import MossModularPipeline
from diffu_moss.step_cache import FBC_HEAD_BLOCK, FirstBlockCache, _BoundedHeadBlockHook

CPU = torch.device("cpu")
EVERY_STEP_SKIPS = 10.0  # a threshold past which, after a pass's first step, the first block's measure always skips
MIXED = 0.6  # a threshold at which the tiny MOSS skips some steps and computes others (its measure: 0.49 to 0.81)


def _settings(model: Path, steps: int) -> BenchSettings:
    """A 64 x 32 page of ``steps`` steps from noise seed 0."""
    return BenchSettings(model=model, device="cpu", size=(64, 32), prompt="A handwritten page.", steps=steps, seed=0)


def _count_computations(pipe: MossModularPipeline) -> list[int]:
    """How many times each joint block of ``pipe``'s DiT computes from now on, by its own forward (its q projection
    runs then alone): a list, by block, that counts as the pipeline draws."""
    counts = [0] * len(pipe.transformer.layers)

    def counter(index: int) -> Callable[[nn.Module, object, object], None]:
        def count(module: nn.Module, args: object, output: object) -> None:
            counts[index] += 1

        return count

    for index, block in enumerate(pipe.transformer.layers):
        block.proj["q"].register_forward_hook(counter(index))
    return counts


def _count_cached_forwards(pipe: MossModularPipeline) -> list[int]:
    """How many times each joint block of ``pipe``'s DiT runs the forward its FirstBlockCache hook wraps (its own, or
    its compiled one) from now on: a list, by block, that counts as the pipeline draws."""
    counts = [0] * len(pipe.transformer.layers)

    def counted(index: int, forward: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        def run(*args: object, **kwargs: object) -> torch.Tensor:
            counts[index] += 1
            return forward(*args, **kwargs)

        return run

    for index, block in enumerate(pipe.transformer.layers):
        registry = HookRegistry.check_if_exists_or_initialize(block)
        hook = registry.get_hook(FBC_HEAD_BLOCK if index == 0 else FBC_BLOCK)
        hook.fn_ref.original_forward = counted(index, hook.fn_ref.original_forward)
    return counts


@pytest.mark.parametrize(
    ("cache", "steps", "computed"),
    [  # each block's computations: block 0 at every step of both passes, block 1 in the conditional pass alone
        pytest.param(FirstBlockCache(threshold=EVERY_STEP_SKIPS), 6, [12, 1, 2], id="unbounded"),
        pytest.param(FirstBlockCache(threshold=EVERY_STEP_SKIPS, warmup_steps=4), 6, [12, 4, 8], id="warmup-4"),
        pytest.param(FirstBlockCache(threshold=EVERY_STEP_SKIPS, warmup_steps=6), 6, [12, 6, 12], id="warmup-all"),
    ],
)
def test_first_block_cache_skips_none_of_each_passs_warmup_steps(
    tiny_moss: Path, cache: FirstBlockCache, steps: int, computed: list[int]
) -> None:
    pipe = load(tiny_moss, MossInference(cache=cache), CPU, steps=steps)
    counts = _count_computations(pipe)

    draw(pipe, _settings(tiny_moss, steps))

    assert counts == computed


@pytest.mark.parametrize(
    ("cache", "computed"),
    [  # eight steps a pass; computed at the steps listed in each id, in both passes
        pytest.param(FirstBlockCache(threshold=EVERY_STEP_SKIPS, max_skipped_in_a_row=3), [16, 2, 4], id="0-4"),
        pytest.param(FirstBlockCache(threshold=EVERY_STEP_SKIPS, max_skipped_in_a_row=2), [16, 3, 6], id="0-3-6"),
        pytest.param(FirstBlockCache(threshold=EVERY_STEP_SKIPS, max_skipped_in_a_row=1), [16, 4, 8], id="0-2-4-6"),
        pytest.param(  # Cache-DiT's defaults, on eight steps
            FirstBlockCache(threshold=EVERY_STEP_SKIPS, warmup_steps=4, max_skipped_in_a_row=3),
            [16, 5, 10],
            id="0-1-2-3-7",
        ),
        pytest.param(
            FirstBlockCache(threshold=EVERY_STEP_SKIPS, warmup_steps=2, max_skipped_in_a_row=2),
            [16, 4, 8],
            id="0-1-4-7",
        ),
    ],
)
def test_first_block_cache_computes_the_step_after_max_skipped_in_a_row_skipped_ones(
    tiny_moss: Path, cache: FirstBlockCache, computed: list[int]
) -> None:
    pipe = load(tiny_moss, MossInference(cache=cache), CPU, steps=8)
    counts = _count_computations(pipe)

    draw(pipe, _settings(tiny_moss, 8))

    assert counts == computed


def test_first_block_cache_bounds_off_set_diffusers_own_hook_and_unbinding_bounds_draw_exactly_as_it(
    tiny_moss: Path,
) -> None:
    settings = _settings(tiny_moss, 8)
    plain = load(tiny_moss, MossInference(cache=FirstBlockCache(threshold=MIXED)), CPU, steps=8)
    plain_counts = _count_computations(plain)
    unbinding = FirstBlockCache(threshold=MIXED, warmup_steps=1, max_skipped_in_a_row=8)  # step 0 computes anyway
    bounded = load(tiny_moss, MossInference(cache=unbinding), CPU, steps=8)
    bounded_counts = _count_computations(bounded)

    drawn, bounded_drawn = draw(plain, settings), draw(bounded, settings)

    head = HookRegistry.check_if_exists_or_initialize(plain.transformer.layers[0]).get_hook(FBC_HEAD_BLOCK)
    assert type(head) is FBCHeadBlockHook
    assert 0 < sum(plain_counts[1:]) < 8 * 3  # the measure skipped some steps, not all
    assert bounded_counts == plain_counts
    assert torch.equal(bounded_drawn.latents, drawn.latents)


def test_first_block_cache_with_compile_blocks_compiles_each_blocks_forward_beneath_its_hooks_drawing_as_eagerly(
    tiny_moss: Path,
) -> None:
    cache = FirstBlockCache(threshold=MIXED, warmup_steps=2, max_skipped_in_a_row=2)
    settings = _settings(tiny_moss, 8)
    eager = load(tiny_moss, MossInference(cache=cache), CPU, steps=8)
    eager_counts = _count_cached_forwards(eager)
    eager_drawn = draw(eager, settings)
    torch._dynamo.reset()
    counters.clear()
    compiled = load(tiny_moss, MossInference(cache=cache, compile_blocks=True), CPU, steps=8)
    compiled_counts = _count_cached_forwards(compiled)

    compiled_drawn = draw(compiled, settings)

    head = HookRegistry.check_if_exists_or_initialize(compiled.transformer.layers[0]).get_hook(FBC_HEAD_BLOCK)
    assert isinstance(head, _BoundedHeadBlockHook)
    assert counters["stats"]["unique_graphs"] == 1  # one whole graph for every block, the hooks outside it
    assert not counters["graph_break"]
    assert 0 < sum(eager_counts[1:]) < 8 * 3  # the cache skipped some steps, and the compiled blocks skip the same
    assert compiled_counts == eager_counts
    cosine = nn.functional.cosine_similarity(compiled_drawn.latents.flatten(), eager_drawn.latents.flatten(), dim=0)
    assert cosine.item() > 0.999  # bfloat16 blocks: compiled, they round otherwise
