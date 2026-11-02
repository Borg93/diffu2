"""MOSS-Image's inference options through the page pipeline on CPU (diffu_moss.inference, by DrawSettings alone): every
option off, the tiny MOSS draws a crop byte for byte as main drew it before them (the DiT reading the whole prompt
unmasked, its rotary frequencies copied to the device once); MXFP8 in the DiT or the text encoder (emulated on the CPU)
draws close to bfloat16 but not as it; the joint blocks run from an AOTInductor package built by diffu-moss-aoti, or
compiled with MXFP8 off and on, close to as they run eagerly; UniPC steps the snapped grid it is given, calling the DiT
at its levels once a pass a step; each step cache computes the joint blocks only at the steps it does not cache, starts
every page afresh, and wraps a package's blocks too; MagCache's calibration measures what diffusers' does, pass by pass;
and what only MOSS-Image takes is refused for another model, as are MXFP8 on a GPU without it, a package or a step cache
beside compile_blocks, and a block cache for a model whose unconditional pass skips its first joint block."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from diffusers.hooks import MagCacheConfig
from diffusers.hooks._helpers import TransformerBlockRegistry
from diffusers.modular_pipelines import ModularPipeline
from pydantic import ValidationError
from torch import nn
from transformers import Qwen3_5Config, Qwen3_5Model
from typer.testing import CliRunner

from diffu_core.repo import ModelIndex
from diffu_moss import aoti
from diffu_moss.init import write_moss_repo
from diffu_moss.modular_pipelines.moss import MossModularPipeline
from diffu_moss.modular_pipelines.moss.before_denoise import Sampler
from diffu_moss.mxfp8 import Precision
from diffu_moss.step_cache import (
    JOINT_BLOCK,
    FirstBlockCache,
    MagCache,
    MagCacheCalibration,
    StepCache,
    TaylorSeerCache,
    calibrated_mag_ratios,
)
from diffu_moss.transformer import JointBlock, MossTransformer
from diffu_page.captions.prompt import PageStyle
from diffu_page.draw.pipeline import DrawSettings, crop_blocks, load_pipeline
from diffu_page.draw.plan import CropPlan, plan_whole_page
from diffu_page.page.layout import load_xml_layout

SETTINGS = DrawSettings(steps=2, device="cpu")
# The tiny MOSS's crop of small_plan from seed 0 as main drew it before these options (27c9fc9, on the CPU): the
# SHA-256 of its RGB bytes, the same at 1, 2 and 4 threads
MAIN_DRAWING_SHA256 = "8f19b95b784939f35b2fa415485d23a69e410ec05953c6cdf74f9321a87c8d75"
CACHED_STEPS = 6  # a step cache's tests: six steps of two passes, the unconditional one skipping block 1 of 3
FIRST_BLOCK = FirstBlockCache(threshold=10.0)  # past the first step, only the first block runs
TAYLORSEER = TaylorSeerCache(cache_interval=2, disable_cache_before_step=1)  # whole at steps 0, 2 and 4
MAGCACHE = MagCache(mag_ratios=(1.0,) * CACHED_STEPS, retention_ratio=0.0)  # whole at 0 and 4: 3 skipped at most
# Qwen3.5 and the DiT as conftest's tiny MOSS has them, but 32 wide, so that every layer MXFP8 takes fits its blocks
WIDE_QWEN3_5 = {
    "text_config": {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
    },
    "vision_config": {
        "depth": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_heads": 2,
        "out_hidden_size": 32,
        "num_position_embeddings": 64,
    },
}
WIDE_MOSS_DIT = {
    "dim": 32,
    "ffn_dim": 64,
    "num_heads": 2,
    "num_layers": 3,
    "text_dim": 32,
    "freq_dim": 16,
    "latent_channels": 16,
}


@pytest.fixture(scope="module")
def wide_moss(tiny_moss: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny MOSS with a text encoder and a DiT 32 wide: its tokenizer and VAE, random weights."""
    tiny_base = Path(ModelIndex.read(tiny_moss).source("tokenizer").repo)
    base = tmp_path_factory.mktemp("wide_moss_base")
    for folder in ("tokenizer", "vae"):
        shutil.copytree(tiny_base / folder, base / folder)
    torch.manual_seed(0)
    Qwen3_5Model(Qwen3_5Config(**WIDE_QWEN3_5)).save_pretrained(base / "text_encoder")
    MossTransformer(**WIDE_MOSS_DIT).save_pretrained(base / "transformer")
    repo = tmp_path_factory.mktemp("wide_moss")
    write_moss_repo(repo, base=base, uncond_skip_layers=(1,))
    return repo


@pytest.fixture
def small_plan(two_column_page: Path, page_style: PageStyle) -> CropPlan:
    """The two-column page planned whole at 64 x 32 px: six lines under a layout caption."""
    return plan_whole_page(load_xml_layout(two_column_page), page_style, pixels=64 * 32)


def _drawn(pipe: ModularPipeline, plan: CropPlan, output: str, settings: DrawSettings = SETTINGS) -> Any:
    """``plan`` drawn from noise seed 0 in ``settings.steps`` steps, its ``output`` (``images``, ``latents``,
    ``prompt_embeds``)."""
    pipe.guider.set_state(step=0, num_inference_steps=settings.steps, timestep=None)
    return pipe(crop_plan=plan, generator=torch.Generator().manual_seed(0), **settings.pipeline_inputs(), output=output)


def _tensor(pipe: ModularPipeline, plan: CropPlan, output: str, settings: DrawSettings = SETTINGS) -> torch.Tensor:
    """``plan``'s ``output`` (``latents`` or ``prompt_embeds``) drawn by :func:`_drawn`, in float32."""
    return _drawn(pipe, plan, output, settings).float()


def _cached(cache: StepCache) -> DrawSettings:
    """Settings that draw in :data:`CACHED_STEPS` steps with the step cache ``cache``."""
    return DrawSettings(steps=CACHED_STEPS, device="cpu", cache=cache)


def _count_computations(pipe: ModularPipeline) -> list[int]:
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


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def test_moss_draws_byte_for_byte_as_main_did_when_every_inference_option_is_off(
    tiny_moss: Path, small_plan: CropPlan
) -> None:
    pipe = load_pipeline(tiny_moss, crop_blocks, loras={}, settings=SETTINGS)

    drawn = _drawn(pipe, small_plan, "images")[0]

    assert hashlib.sha256(drawn.tobytes()).hexdigest() == MAIN_DRAWING_SHA256


@pytest.mark.parametrize(
    ("option", "output"),
    [
        pytest.param({"precision": Precision.MXFP8}, "latents", id="dit"),
        pytest.param({"text_encoder_precision": Precision.MXFP8}, "prompt_embeds", id="text-encoder"),
    ],
)
def test_moss_in_mxfp8_computes_close_to_bf16_but_not_as_it(
    wide_moss: Path, small_plan: CropPlan, option: dict[str, Precision], output: str
) -> None:
    bf16 = _tensor(load_pipeline(wide_moss, crop_blocks, loras={}, settings=SETTINGS), small_plan, output)
    settings = SETTINGS.model_copy(update=option)

    mxfp8 = _tensor(load_pipeline(wide_moss, crop_blocks, loras={}, settings=settings), small_plan, output)

    assert not torch.equal(mxfp8, bf16)
    assert _cosine(mxfp8, bf16) > 0.999


@pytest.fixture(scope="module")
def cli_package(tiny_moss: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny MOSS's joint block compiled for the CPU in its own dtype by diffu-moss-aoti, tuned for 24 tokens."""
    package = tmp_path_factory.mktemp("package") / "block.pt2"
    options = ["--model", str(tiny_moss), "--out", str(package), "--device", "cpu", "--tokens", "24"]
    result = CliRunner().invoke(aoti.app, options)
    assert result.exit_code == 0, result.output
    return package


def test_moss_draws_from_its_aoti_package_without_its_blocks_own_forward_close_to_its_eager_drawing(
    tiny_moss: Path, small_plan: CropPlan, cli_package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eager = _tensor(load_pipeline(tiny_moss, crop_blocks, loras={}, settings=SETTINGS), small_plan, "latents")
    pipe = load_pipeline(
        tiny_moss, crop_blocks, loras={}, settings=SETTINGS.model_copy(update={"aoti_package": cli_package})
    )
    monkeypatch.setattr(JointBlock, "forward", _own_forward_called)

    packaged = _tensor(pipe, small_plan, "latents")

    assert _cosine(packaged, eager) > 0.999


def _own_forward_called(*args: object) -> torch.Tensor:
    raise AssertionError("a joint block ran its own forward, not its package's")


@pytest.mark.parametrize("precision", list(Precision))
def test_moss_draws_with_exactly_its_joint_blocks_compiled_close_to_its_eager_drawing(
    tiny_moss: Path, small_plan: CropPlan, precision: Precision
) -> None:
    settings = SETTINGS.model_copy(update={"precision": precision})
    eager = _tensor(load_pipeline(tiny_moss, crop_blocks, loras={}, settings=settings), small_plan, "latents")
    pipe = load_pipeline(
        tiny_moss, crop_blocks, loras={}, settings=settings.model_copy(update={"compile_blocks": True})
    )

    compiled = _tensor(pipe, small_plan, "latents")

    compiled_modules = {
        type(module).__name__
        for module in pipe.transformer.modules()
        if getattr(module, "_compiled_call_impl", None) is not None  # what nn.Module.compile sets
    }
    assert compiled_modules == {"JointBlock"}
    assert _cosine(compiled, eager) > 0.999


def test_moss_draws_with_unipc_at_its_grids_levels_one_dit_call_a_pass_a_step(
    tiny_moss: Path, small_plan: CropPlan
) -> None:
    euler_settings = SETTINGS.model_copy(update={"steps": 6})
    euler = load_pipeline(tiny_moss, crop_blocks, loras={}, settings=euler_settings)
    _drawn(euler, small_plan, "latents", euler_settings)
    settings = euler_settings.model_copy(update={"sampler": Sampler.UNIPC})
    pipe = load_pipeline(tiny_moss, crop_blocks, loras={}, settings=settings)
    called_at: list[float] = []
    pipe.transformer.register_forward_pre_hook(lambda module, args: called_at.append(float(args[1])))

    _drawn(pipe, small_plan, "latents", settings)

    levels = euler.scheduler.sigmas.clone()  # the snapped grid's six levels, then 0
    levels[0] = 1 - 1e-6  # where UniPC starts instead of 1, to keep its log-SNR finite
    assert pipe.scheduler.sigmas.tolist() == levels.tolist()
    assert called_at == [timestep for timestep in euler.scheduler.timesteps.tolist() for _ in ("cond", "uncond")]


@pytest.mark.parametrize(
    ("cache", "computed"),
    [  # each block's computations in six steps of two passes; the unconditional pass skips block 1
        pytest.param(FIRST_BLOCK, [12, 1, 2], id="first-block"),
        pytest.param(TAYLORSEER, [6, 3, 6], id="taylorseer"),
        pytest.param(MAGCACHE, [4, 2, 4], id="magcache"),
    ],
)
def test_moss_step_cache_computes_the_joint_blocks_only_at_the_steps_it_does_not_cache(
    tiny_moss: Path, small_plan: CropPlan, cache: StepCache, computed: list[int]
) -> None:
    settings = _cached(cache)
    pipe = load_pipeline(tiny_moss, crop_blocks, loras={}, settings=settings)
    counts = _count_computations(pipe)

    _drawn(pipe, small_plan, "latents", settings)

    assert counts == computed


@pytest.mark.parametrize(
    "cache",
    [
        pytest.param(FIRST_BLOCK, id="first-block"),
        pytest.param(TAYLORSEER, id="taylorseer"),
        pytest.param(MAGCACHE, id="magcache"),
    ],
)
def test_moss_step_cache_draws_a_page_after_one_of_another_size_stopped_midway_as_it_drew_it_first(
    tiny_moss: Path, small_plan: CropPlan, two_column_page: Path, page_style: PageStyle, cache: StepCache
) -> None:
    settings = _cached(cache)
    pipe = load_pipeline(tiny_moss, crop_blocks, loras={}, settings=settings)
    first = _tensor(pipe, small_plan, "latents", settings)
    larger = plan_whole_page(load_xml_layout(two_column_page), page_style, pixels=96 * 48)
    _drawn(pipe, larger, "latents", settings.model_copy(update={"steps": 3}))  # stopped midway, as a failed page is

    again = _tensor(pipe, small_plan, "latents", settings)

    assert torch.equal(again, first)


def test_moss_step_cache_wraps_the_joint_blocks_run_from_their_aoti_package(
    tiny_moss: Path, small_plan: CropPlan, cli_package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _cached(FIRST_BLOCK)
    eager = _tensor(load_pipeline(tiny_moss, crop_blocks, loras={}, settings=settings), small_plan, "latents", settings)
    pipe = load_pipeline(
        tiny_moss, crop_blocks, loras={}, settings=settings.model_copy(update={"aoti_package": cli_package})
    )
    own = _count_computations(pipe)
    blocks = {id(block): index for index, block in enumerate(pipe.transformer.layers)}
    packaged = [0] * len(blocks)
    run_package = aoti.PackagedForward.__call__

    def counted(forward: aoti.PackagedForward, *args: torch.Tensor) -> torch.Tensor:
        packaged[blocks[id(forward.block)]] += 1
        return run_package(forward, *args)

    monkeypatch.setattr(aoti.PackagedForward, "__call__", counted)

    drawn = _tensor(pipe, small_plan, "latents", settings)

    assert (own, packaged) == ([0, 0, 0], [12, 1, 2])  # as eagerly: past the first step, only the first block runs
    assert _cosine(drawn, eager) > 0.999


def test_moss_magcache_calibration_measures_each_passs_ratios_as_diffusers_calibration_does(
    tiny_moss: Path, small_plan: CropPlan, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plain = SETTINGS.model_copy(update={"steps": CACHED_STEPS})
    reference = load_pipeline(tiny_moss, crop_blocks, loras={}, settings=plain)
    TransformerBlockRegistry.register(model_class=JointBlock, metadata=JOINT_BLOCK)
    reference.transformer.enable_cache(MagCacheConfig(calibrate=True, num_inference_steps=CACHED_STEPS))
    _drawn(reference, small_plan, "latents", plain)  # diffusers prints each pass's ratios as it ends: cond, uncond
    out = capsys.readouterr().out.splitlines()
    printed = [ast.literal_eval(out[k + 1]) for k, line in enumerate(out) if line.startswith("[MagCache] Calibration")]
    ratios_file = tmp_path / "ratios.jsonl"
    settings = _cached(MagCacheCalibration(ratios_file=ratios_file))
    pipe = load_pipeline(tiny_moss, crop_blocks, loras={}, settings=settings)

    _drawn(pipe, small_plan, "latents", settings)

    recorded = [json.loads(line) for line in ratios_file.read_text(encoding="utf-8").splitlines()]
    assert [line["pass"] for line in recorded] == ["pred_cond", "pred_uncond"]
    assert [line["mag_ratios"] for line in recorded] == printed
    assert calibrated_mag_ratios(ratios_file) == pytest.approx([(a + b) / 2 for a, b in zip(*printed, strict=True)])


@pytest.mark.parametrize(
    ("option", "name"),
    [
        pytest.param({"precision": Precision.MXFP8}, "precision", id="precision"),
        pytest.param({"text_encoder_precision": Precision.MXFP8}, "text_encoder_precision", id="text-encoder"),
        pytest.param({"aoti_package": Path("block.pt2")}, "aoti_package", id="aoti-package"),
        pytest.param({"sampler": Sampler.UNIPC}, "sampler", id="sampler"),
        pytest.param({"cache": FIRST_BLOCK}, "cache", id="cache"),
    ],
)
def test_load_pipeline_refuses_what_only_moss_takes_for_another_model(
    tiny_flux3: Path, option: dict[str, object], name: str
) -> None:
    settings = SETTINGS.model_copy(update=option)

    with pytest.raises(ValueError, match=f"is not MOSS-Image, which alone takes {name}$"):
        load_pipeline(tiny_flux3, crop_blocks, loras={}, settings=settings)


def test_load_pipeline_refuses_mxfp8_on_a_gpu_before_blackwell_before_loading_the_model(
    tiny_moss: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    monkeypatch.setattr(MossModularPipeline, "load_components", _model_loaded)
    settings = SETTINGS.model_copy(update={"precision": Precision.MXFP8, "device": "cuda"})

    with pytest.raises(ValueError, match=r"MXFP8 needs a GPU of compute capability 10\.0 or later \(Blackwell\)"):
        load_pipeline(tiny_moss, crop_blocks, loras={}, settings=settings)


def _model_loaded(*args: object, **kwargs: object) -> None:
    raise AssertionError("the model loaded before MXFP8 was refused")


@pytest.mark.parametrize(
    ("option", "error"),
    [
        pytest.param({"aoti_package": "block.pt2"}, "an AOTInductor package's blocks are compiled already", id="aoti"),
        pytest.param(  # as a run's draw.json gives it
            {"cache": {"kind": "taylorseer"}}, "a step cache decides around each joint block, eagerly", id="cache"
        ),
    ],
)
def test_draw_settings_refuse_beside_compile_blocks_a_package_or_a_step_cache(
    option: dict[str, object], error: str
) -> None:
    with pytest.raises(ValidationError, match=error):
        DrawSettings.model_validate({"compile_blocks": True, **option})


def test_load_pipeline_refuses_a_block_cache_for_a_model_whose_unconditional_pass_skips_its_first_block(
    tiny_moss_base: Path, tmp_path: Path
) -> None:
    write_moss_repo(tmp_path / "model", base=tiny_moss_base, uncond_skip_layers=(0,))

    with pytest.raises(ValueError, match=r"a pass of guidance skips the joint blocks \[0\]"):
        load_pipeline(tmp_path / "model", crop_blocks, loras={}, settings=_cached(FIRST_BLOCK))
