"""MOSS-Image's inference options through the page pipeline on CPU (diffu_moss.inference, by DrawSettings alone): every
option off, the tiny MOSS draws a crop byte for byte as main drew it before them; MXFP8 in the DiT or the text encoder
(emulated on the CPU) draws close to bfloat16 but not as it; the joint blocks run from an AOTInductor package built by
diffu-moss-aoti, or compiled with MXFP8 off and on, close to as they run eagerly; and what only MOSS-Image takes is
refused for another model, as are MXFP8 on a GPU without it and a package beside compile_blocks."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from diffusers.modular_pipelines import ModularPipeline
from pydantic import ValidationError
from transformers import Qwen3_5Config, Qwen3_5Model
from typer.testing import CliRunner

from diffu_core.repo import ModelIndex
from diffu_moss import aoti
from diffu_moss.init import write_moss_repo
from diffu_moss.modular_pipelines.moss import MossModularPipeline
from diffu_moss.mxfp8 import Precision
from diffu_moss.transformer import JointBlock, MossTransformer
from diffu_page.captions.prompt import PageStyle
from diffu_page.draw.pipeline import DrawSettings, crop_blocks, load_pipeline
from diffu_page.draw.plan import CropPlan, plan_whole_page
from diffu_page.page.layout import load_xml_layout

SETTINGS = DrawSettings(steps=2, device="cpu")
# The tiny MOSS's crop of small_plan from seed 0 as main drew it before these options (27c9fc9, on the CPU): the
# SHA-256 of its RGB bytes, the same at 1, 2 and 4 threads
MAIN_DRAWING_SHA256 = "8f19b95b784939f35b2fa415485d23a69e410ec05953c6cdf74f9321a87c8d75"
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


def _drawn(pipe: ModularPipeline, plan: CropPlan, output: str) -> Any:
    """``plan`` drawn from noise seed 0 in two steps, its ``output`` (``images``, ``latents``, ``prompt_embeds``)."""
    pipe.guider.set_state(step=0, num_inference_steps=SETTINGS.steps, timestep=None)
    return pipe(crop_plan=plan, generator=torch.Generator().manual_seed(0), **SETTINGS.pipeline_inputs(), output=output)


def _tensor(pipe: ModularPipeline, plan: CropPlan, output: str) -> torch.Tensor:
    """``plan``'s ``output`` (``latents`` or ``prompt_embeds``) drawn by :func:`_drawn`, in float32."""
    return _drawn(pipe, plan, output).float()


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


@pytest.mark.parametrize(
    ("option", "name"),
    [
        pytest.param({"precision": Precision.MXFP8}, "precision", id="precision"),
        pytest.param({"text_encoder_precision": Precision.MXFP8}, "text_encoder_precision", id="text-encoder"),
        pytest.param({"aoti_package": Path("block.pt2")}, "aoti_package", id="aoti-package"),
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


def test_draw_settings_refuse_an_aoti_package_beside_compile_blocks() -> None:
    with pytest.raises(ValidationError, match="an AOTInductor package's blocks are compiled already"):
        DrawSettings(compile_blocks=True, aoti_package=Path("block.pt2"))
