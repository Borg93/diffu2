"""diffu end to end on CPU with a tiny model repo: drawing from a prompt and from a reference, the text window, the
adaLN-Zero init NeMo trains from, and the latent packing FLUX.3's blocks expect."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from diffusers import Flux3Transformer2DModel
from diffusers.modular_pipelines.flux3.encoders import Flux3VaeEncoderStep
from PIL import Image
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from typer.testing import CliRunner

from diffu.configs import DiffuConfig
from diffu.init import INDEX, app, init_transformer, write_model_repo
from diffu.latents import patchify, unpatchify
from diffu.modular_pipelines.diffu import DiffuModularPipeline
from diffu.modular_pipelines.diffu.encoders import prompt_embeds


@pytest.fixture(scope="module")
def pipe(tiny_diffu: Path) -> DiffuModularPipeline:
    pipe = DiffuModularPipeline.from_pretrained(str(tiny_diffu))
    pipe.load_components(dtype=torch.float32)
    return pipe


def _draw(pipe: DiffuModularPipeline, **inputs: object) -> Image.Image:
    images = pipe(
        prompt="Ett protokoll",
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(0),
        output="images",
        **inputs,
    )
    return images[0]


def test_diffu_draws_an_image_of_the_requested_size(pipe: DiffuModularPipeline) -> None:
    image = _draw(pipe, height=64, width=48)

    assert image.size == (48, 64)


def test_diffu_draws_differently_from_a_reference_image(pipe: DiffuModularPipeline) -> None:
    plain = _draw(pipe, height=64, width=64)

    conditioned = _draw(pipe, height=64, width=64, image=Image.new("RGB", (64, 64), "white"))

    assert conditioned.tobytes() != plain.tobytes()


@pytest.mark.parametrize(
    ("prompt", "window"),
    [
        pytest.param("Ett protokoll", 80, id="one-window"),
        pytest.param("ord " * 100, 160, id="two-windows"),
    ],
)
def test_prompt_embeds_pad_the_window_to_a_multiple_of_80(tiny_qwen35: Path, prompt: str, window: int) -> None:
    text_encoder = Qwen3_5ForConditionalGeneration.from_pretrained(tiny_qwen35)
    tokenizer = AutoTokenizer.from_pretrained(tiny_qwen35)

    embeds = prompt_embeds(
        text_encoder, tokenizer, prompt, layers=(0, 1), max_sequence_length=8192, device=torch.device("cpu")
    )

    assert embeds.shape == (1, window, 2 * 32)


def test_init_transformer_predicts_zero_velocity_once_saved_and_loaded(
    tiny_diffu_config: DiffuConfig, tmp_path: Path
) -> None:
    write_model_repo(init_transformer(tiny_diffu_config, seed=0), tiny_diffu_config, tmp_path)
    transformer = Flux3Transformer2DModel.from_pretrained(tmp_path / "transformer")

    velocity = transformer(
        hidden_states=torch.randn(1, 6, 16),
        encoder_hidden_states=torch.randn(1, 3, 32),
        timestep=torch.tensor([0.5]),
        img_ids=torch.zeros(6, 4),
        txt_ids=torch.zeros(3, 4),
        return_dict=False,
    )[0]

    assert torch.equal(velocity, torch.zeros(1, 6, 16))


def test_diffu_init_refuses_a_directory_that_holds_a_model(tmp_path: Path) -> None:
    (tmp_path / INDEX).write_text("{}")

    result = CliRunner().invoke(app, ["--out", str(tmp_path)])

    assert result.exit_code == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == [INDEX]


def _latents() -> torch.Tensor:
    return torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 4, 6)


def test_patchify_packs_channels_in_flux3s_order() -> None:
    packed = patchify(_latents())

    assert torch.equal(packed, Flux3VaeEncoderStep._patchify_latents(_latents()))


def test_unpatchify_inverts_patchify() -> None:
    packed = patchify(_latents())

    assert torch.equal(unpatchify(packed), _latents())
