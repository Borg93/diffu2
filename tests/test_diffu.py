"""diffu end to end on CPU with a tiny model repo: drawing from a prompt and from a reference, the text window, the
adaLN-Zero init NeMo trains from, a trained checkpoint exported as a model repo, and the latent packing FLUX.3's blocks
expect."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from diffusers import Flux3Transformer2DModel
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep, Flux3VaeEncoderStep
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from typer.testing import CliRunner

from diffu.configs import DiffuConfig
from diffu.export import app as export_app
from diffu.export import export
from diffu.init import app, init_transformer, warm_start, write_model_repo
from diffu.latents import LatentStats, patchify, unpatchify
from diffu.modular_pipelines.diffu import DiffuModularPipeline
from diffu.modular_pipelines.diffu.encoders import prompt_embeds
from diffu.repo import INDEX


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


def test_diffu_draws_differently_from_a_reference_image(pipe: DiffuModularPipeline) -> None:
    plain = _draw(pipe, height=64, width=64)

    conditioned = _draw(pipe, height=64, width=64, image=Image.new("RGB", (64, 64), "white"))

    assert conditioned.tobytes() != plain.tobytes()


@pytest.mark.parametrize(
    "prompt",
    [
        pytest.param("Ett protokoll", id="one-window"),
        pytest.param("ord " * 100, id="two-windows"),
    ],
)
def test_prompt_embeds_are_flux3s(tiny_flux3_snapshot: Path, prompt: str) -> None:
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(tiny_flux3_snapshot, subfolder="text_encoder")
    processor = AutoProcessor.from_pretrained(tiny_flux3_snapshot, subfolder="tokenizer")
    flux3 = SequentialPipelineBlocks.from_blocks_dict({"text_encoder": Flux3TextEncoderStep()}).init_pipeline()
    flux3.update_components(text_encoder=text_encoder, tokenizer=processor)

    embeds = prompt_embeds(
        text_encoder, processor, prompt, layers=(0, 1), max_sequence_length=8192, device=torch.device("cpu")
    )

    expected = flux3(prompt=prompt, text_encoder_out_layers=(0, 1), output="prompt_embeds")
    assert embeds.shape == expected.shape
    assert torch.equal(embeds, expected)


def test_init_transformer_predicts_zero_velocity_once_saved_and_loaded(
    tiny_diffu_config: DiffuConfig, tmp_path: Path
) -> None:
    write_model_repo(init_transformer(tiny_diffu_config, seed=0), tiny_diffu_config, tmp_path)
    transformer = Flux3Transformer2DModel.from_pretrained(tmp_path / "transformer")

    velocity = transformer(
        hidden_states=torch.randn(1, 6, 16),
        encoder_hidden_states=torch.randn(1, 3, 16),
        timestep=torch.tensor([0.5]),
        img_ids=torch.zeros(6, 4),
        txt_ids=torch.zeros(3, 4),
        return_dict=False,
    )[0]

    assert torch.equal(velocity, torch.zeros(1, 6, 16))


def test_diffu_init_refuses_a_directory_that_holds_a_model(tmp_path: Path) -> None:
    (tmp_path / INDEX).write_text("{}")

    result = CliRunner().invoke(app, ["--out", str(tmp_path)])

    assert result.exit_code == 2
    assert sorted(path.name for path in tmp_path.iterdir()) == [INDEX]


def test_warm_start_keeps_a_trained_models_weights_but_the_new_text_projection(
    tiny_diffu: Path, tiny_diffu_config: DiffuConfig
) -> None:
    wider_text = tiny_diffu_config.model_copy(
        update={"text_encoder": tiny_diffu_config.text_encoder.model_copy(update={"layers": (0, 1)})}
    )
    transformer = init_transformer(wider_text, seed=1)
    fresh_projection = transformer.context_embedder.weight.clone()

    fresh = warm_start(transformer, tiny_diffu)

    trained = Flux3Transformer2DModel.from_pretrained(tiny_diffu, subfolder="transformer").state_dict()
    assert fresh == ["context_embedder.weight"]
    assert torch.equal(transformer.context_embedder.weight, fresh_projection)
    assert all(
        torch.equal(weight, trained[name]) for name, weight in transformer.state_dict().items() if name not in fresh
    )


def test_diffu_init_refuses_weights_from_a_directory_that_holds_no_model(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["--out", str(tmp_path / "new"), "--weights-from", str(tmp_path)])

    assert result.exit_code == 2
    assert not (tmp_path / "new").exists()


def test_diffu_init_refuses_latent_stats_of_another_vae_before_it_writes(tmp_path: Path) -> None:
    stats = tmp_path / "stats.json"
    stats.write_text(LatentStats(latents_mean=(0.0,) * 4, latents_std=(1.0,) * 4).model_dump_json())  # the VAE has 16

    result = CliRunner().invoke(app, ["--out", str(tmp_path / "new"), "--latent-stats", str(stats)])

    assert result.exit_code == 2
    assert not (tmp_path / "new").exists()


def _nemo_checkpoint(path: Path, transformer: Flux3Transformer2DModel | None, *, flow_shift: float) -> Path:
    """A NeMo AutoModel checkpoint as diffu-export reads it: the run's config and, in the run's last checkpoint, the
    transformer's consolidated export."""
    path.mkdir()
    (path / "config.yaml").write_text(yaml.safe_dump({"flow_matching": {"flow_shift": flow_shift}}), encoding="utf-8")
    if transformer is not None:
        transformer.save_pretrained(path / "model" / "consolidated")
    return path


def test_diffu_export_loads_the_trained_transformer_at_the_shift_it_trained_at(
    tiny_diffu: Path, tiny_diffu_config: DiffuConfig, tmp_path: Path
) -> None:
    torch.manual_seed(1)
    trained = Flux3Transformer2DModel.from_config(tiny_diffu_config.transformer_config())
    checkpoint = _nemo_checkpoint(tmp_path / "epoch_0_step_9", trained, flow_shift=3.5)
    out = tmp_path / "model"

    result = CliRunner().invoke(
        export_app, ["--checkpoint", str(checkpoint), "--init", str(tiny_diffu), "--out", str(out)]
    )

    assert result.exit_code == 0, result.output
    pipe = DiffuModularPipeline.from_pretrained(str(out))
    pipe.load_components(names=["transformer", "scheduler"])
    assert pipe.scheduler.config.shift == 3.5
    assert torch.equal(pipe.transformer.proj_out.weight, trained.proj_out.weight)


def test_diffu_export_refuses_a_checkpoint_without_a_consolidated_export(tiny_diffu: Path, tmp_path: Path) -> None:
    checkpoint = _nemo_checkpoint(tmp_path / "epoch_0_step_4", None, flow_shift=6.0)

    with pytest.raises(ValueError, match="has no consolidated export"):
        export(checkpoint, tiny_diffu, tmp_path / "model")

    assert not (tmp_path / "model").exists()


def test_diffu_export_refuses_a_directory_that_holds_a_model(
    tiny_diffu: Path, tiny_diffu_config: DiffuConfig, tmp_path: Path
) -> None:
    trained = init_transformer(tiny_diffu_config, seed=1)
    checkpoint = _nemo_checkpoint(tmp_path / "epoch_0_step_9", trained, flow_shift=3.5)
    export(checkpoint, tiny_diffu, tmp_path / "model")
    exported = sorted(path.name for path in (tmp_path / "model").iterdir())

    with pytest.raises(ValueError, match="already holds a model"):
        export(checkpoint, tiny_diffu, tmp_path / "model")

    assert sorted(path.name for path in (tmp_path / "model").iterdir()) == exported


def test_diffu_export_refuses_a_directory_with_a_dangling_transformer_link(tiny_diffu: Path, tmp_path: Path) -> None:
    out = tmp_path / "model"
    out.mkdir()
    (out / "transformer").symlink_to(tmp_path / "gone", target_is_directory=True)

    with pytest.raises(ValueError, match="already holds a model"):
        export(_nemo_checkpoint(tmp_path / "epoch_0_step_4", None, flow_shift=6.0), tiny_diffu, out)


def _latents() -> torch.Tensor:
    return torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 4, 6)


def test_patchify_packs_channels_in_flux3s_order() -> None:
    packed = patchify(_latents())

    assert torch.equal(packed, Flux3VaeEncoderStep._patchify_latents(_latents()))


def test_unpatchify_inverts_patchify() -> None:
    packed = patchify(_latents())

    assert torch.equal(unpatchify(packed), _latents())
