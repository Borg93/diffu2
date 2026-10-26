"""diffu end to end on CPU with a tiny model repo: drawing from a prompt, from a reference and, a glyph model, from its
glyph map, the text window, the adaLN-Zero init NeMo trains from, a trained checkpoint exported as a model repo, and the
latent packing FLUX.3's blocks expect."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from diffusers import Flux3Transformer2DModel
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep, Flux3VaeEncoderStep
from diffusers.modular_pipelines.flux3.modular_blocks_flux3 import Flux3AutoCoreDenoiseStep
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from typer.testing import CliRunner

from diffu.configs import DiffuConfig
from diffu.export import app as export_app
from diffu.export import export
from diffu.init import WarmStart, app, init_transformer, warm_start, write_model_repo
from diffu.latents import LatentStats, patchify, unpatchify
from diffu.modular_pipelines.diffu import DiffuModularPipeline
from diffu.modular_pipelines.diffu.decoders import DiffuDecodeStep
from diffu.modular_pipelines.diffu.encoders import DiffuTextEncoderStep, prompt_embeds
from diffu.modular_pipelines.diffu.modular_blocks_diffu import DiffuAutoVaeEncoderStep
from diffu.repo import INDEX, ModelIndex

GLYPH_SIZE = {"height": 64, "width": 128}  # a 4 x 8 latent grid (FLUX.3 pads a side under 64 px): not square


def _loaded(pipe: DiffuModularPipeline) -> DiffuModularPipeline:
    pipe.load_components(dtype=torch.float32)
    return pipe


@pytest.fixture(scope="module")
def pipe(tiny_diffu: Path) -> DiffuModularPipeline:
    return _loaded(DiffuModularPipeline.from_pretrained(str(tiny_diffu)))


@pytest.fixture(scope="module")
def glyph_pipe(tiny_diffu_glyph: Path) -> DiffuModularPipeline:
    return _loaded(DiffuModularPipeline.from_pretrained(str(tiny_diffu_glyph)))


def _draw(pipe: DiffuModularPipeline, **inputs: object) -> Image.Image:
    images = pipe(
        prompt="Ett protokoll",
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(0),
        output="images",
        **inputs,
    )
    return images[0]


def _drawn(pipe: DiffuModularPipeline, **inputs: object) -> torch.Tensor:
    """The latents ``pipe`` draws, as :func:`_draw` draws its image."""
    return pipe(
        prompt="Ett protokoll",
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(0),
        output="latents",
        **inputs,
    )


def _glyph_map(width: int = 128, height: int = 64) -> Image.Image:
    """A stand-in glyph map: red rising down it and green across it, so no two of its latent cells are alike."""
    down = Image.linear_gradient("L").resize((width, height))
    across = Image.linear_gradient("L").rotate(90).resize((width, height))
    return Image.merge("RGB", (down, across, Image.new("L", (width, height), 128)))


def test_diffu_draws_differently_from_a_reference_image(pipe: DiffuModularPipeline) -> None:
    plain = _draw(pipe, height=64, width=64)

    conditioned = _draw(pipe, height=64, width=64, image=Image.new("RGB", (64, 64), "white"))

    assert conditioned.tobytes() != plain.tobytes()


def _flux3_denoise_blocks() -> SequentialPipelineBlocks:
    """diffu's blocks as they were before the glyph map: FLUX.3's own denoise step between diffu's encoders and
    decoder."""
    return SequentialPipelineBlocks.from_blocks_dict(
        {
            "text_encoder": DiffuTextEncoderStep(),
            "vae_encoder": DiffuAutoVaeEncoderStep(),
            "denoise": Flux3AutoCoreDenoiseStep(),
            "decode": DiffuDecodeStep(),
        }
    )


@pytest.mark.parametrize(
    "reference",
    [pytest.param({}, id="text2image"), pytest.param({"image": Image.new("RGB", (64, 64))}, id="image-conditioned")],
)
def test_diffu_draws_a_model_without_the_glyph_map_as_flux3s_own_denoise_step_does(
    pipe: DiffuModularPipeline, tiny_diffu: Path, reference: dict[str, Image.Image]
) -> None:
    flux3 = _loaded(DiffuModularPipeline(blocks=_flux3_denoise_blocks(), pretrained_model_name_or_path=str(tiny_diffu)))
    expected = _drawn(flux3, height=64, width=64, **reference)

    drawn = _drawn(pipe, height=64, width=64, **reference)

    assert torch.equal(drawn, expected)


def test_a_glyph_model_draws_from_its_glyph_map(glyph_pipe: DiffuModularPipeline) -> None:
    maps = [_glyph_map(), _glyph_map(), Image.new("RGB", (128, 64), "white")]

    drawn = [_draw(glyph_pipe, **GLYPH_SIZE, glyph_image=glyph_image).tobytes() for glyph_image in maps]

    assert drawn[1] == drawn[0]  # the same map at the same seed
    assert drawn[2] != drawn[0]


def test_each_image_token_of_a_glyph_model_takes_its_own_cell_of_the_glyph_map_in_every_pass(
    glyph_pipe: DiffuModularPipeline,
) -> None:
    passes: list[torch.Tensor] = []
    hook = glyph_pipe.transformer.register_forward_pre_hook(
        lambda module, args, kwargs: passes.append(kwargs["hidden_states"]), with_kwargs=True
    )
    try:
        glyph_latents = glyph_pipe(
            prompt="Ett protokoll",
            num_inference_steps=2,
            generator=torch.Generator().manual_seed(0),
            glyph_image=_glyph_map(),
            output="glyph_latents",
            **GLYPH_SIZE,
        )
    finally:
        hook.remove()

    cells = glyph_latents.permute(0, 2, 3, 1).reshape(1, 32, 16)  # row by row: token k is cell (k // 8, k % 8)
    assert [torch.equal(hidden_states[..., 16:], cells) for hidden_states in passes] == [True] * 4  # 2 steps, 2 passes


@pytest.mark.parametrize(
    ("model", "glyph_map_size", "refusal"),
    [
        pytest.param("glyph_pipe", None, "a glyph model draws from a glyph map", id="glyph-model-without-a-map"),
        pytest.param("glyph_pipe", (128, 96), "draw the map at the image's size", id="map-off-the-image"),
        pytest.param("pipe", (128, 64), "takes no glyph map", id="map-to-a-model-without-one"),
    ],
)
def test_diffu_refuses_a_glyph_map_the_model_cannot_draw_from(
    model: str, glyph_map_size: tuple[int, int] | None, refusal: str, request: pytest.FixtureRequest
) -> None:
    drawing: DiffuModularPipeline = request.getfixturevalue(model)
    glyph_image = None if glyph_map_size is None else _glyph_map(*glyph_map_size)

    with pytest.raises(ValueError, match=refusal):
        _draw(drawing, **GLYPH_SIZE, glyph_image=glyph_image)


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

    started = warm_start(transformer, tiny_diffu)

    trained = Flux3Transformer2DModel.from_pretrained(tiny_diffu, subfolder="transformer").state_dict()
    assert started == WarmStart(fresh=["context_embedder.weight"], widened=[])
    assert torch.equal(transformer.context_embedder.weight, fresh_projection)
    assert all(
        torch.equal(weight, trained[name])
        for name, weight in transformer.state_dict().items()
        if name not in started.fresh
    )


def test_a_glyph_model_warm_started_from_a_model_without_the_map_draws_exactly_its_latents(
    tiny_diffu: Path, tiny_diffu_config: DiffuConfig, tmp_path: Path
) -> None:
    glyph_config = tiny_diffu_config.model_copy(update={"glyph": True})
    transformer = init_transformer(glyph_config, seed=1)
    started = warm_start(transformer, tiny_diffu)
    write_model_repo(transformer, glyph_config, tmp_path)
    widened = _loaded(DiffuModularPipeline.from_pretrained(str(tmp_path)))
    before = DiffuModularPipeline(blocks=_flux3_denoise_blocks(), pretrained_model_name_or_path=str(tiny_diffu))
    trained = _drawn(_loaded(before), **GLYPH_SIZE)  # as the trained model drew before the glyph map

    drawn = _drawn(widened, **GLYPH_SIZE, glyph_image=_glyph_map())

    assert started == WarmStart(fresh=[], widened=["x_embedders.image.weight", "proj_out.weight"])
    assert torch.equal(drawn, trained)


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


def test_diffu_export_of_a_glyph_model_keeps_it_a_glyph_model(tiny_diffu_glyph: Path, tmp_path: Path) -> None:
    trained = Flux3Transformer2DModel.from_pretrained(tiny_diffu_glyph, subfolder="transformer")
    checkpoint = _nemo_checkpoint(tmp_path / "epoch_0_step_9", trained, flow_shift=3.0)

    export(checkpoint, tiny_diffu_glyph, tmp_path / "model")

    config = Flux3Transformer2DModel.load_config(tmp_path / "model", subfolder="transformer")
    glyph_channels = ModelIndex.read(tmp_path / "model").glyph_channels
    assert (config["in_channels"], config["cond_in_channels"], glyph_channels) == (32, 16, 16)


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
