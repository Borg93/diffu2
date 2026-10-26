"""A diffu model repo with freshly initialised weights: what NeMo AutoModel trains from and Modular Diffusers loads.

    diffu-init --size S --out runs/diffu/init_s
    diffu-init --size S --out runs/diffu/init_s_vl --weights-from runs/diffu_s_e2_glyph/model
    diffu-init --size S --out runs/diffu/init_s_pages --latent-stats runs/diffu/pages_latent_stats.json
    diffu-init --size S --glyph --out runs/diffu_s_e2c_qwen_init --weights-from runs/diffu_s_vl_init

With ``--weights-from`` the new model starts from a trained diffu's weights wherever their shapes agree; the rest (the
text projection, when the text encoder changed) keep their fresh init. With ``--latent-stats`` (our pages' latent mean
and standard deviation per channel, ``python -m diffu_train.latent_stats``) the repo keeps its own copy of the VAE,
normalizing latents with those stats instead of Qwen-Image's, which were measured on photos and video. With
``--glyph`` the model is a glyph model: it takes the glyph map's latents beside its own on channels, so its image input
and its output are twice as wide; started from a model without the glyph map, those two take the trained weights and
zeros for the map's channels (:func:`warm_start`), so the new model draws what the trained one draws until it trains.

Writes ``<out>/transformer/`` (config and weights) and ``<out>/modular_model_index.json``, which names the pipeline
and its blocks, the text layers, the glyph map's channels, and the text encoder and VAE by pinned Hub revision. NeMo
AutoModel's finetune mode (no ``peft:`` block) loads the transformer from it on every rank, so every rank starts from
the same weights; ``DiffuModularPipeline.from_pretrained(<out>)`` loads the whole model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import diffusers
import torch
import typer
from diffusers import AutoencoderKLQwenImage, Flux3Transformer2DModel
from pydantic import BaseModel, ValidationError
from rich.console import Console
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from diffu.configs import SIZES, DiffuConfig, SizeName, Vae
from diffu.latents import LatentStats
from diffu.modular_pipelines.diffu import DiffuAutoBlocks, DiffuModularPipeline
from diffu.repo import INDEX, ModelIndex, index_entry

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
console = Console()

# The weights a glyph model's glyph map widens, by name, and the axis it widens: the image input's columns and the
# output's rows. Nothing else changes shape (the reference input keeps the latents' width).
GLYPH_WIDENED = {"x_embedders.image.weight": 1, "proj_out.weight": 0}


class WarmStart(BaseModel):
    """What a new model took from a trained diffu (:func:`warm_start`).

    Attributes:
        fresh: The weights that keep their fresh init: the trained model has none of their shape.
        widened: The trained weights widened with zeros for a glyph model's glyph map (:data:`GLYPH_WIDENED`).
    """

    fresh: list[str]
    widened: list[str]


def init_transformer(config: DiffuConfig, *, seed: int) -> Flux3Transformer2DModel:
    """The transformer with random weights, adaLN-Zero.

    Every modulation head, the output norm's modulation and the output projection start at zero (DiT's init): each
    block starts as the identity and the model predicts zero velocity.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        transformer = Flux3Transformer2DModel.from_config(config.transformer_config())
    zeroed = [
        *(head.linear for head in transformer.early_stream_modulations.values()),
        *(head.linear for head in transformer.single_stream_modulations.values()),
        transformer.norm_out.linear,
        transformer.proj_out,
    ]
    with torch.no_grad():
        for layer in zeroed:
            layer.weight.zero_()
    return transformer


def _glyph_widened(
    trained: dict[str, torch.Tensor], own: dict[str, torch.Tensor], glyph_channels: int
) -> dict[str, torch.Tensor]:
    """The trained weights of :data:`GLYPH_WIDENED` that are ``own``'s but ``glyph_channels`` narrower, widened to
    ``own``'s shape: the trained values first, zeros for the glyph map's channels. With the map's input columns at
    zero the glyph model draws what the trained one draws (the input projection has no bias); the map's output rows
    are never trained."""
    widened: dict[str, torch.Tensor] = {}
    for name, axis in GLYPH_WIDENED.items():
        narrower = list(own[name].shape)
        narrower[axis] -= glyph_channels
        weight = trained.get(name)
        if glyph_channels and weight is not None and list(weight.shape) == narrower:
            zeros = list(weight.shape)
            zeros[axis] = glyph_channels
            widened[name] = torch.cat([weight, weight.new_zeros(zeros)], dim=axis)
    return widened


def warm_start(transformer: Flux3Transformer2DModel, source: Path) -> WarmStart:
    """Load into ``transformer`` every weight of the diffu repo at ``source`` with the same name and shape; a glyph
    model started from a model without the glyph map also takes that model's image input and output, widened with
    zeros for the map's channels (its input channels beyond its reference input's).

    Returns:
        The weights that keep their fresh init and the weights widened for the glyph map.
    """
    trained = Flux3Transformer2DModel.from_pretrained(source, subfolder="transformer").state_dict()
    own = transformer.state_dict()
    same = {
        name: trained[name] for name, weight in own.items() if name in trained and trained[name].shape == weight.shape
    }
    glyph_channels = transformer.config.in_channels - transformer.config.cond_in_channels
    widened = _glyph_widened(trained, own, glyph_channels)
    transformer.load_state_dict({**own, **same, **widened})
    return WarmStart(
        fresh=[name for name in own if name not in same and name not in widened],
        widened=[name for name in own if name in widened],
    )


def model_index(config: DiffuConfig, repo: Path) -> ModelIndex:
    """The ``modular_model_index.json`` of a diffu repo at ``repo``: its pipeline and block classes, its text layers,
    its glyph map's channels, and where each component loads from (the transformer from the repo itself)."""
    text, vae = config.text_encoder, config.vae
    return ModelIndex.model_validate(
        {
            "_blocks_class_name": DiffuAutoBlocks.__name__,
            "_class_name": DiffuModularPipeline.__name__,
            "_diffusers_version": diffusers.__version__,
            "text_layers": list(text.layers),
            "glyph_channels": config.glyph_channels,
            "text_encoder": index_entry(
                "transformers",
                Qwen3VLForConditionalGeneration,
                repo=text.repo,
                revision=text.revision,
                subfolder=text.encoder_subfolder,
            ),
            "tokenizer": index_entry(
                "transformers",
                AutoProcessor,
                repo=text.repo,
                revision=text.revision,
                subfolder=text.processor_subfolder,
            ),
            "transformer": index_entry(
                "diffusers", Flux3Transformer2DModel, repo=str(repo), revision=None, subfolder="transformer"
            ),
            "vae": index_entry(
                "diffusers", AutoencoderKLQwenImage, repo=vae.repo, revision=vae.revision, subfolder=vae.subfolder
            ),
        }
    )


def write_vae(vae: Vae, stats: LatentStats, out: Path) -> Vae:
    """``vae``'s weights saved as the repo at ``out``'s own VAE, which normalizes latents with ``stats``.

    Returns:
        The repo's own VAE, to name in its model index.
    """
    model = AutoencoderKLQwenImage.from_pretrained(
        vae.repo, revision=vae.revision, subfolder=vae.subfolder, **stats.model_dump(mode="json")
    )
    model.save_pretrained(out / "vae")
    return vae.model_copy(update={"repo": str(out.resolve()), "revision": None, "subfolder": "vae"})


def read_latent_stats(path: Path, *, channels: int) -> LatentStats:
    """The latent stats in ``path``, checked to have one mean and standard deviation per VAE channel.

    Raises:
        typer.BadParameter: If the file holds no latent stats, or stats of another channel count.
    """
    try:
        stats = LatentStats.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise typer.BadParameter(f"{path} holds no latent stats: {error}", param_hint="--latent-stats") from error
    if len(stats.latents_mean) != channels:
        raise typer.BadParameter(
            f"{path} holds stats of {len(stats.latents_mean)} channels; the VAE has {channels}",
            param_hint="--latent-stats",
        )
    return stats


def write_model_repo(transformer: Flux3Transformer2DModel, config: DiffuConfig, out: Path) -> None:
    """Save ``transformer`` and the model index into ``out``, making it a diffu model repo."""
    transformer.save_pretrained(out / "transformer")
    model_index(config, out.resolve()).write(out)


@app.command()
def main(
    out: Annotated[Path, typer.Option(help="the new model repo's directory")],
    size: Annotated[SizeName, typer.Option(help="the transformer size")] = "S",
    seed: Annotated[int, typer.Option(help="the seed of the random weights")] = 0,
    weights_from: Annotated[
        Path | None, typer.Option(help="a trained diffu repo to start from wherever the shapes agree")
    ] = None,
    latent_stats: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, help="our pages' latent stats, for the repo's own VAE to normalize with"
        ),
    ] = None,
    glyph: Annotated[
        bool, typer.Option(help="a glyph model: the glyph map's latents beside the image latents on channels")
    ] = False,
) -> None:
    """Write a diffu model repo with freshly initialised weights, or a trained one's where they fit."""
    if (out / INDEX).exists():
        raise typer.BadParameter(f"{out} already holds a model; give a new directory", param_hint="--out")
    if weights_from is not None and not (weights_from / INDEX).exists():
        raise typer.BadParameter(f"{weights_from} holds no diffu model", param_hint="--weights-from")
    config = DiffuConfig(size=SIZES[size], glyph=glyph)
    stats = None if latent_stats is None else read_latent_stats(latent_stats, channels=config.vae.latent_channels)
    transformer = init_transformer(config, seed=seed)
    if weights_from is not None:
        started = warm_start(transformer, weights_from)
        console.print(f"from {weights_from}: all weights but {', '.join(started.fresh) or 'none'}")
        if started.widened:
            console.print(f"widened with zeros for the glyph map: {', '.join(started.widened)}")
    if stats is not None:
        config = config.model_copy(update={"vae": write_vae(config.vae, stats, out)})
        console.print(f"its own VAE, normalizing with the stats in {latent_stats}")
    write_model_repo(transformer, config, out)
    parameters = sum(parameter.numel() for parameter in transformer.parameters())
    console.print(f"diffu-{size}: {parameters / 1e6:.1f}M parameters -> {out}")


if __name__ == "__main__":
    app()
