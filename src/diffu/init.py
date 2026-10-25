"""A diffu model repo with freshly initialised weights: what NeMo AutoModel trains from and Modular Diffusers loads.

    diffu-init --size S --out runs/diffu/init_s
    diffu-init --size S --out runs/diffu/init_s_vl --weights-from runs/diffu_s_e2_glyph/model

With ``--weights-from`` the new model starts from a trained diffu's weights wherever their shapes agree; the rest (the
text projection, when the text encoder changed) keep their fresh init.

Writes ``<out>/transformer/`` (config and weights) and ``<out>/modular_model_index.json``, which names the pipeline
and its blocks, the text layers, and the text encoder and VAE by pinned Hub revision. NeMo AutoModel's finetune mode
(no ``peft:`` block) loads the transformer from it on every rank, so every rank starts from the same weights;
``DiffuModularPipeline.from_pretrained(<out>)`` loads the whole model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import diffusers
import torch
import typer
from diffusers import AutoencoderKLQwenImage, Flux3Transformer2DModel
from rich.console import Console
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from diffu.configs import SIZES, DiffuConfig, SizeName
from diffu.modular_pipelines.diffu import DiffuAutoBlocks, DiffuModularPipeline

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
console = Console()

INDEX = "modular_model_index.json"


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


def warm_start(transformer: Flux3Transformer2DModel, source: Path) -> list[str]:
    """Load into ``transformer`` every weight of the diffu repo at ``source`` with the same name and shape.

    Returns:
        The names of the weights that keep their fresh init, because ``source`` has none of that shape.
    """
    trained = Flux3Transformer2DModel.from_pretrained(source, subfolder="transformer").state_dict()
    own = transformer.state_dict()
    fresh = [name for name, weight in own.items() if name not in trained or trained[name].shape != weight.shape]
    transformer.load_state_dict({**own, **{name: trained[name] for name in own if name not in fresh}})
    return fresh


def index_entry(library: str, cls: type, *, repo: str, revision: str | None, subfolder: str) -> list[object]:
    """One component of a modular_model_index.json: its library, its class and where it loads from."""
    spec = {
        "pretrained_model_name_or_path": repo,
        "revision": revision,
        "subfolder": subfolder,
        "type_hint": [library, cls.__name__],
        "variant": None,
    }
    return [library, cls.__name__, spec]


def model_index(config: DiffuConfig, repo: Path) -> dict[str, object]:
    """The ``modular_model_index.json`` of a diffu repo at ``repo``: its pipeline and block classes, its text layers,
    and where each component loads from (the transformer from the repo itself)."""
    text, vae = config.text_encoder, config.vae
    return {
        "_blocks_class_name": DiffuAutoBlocks.__name__,
        "_class_name": DiffuModularPipeline.__name__,
        "_diffusers_version": diffusers.__version__,
        "text_layers": list(text.layers),
        "text_encoder": index_entry(
            "transformers",
            Qwen3VLForConditionalGeneration,
            repo=text.repo,
            revision=text.revision,
            subfolder=text.encoder_subfolder,
        ),
        "tokenizer": index_entry(
            "transformers", AutoProcessor, repo=text.repo, revision=text.revision, subfolder=text.processor_subfolder
        ),
        "transformer": index_entry(
            "diffusers", Flux3Transformer2DModel, repo=str(repo), revision=None, subfolder="transformer"
        ),
        "vae": index_entry(
            "diffusers", AutoencoderKLQwenImage, repo=vae.repo, revision=vae.revision, subfolder=vae.subfolder
        ),
    }


def write_model_repo(transformer: Flux3Transformer2DModel, config: DiffuConfig, out: Path) -> None:
    """Save ``transformer`` and the model index into ``out``, making it a diffu model repo."""
    transformer.save_pretrained(out / "transformer")
    index = json.dumps(model_index(config, out.resolve()), indent=2)
    (out / INDEX).write_text(f"{index}\n", encoding="utf-8")


@app.command()
def main(
    out: Annotated[Path, typer.Option(help="the new model repo's directory")],
    size: Annotated[SizeName, typer.Option(help="the transformer size")] = "S",
    seed: Annotated[int, typer.Option(help="the seed of the random weights")] = 0,
    weights_from: Annotated[
        Path | None, typer.Option(help="a trained diffu repo to start from wherever the shapes agree")
    ] = None,
) -> None:
    """Write a diffu model repo with freshly initialised weights, or a trained one's where they fit."""
    if (out / INDEX).exists():
        raise typer.BadParameter(f"{out} already holds a model; give a new directory", param_hint="--out")
    if weights_from is not None and not (weights_from / INDEX).exists():
        raise typer.BadParameter(f"{weights_from} holds no diffu model", param_hint="--weights-from")
    config = DiffuConfig(size=SIZES[size])
    transformer = init_transformer(config, seed=seed)
    if weights_from is not None:
        fresh = warm_start(transformer, weights_from)
        console.print(f"from {weights_from}: all weights but {', '.join(fresh) or 'none'}")
    write_model_repo(transformer, config, out)
    parameters = sum(parameter.numel() for parameter in transformer.parameters())
    console.print(f"diffu-{size}: {parameters / 1e6:.1f}M parameters -> {out}")


if __name__ == "__main__":
    app()
