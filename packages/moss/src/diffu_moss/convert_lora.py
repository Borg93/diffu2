"""diffu-moss-convert-lora: a MOSS-Image LoRA in LyCORIS's kohya form, as KohakuMossTrainer exports one
(``lora-lycoris.safetensors``), written as a standard PEFT LoRA adapter folder of our DiT (:mod:`diffu_moss.adapters`),
which the page pipeline loads at draw time (``--lora hand=<folder>``). The four hand LoRAs are published so
(``Gabriel/diffu-moss-hand-pahlman``, ``-landskapshandlingar``, ``-overstathallarambetet-1715`` and ``-eight``).

    diffu-moss-convert-lora runs/moss_hand_pahlman/lora/export/lora-lycoris.safetensors --out runs/moss_hand_pahlman/adapter

A LyCORIS LoRA file holds three tensors for each module it adapts, keyed by the module's name with its dots as
underscores after the prefix ``lora_unet_`` (``lora_unet_layers_0_proj_ffn_down`` for ``layers.0.proj.ffn_down``):
``lora_down.weight`` ``[rank, in]``, ``lora_up.weight`` ``[out, rank]`` and ``alpha``, a number; the module's weight
changes by ``alpha / rank * up @ down``. The DiT's own names hold underscores (``ffn_down``), so a key is never split
on them: it is looked up among the DiT's linear layers by their names with dots as underscores. Each module's ``down``
is written as PEFT's ``lora_A`` and its ``up`` as ``lora_B``, in the dtype the file holds them in, under a
``LoraConfig`` of the rank and the alpha most modules have (any other module's in ``rank_pattern`` and
``alpha_pattern``, by exactly its name), the exact list of the modules it adapts (in the DiT's order) and the base
model it was trained on (``--base-model``, ``--base-revision``: MOSS-Image-Base at the revision the repo pins), so that
``lora_alpha / r * lora_B @ lora_A`` is each module's change, exactly. The weights' metadata records where they came
from (a plain LoRA of a KohakuMossTrainer LyCORIS export, and its training step), no path.

A file is refused before anything is written if it is not a plain LoRA (its metadata names another LyCORIS algorithm or
rsLoRA's scale, or it holds another algorithm's tensors: LoHa's, LoKr's, DoRA's, ...), a key names no linear layer of
the DiT, or two, a module's down and up differ in rank or do not fit its layer's widths, or a module lacks one of its
tensors.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import torch
import typer
from peft import LoraConfig
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from diffu_moss.adapters import ADAPTER_CONFIG, ADAPTER_WEIGHTS, PEFT_PREFIX
from diffu_moss.transformer import MossTransformer

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
console = Console()

LYCORIS_PREFIX = "lora_unet_"  # kohya's prefix of the denoiser's modules, which KohakuMossTrainer keeps
DOWN, UP, ALPHA = "lora_down.weight", "lora_up.weight", "alpha"  # a plain LoRA's tensors of a module
PROVENANCE = {"format": "peft", "converted_from": "lycoris (kohakumoss)", "algo": "lora"}  # the weights' metadata


class BaseModelRef(BaseModel):
    """The base model an adapter was trained on, as PEFT's config names it: a Hub repository and its revision."""

    model_config = ConfigDict(frozen=True)

    repo: str
    revision: str | None = None


# the DiT the hand LoRAs trained on, at the snapshot the repo pins
MOSS_IMAGE_BASE = BaseModelRef(
    repo="OpenMOSS-Team/MOSS-Image-Base", revision="85975793d8b3d432f1b6e27f352abaa4c2a4c0d3"
)


class ModuleLora(BaseModel):
    """One module's LoRA as a LyCORIS file holds it: ``down`` ``[rank, in]``, ``up`` ``[out, rank]`` and ``alpha``."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    down: Tensor
    up: Tensor
    alpha: float

    @property
    def rank(self) -> int:
        """The LoRA's rank: the rows of ``down``."""
        return self.down.shape[0]


def check_plain_lora(metadata: Mapping[str, str]) -> None:
    """Check a LyCORIS file's metadata names plain LoRA, if any algorithm (KohakuMossTrainer's record of it, else its
    network arguments', kohya's ``ss_network_args``), and not rsLoRA's scale.

    Raises:
        ValueError: If it names another algorithm, or rsLoRA's scale (alpha over the rank's square root).
    """
    arguments = json.loads(metadata.get("ss_network_args", "{}"))
    algo = metadata.get("kohakumoss_algo", str(arguments.get("algo", "lora")))
    if algo != "lora":
        raise ValueError(f"the file is a LyCORIS {algo} file, not a plain LoRA")
    if arguments.get("rs_lora"):
        raise ValueError("the file's LoRA is scaled as rsLoRA's (alpha over the rank's square root), not a plain LoRA")


def module_loras(tensors: Mapping[str, Tensor], transformer: nn.Module) -> dict[str, ModuleLora]:
    """A LyCORIS file's ``tensors`` as each module's LoRA, by the name of the linear layer of ``transformer`` (on any
    device, the meta device too) it adapts, in the DiT's own order.

    Raises:
        ValueError: If a tensor is no plain LoRA's or there is none, a module lacks one of its tensors, a key names no
            linear layer of the DiT or two, or a module's down and up differ in rank or do not fit its layer's widths.
    """
    by_key: dict[str, dict[str, Tensor]] = defaultdict(dict)
    for key, tensor in tensors.items():
        module, _, part = key.partition(".")
        if part not in (DOWN, UP, ALPHA):
            raise ValueError(f"{key} is no plain LoRA's tensor ({DOWN}, {UP} or {ALPHA}): another algorithm's")
        by_key[module][part] = tensor
    if not by_key:
        raise ValueError("the file holds no LoRA")
    linears: dict[str, list[tuple[str, nn.Linear]]] = defaultdict(list)
    for name, layer in transformer.named_modules():
        if isinstance(layer, nn.Linear):
            linears[LYCORIS_PREFIX + name.replace(".", "_")].append((name, layer))
    loras: dict[str, ModuleLora] = {}
    for key, parts in by_key.items():
        lacking = [part for part in (DOWN, UP, ALPHA) if part not in parts]
        if lacking:
            raise ValueError(f"{key} lacks its {' and '.join(lacking)}")
        found = linears.get(key, [])
        if len(found) != 1:
            names = ", ".join(name for name, _ in found) or "none"
            raise ValueError(f"{key} names {len(found)} linear layers of the DiT ({names}), not one")
        [(name, layer)] = found
        down, up = parts[DOWN], parts[UP]
        if down.dim() != 2 or up.dim() != 2 or down.shape[0] != up.shape[1]:
            raise ValueError(f"{key}'s down {list(down.shape)} and up {list(up.shape)} differ in rank")
        if (down.shape[1], up.shape[0]) != (layer.in_features, layer.out_features):
            raise ValueError(
                f"{key}'s down {list(down.shape)} and up {list(up.shape)} do not fit {name}, {layer.in_features} in "
                f"and {layer.out_features} out"
            )
        loras[name] = ModuleLora(down=down, up=up, alpha=float(parts[ALPHA]))
    return {name: loras[name] for name, _ in transformer.named_modules() if name in loras}


def lora_config(loras: Mapping[str, ModuleLora], base: BaseModelRef) -> LoraConfig:
    """The PEFT config of ``loras`` (by module name) trained on ``base``: the rank and the alpha most modules have, every
    other module's own in ``rank_pattern`` and ``alpha_pattern``, and the modules it adapts, each by its exact name."""
    rank, rank_pattern = _most_and_others({name: lora.rank for name, lora in loras.items()})
    alpha, alpha_pattern = _most_and_others({name: lora.alpha for name, lora in loras.items()})
    return LoraConfig(
        r=rank,
        # PEFT types lora_alpha as an int, but scales by any number: the file's alpha is kept as it is
        lora_alpha=alpha,  # ty: ignore[invalid-argument-type]  # pyrefly: ignore[bad-argument-type]
        target_modules=list(loras),
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        base_model_name_or_path=base.repo,
        revision=base.revision,
    )


def _most_and_others[T](values: Mapping[str, T]) -> tuple[T, dict[str, T]]:
    """The value most modules have (of a tie, the first module's), and every other module's by a PEFT pattern of
    exactly its name: PEFT matches a pattern to the end of a module's name after any dotted path, and ``^`` allows no
    path."""
    most = Counter(values.values()).most_common(1)[0][0]
    return most, {f"^{re.escape(name)}": value for name, value in values.items() if value != most}


def convert_lycoris(lycoris: Path, out: Path, *, transformer: nn.Module, base: BaseModelRef) -> LoraConfig:
    """The LyCORIS LoRA file ``lycoris`` of ``transformer`` (the DiT it was trained on; on the meta device, its layers'
    names and widths are enough), trained on ``base``, written into ``out`` as a PEFT LoRA adapter folder, its weights
    in the dtype the file holds them in and their metadata recording where they came from and the file's step.

    Returns:
        The adapter's config.

    Raises:
        ValueError: If ``out`` holds an adapter already, or the file is not a plain LoRA of the DiT's linear layers
            (:func:`check_plain_lora`, :func:`module_loras`); before anything is written.
    """
    if any((out / name).exists() for name in (ADAPTER_CONFIG, ADAPTER_WEIGHTS)):
        raise ValueError(f"{out} holds an adapter already: give a new folder")
    with safe_open(str(lycoris), "pt") as file:
        metadata = file.metadata() or {}
    check_plain_lora(metadata)
    loras = module_loras(load_file(str(lycoris)), transformer)
    config = lora_config(loras, base)
    weights: dict[str, Tensor] = {}
    for name, lora in loras.items():
        weights[f"{PEFT_PREFIX}{name}.lora_A.weight"] = lora.down
        weights[f"{PEFT_PREFIX}{name}.lora_B.weight"] = lora.up
    step = metadata.get("kohakumoss_step")
    out.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(out / ADAPTER_WEIGHTS), metadata=PROVENANCE | ({} if step is None else {"step": step}))
    # PEFT's own layout (LoraConfig.save_pretrained), but its modules in the DiT's order where PEFT's set has none
    record = config.to_dict() | {"target_modules": list(loras)}
    (out / ADAPTER_CONFIG).write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return config


def dit_layout(transformer: Path | None) -> MossTransformer:
    """The DiT of the folder ``transformer``'s config (MOSS-Image's own without one) on the meta device: its layers'
    names and widths, no weights."""
    with torch.device("meta"):
        if transformer is None:
            return MossTransformer()
        return MossTransformer.from_config(MossTransformer.load_config(str(transformer)))


@app.command()
def main(
    lycoris: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, help="the LoRA in LyCORIS's kohya form (KohakuMossTrainer's export)"
        ),
    ],
    out: Annotated[Path, typer.Option(help="the adapter folder to write (new)")],
    transformer: Annotated[
        Path | None,
        typer.Option(
            help="the DiT folder the LoRA was trained on, whose config.json gives its layers (default: MOSS-Image's own)"
        ),
    ] = None,
    base_model: Annotated[
        str, typer.Option(help="the Hub repository of the base model it was trained on, as the config names it")
    ] = MOSS_IMAGE_BASE.repo,
    base_revision: Annotated[
        str, typer.Option(help="that repository's revision ('' for none)")
    ] = MOSS_IMAGE_BASE.revision or "",
) -> None:
    """Write a MOSS-Image LoRA in LyCORIS's kohya form as a PEFT LoRA adapter folder, which diffu-generate --lora and
    diffu-run --lora load onto the DiT."""
    base = BaseModelRef(repo=base_model, revision=base_revision or None)
    try:
        config = convert_lycoris(lycoris, out, transformer=dit_layout(transformer), base=base)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    console.print(f"{len(config.target_modules or ())} modules at rank {config.r}, alpha {config.lora_alpha} -> {out}")


if __name__ == "__main__":
    app()
