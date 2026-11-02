"""MOSS-Image LoRAs as standard PEFT LoRA adapters of our DiT, loaded onto it at draw time.

An adapter is a folder of PEFT's two files: ``adapter_config.json``, a ``peft.LoraConfig`` (the rank and alpha, the
DiT's modules it adapts by name, and any module's own rank or alpha), and ``adapter_model.safetensors``, each module's
``lora_A`` and ``lora_B`` under PEFT's ``base_model.model.`` prefix. PEFT loads it as it loads any adapter
(``PeftModel.from_pretrained(<MossTransformer>, <folder>)``, and ``merge_and_unload`` merges it), and so does the page
pipeline, through diffusers' adapter methods on the DiT: :func:`load_adapters` loads each under its name, active at a
strength, and :func:`fuse_adapters` merges them into the DiT's weights. A hand is then its adapter (some 150 MB at rank
32) beside the one DiT, not a DiT of its own. ``diffu-moss-convert-lora`` writes a LoRA exported by KohakuMossTrainer
as one (:mod:`diffu_moss.convert_lora`).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path

from peft import LoraConfig, PeftConfig, get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.lora import LoraLayer
from peft.utils import CONFIG_NAME, SAFETENSORS_WEIGHTS_NAME
from pydantic import BaseModel, ConfigDict
from safetensors.torch import load_file
from torch import Tensor

from diffu_moss.transformer import MossTransformer

ADAPTER_CONFIG = CONFIG_NAME  # adapter_config.json
ADAPTER_WEIGHTS = SAFETENSORS_WEIGHTS_NAME  # adapter_model.safetensors
PEFT_PREFIX = "base_model.model."  # what PEFT keys an adapter's weights under: its PeftModel's path to the model


class Adapter(BaseModel):
    """A LoRA adapter folder as read: its config, and its weights by the DiT's own names
    (``layers.0.proj.q.lora_A.weight``)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    config: LoraConfig
    weights: dict[str, Tensor]


def read_adapter(folder: Path) -> Adapter:
    """The LoRA adapter in ``folder``, laid out as PEFT saves one.

    Raises:
        ValueError: If the folder lacks either file, its config is no PEFT LoRA config, or a weight is not keyed under
            PEFT's prefix; naming the folder.
    """
    missing = [name for name in (ADAPTER_CONFIG, ADAPTER_WEIGHTS) if not (folder / name).is_file()]
    if missing:
        raise ValueError(f"{folder} is not a LoRA adapter folder: it has no {' or '.join(missing)}")
    try:
        config = PeftConfig.from_pretrained(str(folder))
    except (TypeError, ValueError) as error:  # PEFT's refusals of a config without a PEFT type, or of none it knows
        raise ValueError(f"{folder / ADAPTER_CONFIG} is no PEFT adapter config: {error}") from error
    if not isinstance(config, LoraConfig):
        raise ValueError(f"{folder} holds no LoRA adapter: its config is a {type(config).__name__}")
    weights = load_file(str(folder / ADAPTER_WEIGHTS))
    unprefixed = sorted(key for key in weights if not key.startswith(PEFT_PREFIX))
    if unprefixed:
        raise ValueError(
            f"{folder}: {len(unprefixed)} weights are not keyed under PEFT's {PEFT_PREFIX!r}, the first "
            f"{unprefixed[0]!r}"
        )
    return Adapter(config=config, weights={key.removeprefix(PEFT_PREFIX): value for key, value in weights.items()})


def load_adapters(transformer: MossTransformer, adapters: Mapping[str, Adapter], *, scale: float) -> None:
    """Each of ``adapters`` loaded onto ``transformer`` under its name, its layers in float32 on the device of the layers
    they adapt, and every one of them active at ``scale`` times its trained strength (1: as trained, 0: not at all;
    diffusers' ``set_adapters``), in place. In float32, a layer's change is added to its output in float32, and fused
    (:func:`fuse_adapters`) into a bfloat16 weight rounded once, as a merge in float32 rounds it (as KohakuMossTrainer
    merged the hands).

    Raises:
        ValueError: If an adapter does not fit the DiT, naming it: a module it adapts is not the DiT's, or its weights
            are not exactly the LoRA weights its config gives the DiT, of their shapes.
    """
    for name, adapter in adapters.items():
        try:
            transformer.add_adapter(copy.deepcopy(adapter.config), adapter_name=name)  # PEFT keeps and changes it
        except ValueError as error:
            raise ValueError(f"the adapter {name!r} does not fit this DiT: {error}") from error
        for layer in transformer.modules():
            if isinstance(layer, LoraLayer) and name in layer.lora_A:
                layer.lora_A[name].float()
                layer.lora_B[name].float()
        wanted = {
            key: tuple(weight.shape)
            for key, weight in get_peft_model_state_dict(transformer, adapter_name=name).items()
        }
        given = {key: tuple(weight.shape) for key, weight in adapter.weights.items()}
        differing = sorted(key for key in wanted.keys() | given.keys() if wanted.get(key) != given.get(key))
        if differing:
            first = differing[0]
            raise ValueError(
                f"the adapter {name!r} does not fit this DiT: {len(differing)} of its weights are not those its config "
                f"gives the DiT, the first {first} (of shape {given.get(first)} in the adapter, {wanted.get(first)} in "
                "the DiT)"
            )
        set_peft_model_state_dict(transformer, adapter.weights, adapter_name=name)
    if adapters:
        transformer.set_adapters(list(adapters), weights=[scale] * len(adapters))


def fuse_adapters(transformer: MossTransformer) -> None:
    """The active adapters of ``transformer`` merged into its weights at their strength (each change computed in the
    float32 of :func:`load_adapters` and rounded once to the weight's dtype) and their layers removed, in place: the DiT
    then computes as a plain one, which compiles, quantizes or runs from a package, and its adapters can no longer be
    switched or rescaled."""
    transformer.fuse_lora()
    transformer.unload_lora()
