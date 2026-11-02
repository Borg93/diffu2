"""MOSS-Image LoRAs as PEFT adapters on CPU (conftest's tiny DiT, random weights): diffu-moss-convert-lora writes a
LyCORIS LoRA of the DiT's modules as a PEFT LoRA adapter folder (its config naming every module, the rank and alpha most
have and any other module's, and the base it trained on; its weights as the file stores them, with where they came from
and no path); PEFT's own loader and the page pipeline's change each module by exactly alpha / rank * up @ down and
nothing else; a file it cannot convert exactly is refused before anything is written, and a folder that is no PEFT LoRA
adapter, or an adapter of another model, is refused when it is loaded."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from peft import PeftModel
from pydantic import BaseModel, ConfigDict
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from typer.testing import CliRunner

from diffu_moss.adapters import ADAPTER_CONFIG, ADAPTER_WEIGHTS, fuse_adapters, load_adapters, read_adapter
from diffu_moss.convert_lora import MOSS_IMAGE_BASE, app, convert_lycoris, dit_layout
from diffu_moss.train.loop import LORA_TARGETS
from diffu_moss.transformer import MossTransformer

if TYPE_CHECKING:
    from conftest import MossAdapterWriter, NemoLoraWriter

RANK, ALPHA = 4, 8.0  # most modules': a scale of 2
OWN_RANK = {"video_embedder": 2}  # a module of a rank of its own
OWN_ALPHA = {"layers.1.proj.ffn_down": 6.0}  # and one of an alpha of its own: a scale of 1.5
KOHAKU_METADATA = {  # an export's metadata as KohakuMossTrainer writes it
    "kohakumoss_algo": "lora",
    "kohakumoss_format": "lycoris",
    "kohakumoss_key_style": "lora_unet",
    "kohakumoss_prefix": "lora_unet",
    "kohakumoss_step": "700",
    "ss_network_alpha": "8.0",
    "ss_network_args": '{"algo": "lora", "bypass_mode": true}',
    "ss_network_dim": "4",
    "ss_network_module": "lycoris.kohya",
}
CONFIG_KEYS = (  # what of a PEFT LoRA config sets which modules change, and by how much
    "peft_type",
    "r",
    "lora_alpha",
    "target_modules",
    "rank_pattern",
    "alpha_pattern",
    "use_rslora",
    "use_dora",
    "bias",
    "base_model_name_or_path",
    "revision",
)

type ModuleLoras = dict[str, tuple[Tensor, Tensor, float]]  # by module name: down, up and alpha


def _key(module: str) -> str:
    """A module's key in a LyCORIS file of KohakuMossTrainer's."""
    return "lora_unet_" + module.replace(".", "_")


def _hand_lora(dit: MossTransformer) -> ModuleLoras:
    """A LoRA of ``dit`` as a hand LoRA adapts MOSS-Image's (every module ``LORA_TARGETS`` names), its down and up
    random in bfloat16 as KohakuMossTrainer stores them, at ``RANK`` and ``ALPHA`` but the modules of ``OWN_RANK`` and
    ``OWN_ALPHA``."""
    generator = torch.Generator().manual_seed(0)
    lora: ModuleLoras = {}
    for name, layer in dit.named_modules():
        if isinstance(layer, nn.Linear) and re.fullmatch(LORA_TARGETS, name):
            rank = OWN_RANK.get(name, RANK)
            down = torch.randn(rank, layer.in_features, generator=generator).to(torch.bfloat16)
            up = torch.randn(layer.out_features, rank, generator=generator).to(torch.bfloat16)
            lora[name] = (down, up, OWN_ALPHA.get(name, ALPHA))
    return lora


def _tensors(lora: ModuleLoras) -> dict[str, Tensor]:
    """The tensors of a LyCORIS file of ``lora``: each module's down, up and alpha (float32, as LyCORIS keeps it)."""
    tensors: dict[str, Tensor] = {}
    for name, (down, up, alpha) in lora.items():
        tensors[f"{_key(name)}.lora_down.weight"] = down
        tensors[f"{_key(name)}.lora_up.weight"] = up
        tensors[f"{_key(name)}.alpha"] = torch.tensor(alpha)
    return tensors


def _write_lycoris(path: Path, tensors: dict[str, Tensor], metadata: dict[str, str] = KOHAKU_METADATA) -> Path:
    save_file(tensors, str(path), metadata=metadata)
    return path


def test_diffu_moss_convert_lora_writes_a_peft_lora_config_of_the_dits_module_names_and_where_it_came_from(
    tiny_moss_dit: Path, tmp_path: Path
) -> None:
    lora = _hand_lora(dit_layout(tiny_moss_dit))
    lycoris = _write_lycoris(tmp_path / "lora-lycoris.safetensors", _tensors(lora))

    result = CliRunner().invoke(
        app, [str(lycoris), "--out", str(tmp_path / "adapter"), "--transformer", str(tiny_moss_dit)]
    )

    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "adapter" / ADAPTER_CONFIG).read_text(encoding="utf-8"))
    assert {key: config[key] for key in CONFIG_KEYS} == {
        "peft_type": "LORA",
        "r": RANK,
        "lora_alpha": ALPHA,
        "target_modules": list(lora),  # every module by its exact name, in the DiT's order
        "rank_pattern": {"^video_embedder": 2},
        "alpha_pattern": {r"^layers\.1\.proj\.ffn_down": 6.0},
        "use_rslora": False,
        "use_dora": False,
        "bias": "none",
        "base_model_name_or_path": "OpenMOSS-Team/MOSS-Image-Base",
        "revision": "85975793d8b3d432f1b6e27f352abaa4c2a4c0d3",
    }
    with safe_open(str(tmp_path / "adapter" / ADAPTER_WEIGHTS), "pt") as weights:
        metadata = weights.metadata()
        stored = {key: weights.get_tensor(key) for key in weights.keys()}  # noqa: SIM118
    assert metadata == {"format": "pt", "converted_from": "lycoris (kohakumoss)", "algo": "lora", "step": "700"}
    assert stored.keys() == {f"base_model.model.{name}.lora_{part}.weight" for name in lora for part in "AB"}
    assert {tensor.dtype for tensor in stored.values()} == {torch.bfloat16}  # as the file stores them
    assert all(
        torch.equal(stored[f"base_model.model.{name}.lora_A.weight"], down)
        and torch.equal(stored[f"base_model.model.{name}.lora_B.weight"], up)
        for name, (down, up, _) in lora.items()
    )


def _merged_by_peft(dit: MossTransformer, adapter: Path) -> MossTransformer:
    """``dit`` with the adapter loaded and merged by PEFT's own loader, as PEFT loads any adapter."""
    return PeftModel.from_pretrained(dit, str(adapter)).merge_and_unload()


def _fused_for_drawing(dit: MossTransformer, adapter: Path) -> MossTransformer:
    """``dit`` with the adapter loaded and fused as the page pipeline loads and fuses one."""
    load_adapters(dit, {"hand": read_adapter(adapter)}, scale=1.0)
    fuse_adapters(dit)
    return dit


@pytest.mark.parametrize(
    ("merged", "dtype"),
    [
        pytest.param(_merged_by_peft, torch.float32, id="peft-float32"),
        # bfloat16, as drawing holds the DiT: each merged weight rounded once, as a merge in float32 rounds it
        pytest.param(_fused_for_drawing, torch.bfloat16, id="draw-time-bfloat16"),
    ],
)
def test_converted_adapter_changes_each_module_by_exactly_alpha_over_rank_up_at_down_and_nothing_else(
    tiny_moss_dit: Path,
    tmp_path: Path,
    merged: Callable[[MossTransformer, Path], MossTransformer],
    dtype: torch.dtype,
) -> None:
    dit = MossTransformer.from_pretrained(str(tiny_moss_dit), torch_dtype=dtype)
    base = {key: tensor.clone() for key, tensor in dit.state_dict().items()}
    lora = _hand_lora(dit)
    lycoris = _write_lycoris(tmp_path / "lora.safetensors", _tensors(lora))
    convert_lycoris(lycoris, tmp_path / "adapter", transformer=dit, base=MOSS_IMAGE_BASE)

    weights = merged(dit, tmp_path / "adapter").state_dict()

    expected = dict(base)
    for name, (down, up, alpha) in lora.items():
        change = alpha / down.shape[0] * (up.float() @ down.float())
        expected[f"{name}.weight"] = (base[f"{name}.weight"].float() + change).to(dtype)
    assert weights.keys() == expected.keys()
    assert [key for key in expected if not torch.equal(weights[key], expected[key])] == []


class Conversion(BaseModel):
    """What a conversion is given: a LyCORIS file's tensors and metadata, the DiT it is of and the folder to write."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tensors: dict[str, Tensor]
    metadata: dict[str, str]
    dit: MossTransformer
    out: Path


def _another_algorithm(conversion: Conversion) -> None:
    conversion.metadata["kohakumoss_algo"] = "loha"


def _rs_lora(conversion: Conversion) -> None:
    conversion.metadata |= {"kohakumoss_algo": "lora", "ss_network_args": '{"algo": "lora", "rs_lora": true}'}


def _dora_tensor(conversion: Conversion) -> None:
    conversion.tensors[f"{_key('layers.0.proj.q')}.dora_scale"] = torch.ones(1, 32)


def _no_tensor(conversion: Conversion) -> None:
    conversion.tensors.clear()


def _no_up(conversion: Conversion) -> None:
    del conversion.tensors[f"{_key('layers.0.proj.q')}.lora_up.weight"]


def _a_layer_the_dit_lacks(conversion: Conversion) -> None:
    conversion.tensors |= _tensors({"layers.7.proj.q": (torch.zeros(4, 32), torch.zeros(32, 4), ALPHA)})


def _two_layers_of_one_name(conversion: Conversion) -> None:
    conversion.dit.add_module("layers_0_proj_q", nn.Linear(32, 32))


def _ranks_differ(conversion: Conversion) -> None:
    conversion.tensors[f"{_key('layers.0.proj.q')}.lora_up.weight"] = torch.zeros(32, 2)


def _other_widths(conversion: Conversion) -> None:
    conversion.tensors[f"{_key('layers.0.proj.q')}.lora_down.weight"] = torch.zeros(RANK, 16)


def _out_taken(conversion: Conversion) -> None:
    conversion.out.mkdir()
    (conversion.out / ADAPTER_CONFIG).write_text("{}", encoding="utf-8")


def _files(folder: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in folder.iterdir()} if folder.exists() else {}


@pytest.mark.parametrize(
    ("spoil", "error"),
    [
        pytest.param(_another_algorithm, "the file is a LyCORIS loha file, not a plain LoRA", id="loha"),
        pytest.param(_rs_lora, r"scaled as rsLoRA's", id="rslora"),
        pytest.param(_dora_tensor, r"dora_scale is no plain LoRA's tensor", id="dora-tensor"),
        pytest.param(_no_tensor, "the file holds no LoRA", id="empty"),
        pytest.param(_no_up, "lora_unet_layers_0_proj_q lacks its lora_up.weight", id="missing-tensor"),
        pytest.param(
            _a_layer_the_dit_lacks,
            r"lora_unet_layers_7_proj_q names 0 linear layers of the DiT \(none\), not one",
            id="no-layer",
        ),
        pytest.param(
            _two_layers_of_one_name,
            r"lora_unet_layers_0_proj_q names 2 linear layers of the DiT \(layers\.0\.proj\.q, layers_0_proj_q\)",
            id="two-layers",
        ),
        pytest.param(_ranks_differ, r"down \[4, 32\] and up \[32, 2\] differ in rank", id="ranks-differ"),
        pytest.param(_other_widths, r"do not fit layers\.0\.proj\.q, 32 in and 32 out", id="widths"),
        pytest.param(_out_taken, "holds an adapter already: give a new folder", id="out-taken"),
    ],
)
def test_convert_lycoris_refuses_a_file_it_cannot_convert_exactly_before_writing_anything(
    tiny_moss_dit: Path, tmp_path: Path, spoil: Callable[[Conversion], None], error: str
) -> None:
    dit = dit_layout(tiny_moss_dit)
    conversion = Conversion(
        tensors=_tensors(_hand_lora(dit)), metadata=dict(KOHAKU_METADATA), dit=dit, out=tmp_path / "adapter"
    )
    spoil(conversion)
    lycoris = _write_lycoris(tmp_path / "lora.safetensors", conversion.tensors, conversion.metadata)
    before = _files(conversion.out)

    with pytest.raises(ValueError, match=error):
        convert_lycoris(lycoris, conversion.out, transformer=conversion.dit, base=MOSS_IMAGE_BASE)

    assert _files(conversion.out) == before


def _no_weights(adapter: Path) -> None:
    (adapter / ADAPTER_WEIGHTS).unlink()


def _no_peft_type(adapter: Path) -> None:
    (adapter / ADAPTER_CONFIG).write_text(json.dumps({"r": 4, "lora_alpha": 4}), encoding="utf-8")


def _loha(adapter: Path) -> None:
    config = {"peft_type": "LOHA", "r": 4, "alpha": 4, "target_modules": ["layers.0.proj.q"]}
    (adapter / ADAPTER_CONFIG).write_text(json.dumps(config), encoding="utf-8")


def _unprefixed(adapter: Path) -> None:
    weights = load_file(adapter / ADAPTER_WEIGHTS)
    save_file(
        {key.removeprefix("base_model.model."): value for key, value in weights.items()}, adapter / ADAPTER_WEIGHTS
    )


@pytest.mark.parametrize(
    ("spoil", "error"),
    [
        pytest.param(_no_weights, "is not a LoRA adapter folder: it has no adapter_model.safetensors", id="no-weights"),
        pytest.param(_no_peft_type, "adapter_config.json is no PEFT adapter config", id="not-peft"),
        pytest.param(_loha, "holds no LoRA adapter: its config is a LoHaConfig", id="loha"),
        pytest.param(_unprefixed, r"2 weights are not keyed under PEFT's 'base_model\.model\.'", id="unprefixed"),
    ],
)
def test_read_adapter_refuses_a_folder_that_is_no_peft_lora_adapter(
    write_moss_adapter: MossAdapterWriter, spoil: Callable[[Path], None], error: str
) -> None:
    adapter = write_moss_adapter("layers.0.proj.q", torch.randn(RANK, 32), torch.randn(32, RANK), alpha=ALPHA)
    spoil(adapter)

    with pytest.raises(ValueError, match=error):
        read_adapter(adapter)


def _wider_dits_adapter(tmp_path: Path, tiny_moss_dit: Path, write_nemo_lora: NemoLoraWriter) -> Path:
    """An adapter of a DiT twice as wide as the tiny one, of the module the tiny one has by that name."""
    with torch.device("meta"):
        wider = MossTransformer.from_config({**MossTransformer.load_config(str(tiny_moss_dit)), "dim": 64})
    tensors = _tensors({"layers.0.proj.q": (torch.zeros(RANK, 64), torch.zeros(64, RANK), ALPHA)})
    adapter = tmp_path / "wider"
    convert_lycoris(
        _write_lycoris(tmp_path / "wider.safetensors", tensors), adapter, transformer=wider, base=MOSS_IMAGE_BASE
    )
    return adapter


def _flux3_lora(tmp_path: Path, tiny_moss_dit: Path, write_nemo_lora: NemoLoraWriter) -> Path:
    """A FLUX.3 Klein LoRA (a NeMo checkpoint): a PEFT adapter folder too, of modules no MOSS DiT has."""
    return write_nemo_lora("single_transformer_blocks.0.attn.to_q", torch.zeros(2, 8), torch.zeros(8, 2), alpha=2)


@pytest.mark.parametrize(
    ("adapter_of", "error"),
    [
        pytest.param(
            _wider_dits_adapter,
            r"2 of its weights are not those its config gives the DiT, the first layers\.0\.proj\.q\.lora_A\.weight "
            r"\(of shape \(4, 64\) in the adapter, \(4, 32\) in the DiT\)",
            id="wider-dit",
        ),
        pytest.param(_flux3_lora, "Target modules .* not found in the base model", id="flux3"),
    ],
)
def test_load_adapters_refuses_an_adapter_of_another_model_by_its_name(
    tiny_moss_dit: Path,
    tmp_path: Path,
    write_nemo_lora: NemoLoraWriter,
    adapter_of: Callable[[Path, Path, NemoLoraWriter], Path],
    error: str,
) -> None:
    adapter = read_adapter(adapter_of(tmp_path, tiny_moss_dit, write_nemo_lora))
    dit = MossTransformer.from_pretrained(str(tiny_moss_dit))

    with pytest.raises(ValueError, match=f"the adapter 'hand' does not fit this DiT: .*{error}"):
        load_adapters(dit, {"hand": adapter}, scale=1.0)
