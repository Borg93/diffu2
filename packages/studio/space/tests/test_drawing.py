"""The demo's models on CPU: the Old Swedish LoRA beside a tiny DiT's own weights."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from peft import get_peft_model_state_dict
from safetensors.torch import save_file

from diffu_moss.train.loop import add_lora
from diffu_moss.train.records import ADAPTER_RECORD, ADAPTER_WEIGHTS, AdapterRecord, LoraSettings
from diffu_moss.transformer import MossTransformer
from pagedemo.drawing import lora_beside

TINY_DIT = {
    "dim": 32,
    "ffn_dim": 16,
    "num_heads": 2,
    "num_layers": 3,
    "text_dim": 8,
    "freq_dim": 8,
    "latent_channels": 4,
}
LORA = LoraSettings(rank=4, alpha=4, target_modules=r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)")


def _velocity(dit: MossTransformer) -> torch.Tensor:
    latents = torch.randn(1, 4, 3, 5, generator=torch.Generator().manual_seed(1))
    text = torch.randn(1, 6, 8, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        return dit(latents, torch.tensor([500.0]), text, torch.ones(1, 6, dtype=torch.bool))


def test_the_old_swedish_lora_beside_the_dit_draws_as_the_lora_does_sharing_every_weight_and_leaving_the_dit_as_it_was(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    dit = MossTransformer(**TINY_DIT).eval()
    trained = copy.deepcopy(dit)
    add_lora(trained, LORA)
    with torch.no_grad():
        for name, parameter in trained.named_parameters():
            if ".lora_" in name:
                parameter.normal_(0, 0.3)  # B no longer zero: the LoRA changes the velocity
    save_file(get_peft_model_state_dict(trained), str(tmp_path / ADAPTER_WEIGHTS))
    AdapterRecord(lora=LORA, step=750).write(tmp_path / ADAPTER_RECORD)
    before = _velocity(dit)

    beside = lora_beside(dit, tmp_path)

    assert torch.allclose(_velocity(beside), _velocity(trained), atol=1e-6)
    assert not torch.allclose(_velocity(beside), before, atol=1e-3)
    assert torch.equal(_velocity(dit), before)  # the DiT itself draws as before
    own = {parameter.data_ptr() for parameter in dit.parameters()}
    added = [name for name, parameter in beside.named_parameters() if parameter.data_ptr() not in own]
    assert added and all(".lora_" in name for name in added)  # the LoRA's weights are all it adds
