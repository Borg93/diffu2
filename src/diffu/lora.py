"""LoRA (low-rank adapters) for Stage-3 fine-tuning of the Diffu backbone.

The backbone is a diffusers ``SD3Transformer2DModel`` (a ``PeftAdapterMixin``), so a LoRA adapter is
injected with peft's ``LoraConfig`` via ``transformer.add_adapter``. Only the adapter trains — the base
backbone, the frozen VAE / DINOv3 / content encoder, and the style resampler all stay frozen — so a run
adapts a pretrained model to a new hand/collection cheaply (a few-MB safetensors adapter, not a 17 GB
checkpoint). Load it back at inference with ``transformer.load_lora_adapter``.

    diffu-train --init-from base.safetensors --lora --lora-rank 16 --hf-dataset Teklia/IAM-line ...
    diffu-generate --ckpt base.safetensors --lora runs/lora/final --text "..." --style ref.png

peft is pinned (see pyproject); ``save_lora_adapter`` writes ``pytorch_lora_weights.safetensors``, which
``load_lora_adapter`` needs by ``weight_name`` (its default looks for a ``.bin``).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Diffu

# What save_lora_adapter writes; load_lora_adapter must be told this name (its default is a .bin).
LORA_WEIGHTS_NAME = "pytorch_lora_weights.safetensors"
ADAPTER_NAME = "default"
# SD3 JointTransformerBlock attention projections (image stream + joint "add" context stream) — the
# standard, well-tested LoRA surface for MMDiT. The custom 2D-RoPE processor reads these same nn.Linears
# (verified: LoRA grads flow through them), so rotary attention is adapted too.
DEFAULT_TARGETS: tuple[str, ...] = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
)


def add_lora(
    model: Diffu,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    targets: tuple[str, ...] | None = None,
    adapter_name: str = ADAPTER_NAME,
) -> int:
    """Inject a LoRA adapter into the backbone and freeze everything else. Returns trainable-param count.

    Call AFTER loading the base weights (``--init-from``) and BEFORE building the optimizer / compiling.
    ``init_lora_weights=True`` zero-inits the B matrix so the adapter is identity at start (the model
    output is unchanged until training moves it).
    """
    from peft import LoraConfig

    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(targets or DEFAULT_TARGETS),
        init_lora_weights=True,
    )
    model.backbone.transformer.add_adapter(cfg, adapter_name=adapter_name)
    # add_adapter alone leaves non-target modules trainable — freeze the WHOLE model, then re-enable only
    # the LoRA tensors, so a LoRA run trains the adapter and nothing else.
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad_(True)
            trainable += p.numel()
    return trainable


def save_lora(model: Diffu, out_dir: str, *, adapter_name: str = ADAPTER_NAME) -> str:
    """Write the backbone's LoRA adapter (safetensors) to ``out_dir``. Returns the weights path."""
    os.makedirs(out_dir, exist_ok=True)
    model.backbone.transformer.save_lora_adapter(out_dir, adapter_name=adapter_name)
    return os.path.join(out_dir, LORA_WEIGHTS_NAME)


def load_lora(
    model: Diffu,
    path: str,
    *,
    adapter_name: str = ADAPTER_NAME,
    scale: float = 1.0,
    weights_name: str = LORA_WEIGHTS_NAME,
) -> None:
    """Load a LoRA adapter (a dir or Hub id holding ``weights_name``) into the backbone for inference.

    ``scale`` in ``[0, 1+]`` controls how strongly the adapter is applied (``set_adapters`` scales the
    attention LoRA — our whole target surface, so it applies fully).
    """
    tr = model.backbone.transformer
    # prefix=None is REQUIRED: save_lora_adapter writes keys WITHOUT a "transformer." prefix, but
    # load_lora_adapter defaults to prefix="transformer" and would silently match nothing (the adapter
    # loads but is never applied). Verified: prefix=None -> active_adapters()==['default'], forward changes.
    tr.load_lora_adapter(path, adapter_name=adapter_name, weight_name=weights_name, prefix=None)
    if scale != 1.0:
        tr.set_adapters(adapter_name, weights=scale)
