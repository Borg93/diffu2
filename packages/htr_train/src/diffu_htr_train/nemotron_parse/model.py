"""The model as the trainer trains it (its remote code, ``hf_nemotron_parse_modeling.py`` at the pinned revision):

- loaded in float32, so the optimizer updates float32 weights (the forward runs in bfloat16 under the Trainer's
  autocast, ``--bf16 true``); 631 M parameters, 2.5 GB, about 10 GB with AdamW's state;
- its parts (:func:`freeze`): ``encoder.model_encoder``, the C-RADIOv4-SO400M vision encoder (27 blocks of 1152 over
  every 16 x 16 patch); the adapter, the rest of ``encoder`` (its 1D convolutions, norms and summary projection, which
  bring the page's features to the decoder's 1,024); and ``decoder`` with ``lm_head`` (tied to its embedding);
- the bundled one-step MTP head (``decoder.extra_heads``, ``decoder.extra_proj``) is never trained: its forward does
  not use it (only vLLM's MTP proposer does), so it gets no gradient; it is frozen, so DDP has no parameter without one,
  and saved as shipped. After fine-tuning, vLLM's one-token MTP proposals will be accepted less often (the output is
  the same; only its speed);
- the model does not support transformers' gradient checkpointing (``supports_gradient_checkpointing`` is unset), so
  :func:`checkpoint_activations` recomputes each vision block and decoder layer in the backward pass itself, its
  parameter names unchanged;
- each checkpoint gets the model's tokenizer, processor, generation config and helpers beside its weights
  (:class:`CopyModelFiles`), so it is read and served as the model's own repo is. ``save_pretrained`` writes the
  weights, the config and the model's code (its ``auto_map``), but names the architecture by its Python class
  (``NemotronParseForConditionalGeneration``); the repo's name is put back (``RadioMBartPrefixForCausalLM``, which
  NVIDIA's vLLM plugin registers and serves). A checkpoint's weights are float32 (serve it with ``--dtype bfloat16``);
  the model a run ends with is written in bfloat16, as the repo ships it (:func:`export`).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, PreTrainedModel, TrainerCallback

from diffu_htr_train.nemotron_parse.options import TrainerOptions

# The files a checkpoint takes from the model's repo: its tokenizer, processor and generation config, its code, and
# the helpers it is served and read with (postprocessing, the vLLM plugin), never its weights, config or draft
MODEL_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "processor_config.json",
    "generation_config.json",
    "chat_template.jinja",
    "hf_*.py",
    "postprocessing.py",
    "latex2html.py",
    "nemotron_parse/*",
    "vllm_tied_patch/*",
    "pyproject.toml",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "TOKENIZER_ATTRIBUTION.md",
)


def load_model(options: TrainerOptions) -> PreTrainedModel:
    """The model ``options`` names, in float32: a Hub repo at its revision, or a local folder (a checkpoint of ours)."""
    return AutoModel.from_pretrained(
        options.model, revision=options.revision, trust_remote_code=True, dtype=torch.float32
    )


def model_files(options: TrainerOptions) -> Path:
    """A folder holding the model's :data:`MODEL_FILES`: its Hub snapshot at the revision (those files alone fetched),
    or the local folder it is."""
    local = Path(options.model).expanduser()
    if local.is_dir():
        return local
    return Path(snapshot_download(options.model, revision=options.revision, allow_patterns=list(MODEL_FILES)))


def _parts(model: nn.Module) -> dict[str, list[nn.Module]]:
    """The model's parts by name: the vision encoder, the adapter, the MTP head (the decoder and LM head are the
    rest)."""
    encoder: Any = model.encoder
    decoder: Any = model.decoder
    adapter = [module for name, module in encoder.named_children() if name != "model_encoder"]
    return {
        "vision_encoder": [encoder.model_encoder],
        "adapter": adapter,
        "mtp_head": [decoder.extra_heads, decoder.extra_proj],
    }


def freeze(model: nn.Module, *, vision_encoder: bool, adapter: bool) -> dict[str, int]:
    """The vision encoder and the adapter frozen as asked, the MTP head always: the trainable parameters by part
    afterwards (``decoder`` the decoder and its tied LM head)."""
    parts = _parts(model)
    for name, frozen in (("vision_encoder", vision_encoder), ("adapter", adapter), ("mtp_head", True)):
        for module in parts[name]:
            module.requires_grad_(not frozen)
    counts = {
        name: sum(p.numel() for module in modules for p in module.parameters() if p.requires_grad)
        for name, modules in parts.items()
    }
    decoder: Any = model.decoder
    counts["decoder"] = sum(
        p.numel() for name, p in decoder.named_parameters() if p.requires_grad and not name.startswith("extra_")
    )
    return counts


def _recomputed(module: nn.Module) -> None:
    """``module``'s forward recomputed in the backward pass while it trains (non-reentrant: other arguments, such as
    the decoder's mask, pass through), as it is otherwise."""
    forward: Callable[..., Any] = module.forward

    def recomputing(*args: Any, **kwargs: Any) -> Any:
        if module.training and torch.is_grad_enabled():
            return checkpoint(forward, *args, use_reentrant=False, **kwargs)
        return forward(*args, **kwargs)

    module.forward = recomputing


def checkpoint_activations(model: nn.Module) -> int:
    """Every vision block and decoder layer recomputed in the backward pass (:func:`_recomputed`): how many."""
    encoder: Any = model.encoder
    decoder: Any = model.decoder
    modules = [*encoder.model_encoder.radio_model.model.blocks, *decoder.layers]
    for module in modules:
        _recomputed(module)
    return len(modules)


def copy_model_files(source: Path, folder: Path) -> list[str]:
    """The :data:`MODEL_FILES` of ``source`` copied into ``folder``, but those already there (what ``save_pretrained``
    wrote): their paths in ``folder``."""
    copied: list[str] = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix()
        if not path.is_file() or not any(fnmatch(relative, pattern) for pattern in MODEL_FILES):
            continue
        target = folder / relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(relative)
    return copied


def as_the_repo(folder: Path, source: Path, architectures: list[str]) -> None:
    """A saved model in ``folder`` made to read as the model's repo does: its config naming ``architectures`` (the
    repo's, which ``save_pretrained`` replaced by the Python class's) and its other files beside it
    (:func:`copy_model_files` from ``source``)."""
    config = folder / "config.json"
    saved = json.loads(config.read_text(encoding="utf-8"))
    saved["architectures"] = architectures
    config.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    copy_model_files(source, folder)


class CopyModelFiles(TrainerCallback):
    """Each checkpoint made to read as the model's repo does (:func:`as_the_repo`), by the first process."""

    def __init__(self, source: Path, architectures: list[str]) -> None:
        self.source = source
        self.architectures = architectures

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.is_world_process_zero:
            folder = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            as_the_repo(folder, self.source, self.architectures)


def export(trainer: Any, source: Path, architectures: list[str]) -> Path:
    """The model a run ends with (the best by eval loss, when the Trainer loaded it back) written to the run's output
    folder in bfloat16, as the repo ships its weights, and made to read as the repo does (:func:`as_the_repo`): that
    folder."""
    trainer.model.to(torch.bfloat16)
    trainer.save_model()
    folder = Path(trainer.args.output_dir)
    if trainer.is_world_process_zero():
        as_the_repo(folder, source, architectures)
    return folder
