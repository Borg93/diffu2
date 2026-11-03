"""Tiny random models of the readers' shapes, for the envs lane: run by a tool environment's own Python (the workspace's
transformers has no ``minicpm_ocr``), never imported by the tests.

    <env>/.venv/bin/python envs_tiny.py minicpm-ocr OUT    # MiniCPM-OCR 1.0's config shrunk, its tokenizer, processor

Only the configs, tokenizers and processors are fetched (at pinned commits), never weights.
"""

from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download

MINICPM_OCR = ("openbmb/MiniCPM-OCR-1.0", "2697d4e93dca82784abf858ae57206a49629cb6b")
FILES = ["*.json", "*.jinja", "*.txt", "*.model"]  # configs, tokenizer, processor, chat template: no weights


def _shrink_text(config: transformers.PretrainedConfig) -> transformers.PretrainedConfig:
    """``config`` (a Qwen3.5 language model's) as a copy of four full-attention layers and width 64: the linear
    attention layers' kernel (flash-linear-attention's) has no CPU path, and the lane runs on CPU."""
    text = copy.deepcopy(config)
    text.num_hidden_layers = 4
    text.layer_types = ["full_attention"] * 4
    text.hidden_size = 64
    text.intermediate_size = 128
    text.num_attention_heads = 2
    text.num_key_value_heads = 1
    text.head_dim = 32
    text.linear_num_key_heads = 2
    text.linear_num_value_heads = 2
    text.linear_key_head_dim = 16
    text.linear_value_head_dim = 16
    return text


def _copy_files(snapshot: Path, out: Path) -> None:
    """The snapshot's tokenizer, processor and template files beside the model (its config is the tiny one)."""
    for path in snapshot.iterdir():
        if path.is_file() and path.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(path, out / path.name)


def tiny_minicpm_ocr(out: Path) -> None:
    """A tiny random MiniCPM-OCR 1.0 written to ``out``: its pinned config with a four-layer language model
    (:func:`_shrink_text`) and a two-layer vision encoder of width 32, its weights drawn by seed 0, and the snapshot's
    tokenizer, processor and chat template beside it."""
    snapshot = Path(snapshot_download(MINICPM_OCR[0], revision=MINICPM_OCR[1], allow_patterns=FILES))
    config = transformers.AutoConfig.from_pretrained(snapshot)
    config.text_config = _shrink_text(config.text_config)
    vision = config.vision_config
    vision.num_hidden_layers = 2
    vision.hidden_size = 32
    vision.intermediate_size = 64
    vision.num_attention_heads = 2
    config.insert_layer_id = 0
    torch.manual_seed(0)
    model = getattr(transformers, config.architectures[0])(config)
    model.save_pretrained(out)
    _copy_files(snapshot, out)


TINY_MODELS = {"minicpm-ocr": tiny_minicpm_ocr}


if __name__ == "__main__":
    kind, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    TINY_MODELS[kind](out)
