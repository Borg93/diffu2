"""Caption dropout on samples shaped like NeMo's image-edit items: a dropped pair trains on the cached empty prompt
with its target and reference, every other pair as it was."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from diffu_train.dropout import CaptionDropout, EmptyPrompt


def _sample(k: int) -> dict[str, Any]:
    """A pair as NeMo's ImageEditDataset yields it, its caption 160 tokens long."""
    return {
        "target_latent": torch.full((4, 2, 2), float(k)),
        "context_latents": [torch.full((4, 2, 2), -float(k))],
        "prompt_embeddings": torch.full((160, 8), float(k), dtype=torch.float16),
        "prompt_attention_mask": torch.ones(160, dtype=torch.long),
        "conditioning_tensors": {},
        "metadata": {"target_token_length": 4, "context_token_lengths": [4], "text_token_length": 160},
    }


def test_caption_dropout_trains_a_dropped_pair_on_the_empty_prompt_and_keeps_the_rest(tmp_path: Path) -> None:
    EmptyPrompt(embeddings=torch.full((80, 8), 7.0, dtype=torch.float16), mask=torch.ones(80, dtype=torch.long)).save(
        tmp_path
    )
    pairs = [_sample(0), _sample(1)]
    dataset = CaptionDropout(pairs, EmptyPrompt.load(tmp_path), share=0.1, draw=iter([0.05, 0.5]).__next__)

    dropped, kept = dataset[0], dataset[1]

    assert torch.equal(dropped["prompt_embeddings"], torch.full((80, 8), 7.0, dtype=torch.float16))
    assert torch.equal(dropped["prompt_attention_mask"], torch.ones(80, dtype=torch.long))
    assert dropped["metadata"] == {"target_token_length": 4, "context_token_lengths": [4], "text_token_length": 80}
    assert dropped["target_latent"] is pairs[0]["target_latent"]
    assert dropped["context_latents"] is pairs[0]["context_latents"]
    assert pairs[0]["metadata"]["text_token_length"] == 160  # the dataset's own sample is left as it was
    assert kept == pairs[1]
