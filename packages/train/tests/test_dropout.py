"""Caption dropout on items shaped like NeMo's image-edit and text-to-image items: a dropped item trains on the cached
empty prompt with its images, every other item as it was."""

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
    blank = EmptyPrompt.load(tmp_path).into_image_edit
    dataset = CaptionDropout(pairs, blank, share=0.1, draw=iter([0.05, 0.5]).__next__)

    dropped, kept = dataset[0], dataset[1]

    assert torch.equal(dropped["prompt_embeddings"], torch.full((80, 8), 7.0, dtype=torch.float16))
    assert torch.equal(dropped["prompt_attention_mask"], torch.ones(80, dtype=torch.long))
    assert dropped["metadata"] == {"target_token_length": 4, "context_token_lengths": [4], "text_token_length": 80}
    assert dropped["target_latent"] is pairs[0]["target_latent"]
    assert dropped["context_latents"] is pairs[0]["context_latents"]
    assert pairs[0]["metadata"]["text_token_length"] == 160  # the dataset's own sample is left as it was
    assert kept == pairs[1]


def test_caption_dropout_trains_a_dropped_crop_of_diffu_on_the_empty_prompt_and_keeps_the_rest() -> None:
    empty = EmptyPrompt(embeddings=torch.full((80, 8), 7.0), mask=torch.ones(80, dtype=torch.long))
    crops = [
        {"latent": torch.full((64, 2, 2), float(k)), "prompt_embeds": torch.full((160, 8), float(k))} for k in (0, 1)
    ]
    dataset = CaptionDropout(crops, empty.into_text_to_image, share=0.1, draw=iter([0.05, 0.5]).__next__)

    dropped, kept = dataset[0], dataset[1]

    assert torch.equal(dropped["prompt_embeds"], torch.full((80, 8), 7.0))
    assert dropped["latent"] is crops[0]["latent"]
    assert kept == crops[1]
