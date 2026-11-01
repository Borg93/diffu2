"""The runner's own fixtures: a tiny HTR judge in place of the Swedish TrOCR model, so a run's actors judge every line
without loading the real one (1.3 GB)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from transformers import (
    GenerationConfig,
    TrOCRConfig,
    TrOCRProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
    ViTConfig,
)

from diffu_page.settings import settings

PROCESSOR_FILES = [
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]
TINY_IMAGE = 32  # px: a line box is read at 32 x 32, two 16-px patches a side


@pytest.fixture(scope="session")
def tiny_htr(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A TrOCR model directory as the judge reads one: a tiny ViT encoder and TrOCR decoder (random weights, greedy
    decoding) with the Swedish HTR model's own tokenizer and image processor (its pinned revision's, in the Hugging Face
    cache), reading each line box at 32 x 32."""
    snapshot = Path(
        snapshot_download(settings.htr_model, revision=settings.htr_revision, allow_patterns=PROCESSOR_FILES)
    )
    processor = TrOCRProcessor.from_pretrained(snapshot)
    processor.image_processor.size = {"height": TINY_IMAGE, "width": TINY_IMAGE}
    tokens = {"bos_token_id": 0, "pad_token_id": 1, "eos_token_id": 2}
    encoder = ViTConfig(
        image_size=TINY_IMAGE,
        patch_size=16,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    decoder = TrOCRConfig(
        vocab_size=len(processor.tokenizer),
        d_model=16,
        decoder_layers=1,
        decoder_attention_heads=2,
        decoder_ffn_dim=32,
        max_position_embeddings=256,
        decoder_start_token_id=2,
        **tokens,
    )
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder, decoder)
    config.decoder_start_token_id = 0
    config.pad_token_id = tokens["pad_token_id"]
    config.eos_token_id = tokens["eos_token_id"]
    torch.manual_seed(0)
    model = VisionEncoderDecoderModel(config)
    model.generation_config = GenerationConfig(decoder_start_token_id=0, num_beams=1, **tokens)
    out = tmp_path_factory.mktemp("htr")
    model.save_pretrained(out)
    processor.save_pretrained(out)
    return out
