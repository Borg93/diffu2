"""Fixtures shared by every workspace member's tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from huggingface_hub import snapshot_download

TINY_FLUX3 = "hf-internal-testing/tiny-flux3-image-modular"
TINY_FLUX3_REVISION = "9130f96f124d6c6fdd1beab63cfb6d2f1c8658a1"  # pinned: a changed fixture must not change results


@pytest.fixture(scope="session")
def tiny_model() -> Path:
    """diffusers' tiny FLUX.3 test model (a few MB), downloaded once into the Hugging Face cache."""
    return Path(snapshot_download(TINY_FLUX3, revision=TINY_FLUX3_REVISION))
