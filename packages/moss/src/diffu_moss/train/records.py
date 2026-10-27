"""What a reference training writes beside its outputs, so each step reads what the step before it made: a cache's
record (``cache.json``, written last), a checkpoint's LoRA (``adapter.json`` beside its weights) and an export's
record (``export.json`` beside the merged DiT, which ``diffu-moss-init`` reads)."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict

CACHE_RECORD = "cache.json"
ADAPTER_RECORD = "adapter.json"
ADAPTER_WEIGHTS = "adapter.safetensors"
EXPORT_RECORD = "export.json"


class JsonRecord(BaseModel):
    """A record kept as one JSON file."""

    model_config = ConfigDict(frozen=True)

    @classmethod
    def read(cls, path: Path) -> Self:
        """The record at ``path``."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def write(self, path: Path) -> None:
        """The record written to ``path``."""
        path.write_text(f"{self.model_dump_json(indent=1)}\n", encoding="utf-8")


class LoraSettings(JsonRecord):
    """A LoRA's shape: its rank, its alpha (the update is scaled by alpha / rank) and the modules it adapts (a regex
    matched in full against the DiT's module names)."""

    rank: int
    alpha: int
    target_modules: str


class AdapterRecord(JsonRecord):
    """A checkpoint's LoRA: its shape and the optimizer step it was saved at."""

    lora: LoraSettings
    step: int


class CacheRecord(JsonRecord):
    """How a cache was made: from which pairs and model, how many pairs it holds and how they were drawn into the mix
    of prompt forms (``diffu_moss.train.pairs.Variant``), and how many pairs were left out for sharing a line with their
    reference."""

    pairs: str
    model: str
    seed: int
    pairs_read: int
    pairs_sharing_lines: int
    entries: dict[str, int]


class ExportRecord(JsonRecord):
    """A merged DiT's provenance: the Base DiT it was merged into, the checkpoint and LoRA merged, and whether it was
    trained to take a reference image (``diffu-moss-init`` writes that into the model index's ``takes_reference``)."""

    takes_reference: bool
    base: str
    checkpoint: str
    adapter: AdapterRecord
