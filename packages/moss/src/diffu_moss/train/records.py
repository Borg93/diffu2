"""What a reference training writes beside its outputs, so each step reads what the step before it made: a cache's
record (``cache.json``, written last), a checkpoint's LoRA (``adapter.json`` beside its weights), an export's record
(``export.json`` beside a merged or fully fine-tuned DiT, which ``diffu-moss-init`` reads) and a full fine-tune's
progress (``progress.json`` in a checkpoint's state, which a resumed run starts from)."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from diffu_moss.train.pairs import PromptMix, SampleSize

CACHE_RECORD = "cache.json"
ADAPTER_RECORD = "adapter.json"
ADAPTER_WEIGHTS = "adapter.safetensors"
EXPORT_RECORD = "export.json"
PROGRESS_RECORD = "progress.json"


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


def _half(size: SampleSize) -> bool:
    """Whether a record's sample size is the first rule, which a record written before the rule leaves out."""
    return size is SampleSize.HALF


class AdapterRecord(JsonRecord):
    """A checkpoint's LoRA: its shape, the optimizer step it was saved at and how its training's samples were sized
    (left out when shrunk under half a megapixel, as a record was written before)."""

    lora: LoraSettings
    step: int
    sample_size: SampleSize = Field(default=SampleSize.HALF, exclude_if=_half)


class CacheRecord(JsonRecord):
    """How a cache was made: from which pairs and model, how many pairs it holds and how they were drawn into the mix
    of prompt forms (``mix``, ``diffu_moss.train.pairs.Variant``), how many pairs were left out for sharing a line with
    their reference, whether it holds the latents alone (no prompt embeddings: a training encodes them), how its
    samples were sized (``sample_size``), the share of its entries in the sample form asked to take a look change and
    how many took each kind (``diffu_moss.train.restyle``), and the share asked to take a hand change, how many took
    one, how many of those pages left no room for a slant (their changes of stroke and width alone) and why any others
    were refused. Each is left out of a cache without it (the named mix, the samples shrunk, no change), which is
    written as before."""

    pairs: str
    model: str
    seed: int
    pairs_read: int
    pairs_sharing_lines: int
    entries: dict[str, int]
    latents_only: bool = False
    restyle_share: float = Field(default=0.0, exclude_if=lambda share: share == 0)
    restyled: dict[str, int] = Field(default_factory=dict, exclude_if=lambda kinds: not kinds)
    mix: PromptMix = Field(default=PromptMix.NAMED, exclude_if=lambda mix: mix is PromptMix.NAMED)
    sample_size: SampleSize = Field(default=SampleSize.HALF, exclude_if=_half)
    hand_restyle_share: float = Field(default=0.0, exclude_if=lambda share: share == 0)
    hand_restyled: int = Field(default=0, exclude_if=lambda count: count == 0)
    hand_no_slant_room: int = Field(default=0, exclude_if=lambda count: count == 0)
    hand_refused: dict[str, int] = Field(default_factory=dict, exclude_if=lambda reasons: not reasons)


class ExportRecord(JsonRecord):
    """A trained DiT's provenance: the DiT it was trained from (``base``: the Base DiT, or a full fine-tune's checkpoint
    it went on from, whose own record names its), the checkpoint it is (or whose LoRA was merged into Base;
    ``adapter`` is None for a full fine-tune, whose own weights are the DiT), whether it was trained to take a
    reference image and how its samples were sized (``diffu-moss-init`` writes both into the model index:
    ``takes_reference``, and ``sample_size`` when not shrunk under half a megapixel, left out here then too)."""

    takes_reference: bool
    base: str
    checkpoint: str
    adapter: AdapterRecord | None = None
    sample_size: SampleSize = Field(default=SampleSize.HALF, exclude_if=_half)


class Progress(JsonRecord):
    """Where a full fine-tune stood when its state was saved: its last step, its settings (``TrainSettings`` as JSON),
    its random generator's state, and its place in the sample order (the current pass's order and how much of it was
    drawn), so a run resumed from the state draws on as the run would have."""

    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")

    step: int
    settings: str
    generator: bytes
    order: list[int]
    drawn: int
