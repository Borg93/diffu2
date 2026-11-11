"""The trainer's own options (:class:`TrainerOptions`), passed to it as one JSON (``--options``) beside transformers'
TrainingArguments flags; torch-free, so htr-train renders them where the trainer's environment is not."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# The README's generation budget (max_new_tokens 8744): a page whose answer is longer cannot be read whole anyway
MAX_TARGET_TOKENS = 8744


class TrainerOptions(BaseModel):
    """What the trainer does beyond TrainingArguments: the model (a Hub repo at its revision, or a local folder), the
    training sets and the dev set (``diffu-reader-set --model nemotron-parse`` samples), which parts learn (the decoder
    always; the RADIO vision encoder and the adapter between them, its convolutions and norms, unless frozen; the
    bundled one-step MTP head never: the forward does not use it, see :mod:`diffu_htr_train.nemotron_parse.model`), how
    much more a layout token (a box coordinate or a class) weighs in the loss than a text token (NVIDIA's recipe: 10),
    the longest answer kept (in tokens, its end included; a longer page is left out and counted), and whether the vision
    encoder's blocks and the decoder's layers recompute their activations in the backward pass."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    revision: str | None = None
    train: list[Path] = Field(min_length=1)
    dev: Path | None = None
    freeze_vision_encoder: bool
    freeze_adapter: bool
    coordinate_weight: float = Field(default=10.0, gt=0)
    max_target_tokens: int = Field(default=MAX_TARGET_TOKENS, gt=0)
    activation_checkpointing: bool = False
