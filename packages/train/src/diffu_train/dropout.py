"""Caption dropout for FLUX.3 image-edit training: a share of the pairs trained on the empty prompt in place of their
caption, as diffusers' FLUX.3 Klein trainer does it (``--caption_dropout``, 0.1 recommended).

Classifier-free guidance at generation contrasts the caption with the empty prompt ``""``, encoded like any caption
(diffusers' ``Flux3TextEncoderStep``). Trained only on captioned pairs, the LoRA never sees that unconditional input
with a reference, so guidance pushes every drawing towards the words and away from what the reference alone would
give. NeMo's own ``cfg_dropout`` zeroes the text embeddings instead, an input generation never makes; this swaps in
the real encoding of ``""``, cached once beside the pairs (``diffu_train.cache_pairs``). No NeMo import, so it runs
and tests off the NeMo environment; ``diffu_train.nemo`` wraps NeMo's image-edit dataset with it.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

import torch
from pydantic import BaseModel, ConfigDict
from torch.utils.data import Dataset

EMPTY_PROMPT_FILE = "empty_prompt.pt"  # in a pair cache dir

type Sample = Mapping[str, Any]  # one item of NeMo's ImageEditDataset


class SampleSource(Protocol):
    """What caption dropout draws its samples from: NeMo's ImageEditDataset, or any sized, indexed source."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int, /) -> Sample: ...


def uniform() -> float:
    """A draw from the process's own ``random`` (by name, not a bound method: a spawned worker gets its own)."""
    return random.random()


class EmptyPrompt(BaseModel):
    """The empty prompt as NeMo's FLUX.3 text encoding caches a caption: its embeddings and attention mask."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    embeddings: torch.Tensor  # [sequence, hidden]
    mask: torch.Tensor  # [sequence]

    @classmethod
    def load(cls, cache_dir: Path) -> EmptyPrompt:
        """The empty prompt cached in ``cache_dir``.

        Raises:
            FileNotFoundError: If the cache has none (it predates caption dropout: run diffu_train.cache_pairs).
        """
        path = cache_dir / EMPTY_PROMPT_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: cache the pairs with diffu_train.cache_pairs to write it")
        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls(embeddings=data["prompt_embeddings"], mask=data["prompt_attention_mask"])

    def save(self, cache_dir: Path) -> None:
        """Write the empty prompt into ``cache_dir``, where ``load`` reads it."""
        torch.save(
            {"prompt_embeddings": self.embeddings, "prompt_attention_mask": self.mask}, cache_dir / EMPTY_PROMPT_FILE
        )


class CaptionDropout(Dataset[dict[str, Any]]):
    """A NeMo image-edit dataset whose samples each carry the empty prompt instead of their caption with probability
    ``share``: target and reference stay, and the sample's text token count follows its prompt.

    Each draw is ``draw()``, by default Python's ``random`` (``uniform``), which the DataLoader seeds in every worker
    from its generator each epoch: a pair loses its caption in some epochs and not others, reproducibly.
    """

    def __init__(
        self, dataset: SampleSource, empty: EmptyPrompt, share: float, draw: Callable[[], float] = uniform
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._empty = empty
        self._share = share
        self._draw = draw

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self._dataset[index])
        if self._draw() < self._share:
            sample["prompt_embeddings"] = self._empty.embeddings
            sample["prompt_attention_mask"] = self._empty.mask
            sample["metadata"] = {**sample["metadata"], "text_token_length": self._empty.embeddings.shape[0]}
        return sample
