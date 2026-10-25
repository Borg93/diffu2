"""Caption dropout: a share of the training items trained on the empty prompt in place of their caption, as diffusers'
FLUX.3 Klein trainer does it (``--caption_dropout``, 0.1 recommended), for FLUX.3's image-edit pairs and diffu's
text-to-image crops alike.

Classifier-free guidance at generation contrasts the caption with the empty prompt ``""``, encoded like any caption
(diffusers' ``Flux3TextEncoderStep``, diffu's ``DiffuTextEncoderStep``). A model trained only on captioned items never
sees that unconditional input, so guidance pushes against a prediction it never learned (diffu's first run drew noise
at guidance 4 and a page without guidance). NeMo's own ``cfg_dropout`` zeroes the text embeddings instead, an input
generation never makes; this swaps in the real encoding of ``""``, cached once beside the items
(``diffu_train.cache_pairs`` for FLUX.3's pairs, ``diffu_train.cache_diffu`` for diffu's crops). No NeMo import, so it
runs and tests off the NeMo environment; ``diffu_train.nemo`` wraps NeMo's datasets with it.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, Self

import torch
from pydantic import BaseModel, ConfigDict
from torch.utils.data import Dataset

EMPTY_PROMPT_FILE = "empty_prompt.pt"  # beside a cache's metadata.json

type Sample = Mapping[str, Any]  # one item of NeMo's ImageEditDataset or TextToImageDataset
type Item = dict[str, Any]  # an item as the dataset hands it on
type Blank = Callable[[Sample], Item]  # an item with the empty prompt in place of its caption


class SampleSource(Protocol):
    """What caption dropout draws its samples from: NeMo's ImageEditDataset or TextToImageDataset, or any sized,
    indexed source."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int, /) -> Sample: ...


def uniform() -> float:
    """A draw from the process's own ``random`` (by name, not a bound method: a spawned worker gets its own)."""
    return random.random()


class EmptyPrompt(BaseModel):
    """The empty prompt as a cache holds a caption: its embeddings and attention mask."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    embeddings: torch.Tensor  # [sequence, hidden]
    mask: torch.Tensor  # [sequence]

    @classmethod
    def load(cls, cache_dir: Path) -> Self:
        """The empty prompt cached in ``cache_dir``.

        Raises:
            FileNotFoundError: If the cache has none (it predates caption dropout: cache it again).
        """
        path = cache_dir / EMPTY_PROMPT_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: cache with diffu_train.cache_pairs (FLUX.3) or diffu_train.cache_diffu (diffu)"
            )
        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls(embeddings=data["prompt_embeddings"], mask=data["prompt_attention_mask"])

    def save(self, cache_dir: Path) -> None:
        """Write the empty prompt into ``cache_dir``, where ``load`` reads it."""
        torch.save(
            {"prompt_embeddings": self.embeddings, "prompt_attention_mask": self.mask}, cache_dir / EMPTY_PROMPT_FILE
        )

    def into_image_edit(self, sample: Sample) -> Item:
        """An image-edit pair (NeMo's ImageEditDataset item) with the empty prompt in place of its caption: target and
        reference stay, and its text token count follows the prompt."""
        return {
            **sample,
            "prompt_embeddings": self.embeddings,
            "prompt_attention_mask": self.mask,
            "metadata": {**sample["metadata"], "text_token_length": self.embeddings.shape[0]},
        }

    def into_text_to_image(self, sample: Sample) -> Item:
        """A text-to-image crop (NeMo's TextToImageDataset item, whose mask NeMo's collate makes from its length) with
        the empty prompt in place of its caption."""
        return {**sample, "prompt_embeds": self.embeddings, "prompt": ""}


class CaptionDropout(Dataset[Item]):
    """A NeMo dataset whose items each carry the empty prompt instead of their caption with probability ``share``,
    put in by ``blank`` (``EmptyPrompt.into_image_edit`` or ``into_text_to_image``).

    Each draw is ``draw()``, by default Python's ``random`` (``uniform``), which the DataLoader seeds once in each
    worker as it starts (NeMo's workers persist across epochs, so the draws run on from epoch to epoch): an item loses
    its caption in some epochs and not others. A resumed run's workers restart their streams from the run's seed.
    """

    def __init__(
        self, dataset: SampleSource, blank: Blank, *, share: float, draw: Callable[[], float] = uniform
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._blank = blank
        self._share = share
        self._draw = draw

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Item:
        sample = self._dataset[index]
        return self._blank(sample) if self._draw() < self._share else dict(sample)
