"""The loss NVIDIA fine-tunes Nemotron-Parse with: NeMo AutoModel's ``NemotronParseLoss`` (at ``58487ab5``,
``nemo_automodel/components/models/nemotron_parse/nemotron_parse_loss.py``; its tutorial ``tutorials/nemotron-parse``
sets ``coordinate_weight: 10``): token cross-entropy in float32, a layout token's ten times a text token's, summed and
divided by the number of tokens learnt (unweighted), over the whole optimizer step when transformers' Trainer counts
them (``num_items_in_batch``: every micro-batch of the step, on every device).

One difference, on purpose: theirs weighs every label id from ``class_token_start_idx`` on (50000 in the tutorial),
which in 1.1's vocabulary are its control, coordinate and class tokens. In 3.5's they are not: ids 50011-52314 are its
2,304 coordinates, 52315-52330 its classes and ``\\<|unk|\\>``, and ids from 52331 to 72504 characters and punctuation
added later (``' -'``, ``' /'``, ``' ('`` and ``„`` among them: 0.13 % of our pages' text tokens). So the threshold
would weigh punctuation ten times, and 3.5's own ``class_token_start_idx`` (52315) would weigh no coordinate at all.
Here the layout tokens are named: every ``<x_...>``, ``<y_...>`` and ``<class_...>`` in its tokenizer
(:func:`layout_mask`).
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
import torch.nn.functional as F

from diffu_htr_train.nemotron_parse.data import IGNORE

LAYOUT_PREFIXES = ("<x_", "<y_", "<class_")  # its coordinate and class tokens


class Vocabulary(Protocol):
    """The part of the tokenizer the loss reads: its tokens and their ids."""

    def get_vocab(self) -> dict[str, int]: ...


def layout_mask(tokenizer: Vocabulary, vocab_size: int) -> torch.Tensor:
    """For each of the model's ``vocab_size`` output ids, whether it is a layout token (a coordinate or a class)."""
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    ids = [index for token, index in tokenizer.get_vocab().items() if token.startswith(LAYOUT_PREFIXES)]
    mask[torch.tensor(ids, dtype=torch.long)] = True
    return mask


class LayoutWeightedLoss:
    """transformers' ``compute_loss_func``: the answer's tokens' cross-entropy (each position predicting the next
    label, :data:`IGNORE` skipped), a layout token's (``layout``, :func:`layout_mask`) ``weight`` times, summed and
    divided by the tokens learnt: the step's, when the Trainer counts them, else this batch's."""

    def __init__(self, layout: torch.Tensor, weight: float) -> None:
        self.layout = layout
        self.weight = weight

    def __call__(
        self, outputs: Any, labels: torch.Tensor, num_items_in_batch: torch.Tensor | int | None = None
    ) -> torch.Tensor:
        logits = outputs.logits[:, :-1].float()
        targets = labels[:, 1:].to(logits.device)
        losses = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=IGNORE, reduction="none")
        layout = self.layout.to(targets.device)[targets.clamp(min=0)]
        weighted = torch.where(layout, losses * self.weight, losses)
        count = (targets != IGNORE).sum() if num_items_in_batch is None else num_items_in_batch
        return weighted.sum() / count
