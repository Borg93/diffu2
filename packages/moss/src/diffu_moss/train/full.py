"""The full fine-tune of ``diffu-moss-train-ref --full``: every weight of MOSS-Image's DiT trained on the reference
pairs, by MOSS-Image's own optimizer recipe (its technical report, Appendix A.2 and its Table 7, which
KohakuMossTrainer's full fine-tune follows too):

- torch's Muon on the joint blocks' weight matrices: Nesterov momentum 0.95, 5 Newton-Schulz steps, its step matched to
  AdamW's RMS (``match_rms_adamw``); AdamW on every other parameter, betas (0.9, 0.95), eps 1e-8; both at 5e-5,
  constant after the warmup;
- weight decay 0.01, but none on biases, norms and the embedders (the latents', the prompt's and the timestep's);
- the last joint block's q/k norm weights decayed towards one instead, ``w -= lr * 0.01 * (w - 1)``;
- gradients clipped at 1 (the loop's), and an exponential moving average of the weights (the report's 0.9999) when
  asked for: off by default, as its float32 copy is 20 GB more than the Spark holds.

The weights stay in float32 and the forward runs in bfloat16 autocast, its weight cache off so a block's bfloat16
copies of its weights live only while it runs; the joint blocks' activations are checkpointed. That is the layout
FSDP2's MixedPrecisionPolicy takes, with no float32 master copy of our own. On the real DiT (5.08 B parameters, 4.76 B
of them the joint blocks' matrices) it holds 20.3 GB of weights, 20.3 GB of gradients, 19.0 GB of Muon momentum and
2.6 GB of AdamW moments: about 73 GB at its peak with the activations of a 1 MP target and a 0.5 MP reference.

A checkpoint (``checkpoints/step_NNNNNN/``) holds the DiT itself: ``transformer/``, its weights in bfloat16 as a
diffusers folder with its ``export.json``, which ``diffu-moss-init --transformer`` draws with as it is; ``ema/``, the
same of the average, when kept; and with ``save_state`` ``state/``, what a resumed run starts from (about 42 GB): the
weights in float32 (a diffusers folder too), the average's (``ema.safetensors``), the optimizer's state
(``optimizer.pt``) and, written last, the loop's place (``progress.json``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Self

import torch
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR

from diffu_moss.train.records import EXPORT_RECORD, PROGRESS_RECORD, ExportRecord, Progress
from diffu_moss.transformer import MossTransformer

COMPUTE_DTYPE = torch.bfloat16
EMBEDDERS = ("video_embedder.", "text_embedder.proj.", "time_embedding.")  # the latents', the prompt's, the timestep's
DIT = "transformer"
AVERAGE = "ema"
STATE = "state"
AVERAGE_WEIGHTS = "ema.safetensors"
OPTIMIZER_STATE = "optimizer.pt"


class FullSettings(BaseModel):
    """How a full fine-tune steps (the module's docstring): its optimizers' settings, the decay of the average of its
    weights (None: no average), and whether each checkpoint keeps the state a run resumes from."""

    model_config = ConfigDict(frozen=True)

    learning_rate: float = 5e-5
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    momentum: float = 0.95
    weight_decay: float = 0.01
    ema: float | None = Field(default=None, gt=0, lt=1)
    save_state: bool = False


def full_transformer(folder: Path, *, device: str) -> MossTransformer:
    """The DiT at ``folder`` (a diffusers-format folder) in float32 on ``device``, every weight trained, its joint
    blocks checkpointed."""
    transformer = MossTransformer.from_pretrained(str(folder), torch_dtype=torch.float32)
    transformer.requires_grad_(True)
    transformer.enable_gradient_checkpointing()
    return transformer.to(device).train()


class FullOptimizer:
    """MOSS-Image's optimizer over every weight of ``transformer`` (the module's docstring), each step's learning rate
    ``settings.learning_rate`` times ``schedule`` of the steps done.

    Attributes:
        muon: torch's Muon over the joint blocks' weight matrices.
        adamw: AdamW over the rest, in two groups: weight-decayed, then not (biases, norms, embedders).
    """

    def __init__(
        self, transformer: MossTransformer, settings: FullSettings, *, schedule: Callable[[int], float]
    ) -> None:
        matrices, decayed, undecayed = [], [], []
        for name, parameter in transformer.named_parameters():
            if name.startswith("layers.") and parameter.ndim == 2:
                matrices.append((name, parameter))
            elif parameter.ndim < 2 or name.startswith(EMBEDDERS):
                undecayed.append((name, parameter))
            else:
                decayed.append((name, parameter))
        self.muon = torch.optim.Muon(
            matrices,
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
            momentum=settings.momentum,
            nesterov=True,
            adjust_lr_fn="match_rms_adamw",
        )
        self.adamw = torch.optim.AdamW(
            [{"params": decayed, "weight_decay": settings.weight_decay}, {"params": undecayed, "weight_decay": 0.0}],
            lr=settings.learning_rate,
            betas=settings.betas,
            eps=settings.eps,
        )
        last = len(transformer.layers) - 1
        towards_one = {f"layers.{last}.q_norm.weight", f"layers.{last}.k_norm.weight"}
        self._towards_one = [parameter for name, parameter in transformer.named_parameters() if name in towards_one]
        self._weight_decay = settings.weight_decay
        self._schedules = [LambdaLR(optimizer, schedule) for optimizer in (self.muon, self.adamw)]

    @property
    def learning_rate(self) -> float:
        """The learning rate the next step takes."""
        return float(self.adamw.param_groups[0]["lr"])

    def step(self) -> None:
        """Every weight stepped from its gradient, then the learning rate scheduled for the next step."""
        learning_rate = self.learning_rate
        with torch.no_grad():
            for weight in self._towards_one:  # from the weight before its step, as AdamW decays a weight
                weight.sub_(weight - 1, alpha=learning_rate * self._weight_decay)
        self.muon.step()
        self.adamw.step()
        for schedule in self._schedules:
            schedule.step()

    def zero_grad(self) -> None:
        """Every gradient dropped."""
        self.muon.zero_grad(set_to_none=True)
        self.adamw.zero_grad(set_to_none=True)

    def state_dict(self) -> dict[str, Any]:
        """The optimizers' and their schedules' states."""
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
            "schedules": [schedule.state_dict() for schedule in self._schedules],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """The optimizers and their schedules set to ``state`` (:meth:`state_dict`'s)."""
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])
        for schedule, saved in zip(self._schedules, state["schedules"], strict=True):
            schedule.load_state_dict(saved)


class WeightAverage:
    """An exponential moving average of a DiT's weights in float32: each :meth:`update` moves every averaged weight
    ``1 - decay`` of the way to the DiT's (``average + (1 - decay) * (weight - average)``).

    Attributes:
        weights: The averaged weights, by the DiT's parameter names.
        decay: The share of the average an update keeps.
    """

    def __init__(self, weights: dict[str, Tensor], decay: float) -> None:
        self.weights = weights
        self.decay = decay

    @classmethod
    def of(cls, transformer: MossTransformer, decay: float) -> Self:
        """An average starting at ``transformer``'s weights."""
        return cls({name: weight.detach().float().clone() for name, weight in transformer.named_parameters()}, decay)

    def update(self, transformer: MossTransformer) -> None:
        """Every averaged weight moved towards ``transformer``'s."""
        with torch.no_grad():
            for name, weight in transformer.named_parameters():
                self.weights[name].lerp_(weight, 1 - self.decay)


def _save_dit(
    transformer: MossTransformer, weights: Mapping[str, Tensor], directory: Path, record: ExportRecord
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    transformer.save_config(str(directory))
    bfloat16 = {name: weight.detach().to("cpu", torch.bfloat16).contiguous() for name, weight in weights.items()}
    save_file(bfloat16, str(directory / SAFETENSORS_WEIGHTS_NAME))
    record.write(directory / EXPORT_RECORD)


class FullTraining:
    """A full fine-tune in training (the module's docstring), from the DiT at ``base`` or, with ``resume``, from that
    checkpoint's state: the DiT's weights, the optimizer's state and the average's as they were saved.

    Attributes:
        transformer: The DiT, every weight trained in float32.
        parameters: Its weights.
        optimizer: Their optimizer.
        average: The average of its weights, or None.
    """

    compute_dtype: torch.dtype | None = COMPUTE_DTYPE

    def __init__(
        self,
        base: Path,
        settings: FullSettings,
        *,
        schedule: Callable[[int], float],
        device: str,
        resume: Path | None = None,
    ) -> None:
        self._base = base
        self._save_state = settings.save_state
        state = None if resume is None else resume / STATE
        self.transformer = full_transformer(base if state is None else state, device=device)
        self.parameters: list[nn.Parameter] = list(self.transformer.parameters())
        self.optimizer = FullOptimizer(self.transformer, settings, schedule=schedule)
        self.average: WeightAverage | None = None
        if state is not None:
            self.optimizer.load_state_dict(torch.load(state / OPTIMIZER_STATE, map_location=device, weights_only=True))
        if settings.ema is None:
            return
        if state is None:
            self.average = WeightAverage.of(self.transformer, settings.ema)
        else:
            self.average = WeightAverage(load_file(str(state / AVERAGE_WEIGHTS), device=device), settings.ema)

    def step(self) -> float:
        """One step of every weight from its gradient (the gradients then dropped), and the average moved after it.

        Returns:
            The learning rate it stepped at.
        """
        learning_rate = self.optimizer.learning_rate
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.average is not None:
            self.average.update(self.transformer)
        return learning_rate

    def save(self, directory: Path, progress: Progress) -> None:
        """The checkpoint of ``progress``'s step into ``directory`` (the module's docstring)."""
        record = ExportRecord(takes_reference=True, base=str(self._base.resolve()), checkpoint=str(directory.resolve()))
        _save_dit(self.transformer, dict(self.transformer.named_parameters()), directory / DIT, record)
        if self.average is not None:
            _save_dit(self.transformer, self.average.weights, directory / AVERAGE, record)
        if not self._save_state:
            return
        state = directory / STATE
        (state / PROGRESS_RECORD).unlink(missing_ok=True)  # a stopped run's, until this state is whole
        self.transformer.save_pretrained(str(state))
        if self.average is not None:
            save_file(
                {name: weight.contiguous() for name, weight in self.average.weights.items()},
                str(state / AVERAGE_WEIGHTS),
            )
        torch.save(self.optimizer.state_dict(), state / OPTIMIZER_STATE)
        progress.write(state / PROGRESS_RECORD)
