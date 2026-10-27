"""diffu-moss-train-ref: a LoRA that teaches MOSS-Image's DiT to draw from a reference image, trained on a cache of
style pairs (``diffu-moss-cache-pairs``).

    diffu-moss-train-ref --transformer <MOSS-Image-Base snapshot>/transformer --cache data_flux3/cache_moss_ref_v3d \\
        --val-cache data_flux3/cache_moss_ref_v3d_val --out runs/moss_ref_pilot/lora --steps 1500

The recipe of the 1715 hand LoRA (KohakuMossTrainer's, for parity): LoRA rank 32, alpha 32, on the attention (q, k, v,
o) and feed-forward (ffn_up_gate, ffn_down) of every joint block, the text refiner's blocks (qkv, proj, ffn_gate,
ffn_up, ffn_down), the prompt's projection (text_embedder.proj), the latents' embedder (video_embedder) and the output
(final_linear_video); the Base DiT frozen in bfloat16, the LoRA in float32; AdamW at 1e-4, betas (0.9, 0.999), weight
decay 1e-4, gradients clipped at 1, the learning rate constant after 50 warmup steps; one sample per micro-batch, two
per step. Each sample is the drawn image's clean latents noised to a level drawn as MOSS-Image draws it
(:mod:`diffu_moss.train.sigmas`), and the loss is flow matching's: the mean squared error between the DiT's velocity and
noise minus image over the drawn image's tokens; a reference enters clean and is not predicted
(:func:`flow_matching_loss`). The joint blocks' activations are checkpointed, as a reference adds half the drawn
image's tokens again (a 1 MP target, a 0.5 MP reference).

Writes under ``--out``: ``train_log.jsonl`` (a line a step: loss, learning rate, gradient norm, each sample's pair,
form and noise level; a validation's losses by form on its own lines), and every ``--save-every`` steps and at the end
``checkpoints/step_NNNNNN/``: the LoRA's weights (``adapter.safetensors``) and shape (``adapter.json``), which
``diffu-moss-export-ref`` merges into the DiT. With ``--val-cache`` (a cache of every form, ``--every-form``) the loss
of each form is scored every ``--val-every`` steps at a fixed noise level and noise per pair (the same in every run),
the same for a pair's three forms, so the forms' losses compare: does the reference lower it?
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from statistics import fmean
from typing import Annotated

import torch
import typer
from peft import LoraConfig, get_peft_model_state_dict
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from safetensors.torch import save_file
from torch import Tensor
from torch.nn import functional as F

from diffu_moss.train.cache import CachedSample, CacheEntry, load_sample, read_entries
from diffu_moss.train.records import ADAPTER_RECORD, ADAPTER_WEIGHTS, AdapterRecord, LoraSettings
from diffu_moss.train.sigmas import draw_sigmas
from diffu_moss.transformer import MossTransformer

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
console = Console()

LORA_TARGETS = (
    r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)"
    r"|text_embedder\.blocks\.\d+\.(qkv|proj|ffn_gate|ffn_up|ffn_down)"
    r"|text_embedder\.proj|video_embedder|final_linear_video"
)
LOG = "train_log.jsonl"
CHECKPOINTS = "checkpoints"
VALIDATION_SEED = 0  # every run scores at the same noise, so runs of different seeds compare


class TrainSettings(BaseModel):
    """How a reference LoRA is trained (the module's docstring): its shape, the optimizer, the steps and samples, and
    when it is saved and scored."""

    model_config = ConfigDict(frozen=True)

    steps: int = Field(default=1500, ge=1)
    rank: int = Field(default=32, ge=1)
    alpha: int = Field(default=32, ge=1)
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup: int = Field(default=50, ge=1)
    grad_accumulation: int = Field(default=2, ge=1)
    save_every: int = Field(default=250, ge=1)
    val_every: int = Field(default=250, ge=1)
    seed: int = 0
    device: str = "cuda"

    @property
    def lora(self) -> LoraSettings:
        """The LoRA's shape."""
        return LoraSettings(rank=self.rank, alpha=self.alpha, target_modules=LORA_TARGETS)


class StepLog(BaseModel):
    """One optimizer step: the mean loss of its samples, the learning rate it stepped at, the gradients' norm before
    clipping, and each sample's pair, prompt form and noise level."""

    step: int
    loss: float
    learning_rate: float
    grad_norm: float
    pairs: list[str]
    variants: list[str]
    sigmas: list[float]
    seconds: float


class ValidationLog(BaseModel):
    """A validation at ``step``: the mean loss of each prompt form over the validation cache."""

    step: int
    val_loss: dict[str, float]


def lora_config(lora: LoraSettings) -> LoraConfig:
    """``lora`` as PEFT's config: B initialised to zero, so the adapted DiT starts as the Base DiT."""
    return LoraConfig(r=lora.rank, lora_alpha=lora.alpha, target_modules=lora.target_modules, init_lora_weights=True)


def add_lora(transformer: MossTransformer, lora: LoraSettings) -> None:
    """A LoRA of shape ``lora`` added to ``transformer``, its weights in float32 whatever the DiT's dtype: diffusers
    adds them in the DiT's bfloat16, where AdamW's steps of 1e-4 on weights of 0.1 fall below the format's resolution
    and are lost."""
    transformer.add_adapter(lora_config(lora))
    for name, parameter in transformer.named_parameters():
        if ".lora_" in name:
            parameter.data = parameter.data.float()


def lora_transformer(base: Path, lora: LoraSettings, *, device: str, seed: int) -> MossTransformer:
    """The DiT at ``base`` (a diffusers-format folder) frozen in bfloat16 on ``device``, with a fresh LoRA of shape
    ``lora`` (float32, its A drawn after seeding torch with ``seed``), its joint blocks checkpointed."""
    transformer = MossTransformer.from_pretrained(str(base), torch_dtype=torch.bfloat16)
    transformer.requires_grad_(False)
    torch.manual_seed(seed)
    add_lora(transformer, lora)
    transformer.enable_gradient_checkpointing()
    return transformer.to(device).train()


def flow_matching_loss(transformer: MossTransformer, sample: CachedSample, sigma: float, noise: Tensor) -> Tensor:
    """The flow-matching loss of one sample at noise level ``sigma``: the drawn image's clean latents noised to it
    (``(1 - sigma) * image + sigma * noise``), the DiT's velocity for them from the prompt and the clean reference (if
    the sample has one), and its mean squared error from ``noise - image``, over the drawn image's tokens alone."""
    image = sample.target.float()
    noisy = (1 - sigma) * image + sigma * noise
    text = sample.prompt_embeds
    mask = torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)
    timestep = torch.tensor([sigma * 1000], device=noisy.device)
    references = None if sample.reference is None else [sample.reference]
    velocity = transformer(noisy, timestep, text, mask, references=references)
    return F.mse_loss(velocity.float(), noise - image)


def sample_order(count: int, generator: torch.Generator) -> Iterator[int]:
    """Entry indices for ever: each pass over the ``count`` entries in an order drawn from ``generator``."""
    while True:
        yield from torch.randperm(count, generator=generator).tolist()


def validation_losses(
    transformer: MossTransformer, cache: Path, entries: list[CacheEntry], *, device: str, seed: int
) -> dict[str, float]:
    """The mean loss of each prompt form over ``entries``, each pair at its own fixed noise level and noise (drawn
    from ``seed`` and the pair's place), the same for every form of the pair."""
    pairs = {pair: k for k, pair in enumerate(dict.fromkeys(entry.pair for entry in entries))}
    losses: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for entry in entries:
            sample = load_sample(cache, entry, device)
            generator = torch.Generator().manual_seed(seed + pairs[entry.pair])
            sigma = draw_sigmas(1, generator).item()
            noise = torch.randn(sample.target.shape, generator=generator).to(device)
            losses[entry.variant.value].append(flow_matching_loss(transformer, sample, sigma, noise).item())
    return {variant: fmean(values) for variant, values in losses.items()}


def save_checkpoint(transformer: MossTransformer, out: Path, step: int, lora: LoraSettings) -> Path:
    """The LoRA's weights and shape at ``step``, under ``out``'s checkpoints.

    Returns:
        The checkpoint's directory.
    """
    directory = out / CHECKPOINTS / f"step_{step:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    weights = {
        name: tensor.detach().cpu().contiguous() for name, tensor in get_peft_model_state_dict(transformer).items()
    }
    save_file(weights, str(directory / ADAPTER_WEIGHTS))
    AdapterRecord(lora=lora, step=step).write(directory / ADAPTER_RECORD)
    return directory


def _log(out: Path, record: BaseModel) -> None:
    with (out / LOG).open("a", encoding="utf-8") as log:
        log.write(f"{record.model_dump_json()}\n")


def accumulate_gradients(
    transformer: MossTransformer, cache: Path, samples: list[CacheEntry], generator: torch.Generator, device: str
) -> tuple[list[float], list[float]]:
    """One step's gradients accumulated into the LoRA: the mean of ``samples``' flow-matching losses, each at a noise
    level and noise drawn from ``generator``.

    Returns:
        Each sample's loss and noise level.
    """
    losses, sigmas = [], []
    for entry in samples:
        sample = load_sample(cache, entry, device)
        sigma = draw_sigmas(1, generator).item()
        noise = torch.randn(sample.target.shape, generator=generator).to(device)
        loss = flow_matching_loss(transformer, sample, sigma, noise)
        (loss / len(samples)).backward()
        losses.append(loss.item())
        sigmas.append(sigma)
    return losses, sigmas


def _validate(transformer: MossTransformer, val_cache: Path, out: Path, step: int, settings: TrainSettings) -> None:
    transformer.eval()
    entries = read_entries(val_cache)
    val_loss = validation_losses(transformer, val_cache, entries, device=settings.device, seed=VALIDATION_SEED)
    transformer.train()
    _log(out, ValidationLog(step=step, val_loss=val_loss))
    console.print(f"step {step}: val loss {json.dumps({form: round(loss, 4) for form, loss in val_loss.items()})}")


def train(base: Path, cache: Path, out: Path, settings: TrainSettings, *, val_cache: Path | None = None) -> Path:
    """A reference LoRA trained on the DiT at ``base`` from the cache at ``cache`` (the module's docstring).

    Returns:
        The last checkpoint's directory.

    Raises:
        ValueError: If a cache is not finished or holds no entry, or ``out`` holds a training's log already.
    """
    if (out / LOG).exists():
        raise ValueError(f"{out} holds a training already ({LOG}); give a new directory")
    entries = read_entries(cache)
    if not entries:  # the sample order would wait for ever for an entry
        raise ValueError(f"{cache} holds no entry to train on")
    out.mkdir(parents=True, exist_ok=True)
    transformer = lora_transformer(base, settings.lora, device=settings.device, seed=settings.seed)
    trainable = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=settings.learning_rate, betas=settings.betas, weight_decay=settings.weight_decay
    )
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda done: min(1.0, (done + 1) / settings.warmup))
    generator = torch.Generator().manual_seed(settings.seed)
    order = sample_order(len(entries), generator)
    checkpoint = out
    for step in range(1, settings.steps + 1):
        started = time.monotonic()
        samples = [entries[next(order)] for _ in range(settings.grad_accumulation)]
        losses, sigmas = accumulate_gradients(transformer, cache, samples, generator, settings.device)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, settings.grad_clip).item()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        seconds = time.monotonic() - started
        logged = StepLog(
            step=step,
            loss=fmean(losses),
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            pairs=[entry.pair for entry in samples],
            variants=[entry.variant.value for entry in samples],
            sigmas=sigmas,
            seconds=seconds,
        )
        _log(out, logged)
        if step % 10 == 0 or step == 1:
            console.print(f"step {step}: loss {fmean(losses):.4f}, grad norm {grad_norm:.3f}, {seconds:.1f} s")
        if step % settings.save_every == 0 or step == settings.steps:
            checkpoint = save_checkpoint(transformer, out, step, settings.lora)
        if val_cache is not None and (step % settings.val_every == 0 or step == settings.steps):
            _validate(transformer, val_cache, out, step, settings)
    return checkpoint


@app.command()
def main(
    transformer: Annotated[
        Path, typer.Option(help="the Base DiT's folder (diffusers format), the snapshot's transformer/")
    ],
    cache: Annotated[Path, typer.Option(help="a finished diffu-moss-cache-pairs cache")],
    out: Annotated[Path, typer.Option(help="the run's directory: its log and checkpoints")],
    val_cache: Annotated[
        Path | None, typer.Option(help="a validation cache (diffu-moss-cache-pairs --every-form)")
    ] = None,
    steps: Annotated[int, typer.Option(min=1, help="optimizer steps")] = 1500,
    save_every: Annotated[int, typer.Option(min=1, help="a checkpoint every this many steps, and at the end")] = 250,
    val_every: Annotated[
        int, typer.Option(min=1, help="score --val-cache every this many steps, and at the end")
    ] = 250,
    learning_rate: Annotated[float, typer.Option(help="AdamW's learning rate, constant after the warmup")] = 1e-4,
    seed: Annotated[int, typer.Option(help="the sample order, noise levels, noise and the LoRA's start")] = 0,
    device: Annotated[str, typer.Option(help="where the DiT trains")] = "cuda",
) -> None:
    """Train a LoRA that teaches MOSS-Image's DiT to draw from a reference image."""
    settings = TrainSettings(
        steps=steps, save_every=save_every, val_every=val_every, learning_rate=learning_rate, seed=seed, device=device
    )
    try:
        checkpoint = train(transformer, cache, out, settings, val_cache=val_cache)
    except ValueError as error:  # an unfinished cache or a used run directory: the arguments' fault
        raise typer.BadParameter(str(error)) from error
    console.print(f"the LoRA after {settings.steps} steps -> {checkpoint}")


if __name__ == "__main__":
    app()
