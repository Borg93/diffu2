"""diffu-moss-train-ref: teaches MOSS-Image's DiT to draw from a reference image, trained on a cache of style pairs
(``diffu-moss-cache-pairs``): a LoRA, or with ``--full`` every weight of the DiT (:mod:`diffu_moss.train.full`).

    diffu-moss-train-ref --transformer <MOSS-Image-Base snapshot>/transformer --cache data_flux3/cache_moss_ref_v3d \\
        --val-cache data_flux3/cache_moss_ref_v3d_val --out runs/moss_ref_pilot/lora --steps 1500
    diffu-moss-train-ref --full --transformer <MOSS-Image-Base snapshot>/transformer \\
        --cache data_flux3/cache_moss_ref_v3d --val-cache data_flux3/cache_moss_ref_v3d_val \\
        --out runs/moss_ref_full/full --steps 500

The LoRA follows the recipe of the 1715 hand LoRA (KohakuMossTrainer's, for parity): LoRA rank 32, alpha 32, on the
attention (q, k, v, o) and feed-forward (ffn_up_gate, ffn_down) of every joint block, the text refiner's blocks (qkv,
proj, ffn_gate, ffn_up, ffn_down), the prompt's projection (text_embedder.proj), the latents' embedder (video_embedder)
and the output (final_linear_video); the Base DiT frozen in bfloat16, the LoRA in float32; AdamW at 1e-4, betas (0.9,
0.999), weight decay 1e-4. The full fine-tune follows MOSS-Image's own recipe: Muon on the joint blocks' matrices and
AdamW on the rest at 5e-5, every weight in float32, the forward in bfloat16. Both clip the gradients at 1, keep the
learning rate constant after 50 warmup steps and take two samples a step, one a micro-batch. Each sample is the drawn
image's clean latents noised to a level drawn as MOSS-Image draws it (:mod:`diffu_moss.train.sigmas`; ``--sigma-shift``
moves the levels towards the clean end, where letter strokes are decided), and the loss is
flow matching's: the mean squared error between the DiT's velocity and noise minus image over the drawn image's
tokens; a reference enters clean and is not predicted (:func:`flow_matching_loss`). The joint blocks' activations are
checkpointed, as a reference adds half the drawn image's tokens again (a 1 MP target, a 0.5 MP reference).

Writes under ``--out``: ``train_log.jsonl`` (a line a step: loss, learning rate, gradient norm, its time and, on a CUDA
device, its peak memory, each sample's pair, form and noise level; a validation's losses by form on its own lines),
and every ``--save-every`` steps and at the end ``checkpoints/step_NNNNNN/``: the LoRA's weights
(``adapter.safetensors``) and shape (``adapter.json``), which ``diffu-moss-export-ref`` merges into the DiT, or the
fine-tuned DiT itself (``transformer/``, which ``diffu-moss-init`` draws with as it is). With ``--val-cache`` (a cache
of every form, ``--every-form``) the loss of each form is scored every ``--val-every`` steps at a fixed noise level and
noise per pair (the same in every run), the same for a pair's three forms, so the forms' losses compare: does the
reference lower it? A full fine-tune that saves its state (``--save-state``) resumes from a checkpoint (``--resume``)
as it would have gone on: the same samples, noise and steps, its log cut back to the checkpoint's step. With
``--limit-pairs N`` a run trains on the cache's first N pairs drawn with their reference alone: an overfit test, whose
model ``diffu_page.cli.draw_pairs`` draws the same pairs back. With ``--encode-text`` each sample's prompt is encoded in
its step by the frozen text encoder (:class:`diffu_moss.train.cache.PromptEncoder`, loaded as the cache loads it from
``--model``, by default the model directory the cache was made with) instead of read from the cache, which then needs
its latents alone (``diffu-moss-cache-pairs --latents-only``): the same embeddings, at the cost of the encoder's
16.8 GB in bfloat16 beside the DiT and a forward of it per sample.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from statistics import fmean
from typing import Annotated, Self

import torch
import typer
from peft import LoraConfig, get_peft_model_state_dict
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR

from diffu_moss.train.cache import (
    CachedSample,
    CacheEntry,
    PromptEncoder,
    load_sample,
    read_entries,
    read_record,
    target_shape,
    with_reference,
)
from diffu_moss.train.distributed import Ranks, clip_sharded_gradients, joined, left
from diffu_moss.train.full import STATE, FullSettings, FullTraining
from diffu_moss.train.records import (
    ADAPTER_RECORD,
    ADAPTER_WEIGHTS,
    PROGRESS_RECORD,
    AdapterRecord,
    LoraSettings,
    Progress,
)
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
# what a resumed run may change: how long it runs, how often it saves and scores, where, whether it saves its state, and
# whether it encodes its prompts (the same embeddings as cached)
RESUMABLE_CHANGES = {
    "steps": True,
    "save_every": True,
    "val_every": True,
    "device": True,
    "encode_text": True,
    "full": {"save_state": True},
}


class TrainSettings(BaseModel):
    """How a reference DiT is trained (the module's docstring): the LoRA's shape and optimizer, or with ``full`` a full
    fine-tune's; the steps and samples (with ``limit_pairs``, only the cache's first that many pairs drawn with their
    reference: an overfit test); whether each prompt is encoded in the step (``encode_text``) or read from the cache;
    the shift of the drawn noise levels (``sigma_shift``, :mod:`diffu_moss.train.sigmas`; 1: MOSS-Image's own); and
    when it is saved and scored."""

    model_config = ConfigDict(frozen=True)

    steps: int = Field(default=1500, ge=1)
    rank: int = Field(default=32, ge=1)
    alpha: int = Field(default=32, ge=1)
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    full: FullSettings | None = None
    limit_pairs: int | None = Field(default=None, ge=1)
    encode_text: bool = False
    sigma_shift: float = Field(default=1.0, gt=0)
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
    clipping, each sample's pair, prompt form and noise level, its time and, on a CUDA device, the most memory its
    tensors held at once (GB)."""

    step: int
    loss: float
    learning_rate: float
    grad_norm: float
    pairs: list[str]
    variants: list[str]
    sigmas: list[float]
    seconds: float
    peak_memory_gb: float | None = None


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


def warmup(steps: int) -> Callable[[int], float]:
    """The learning rate's factor after a number of steps done: rising linearly to 1 over ``steps`` steps."""
    return lambda done: min(1.0, (done + 1) / steps)


def flow_matching_loss(
    transformer: MossTransformer,
    sample: CachedSample,
    sigma: float,
    noise: Tensor,
    *,
    compute_dtype: torch.dtype | None = None,
) -> Tensor:
    """The flow-matching loss of one sample at noise level ``sigma``: the drawn image's clean latents noised to it
    (``(1 - sigma) * image + sigma * noise``), the DiT's velocity for them from the prompt and the clean reference (if
    the sample has one), and its mean squared error from ``noise - image``, over the drawn image's tokens alone. With
    ``compute_dtype`` the DiT runs in that dtype's autocast (its weight cache off), else in its weights' own dtypes."""
    image = sample.target.float()
    noisy = (1 - sigma) * image + sigma * noise
    text = sample.prompt_embeds
    mask = torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)
    timestep = torch.tensor([sigma * 1000], device=noisy.device)
    references = None if sample.reference is None else [sample.reference]
    precision = (
        nullcontext()
        if compute_dtype is None
        else torch.autocast(noisy.device.type, dtype=compute_dtype, cache_enabled=False)
    )
    with precision:
        velocity = transformer(noisy, timestep, text, mask, references=references)
    return F.mse_loss(velocity.float(), noise - image)


class SampleOrder:
    """Entry indices for ever: each pass over ``count`` entries in an order drawn from ``generator`` when the pass
    begins; resumed, from ``order`` (the current pass's) with ``drawn`` of it drawn.

    Attributes:
        order: The current pass's order.
        drawn: How much of it was drawn.
    """

    def __init__(
        self, count: int, generator: torch.Generator, *, order: list[int] | None = None, drawn: int = 0
    ) -> None:
        self._count = count
        self._generator = generator
        self.order = [] if order is None else order
        self.drawn = drawn

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> int:
        if self.drawn == len(self.order):
            self.order = torch.randperm(self._count, generator=self._generator).tolist()
            self.drawn = 0
        self.drawn += 1
        return self.order[self.drawn - 1]


def validation_losses(
    transformer: MossTransformer,
    cache: Path,
    entries: list[CacheEntry],
    *,
    device: str,
    seed: int,
    compute_dtype: torch.dtype | None = None,
    encoder: PromptEncoder | None = None,
    ranks: Ranks | None = None,
) -> dict[str, float]:
    """The mean loss of each prompt form over ``entries``, each pair at its own fixed noise level and noise (drawn
    from ``seed`` and the pair's place), the same for every form of the pair; the prompts encoded by ``encoder`` when
    given, else cached. Over several ranks each scores the entries it owns in turn (a rank short of one in the last
    round scores the first entry again and drops it: a sharded DiT's forward needs every rank), and the losses are
    summed over the ranks before the means, taken in entry order as one process takes them."""
    ranks = Ranks() if ranks is None else ranks
    pairs = {pair: k for k, pair in enumerate(dict.fromkeys(entry.pair for entry in entries))}
    rounds = -(-len(entries) // ranks.world)
    scored = [0.0] * len(entries)
    with torch.no_grad():
        for index in range(ranks.rank, rounds * ranks.world, ranks.world):
            entry = entries[index if index < len(entries) else 0]
            sample = load_sample(cache, entry, device, encoder)
            generator = torch.Generator().manual_seed(seed + pairs[entry.pair])
            sigma = draw_sigmas(1, generator).item()
            noise = torch.randn(sample.target.shape, generator=generator).to(device)
            loss = flow_matching_loss(transformer, sample, sigma, noise, compute_dtype=compute_dtype)
            if index < len(entries):
                scored[index] = loss.item()
    losses: dict[str, list[float]] = defaultdict(list)
    for entry, loss in zip(entries, ranks.summed(scored, device), strict=True):
        losses[entry.variant.value].append(loss)
    return {variant: fmean(values) for variant, values in losses.items()}


class LoraTraining:
    """A reference LoRA in training (the module's docstring): the DiT at ``base`` with a fresh LoRA, AdamW over the
    LoRA's weights, and the warmup.

    Attributes:
        transformer: The DiT, frozen in bfloat16, with its LoRA in float32.
        parameters: The LoRA's weights.
    """

    compute_dtype: torch.dtype | None = None  # the frozen DiT in its bfloat16 and the LoRA in its float32

    def __init__(self, base: Path, settings: TrainSettings) -> None:
        self.transformer = lora_transformer(base, settings.lora, device=settings.device, seed=settings.seed)
        self.parameters: list[nn.Parameter] = [
            parameter for parameter in self.transformer.parameters() if parameter.requires_grad
        ]
        self._optimizer = torch.optim.AdamW(
            self.parameters, lr=settings.learning_rate, betas=settings.betas, weight_decay=settings.weight_decay
        )
        self._schedule = LambdaLR(self._optimizer, warmup(settings.warmup))
        self._lora = settings.lora

    def step(self) -> float:
        """One AdamW step of the LoRA from its gradients (then dropped).

        Returns:
            The learning rate it stepped at.
        """
        learning_rate = float(self._optimizer.param_groups[0]["lr"])
        self._optimizer.step()
        self._schedule.step()
        self._optimizer.zero_grad(set_to_none=True)
        return learning_rate

    def save(self, directory: Path, progress: Progress) -> None:
        """The LoRA's weights and shape at ``progress``'s step, into ``directory``."""
        weights = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in get_peft_model_state_dict(self.transformer).items()
        }
        save_file(weights, str(directory / ADAPTER_WEIGHTS))
        AdapterRecord(lora=self._lora, step=progress.step).write(directory / ADAPTER_RECORD)


def _log(out: Path, record: BaseModel) -> None:
    with (out / LOG).open("a", encoding="utf-8") as log:
        log.write(f"{record.model_dump_json()}\n")


def accumulate_gradients(
    transformer: MossTransformer,
    cache: Path,
    samples: list[CacheEntry],
    generator: torch.Generator,
    device: str,
    *,
    compute_dtype: torch.dtype | None = None,
    encoder: PromptEncoder | None = None,
    sigma_shift: float = 1.0,
    ranks: Ranks | None = None,
) -> tuple[list[float], list[float]]:
    """One step's gradients accumulated into the trained weights: the mean of ``samples``' flow-matching losses, each
    at a noise level (shifted by ``sigma_shift``) and noise drawn from ``generator``, its prompt encoded by ``encoder``
    when given. Over several ranks every rank draws every sample's level and noise in turn, as one process does, and
    computes the samples it owns (:meth:`Ranks.owns`), each loss divided by its own count: FSDP2 averages the ranks'
    gradients into the mean over ``samples``.

    Returns:
        Each sample's loss (summed over the ranks) and noise level.
    """
    ranks = Ranks() if ranks is None else ranks
    owned = len(samples) // ranks.world
    losses, sigmas = [0.0] * len(samples), []
    for index, entry in enumerate(samples):
        if not ranks.owns(index):  # its level and noise drawn all the same, so every rank's generator goes on alike
            sigmas.append(draw_sigmas(1, generator, sigma_shift).item())
            torch.randn(target_shape(cache, entry), generator=generator)
            continue
        sample = load_sample(cache, entry, device, encoder)
        sigma = draw_sigmas(1, generator, sigma_shift).item()
        noise = torch.randn(sample.target.shape, generator=generator).to(device)
        loss = flow_matching_loss(transformer, sample, sigma, noise, compute_dtype=compute_dtype)
        (loss / owned).backward()
        losses[index] = loss.item()
        sigmas.append(sigma)
    return ranks.summed(losses, device), sigmas


def _validate(
    training: LoraTraining | FullTraining,
    val_cache: Path,
    out: Path,
    step: int,
    settings: TrainSettings,
    encoder: PromptEncoder | None,
    ranks: Ranks,
) -> None:
    training.transformer.eval()
    entries = read_entries(val_cache)
    val_loss = validation_losses(
        training.transformer,
        val_cache,
        entries,
        device=settings.device,
        seed=VALIDATION_SEED,
        compute_dtype=training.compute_dtype,
        encoder=encoder,
        ranks=ranks,
    )
    training.transformer.train()
    if not ranks.main:
        return
    _log(out, ValidationLog(step=step, val_loss=val_loss))
    console.print(f"step {step}: val loss {json.dumps({form: round(loss, 4) for form, loss in val_loss.items()})}")


def resumed_progress(out: Path, checkpoint: Path, settings: TrainSettings) -> Progress:
    """The progress saved in ``checkpoint``'s state, checked to be a checkpoint of the run in ``out`` that trained as
    ``settings`` do (but for :data:`RESUMABLE_CHANGES`).

    Raises:
        ValueError: If ``settings`` are not a full fine-tune's, ``checkpoint`` is not one of ``out``'s or holds no
            state, or it was trained otherwise.
    """
    if settings.full is None:
        raise ValueError("only a full fine-tune (--full) saves the state a run resumes from")
    if checkpoint.resolve().parent != (out / CHECKPOINTS).resolve():
        raise ValueError(f"{checkpoint} is not a checkpoint of {out}: a run resumes in its own directory")
    record = checkpoint / STATE / PROGRESS_RECORD
    if not record.is_file():
        raise ValueError(f"{checkpoint} holds no state to resume from (a run saves it with --save-state)")
    progress = Progress.read(record)
    trained = TrainSettings.model_validate_json(progress.settings)
    if trained.model_dump(exclude=RESUMABLE_CHANGES) != settings.model_dump(exclude=RESUMABLE_CHANGES):
        raise ValueError(f"{checkpoint} was trained with other settings: {progress.settings}")
    return progress


def _cut_log(out: Path, step: int) -> None:
    """``out``'s log cut back to the lines of its steps up to ``step``, for a run resumed at it to go on from."""
    lines = (out / LOG).read_text(encoding="utf-8").splitlines(keepends=True)
    (out / LOG).write_text("".join(line for line in lines if json.loads(line)["step"] <= step), encoding="utf-8")


def _progress(step: int, settings: TrainSettings, generator: torch.Generator, order: SampleOrder) -> Progress:
    generator_state = bytes(generator.get_state().numpy())
    return Progress(
        step=step, settings=settings.model_dump_json(), generator=generator_state, order=order.order, drawn=order.drawn
    )


def _peak_memory_gb(device: torch.device) -> float | None:
    return torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None


def prompt_encoder(
    cache: Path, val_cache: Path | None, settings: TrainSettings, model: Path | None
) -> PromptEncoder | None:
    """The text encoder a run encodes its prompts by: with ``settings.encode_text``, the MOSS model directory
    ``model``'s, or by default that of the model directory ``cache`` was made with; else None.

    Raises:
        ValueError: If a cache is not finished, or holds no prompt embeddings (a latents-only cache) and the run does
            not encode them.
    """
    record = read_record(cache)
    if settings.encode_text:
        return PromptEncoder(Path(record.model) if model is None else model, settings.device)
    for given in (cache, val_cache):
        if given is not None and read_record(given).latents_only:
            raise ValueError(f"{given} holds no prompt embeddings (a latents-only cache): train with --encode-text")
    return None


def train(
    base: Path,
    cache: Path,
    out: Path,
    settings: TrainSettings,
    *,
    val_cache: Path | None = None,
    resume: Path | None = None,
    model: Path | None = None,
    ranks: Ranks | None = None,
) -> Path:
    """The DiT at ``base`` trained from the cache at ``cache`` (the module's docstring): a LoRA, or with
    ``settings.full`` every weight, from the start or from ``out``'s checkpoint ``resume``; with
    ``settings.encode_text`` its prompts encoded by the text encoder of :func:`prompt_encoder` (``model``'s). Over
    several ``ranks`` (a ``torchrun`` launch, :mod:`diffu_moss.train.distributed`) a full fine-tune is sharded by
    FSDP2, each step's samples shared out among the ranks, and the main rank writes the log and checkpoints.

    Returns:
        The last checkpoint's directory.

    Raises:
        ValueError: If a cache is not finished, holds no entry to train on or no prompt embeddings to read, ``out``
            holds a training's log already (unless resumed), or the run cannot resume from ``resume``
            (:func:`resumed_progress`; its cache's entries are others); over several ranks, if it trains a LoRA or its
            samples a step do not share out evenly among them.
    """
    ranks = Ranks() if ranks is None else ranks
    if ranks.mesh is not None and settings.full is None:
        raise ValueError("a run over several processes shards a full fine-tune: give --full")
    if settings.grad_accumulation % ranks.world:
        raise ValueError(f"{settings.grad_accumulation} samples a step do not share out among {ranks.world} ranks")
    if resume is None and (out / LOG).exists():
        raise ValueError(f"{out} holds a training already ({LOG}); give a new directory")
    entries = read_entries(cache)
    if settings.limit_pairs is not None:
        entries = with_reference(entries, settings.limit_pairs)
    if not entries:  # the sample order would wait for ever for an entry
        raise ValueError(f"{cache} holds no entry to train on")
    progress = None if resume is None else resumed_progress(out, resume, settings)
    if progress is not None and len(progress.order) != len(entries):
        raise ValueError(f"{cache} holds {len(entries)} entries; the run resumed drew from {len(progress.order)}")
    encoder = prompt_encoder(cache, val_cache, settings, model)
    out.mkdir(parents=True, exist_ok=True)
    training: LoraTraining | FullTraining
    if settings.full is None:
        training = LoraTraining(base, settings)
    else:
        schedule = warmup(settings.warmup)
        training = FullTraining(
            base, settings.full, schedule=schedule, device=settings.device, resume=resume, ranks=ranks
        )
    generator = torch.Generator().manual_seed(settings.seed)
    order = SampleOrder(len(entries), generator)
    done = 0
    if progress is not None:
        generator.set_state(torch.frombuffer(bytearray(progress.generator), dtype=torch.uint8))
        order = SampleOrder(len(entries), generator, order=progress.order, drawn=progress.drawn)
        done = progress.step
        if ranks.main:
            _cut_log(out, done)
    device = torch.device(settings.device)
    checkpoint = out
    for step in range(done + 1, settings.steps + 1):
        started = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        samples = [entries[next(order)] for _ in range(settings.grad_accumulation)]
        losses, sigmas = accumulate_gradients(
            training.transformer,
            cache,
            samples,
            generator,
            settings.device,
            compute_dtype=training.compute_dtype,
            encoder=encoder,
            sigma_shift=settings.sigma_shift,
            ranks=ranks,
        )
        if ranks.mesh is None:
            grad_norm = torch.nn.utils.clip_grad_norm_(training.parameters, settings.grad_clip).item()
        else:
            grad_norm = clip_sharded_gradients(training.parameters, settings.grad_clip)
        learning_rate = training.step()
        seconds = time.monotonic() - started
        peak = _peak_memory_gb(device)
        logged = StepLog(
            step=step,
            loss=fmean(losses),
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            pairs=[entry.pair for entry in samples],
            variants=[entry.variant.value for entry in samples],
            sigmas=sigmas,
            seconds=seconds,
            peak_memory_gb=peak,
        )
        if ranks.main:
            _log(out, logged)
        if ranks.main and (step % 10 == 0 or step == done + 1):
            memory = "" if peak is None else f", {peak:.1f} GB"
            console.print(f"step {step}: loss {fmean(losses):.4f}, grad norm {grad_norm:.3f}, {seconds:.1f} s{memory}")
        if step % settings.save_every == 0 or step == settings.steps:
            checkpoint = out / CHECKPOINTS / f"step_{step:06d}"
            if ranks.main:
                checkpoint.mkdir(parents=True, exist_ok=True)
            training.save(checkpoint, _progress(step, settings, generator, order))
        if val_cache is not None and (step % settings.val_every == 0 or step == settings.steps):
            _validate(training, val_cache, out, step, settings, encoder, ranks)
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
    full: Annotated[
        bool,
        typer.Option("--full", help="train every weight of the DiT (MOSS-Image's recipe, Muon and AdamW), not a LoRA"),
    ] = False,
    steps: Annotated[int, typer.Option(min=1, help="optimizer steps")] = 1500,
    samples_per_step: Annotated[
        int,
        typer.Option(
            min=1,
            help="samples a step, one a micro-batch; under torchrun shared out evenly among the ranks (a multiple of "
            "their number)",
        ),
    ] = 2,
    limit_pairs: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="train only on the cache's first this many pairs drawn with their reference: an overfit test "
            "(diffu_page.cli.draw_pairs draws them back)",
        ),
    ] = None,
    save_every: Annotated[int, typer.Option(min=1, help="a checkpoint every this many steps, and at the end")] = 250,
    val_every: Annotated[
        int, typer.Option(min=1, help="score --val-cache every this many steps, and at the end")
    ] = 250,
    learning_rate: Annotated[
        float | None,
        typer.Option(help="the learning rate, constant after the warmup (default: 1e-4 for a LoRA, 5e-5 with --full)"),
    ] = None,
    ema: Annotated[
        float | None,
        typer.Option(
            help="with --full: keep an average of the weights at this decay (the report's 0.9999; 20 GB more in "
            "float32), saved beside each checkpoint (default: none)"
        ),
    ] = None,
    save_state: Annotated[
        bool,
        typer.Option(
            "--save-state",
            help="with --full: save the float32 weights and the optimizer's state in each checkpoint (about 42 GB), "
            "to resume from",
        ),
    ] = False,
    resume: Annotated[
        Path | None, typer.Option(help="with --full: go on from a checkpoint of --out saved with --save-state")
    ] = None,
    encode_text: Annotated[
        bool,
        typer.Option(
            "--encode-text",
            help="encode each prompt in its step by the frozen text encoder (16.8 GB more) instead of reading it from "
            "the cache, which then needs its latents alone (diffu-moss-cache-pairs --latents-only)",
        ),
    ] = False,
    model: Annotated[
        Path | None,
        typer.Option(
            help="with --encode-text: the MOSS model directory whose text encoder encodes the prompts (default: the "
            "one the cache was made with)"
        ),
    ] = None,
    sigma_shift: Annotated[
        float,
        typer.Option(
            help="shift the drawn noise levels: sigma -> k sigma / (1 + (k - 1) sigma); under 1 towards the clean end, "
            "where letter strokes are decided (1/3 moves the median 0.78 to 0.54); 1 draws MOSS-Image's own (the "
            "validation always does)",
        ),
    ] = 1.0,
    seed: Annotated[int, typer.Option(help="the sample order, noise levels, noise and the LoRA's start")] = 0,
    device: Annotated[str, typer.Option(help="where the DiT trains")] = "cuda",
) -> None:
    """Train MOSS-Image's DiT to draw from a reference image: a LoRA, or every weight (--full); under torchrun, every
    weight sharded over the ranks by FSDP2."""
    if not full and (ema is not None or save_state or resume is not None):
        raise typer.BadParameter("--ema, --save-state and --resume are a full fine-tune's: give --full")
    if model is not None and not encode_text:
        raise typer.BadParameter("--model gives the text encoder of --encode-text: give --encode-text")
    rate = {} if learning_rate is None else {"learning_rate": learning_rate}  # else each kind's own default
    ranks, device = joined(device)
    try:  # a ValidationError (an --ema outside (0, 1)) is a ValueError too
        fine_tune = FullSettings.model_validate({"ema": ema, "save_state": save_state, **rate}) if full else None
        run = {
            "steps": steps,
            "limit_pairs": limit_pairs,
            "encode_text": encode_text,
            "sigma_shift": sigma_shift,
            "grad_accumulation": samples_per_step,
            "save_every": save_every,
            "val_every": val_every,
        }
        settings = TrainSettings.model_validate(
            {**run, "seed": seed, "device": device, "full": fine_tune, **({} if full else rate)}
        )
        checkpoint = train(
            transformer, cache, out, settings, val_cache=val_cache, resume=resume, model=model, ranks=ranks
        )
    except ValueError as error:  # an unfinished cache, a used run directory, a checkpoint not to resume: the arguments'
        raise typer.BadParameter(str(error)) from error
    finally:
        left()
    if not ranks.main:
        return
    trained = "the LoRA" if settings.full is None else "the DiT"
    console.print(f"{trained} after {settings.steps} steps -> {checkpoint}")


if __name__ == "__main__":
    app()
