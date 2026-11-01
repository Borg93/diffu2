"""python -m diffu_moss.bench_inference: MOSS-Image's drawing speed, memory and fidelity, one configuration at a time:
bfloat16, MXFP8 in the DiT's joint blocks, and MXFP8 in the DiT and the text encoder, each with its joint blocks run
eagerly, compiled at load (``torch.compile``) or from an AOTInductor package (``diffu-moss-aoti``).

    python -m diffu_moss.bench_inference --model runs/moss/base --package-bf16 runs/aoti/moss_bf16.pt2 \\
        --package-mxfp8 runs/aoti/moss_mxfp8.pt2 --out runs/bench/inference.jsonl --images runs/bench/pages

Each configuration loads the model afresh and prepares it as the page pipeline's ``load_pipeline`` does
(:func:`diffu_moss.inference.prepare_inference`), then draws one page twice from the same prompt and noise: the first
drawing pays for any compiling, the second is timed. One JSON line per configuration (printed, and appended to
``--out``):

- ``load_seconds``: loading the components onto the device and preparing them (quantizing, loading a package;
  compiling waits for the first call); ``first_step_seconds``: the first drawing, from its start (encoding its prompts)
  to the end of its first denoising step (compiling included); ``ready_seconds``: their sum;
- ``step_seconds``: the second drawing's denoising steps (both passes of guidance and the scheduler's step), their mean
  (and each in ``step_seconds_each``), timed by CUDA events; ``text_seconds``: one prompt encoded, warm, the mean of 3;
- ``load_peak_gb`` and ``run_peak_gb``: ``torch.cuda.max_memory_allocated`` over the load, then over the drawings;
- ``velocity_cosine`` and ``latents_cosine``: the cosine similarity of the DiT's first velocity (the first step's pass
  with the prompt) and of the final latents to the first configuration's, at the same noise.

The first configuration measured is the reference (bf16_bf16_eager by default). ``--images`` saves each
configuration's page, to look at beside the reference's. A configuration that fails gets a line of its own
(``config``, ``error``) and the next one runs, unless the device ran out of memory or failed: the run stops there.
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Self

import torch
import typer
from PIL import Image
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from torch import Tensor
from torch.nn import functional as F

from diffu.modular_pipelines.loading import COMPONENT_DTYPES, require_loaded
from diffu_moss.aoti import device_arch
from diffu_moss.inference import MossInference, prepare_inference
from diffu_moss.layout_prompt import LAYOUT_META
from diffu_moss.modular_pipelines.moss import MossAutoBlocks, MossModularPipeline
from diffu_moss.modular_pipelines.moss.encoders import prompt_embeds, prompt_tokens
from diffu_moss.mxfp8 import Precision

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
console = Console(stderr=True)

TEXT_REPEATS = 3
PAGE_LINES = 32  # the synthetic page's handwritten lines: a full page's prompt
PRECISIONS = ((Precision.BF16, Precision.BF16), (Precision.MXFP8, Precision.BF16), (Precision.MXFP8, Precision.MXFP8))
# Errors after which the device cannot be trusted: the run stops there (an out-of-memory once froze the DGX Spark)
DEVICE_ERRORS = (torch.OutOfMemoryError, torch.AcceleratorError)


class Blocks(StrEnum):
    """How the DiT's joint blocks run."""

    EAGER = "eager"
    COMPILED = "compiled"
    PACKAGED = "packaged"


class BenchConfig(BaseModel):
    """One configuration: the precision of the DiT's joint blocks and of the text encoder, and how the blocks run."""

    model_config = ConfigDict(frozen=True)

    dit: Precision
    text_encoder: Precision
    blocks: Blocks

    @property
    def name(self) -> str:
        """``<dit>_<text encoder>_<blocks>``, as ``--config`` names it."""
        return f"{self.dit}_{self.text_encoder}_{self.blocks}"

    def inference(self, packages: dict[Precision, Path]) -> MossInference:
        """The options that draw in this configuration, a packaged one from ``packages`` (by the DiT's precision).

        Raises:
            ValueError: If it runs from a package and none is given for its precision.
        """
        package = None
        if self.blocks is Blocks.PACKAGED:
            if self.dit not in packages:
                raise ValueError(f"{self.name} needs a package of the {self.dit} joint block (--package-{self.dit})")
            package = packages[self.dit]
        return MossInference(
            precision=self.dit,
            text_encoder_precision=self.text_encoder,
            compile_blocks=self.blocks is Blocks.COMPILED,
            aoti_package=package,
        )


CONFIGS = {
    config.name: config
    for config in (
        BenchConfig(dit=dit, text_encoder=text_encoder, blocks=blocks)
        for dit, text_encoder in PRECISIONS
        for blocks in Blocks
    )
}


class BenchSettings(BaseModel):
    """What every configuration draws: the model on its device, the page's size (width, height), prompt, steps and
    noise seed."""

    model_config = ConfigDict(frozen=True)

    model: Path
    device: str
    size: tuple[int, int]
    prompt: str
    steps: int
    seed: int


class BenchRecord(BaseModel):
    """One configuration's measures (the module's docstring); the cosines are set once the reference is drawn."""

    config: str
    dit: Precision
    text_encoder: Precision
    blocks: Blocks
    package: str | None
    device: str
    arch: str
    torch: str
    size: tuple[int, int]
    steps: int
    prompt_tokens: int
    load_seconds: float
    first_step_seconds: float
    ready_seconds: float
    step_seconds: float
    step_seconds_each: list[float]
    text_seconds: float
    load_peak_gb: float | None
    run_peak_gb: float | None
    velocity_cosine: float | None = None
    latents_cosine: float | None = None


class BenchFailure(BaseModel):
    """A configuration that could not be measured: its name and the error that stopped it."""

    config: str
    error: str

    @classmethod
    def of(cls, config: BenchConfig, error: Exception) -> Self:
        """``config`` stopped by ``error``."""
        return cls(config=config.name, error=f"{type(error).__name__}: {error}")


class Drawing(BaseModel):
    """A drawing's first velocity (on the CPU), final latents (on the CPU) and page, the seconds of each of its steps,
    and the seconds from its start to the end of its first step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    velocity: Tensor
    latents: Tensor
    page: Image.Image
    steps: list[float]
    to_first_step: float


class Clock:
    """Marks on a device's timeline: CUDA events on a GPU (read once synchronized), the wall clock elsewhere."""

    def __init__(self, device: torch.device) -> None:
        self.cuda = device.type == "cuda"
        self.marks: list[Any] = []

    def mark(self) -> None:
        """A mark now, in the device's stream order."""
        if self.cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.marks.append(event)
        else:
            self.marks.append(time.perf_counter())

    def intervals(self) -> list[float]:
        """The seconds between consecutive marks."""
        pairs = zip(self.marks, self.marks[1:], strict=False)
        if not self.cuda:
            return [later - earlier for earlier, later in pairs]
        torch.cuda.synchronize()
        return [earlier.elapsed_time(later) / 1000 for earlier, later in pairs]


def synchronize(device: torch.device) -> None:
    """Wait for ``device``'s queued work (a GPU's; nothing elsewhere)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_gb(device: torch.device) -> float | None:
    """The most memory allocated on ``device`` since its peak was reset, in GB (GPUs only)."""
    return torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None


def reset_device(device: torch.device) -> None:
    """Free what the last configuration left (its compiled code too) and start the peak afresh."""
    torch.compiler.reset()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def synthetic_prompt(width: int, height: int, lines: int = PAGE_LINES) -> str:
    """A MOSS Layout prompt of a handwritten page of ``lines`` lines on a ``width`` x ``height`` canvas, laid out as the
    page pipeline compiles one (:mod:`diffu_moss.layout_prompt`): a summary, then a text item per line in its box."""
    pitch = height // (lines + 2)
    items = [
        {
            "label": "text",
            "caption": "a handwritten line in dark brown iron-gall ink, slanted to the right",
            "text": f"Anno {1700 + k} den {k % 28 + 1} Martii inställte sig för Rätten Hans Persson ifrån Byn N:o {k}",
            "bbox": [(k + 1) * pitch - height // 2, 40 - width // 2, (k + 2) * pitch - height // 2, width // 2 - 40],
        }
        for k in range(lines)
    ]
    summary = {
        "meta": LAYOUT_META,
        "canvas": "single",
        "caption": "A flat, evenly lit archival scan of a handwritten page of court minutes on aged cream paper.",
    }
    prompt = {"summary": summary, "items": items, "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    return json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))


def load(model: Path, options: MossInference, device: torch.device) -> MossModularPipeline:
    """The MOSS-Image repo ``model`` loaded as the page pipeline loads it (its own blocks, the VAE in float32, the rest
    in bfloat16) on ``device``, prepared as ``options`` ask."""
    pipe = MossModularPipeline(blocks=MossAutoBlocks(), pretrained_model_name_or_path=str(model))
    pipe.load_components(dtype=COMPONENT_DTYPES)
    require_loaded(pipe)
    pipe.to(device)
    prepare_inference(pipe, options)
    return pipe


def draw(pipe: MossModularPipeline, settings: BenchSettings) -> Drawing:
    """The page of ``settings`` drawn by ``pipe``, timed: a mark when the DiT is first called and after each scheduler
    step, and the wall clock from the start to the first step's end (synchronized there)."""
    device = pipe._execution_device
    clock = Clock(device)
    velocities: list[Tensor] = []
    first_step_end: list[float] = []

    def before_dit(module: torch.nn.Module, args: tuple[object, ...]) -> None:
        if not clock.marks:
            clock.mark()

    def after_dit(module: torch.nn.Module, args: tuple[object, ...], output: Tensor) -> None:
        if not velocities:
            velocities.append(output.detach().float().cpu())

    scheduler_step: Callable[..., Any] = pipe.scheduler.step

    def timed_step(*args: Any, **kwargs: Any) -> Any:
        stepped = scheduler_step(*args, **kwargs)
        clock.mark()
        if not first_step_end:
            synchronize(device)
            first_step_end.append(time.perf_counter())
        return stepped

    hooks = [
        pipe.transformer.register_forward_pre_hook(before_dit),
        pipe.transformer.register_forward_hook(after_dit),
    ]
    pipe.scheduler.step = timed_step
    pipe.guider.set_state(step=0, num_inference_steps=settings.steps, timestep=None)
    start = time.perf_counter()
    try:
        with torch.no_grad():
            drawn = pipe(
                prompt=settings.prompt,
                width=settings.size[0],
                height=settings.size[1],
                num_inference_steps=settings.steps,
                generator=torch.Generator().manual_seed(settings.seed),
                output=["latents", "images"],
            )
    finally:
        for hook in hooks:
            hook.remove()
        pipe.scheduler.step = scheduler_step
    return Drawing(
        velocity=velocities[0],
        latents=drawn["latents"].detach().float().cpu(),
        page=drawn["images"][0],
        steps=clock.intervals(),
        to_first_step=first_step_end[0] - start,
    )


def text_seconds(pipe: MossModularPipeline, prompt: str) -> float:
    """The mean seconds of encoding ``prompt``, warm, over :data:`TEXT_REPEATS` encodings."""
    device = pipe._execution_device
    clock = Clock(device)
    with torch.no_grad():
        prompt_embeds(pipe.text_encoder, pipe.tokenizer, prompt, device=device)
        clock.mark()
        for _ in range(TEXT_REPEATS):
            prompt_embeds(pipe.text_encoder, pipe.tokenizer, prompt, device=device)
            clock.mark()
    times = clock.intervals()
    return sum(times) / len(times)


def cosine(a: Tensor, b: Tensor) -> float:
    """The cosine similarity of two tensors as vectors, in float64."""
    return F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


def measure(config: BenchConfig, options: MossInference, settings: BenchSettings) -> tuple[BenchRecord, Drawing]:
    """One configuration loaded and measured (the module's docstring), with its timed drawing."""
    device = torch.device(settings.device)
    reset_device(device)
    start = time.perf_counter()
    pipe = load(settings.model, options, device)
    synchronize(device)
    load_seconds = time.perf_counter() - start
    load_peak = peak_gb(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    first = draw(pipe, settings)
    timed = draw(pipe, settings)
    record = BenchRecord(
        config=config.name,
        dit=config.dit,
        text_encoder=config.text_encoder,
        blocks=config.blocks,
        package=None if options.aoti_package is None else str(options.aoti_package),
        device=torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
        arch=device_arch(device),
        torch=torch.__version__,
        size=settings.size,
        steps=len(timed.steps),
        prompt_tokens=prompt_tokens(pipe.tokenizer, settings.prompt),
        load_seconds=load_seconds,
        first_step_seconds=first.to_first_step,
        ready_seconds=load_seconds + first.to_first_step,
        step_seconds=sum(timed.steps) / len(timed.steps),
        step_seconds_each=timed.steps,
        text_seconds=text_seconds(pipe, settings.prompt),
        load_peak_gb=load_peak,
        run_peak_gb=peak_gb(device),
    )
    return record, timed


def chosen_configs(names: list[str] | None) -> list[BenchConfig]:
    """The configurations ``--config`` names, in its order (every one when it names none).

    Raises:
        typer.BadParameter: If it names one there is not.
    """
    if not names:
        return list(CONFIGS.values())
    unknown = [name for name in names if name not in CONFIGS]
    if unknown:
        raise typer.BadParameter(f"no configuration {unknown}; there are {list(CONFIGS)}", param_hint="--config")
    return [CONFIGS[name] for name in names]


def emit(record: BaseModel, out: Path | None) -> None:
    """``record`` as a JSON line on stdout, and appended to ``out`` if given."""
    line = record.model_dump_json()
    print(line, flush=True)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as lines:
            lines.write(line + "\n")


@app.command()
def main(
    model: Annotated[Path, typer.Option(help="a MOSS-Image model directory (diffu-moss-init's)")],
    out: Annotated[
        Path | None, typer.Option(help="a JSON-lines file each configuration's record is appended to")
    ] = None,
    images: Annotated[Path | None, typer.Option(help="a directory for each configuration's page (PNG)")] = None,
    config: Annotated[
        list[str] | None,
        typer.Option(
            help="a configuration to measure, <dit>_<text encoder>_<blocks>; repeat it; the first is the "
            "reference (default: all nine, bf16_bf16_eager first)"
        ),
    ] = None,
    package_bf16: Annotated[Path | None, typer.Option(help="the bf16 joint block's package (diffu-moss-aoti)")] = None,
    package_mxfp8: Annotated[Path | None, typer.Option(help="the mxfp8 joint block's package")] = None,
    size: Annotated[str, typer.Option(help="the page's <width>x<height> in px, multiples of 16")] = "1024x1024",
    steps: Annotated[int, typer.Option(min=1, help="denoising steps per drawing")] = 10,
    seed: Annotated[int, typer.Option(help="the noise seed every configuration draws from")] = 0,
    prompt_file: Annotated[
        Path | None, typer.Option(help="a file holding the prompt (default: a synthetic page's Layout prompt)")
    ] = None,
    device: Annotated[str, typer.Option(help="where the model runs")] = "cuda",
) -> None:
    """Measure MOSS-Image's drawing speed, memory and fidelity in each precision and way of running its blocks."""
    width, _, height = size.partition("x")
    settings_size = (int(width), int(height))
    configs = chosen_configs(config)
    packages = {
        precision: path
        for precision, path in ((Precision.BF16, package_bf16), (Precision.MXFP8, package_mxfp8))
        if path is not None
    }
    try:
        options = [chosen.inference(packages) for chosen in configs]  # every configuration checked before any runs
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    prompt = prompt_file.read_text(encoding="utf-8") if prompt_file else synthetic_prompt(*settings_size)
    settings = BenchSettings(model=model, device=device, size=settings_size, prompt=prompt, steps=steps, seed=seed)
    reference: Drawing | None = None
    for chosen, option in zip(configs, options, strict=True):
        console.print(f"[bold]{chosen.name}[/bold]")
        try:
            record, drawing = measure(chosen, option, settings)
        except DEVICE_ERRORS as error:
            emit(BenchFailure.of(chosen, error), out)
            raise
        except Exception as error:  # one configuration failing leaves the others to measure
            console.print_exception()
            emit(BenchFailure.of(chosen, error), out)
            continue
        reference = drawing if reference is None else reference
        emit(
            record.model_copy(
                update={
                    "velocity_cosine": cosine(drawing.velocity, reference.velocity),
                    "latents_cosine": cosine(drawing.latents, reference.latents),
                }
            ),
            out,
        )
        if images is not None:
            images.mkdir(parents=True, exist_ok=True)
            drawing.page.save(images / f"{chosen.name}.png")


if __name__ == "__main__":
    app()
