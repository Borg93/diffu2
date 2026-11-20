"""The model the demo draws with, loaded once, and a planned page drawn and read back with it.

The model is the page pipeline's own (:func:`diffu_page.draw.pipeline.load_pipeline` on a MOSS-Image model directory,
``diffu-moss-init``'s): MOSS-Image-Base's text encoder, tokenizer and VAE around the fine-tuned DiT, both downloaded
at pinned revisions, and the pipeline's line judge, a Swedish HTR model that reads every drawn line back. A distilled
student, when one is given (a ``diffu-moss-distill`` checkpoint: a LoRA of the DiT that draws in a few steps without
guidance), is loaded beside the DiT's weights, unmerged, and switched on for a fast page and off for a guided one.
:func:`draw` runs the crop pipeline as ``diffu_page.draw.pipeline.draw_crop`` does, the guidance scale set per page,
and keeps the drawn page when the read-back fails: a page costs its author GPU time, its labels little of it.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import torch
from diffusers.modular_pipelines import ModularPipeline
from huggingface_hub import snapshot_download
from peft import set_peft_model_state_dict
from PIL import Image
from pydantic import BaseModel, ConfigDict
from safetensors.torch import load_file

from diffu_moss.init import write_moss_repo
from diffu_moss.train.distill import DISTILL_RECORD, DistillRecord
from diffu_moss.train.loop import add_lora
from diffu_moss.train.records import ADAPTER_RECORD, ADAPTER_WEIGHTS, AdapterRecord
from diffu_moss.transformer import MossTransformer
from diffu_page.draw.guidance import ReferenceGuidance
from diffu_page.draw.pipeline import DrawSettings, crop_blocks, load_pipeline
from diffu_page.draw.plan import CropPlan
from diffu_page.page.crops.geometry import LayoutBox
from diffu_page.quality.judge import Judge, LineVerdict, TrOCRJudge

BASE_REPO = "OpenMOSS-Team/MOSS-Image-Base"
BASE_REVISION = "85975793d8b3d432f1b6e27f352abaa4c2a4c0d3"
BASE_FILES = ["text_encoder/*", "tokenizer/*", "vae/*"]
DIT_REPO = "Gabriel/diffu-moss-above"
DIT_REVISION = "3d5d3d966a0d32d4b9573c3d6a6d2324d19bb5a8"
DIT_FILES = ["config.json", "export.json", "*.safetensors"]
BASE_ENV, DIT_ENV = "PAGEDEMO_BASE", "PAGEDEMO_DIT"  # local folders in place of the downloads, for a run off the Hub
STUDENT_ENV = "PAGEDEMO_STUDENT"  # a student's checkpoint folder in place of the Space's own
STUDENT_FILES = (ADAPTER_WEIGHTS, ADAPTER_RECORD, DISTILL_RECORD)


class Models:
    """What a page is drawn and read with: the crop pipeline and the line judge, on their device, and how the
    pipeline's student draws (None: no student is loaded)."""

    def __init__(self, pipe: ModularPipeline, judge: Judge, student: DistillRecord | None = None) -> None:
        self.pipe = pipe
        self.judge = judge
        self.student = student


class Drawn(BaseModel):
    """A drawn page: its image, the judge's verdict on each line of its plan (None when the lines could not be read
    back, with why), and how long the drawing and the reading took, seconds."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Image.Image
    verdicts: list[LineVerdict] | None
    read_error: str = ""
    draw_seconds: float
    read_seconds: float


def model_directory() -> Path:
    """A MOSS-Image model directory of the Base snapshot's components around the fine-tuned DiT, each downloaded at
    its pinned revision (or the local folders ``PAGEDEMO_BASE`` and ``PAGEDEMO_DIT`` name)."""
    base = os.environ.get(BASE_ENV) or snapshot_download(BASE_REPO, revision=BASE_REVISION, allow_patterns=BASE_FILES)
    dit = os.environ.get(DIT_ENV) or snapshot_download(DIT_REPO, revision=DIT_REVISION, allow_patterns=DIT_FILES)
    out = Path(tempfile.mkdtemp(prefix="diffu-moss-"))
    write_moss_repo(out, base=Path(base), transformer=Path(dit))
    return out


def student_folder(own: Path) -> Path | None:
    """The student's checkpoint folder: the one ``PAGEDEMO_STUDENT`` names, else ``own`` (the Space's, staged with
    it) when it holds a checkpoint's files; None when there is neither: the pages are then drawn guided only."""
    named = os.environ.get(STUDENT_ENV)
    folder = Path(named) if named else own
    return folder if all((folder / name).is_file() for name in STUDENT_FILES) else None


def add_student(transformer: MossTransformer, checkpoint: Path) -> DistillRecord:
    """The student saved in ``checkpoint`` added to ``transformer`` as it was trained: its LoRA in float32 beside the
    DiT's weights, not merged into them (a merge into bfloat16 rounds most of a short distillation's update away).

    Returns:
        How the student is drawn: its steps and noise levels' shift, without guidance.

    Raises:
        ValueError: If the checkpoint's weights are not its LoRA's.
    """
    add_lora(transformer, AdapterRecord.read(checkpoint / ADAPTER_RECORD).lora)
    loaded = set_peft_model_state_dict(transformer, load_file(str(checkpoint / ADAPTER_WEIGHTS)))
    if loaded.unexpected_keys:
        raise ValueError(f"{checkpoint}: weights that are not its LoRA's")
    return DistillRecord.read(checkpoint / DISTILL_RECORD)


def load_models(device: str, *, student: Path) -> Models:
    """The crop pipeline of the fine-tuned model, its student if there is one (:func:`student_folder`, the Space's
    own in ``student``), and the line judge, on ``device``. The pipeline is built on the CPU and moved whole, the
    student's LoRA with it."""
    pipe = load_pipeline(model_directory(), crop_blocks, loras={}, settings=DrawSettings(device="cpu"))
    folder = student_folder(student)
    record = None if folder is None else add_student(pipe.transformer, folder)
    pipe.to(device)
    return Models(pipe=pipe, judge=TrOCRJudge(device=device), student=record)


def draw(
    models: Models,
    plan: CropPlan,
    sample: Image.Image,
    sample_lines: list[LayoutBox],
    *,
    steps: int,
    guidance: float,
    seed: int,
    fast: bool = False,
) -> Drawn:
    """``plan`` drawn from noise seed ``seed`` in ``steps`` steps at guidance scale ``guidance`` (exactly 1: one pass
    a step), or ``fast``: by the student, in its own steps without guidance; from ``sample`` (cut by ``sample_lines``
    when the plan is not a whole page's), then every line read back by the judge.

    Raises:
        ValueError: If a fast page is asked of models without a student.
    """
    pipe = models.pipe
    if fast and models.student is None:
        raise ValueError("a fast page needs the distilled student, and none is loaded")
    if models.student is not None:
        if fast:
            pipe.transformer.enable_adapters()
            steps, guidance = models.student.student_steps, models.student.guidance_scale
        else:
            pipe.transformer.disable_adapters()
    pipe.update_components(guider=ReferenceGuidance(guidance_scale=guidance))
    # the text encoder encodes the negative prompt only if the guider guides at its current step: the first, here
    pipe.guider.set_state(step=0, num_inference_steps=steps, timestep=None)
    reference: dict[str, object] = {"image": sample}
    if sample_lines:
        reference["reference_lines"] = list(sample_lines)
    started = time.perf_counter()
    image = pipe(
        crop_plan=plan,
        generator=torch.Generator().manual_seed(seed),
        num_inference_steps=steps,
        **reference,
        output="images",
    )[0]
    drawn = time.perf_counter()
    verdicts: list[LineVerdict] | None = None
    read_error = ""
    try:
        verdicts = models.judge(image, plan.lines)
    except Exception as error:  # the page is drawn and paid for: it is returned without its read-back
        read_error = f"{type(error).__name__}: {error}"
    return Drawn(
        image=image,
        verdicts=verdicts,
        read_error=read_error,
        draw_seconds=drawn - started,
        read_seconds=time.perf_counter() - drawn,
    )
