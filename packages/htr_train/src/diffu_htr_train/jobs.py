"""A training run as one YAML file (:class:`RunSpec`): what to train (one of the four jobs, a discriminated union on
``kind``), on which sets, with which resources; loaded and checked before any tool, GPU or model is touched
(:func:`load_run_spec`, :func:`check_run`).

The three ms-swift jobs share :class:`SwiftSFT`, whose fields are ms-swift's own flags (``swift sft``), each passed only
when it is set, so a spec states every choice it makes and leaves ms-swift's defaults where it does not. Two defaults
are traps, so the spec must state them: ``--freeze_vit`` and ``--freeze_aligner`` default to true in ms-swift even for a
full fine-tune (``swift/arguments/tuner_args.py:123-124`` at ms-swift ``ef4953a4``), and OpenBMB's MiniCPM-OCR recipe
trains the vision encoder. The draft job (:class:`DSparkJob`) is speculators', whose data comes from the target model
served by vLLM.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, model_validator

from diffu_htr_train.evaluate.in_training import CerSet
from diffu_htr_train.jev.prompt import prompt_digest
from diffu_htr_train.output import validation_problems
from diffu_htr_train.records import Sample, SampleMeta, meta_path, read_jsonl, read_meta, target_record_path

COMMIT = re.compile(r"^[0-9a-f]{40}$")
# The one model whose code is run from the Hub: Unlimited-OCR 2's, at the revision its environment was built against
UNLIMITED_OCR_2 = "baidu/Unlimited-OCR-2"
UNLIMITED_OCR_2_REVISION = "0480966938291445591f69f08a13b041c073a96a"
# vLLM flags a draft's spec may not pass itself, and why
REFUSED_SERVE_FLAGS = {
    "--trust-remote-code": "MiniCPM-OCR is native in vLLM; no remote code runs but Unlimited-OCR 2's",
    "--allowed-local-media-path": "it is the job's media_root",
    "--served-model-name": "the target is served by its path, which the responses record",
}


class JobRefused(ValueError):
    """A run that cannot start as asked: its spec, data, model or resources do not go together."""


class Kind(StrEnum):
    """What a run trains."""

    MINICPM_OCR = "minicpm-ocr"
    UNLIMITED_OCR_2 = "unlimited-ocr-2"
    JEV = "jev"
    DSPARK = "dspark"


class ToolEnv(StrEnum):
    """The tool environments under ``packages/htr_train/envs/``, each its own uv project and lock."""

    MINICPM = "minicpm"  # ms-swift + transformers 5.22: MiniCPM-OCR and Jev
    UOCR2 = "uocr2"  # ms-swift + transformers 4.52.1: Unlimited-OCR 2 (its remote code)
    SPECULATORS = "speculators"
    VLLM = "vllm"


class Where(StrEnum):
    """Where a run is launched: the DGX Spark, through its memory guard, a Slurm cluster, or a server's own GPUs (one
    node, the GPUs ``HTR_TRAIN_LOCAL_GPUS`` names)."""

    SPARK = "spark"
    SLURM = "slurm"
    LOCAL = "local"


class ModelRef(BaseModel):
    """A model to load: a Hub repo at a full commit, or a local folder (a checkpoint of ours, a staged snapshot),
    written as a path (starting with ``/``, ``.`` or ``~``) so a spec means the same from any folder."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str
    revision: str | None = None

    @property
    def is_local(self) -> bool:
        """Whether the model is a folder on disk rather than a Hub repo: its value is written as a path."""
        return self.repo.startswith(("/", ".", "~"))

    @model_validator(mode="after")
    def _pinned(self) -> Self:
        if self.is_local and self.revision is not None:
            raise ValueError(f"{self.repo} is a local folder: it has no revision")
        if not self.is_local and (self.revision is None or not COMMIT.match(self.revision)):
            raise ValueError(
                f"{self.repo} is a Hub repo (a local folder is written as a path: /, ./ or ~/): pin it to a full "
                f"40-character commit, not {self.revision!r}"
            )
        return self


class DataPart(BaseModel):
    """A training set (a ``diffu-reader-set`` or ``htr-train jev-set`` JSONL, its ``.meta.jsonl`` beside it) and how
    many of its samples to draw (ms-swift's ``path#N``; None: all of them)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path
    samples: int | None = Field(default=None, gt=0)


class DataRef(BaseModel):
    """A run's sets: the training parts, mixed by ms-swift's sampling, and the dev set it is evaluated on (cut by
    volume, ``htr-train split-dev``; None: no evaluation)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train: list[DataPart] = Field(min_length=1)
    dev: Path | None = None


class Resources(BaseModel):
    """What a run needs: nodes and GPUs (the Spark has one), the memory its guard checks for (``--need-gb``) and its
    time on a cluster."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    nodes: int = Field(default=1, ge=1)
    gpus_per_node: int = Field(default=1, ge=1)
    need_gb: int = Field(gt=0)
    hours: float = Field(gt=0)
    cpus_per_task: int = Field(default=16, ge=1)

    @property
    def gpus(self) -> int:
        """The run's GPUs over all its nodes."""
        return self.nodes * self.gpus_per_node


class LoraSpec(BaseModel):
    """A LoRA's shape (ms-swift's ``--lora_rank``, ``--lora_alpha``, ``--target_modules``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    target_modules: str = "all-linear"


type Strategy = Literal["no", "steps", "epoch"]


class SwiftSFT(BaseModel):
    """What an ms-swift SFT job states, each field one ``swift sft`` flag (None: not passed, ms-swift's default).

    ``freeze_vit`` and ``freeze_aligner`` have no default (see the module's note); ``global_batch`` is the batch over
    every GPU, from which the gradient accumulation follows. ``env`` is the template's settings, read by ms-swift from
    the environment (``DOWNSAMPLE_MODE``, ``MAX_SLICE_NUMS``, ``IMAGE_MAX_TOKEN_NUM``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: ModelRef
    tuner: Literal["full", "lora", "lora_llm"]
    freeze_vit: bool
    freeze_aligner: bool
    freeze_llm: bool | None = None
    lora: LoraSpec | None = None
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    # optimisation
    learning_rate: float = Field(gt=0)
    epochs: float | None = Field(default=None, gt=0)
    max_steps: int | None = Field(default=None, gt=0)
    global_batch: int = Field(gt=0)
    per_device_batch: int = Field(default=1, gt=0)
    lr_scheduler: str | None = None
    warmup_ratio: float | None = Field(default=None, ge=0, lt=1)
    optim: str | None = None
    adam_beta1: float | None = None
    adam_beta2: float | None = None
    adam_epsilon: float | None = None
    weight_decay: float | None = None
    max_grad_norm: float | None = None
    seed: int | None = None
    data_seed: int | None = None
    # sequences and memory
    max_length: int = Field(gt=0)
    packing: bool | None = None
    packing_length: int | None = Field(default=None, gt=0)
    packing_num_proc: int | None = Field(default=None, gt=0)
    padding_free: bool | None = None
    attn_impl: Literal["sdpa", "flash_attn", "eager"] | None = None
    deepspeed: str | None = None  # zero2, zero3 or a config file
    gradient_checkpointing: bool | None = None
    vit_gradient_checkpointing: bool | None = None
    gradient_checkpointing_kwargs: dict[str, bool] | None = None
    # data loading
    dataset_num_proc: int | None = Field(default=None, gt=0)
    dataloader_num_workers: int | None = Field(default=None, ge=0)
    load_from_cache_file: bool | None = None
    dataset_shuffle: bool | None = None
    train_dataloader_shuffle: bool | None = None
    # evaluation, checkpoints and logs
    eval_strategy: Strategy | None = None
    eval_steps: int | None = Field(default=None, gt=0)
    save_strategy: Strategy | None = None
    save_steps: int | None = Field(default=None, gt=0)
    save_total_limit: int | None = Field(default=None, gt=0)
    save_only_model: bool | None = None
    best_by_eval_loss: bool = False
    per_device_eval_batch_size: int | None = Field(default=None, gt=0)
    logging_steps: int | None = Field(default=None, gt=0)
    ddp_timeout: int | None = Field(default=None, gt=0)
    report_to: Literal["trackio", "tensorboard", "none"] = "trackio"
    # samples the model in training reads at each evaluation, their CER logged as eval_cer_<name>
    # (diffu_htr_train.evaluate.in_training); cer_at_start reads them once when training starts or resumes too
    cer_sets: dict[Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]+$")], CerSet] = Field(default_factory=dict)
    cer_at_start: bool = False
    env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (self.epochs is None) == (self.max_steps is None):
            raise ValueError("give the run's length as epochs or as max_steps, one of them")
        if (self.lora is not None) != (self.tuner != "full"):
            raise ValueError(f"a {self.tuner} tuner {'needs' if self.tuner != 'full' else 'takes no'} lora settings")
        if self.packing and self.attn_impl != "flash_attn":
            raise ValueError("ms-swift packs sequences only with attn_impl flash_attn")
        if self.best_by_eval_loss:
            self._check_best_by_eval_loss()
        if self.cer_sets and self.eval_strategy in (None, "no") and not self.cer_at_start:
            raise ValueError("cer_sets are read at each evaluation: give an eval_strategy that evaluates")
        return self

    def _check_best_by_eval_loss(self) -> None:
        """transformers' rules for ``load_best_model_at_end``, refused here rather than when the Trainer starts on a
        cluster: it saves when it evaluates, and with DeepSpeed it needs the optimizer's state saved too."""
        if self.eval_strategy in (None, "no") or self.save_strategy != self.eval_strategy:
            raise ValueError("best_by_eval_loss needs save_strategy equal to an eval_strategy that evaluates")
        if self.eval_strategy == "steps" and (
            self.eval_steps is None or self.save_steps is None or self.save_steps % self.eval_steps
        ):
            raise ValueError("best_by_eval_loss needs save_steps a multiple of eval_steps")
        if self.deepspeed is not None and self.save_only_model:
            raise ValueError(
                "best_by_eval_loss cannot load a best checkpoint saved with save_only_model under DeepSpeed"
            )


class MiniCPMOCRJob(SwiftSFT):
    """MiniCPM-OCR 1.0 as a line or region recogniser (``minicpm_ocr``), as OpenBMB fine-tunes it on handwritten lines:
    the empty thinking prefix added to each answer and left out of the loss."""

    kind: Literal[Kind.MINICPM_OCR] = Kind.MINICPM_OCR
    add_non_thinking_prefix: bool = True
    loss_scale: str | None = "ignore_empty_think"


class UnlimitedOCR2Job(SwiftSFT):
    """Unlimited-OCR 2 as a page reader (``unlimited-ocr``), LoRA or full; its remote code runs, at its pinned
    revision only."""

    kind: Literal[Kind.UNLIMITED_OCR_2] = Kind.UNLIMITED_OCR_2

    @model_validator(mode="after")
    def _logs_where_its_transformers_can(self) -> Self:
        if self.report_to == "trackio":
            raise ValueError(
                "Unlimited-OCR 2 trains under transformers 4.52.1, whose TrainingArguments has no trackio project: "
                "set report_to tensorboard"
            )
        return self

    @model_validator(mode="after")
    def _pinned_remote_code(self) -> Self:
        if self.model.is_local:
            return self
        if (self.model.repo, self.model.revision) != (UNLIMITED_OCR_2, UNLIMITED_OCR_2_REVISION):
            raise ValueError(
                f"Unlimited-OCR 2 runs its own code: only {UNLIMITED_OCR_2}@{UNLIMITED_OCR_2_REVISION} is trusted"
            )
        return self


class JevJob(SwiftSFT):
    """MiniCPM-OCR-Jev (a Qwen3.5 sequence classifier, ``qwen3_5``) fine-tuned on its own input form: a crop, a
    candidate text and that text rendered, labelled 1 when the text is what the crop shows. ``renderer`` is the
    renderer version the pairs were built with (:func:`diffu_htr_train.jev.render.renderer_version`)."""

    kind: Literal[Kind.JEV] = Kind.JEV
    renderer: str


class DSparkJob(BaseModel):
    """A DSpark draft for a fine-tuned MiniCPM-OCR, trained by speculators on hidden states its target served by vLLM
    gives: ``responses`` are that same target's answers (``htr-train dspark-set``), checked against ``target``.

    ``target_layer_ids`` mirror OpenBMB's released draft (``[3, 7, 11, 15, 19]`` in Z Lab's convention, which reads
    ``hidden_states[id + 1]``); vLLM is also asked for the last layer (``target_layers``), as speculators' own launcher
    adds it. vLLM may read local images below ``media_root`` only (the responses' crops); ``serve_args`` are vLLM's
    other flags, never ``--trust-remote-code`` (MiniCPM-OCR is native in vLLM). ``draft_vocab_size`` is speculators'
    DSpark example's (its default is the whole vocabulary); OpenBMB's released draft crops it to 114,688 tokens. ``max_anchors`` and
    ``confidence_head_alpha`` are speculators' DSpark example's; the cosine schedule with 4 % warmup is its DSpark
    research runs' (speculators' default is linear, no warmup); two dataloader workers a rank, as those runs found the
    default 12 exhausts a node's threads beside vLLM. The best checkpoint by validation loss is always kept
    (``--save-best``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[Kind.DSPARK] = Kind.DSPARK
    target: ModelRef
    target_layers: int = Field(default=24, gt=0)
    responses: Path
    media_root: Path
    serve_args: list[str] = Field(default_factory=list)
    vllm_memory: float = Field(default=0.35, gt=0, lt=1)
    port: int = Field(default=8000, ge=1024, le=65535)
    target_layer_ids: list[int] = Field(default_factory=lambda: [4, 8, 12, 16, 20])
    block_size: int = Field(default=15, gt=0)
    num_layers: int = Field(default=5, gt=0)
    draft_vocab_size: int = Field(default=32000, gt=0)
    mask_token_id: int = 248076
    markov_rank: int = 256
    markov_head_type: Literal["vanilla", "gated", "rnn"] = "vanilla"
    loss_fn: dict[str, float] = Field(default_factory=lambda: {"ce": 0.1, "tv": 0.9})
    confidence_head_alpha: float = Field(default=1.0, ge=0)
    max_anchors: int = Field(default=3072, gt=0)
    lr: float = Field(default=3e-4, gt=0)
    scheduler: Literal["linear", "cosine", "none"] = "cosine"
    warmup_ratio: float = Field(default=0.04, ge=0, lt=1)
    epochs: int = Field(default=5, gt=0)
    num_workers: int = Field(default=2, ge=0)
    seq_length: int = Field(default=4096, gt=0)
    max_samples: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _layers_in_target(self) -> Self:
        if not self.target.is_local:
            raise ValueError(
                f"the target {self.target.repo} must be a local folder (its published commit's snapshot): speculators "
                "loads a Hub model at its latest commit, which may not be the one the responses came from"
            )
        ids = self.target_layer_ids
        if not ids or min(ids) < 0 or max(ids) >= self.target_layers or len(set(ids)) != len(ids):
            raise ValueError(f"target_layer_ids {ids} must be distinct layers of the target's {self.target_layers}")
        refused = [flag for flag in self.serve_args if flag.split("=")[0] in REFUSED_SERVE_FLAGS]
        if refused:
            raise ValueError(f"serve_args may not hold {refused[0]}: {REFUSED_SERVE_FLAGS[refused[0].split('=')[0]]}")
        return self


type Job = Annotated[MiniCPMOCRJob | UnlimitedOCR2Job | JevJob | DSparkJob, Field(discriminator="kind")]
type SwiftJob = MiniCPMOCRJob | UnlimitedOCR2Job | JevJob  # the jobs ms-swift trains


class RunSpec(BaseModel):
    """One YAML file: the run's name (its folder ``<runs>/<name>/`` and its trackio run), what variant it is (for the
    runs compared later: which sets, mixed how, scaled or not), its job, its sets and its resources."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str = ""
    job: Job
    data: DataRef | None = None
    resources: Resources

    @model_validator(mode="after")
    def _data_for_sft(self) -> Self:
        if isinstance(self.job, DSparkJob) != (self.data is None):
            raise ValueError("an ms-swift job trains on data; a DSpark job on its responses alone (no data)")
        if (
            not isinstance(self.job, DSparkJob)
            and self.job.dataset_shuffle is False
            and self.data is not None
            and any(part.samples for part in self.data.train)
        ):
            raise ValueError(
                "a training part draws samples (#N) but dataset_shuffle is false, so ms-swift would take its first N "
                "rows: set dataset_shuffle true"
            )
        if isinstance(self.job, DSparkJob) and self.resources.gpus != 1:
            raise ValueError(
                f"a DSpark run trains on one GPU beside its own vLLM on one node, not {self.resources.gpus} GPUs"
            )
        if not isinstance(self.job, DSparkJob) and self.job.global_batch % (
            self.resources.gpus * self.job.per_device_batch
        ):
            raise ValueError(
                f"global_batch {self.job.global_batch} is no multiple of {self.resources.gpus} GPUs x "
                f"{self.job.per_device_batch} a device"
            )
        return self


def load_run_spec(path: Path) -> RunSpec:
    """The run spec in the YAML file at ``path``.

    Raises:
        JobRefused: If the file cannot be read, is not YAML, or is not a run spec.
    """
    try:
        given = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise JobRefused(f"{path} is not a YAML run spec: {error}") from error
    try:
        return RunSpec.model_validate(given)
    except ValidationError as error:
        raise JobRefused(f"{path}: {validation_problems(error, 'spec')}") from error


def _check_set(path: Path) -> list[SampleMeta]:
    """A set's meta, its samples read, their images checked to exist and its meta to match it line for line.

    Raises:
        JobRefused: If the set, its meta or an image it names is missing, a line is not a sample or a sample's meta,
            or the meta has another number of lines.
    """
    beside = meta_path(path)
    for needed in (path, beside):
        if not needed.is_file():
            raise JobRefused(f"{needed} does not exist: build the set and its meta with diffu-reader-set")
    try:
        rows = read_jsonl(path, Sample)
        meta = read_meta(beside)
    except ValueError as error:
        raise JobRefused(str(error)) from error
    if not rows:
        raise JobRefused(f"{path} holds no samples")
    for number, row in enumerate(rows, start=1):
        missing = [image for image in row.images if not Path(image).is_file()]
        if missing:
            raise JobRefused(f"{path}:{number} names an image that does not exist: {missing[0]}")
    if len(meta) != len(rows):
        raise JobRefused(f"{beside} has {len(meta)} lines for {len(rows)} samples in {path.name}")
    return meta


def _check_trainable(path: Path, meta: SampleMeta) -> None:
    """A sample may be trained on: a drawn one always, a real one only when its set was split by the GT's test lists
    and it is on the training side (a real sample of an unsplit set may be of a test page).

    Raises:
        JobRefused: If it is a real sample of a test page, or of an unsplit set.
    """
    if not meta.is_of("train"):
        why = "a sample of a GT test page" if meta.split == "test" else "a real sample of a set not split by the GT"
        raise JobRefused(
            f"{path} holds {meta.id}, {why}: never train on one (build real sets with --split-from-gt-test-lists)"
        )


def check_data(data: DataRef) -> None:
    """The leak checks, once at plan time: every set and image exists, no real sample but the GT's training pages' is
    trained on (:func:`_check_trainable`), and the dev set's volumes are named and none is a training set's.

    Raises:
        JobRefused: If a set or image is missing, a set holds a sample not to train on, a dev sample names no volume,
            or a volume is on both sides.
    """
    train_volumes: set[tuple[str, str]] = set()
    for part in data.train:
        for meta in _check_set(part.path):
            _check_trainable(part.path, meta)
            if meta.volume:
                train_volumes.add((meta.collection, meta.volume))
    if data.dev is None:
        return
    for meta in _check_set(data.dev):
        _check_trainable(data.dev, meta)
        if not meta.volume:
            raise JobRefused(f"{data.dev}: {meta.id} names no volume, so no check can keep it apart from training")
        if (meta.collection, meta.volume) in train_volumes:
            raise JobRefused(
                f"{data.dev} shares the volume {meta.collection}/{meta.volume} with the training sets: "
                "cut the dev set by volume (htr-train split-dev)"
            )


def check_jev_pairs(job: JevJob, data: DataRef) -> None:
    """A Jev run's pairs were all rendered by the job's renderer and built in this package's prompt.

    Raises:
        JobRefused: If a training or dev pair's meta names another renderer or prompt, or none.
    """
    prompt = prompt_digest()
    for path in [part.path for part in data.train] + ([data.dev] if data.dev is not None else []):
        for meta in read_meta(meta_path(path)):
            if meta.renderer != job.renderer:
                raise JobRefused(
                    f"{path}: {meta.id} was rendered by {meta.renderer or 'no renderer'}, the job trains on "
                    f"{job.renderer}: a new renderer is a new input, rebuild the pairs"
                )
            if meta.prompt != prompt:
                raise JobRefused(
                    f"{path}: {meta.id} was built in the prompt {meta.prompt or 'of no record'}, Jev trains in "
                    f"{prompt} (diffu_htr_train.jev.prompt): rebuild the pairs"
                )


class ResponsesMeta(BaseModel):
    """What ``htr-train dspark-set`` records of a response set (``<stem>.target.json``): the target folder that
    answered (as an absolute path; vLLM served it by that name), the samples answered and failed, the share of prompts
    allowed to fail, and the answers' length at most."""

    model_config = ConfigDict(frozen=True)

    target: str
    samples: int = Field(ge=0)
    failed: int = Field(ge=0)
    max_failed: float = Field(ge=0, lt=1)
    max_tokens: int = Field(gt=0)

    @property
    def fit_to_train(self) -> bool:
        """Whether the set holds answers and no more of its prompts failed than allowed."""
        return self.samples > 0 and self.failed <= self.max_failed * (self.samples + self.failed)


def check_responses(job: DSparkJob) -> None:
    """A DSpark run's responses were generated by its own target (the on-policy check).

    Raises:
        JobRefused: If the responses or their record are missing, another model or revision answered them, or the set
            is not fit to train on (:attr:`ResponsesMeta.fit_to_train`).
    """
    record = target_record_path(job.responses)
    if not job.responses.is_file() or not record.is_file():
        raise JobRefused(f"{job.responses} and its {record.name} must exist: build them with htr-train dspark-set")
    try:
        made_by = ResponsesMeta.model_validate_json(record.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise JobRefused(f"{record}: {validation_problems(error, 'record')}") from error
    target = Path(job.target.repo).expanduser().resolve()
    if Path(made_by.target) != target:
        raise JobRefused(
            f"{job.responses} was answered by {made_by.target}, the draft is for {target}: a draft learns its own "
            "target's answers"
        )
    if not made_by.fit_to_train:
        raise JobRefused(
            f"{job.responses} holds {made_by.samples} answers and {made_by.failed} failed prompts (at most "
            f"{made_by.max_failed:.0%} may fail): rebuild it with htr-train dspark-set"
        )


def check_run(spec: RunSpec) -> None:
    """Every check a run passes before it is planned: its data (:func:`check_data`), a Jev run's renderer and prompt, a DSpark
    run's responses, and a local model that exists.

    Raises:
        JobRefused: If any check fails.
    """
    if isinstance(spec.job, DSparkJob):
        check_responses(spec.job)
        model = spec.job.target
    else:
        if spec.data is None:
            raise JobRefused(f"{spec.name} trains on no data")
        check_data(spec.data)
        if isinstance(spec.job, JevJob):
            check_jev_pairs(spec.job, spec.data)
        model = spec.job.model
    if model.is_local and not Path(model.repo).expanduser().is_dir():
        raise JobRefused(f"{model.repo} is no model folder")


def check_where(
    spec: RunSpec, where: Where, *, memory_gb: int, floor_gb: int, local_gpus: tuple[int, ...] = ()
) -> None:
    """A run fits where it is launched: on the Spark, one GPU, and its memory plus the guard's floor within the box's;
    on a server's own GPUs (``local_gpus``), one node and as many GPUs as it names.

    Raises:
        JobRefused: If a Spark run asks for more than one GPU or more memory than the box has, or a local run for
            more than one node or another number of GPUs than are named.
    """
    if where is Where.LOCAL:
        if spec.resources.nodes != 1:
            raise JobRefused(
                f"{spec.name} asks for {spec.resources.nodes} nodes: a local run has one; launch it on slurm"
            )
        if len(local_gpus) != spec.resources.gpus_per_node:
            raise JobRefused(
                f"{spec.name} asks for {spec.resources.gpus_per_node} GPUs and HTR_TRAIN_LOCAL_GPUS names "
                f"{len(local_gpus)} ({','.join(map(str, local_gpus)) or 'none'}): name as many as it takes"
            )
        return
    if where is not Where.SPARK:
        return
    if spec.resources.gpus != 1:
        raise JobRefused(f"{spec.name} asks for {spec.resources.gpus} GPUs: the Spark has one; launch it on slurm")
    if spec.resources.need_gb + floor_gb > memory_gb:
        raise JobRefused(
            f"{spec.name} needs {spec.resources.need_gb} GB and the guard keeps {floor_gb} GB free: the Spark has "
            f"{memory_gb} GB"
        )
