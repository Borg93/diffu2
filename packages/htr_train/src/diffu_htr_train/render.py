"""A run spec rendered to the exact processes it starts (:class:`Command`): which tool environment, which argv, which
environment variables. :func:`swift_sft_argv` is the one place that knows ``swift sft``'s flag names; each kind's
renderer adds what its model needs (:func:`render` picks it by the job). Every tool is called by its path in its own
environment, never imported.

ms-swift's choices made for every run, beyond a spec's fields: the Hub is Hugging Face's (``--use_hf true``; ms-swift
defaults to ModelScope); a set is never split by sample (``--split_dataset_ratio 0``: lines of one page would land on
both sides), the dev set is given whole (``--val_dataset``); checkpoints go straight into ``<run>/output``
(``--add_version false``); trackio logs to the local project under the run's name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Self, assert_never

from pydantic import BaseModel, ConfigDict, Field, model_validator

from diffu_htr_train.evaluate.in_training import AT_START_ENV, CER_SETS_ENV
from diffu_htr_train.jobs import (
    DataRef,
    DSparkJob,
    JevJob,
    Kind,
    MiniCPMOCRJob,
    ModelRef,
    RunSpec,
    SwiftJob,
    SwiftSFT,
    ToolEnv,
    UnlimitedOCR2Job,
)

type Flag = tuple[str, object]  # a flag and its value; None leaves the flag out


class Command(BaseModel):
    """One process a run starts: in which tool environment, with which argv and extra environment. A ``serve``
    command runs in the background for the commands after it and is ready when ``ready`` answers (a URL); a ``run``
    command has none. A command with ``resume_from`` (ms-swift's output folder) resumes from the newest whole
    checkpoint there when its steps run again, so a stopped training is continued by running its steps again."""

    model_config = ConfigDict(frozen=True)

    env: ToolEnv
    argv: list[str]
    environ: dict[str, str] = Field(default_factory=dict)
    role: Literal["run", "serve"] = "run"
    ready: str | None = None
    resume_from: Path | None = None

    @model_validator(mode="after")
    def _ready_when_served(self) -> Self:
        if (self.role == "serve") != (self.ready is not None):
            raise ValueError(f"a {self.role} command {'needs a' if self.role == 'serve' else 'takes no'} ready URL")
        return self


class RunPaths(BaseModel):
    """Where a run's files go: its folder, ms-swift's or speculators' output in it, and the tool environments."""

    model_config = ConfigDict(frozen=True)

    root: Path
    envs: Path
    trackio_project: str

    @property
    def output(self) -> Path:
        """ms-swift's checkpoints (``--output_dir``), speculators' (``--save-path``)."""
        return self.root / "output"

    def tool(self, env: ToolEnv, name: str) -> str:
        """A tool's executable in its environment."""
        return str(self.envs / env / ".venv" / "bin" / name)


def value(given: object) -> str:
    """A flag's value as ms-swift and speculators parse it: booleans in lower case, dicts as JSON."""
    match given:
        case bool():
            return str(given).lower()
        case dict():
            return json.dumps(given)
        case _:
            return str(given)


def flags(pairs: list[Flag]) -> list[str]:
    """``pairs`` as argv, each flag followed by its value, a flag whose value is None left out."""
    return [part for flag, given in pairs if given is not None for part in (flag, value(given))]


def model_flags(model: ModelRef) -> list[Flag]:
    """The model to load: a local folder by its absolute path, a Hub repo at its commit."""
    repo = str(Path(model.repo).expanduser().resolve()) if model.is_local else model.repo
    return [("--model", repo), ("--model_revision", model.revision)]


def dataset_args(data: DataRef) -> list[str]:
    """``--dataset`` with each training part (``path#N`` for N of its samples), the dev set as ``--val_dataset``, and
    no split by sample."""
    parts = [
        f"{part.path.resolve()}#{part.samples}" if part.samples else str(part.path.resolve()) for part in data.train
    ]
    return [
        "--dataset",
        *parts,
        *flags([("--val_dataset", data.dev.resolve() if data.dev else None), ("--split_dataset_ratio", 0)]),
    ]


def swift_sft_argv(job: SwiftSFT, spec: RunSpec, paths: RunPaths) -> list[str]:
    """The flags every ``swift sft`` run passes, from ``job``'s fields (each set field one flag) and the spec's data
    and resources; the model's own flags (its type and template) are its renderer's."""
    data = spec.data
    if data is None:
        raise ValueError(f"{spec.name} trains on no data")
    accumulation = job.global_batch // (spec.resources.gpus * job.per_device_batch)
    lora = job.lora
    best = job.best_by_eval_loss or None
    trackio = job.report_to == "trackio"
    return [
        *flags(
            [
                ("--tuner_type", job.tuner),
                ("--torch_dtype", job.torch_dtype),
                ("--freeze_vit", job.freeze_vit),
                ("--freeze_aligner", job.freeze_aligner),
                ("--freeze_llm", job.freeze_llm),
                ("--lora_rank", lora.rank if lora else None),
                ("--lora_alpha", lora.alpha if lora else None),
                ("--target_modules", lora.target_modules if lora else None),
                ("--num_train_epochs", job.epochs),
                ("--max_steps", job.max_steps),
                ("--per_device_train_batch_size", job.per_device_batch),
                ("--gradient_accumulation_steps", accumulation),
                ("--learning_rate", job.learning_rate),
                ("--lr_scheduler_type", job.lr_scheduler),
                ("--warmup_ratio", job.warmup_ratio),
                ("--optim", job.optim),
                ("--adam_beta1", job.adam_beta1),
                ("--adam_beta2", job.adam_beta2),
                ("--adam_epsilon", job.adam_epsilon),
                ("--weight_decay", job.weight_decay),
                ("--max_grad_norm", job.max_grad_norm),
                ("--seed", job.seed),
                ("--data_seed", job.data_seed),
                ("--max_length", job.max_length),
                ("--packing", job.packing),
                ("--packing_length", job.packing_length),
                ("--packing_num_proc", job.packing_num_proc),
                ("--padding_free", job.padding_free),
                ("--attn_impl", job.attn_impl),
                ("--deepspeed", job.deepspeed),
                ("--gradient_checkpointing", job.gradient_checkpointing),
                ("--vit_gradient_checkpointing", job.vit_gradient_checkpointing),
                ("--gradient_checkpointing_kwargs", job.gradient_checkpointing_kwargs),
                ("--dataset_num_proc", job.dataset_num_proc),
                ("--dataloader_num_workers", job.dataloader_num_workers),
                ("--load_from_cache_file", job.load_from_cache_file),
                ("--dataset_shuffle", job.dataset_shuffle),
                ("--train_dataloader_shuffle", job.train_dataloader_shuffle),
                ("--eval_strategy", job.eval_strategy),
                ("--eval_steps", job.eval_steps),
                ("--save_strategy", job.save_strategy),
                ("--save_steps", job.save_steps),
                ("--save_total_limit", job.save_total_limit),
                ("--save_only_model", job.save_only_model),
                ("--load_best_model_at_end", best),
                ("--metric_for_best_model", "eval_loss" if best else None),
                ("--greater_is_better", False if best else None),
                ("--per_device_eval_batch_size", job.per_device_eval_batch_size),
                ("--logging_steps", job.logging_steps),
                ("--ddp_timeout", job.ddp_timeout),
                ("--output_dir", paths.output.resolve()),
                ("--add_version", False),
                ("--use_hf", True),
                ("--report_to", job.report_to),
                ("--project", paths.trackio_project if trackio else None),
                ("--run_name", spec.name if trackio else None),
                ("--external_plugins", cer_plugin(paths) if job.cer_sets else None),
                ("--callbacks", "cer" if job.cer_sets else None),
            ]
        ),
        *dataset_args(data),
    ]


def cer_plugin(paths: RunPaths) -> Path:
    """The ms-swift plugin registering the ``cer`` callback, in the checkout the tool environments sit in."""
    return (paths.envs.parent / "src" / "diffu_htr_train" / "evaluate" / "swift_plugin.py").resolve()


def sft_environ(job: SwiftSFT, spec: RunSpec) -> dict[str, str]:
    """A run's environment: its template's settings, ``NPROC_PER_NODE`` when ms-swift starts a process a GPU, and the
    sets the ``cer`` callback reads (their lines' paths absolute)."""
    gpus = spec.resources.gpus_per_node
    cer = {
        name: cer_set.model_copy(update={"lines": cer_set.lines.resolve()}).model_dump(mode="json")
        for name, cer_set in job.cer_sets.items()
    }
    return {
        **job.env,
        **({"NPROC_PER_NODE": str(gpus)} if gpus > 1 or spec.resources.nodes > 1 else {}),
        **({CER_SETS_ENV: json.dumps(cer), AT_START_ENV: str(job.cer_at_start).lower()} if cer else {}),
    }


SWIFT_ENVS: dict[Kind, ToolEnv] = {  # the environment each ms-swift job runs in
    Kind.MINICPM_OCR: ToolEnv.MINICPM,
    Kind.JEV: ToolEnv.MINICPM,
    Kind.UNLIMITED_OCR_2: ToolEnv.UOCR2,
}


def _swift_sft(model: list[Flag], *, job: SwiftJob, spec: RunSpec, paths: RunPaths) -> Command:
    """A ``swift sft`` command: the model and its own flags, then every run's (:func:`swift_sft_argv`). It resumes
    from its output folder when run again, unless its checkpoints hold the model only (no optimizer state to resume
    from)."""
    env = SWIFT_ENVS[job.kind]
    argv = [paths.tool(env, "swift"), "sft", *flags(model), *swift_sft_argv(job, spec, paths)]
    resume = None if job.save_only_model else paths.output.resolve()
    return Command(env=env, argv=argv, environ=sft_environ(job, spec), resume_from=resume)


def render_minicpm_ocr(job: MiniCPMOCRJob, spec: RunSpec, paths: RunPaths) -> list[Command]:
    """MiniCPM-OCR's full fine-tune (``minicpm_ocr``): the empty thinking prefix added to each answer and, with
    ``ignore_empty_think``, left out of the loss, as OpenBMB's recipe trains it."""
    own: list[Flag] = [
        ("--model_type", "minicpm_ocr"),
        ("--template", "minicpm_ocr"),
        ("--add_non_thinking_prefix", job.add_non_thinking_prefix),
        ("--loss_scale", job.loss_scale),
    ]
    return [_swift_sft([*model_flags(job.model), *own], job=job, spec=spec, paths=paths)]


def render_unlimited_ocr2(job: UnlimitedOCR2Job, spec: RunSpec, paths: RunPaths) -> list[Command]:
    """Unlimited-OCR 2 (``unlimited-ocr``, named: its architecture is shared with v1, which ms-swift would pick)."""
    own: list[Flag] = [("--model_type", "unlimited-ocr")]
    return [_swift_sft([*model_flags(job.model), *own], job=job, spec=spec, paths=paths)]


def render_jev(job: JevJob, spec: RunSpec, paths: RunPaths) -> list[Command]:
    """Jev as a two-class sequence classifier, as OpenBMB fine-tunes it: model type and template ``qwen3_5`` both named
    (its architecture maps to three model types, and its repo matches none of ms-swift's ``qwen3_5`` model groups,
    whose templates are ``qwen3_5`` and ``qwen3_8``, so neither is picked by itself)."""
    own: list[Flag] = [
        ("--model_type", "qwen3_5"),
        ("--template", "qwen3_5"),
        ("--task_type", "seq_cls"),
        ("--num_labels", 2),
        ("--problem_type", "single_label_classification"),
    ]
    return [_swift_sft([*model_flags(job.model), *own], job=job, spec=spec, paths=paths)]


def _target(job: DSparkJob) -> str:
    """A draft's target as vLLM serves it and speculators loads it: its folder's absolute path."""
    return str(Path(job.target.repo).expanduser().resolve())


def _served_url(job: DSparkJob) -> str:
    """Where a draft's target is served."""
    return f"http://localhost:{job.port}"


def _prepared(paths: RunPaths) -> Path:
    """Where speculators writes a draft's prepared responses, and its training reads them."""
    return (paths.root / "data").resolve()


def vllm_serve(job: DSparkJob, paths: RunPaths, *, extra: list[Flag], switches: tuple[str, ...] = ()) -> Command:
    """vLLM serving a draft's target, in the background: by its path (``vllm serve`` takes it first), then
    ``switches``, under that name (the name its responses record), reading local images below the job's
    ``media_root`` only, with ``extra`` flags and the job's own."""
    target = _target(job)
    argv = [
        paths.tool(ToolEnv.VLLM, "vllm"),
        "serve",
        target,
        *switches,
        *flags(
            [
                ("--served-model-name", target),
                ("--port", job.port),
                ("--gpu-memory-utilization", job.vllm_memory),
                ("--allowed-local-media-path", job.media_root.resolve()),
                *extra,
            ]
        ),
        *job.serve_args,
    ]
    return Command(env=ToolEnv.VLLM, argv=argv, role="serve", ready=f"{_served_url(job)}/health")


def render_responses_server(job: DSparkJob, paths: RunPaths) -> Command:
    """The server ``htr-train dspark-set`` asks for a draft's responses: vLLM serving the target as the draft's run
    serves it, without the hidden-state extraction (greedy answers need none)."""
    return vllm_serve(job, paths, extra=[])


def _dspark_serve(job: DSparkJob, paths: RunPaths) -> Command:
    """vLLM serving a draft's target with the hidden states of ``target_layer_ids`` and of its last layer (as
    speculators' launcher asks for it) extracted into the run's folder for speculators; its render endpoints on
    (``--enable-scale-out``, which speculators' ``prepare-data`` renders each conversation through, and which answer
    403 without it; speculators' own launcher always adds it)."""
    layers = [*job.target_layer_ids, job.target_layers]
    speculative = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": layers}},
    }
    transfer = {
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {"shared_storage_path": str((paths.root / "hidden_states").resolve())},
    }
    extra: list[Flag] = [
        ("--max-model-len", job.seq_length + 1),
        ("--speculative_config", speculative),
        ("--kv_transfer_config", transfer),
    ]
    return vllm_serve(job, paths, extra=extra, switches=("--enable-scale-out",))


def _dspark_prepare(job: DSparkJob, paths: RunPaths) -> Command:
    """speculators preparing a draft's responses as the served target renders them: counting their tokens, for the
    draft's reduced vocabulary (speculators otherwise skips it), over any earlier preparation (which it otherwise keeps
    silently)."""
    argv = [
        paths.tool(ToolEnv.SPECULATORS, "speculators"),
        "prepare-data",
        *flags(
            [
                ("--model", _target(job)),
                ("--data", job.responses.resolve()),
                ("--output", _prepared(paths)),
                ("--seq-length", job.seq_length),
                ("--render-endpoint", _served_url(job)),
                ("--max-samples", job.max_samples),
            ]
        ),
        "--no-skip-token-freq",
        "--overwrite",
    ]
    return Command(env=ToolEnv.SPECULATORS, argv=argv)


def _dspark_train(job: DSparkJob, paths: RunPaths, *, name: str) -> Command:
    """speculators training a draft on one GPU against the live server, logged to trackio as ``name``, its best
    checkpoint kept: ``speculators train`` itself, as speculators' docs run one GPU (under ``torchrun`` it also claims
    an accelerator before anything else, which a CPU check cannot give it)."""
    argv = [
        paths.tool(ToolEnv.SPECULATORS, "speculators"),
        "train",
        *flags(
            [
                ("--verifier-name-or-path", _target(job)),
                ("--data-path", _prepared(paths)),
                ("--vllm-endpoint", f"{_served_url(job)}/v1"),
                ("--on-missing", "generate"),
                ("--save-path", paths.output.resolve()),
                ("--speculator-type", "dspark"),
                ("--num-layers", job.num_layers),
                ("--block-size", job.block_size),
            ]
        ),
        "--target-layer-ids",
        *(str(layer) for layer in job.target_layer_ids),
        *flags(
            [
                ("--draft-vocab-size", job.draft_vocab_size),
                ("--mask-token-id", job.mask_token_id),
                ("--markov-rank", job.markov_rank),
                ("--markov-head-type", job.markov_head_type),
            ]
        ),
        "--enable-confidence-head",
        "--confidence-head-with-markov",
        *flags(
            [
                ("--confidence-head-alpha", job.confidence_head_alpha),
                ("--max-anchors", job.max_anchors),
                ("--loss-fn", job.loss_fn),
                ("--lr", job.lr),
                ("--scheduler-type", job.scheduler),
                ("--scheduler-warmup-ratio", job.warmup_ratio),
                ("--epochs", job.epochs),
                ("--total-seq-len", job.seq_length),
                ("--num-workers", job.num_workers),
                ("--logger", "trackio"),
                ("--run-name", name),
            ]
        ),
        "--save-best",
    ]
    return Command(env=ToolEnv.SPECULATORS, argv=argv)


def render_dspark(job: DSparkJob, spec: RunSpec, paths: RunPaths) -> list[Command]:
    """A DSpark draft trained online: vLLM serves the target and gives speculators the rendered conversations and the
    hidden states (:func:`_dspark_serve`); speculators prepares the responses (:func:`_dspark_prepare`) and trains the
    draft against the live server (:func:`_dspark_train`)."""
    return [_dspark_serve(job, paths), _dspark_prepare(job, paths), _dspark_train(job, paths, name=spec.name)]


def render(spec: RunSpec, paths: RunPaths) -> list[Command]:
    """The processes ``spec`` starts, in order, by its job's renderer."""
    job = spec.job
    match job:
        case MiniCPMOCRJob():
            return render_minicpm_ocr(job, spec, paths)
        case UnlimitedOCR2Job():
            return render_unlimited_ocr2(job, spec, paths)
        case JevJob():
            return render_jev(job, spec, paths)
        case DSparkJob():
            return render_dspark(job, spec, paths)
        case _:
            assert_never(job)
