"""A run spec renders to the commands the tools' own authors publish: OpenBMB's MiniCPM-OCR recipe for handwritten
lines (``docs/swift_minicpm_ocr_french_en.md``) and its Jev fine-tune (``docs/finetune_jev_en.md``), ms-swift's
Unlimited-OCR 2 example (``examples/models/unlimited_ocr2/train.sh`` at ``ef4953a4``) and speculators' DSpark example
(``examples/train/dspark_qwen3_0_6b_sharegpt_online.sh``), flag for flag, less the choices we make on purpose, each
named in the case (:data:`OURS` and each case's ``ours``); LightOnOCR-3, which no one publishes a fine-tune of, renders
OpenBMB's recipe and ms-swift's Qwen3.5 example (``examples/models/qwen3_5/transformers.sh``) but for its own flags and
image bounds; a run left to log to trackio logs to the local project under its name."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import JsonValue

from diffu_htr_train.evaluate.in_training import CerSet, at_start, cer_sets
from diffu_htr_train.jobs import MiniCPMOCRJob, RunSpec, ToolEnv
from diffu_htr_train.render import RunPaths, render

if TYPE_CHECKING:
    from conftest import MakeSet

# ms-swift flags every run passes (diffu_htr_train.render's note): the Hub is Hugging Face's, no split by sample,
# checkpoints straight into the run's output
OURS = {"--use_hf": ["true"], "--add_version": ["false"], "--split_dataset_ratio": ["0"]}


def _flags(argv: list[str]) -> dict[str, list[float | str]]:
    """An argv's flags, each with the values after it, values as numbers where they are numbers (so ``0.0`` and ``0``,
    ``5e-06`` and ``5.0e-6`` are one value)."""

    def number(word: str) -> float | str:
        try:
            return float(word)
        except ValueError:
            return word

    parsed: dict[str, list[float | str]] = {}
    flag = ""
    for word in argv:
        if word.startswith("--"):
            flag = word
            parsed[flag] = []
        else:
            parsed[flag].append(number(word))
    return parsed


def _published(text: str) -> list[str]:
    """A published command's words after ``swift sft``, as bash splits them."""
    return shlex.split(text.replace("\\\n", " "))


FRENCH = """
  --model "$MODEL_PATH" --model_type minicpm_ocr --template minicpm_ocr --add_non_thinking_prefix true
  --dataset "$TRAIN_DATA" --split_dataset_ratio 0.0 --deepspeed "$DOC_DIR/assets/minicpm-ocr/ds_zero2.json"
  --tuner_type full --torch_dtype bfloat16 --freeze_vit false --freeze_aligner true --freeze_llm false --packing false
  --padding_free false --max_length 4096 --num_train_epochs 3.0 --per_device_train_batch_size 1
  --gradient_accumulation_steps 16 --gradient_checkpointing true --vit_gradient_checkpointing true
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' --learning_rate 5e-06 --lr_scheduler_type cosine
  --warmup_ratio 0.05 --optim adamw_torch_fused --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-08
  --weight_decay 0.1 --max_grad_norm 1.0 --logging_steps 1 --save_total_limit 3 --save_only_model false
  --load_from_cache_file false --dataset_num_proc 8 --dataloader_num_workers 8 --attn_impl sdpa
  --loss_scale ignore_empty_think --eval_strategy epoch --output_dir "$OUTPUT_DIR" --report_to none --seed 42
  --data_seed 42 --add_version false --use_hf true --dataset_shuffle false --train_dataloader_shuffle true
  --val_dataset "$VAL_DATA" --save_strategy epoch --load_best_model_at_end true --metric_for_best_model eval_loss
  --greater_is_better false --per_device_eval_batch_size 1
"""
FRENCH_SPEC = {
    "kind": "minicpm-ocr",
    "tuner": "full",
    "freeze_vit": False,
    "freeze_aligner": True,
    "freeze_llm": False,
    "learning_rate": 5e-6,
    "epochs": 3,
    "global_batch": 128,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.05,
    "optim": "adamw_torch_fused",
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_epsilon": 1e-8,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "seed": 42,
    "data_seed": 42,
    "max_length": 4096,
    "packing": False,
    "padding_free": False,
    "attn_impl": "sdpa",
    "deepspeed": "zero2",
    "gradient_checkpointing": True,
    "vit_gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "dataset_num_proc": 8,
    "dataloader_num_workers": 8,
    "load_from_cache_file": False,
    "dataset_shuffle": False,
    "train_dataloader_shuffle": True,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 3,
    "save_only_model": False,
    "best_by_eval_loss": True,
    "per_device_eval_batch_size": 1,
    "logging_steps": 1,
    "report_to": "none",
    "env": {"DOWNSAMPLE_MODE": "4x"},
}

JEV = """
  --model "${MODEL_PATH}" --dataset "${TRAIN_DATA}" --task_type seq_cls --num_labels 2
  --problem_type single_label_classification --tuner_type full --torch_dtype bfloat16 --freeze_vit true
  --freeze_aligner true --freeze_llm false --num_train_epochs 3 --per_device_train_batch_size 1
  --gradient_accumulation_steps 1 --learning_rate 1e-5 --lr_scheduler_type cosine --warmup_ratio 0.05
  --weight_decay 0.1 --max_grad_norm 1.0 --max_length 12000 --packing true --packing_length 12000
  --packing_num_proc 8 --deepspeed zero3 --gradient_checkpointing true --attn_impl flash_attn --logging_steps 10
  --eval_strategy no --save_strategy steps --save_steps 500 --save_total_limit 40 --save_only_model true
  --dataset_num_proc 48 --dataloader_num_workers 4 --ddp_timeout 10800 --seed 42 --output_dir "${OUTPUT_DIR}"
  --report_to tensorboard
"""
JEV_SPEC = {
    "kind": "jev",
    "renderer": "htr-train-jev-render-1+000000000000",
    "tuner": "full",
    "freeze_vit": True,
    "freeze_aligner": True,
    "freeze_llm": False,
    "learning_rate": 1e-5,
    "epochs": 3,
    "global_batch": 8,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "seed": 42,
    "max_length": 12000,
    "packing": True,
    "packing_length": 12000,
    "packing_num_proc": 8,
    "attn_impl": "flash_attn",
    "deepspeed": "zero3",
    "gradient_checkpointing": True,
    "dataset_num_proc": 48,
    "dataloader_num_workers": 4,
    "eval_strategy": "no",
    "save_strategy": "steps",
    "save_steps": 500,
    "save_total_limit": 40,
    "save_only_model": True,
    "logging_steps": 10,
    "ddp_timeout": 10800,
    "report_to": "tensorboard",
    "env": {"IMAGE_MAX_TOKEN_NUM": "2048", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
}

UOCR2 = """
  --model PaddlePaddle/Unlimited-OCR-2 --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#1000'
  --load_from_cache_file true --split_dataset_ratio 0.01 --tuner_type lora --torch_dtype bfloat16 --num_train_epochs 1
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --learning_rate 1e-4 --lora_rank 8 --lora_alpha 32
  --target_modules all-linear --freeze_vit true --freeze_aligner true --gradient_accumulation_steps 16 --eval_steps 50
  --save_steps 50 --save_total_limit 2 --logging_steps 5 --max_length 4096 --output_dir output --warmup_ratio 0.05
  --dataset_num_proc 4 --dataloader_num_workers 4
"""
UOCR2_SPEC = {
    "kind": "unlimited-ocr-2",
    "tuner": "lora",
    "lora": {"rank": 8, "alpha": 32, "target_modules": "all-linear"},
    "freeze_vit": True,
    "freeze_aligner": True,
    "learning_rate": 1e-4,
    "epochs": 1,
    "global_batch": 16,
    "warmup_ratio": 0.05,
    "max_length": 4096,
    "load_from_cache_file": True,
    "dataset_num_proc": 4,
    "dataloader_num_workers": 4,
    "eval_steps": 50,
    "save_steps": 50,
    "save_total_limit": 2,
    "per_device_eval_batch_size": 1,
    "logging_steps": 5,
    "report_to": "tensorboard",
}


def _paths(tmp_path: Path) -> RunPaths:
    return RunPaths(root=tmp_path / "runs" / "r", envs=tmp_path / "envs", trackio_project="diffu")


@pytest.mark.parametrize(
    ("published", "job", "gpus", "dev", "environ", "ours"),
    [
        pytest.param(
            FRENCH,
            FRENCH_SPEC,
            8,
            True,
            {"DOWNSAMPLE_MODE": "4x", "NPROC_PER_NODE": "8"},
            # ms-swift's own zero2 config is OpenBMB's ds_zero2.json key for key (both read at their commits)
            {"--deepspeed": ["zero2"]},
            id="minicpm-ocr-openbmb-french-lines",
        ),
        pytest.param(
            JEV,
            JEV_SPEC,
            8,
            False,
            {
                "IMAGE_MAX_TOKEN_NUM": "2048",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "NPROC_PER_NODE": "8",
            },
            # its architecture maps to three ms-swift model types, and its repo to no qwen3_5 model group (whose
            # templates are qwen3_5 and qwen3_8): both named
            {"--model_type": ["qwen3_5"], "--template": ["qwen3_5"], **OURS},
            id="jev-openbmb-seq-cls",
        ),
        pytest.param(
            UOCR2,
            UOCR2_SPEC,
            1,
            True,
            # its sparse attention routed as its own inference routes it (render.UNLIMITED_OCR_2_ENV)
            {"DASH_STAGE0": "head_cls", "PAGE_SCOPE_PREFILL": "1"},
            # the Hugging Face repo at its pinned commit, not ModelScope's; named (its architecture is v1's too); our
            # set with its dev set, not a split by sample; ms-swift's default logger, named
            {
                "--model": ["baidu/Unlimited-OCR-2"],
                "--model_revision": ["0480966938291445591f69f08a13b041c073a96a"],
                "--model_type": ["unlimited-ocr"],
                "--report_to": ["tensorboard"],
                **OURS,
            },
            id="unlimited-ocr-2-ms-swift-lora-example",
        ),
    ],
)
def test_render_gives_the_published_swift_sft_command_but_for_our_choices(
    tmp_path: Path,
    make_set: MakeSet,
    published: str,
    job: dict[str, JsonValue],
    gpus: int,
    dev: bool,
    environ: dict[str, str],
    ours: dict[str, list[str]],
) -> None:
    model = (tmp_path / "model").resolve()
    model.mkdir()
    train = make_set("train", [("a_0_0", "Anno 1715", "GT-A", "V1", "train")]).resolve()
    val = make_set("dev", [("b_0_0", "den 4 maj", "GT-A", "V2", "train")]).resolve()
    paths = _paths(tmp_path)
    model_repo = {"repo": "baidu/Unlimited-OCR-2", "revision": "0480966938291445591f69f08a13b041c073a96a"}
    spec = RunSpec.model_validate(
        {
            "name": "r",
            "job": {**job, "model": model_repo if job["kind"] == "unlimited-ocr-2" else {"repo": str(model)}},
            "data": {"train": [{"path": str(train)}], "dev": str(val) if dev else None},
            "resources": {"gpus_per_node": gpus, "need_gb": 40, "hours": 1},
        }
    )

    [command] = render(spec, paths)

    words = (
        published.replace('"$MODEL_PATH"', str(model))
        .replace('"${MODEL_PATH}"', str(model))
        .replace('"$TRAIN_DATA"', str(train))
        .replace('"${TRAIN_DATA}"', str(train))
        .replace("'AI-ModelScope/LaTeX_OCR:human_handwrite#1000'", str(train))
        .replace('"$VAL_DATA"', str(val))
        .replace('"$OUTPUT_DIR"', str(paths.output.resolve()))
        .replace('"${OUTPUT_DIR}"', str(paths.output.resolve()))
        .replace("--output_dir output", f"--output_dir {paths.output.resolve()}")
    )
    expected = _flags(_published(words)) | _flags([word for flag, values in ours.items() for word in (flag, *values)])
    if dev and "--val_dataset" not in expected:
        expected["--val_dataset"] = [str(val)]
    env = ToolEnv.UOCR2 if job["kind"] == "unlimited-ocr-2" else ToolEnv.MINICPM
    assert command.argv[:2] == [str(tmp_path / "envs" / env / ".venv" / "bin" / "swift"), "sft"]
    assert _flags(command.argv[2:]) == expected
    assert command.environ == environ


# ms-swift's Qwen3.5 example (examples/models/qwen3_5/transformers.sh at ef4953a4), a LoRA of Qwen3.5-35B-A3B, less its
# mixture-of-experts flags (--experts_impl grouped_mm --router_aux_loss_coef 1e-3: LightOnOCR-3-4B is dense) and
# --group_by_length true (a sampler our specs do not choose); its environment: IMAGE_MAX_TOKEN_NUM=1024,
# VIDEO_MAX_TOKEN_NUM=128, FPS_MAX_FRAMES=12, NPROC_PER_NODE=4
QWEN35_LORA = """
  --model Qwen/Qwen3.5-35B-A3B --tuner_type lora --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#2000'
  --load_from_cache_file true --add_non_thinking_prefix true --split_dataset_ratio 0.01 --torch_dtype bfloat16
  --num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --learning_rate 1e-4
  --lora_rank 8 --lora_alpha 32 --target_modules all-linear --gradient_accumulation_steps 1
  --output_dir output/Qwen3.5-35B-A3B --eval_steps 50 --save_steps 50 --save_total_limit 2 --logging_steps 5
  --max_length 2048 --warmup_ratio 0.05 --dataset_num_proc 4 --dataloader_num_workers 4 --deepspeed zero3
"""
QWEN35_LORA_SPEC = {
    "kind": "lightonocr-3",
    "tuner": "lora",
    "lora": {"rank": 8, "alpha": 32, "target_modules": "all-linear"},
    "freeze_vit": True,
    "freeze_aligner": True,
    "learning_rate": 1e-4,
    "epochs": 1,
    "global_batch": 16,
    "per_device_batch": 4,
    "warmup_ratio": 0.05,
    "max_length": 2048,
    "load_from_cache_file": True,
    "dataset_num_proc": 4,
    "dataloader_num_workers": 4,
    "deepspeed": "zero3",
    "eval_steps": 50,
    "save_steps": 50,
    "save_total_limit": 2,
    "per_device_eval_batch_size": 4,
    "logging_steps": 5,
    "report_to": "tensorboard",
}
# LightOnOCR-3-4B's processor's bounds on an image, in ms-swift's qwen3_5 template's tokens of 32 x 32 px: 65,536 and
# 16,777,216 px (its preprocessor_config.json at b71010f0)
LIGHTON_IMAGE_TOKENS = {"IMAGE_MIN_TOKEN_NUM": "64", "IMAGE_MAX_TOKEN_NUM": "16384"}
# Every LightOnOCR-3 run: model type and template named (its architecture maps to several model types, its repo to no
# model group), and the empty thinking block its chat template writes with thinking off added to each answer and left
# out of the loss, so a sample is its own template's sequence and the loss starts where its generation prompt ends
LIGHTON_OWN = {"--model_type": ["qwen3_5"], "--template": ["qwen3_5"], "--add_non_thinking_prefix": ["true"]}
LIGHTON_OWN |= {"--loss_scale": ["ignore_empty_think"]}


@pytest.mark.parametrize(
    ("published", "job", "gpus", "environ", "ours"),
    [
        pytest.param(
            FRENCH,
            {key: given for key, given in FRENCH_SPEC.items() if key != "env"} | {"kind": "lightonocr-3"},
            8,
            {**LIGHTON_IMAGE_TOKENS, "NPROC_PER_NODE": "8"},
            # OpenBMB's full recipe for MiniCPM-OCR (its vision encoder trained, its projector frozen) with
            # LightOnOCR-3's own flags for MiniCPM-OCR's, and no DOWNSAMPLE_MODE (MiniCPM-OCR's slicing)
            {"--deepspeed": ["zero2"], **LIGHTON_OWN},
            id="full-as-openbmbs-minicpm-ocr-recipe",
        ),
        pytest.param(
            QWEN35_LORA,
            QWEN35_LORA_SPEC,
            4,
            {**LIGHTON_IMAGE_TOKENS, "NPROC_PER_NODE": "4"},
            # its own flags and image bounds (no video); its two freezes stated (ms-swift's defaults, which a spec
            # must state); our dev set, not a split by sample; ms-swift's default logger, named
            {
                **LIGHTON_OWN,
                "--freeze_vit": ["true"],
                "--freeze_aligner": ["true"],
                "--report_to": ["tensorboard"],
                **OURS,
            },
            id="lora-as-ms-swifts-qwen3-5-example",
        ),
    ],
)
def test_render_gives_lightonocr_3_the_published_recipe_but_for_its_own_flags_and_image_bounds(
    tmp_path: Path,
    make_set: MakeSet,
    published: str,
    job: dict[str, JsonValue],
    gpus: int,
    environ: dict[str, str],
    ours: dict[str, list[str]],
) -> None:
    model = (tmp_path / "model").resolve()
    model.mkdir()
    train = make_set("train", [("a_0_0", "Anno 1715", "GT-A", "V1", "train")]).resolve()
    val = make_set("dev", [("b_0_0", "den 4 maj", "GT-A", "V2", "train")]).resolve()
    paths = _paths(tmp_path)
    spec = RunSpec.model_validate(
        {
            "name": "r",
            "job": {**job, "model": {"repo": str(model)}},
            "data": {"train": [{"path": str(train)}], "dev": str(val)},
            "resources": {"gpus_per_node": gpus, "need_gb": 60, "hours": 1},
        }
    )

    [command] = render(spec, paths)

    words = (
        published.replace('"$MODEL_PATH"', str(model))
        .replace("--model Qwen/Qwen3.5-35B-A3B", f"--model {model}")
        .replace('"$TRAIN_DATA"', str(train))
        .replace("'AI-ModelScope/LaTeX_OCR:human_handwrite#2000'", str(train))
        .replace('"$VAL_DATA"', str(val))
        .replace('"$OUTPUT_DIR"', str(paths.output.resolve()))
        .replace("--output_dir output/Qwen3.5-35B-A3B", f"--output_dir {paths.output.resolve()}")
    )
    expected = _flags(_published(words)) | _flags([word for flag, values in ours.items() for word in (flag, *values)])
    expected.setdefault("--val_dataset", [str(val)])
    assert command.argv[:2] == [str(tmp_path / "envs" / ToolEnv.MINICPM / ".venv" / "bin" / "swift"), "sft"]
    assert _flags(command.argv[2:]) == expected
    assert command.environ == environ


def test_render_gives_speculators_dspark_example_for_our_target_served_by_vllm(tmp_path: Path) -> None:
    target = (tmp_path / "target").resolve()
    target.mkdir()
    responses = (tmp_path / "responses.jsonl").resolve()
    paths = _paths(tmp_path)
    spec = RunSpec.model_validate(
        {
            "name": "draft",
            "job": {
                "kind": "dspark",
                "target": {"repo": str(target)},
                "responses": str(responses),
                "media_root": str(tmp_path),
                "serve_args": ["--trust-request-mm-kwargs"],
            },
            "resources": {"need_gb": 60, "hours": 1, "gpus_per_node": 2, "cpus_per_task": 384},
        }
    )

    serve, prepare, train = render(spec, paths)

    url = "http://localhost:8000"
    tool = tmp_path / "envs"
    assert (serve.role, serve.ready, serve.env) == ("serve", f"{url}/health", ToolEnv.VLLM)
    # vLLM on the run's first GPU, the trainer on its last, as speculators' DSpark example runs them
    assert (serve.device, prepare.device, train.device) == ("first", None, "last")
    # vLLM's front end as speculators' launch_vllm.py starts it: on their 384-CPU H100 node, 18 API servers of 2
    # renderer threads for prepare-data's 72 workers, every native thread pool bounded
    assert serve.environ == {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "2",
    }
    assert serve.argv == [
        str(tool / "vllm" / ".venv" / "bin" / "vllm"),
        "serve",
        str(target),
        "--enable-scale-out",
        "--served-model-name",
        str(target),
        "--port",
        "8000",
        "--gpu-memory-utilization",
        "0.35",
        "--allowed-local-media-path",
        str(tmp_path.resolve()),
        "--max-model-len",
        "4097",
        "--speculative_config",
        '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": '
        '{"eagle_aux_hidden_state_layer_ids": [4, 8, 12, 16, 20, 24]}}}',
        "--kv_transfer_config",
        '{"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer", "kv_connector_extra_config": '
        f'{{"shared_storage_path": "{(paths.root / "hidden_states").resolve()}"}}}}',
        *("--api-server-count", "18", "--renderer-num-workers", "2"),
        "--trust-request-mm-kwargs",
    ]
    data = (paths.root / "data").resolve()
    assert prepare.argv == [
        str(tool / "speculators" / ".venv" / "bin" / "speculators"),
        "prepare-data",
        *("--model", str(target), "--data", str(responses), "--output", str(data)),
        *("--seq-length", "4096", "--render-endpoint", url, "--num-preprocessing-workers", "72"),
        # the token counts the reduced draft vocabulary is built from (speculators skips them by default), and never
        # an earlier preparation kept in its place
        *("--no-skip-token-freq", "--overwrite"),
    ]
    # the example's training flags, with OpenBMB's released draft's shape (5 layers, 15 tokens a block, its mask token
    # and layers) for the example's Qwen3-0.6B ones; the cosine schedule, 4 % warmup and 2 workers of speculators'
    # DSpark research runs; a checkpoint every tenth of an epoch, as their Kimi-K3 DSpark example saves (--save-best
    # would save no checkpoint mid-epoch); trackio for logging
    assert train.argv == [
        str(tool / "speculators" / ".venv" / "bin" / "speculators"),
        "train",
        *("--verifier-name-or-path", str(target), "--data-path", str(data), "--vllm-endpoint", f"{url}/v1"),
        *("--on-missing", "generate", "--save-path", str(paths.output.resolve()), "--speculator-type", "dspark"),
        *("--num-layers", "5", "--block-size", "15", "--target-layer-ids", "4", "8", "12", "16", "20"),
        *("--draft-vocab-size", "32000", "--mask-token-id", "248076", "--markov-rank", "256"),
        *("--markov-head-type", "vanilla", "--enable-confidence-head", "--confidence-head-with-markov"),
        *("--confidence-head-alpha", "1.0", "--max-anchors", "3072"),
        *("--loss-fn", '{"ce": 0.1, "tv": 0.9}', "--lr", "0.0003", "--scheduler-type", "cosine"),
        *("--scheduler-warmup-ratio", "0.04", "--epochs", "5", "--total-seq-len", "4096", "--num-workers", "2"),
        *("--logger", "trackio", "--run-name", "draft", "--checkpoint-freq", "0.1"),
    ]


def test_render_logs_a_run_left_to_trackio_to_the_local_project_under_its_name(
    tmp_path: Path, make_set: MakeSet
) -> None:
    train = make_set("train", [("a_0_0", "Anno 1715", "GT-A", "V1", "train")])
    job = {**FRENCH_SPEC, "model": {"repo": str(tmp_path)}}
    del job["report_to"]
    spec = RunSpec.model_validate(
        {
            "name": "lines",
            "job": job,
            "data": {"train": [{"path": str(train)}]},
            "resources": {"gpus_per_node": 8, "need_gb": 40, "hours": 1},
        }
    )

    [command] = render(spec, _paths(tmp_path))

    logged = _flags(command.argv[2:])
    assert {flag: logged[flag] for flag in ("--report_to", "--project", "--run_name")} == {
        "--report_to": ["trackio"],
        "--project": ["diffu"],
        "--run_name": ["lines"],
    }


def test_render_has_a_run_with_cer_sets_read_them_through_the_plugin_in_its_checkout(
    tmp_path: Path, make_set: MakeSet
) -> None:
    train = make_set("train", [("a_0_0", "Anno 1715", "GT-A", "V1", "train")])
    lines = tmp_path / "dev_lines.jsonl"
    lines.write_text("", encoding="utf-8")
    job = {**FRENCH_SPEC, "model": {"repo": str(tmp_path)}, "cer_sets": {"dev_lines": {"lines": str(lines)}}}
    spec = RunSpec.model_validate(
        {
            "name": "lines",
            "job": {**job, "cer_at_start": True},
            "data": {"train": [{"path": str(train)}]},
            "resources": {"gpus_per_node": 2, "need_gb": 40, "hours": 1},
        }
    )
    checkout = Path(__file__).resolve().parents[1]

    [command] = render(spec, RunPaths(root=tmp_path / "r", envs=checkout / "envs", trackio_project="diffu"))

    given = _flags(command.argv[2:])
    assert given["--callbacks"] == ["cer"]
    assert Path(str(given["--external_plugins"][0])).is_file()
    assert cer_sets(command.environ) == {"dev_lines": CerSet(lines=lines.resolve())}
    assert at_start(command.environ)


def test_a_run_reading_cer_sets_without_evaluating_is_refused(tmp_path: Path) -> None:
    job = {
        **FRENCH_SPEC,
        "model": {"repo": str(tmp_path)},
        "eval_strategy": "no",
        "best_by_eval_loss": False,
        "cer_sets": {"dev": {"lines": str(tmp_path / "l.jsonl")}},
    }

    with pytest.raises(ValueError, match="cer_sets are read at each evaluation"):
        MiniCPMOCRJob.model_validate(job)
