"""Nemotron 3.5 Parse's run renders to our trainer under torchrun with NVIDIA's own fine-tuning recipe (NeMo AutoModel's
``tutorials/nemotron-parse/nemotron_parse_config.yaml`` at ``58487ab5``), flag for flag, less our choices; it is refused
when its model is another, or the pinned one at another revision, or when it is to run on more than one node, draw from
its sets, evaluate without a dev set or keep a best checkpoint it does not save; the loss weighs the model's layout
tokens (its coordinates and classes, named) ten times, as NVIDIA's does, and no other; and the dmlpai01 specs load."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import torch
import torch.nn.functional as F
from pydantic import BaseModel, ValidationError

from diffu_htr_train.jobs import RunSpec, ToolEnv, load_run_spec
from diffu_htr_train.nemotron_parse import REPO, REVISION
from diffu_htr_train.nemotron_parse.loss import LayoutWeightedLoss, layout_mask
from diffu_htr_train.nemotron_parse.options import TrainerOptions
from diffu_htr_train.render import RunPaths, render

if TYPE_CHECKING:
    from conftest import MakeSet

MODEL = {"repo": REPO, "revision": REVISION}
LION_CONFIGS = Path(__file__).parents[1] / "lion_data_prep" / "configs"

# NVIDIA's tutorial (step_scheduler, rng, optimizer, loss_fn, freeze_config, dataloader), as our job's fields: global
# batch 8 of 1 a device, 50 steps, a checkpoint and an evaluation every 25, seed 42, AdamW at 5e-5 with weight decay
# 0.01 and betas (0.9, 0.999), bfloat16, coordinate tokens weighted 10, no dataloader workers. Its freeze_vision_tower
# freezes the modules named vision, visual or image_encoder (AutoModel's model_utils.py), which Nemotron-Parse has none
# of, so it trains the whole model: nothing frozen here either.
TUTORIAL = {
    "kind": "nemotron-parse",
    "model": MODEL,
    "freeze_vision_encoder": False,
    "freeze_adapter": False,
    "coordinate_weight": 10.0,
    "bf16": True,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "max_steps": 50,
    "global_batch": 8,
    "per_device_batch": 1,
    "seed": 42,
    "dataloader_num_workers": 0,
    "eval_strategy": "steps",
    "eval_steps": 25,
    "save_strategy": "steps",
    "save_steps": 25,
    "report_to": "none",
}


def _page_set(make_set: MakeSet, name: str, volume: str) -> Path:
    """A set of one page (a line set's shape is enough for what a run's checks read)."""
    return make_set(name, [(f"{name}_p", "Anno 1715", "GT-A", volume, "train")])


def _spec(make_set: MakeSet, job: dict[str, Any], **resources: Any) -> dict[str, Any]:
    return {
        "name": "np",
        "job": job,
        "data": {
            "train": [{"path": str(_page_set(make_set, "train", "V1"))}],
            "dev": str(_page_set(make_set, "dev", "V2")),
        },
        "resources": {"need_gb": 40, "hours": 1, **resources},
    }


def _flags(argv: list[str]) -> dict[str, list[str]]:
    """An argv's flags, each with the words after it."""
    parsed: dict[str, list[str]] = {}
    flag = ""
    for word in argv:
        if word.startswith("--"):
            flag = word
            parsed[flag] = []
        else:
            parsed[flag].append(word)
    return parsed


def test_render_gives_nvidias_tutorial_recipe_to_our_trainer_under_torchrun(tmp_path: Path, make_set: MakeSet) -> None:
    spec = RunSpec.model_validate(_spec(make_set, TUTORIAL))
    paths = RunPaths(root=tmp_path / "runs" / "np", envs=tmp_path / "envs", trackio_project="diffu")

    [command] = render(spec, paths)

    env = tmp_path / "envs" / "nemotron_parse" / ".venv" / "bin"
    assert command.env is ToolEnv.NEMOTRON_PARSE
    assert command.argv[:6] == [str(env / "torchrun"), "--standalone", "--nproc-per-node", "1"] + [
        str(env / "htr-nemotron-parse"),
        "--options",
    ]
    assert TrainerOptions.model_validate_json(command.argv[6]) == TrainerOptions(
        model=REPO,
        revision=REVISION,
        train=[(tmp_path / "sets" / "train.jsonl").resolve()],
        dev=(tmp_path / "sets" / "dev.jsonl").resolve(),
        freeze_vision_encoder=False,
        freeze_adapter=False,
        coordinate_weight=10.0,
    )
    assert _flags(command.argv[7:]) == {
        "--output_dir": [str(paths.output.resolve())],
        "--bf16": ["true"],
        "--max_steps": ["50"],
        "--per_device_train_batch_size": ["1"],
        "--gradient_accumulation_steps": ["8"],
        "--learning_rate": ["5e-05"],
        "--adam_beta1": ["0.9"],
        "--adam_beta2": ["0.999"],
        "--weight_decay": ["0.01"],
        "--seed": ["42"],
        "--dataloader_num_workers": ["0"],
        "--eval_strategy": ["steps"],
        "--eval_steps": ["25"],
        "--save_strategy": ["steps"],
        "--save_steps": ["25"],
        "--report_to": ["none"],
    }
    # a stopped run is continued by running its steps again (the Trainer's own --resume_from_checkpoint)
    assert command.resume_from == paths.output.resolve()


def test_render_runs_a_process_a_gpu_logs_to_trackio_and_reads_a_warmup_ratio_as_transformers_5_does(
    tmp_path: Path, make_set: MakeSet
) -> None:
    job = {**TUTORIAL, "report_to": "trackio", "warmup_ratio": 0.05, "global_batch": 16, "per_device_batch": 4}
    spec = RunSpec.model_validate(_spec(make_set, job, gpus_per_node=2))

    [command] = render(spec, RunPaths(root=tmp_path / "r", envs=tmp_path / "envs", trackio_project="diffu"))

    given = _flags(command.argv[7:])
    assert command.argv[2:4] == ["--nproc-per-node", "2"]
    assert {flag: given[flag] for flag in ("--gradient_accumulation_steps", "--per_device_train_batch_size")} == {
        "--gradient_accumulation_steps": ["2"],
        "--per_device_train_batch_size": ["4"],
    }
    # warmup_ratio is deprecated in transformers 5.14; its warmup_steps reads a fraction below 1 as the ratio
    assert given["--warmup_steps"] == ["0.05"]
    assert {flag: given[flag] for flag in ("--report_to", "--project", "--run_name")} == {
        "--report_to": ["trackio"],
        "--project": ["diffu"],
        "--run_name": ["np"],
    }


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        pytest.param(
            {"job": {"model": {"repo": REPO, "revision": "0" * 40}}},
            "only nvidia/Nemotron-3.5-Parse@72ad4d71dd827135ee581502363e14748008900a is trusted",
            id="another revision of the model",
        ),
        pytest.param(
            {"job": {"model": {"repo": "nvidia/NVIDIA-Nemotron-Parse-v1.1", "revision": "1" * 40}}},
            "only nvidia/Nemotron-3.5-Parse",
            id="another model's code",
        ),
        pytest.param({"resources": {"nodes": 2}}, "trains on one node", id="two nodes"),
        pytest.param({"samples": 1}, "reads its training sets whole", id="a draw of a set"),
        pytest.param({"dev": None}, "needs a dev set", id="evaluating without a dev set"),
        pytest.param(
            {"job": {"best_by_eval_loss": True, "save_strategy": "no"}},
            "best_by_eval_loss needs save_strategy equal to an eval_strategy",
            id="a best checkpoint never saved",
        ),
        pytest.param({"job": {"epochs": 2}}, "epochs or as max_steps, one of them", id="two lengths"),
        pytest.param({"job": {"freeze_vision_encoder": None}}, "freeze_vision_encoder", id="freezing left unsaid"),
    ],
)
def test_a_nemotron_parse_run_is_refused_when_its_spec_does_not_go_together(
    make_set: MakeSet, change: dict[str, Any], refusal: str
) -> None:
    spec = _spec(make_set, dict(TUTORIAL))
    spec["job"].update(change.get("job", {}))
    spec["resources"].update(change.get("resources", {}))
    if "samples" in change:
        spec["data"]["train"][0]["samples"] = change["samples"]
    if "dev" in change:
        spec["data"]["dev"] = change["dev"]
    spec["job"] = {key: value for key, value in spec["job"].items() if value is not None}

    with pytest.raises(ValidationError, match=refusal):
        RunSpec.model_validate(spec)


def test_the_dmlpai01_specs_load_and_train_on_the_nemotron_parse_sets_dev_split() -> None:
    for name in ("smoke_nemotron_parse_2gpu.yaml", "nemotron_parse_all.yaml"):
        spec = load_run_spec(LION_CONFIGS / name)

        assert (spec.job.kind, spec.resources.gpus_per_node) == ("nemotron-parse", 2)
        assert spec.data is not None
        assert [str(part.path) for part in spec.data.train] == ["data/htr/nemotron_parse_pages.train.jsonl"]
        assert str(spec.data.dev) == "data/htr/nemotron_parse_pages.dev.jsonl"


class _Vocabulary(BaseModel):
    """A tokenizer's vocabulary, the part of it the loss reads."""

    tokens: dict[str, int]

    def get_vocab(self) -> dict[str, int]:
        return dict(self.tokens)


# A vocabulary laid out as 3.5's: text, then the coordinates and classes with an added token among them, and one of the
# punctuation tokens added after them (" -" is 72262 in 3.5's)
TOKENIZER = _Vocabulary(
    tokens={"<s>": 0, "</s>": 2, "a": 5, "<x_0.5>": 7, "<y_0.25>": 8, "<class_Text>": 9, "\\<|unk|\\>": 10, " -": 11}
)


def test_the_layout_tokens_are_the_coordinates_and_classes_by_name_not_every_id_from_a_threshold() -> None:
    assert layout_mask(TOKENIZER, 12).nonzero().flatten().tolist() == [7, 8, 9]


def test_the_loss_weighs_layout_tokens_ten_times_skips_the_prompt_and_divides_by_the_tokens_learnt() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 12)
    # each row: a prompt token (never learnt), then the answer; the second row padded at its end
    labels = torch.tensor([[-100, 5, 7, 9], [-100, 11, 2, -100]])
    loss = LayoutWeightedLoss(layout_mask(TOKENIZER, 12), 10.0)

    each = F.cross_entropy(logits[:, :-1].transpose(1, 2), labels[:, 1:], ignore_index=-100, reduction="none")
    weighted = each[0, 0] + 10 * each[0, 1] + 10 * each[0, 2] + each[1, 0] + each[1, 1]
    outputs = SimpleNamespace(logits=logits)
    assert torch.allclose(loss(outputs, labels), weighted / 5)
    # over a step of several batches, on every device, the Trainer's count of the step's tokens divides instead
    assert torch.allclose(loss(outputs, labels, num_items_in_batch=torch.tensor(20)), weighted / 20)
