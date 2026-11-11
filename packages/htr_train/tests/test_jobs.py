"""A run is refused before anything starts when its spec, data, model or resources do not go together: a spec that is
not YAML, an unpinned Hub model, a local model with a revision or one that does not exist, remote code for any model but
Unlimited-OCR 2 at its revision, a batch that does not divide, packing without flash attention, a test page or a dev
volume in training, a missing image or meta line, pairs of another renderer, a draft's responses from another target,
a draft's target from the Hub or its layers outside it, a LightOnOCR-3 run reading a page otherwise than LightOnOCR-3
reads it (sized by a spec's env, packed, read for CER, or larger than its client sends), and a Spark run with more
than its one GPU or memory; every shipped spec loads."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from pydantic import JsonValue

from diffu_htr_train.jev.prompt import prompt_digest
from diffu_htr_train.jobs import JobRefused, ModelRef, RunSpec, Where, load_run_spec
from diffu_htr_train.launch import plan_run
from diffu_htr_train.settings import Settings

if TYPE_CHECKING:
    from conftest import MakeSet

CONFIGS = Path(__file__).parents[1] / "configs"
MINICPM = {"repo": "openbmb/MiniCPM-OCR-1.0", "revision": "2697d4e93dca82784abf858ae57206a49629cb6b"}

type Change = Callable[[dict[str, Any], Path], None]  # a change to a valid spec, given its sets' folder


def _spec(train: Path, dev: Path) -> dict[str, Any]:
    """A valid MiniCPM-OCR run on ``train`` and ``dev``."""
    return {
        "name": "r",
        "job": {
            "kind": "minicpm-ocr",
            "model": dict(MINICPM),
            "tuner": "full",
            "freeze_vit": False,
            "freeze_aligner": True,
            "learning_rate": 5e-6,
            "max_steps": 2,
            "global_batch": 4,
            "max_length": 4096,
        },
        "data": {"train": [{"path": str(train)}], "dev": str(dev)},
        "resources": {"need_gb": 40, "hours": 1},
    }


def _set_job(**fields: JsonValue) -> Change:
    def change(spec: dict[str, Any], _: Path) -> None:
        spec["job"].update(fields)

    return change


def _drop_job(field: str) -> Change:
    def change(spec: dict[str, Any], _: Path) -> None:
        del spec["job"][field]

    return change


def _set_resources(**fields: JsonValue) -> Change:
    def change(spec: dict[str, Any], _: Path) -> None:
        spec["resources"].update(fields)

    return change


def _edit_meta(name: str, edit: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]) -> Change:
    def change(_: dict[str, Any], sets: Path) -> None:
        path = sets / f"{name}.meta.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        path.write_text("".join(json.dumps(row) + "\n" for row in edit(rows)), encoding="utf-8")

    return change


def _first(**fields: JsonValue) -> Callable[[list[dict[str, Any]]], list[dict[str, Any]]]:
    return lambda rows: [{**rows[0], **fields}, *rows[1:]]


def _remove_image(_: dict[str, Any], sets: Path) -> None:
    (sets / "train_lines" / "a_0_0.jpg").unlink()


def _model_not_there(spec: dict[str, Any], sets: Path) -> None:
    spec["job"]["model"] = {"repo": str(sets / "no_model")}


def _sampled_in_order(spec: dict[str, Any], _: Path) -> None:
    spec["job"]["dataset_shuffle"] = False
    spec["data"]["train"][0]["samples"] = 1


def _uocr2_logged_to_trackio(spec: dict[str, Any], _: Path) -> None:
    model = {"repo": "baidu/Unlimited-OCR-2", "revision": "0480966938291445591f69f08a13b041c073a96a"}
    spec["job"].update(kind="unlimited-ocr-2", model=model, report_to="trackio")


def _uocr2_elsewhere(spec: dict[str, Any], _: Path) -> None:
    spec["job"].update(
        kind="unlimited-ocr-2",
        model={"repo": "someone/Unlimited-OCR-2", "revision": "a" * 40},
        report_to="tensorboard",
    )


@pytest.mark.parametrize(
    ("change", "where", "refusal"),
    [
        pytest.param(
            _set_job(model={"repo": "openbmb/MiniCPM-OCR-1.0", "revision": "main"}),
            Where.SPARK,
            "pin it to a full 40-character commit, not 'main'",
            id="hub-model-unpinned",
        ),
        pytest.param(
            _set_job(trust_remote_code=True),
            Where.SPARK,
            "job.minicpm-ocr.trust_remote_code: Extra inputs are not permitted",
            id="remote-code-asked-for",
        ),
        pytest.param(
            _uocr2_elsewhere,
            Where.SPARK,
            "Unlimited-OCR 2 runs its own code: only baidu/Unlimited-OCR-2@0480966938291445591f69f08a13b041c073a96a",
            id="remote-code-of-another-revision",
        ),
        pytest.param(
            _drop_job("freeze_vit"), Where.SPARK, "job.minicpm-ocr.freeze_vit: Field required", id="freeze-vit-unstated"
        ),
        pytest.param(
            _set_job(epochs=1),
            Where.SPARK,
            "give the run's length as epochs or as max_steps, one of them",
            id="length-twice",
        ),
        pytest.param(_set_job(tuner="lora"), Where.SPARK, "a lora tuner needs lora settings", id="lora-without-shape"),
        pytest.param(
            _set_job(eval_strategy="steps", eval_steps=0, save_strategy="steps", save_steps=10),
            Where.SPARK,
            "job.minicpm-ocr.eval_steps: Input should be greater than 0",
            id="zero-steps",
        ),
        pytest.param(
            _set_job(model={"repo": "./model", "revision": "a" * 40}),
            Where.SPARK,
            "./model is a local folder: it has no revision",
            id="local-model-with-a-revision",
        ),
        pytest.param(_model_not_there, Where.SPARK, "no_model is no model folder", id="local-model-missing"),
        pytest.param(
            _set_job(model={"repo": "runs/htr/lines/output"}),
            Where.SPARK,
            "runs/htr/lines/output is a Hub repo \\(a local folder is written as a path",
            id="local-checkpoint-not-written-as-a-path",
        ),
        pytest.param(
            _sampled_in_order,
            Where.SPARK,
            "draws samples \\(#N\\) but dataset_shuffle is false",
            id="samples-drawn-in-order",
        ),
        pytest.param(
            _uocr2_logged_to_trackio,
            Where.SPARK,
            "transformers 4.52.1, whose TrainingArguments has no trackio project",
            id="uocr2-to-trackio",
        ),
        pytest.param(
            _set_job(packing=True, attn_impl="sdpa"),
            Where.SPARK,
            "ms-swift packs sequences only with attn_impl flash_attn",
            id="packing-without-flash-attn",
        ),
        pytest.param(
            _set_job(
                best_by_eval_loss=True, eval_strategy="steps", eval_steps=10, save_strategy="steps", save_steps=15
            ),
            Where.SPARK,
            "best_by_eval_loss needs save_steps a multiple of eval_steps",
            id="best-saved-off-its-evaluations",
        ),
        pytest.param(
            _set_job(
                best_by_eval_loss=True,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_only_model=True,
                deepspeed="zero3",
            ),
            Where.SPARK,
            "cannot load a best checkpoint saved with save_only_model under DeepSpeed",
            id="best-of-model-only-checkpoints-under-deepspeed",
        ),
        pytest.param(
            _set_resources(nodes=1, gpus_per_node=3),
            Where.SLURM,
            "global_batch 4 is no multiple of 3 GPUs x 1 a device",
            id="batch-does-not-divide",
        ),
        pytest.param(
            _edit_meta("train", _first(split="test")),
            Where.SPARK,
            "holds a_0_0, a sample of a GT test page: never train on one",
            id="test-page-in-training",
        ),
        pytest.param(
            _edit_meta("train", _first(split=None)),
            Where.SPARK,
            "holds a_0_0, a real sample of a set not split by the GT: never train on one",
            id="real-page-of-an-unsplit-set",
        ),
        pytest.param(
            _edit_meta("dev", _first(volume="")),
            Where.SPARK,
            "b_0_0 names no volume, so no check can keep it apart from training",
            id="dev-without-volume",
        ),
        pytest.param(
            _edit_meta("dev", _first(volume="V1")),
            Where.SPARK,
            "shares the volume GT-A/V1 with the training sets: cut the dev set by volume",
            id="dev-volume-in-training",
        ),
        pytest.param(_remove_image, Where.SPARK, "names an image that does not exist", id="image-missing"),
        pytest.param(
            _edit_meta("train", lambda rows: rows[:1]),
            Where.SPARK,
            "has 1 lines for 2 samples in train.jsonl",
            id="meta-short",
        ),
        pytest.param(
            _set_resources(nodes=2),
            Where.SPARK,
            "r asks for 2 GPUs: the Spark has one; launch it on slurm",
            id="spark-with-two-gpus",
        ),
        pytest.param(
            _set_resources(need_gb=110),
            Where.SPARK,
            "r needs 110 GB and the guard keeps 15 GB free: the Spark has 119 GB",
            id="spark-memory",
        ),
    ],
)
def test_plan_run_refuses_a_run_that_does_not_go_together(
    tmp_path: Path,
    make_set: MakeSet,
    settings: Settings,
    change: Change,
    where: Where,
    refusal: str,
) -> None:
    train = make_set("train", [("a_0_0", "Anno", "GT-A", "V1", "train"), ("a_0_1", "1715", "GT-A", "V1", "train")])
    dev = make_set("dev", [("b_0_0", "den 4 maj", "GT-A", "V2", "train")])
    spec = _spec(train, dev)
    change(spec, train.parent)
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")

    with pytest.raises(JobRefused, match=refusal):
        plan_run(load_run_spec(path), where, settings)


def test_load_run_spec_refuses_a_file_that_is_not_yaml(tmp_path: Path) -> None:
    path = tmp_path / "spec.yaml"
    path.write_text("job: [kind: jev\n", encoding="utf-8")

    with pytest.raises(JobRefused, match="spec.yaml is not a YAML run spec"):
        load_run_spec(path)


def load_run_spec_from(folder: Path, spec: dict[str, Any]) -> RunSpec:
    """``spec`` through the YAML file a user writes."""
    path = folder / "spec.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return load_run_spec(path)


def _pairs_by(renderer: str, prompt: str, make_set: MakeSet) -> Path:
    pairs = make_set("pairs", [("a_0_0_0", "Anno", "GT-A", "V1", "train")])
    meta = pairs.with_name("pairs.meta.jsonl")
    row = json.loads(meta.read_text(encoding="utf-8"))
    meta.write_text(json.dumps({**row, "renderer": renderer, "prompt": prompt}) + "\n", encoding="utf-8")
    return pairs


@pytest.mark.parametrize(
    ("renderer", "prompt", "refusal"),
    [
        pytest.param(
            "htr-train-jev-render-1+111111111111",
            prompt_digest(),
            "rendered by htr-train-jev-render-1\\+111111111111, the job trains on",
            id="another-renderer",
        ),
        pytest.param(
            "htr-train-jev-render-1+222222222222",
            "0123456789ab",
            "was built in the prompt 0123456789ab, Jev trains in",
            id="another-prompt",
        ),
    ],
)
def test_plan_run_refuses_jev_pairs_of_another_renderer_or_prompt(
    tmp_path: Path, make_set: MakeSet, settings: Settings, renderer: str, prompt: str, refusal: str
) -> None:
    pairs = _pairs_by(renderer, prompt, make_set)
    model = tmp_path / "jev"
    model.mkdir()
    spec = load_run_spec_from(
        tmp_path,
        {
            "name": "jev",
            "job": {
                **_spec(pairs, pairs)["job"],
                "kind": "jev",
                "model": {"repo": str(model)},
                "freeze_vit": True,
                "renderer": "htr-train-jev-render-1+222222222222",
            },
            "data": {"train": [{"path": str(pairs)}]},
            "resources": {"need_gb": 40, "hours": 1},
        },
    )

    with pytest.raises(JobRefused, match=refusal):
        plan_run(spec, Where.SPARK, settings)


LIGHTONOCR_3 = {"repo": "lightonai/LightOnOCR-3-4B", "revision": "b71010f095dc3735de7ee6b9969bff8f8f50b39a"}


@pytest.mark.parametrize(
    ("fields", "page", "refusal"),
    [
        pytest.param(
            {"env": {"IMAGE_MAX_TOKEN_NUM": "1024"}},
            (1182, 2048),
            "env may not set IMAGE_MAX_TOKEN_NUM: LightOnOCR-3's pages are sized as its processor sizes them",
            id="image-tokens-in-env",
        ),
        pytest.param(
            {"env": {"MAX_PIXELS": "4194304"}}, (1182, 2048), "env may not set MAX_PIXELS", id="pixels-in-env"
        ),
        pytest.param(
            {"packing": True, "attn_impl": "flash_attn"},
            (1182, 2048),
            "carries a packed sample's state into the next in envs/minicpm: train LightOnOCR-3 unpacked",
            id="packed",
        ),
        pytest.param({"padding_free": True}, (1182, 2048), "train LightOnOCR-3 unpacked", id="padding-free"),
        pytest.param(
            {"cer_sets": {"dev": {"lines": "dev_lines.jsonl"}}, "eval_strategy": "steps", "eval_steps": 1},
            (1182, 2048),
            "cer_sets read lines in MiniCPM-OCR's prompt: a LightOnOCR-3 run is evaluated by its loss",
            id="cer-sets",
        ),
        pytest.param(
            {},
            (1500, 2600),
            "train.jsonl:1 shows .*a_0_0.jpg at a longest edge of 2600 px: LightOnOCR-3 reads pages fitted to 2048 px",
            id="page-larger-than-its-client-sends",
        ),
    ],
)
def test_plan_run_refuses_a_lightonocr_3_run_that_would_read_a_page_otherwise_than_lightonocr_3_reads_it(
    tmp_path: Path,
    make_set: MakeSet,
    settings: Settings,
    fields: dict[str, JsonValue],
    page: tuple[int, int],
    refusal: str,
) -> None:
    train = make_set("train", [("a_0_0", "Anno 1715", "GT-A", "V1", "train")], size=page)
    dev = make_set("dev", [("b_0_0", "den 4 maj", "GT-A", "V2", "train")], size=(1182, 2048))
    spec = _spec(train, dev)
    spec["job"].update(kind="lightonocr-3", model=dict(LIGHTONOCR_3), **fields)

    with pytest.raises(JobRefused, match=refusal):
        plan_run(load_run_spec_from(tmp_path, spec), Where.SPARK, settings)


def test_plan_run_plans_a_lightonocr_3_run_on_pages_fitted_as_its_client_fits_them_with_its_processors_bounds(
    tmp_path: Path, make_set: MakeSet, settings: Settings
) -> None:
    train = make_set("train", [("a_0_0", "Anno 1715", "GT-A", "V1", "train")], size=(2048, 1446))
    dev = make_set("dev", [("b_0_0", "den 4 maj", "GT-A", "V2", "train")], size=(1182, 2048))
    spec = _spec(train, dev)
    spec["job"].update(kind="lightonocr-3", model=dict(LIGHTONOCR_3))

    plan = plan_run(load_run_spec_from(tmp_path, spec), Where.SPARK, settings)

    [command] = plan.commands
    assert command.environ == {"IMAGE_MIN_TOKEN_NUM": "64", "IMAGE_MAX_TOKEN_NUM": "16384"}


@pytest.mark.parametrize(
    ("answered_by", "samples", "job_fields", "gpus", "refusal"),
    [
        pytest.param("other", 1, {}, 1, "answered by {other}, the draft is for {target}", id="another-target"),
        pytest.param("target", 0, {}, 1, "holds 0 answers and 3 failed prompts", id="no-answers"),
        pytest.param(
            "target", 2, {}, 1, "holds 2 answers and 3 failed prompts \\(at most 50% may fail\\)", id="too-many-failed"
        ),
        pytest.param(
            "target",
            5,
            {},
            3,
            "a DSpark run serves its target on the first of its GPUs and trains on the last, on one node: one GPU",
            id="three-gpus",
        ),
        pytest.param(
            "target",
            5,
            {"checkpoint_freq": 1.5},
            1,
            "checkpoint_freq 1.5: a fraction of an epoch \\(at most 1\\) or a whole number of epochs",
            id="checkpoints-every-epoch-and-a-half",
        ),
        pytest.param(
            "target",
            1,
            {"serve_args": ["--trust-remote-code"]},
            1,
            "serve_args may not hold --trust-remote-code: MiniCPM-OCR is native in vLLM",
            id="remote-code",
        ),
        pytest.param(
            "target",
            1,
            {"target_layer_ids": [4, 24]},
            1,
            "target_layer_ids \\[4, 24\\] must be distinct layers of the target's 24",
            id="layer-past-the-target",
        ),
        pytest.param(
            "target",
            1,
            {"target": {"repo": "openbmb/MiniCPM-OCR-1.0", "revision": "2697d4e93dca82784abf858ae57206a49629cb6b"}},
            1,
            "the target openbmb/MiniCPM-OCR-1.0 must be a local folder",
            id="target-from-the-hub",
        ),
    ],
)
def test_plan_run_refuses_a_draft_not_on_its_own_targets_answers_or_on_more_than_two_gpus(
    tmp_path: Path,
    settings: Settings,
    answered_by: str,
    samples: int,
    job_fields: dict[str, JsonValue],
    gpus: int,
    refusal: str,
) -> None:
    folders = {name: (tmp_path / name).resolve() for name in ("target", "other")}
    folders["target"].mkdir()
    responses = tmp_path / "responses.jsonl"
    responses.write_text("{}\n", encoding="utf-8")
    record = {"target": str(folders[answered_by]), "samples": samples, "failed": 3, "max_failed": 0.5}
    record["max_tokens"] = 512
    (tmp_path / "responses.target.json").write_text(json.dumps(record), encoding="utf-8")
    job = {
        "kind": "dspark",
        "target": {"repo": str(folders["target"])},
        "responses": str(responses),
        "media_root": str(tmp_path),
        **job_fields,
    }
    resources = {"need_gb": 60, "hours": 1, "gpus_per_node": gpus}

    with pytest.raises(JobRefused, match=refusal.format(**folders)):
        plan_run(
            load_run_spec_from(tmp_path, {"name": "draft", "job": job, "resources": resources}), Where.SPARK, settings
        )


def test_model_ref_reads_a_hub_id_as_a_hub_repo_whatever_folders_the_current_one_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "openbmb" / "MiniCPM-OCR-1.0").mkdir(parents=True)  # a folder named as the Hub id

    model = ModelRef.model_validate(MINICPM)

    assert not model.is_local


def test_every_shipped_run_spec_loads() -> None:
    """Kept although it reads the repository's files: a spec field renamed or tightened breaks the shipped specs, and
    nothing else loads them before a run is launched."""
    specs = sorted([*CONFIGS.glob("*/*.yaml"), *(CONFIGS.parent / "lion_data_prep" / "configs").glob("*.yaml")])

    loaded = [load_run_spec(path) for path in specs]

    assert len(loaded) == len(specs) > 0
