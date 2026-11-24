"""CPU tests of our RLVR task: the reward, the selection rule, the task contract and the configuration.

    cd runs/htr/rlvr && CUDA_VISIBLE_DEVICES= PYTHONPATH=/mnt/scratch/gabriel/src/Nemotron/src:. \\
        /mnt/scratch/gabriel/envs/parse_rlvr/bin/uv run --no-project --with pytest \\
        --python /mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python pytest tests -q

The tests on real pages (our reader's saved dev readings, the dataset bundle) are skipped where those are missing.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import htr_evaluate
import htr_select
import pytest
from htr_reward import boxes, components, transcription
from htr_settings import settings

REPO = Path(__file__).resolve().parents[4]
DATA = Path("/mnt/scratch/gabriel/htr/rlvr/data")
READS = REPO / "runs/htr/nemotron-parse-selftrain-finetune2/dev_reads"
NEMOTRON = Path("/mnt/scratch/gabriel/src/Nemotron/src/nemotron/recipes/parse")

TRUTH = (
    "<x_0.10><y_0.10>Anno 1642 den 3 Martii\nhölts laga ting<x_0.90><y_0.20><class_Section-header>\n\n"
    "<x_0.10><y_0.25>Samma dag kom för rätten Erich Olofsson\noch kärade till Per Larsson om en ko"
    "<x_0.90><y_0.60><class_Text>"
)


def rows(split: str) -> dict[str, dict]:
    return {row["id"]: row for row in map(json.loads, (DATA / f"{split}.jsonl").read_text().splitlines())}


# ---- reward ---------------------------------------------------------------------------------------------------------


def test_truth_scores_one():
    parts = components(TRUTH, TRUTH, ended=True)
    assert parts["reward"] == 1.0 and parts["cer"] == 0.0 and parts["boxes"] == 1.0 and parts["gate"] == 1.0


def test_boxes_are_read_from_blocks():
    assert boxes(TRUTH) == [[0.10, 0.10, 0.90, 0.20], [0.10, 0.25, 0.90, 0.60]]
    assert boxes("no blocks here") == []
    assert "tabular" not in transcription("\\begin{tabular}{ll}a & b\\\\c & d\\end{tabular}")


def test_text_and_boxes_are_separate_parts():
    wrong_text = TRUTH.replace("Erich Olofsson", "Erik Olsson")
    parts = components(wrong_text, TRUTH)
    assert parts["boxes"] == 1.0 and 0 < parts["text"] < 1.0
    assert parts["reward"] == pytest.approx(0.8 * parts["text"] + 0.2)
    moved = TRUTH.replace("<x_0.10><y_0.25>", "<x_0.10><y_0.55>")  # the second block's box no longer overlaps enough
    parts = components(moved, TRUTH)
    assert (
        parts["text"] == 1.0 and parts["boxes"] < 1.0 and parts["reward"] == pytest.approx(0.8 + 0.2 * parts["boxes"])
    )
    plain = components("Anno 1642 den 3 Martii hölts laga ting", TRUTH)
    assert plain["boxes"] == 0.0 and plain["predicted_boxes"] == 0.0


def test_weights_are_options():
    wrong = TRUTH.replace("<x_0.10><y_0.25>", "<x_0.10><y_0.55>")
    assert components(wrong, TRUTH, weights={"text": 1.0, "boxes": 0.0})["reward"] == 1.0
    assert components(wrong, TRUTH, weights={"text": 0.0, "boxes": 1.0})["reward"] < 1.0


def test_gate_zeroes_loops_caps_and_missing_ends():
    looping = TRUTH + "\n\n<x_0.1><y_0.7>" + "Moni " * 10
    assert components(looping, TRUTH)["looped"] == 1.0
    assert components(looping, TRUTH)["reward"] == 0.0
    assert components(looping, TRUTH, loop={"ngram_size": 24, "min_repeats": 4, "token_run_limit": 32})["looped"] == 0.0
    assert components(TRUTH, TRUTH, capped=True)["reward"] == 0.0
    assert components(TRUTH, TRUTH, ended=False)["reward"] == 0.0
    assert components(TRUTH, TRUTH, token_ids=[5, 6, 2], eos_token_id=2)["ended"] == 1.0
    assert components(TRUTH, TRUTH, token_ids=[5, 6, 7], eos_token_id=2)["reward"] == 0.0
    assert components(TRUTH, TRUTH)["ended"] == -1.0 and components(TRUTH, TRUTH)["reward"] == 1.0  # end not known


def test_repeats_the_truth_has_are_not_a_loop():
    dots = "<x_0.1><y_0.1>Summa " + ". " * 40 + "13<x_0.9><y_0.2><class_Text>"
    assert components(dots, dots)["looped"] == 0.0


@pytest.mark.skipif(not (READS / "step-3000.jsonl").exists() or not DATA.exists(), reason="needs our dev readings")
def test_text_part_is_our_dev_cer():
    monitor = rows("monitor")
    reads = [json.loads(line) for line in (READS / "step-3000.jsonl").read_text().splitlines()]
    assert [read["id"] for read in reads] == list(monitor)  # the monitor split is the dev draw, in its order
    edits = chars = 0.0
    for read in reads:
        parts = components(read["reading"], monitor[read["id"]]["target"])
        assert parts["text"] == pytest.approx(max(0.0, 1 - read["edits"] / max(1, read["chars"])), abs=1e-9)
        edits, chars = edits + parts["cer"] * parts["chars"], chars + parts["chars"]
    assert edits / chars == pytest.approx(sum(r["edits"] for r in reads) / sum(r["chars"] for r in reads), abs=1e-9)


@pytest.mark.skipif(not (READS / "step-250.jsonl").exists() or not DATA.exists(), reason="needs our dev readings")
def test_known_loop_is_gated():
    reads = {r["id"]: r for r in map(json.loads, (READS / "step-250.jsonl").read_text().splitlines())}
    reading = reads["483ab0a6287c"]["reading"]
    assert reading.endswith("Moni Moni Moni")
    parts = components(reading, rows("monitor")["483ab0a6287c"]["target"])
    assert parts["looped"] == 1.0 and parts["gate"] == 0.0 and parts["reward"] == 0.0 and parts["text"] > 0


# ---- selection ------------------------------------------------------------------------------------------------------


def answer(reward: float, looped: bool = False, capped: bool = False) -> dict:
    return {"reward": reward, "looped": float(looped), "capped": float(capped)}


def pool(count: int = 100) -> tuple[list[dict], list[dict], list[dict]]:
    """Candidates: 10 failing (5 mixed, 5 all failing), 40 with a spread, 50 flat; and the fixed splits' rows."""
    rng = random.Random(7)
    candidates = [{"id": f"p{i}", "context_id": f"v{i % 12}", "split": "train"} for i in range(count)]
    groups = []
    for i, row in enumerate(candidates):
        if i < 5:
            sampled = [answer(0.9), answer(0.0, looped=True), answer(0.8), answer(0.9)]
        elif i < 10:
            sampled = [answer(0.0, capped=True)] * 4
        elif i < 50:
            sampled = [answer(0.5 + rng.uniform(-0.01, 0.01) * (i - 9)) for _ in range(4)]
        else:
            sampled = [answer(0.95)] * 4
        groups.append({"id": row["id"], "answers": sampled})
    excluded = [dict(row) for row in candidates] + [
        {"id": "m1", "context_id": "monitor-volume", "split": "monitor"},
        {"id": "t1", "context_id": "test-volume", "split": "test"},
    ]
    return candidates, groups, excluded


def test_selection_quotas():
    candidates, groups, excluded = pool()
    selected = htr_select.select_training_pages(candidates, groups, pages=40, seed=1, excluded=excluded)
    ids = [row["id"] for row in selected]
    index = {row["id"]: i for i, row in enumerate(candidates)}
    picked = sorted(index[i] for i in ids)
    assert len(ids) == len(set(ids)) == 40
    assert set(range(10)) <= set(picked)  # a quarter of 40 = all ten failing pages
    informed = [i for i in picked if 10 <= i < 50]
    assert len(informed) >= 20  # 30 by a rule (10 failing + 20 by spread); random picks may add more
    by_spread = sorted(range(10, 50), key=lambda i: -htr_select.spread(groups[i]["answers"]))
    assert set(by_spread[:20]) <= set(picked)
    assert sum(i >= 50 for i in picked) >= 1  # the random quarter reaches the flat pages


def test_mixed_failures_come_before_uniform_ones():
    candidates, groups, excluded = pool()
    selected = htr_select.select_training_pages(candidates, groups, pages=20, seed=1, excluded=excluded)
    assert {f"p{i}" for i in range(5)} <= {row["id"] for row in selected}  # 5 failing places: the mixed pages


def test_selection_is_seeded_and_fills_without_failures():
    candidates, groups, excluded = pool()
    first = htr_select.select_training_pages(candidates, groups, pages=40, seed=3, excluded=excluded)
    again = htr_select.select_training_pages(candidates, groups, pages=40, seed=3, excluded=excluded)
    other = htr_select.select_training_pages(candidates, groups, pages=40, seed=4, excluded=excluded)
    assert first == again and [r["id"] for r in first] != [r["id"] for r in other]
    for group in groups[:10]:
        group["answers"] = [answer(0.95)] * 4  # no failing page: the spread rule takes their places
    selected = htr_select.select_training_pages(candidates, groups, pages=40, seed=3, excluded=excluded)
    assert sum(htr_select.spread(g["answers"]) > 0 for g in groups if g["id"] in {r["id"] for r in selected}) >= 30
    report = htr_select.selection_report(candidates, groups, selected)
    assert report["selected"] == 40 and report["selected_with_a_failed_answer"] == 0


def test_selection_refuses_held_out_volumes_and_mismatches():
    candidates, groups, excluded = pool()
    select = htr_select.select_training_pages
    with pytest.raises(ValueError, match="outside the monitor and test"):
        select(candidates, groups, pages=10, seed=1, excluded=[{"id": "x", "context_id": "v3", "split": "test"}])
    with pytest.raises(ValueError, match="monitor or test pages"):
        select(candidates, groups, pages=10, seed=1, excluded=[{"id": "p3", "context_id": "other", "split": "monitor"}])
    with pytest.raises(ValueError, match="train split"):
        select([dict(candidates[0], split="test"), *candidates[1:]], groups, pages=10, seed=1, excluded=excluded)
    with pytest.raises(ValueError, match="IDs differ"):
        select(candidates, groups[::-1], pages=10, seed=1, excluded=excluded)
    with pytest.raises(ValueError, match="repeated"):
        select([candidates[0]] * 2, [groups[0]] * 2, pages=1, seed=1, excluded=excluded)
    with pytest.raises(ValueError, match="enough candidates"):
        select(candidates, groups, pages=101, seed=1, excluded=excluded)


# ---- task contract, environment, configuration ----------------------------------------------------------------------


def test_task_loads_for_both_stages():
    from nemotron.recipes.parse.training.task import load_task

    for stage in ("sft", "rlvr"):
        task = load_task("htr_task.HtrTask", stage=stage)
        assert task.environment_class == "htr_environment.HtrEnvironment" and task.reward_metric == "reward"
    assert (
        task.prompt_file.read_text() == "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
    )
    assert task.prompt_file.read_text() == (NEMOTRON / "dental/prompt.txt").read_text()


@pytest.mark.skipif(not DATA.exists(), reason="needs the dataset bundle")
def test_candidates_selection_and_manifest_roundtrip(tmp_path):
    from nemotron.recipes.parse.training.data import load_pages
    from nemotron.recipes.parse.training.task import load_task

    task = load_task("htr_task.HtrTask", stage="rlvr")
    curriculum = tmp_path / "curriculum"
    curriculum.mkdir()
    candidates = task.create_candidates(curriculum, seed=31415, contexts=32)
    assert len({row["id"] for row in candidates}) == 32
    assert all(Path(row["image"]).is_absolute() and Path(row["image"]).is_file() for row in candidates)
    assert all(Path(row["image"]).is_relative_to(curriculum) for row in candidates)
    again = task.create_candidates(tmp_path / "again", seed=31415, contexts=32)
    assert [row["id"] for row in again] == [row["id"] for row in candidates]
    rng = random.Random(0)
    groups = [
        {"id": row["id"], "answers": [answer(rng.random(), looped=rng.random() < 0.1) for _ in range(4)]}
        for row in candidates
    ]
    fixed = load_pages(DATA / "train.jsonl") + load_pages(DATA / "monitor.jsonl") + load_pages(DATA / "test.jsonl")
    selected = task.select_candidates(candidates, groups, pages=8, seed=42, excluded=fixed)
    assert len(selected) == 8
    for row in selected:  # as stage2_rlvr/train.py writes rl_train.jsonl
        row["image"] = str(Path(row["image"]).relative_to(curriculum))
    (curriculum / "rl_train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected))
    assert all(Path(row["image"]).is_file() and row["target"] for row in load_pages(curriculum / "rl_train.jsonl"))
    with pytest.raises(ValueError):  # a monitor page offered as a candidate is refused
        monitor = load_pages(DATA / "monitor.jsonl")[:1]
        task.select_candidates(
            monitor, [{"id": monitor[0]["id"], "answers": [answer(0.5)]}], pages=1, seed=1, excluded=fixed
        )


def test_environment_scores_and_logs(tmp_path):
    from htr_environment import LOGGED, HtrEnvironment

    cfg = {
        "metrics_file": str(tmp_path / "train-components.jsonl"),
        "phase": "train",
        "weights": {"text": 0.8, "boxes": 0.2},
    }
    environment = HtrEnvironment.__ray_metadata__.modified_class(cfg)
    row = {"id": "a", "target": TRUTH}
    logs = [
        [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": TRUTH}],
        [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": TRUTH + " Moni" * 12}],
    ]
    result = environment.step(logs, [row, row], return_extracted_answer=True)
    assert result.rewards.tolist() == [1.0, 0.0] and result.terminateds.tolist() == [1.0, 1.0]
    assert result.answers[0] == TRUTH and len(result.observations) == 2
    record = json.loads((tmp_path / "train-components.jsonl").read_text())
    assert record["phase"] == "train" and [set(s) for s in record["samples"]] == [set(LOGGED)] * 2
    assert [s["looped"] for s in record["samples"]] == [0.0, 1.0]


def test_summary_reports_sampled_and_greedy():
    def reading(text: str, **how) -> dict:
        return dict(answer=text, tokens=10, **components(text, TRUTH, **how))

    groups = [
        {
            "id": "a",
            "answers": [reading(TRUTH, ended=True), reading(TRUTH + " Moni" * 12, capped=True, ended=False)],
            "greedy": {"1": reading(TRUTH + " Moni" * 12, capped=True, ended=False), "1.1": reading(TRUTH, ended=True)},
        },
        {
            "id": "b",
            "answers": [reading(TRUTH, ended=True)] * 2,
            "greedy": {"1": reading(TRUTH, ended=True), "1.1": reading(TRUTH, ended=True)},
        },
    ]
    summary = htr_evaluate.summarize(groups)
    assert summary["pages"] == 2 and summary["answers"] == 4 and summary["samples"] == 2
    assert summary["sampled"]["looped"] == 0.25 and summary["sampled"]["ended"] == 0.75
    assert summary["pages_with_a_looped_or_capped_answer"] == 1
    assert summary["greedy"]["1"]["looped"] == 0.5 and summary["greedy"]["1.1"]["looped"] == 0.0
    assert summary["greedy"]["1.1"]["cer_pooled"] == 0.0 and summary["greedy"]["1"]["capped"] == 0.5
    json.dumps(summary)
    with pytest.raises(ValueError):
        htr_evaluate.summarize([dict(groups[0], answers=groups[0]["answers"][:1]), groups[1]])


def test_protocol_follows_the_training_yaml():
    values = settings()
    protocol = htr_evaluate.evaluation_protocol(4, values, [])
    assert protocol["generation"]["temperature"] == values["temperature"] and protocol["generation"]["n"] == 4
    assert protocol["generation"]["repetition_penalty"] == 1.0 and protocol["greedy"] == []
    protocol = htr_evaluate.evaluation_protocol(8, values, values["greedy_repetition_penalties"])
    assert [g["repetition_penalty"] for g in protocol["greedy"]] == [1.0, 1.1]
    assert all(g["temperature"] == 0.0 for g in protocol["greedy"])
    assert protocol["engine"]["max_model_len"] == values["max_model_len"]


def test_training_configuration_resolves(tmp_path):
    from nemo_rl.algorithms.grpo import MasterConfig
    from nemotron.recipes.parse.training.config import rlvr_config

    here = Path(__file__).resolve().parents[1]
    config = rlvr_config(
        here / "grpo_htr.yaml", tmp_path / "model", tmp_path / "curriculum", tmp_path, prompt_file=here / "prompt.txt"
    )
    MasterConfig(**config)
    grpo, policy, generation = config["grpo"], config["policy"], config["policy"]["generation"]
    # the recipe's algorithm, untouched
    assert (grpo["max_num_steps"], grpo["num_prompts_per_step"], grpo["num_generations_per_prompt"]) == (128, 16, 4)
    assert policy["optimizer"]["kwargs"]["lr"] == 1.0e-6 and policy["train_global_batch_size"] == 64
    assert config["loss_fn"]["reference_policy_kl_penalty"] == 0.1 and config["loss_fn"]["ratio_clip_max"] == 0.2
    assert policy["hf_config_overrides"] == {"parse_train_scope": "decoder"}
    # ours: truncated answers stay in the loss with reward 0
    assert grpo["overlong_filtering"] is False
    assert grpo["reward_shaping"]["enabled"] is True and grpo["reward_shaping"]["stop_properly_penalty_coef"] == 0.0
    # ours: room for the longest page, an answer and the observation
    assert 3329 + 6 + generation["max_new_tokens"] + 32 <= policy["max_total_sequence_length"]
    assert generation["vllm_cfg"]["max_model_len"] == policy["max_total_sequence_length"]
    assert generation["temperature"] == generation["val_temperature"] == 1.0
    for name in ("parse-page", "parse-page-validation"):
        assert config["env"][name]["weights"] == {"text": 0.8, "boxes": 0.2}
        assert config["env"][name]["loop"]["ngram_size"] == 6
    assert set(config["env"]) == {"parse-page", "parse-page-validation"}
    assert config["data"]["train"]["data_path"] == str(tmp_path / "curriculum" / "rl_train.jsonl")


def test_temperature_is_one_option_for_trainer_and_hooks(tmp_path, monkeypatch):
    from nemo_rl.algorithms.grpo import MasterConfig
    from nemotron.recipes.parse.training.config import rlvr_config

    here = Path(__file__).resolve().parents[1]
    cooler = tmp_path / "grpo_htr_t06.yaml"
    cooler.write_text(f"defaults: {here / 'grpo_htr.yaml'}\npolicy:\n  generation:\n    temperature: 0.6\n")
    config = rlvr_config(cooler, tmp_path / "m", tmp_path / "c", tmp_path, prompt_file=here / "prompt.txt")
    MasterConfig(**config)
    assert config["policy"]["generation"]["temperature"] == config["policy"]["generation"]["val_temperature"] == 0.6
    assert config["grpo"]["overlong_filtering"] is False and config["policy"]["max_total_sequence_length"] == 5504
    monkeypatch.setenv("HTR_RLVR_CONFIG", str(cooler))
    settings.cache_clear()
    try:
        assert settings()["temperature"] == 0.6 and settings()["weights"] == {"text": 0.8, "boxes": 0.2}
    finally:
        monkeypatch.delenv("HTR_RLVR_CONFIG")
        settings.cache_clear()
