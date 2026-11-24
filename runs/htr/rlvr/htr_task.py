"""Our handwritten pages as a task of NVIDIA's Parse recipe (docs/nemotron/parse/adaptation.md, "Task contract").

Select it with ``task=htr_task.HtrTask`` and this folder on ``PYTHONPATH`` (``run_rlvr.sh`` does both).

* ``monitor`` (SFT): greedy readings of the monitor pages with the recipe's ``predict``, scored with the reward.
* ``create_candidates``: a seeded random draw of ground-truth pages from the dataset's train split (pages of volumes
  that are in neither monitor nor test), linked below the run's curriculum folder. The recipe renders fresh pages
  here; a handwriting reward needs a true text, so ours are pages the reader was trained on.
* ``select_candidates``: ``htr_select`` (pages with a looped or capped answer, then the largest spread, then random).
* ``evaluate`` / ``summarize``: ``htr_evaluate`` (sampled answers, and greedy readings except while mining).
* ``environment_class``: ``htr_environment.HtrEnvironment``; ``reward_metric``: its ``reward`` component.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import htr_evaluate
import htr_select
from htr_reward import components
from htr_settings import data_dir, settings

PROMPT_TOKENS = 6  # </s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>
OBSERVATION_TOKENS = 32  # room for the environment's "Reward: 0.87" after the answer


class HtrTask:
    """The hooks the SFT and RLVR stages call."""

    prompt_file = Path(__file__).with_name("prompt.txt")
    environment_class = "htr_environment.HtrEnvironment"
    reward_metric = "reward"

    def monitor(self, model, processor, rows):
        from nemotron.recipes.parse.training.sft import predict

        weights, loop = settings()["weights"], settings()["loop"]
        predictions = predict(model, processor, rows)
        for prediction, row in zip(predictions, rows, strict=True):
            ended = bool(prediction["ended"])  # predict() saw the token ids; an answer without EOS hit its cap
            prediction.update(
                components(prediction["prediction"], row["target"], weights=weights, loop=loop, ended=ended)
            )
        pooled = sum(p["cer"] * p["chars"] for p in predictions) / max(1.0, sum(p["chars"] for p in predictions))
        summary = {key: sum(p[key] for p in predictions) / len(predictions) for key in ("reward", "cer", "looped")}
        return predictions, dict(summary, cer_pooled=pooled, ended=sum(p["ended"] for p in predictions) / len(rows))

    def eligible(self) -> list[dict]:
        """The train pages an answer can be generated and trained on whole, within the YAML's limits."""
        from nemotron.recipes.parse.training.data import load_pages

        limits = settings()
        prompt = self.prompt_file.read_text()
        rows = []
        for row in load_pages(data_dir() / "train.jsonl"):
            prefix = row["visual_tokens"] + PROMPT_TOKENS
            if (
                row["prompt"] == prompt
                and row["target"].strip()
                and row["target_tokens"] <= limits["max_new_tokens"]
                and prefix + limits["max_new_tokens"] + OBSERVATION_TOKENS <= limits["max_model_len"]
            ):
                rows.append(row)
        return rows

    def create_candidates(self, output, *, seed, contexts):
        """``contexts`` pages (the stage's ``candidate_contexts``) drawn by ``seed``, images linked below ``output``."""
        output = Path(output)
        pool = sorted(self.eligible(), key=lambda row: row["id"])
        if contexts > len(pool):
            raise ValueError(f"Asked for {contexts} candidates; the train split has {len(pool)} eligible pages")
        rows = random.Random(seed).sample(pool, contexts)
        images = output / "rl_candidates"
        images.mkdir(parents=True, exist_ok=True)
        for row in rows:
            source = Path(row["image"])
            link = images / source.name
            os.symlink(source.resolve(), link)
            row["image"] = str(link)
        (output / "rl_candidates.jsonl").write_text(
            "".join(json.dumps(dict(row, image=str(Path(row["image"]).relative_to(output)))) + "\n" for row in rows)
        )
        return rows

    def select_candidates(self, candidates, answers, *, pages, seed, excluded):
        shares = settings()
        selected = htr_select.select_training_pages(
            candidates,
            answers,
            pages=pages,
            seed=seed,
            excluded=excluded,
            failing=shares["failing_share"],
            informed=shares["informed_share"],
        )
        print("RL pages:", json.dumps(htr_select.selection_report(candidates, answers, selected)), flush=True)
        return selected

    def evaluate(self, model, rows, output, *, samples, processor_source):
        """Sampled answers for every page; greedy readings too, except for the mining pass (``output`` = mining)."""
        greedy = [] if Path(output).name == "mining" else settings()["greedy_repetition_penalties"]
        return htr_evaluate.evaluate_isolated(
            model, rows, output, samples=samples, processor_source=processor_source, settings=settings(), greedy=greedy
        )

    def summarize(self, answers):
        return htr_evaluate.summarize(answers)
