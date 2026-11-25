"""Which candidate pages RLVR trains on: the pages whose sampled answers differ, and the pages that fail to end.

NVIDIA's dental curriculum (``nemotron/recipes/parse/dental/curriculum.py``) has a right/wrong score and takes the
pages the model gets right about half the time (75 %), the rest at random (25 %). Our reward is continuous, so
"about half right" becomes "the sampled answers disagree": GRPO's advantage is each answer's reward minus the mean of
its page's answers, so a page whose answers all score the same teaches nothing, however low the score.

Of the ``pages`` asked for:

* ``FAILING`` (a quarter; the training YAML's ``htr.failing_share``, 0.5 in grpo_htr_v2.yaml) are pages where at least one sampled answer looped or ran to the token cap, those with the
  largest spread first (a page where every answer loops has no spread; it comes after the mixed ones). Fewer are
  taken if fewer exist; the spread rule takes the places left.
* up to ``INFORMED`` (three quarters, the failing ones counted in) are the remaining pages with the largest spread
  of reward over their answers, only pages with a spread above zero.
* the rest are drawn at random from what is left, as theirs are.

No page of a monitor or test volume is taken, and no page twice. Unlike the dental rule, several pages of one volume
may be taken (ours are real volumes, 144 of them, not a generator's seeds).
"""

from __future__ import annotations

import random
import statistics

FAILING = 0.25
INFORMED = 0.75


def spread(answers: list[dict]) -> float:
    """The standard deviation of the reward over a page's sampled answers."""
    return statistics.pstdev(answer["reward"] for answer in answers)


def failed(answers: list[dict]) -> bool:
    """Whether any of the page's sampled answers looped or ran to the token cap."""
    return any(answer["looped"] > 0 or answer["capped"] > 0 for answer in answers)


def select_training_pages(
    candidates: list[dict],
    answers: list[dict],
    *,
    pages: int,
    seed: int,
    excluded: list[dict],
    failing: float = FAILING,
    informed: float = INFORMED,
) -> list[dict]:
    """The ``pages`` candidates to train on, in a seeded random order.

    Args:
        candidates: Candidate rows (``id``, ``context_id``, ``split`` = "train", ...).
        answers: A group a candidate in the same order: ``{"id", "answers": [{"reward", "looped", "capped"}, ...]}``.
        pages: How many to select.
        seed: The seed of every random choice here.
        excluded: The fixed dataset's rows; those of the monitor and test splits fence off their volumes and ids.
        failing: The share to take from pages with a looped or capped answer.
        informed: The share taken by a rule (failing and spread together); the rest is random.

    Raises:
        ValueError: When groups and candidates do not match, a candidate is repeated, is not a train page, or is
            from a monitor or test volume.
    """
    if len(candidates) != len(answers) or pages > len(candidates):
        raise ValueError("Need a sampled answer group for each candidate and enough candidates")
    if [row["id"] for row in candidates] != [row["id"] for row in answers]:
        raise ValueError("Candidate/answer IDs differ")
    if not all(group["answers"] for group in answers):
        raise ValueError("Each candidate needs sampled answers")
    if not 0 <= failing <= informed <= 1:
        raise ValueError("Need 0 <= failing <= informed <= 1")
    ids = [row["id"] for row in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("A candidate page is repeated")
    if any(row.get("split") != "train" for row in candidates):
        raise ValueError("RL candidates must be pages of the dataset's train split")
    held_out = [row for row in excluded if row.get("split") != "train"]
    if not {row["context_id"] for row in candidates}.isdisjoint(row["context_id"] for row in held_out):
        raise ValueError("RL candidates must be from volumes outside the monitor and test splits")
    if not set(ids).isdisjoint(row["id"] for row in held_out):
        raise ValueError("RL candidates must not be monitor or test pages")

    rng = random.Random(seed)
    spreads = [spread(group["answers"]) for group in answers]
    order = list(range(len(candidates)))
    rng.shuffle(order)  # ties are broken by the seed, not by the manifest's order
    by_spread = sorted(order, key=lambda i: -spreads[i])
    failures = [i for i in by_spread if failed(answers[i]["answers"])]
    selected = failures[: round(pages * failing)]
    taken = set(selected)
    wanted = int(pages * informed) - len(selected)
    selected += [i for i in by_spread if i not in taken and spreads[i] > 0][:wanted]
    taken = set(selected)
    remaining = [i for i in order if i not in taken]
    selected += remaining[: pages - len(selected)]
    rng.shuffle(selected)
    return [candidates[i] for i in selected]


def selection_report(candidates: list[dict], answers: list[dict], selected: list[dict]) -> dict:
    """What the selection took, for the run's log: counts of failing pages, spreads and volumes."""
    groups = {group["id"]: group["answers"] for group in answers}
    chosen = [groups[row["id"]] for row in selected]

    def mean(values: list[float]) -> float:
        return sum(values) / max(1, len(values))

    reasons = sorted({row.get("candidate_reason", "random") for row in candidates})

    def by_reason(rows: list[dict], *, failing: bool = False) -> dict[str, int]:
        return {
            reason: sum(
                row.get("candidate_reason", "random") == reason and (not failing or failed(groups[row["id"]]))
                for row in rows
            )
            for reason in reasons
        }

    return {
        "candidates": len(candidates),
        "candidates_with_a_failed_answer": sum(failed(group) for group in groups.values()),
        "candidates_with_spread": sum(spread(group) > 0 for group in groups.values()),
        "selected": len(selected),
        "selected_with_a_failed_answer": sum(failed(group) for group in chosen),
        "selected_with_a_failed_answer_and_no_spread": sum(failed(group) and spread(group) == 0 for group in chosen),
        "candidates_by_reason": by_reason(candidates),
        "candidates_with_a_failed_answer_by_reason": by_reason(candidates, failing=True),
        "selected_by_reason": by_reason(selected),
        "selected_with_a_failed_answer_by_reason": by_reason(selected, failing=True),
        "selected_volumes": len({row["context_id"] for row in selected}),
        "mean_spread_candidates": mean([spread(group) for group in groups.values()]),
        "mean_spread_selected": mean([spread(group) for group in chosen]),
        "mean_reward_candidates": mean([a["reward"] for group in groups.values() for a in group]),
        "mean_reward_selected": mean([a["reward"] for group in chosen for a in group]),
    }
