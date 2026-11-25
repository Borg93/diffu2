"""Which train pages are offered to mining as RL candidates: a seeded random draw (pilot 1), or a targeted draw.

Pilot 1 drew 1,024 pages at random: 13 had a sampled answer that looped or ran to the cap, the mean reward spread was
0.046, and its gain came from 2 of the 4 test pages that loop. The targeted draw offers mining the kinds of page where
the reader fails, so the selection (``htr_select``) has failing pages and spread to choose from:

* ``table``: the truth has a table (``\\begin{tabular}``);
* ``dark``: the page's thumbnail is dark (mean grey level below ``dark_brightness``; measured by
  ``build_dataset_v2.py`` as ``brightness``; absent in the v1 bundle, where no page is of this kind);
* ``list``: the truth is a list or a ledger: at least ``list_min_lines`` visible lines and either their median length
  is below ``list_median_chars`` characters, or at least ``list_marked_share`` of them start with a dash or a ditto
  mark (— – - „ ") or hold a dot leader (". . .");
* ``long``: the truth's token count is in the top ``1 - long_quantile`` of the eligible pages.

Kinds are taken in ``order``; each takes at most ``quota[kind]`` of the candidates, drawn at random (the stage's
``candidate_seed``) from its eligible pages not already taken; the rest is drawn at random from all the others. A
page may be of several kinds; ``candidate_kinds`` lists them all, ``candidate_reason`` the one it was taken for.
"""

from __future__ import annotations

import random
import re
import statistics
from typing import Any

KINDS = ("table", "dark", "list", "long")
LAYOUT = re.compile(r"<(?:x|y)_[0-9.]+>|<class_[^>]+>")
TABLE_MARKUP = re.compile(r"\\begin\{tabular\}\{[^}]*\}|\\end\{tabular\}|\\linebreak\{\}|\\\\")
MARK = re.compile(r"^[—–\-„\"]")
LEADER = re.compile(r"\.\s?\.\s?\.")

RANDOM = {"rule": "random"}
TARGETED = {
    "rule": "targeted",
    "order": ["table", "dark", "list", "long"],
    "quota": {"table": 1.0, "dark": 0.125, "list": 0.375, "long": 0.1875},
    "list_min_lines": 8,
    "list_median_chars": 15,
    "list_marked_share": 0.3,
    "long_quantile": 0.9,
    "dark_brightness": 100.0,
}


def visible_lines(target: str) -> list[str]:
    """The truth's text lines: block boxes and classes and table markup out (a table row is a line), blanks dropped."""
    text = TABLE_MARKUP.sub("\n", LAYOUT.sub("\n", target or ""))
    return [line.strip() for line in text.split("\n") if line.strip()]


def page_features(target: str) -> dict[str, Any]:
    """What the list rule reads from a truth; ``build_dataset_v2.py`` stores these in each manifest row."""
    lines = visible_lines(target)
    return {
        "has_table": "\\begin{tabular}" in (target or ""),
        "lines": len(lines),
        "line_median": float(statistics.median(len(line) for line in lines)) if lines else 0.0,
        "marked_lines": sum(bool(MARK.match(line)) for line in lines),
        "leader_lines": sum(bool(LEADER.search(line)) for line in lines),
    }


def rules(config: dict[str, Any] | None) -> dict[str, Any]:
    """The candidate rule of the training YAML's ``htr.candidates``, completed from ``TARGETED``; random if unset."""
    if not config or config.get("rule", "random") == "random":
        return dict(RANDOM)
    if config["rule"] != "targeted":
        raise ValueError(f"Unknown candidate rule: {config['rule']}")
    merged = {**TARGETED, **config, "quota": {**TARGETED["quota"], **config.get("quota", {})}}
    unknown = set(merged["order"]) - set(KINDS)
    if unknown:
        raise ValueError(f"Unknown candidate kinds: {sorted(unknown)}")
    if not all(0 <= float(merged["quota"][kind]) <= 1 for kind in merged["order"]):
        raise ValueError("Each kind's quota is a share of the candidates, 0 to 1")
    return merged


def long_threshold(rows: list[dict], quantile: float) -> int:
    """The target token count at ``quantile`` of ``rows`` (a page at or above it is long)."""
    counts = sorted(row["target_tokens"] for row in rows)
    return counts[min(len(counts) - 1, int(quantile * len(counts)))]


def kinds_of(row: dict, rule: dict[str, Any], long_from: int) -> list[str]:
    """The kinds of one page under ``rule`` (features are computed from the target where the row lacks them)."""
    features = row if "lines" in row else {**row, **page_features(row["target"])}
    lines = features["lines"]
    marked = (features["marked_lines"] + features["leader_lines"]) / lines if lines else 0.0
    found = []
    if features.get("has_table", "\\begin{tabular}" in row["target"]):
        found.append("table")
    if features.get("brightness") is not None and features["brightness"] < rule["dark_brightness"]:
        found.append("dark")
    if lines >= rule["list_min_lines"] and (
        features["line_median"] < rule["list_median_chars"] or marked >= rule["list_marked_share"]
    ):
        found.append("list")
    if row["target_tokens"] >= long_from:
        found.append("long")
    return found


def draw(pool: list[dict], contexts: int, seed: int, rule: dict[str, Any]) -> tuple[list[dict], dict[str, Any]]:
    """``contexts`` pages of ``pool`` by ``rule``, in a seeded order, each with ``candidate_kinds`` and
    ``candidate_reason``; and a report of the counts (eligible and taken, by kind).

    Raises:
        ValueError: When the pool has fewer pages than asked for.
    """
    if contexts > len(pool):
        raise ValueError(f"Asked for {contexts} candidates; the train split has {len(pool)} eligible pages")
    pool = sorted(pool, key=lambda row: row["id"])
    rng = random.Random(seed)
    if rule["rule"] == "random":
        rows = rng.sample(pool, contexts)  # pilot 1's draw, unchanged
        return [dict(row) for row in rows], {"rule": "random", "candidates": len(rows), "eligible": len(pool)}
    long_from = long_threshold(pool, rule["long_quantile"])
    kinds = {row["id"]: kinds_of(row, rule, long_from) for row in pool}
    taken: dict[str, str] = {}
    for kind in rule["order"]:
        members = [row for row in pool if kind in kinds[row["id"]] and row["id"] not in taken]
        room = min(len(members), int(contexts * float(rule["quota"][kind])), contexts - len(taken))
        for row in rng.sample(members, room):
            taken[row["id"]] = kind
    rest = [row for row in pool if row["id"] not in taken]
    for row in rng.sample(rest, contexts - len(taken)):
        taken[row["id"]] = "random"
    rows = [
        dict(row, candidate_kinds=",".join(kinds[row["id"]]), candidate_reason=taken[row["id"]])
        for row in pool
        if row["id"] in taken
    ]
    rng.shuffle(rows)
    report = {
        "rule": "targeted",
        "candidates": len(rows),
        "eligible": len(pool),
        "long_from_target_tokens": long_from,
        "eligible_by_kind": {kind: sum(kind in found for found in kinds.values()) for kind in KINDS},
        "taken_for": {reason: sum(r["candidate_reason"] == reason for r in rows) for reason in (*rule["order"], "random")},
        "candidates_by_kind": {kind: sum(kind in kinds[r["id"]] for r in rows) for kind in KINDS},
        "candidates_of_no_kind": sum(not kinds[r["id"]] for r in rows),
        "settings": {key: value for key, value in rule.items() if key != "rule"},
    }
    return rows, report
