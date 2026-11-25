"""Two checkpoints' scored readings compared page by page, with a bootstrap interval over pages (CPU).

    /mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python runs/htr/rlvr/compare_eval.py BASE NEW [--json OUT]

BASE and NEW are ``eval_score.py`` TSVs of the same reader and split, or two ``eval_pilot.sh`` folders
(``/mnt/scratch/gabriel/htr/rlvr/eval/NAME``, or just NAME): then every reader and split both have is compared.

For each pair: pooled CER (edits over the truth's characters) of BASE and NEW, the gain in points with a 95 %
bootstrap interval (10,000 draws of pages with replacement, seed 0, as ``compare_reads.py``) and the share of draws
with a gain; pages better, worse, the same (by edits) and with identical readings; the same on the pages neither
reading capped or looped (where a gain is not one page that stopped looping); by kind of page; the pages that moved
most.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

EVAL = Path("/mnt/scratch/gabriel/htr/rlvr/eval")


def load(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in rows:
        for key in ("chars", "edits", "looped", "capped"):
            row[key] = int(row[key])
    return {row["id"]: row for row in rows}


def rate(rows: list[dict], key: str = "edits") -> float:
    return 100 * sum(r[key] for r in rows) / max(1, sum(r["chars"] for r in rows))


def paired(base: dict[str, dict], new: dict[str, dict], ids: list[str], draws: int = 10000) -> dict:
    """BASE against NEW on ``ids``: rates, gain with its interval, page counts, the most moved pages."""
    if not ids:
        return {"pages": 0}
    old_edits = np.array([base[i]["edits"] for i in ids])
    new_edits = np.array([new[i]["edits"] for i in ids])
    chars = np.array([new[i]["chars"] for i in ids])
    picks = np.random.default_rng(0).integers(0, len(ids), (draws, len(ids)))
    gain = 100 * (old_edits[picks].sum(1) - new_edits[picks].sum(1)) / np.maximum(1, chars[picks].sum(1))
    moved = sorted(zip(ids, (new_edits - old_edits).tolist(), strict=True), key=lambda pair: pair[1])
    old, now = 100 * old_edits.sum() / max(1, chars.sum()), 100 * new_edits.sum() / max(1, chars.sum())
    return {
        "pages": len(ids),
        "chars": int(chars.sum()),
        "cer_base": round(float(old), 3),
        "cer_new": round(float(now), 3),
        "gain_points": round(float(old - now), 3),
        "gain_95": [round(float(np.percentile(gain, 2.5)), 3), round(float(np.percentile(gain, 97.5)), 3)],
        "share_of_draws_with_a_gain": round(float(np.mean(gain > 0)), 3),
        "better": int((new_edits < old_edits).sum()),
        "worse": int((new_edits > old_edits).sum()),
        "same_edits": int((new_edits == old_edits).sum()),
        "identical_readings": sum(base[i]["raw_sha1"] == new[i]["raw_sha1"] for i in ids),
        "most_better": [(page, change) for page, change in moved[:5] if change < 0],
        "most_worse": [(page, change) for page, change in moved[::-1][:5] if change > 0],
    }


def compare(base_path: Path, new_path: Path) -> dict:
    base, new = load(base_path), load(new_path)
    if set(base) != set(new):
        raise ValueError(f"{base_path} and {new_path} do not read the same pages")
    ids = sorted(base)
    failing = lambda rows, i: rows[i]["capped"] or rows[i]["looped"]  # noqa: E731
    result = {
        "base": str(base_path),
        "new": str(new_path),
        "all": paired(base, new, ids),
        "neither_capped_nor_looped": paired(base, new, [i for i in ids if not failing(base, i) and not failing(new, i)]),
        "capped_or_looped": {
            "base": sorted(i for i in ids if failing(base, i)),
            "new": sorted(i for i in ids if failing(new, i)),
        },
        "by_kind": {},
    }
    for kind in ("table", "list", "dark"):
        part = [i for i in ids if kind in new[i]["kinds"].split(",")]
        if part:
            result["by_kind"][kind] = paired(base, new, part)
    return result


def show(name: str, result: dict) -> None:
    def line(label: str, part: dict) -> str:
        if not part.get("pages"):
            return f"  {label}: no pages"
        return (
            f"  {label}: {part['cer_base']:.2f} -> {part['cer_new']:.2f} %, gain {part['gain_points']:+.2f} points "
            f"(95 % {part['gain_95'][0]:+.2f} to {part['gain_95'][1]:+.2f}; share of draws with a gain "
            f"{part['share_of_draws_with_a_gain']:.3f}) on {part['pages']} pages; better {part['better']}, worse "
            f"{part['worse']}, same {part['same_edits']} (identical readings {part['identical_readings']})"
        )

    print(f"== {name}")
    print(line("all", result["all"]))
    print(line("neither capped nor looped", result["neither_capped_nor_looped"]))
    for kind, part in result["by_kind"].items():
        print(line(kind, part))
    print(f"  capped or looped: base {result['capped_or_looped']['base']}, new {result['capped_or_looped']['new']}")
    whole = result["all"]
    print("  most better: " + "; ".join(f"{p} {c:+d} edits" for p, c in whole["most_better"]))
    print("  most worse: " + "; ".join(f"{p} {c:+d} edits" for p, c in whole["most_worse"]))


def folder(value: str) -> Path:
    path = Path(value)
    return path if path.is_dir() and path.is_absolute() else (path if path.is_dir() else EVAL / value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("base")
    parser.add_argument("new")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    results = {}
    if args.base.endswith(".tsv"):
        results[Path(args.new).name] = compare(Path(args.base), Path(args.new))
    else:
        base, new = folder(args.base) / "scores", folder(args.new) / "scores"
        for path in sorted(base.glob("*.tsv")):
            if (new / path.name).is_file():
                results[path.stem] = compare(path, new / path.name)
        if not results:
            raise FileNotFoundError(f"No reader and split scored in both {base} and {new}")
    for name, result in results.items():
        show(name, result)
    if args.json:
        args.json.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
