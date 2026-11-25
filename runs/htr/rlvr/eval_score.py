"""Score one reader's readings of one split page by page: a TSV a page and a JSON summary (CPU).

    PYTHONPATH=/mnt/scratch/gabriel/src/Nemotron/src:runs/htr/rlvr /mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python \\
        runs/htr/rlvr/eval_score.py READS.jsonl SPLIT OUT_PREFIX [--data DIR] [--checkpoint NAME]

Two error rates, both edits over the truth's characters:

* ``cer``: the text our test and dev reads are scored by (``checkpoint_cer.page_text``: boxes, classes and table
  markup out, white space as one space; NFC, ends trimmed, as ``compare_reads.py``); exact Levenshtein.
* ``edits_unordered``: the order-free distance: the truth's lines matched one to one to the reading's lines at the
  least total cost (Hungarian; a line left unmatched costs its length), each line's text as ``cer``'s. A block or a
  margin column read in another order than the truth's then costs little (CURATION E72: about three pages in each of
  dev and test differ from the truth mostly in order). Pooled over the truth's characters it is the order-free CER;
  a line split or joined otherwise than the truth's costs more here than in ``cer``.
* ``reward_cer``: the reward's text part (``htr_reward.components``: NVIDIA's ``text_quality`` normalisation), the
  number pilot 1's before/after reported (``cer_pooled``).

And from the reward: ``looped`` (the loop detector of the training YAML), ``capped`` (the reader stopped at its token
limit), ``ended``, ``gate``, ``boxes``, ``reward``. Kinds of page (``table``, ``list``, ``dark``) by ``htr_candidates``.

Writes ``OUT_PREFIX.tsv`` and ``OUT_PREFIX.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from htr_candidates import kinds_of, rules
from htr_reward import components
from htr_settings import settings
from nemotron.recipes.parse.document_rewards.baselines import _distance

DATA = Path("/mnt/scratch/gabriel/htr/rlvr/data_v2")
LAYOUT = re.compile(r"<(?:x|y)_[0-9.]+>|<class_[A-Za-z-]+>")
TABLE = re.compile(r"\\begin\{tabular\}\{[^}]*\}|\\end\{tabular\}|\\linebreak\{\}|\\\\| & ")
COLUMNS = ("id", "split", "volume", "kinds", "chars", "edits", "cer", "edits_unordered", "reward_cer", "looped", "capped", "ended", "gate",
           "boxes", "reward", "tokens", "finish", "seconds", "raw_sha1")


def page_text(answer: str) -> str:
    """``checkpoint_cer.page_text``, then NFC and the ends trimmed (``compare_reads.py``)."""
    text = " ".join(TABLE.sub(" ", LAYOUT.sub("\n", answer)).replace("#", " ").split())
    return unicodedata.normalize("NFC", text).strip()


def lines_of(answer: str) -> list[str]:
    """The answer's lines (blocks and line breaks), each as ``page_text`` makes a text."""
    return [line for line in (page_text(part) for part in TABLE.sub("\n", LAYOUT.sub("\n", answer)).split("\n")) if line]


def unordered_edits(truth: str, reading: str) -> int:
    """Edits when the truth's lines are matched to the reading's lines in the order that costs least."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    ours, theirs = lines_of(truth), lines_of(reading)
    n, m = len(ours), len(theirs)
    if not n or not m:
        return sum(map(len, ours)) + sum(map(len, theirs))
    big = 10**9
    cost = np.zeros((n + m, n + m), dtype=np.int64)
    cost[:n, :m] = [[_distance(a, b) for b in theirs] for a in ours]
    cost[:n, m:] = big
    cost[n:, :m] = big
    cost[np.arange(n), m + np.arange(n)] = [len(a) for a in ours]  # a truth line left out
    cost[n + np.arange(m), np.arange(m)] = [len(b) for b in theirs]  # a reading line with no truth
    rows, columns = linear_sum_assignment(cost)
    return int(cost[rows, columns].sum())


def score(read: dict, row: dict, weights: dict, loop: dict) -> dict:
    truth, reading = page_text(row["target"]), page_text(read["raw"])
    edits = _distance(truth, reading)
    capped = read.get("finish") == "length"
    parts = components(read["raw"], row["target"], weights=weights, loop=loop, ended=read.get("ended"), capped=capped)
    rule = rules({"rule": "targeted"})
    return {
        "id": row["id"],
        "split": row["split"],
        "volume": f"{row['collection']}/{row['volume']}",
        "kinds": ",".join(kinds_of(row, rule, long_from=10**9)) or "-",
        "chars": len(truth),
        "edits": edits,
        "cer": edits / max(1, len(truth)),
        "edits_unordered": unordered_edits(row["target"], read["raw"]),
        "reward_cer": parts["cer"],
        "reward_chars": parts["chars"],
        "looped": int(parts["looped"]),
        "capped": int(capped),
        "ended": "" if read.get("ended") is None else int(bool(read["ended"])),
        "gate": parts["gate"],
        "boxes": parts["boxes"],
        "reward": parts["reward"],
        "tokens": read.get("tokens"),
        "finish": read.get("finish"),
        "seconds": read.get("seconds"),
        "raw_sha1": hashlib.sha1(read["raw"].encode()).hexdigest()[:12],
    }


def summary(scored: list[dict]) -> dict:
    def pooled(rows: list[dict], what: str = "cer") -> float | None:
        if not rows:
            return None
        if what == "reward_cer":
            return sum(r["reward_cer"] * r["reward_chars"] for r in rows) / max(1.0, sum(r["reward_chars"] for r in rows))
        return sum(r["edits"] for r in rows) / max(1, sum(r["chars"] for r in rows))

    failed = [r for r in scored if r["capped"] or r["looped"]]
    clean = [r for r in scored if not (r["capped"] or r["looped"])]
    return {
        "pages": len(scored),
        "cer_pooled": pooled(scored),
        "cer_mean": sum(r["cer"] for r in scored) / len(scored),
        "reward_cer_pooled": pooled(scored, "reward_cer"),
        "reward_mean": sum(r["reward"] for r in scored) / len(scored),
        "pages_capped": sum(r["capped"] for r in scored),
        "pages_looped": sum(r["looped"] for r in scored),
        "pages_capped_or_looped": [r["id"] for r in failed],
        "cer_pooled_without_capped_or_looped": pooled(clean),
        "tokens_mean": sum(r["tokens"] or 0 for r in scored) / len(scored),
        "finish": {reason: sum(r["finish"] == reason for r in scored) for reason in sorted({str(r["finish"]) for r in scored})},
        "by_kind": {
            kind: {"pages": len(part), "cer_pooled": pooled(part)}
            for kind in ("table", "list", "dark")
            if (part := [r for r in scored if kind in r["kinds"].split(",")])
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("reads", type=Path)
    parser.add_argument("split", choices=("test", "monitor"))
    parser.add_argument("out", type=Path, help="prefix of the .tsv and .json written")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--checkpoint", default="", help="a name for the summary")
    args = parser.parse_args()

    rows = [json.loads(line) for line in (args.data / f"{args.split}.jsonl").read_text("utf-8").split("\n") if line.strip()]
    reads = {r["id"]: r for r in map(json.loads, (l for l in args.reads.read_text("utf-8").split("\n") if l.strip()))}
    if set(reads) != {row["id"] for row in rows}:
        raise ValueError(f"{args.reads} does not read exactly the {len(rows)} pages of {args.split}")
    values = settings()
    scored = [score(reads[row["id"]], row, values["weights"], values["loop"]) for row in rows]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with Path(f"{args.out}.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scored)
    result = {"checkpoint": args.checkpoint, "reads": str(args.reads), "split": args.split, "data": str(args.data),
              "loop": values["loop"], **summary(scored)}
    Path(f"{args.out}.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"{args.out.name}: CER {100 * result['cer_pooled']:.2f} % pooled over {result['pages']} pages "
          f"(reward CER {100 * result['reward_cer_pooled']:.2f} %); capped {result['pages_capped']}, "
          f"looped {result['pages_looped']}; without them {100 * (result['cer_pooled_without_capped_or_looped'] or 0):.2f} %")


if __name__ == "__main__":
    main()
