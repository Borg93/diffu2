"""Pages whose ground truth the readers agree against: candidates for label errors (CURATION E72's evaluation pages).

For each page of a split, the guarded readings of every model evaluated (``/mnt/scratch/gabriel/htr/rlvr/eval/MODEL/
reads/READER.SPLIT.jsonl``) are made plain text as the dev CER makes them (``checkpoint_cer.page_text``). The medoid
reading (least summed distance to the others) stands for the readers; their agreement is the median distance of the
readings to it over its length. A page is suspect when the readers agree closely and the medoid is far from the truth,
or when the medoid's length is far from the truth's (truth partial, or text in the truth that no reader sees).

    uv run --with rapidfuzz python runs/htr/verify/label_check.py [SPLIT ...] > /mnt/scratch/gabriel/htr/rlvr/eval/label_check.txt
"""

import json
import statistics as st
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from diffu_htr_train.nemotron_parse.checkpoint_cer import page_text

EVAL = Path("/mnt/scratch/gabriel/htr/rlvr/eval")
DATA = Path("/mnt/scratch/gabriel/htr/rlvr/data_v2")
MODELS = ("sft", "pilot1")
READERS = ("hf_norepeat10", "served_stop", "vllm_p1.1")


def lines(path: Path) -> list[str]:
    return [line for line in path.read_text("utf-8").split("\n") if line]


def main(splits: list[str]) -> None:
    for split in splits:
        truth = {}
        for line in lines(DATA / f"{split}.jsonl"):
            row = json.loads(line)
            truth[row["id"]] = (page_text(row["target"]), str(DATA / row["image"]))
        reads: dict[str, list[str]] = {page: [] for page in truth}
        for model in MODELS:
            for reader in READERS:
                for line in lines(EVAL / model / "reads" / f"{reader}.{split}.jsonl"):
                    row = json.loads(line)
                    if row["id"] in reads:
                        reads[row["id"]].append(page_text(row["raw"]))
        found = []
        for page, (text, image) in truth.items():
            readings = reads[page]
            medoid = min(readings, key=lambda r: sum(Levenshtein.distance(r, o) for o in readings))
            agreement = st.median(Levenshtein.distance(medoid, o) for o in readings) / max(1, len(medoid))
            cer = Levenshtein.distance(text, medoid) / max(1, len(text))
            ratio = len(medoid) / max(1, len(text))
            found.append((page, len(text), cer, agreement, ratio, image, medoid, text))
        suspects = [f for f in found if (f[2] > 0.10 and f[3] < 0.03) or f[4] > 1.3 or f[4] < 0.77]
        print(f"== {split}: {len(found)} pages; {len(suspects)} suspect")
        print("   agreement among readings (median over pages):", round(st.median(f[3] for f in found), 4))
        for page, chars, cer, agreement, ratio, image, medoid, text in sorted(suspects, key=lambda f: f[3] - f[2]):
            print(f"{page}\tchars {chars}\tmedoid CER {cer:.3f}\treaders' spread {agreement:.3f}\tlength ratio {ratio:.2f}\t{image}")
            ops = Levenshtein.opcodes(text, medoid)
            chunks = sorted(
                ((max(o.src_end - o.src_start, o.dest_end - o.dest_start), o) for o in ops if o.tag != "equal"),
                key=lambda c: -c[0],
            )[:3]
            for _, o in chunks:
                print(f"    {o.tag}: truth «{text[o.src_start:o.src_end][:160]}» readers «{medoid[o.dest_start:o.dest_end][:160]}»")


if __name__ == "__main__":
    main(sys.argv[1:] or ["test", "monitor"])
