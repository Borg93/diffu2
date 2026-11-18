"""The old pipeline as one reader on the 200 test pages the page readers read (CURATION E43): YOLO regions, YOLO lines
within them, TrOCR on each line, as htrflow runs them, scored on the whole page as Nemotron Parse is.

    uv run python runs/htr/verify/old_pipeline_test_read.py prepare OUT [LIMIT]   # links to the page images, the truth, the pipelines
    uv run python runs/htr/verify/old_pipeline_test_read.py score OUT RUN ORDER   # htrflow's text files -> OUT/RUN_ORDER.jsonl

``prepare`` writes OUT/images/<id>.jpg (links to the pages the readers were given), OUT/pages.jsonl (id, subset, the
truth as the other reads hold it) and a pipeline a recogniser:

    swe2   Riksarkivet/trocr-base-handwritten-hist-swe-2, the public TrOCR of the 2024 run that labelled the pools
           (its label files name the two YOLO models and their revisions, and the recogniser only as "./model")
    lion   the Swedish Lion 2026, large, character vocabulary: today's TrOCR, with its card's generation setting and
           the 128 tokens a line it was trained with (the local copy names no limit, and transformers then stops a
           line at 20 characters), greedy, as our line evaluation read it

Each pipeline exports the page's text twice: ``lines_order`` (regions as YOLO gives them, lines top-down: the card's
pipeline) and ``reading_order`` (htrflow's ReadingOrderMarginalia for single pages). runs/htr/old_pipeline_test.sh
runs both and scores them with ``htr-eval cer``.
"""
import json
import re
import sys
from pathlib import Path

LAYOUT = re.compile(r"<(?:x|y)_[0-9.]+>|<class_[A-Za-z-]+>")  # as nemotron_parse_test_read.py makes the truth's text
TABLE = re.compile(r"\\begin\{tabular\}\{[^}]*\}|\\end\{tabular\}|\\linebreak\{\}|\\\\| & ")
YOLO = {"regions": ("Riksarkivet/yolov9-regions-1", "7c44178d85926b4a096c55c89bf224855a201fbf"),
        "lines": ("Riksarkivet/yolov9-lines-within-regions-1", "ea2f8987cba316abc62762f3030266ec8875338d")}
TROCR = {"swe2": ("Riksarkivet/trocr-base-handwritten-hist-swe-2", ""),
         "lion": ("/mnt/work/gabriel/models/swedish-lion-trocr-large-handwritten-char", "\n      interpolate_pos_encoding: true\n      max_new_tokens: 128")}
PIPELINE = """steps:
- step: Segmentation
  settings:
    model: yolo
    model_settings:
      model: {regions[0]}
      revision: {regions[1]}
    generation_settings:
      batch_size: 8
- step: Segmentation
  settings:
    model: yolo
    model_settings:
      model: {lines[0]}
      revision: {lines[1]}
    generation_settings:
      batch_size: 16
- step: TextRecognition
  settings:
    model: TrOCR
    model_settings:
      model: {trocr}
    generation_settings:
      batch_size: 32{generation}
- step: OrderLines
- step: Export
  settings:
    format: txt
    dest: {out}/lines_order
- step: ReadingOrderMarginalia
  settings:
    two_page: false
- step: Export
  settings:
    format: txt
    dest: {out}/reading_order
- step: Export
  settings:
    format: alto
    dest: {out}/alto
"""


def text(answer: str) -> str:
    return " ".join(TABLE.sub(" ", LAYOUT.sub("\n", answer)).replace("#", " ").split())


if __name__ == "__main__":
    what, out = sys.argv[1], Path(sys.argv[2]).resolve()
    if what == "prepare":
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
        wanted = sorted(json.loads(line)["id"] for shard in (0, 1) for line in open(f"runs/htr/uocr2/test_read/shard{shard}.jsonl", encoding="utf-8"))[:limit]
        rows = open("data/htr/nemotron_parse_pages.test.jsonl", encoding="utf-8").read().splitlines()
        metas = [json.loads(line) for line in open("data/htr/nemotron_parse_pages.test.meta.jsonl", encoding="utf-8")]
        index = {meta["id"]: i for i, meta in enumerate(metas)}
        (out / "images").mkdir(parents=True, exist_ok=True)
        with open(out / "pages.jsonl", "w", encoding="utf-8") as pages:
            for page in wanted:
                row, meta = json.loads(rows[index[page]]), metas[index[page]]
                link = out / "images" / f"{page}.jpg"
                if not link.is_symlink():
                    link.symlink_to(row["images"][0])
                pages.write(json.dumps({"id": page, "subset": meta["collection"] or meta["volume"], "text": text(row["messages"][1]["content"])}, ensure_ascii=False) + "\n")
        for run, (trocr, generation) in TROCR.items():
            (out / f"{run}.yaml").write_text(PIPELINE.format(regions=YOLO["regions"], lines=YOLO["lines"], trocr=trocr, generation=generation, out=out / run), encoding="utf-8")
        print(len(wanted), "pages prepared in", out)
    else:
        run, order = sys.argv[3], sys.argv[4]
        read = {path.stem: path.read_text(encoding="utf-8") for path in (out / run / order).rglob("*.txt")}
        pages = [json.loads(line) for line in open(out / "pages.jsonl", encoding="utf-8")]
        with open(out / f"{run}_{order}.jsonl", "w", encoding="utf-8") as scored:
            for page in pages:  # a page htrflow found no line on has no file: an empty reading
                scored.write(json.dumps({**page, "prediction": " ".join(read.get(page["id"], "").split())}, ensure_ascii=False) + "\n")
        print(len(pages), "pages;", sum(page["id"] not in read for page in pages), "without a reading;", len(read) - sum(page["id"] in read for page in pages), "readings of no page")
