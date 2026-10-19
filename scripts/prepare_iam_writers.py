"""Source-specific recipe: Teklia/IAM-line with REAL IAM writer ids, as a ``--data-dir`` manifest.

``Teklia/IAM-line`` (the Aachen split: 6482 / 976 / 2915 lines) has only ``image`` + ``text``, so under
``--hf-dataset`` every style reference is a random other writer and ``--style-canvas`` can't learn to copy
a hand. DiffusionPen's preprocessed IAM *words* (``konnik/DiffusionPen`` -> ``saved_iam_data/*.pt``) carry
each word's IAM path (``.../c04-110-01-00``: form-line-word) and its writer (via IAM's forms.txt), so they
give a form -> writer map. This recipe matches each Teklia line to its IAM form by transcription and
writes ``out_dir/lines/*.png`` + ``{train,val,test}.jsonl`` with ``{image, text, writer_id, form_id}``.

Matching (Teklia's rows are sorted by IAM form id, the lines shuffled within each form):
  1. exact match of the alphanumeric-only, lower-cased text against the IAM line built from its words.
     IAM reuses source texts across writers (a01-003 / a01-003u / a01-003x), and DiffusionPen drops
     mis-segmented words, so a line can match only its twin from another writer: keep the largest set of
     exact matches whose form ids never decrease down the rows; the rest are overruled;
  2. every other row sits between two kept matches: same form on both sides -> that form, else the best
     fuzzy match among the lines of the forms in between.
Lines still unresolved are dropped (counted in the report).

    hf download konnik/DiffusionPen --include "saved_iam_data/*" --local-dir data_raw/diffusionpen
    uv run --extra hf python scripts/prepare_iam_writers.py --out-dir data_iam
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import os
import re
import urllib.request

# DiffusionPen indexes writers per split; these map the index back to the IAM writer name (e.g. "049").
_WRITER_DICT_URL = "https://raw.githubusercontent.com/koninik/DiffusionPen/main/writers_dict_{}.json"
# Teklia split -> the DiffusionPen word file holding its forms (DiffusionPen's train = Aachen train + val).
_DP_SPLIT = {"train": "train", "validation": "train", "test": "test"}
_OUT_NAME = {"train": "train", "validation": "val", "test": "test"}


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", s.lower())


def load_dp_lines(pt_dir: str) -> dict[str, dict[str, tuple[str, str]]]:
    """DiffusionPen split -> {IAM line id: (normalized line text, writer)} from its word tuples."""
    import PIL.Image
    import torch

    out: dict[str, dict[str, tuple[str, str]]] = {}
    for split in ("train", "test"):
        with urllib.request.urlopen(_WRITER_DICT_URL.format(split)) as r:
            inv = {v: k for k, v in json.load(r).items()}
        # The .pt is a pickle of (PIL word, text, writer index, IAM path) tuples. weights_only with only
        # PIL.Image.Image allowed: no other global can run, so the untrusted pickle can't execute code.
        with torch.serialization.safe_globals([PIL.Image.Image]):
            words = torch.load(os.path.join(pt_dir, f"{split}_word_IAM.pt"), weights_only=True)
        by_line: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
        writer: dict[str, str] = {}
        for _, text, w, path in words:
            line_id, idx = path.rsplit("/", 1)[1].rsplit("-", 1)
            by_line[line_id].append((int(idx), text))
            writer[line_id] = inv[w]
        del words
        out[split] = {
            lid: (_norm("".join(t for _, t in sorted(ws))), writer[lid]) for lid, ws in by_line.items()
        }
        print(
            f"DiffusionPen {split}: {len(out[split])} lines, {len(set(writer.values()))} writers", flush=True
        )
    return out


def _monotone_chain(cands: list[list[int]], n_ranks: int) -> list[int | None]:
    """One candidate rank per row (or None) so the picks never decrease down the rows, picking as many rows
    as possible: a longest non-decreasing subsequence over candidate sets (Fenwick prefix-max)."""
    tree = [(0, -1)] * (n_ranks + 1)  # 1-based: (chain length, node) of the best chain ending at rank <= r
    nodes: list[tuple[int, int, int]] = []  # (row, rank, previous node)

    def query(r: int) -> tuple[int, int]:
        best, r = (0, -1), r + 1
        while r > 0:
            best, r = max(best, tree[r]), r - (r & -r)
        return best

    for i, ranks in enumerate(cands):
        for r in sorted(ranks, reverse=True):  # descending: a row can't chain onto its own other candidate
            length, prev = query(r)
            nodes.append((i, r, prev))
            k, val = r + 1, (length + 1, len(nodes) - 1)
            while k <= n_ranks:
                tree[k], k = max(tree[k], val), k + (k & -k)
    picked: list[int | None] = [None] * len(cands)
    node = query(n_ranks - 1)[1]
    while node >= 0:
        i, r, node = nodes[node]
        picked[i] = r
    return picked


def match_forms(texts: list[str], iam: dict[str, tuple[str, str]]) -> tuple[list[str | None], list[str]]:
    """Each Teklia row's IAM form id (or None), and the step that resolved it."""
    form = lambda lid: lid.rsplit("-", 1)[0]  # noqa: E731
    forms_sorted = sorted({form(lid) for lid in iam})
    rank = {f: r for r, f in enumerate(forms_sorted)}
    by_key: dict[str, set[str]] = collections.defaultdict(set)
    lines_of: dict[str, list[str]] = collections.defaultdict(list)
    for lid, (key, _) in iam.items():
        by_key[key].add(form(lid))
        lines_of[form(lid)].append(lid)

    keys = [_norm(t) for t in texts]
    cands = [[rank[f] for f in by_key.get(k, ())] for k in keys]
    picked = _monotone_chain(cands, len(forms_sorted))
    forms: list[str | None] = [None if r is None else forms_sorted[r] for r in picked]
    how = ["exact" if r is not None else "" for r in picked]

    prev_rank: list[int | None] = []
    last = None
    for r in picked:
        last = r if r is not None else last
        prev_rank.append(last)
    next_rank: list[int | None] = [None] * len(picked)
    last = None
    for i in range(len(picked) - 1, -1, -1):
        last = picked[i] if picked[i] is not None else last
        next_rank[i] = last
    for i, r in enumerate(picked):
        if r is not None:
            continue
        lo, hi = prev_rank[i], next_rank[i]
        lo = hi if lo is None else lo
        hi = lo if hi is None else hi
        if lo is None or hi is None:  # no exact match anywhere in the split
            continue
        if lo == hi:  # rows are sorted by form: between two rows of one form means that form
            forms[i], how[i] = forms_sorted[lo], "inside-run"
            continue
        pool = [lid for f in forms_sorted[lo : hi + 1] for lid in lines_of[f]]
        scored = [(difflib.SequenceMatcher(None, keys[i], iam[lid][0]).ratio(), lid) for lid in pool]
        score, lid = max(scored, default=(0.0, None))
        if lid is not None and score >= 0.6:
            forms[i], how[i] = form(lid), "fuzzy"
    for i, c in enumerate(cands):
        if c and picked[i] is None:
            how[i] = f"{how[i] or 'unresolved'} (exact overruled)"
    return forms, how


def main() -> None:
    ap = argparse.ArgumentParser(description="Teklia/IAM-line + IAM writer ids -> a --data-dir manifest.")
    ap.add_argument(
        "--pt-dir", default="data_raw/diffusionpen/saved_iam_data", help="DiffusionPen *_word_IAM.pt"
    )
    ap.add_argument("--out-dir", default="data_iam")
    ap.add_argument("--dry-run", action="store_true", help="report the matching, write nothing")
    args = ap.parse_args()

    from datasets import load_dataset

    iam = load_dp_lines(args.pt_dir)
    form_writer = {lid.rsplit("-", 1)[0]: w for split in iam.values() for lid, (_, w) in split.items()}
    ds = load_dataset("Teklia/IAM-line")
    os.makedirs(os.path.join(args.out_dir, "lines"), exist_ok=True)
    for split, name in _OUT_NAME.items():
        forms, how = match_forms(ds[split]["text"], iam[_DP_SPLIT[split]])
        split_forms = {f for f in forms if f}
        print(
            f"{split}: {len(forms)} lines -> {dict(collections.Counter(h or 'unresolved' for h in how))}; "
            f"{len(split_forms)} forms, {len({form_writer[f] for f in split_forms})} writers",
            flush=True,
        )
        if args.dry_run:
            continue
        written = 0
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as mf:
            for i, (row, f) in enumerate(zip(ds[split], forms, strict=True)):
                text = str(row["text"]).strip()
                if f is None or not text:
                    continue
                dst = os.path.join(args.out_dir, "lines", f"{name}_{i:05d}.png")
                row["image"].convert("RGB").save(dst)
                rec = {"image": dst, "text": text, "writer_id": f"iam{form_writer[f]}", "form_id": f}
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
        print(f"  wrote {written} rows -> {os.path.join(args.out_dir, name + '.jsonl')}", flush=True)


if __name__ == "__main__":
    main()
