"""The pool scans the classifiers are least sure of, in groups for the marking gallery (CURATION E63).

    uv run python runs/htr/curate/prototypes/unsure_groups.py OUT.parquet NAMES.json

Scans nobody has marked yet, each group up to 240 scans drawn at random, the guess the classifier's own answer:
print in a training set, cut or not on the edge, odd layouts, tables on the edge, and handwriting the kind head doubts.
"""
import json, sys
import lance, pandas as pd

HOME = "runs/htr/curate"
marked = set(lance.dataset(f"{HOME}/cut_marks.lance").to_table(columns=["image"]).column("image").to_pylist())
t = lance.dataset(f"{HOME}/scans.lance").to_table(columns=["image", "pool", "scan", "cut_class", "cut_p", "is_table", "table_p", "kind", "kind_p", "fold_found"],
                                                 filter="pool != 'gt' AND cut_class IS NOT NULL AND lines > 0").to_pandas()
t = t[~t.image.isin(marked)]
in_sets = set().union(*(pd.read_parquet(f"/mnt/scratch/gabriel/htr/{name}/scans.parquet", columns=["scan"]).scan for name in ("pool_v2", "pool_v3")))
taken = t.scan.isin(in_sets)
GROUPS = [  # name, rows, the guess shown
    ("Called print or type, in a training set", taken & (t.kind == "printed_or_typed_text"), "print"),
    ("In a training set, the cut least sure (cut_p under 0.65)", taken & (t.cut_p < 0.65), "cut"),
    ("Called odd layout, sure (p of cut under 0.2)", (t.cut_class == "odd") & (t.cut_p < 0.2), "odd"),
    ("Called odd layout, close to cut (p of cut 0.3 to 0.5)", (t.cut_class == "odd") & t.cut_p.between(0.3, 0.5), "odd"),
    ("Called do not cut, close to cut", (t.cut_class == "no_cut") & t.cut_p.between(0.25, 0.5), "no_cut"),
    ("Called single page, close to cut", (t.cut_class == "single") & t.cut_p.between(0.25, 0.5), "single"),
    ("In a training set, table or not on the edge (0.35 to 0.5)", taken & t.table_p.between(0.35, 0.5), "cut"),
    ("Called a table, on the edge (0.5 to 0.65), cut class cut", (t.cut_class == "cut") & t.table_p.between(0.5, 0.65), "cut"),
    ("In a training set, the kind head doubts handwriting (kind_p under 0.55)", taken & t.kind.str.startswith("handwritten") & (t.kind_p < 0.55), "cut"),
    ("Called not a page, close to cut", (t.cut_class == "not_a_page") & t.cut_p.between(0.2, 0.5), "not_a_page"),
]
parts, names = [], {}
for number, (name, rows, guess) in enumerate(GROUPS):
    part = t[rows]; names[number] = f"{name} ({len(part):,} in the pools)"
    part = part.sample(min(240, len(part)), random_state=0)[["image", "pool"]].assign(cluster=number, guess=guess); parts.append(part)
    print(f"{len(part):4} shown of {int(rows.sum()):9,}  {name}")
out = pd.concat(parts).drop_duplicates("image"); out.to_parquet(sys.argv[1]); json.dump(names, open(sys.argv[2], "w"))
print(len(out), "scans in", len(GROUPS), "groups")
