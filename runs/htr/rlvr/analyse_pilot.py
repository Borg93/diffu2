"""Where a pilot's change against SFT comes from (CURATION E72): the guarded reader's page scores of two evaluations,
split by whether the page's volume gave RL pages, by kind of page, by how hard the page was, by volume; the kinds of
edit (substituted, missed, added characters) and the character pairs that changed most; the run-to-run noise.

    uv run --no-project --with rapidfuzz --with numpy python runs/htr/rlvr/analyse_pilot.py BASE NEW RUN_DIR [NOISE_A NOISE_B]
"""
import csv, json, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from rapidfuzz.distance import Levenshtein

E = Path("/mnt/scratch/gabriel/htr/rlvr/eval"); D = Path("/mnt/scratch/gabriel/htr/rlvr/data_v2")
PARTIAL = {"e6e45ddcae8c_right", "4c7c6e97ab16_left"}; READER = "hf_norepeat10"
LAYOUT = re.compile(r"<(?:x|y)_[0-9.]+>|<class_[A-Za-z-]+>")
TABLE = re.compile(r"\\begin\{tabular\}\{[^}]*\}|\\end\{tabular\}|\\linebreak\{\}|\\\\| & ")
def text(a): return unicodedata.normalize("NFC", " ".join(TABLE.sub(" ", LAYOUT.sub("\n", a)).replace("#", " ").split())).strip()
def lines(p): return [l for l in Path(p).read_text("utf-8").split("\n") if l]
def scores(run, sp): return {r["id"]: r for r in csv.DictReader(open(E / run / "scores" / f"{READER}.{sp}.tsv"), delimiter="\t") if r["id"] not in PARTIAL}
def reads(run, sp): return {json.loads(l)["id"]: text(json.loads(l)["raw"]) for l in lines(E / run / "reads" / f"{READER}.{sp}.jsonl")}
def gain(a, b, ids, key="edits"):
    if not ids: return "no pages"
    ea = np.array([int(a[i][key]) for i in ids]); eb = np.array([int(b[i][key]) for i in ids]); c = np.array([int(a[i]["chars"]) for i in ids])
    pick = np.random.default_rng(0).integers(0, len(ids), (4000, len(ids))); g = 100 * (ea[pick].sum(1) - eb[pick].sum(1)) / c[pick].sum(1)
    return f"{len(ids):4} pages  {100*ea.sum()/c.sum():6.2f} -> {100*eb.sum()/c.sum():6.2f} %  gain {100*(ea.sum()-eb.sum())/c.sum():+.2f} ({np.quantile(g,.025):+.2f} to {np.quantile(g,.975):+.2f})  better {int((eb<ea).sum())} worse {int((eb>ea).sum())}"

base, new, run = sys.argv[1], sys.argv[2], Path(sys.argv[3])
rl = [json.loads(l) for l in lines(run / "curriculum" / "rl_train.jsonl")]
rl_vol = Counter(f"{r['collection']}/{r['volume']}" for r in rl)
print(f"RL pages: {len(rl)} of {len(rl_vol)} volumes")
for sp in ("test", "monitor"):
    a, b = scores(base, sp), scores(new, sp); ids = sorted(a)
    print(f"\n=== {sp}: {base} -> {new} (ordered CER, then order-free)")
    print("  all            ", gain(a, b, ids)); print("  all, order-free", gain(a, b, ids, "edits_unordered"))
    if sp == "test":
        n = {i: rl_vol.get(a[i]["volume"], 0) for i in ids}
        for lab, part in (("volume gave no RL page ", [i for i in ids if n[i] == 0]), ("volume gave 1-2        ", [i for i in ids if 1 <= n[i] <= 2]), ("volume gave 3 or more  ", [i for i in ids if n[i] >= 3])):
            print(f"  {lab}", gain(a, b, part), "| order-free", gain(a, b, part, "edits_unordered")[11:])
    print("  -- by kind of page")
    for k in ("list", "table", "dark", "-"):
        print(f"  {k:6}", gain(a, b, [i for i in ids if k in a[i]["kinds"].split(",")]))
    print("  -- by how hard the page was for SFT (its CER)")
    cer = {i: int(a[i]["edits"]) / max(1, int(a[i]["chars"])) for i in ids}
    for lo, hi in ((0, .02), (.02, .05), (.05, .10), (.10, .20), (.20, 9)):
        print(f"  {lo:.2f}-{hi:.2f}", gain(a, b, [i for i in ids if lo <= cer[i] < hi]))
    print("  -- by page length (truth characters)")
    for lo, hi in ((0, 400), (400, 900), (900, 1500), (1500, 99999)):
        print(f"  {lo:5}-{hi:<5}", gain(a, b, [i for i in ids if lo <= int(a[i]["chars"]) < hi]))
    print("  -- by volume (those with 8 pages or more)")
    by = defaultdict(list)
    for i in ids: by[a[i]["volume"]].append(i)
    for v, part in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(part) >= 8: print(f"  {v.split('/')[-1][:38]:38} RL pages {rl_vol.get(v,0):3}", gain(a, b, part))
    truth = {json.loads(l)["id"]: text(json.loads(l)["target"]) for l in lines(D / f"{sp}.jsonl")}
    ra, rb = reads(base, sp), reads(new, sp)
    kinds = {}; subs = {}
    for name, rd in ((base, ra), (new, rb)):
        k = Counter(); s = Counter()
        for i in ids:
            for op in Levenshtein.editops(truth[i], rd[i]):
                k[op.tag] += 1
                if op.tag == "replace": s[(truth[i][op.src_pos], rd[i][op.dest_pos])] += 1
        kinds[name], subs[name] = k, s
    tot = sum(len(truth[i]) for i in ids)
    print("  -- kinds of edit, per 100 truth characters (replace = wrong character, delete = missed, insert = added)")
    for name in (base, new): print(f"  {name[:26]:26}", {t: round(100 * kinds[name][t] / tot, 2) for t in ("replace", "delete", "insert")})
    diff = {p: subs[new][p] - subs[base][p] for p in set(subs[base]) | set(subs[new])}
    print("  substitutions (truth>read) fewer after RL:", [(f"{p[0]}>{p[1]}", d) for p, d in sorted(diff.items(), key=lambda x: x[1])[:10]])
    print("  substitutions more after RL:              ", [(f"{p[0]}>{p[1]}", d) for p, d in sorted(diff.items(), key=lambda x: -x[1])[:10]])
if len(sys.argv) > 5:
    print(f"\n=== run-to-run noise: {sys.argv[4]} against {sys.argv[5]} (the same settings and update count, two runs)")
    for sp in ("test", "monitor"):
        a, b = scores(sys.argv[4], sp), scores(sys.argv[5], sp); print(f"  {sp:8}", gain(a, b, sorted(a)), "| order-free", gain(a, b, sorted(a), "edits_unordered")[11:])
