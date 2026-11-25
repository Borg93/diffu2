"""Where our readings of the 1,000 test lines lose to TrOCR's: by subset, by line, by character operation."""

import json
import sys
from collections import Counter, defaultdict

from diffu_htr_train.evaluate.cer import edits, whitespace

ours_path, trocr_path = sys.argv[1], sys.argv[2]
ours = {r["id"]: r for r in map(json.loads, open(ours_path, encoding="utf-8"))}
trocr = {r["id"]: r for r in map(json.loads, open(trocr_path, encoding="utf-8"))}
ids = [i for i in ours if i in trocr]
print(f"joined {len(ids)} lines")


def ops(ref: str, hyp: str) -> list[tuple[str, str]]:
    """Edit operations (ref piece, hyp piece) of a minimal alignment; '' marks an insertion or deletion."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    out, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                out.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            out.append((ref[i - 1], ""))
            i -= 1
        else:
            out.append(("", hyp[j - 1]))
            j -= 1
    return out


by_subset: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # lines, ref chars, our edits, trocr edits
deltas = []
our_ops: Counter = Counter()
tr_ops: Counter = Counter()
for i in ids:
    ref = whitespace(ours[i]["text"])
    a, b = whitespace(ours[i]["prediction"]), whitespace(trocr[i]["prediction"])
    ea, eb = edits(ref, a), edits(ref, b)
    s = by_subset[ours[i]["subset"]]
    s[0] += 1
    s[1] += len(ref)
    s[2] += ea
    s[3] += eb
    deltas.append((ea - eb, i, ref, a, b))
    our_ops.update(ops(ref, a))
    tr_ops.update(ops(ref, b))

total = [sum(v[k] for v in by_subset.values()) for k in range(4)]
print(f"\ncorpus: ours {total[2] / total[1]:.2%}  trocr {total[3] / total[1]:.2%}  extra edits {total[2] - total[3]}")
print("\nby subset (lines, chars, ours, trocr, extra edits = share of the gap):")
for name, (n, c, ea, eb) in sorted(by_subset.items(), key=lambda kv: kv[1][3] - kv[1][2]):
    print(f"  {name:28} {n:4} {c:6}  {ea / c:6.2%} {eb / c:6.2%}  {ea - eb:+5}")

deltas.sort(reverse=True)
worse = sum(1 for d, *_ in deltas if d > 0)
better = sum(1 for d, *_ in deltas if d < 0)
print(f"\nlines: ours worse on {worse}, better on {better}, equal on {len(deltas) - worse - better}")
top = sum(d for d, *_ in deltas[:20])
print(f"the 20 lines where we lose most account for {top} extra edits")
for d, i, ref, a, b in deltas[:12]:
    print(f"  {d:+3} {i}\n      ref  {ref!r}\n      ours {a!r}\n      tr   {b!r}")

diff = Counter({k: our_ops[k] - tr_ops[k] for k in set(our_ops) | set(tr_ops)})
print("\noperations we make more than TrOCR (ref -> hyp, ours, trocr, extra):")
for (r, h), x in diff.most_common(25):
    print(f"  {r!r:>6} -> {h!r:6} {our_ops[(r, h)]:4} {tr_ops[(r, h)]:4} {x:+4}")
print("\noperations TrOCR makes more than us:")
for (r, h), x in sorted(diff.items(), key=lambda kv: kv[1])[:10]:
    print(f"  {r!r:>6} -> {h!r:6} {our_ops[(r, h)]:4} {tr_ops[(r, h)]:4} {x:+4}")
