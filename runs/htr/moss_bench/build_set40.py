"""Build set40: 40 pages (20 two-page openings) whose ALTO templates are clean, for judging diffu-generate's speed
options on pages a reader can read back.

Picks scans from runs/htr/curate/train_pages.lance whose two pages are Swedish handwritten running text (no table),
read by our reader to the end (read_finish 'stop', not looped, not turned), 20 to 40 lines and over 600 characters each,
with the old label (the ALTO file's text) within MAX_DISAGREEMENT of our reader's text; the scan an opening with both
pages written (scans.lance's layout). One scan per volume, the volumes spread over the pools' clusters. Each picked ALTO
file is then planned on CPU exactly as `diffu-generate --whole-page --page-megapixels 2` plans it (no model is loaded),
and kept only if it plans to its two pages, one whole-page crop each, whose line counts are near the labels'.

Writes under OUT: templates.txt (the ALTO paths), pages.tsv (one row a page, with its sample), draw.sh (the set drawn
with a settings file $1 into $2, one diffu-generate call per sample group; needs the GPU, not run here), plan.tsv (the
planning result per page).

    CUDA_VISIBLE_DEVICES= uv run python runs/htr/moss_bench/build_set40.py --exclude A0068544 C0100842 C0025481 A0074145

--exclude drops scans or volumes found unclean by eye (the four above, 27 Nov: a name list and a faint signed page;
faint pasted sheets with struck lines; a loose slip laid over the page; lines running into the gutter, their ends
lost from the ALTO text); the pick is otherwise deterministic.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import lance
import pandas as pd

TRAIN = Path("runs/htr/curate/train_pages.lance")
SCANS = Path("runs/htr/curate/scans.lance")
HTR = Path("/mnt/scratch/gabriel/htr")
OUT = Path("/mnt/scratch/gabriel/moss_bench/set40")
SAMPLES = Path("/mnt/scratch/gabriel/moss_bench/pagedemo/samples")
GROUPS = {  # group name: the sample its scans are drawn in (four hands, 15th to 18th century)
    "gota1701": SAMPLES / "gota-hovratt-1701.jpg",
    "jonkoping1590": SAMPLES / "jonkopings-radhusratt-1584-1595.jpg",
    "deputation1765": SAMPLES / "standernas-deputation-1765.jpg",
    "nikodemus1400s": SAMPLES / "nikodemusevangeliet-1400s.jpg",
}
MODEL = "/mnt/scratch/gabriel/models/moss/moss-diffu-moss-above"
N_SCANS = 20
MAX_DISAGREEMENT = 0.05
PAGE_PIXELS = 2_000_000  # --page-megapixels 2
TITLE = "Swedish archival records"  # required by diffu-generate; the sample caption names no archive
COLUMNS = [
    "page", "set", "pool", "volume", "scan", "side", "page_image", "label_chars", "label_lines", "read_finish",
    "read_disagreement", "read_looped", "language", "kind", "is_table", "page_turn", "cut_class",
]  # fmt: skip


def alto_path(row: pd.Series) -> Path:
    return HTR / row["set"] / row["volume"] / "alto" / f"{row['scan']}.xml"


def candidates() -> tuple[pd.DataFrame, list[str]]:
    """The pages of every scan whose two pages pass, and the counts at each filter."""
    t = lance.dataset(TRAIN).to_table(columns=COLUMNS).to_pandas()
    log = [f"pages in train_pages.lance: {len(t)}"]
    keep = pd.Series(True, index=t.index)
    for name, test in [
        ("language swedish", t.language.eq("swedish")),
        ("kind handwritten_text", t.kind.eq("handwritten_text")),
        ("not a table", ~t.is_table.fillna(True)),
        (f"read_disagreement < {MAX_DISAGREEMENT}", t.read_disagreement < MAX_DISAGREEMENT),
        ("read_finish stop", t.read_finish.eq("stop")),
        ("not looped", ~t.read_looped.fillna(True)),
        ("not turned", t.page_turn.eq(0)),
        ("20-40 label lines", t.label_lines.between(20, 40)),
        ("label_chars > 600", t.label_chars > 600),
        ("cut_class cut", t.cut_class.eq("cut")),
    ]:
        keep &= test
        log.append(f"  + {name}: {int(keep.sum())} pages")
    t["ok"] = keep
    per_scan = t.groupby("scan").agg(pages=("page", "size"), ok=("ok", "sum"))
    both = per_scan.index[(per_scan.pages == 2) & (per_scan.ok == 2)]
    t = t[t.scan.isin(both)]
    log.append(f"scans with both pages passing: {len(both)} ({t.volume.nunique()} volumes)")
    s = lance.dataset(SCANS).to_table(columns=["scan", "layout", "pool_cluster_name"]).to_pandas()
    t = t.merge(s, on="scan", how="left")
    t = t[t.layout.eq("opening_both_pages_written")]
    log.append(f"  + scan layout opening_both_pages_written: {t.scan.nunique()} scans ({t.volume.nunique()} volumes)")
    t = t.assign(alto=t.apply(alto_path, axis=1))
    t = t[t.alto.map(Path.is_file)]
    log.append(
        f"  + ALTO file at <set>/<volume>/alto/<scan>.xml: {t.scan.nunique()} scans ({t.volume.nunique()} volumes)"
    )
    return t, log


def shortlist(t: pd.DataFrame, exclude: set[str], rng: random.Random) -> list[str]:
    """Scans in pick order: per volume its scan with the lowest worse-page disagreement; volumes taken round-robin
    over the pool clusters (each cluster's volumes in random order), so the set spreads over kinds of books. Excluded
    scans and volumes are skipped after the order is drawn, so excluding one moves no other pick."""
    worst = t.groupby("scan").agg(
        d=("read_disagreement", "max"), volume=("volume", "first"), cluster=("pool_cluster_name", "first")
    )
    best = worst.sort_values("d").groupby("volume").head(1)
    by_cluster: dict[str, list[str]] = {}
    for cluster, part in best.groupby("cluster"):
        scans = part.index.tolist()
        rng.shuffle(scans)
        by_cluster[str(cluster)] = scans
    order: list[str] = []
    clusters = sorted(by_cluster)
    rng.shuffle(clusters)
    while any(by_cluster.values()):
        for cluster in clusters:
            if by_cluster[cluster]:
                order.append(by_cluster[cluster].pop())
    return [scan for scan in order if scan not in exclude and worst.volume[scan] not in exclude]


def plan_group(altos: list[Path], sample: Path) -> dict[Path, list[dict]]:
    """The templates planned as one diffu-generate call with --templates and --text-from each of ``altos``,
    --reference ``sample`` --reference-caption sample --whole-page --page-megapixels 2 --seed 7 plans them (the
    command's own code, CPU, no model loaded): per template, per page its side, crop count, planned lines and crop
    size, or {"error": ...} for a page that cannot be planned."""
    from diffu_page.captions.look import Ink, Paper, ScanLook
    from diffu_page.captions.prompt import PageStyle
    from diffu_page.captions.provenance import Provenance
    from diffu_page.captions.restyle import ReferenceCaption
    from diffu_page.cli.generate import page_groups, page_reference, page_text, plan_groups
    from diffu_page.draw.jobs import PageRequest, ResolvedPage, single_pages

    style = PageStyle(
        provenance=Provenance(title=TITLE, years=""),
        look=ScanLook(greyscale=False, paper=Paper.CREAM, ink=Ink.DARK_BROWN),
    )
    text = " ".join(page_text(path) for path in altos)
    reference = page_reference(sample, ReferenceCaption.SAMPLE)
    requests = [
        PageRequest(template=a, text=text, style=style, seed=7, reference=reference, page_pixels=PAGE_PIXELS)
        for a in altos
    ]
    resolved = [r.resolve({}, {}) for r in requests]
    pages = [ResolvedPage(page=single, loras=p.loras) for p in resolved for single in single_pages(p.page)]
    jobs, failed = plan_groups(pages, page_groups(pages))
    out: dict[Path, list[dict]] = {a: [] for a in altos}
    for index, rp in enumerate(pages):
        name = rp.page.name_at(index)
        side = str(rp.page.side.value) if rp.page.side else "none"
        mine = [job for _, job in jobs if job.name.startswith(name)]
        if name in failed:
            out[rp.page.template].append({"side": side, "error": failed[name]})
            continue
        out[rp.page.template].append(
            {
                "side": side,
                "crops": len(mine),
                "plan_lines": sum(len(job.plan.lines) for job in mine),
                "size": f"{mine[0].plan.width}x{mine[0].plan.height}" if mine else "",
            }
        )
    return out


def plan(alto: Path) -> list[dict]:
    """The template planned alone with its own text (the screening pass); raises if a page cannot be planned."""
    planned = plan_group([alto], GROUPS["gota1701"])[alto]
    errors = [p["error"] for p in planned if "error" in p]
    if errors:
        raise ValueError("; ".join(errors))
    return planned


def write_own_text_jobs(sel: pd.DataFrame) -> list[str]:
    """jobs.jsonl and draw_jobs.sh: the same 40 pages in one diffu-generate call by a job file, each page written with
    its own text (its side of the ALTO file, split as the planner splits it) in its group's sample, at 2 megapixels
    (a job line's page_pixels is --whole-page's) and seed 7, named and seeded by <scan>_<side>. draw.sh's calls write
    every page of a group with the joined text from its start, so its 40 pages hold only 4 texts. Planned here from
    the job file as diffu-generate reads it; returns the log lines."""
    import json

    from diffu_page.cli.generate import page_groups, plan_groups
    from diffu_page.draw.jobs import read_page_jobs
    from diffu_page.page.layout import load_xml_layout, scan_pages
    from diffu_page.page.layout.text_refill import running_text

    style = {
        "provenance": {"title": TITLE, "years": ""},
        "look": {"greyscale": False, "paper": "cream", "ink": "dark brown"},
    }
    rows = []
    for row in sel.itertuples():
        page = {str(p.side.value): p for p in scan_pages(load_xml_layout(row.alto)) if p.side is not None}[row.side]
        rows.append(
            {
                "template": str(row.alto),
                "side": row.side,
                "text": running_text(page.layout),
                "style": style,
                "seed": 7,
                "reference": {"image": row.sample, "caption": "sample"},
                "page_pixels": PAGE_PIXELS,
                "id": f"{row.scan}_{row.side}",
            }
        )
    path = OUT / "jobs.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    pages = read_page_jobs(path, presets={}, loras={})
    jobs, failed = plan_groups(pages, page_groups(pages))
    first = {job.name: job.plan.lines[job.plan.reading_order[0]].text for _, job in jobs}
    script = [
        "#!/bin/bash",
        "# set40 drawn by its job file, every page with its own text: draw_jobs.sh <settings.json> <out dir>.",
        "# Built by runs/htr/moss_bench/build_set40.py. Needs the GPU (CUDA_VISIBLE_DEVICES set by the caller).",
        'set -u; [ $# -eq 2 ] || { echo "usage: $0 <settings.json> <out dir>"; exit 2; }',
        "cd /home/gabriel/Desktop/diffu2 || exit 1",
        f'uv run diffu-generate --model {MODEL} --jobs {path} --draw-settings "$1" --out "$2"',
    ]
    (OUT / "draw_jobs.sh").write_text("\n".join(script) + "\n", encoding="utf-8")
    (OUT / "draw_jobs.sh").chmod(0o755)
    return [
        f"jobs.jsonl: {len(pages)} pages planned to {len(jobs)} crops, {len(failed)} failed {dict(failed)}",
        f"  distinct first lines among the planned pages: {len(set(first.values()))}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--exclude", nargs="*", default=[], help="scans or volumes to leave out")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    t, log = candidates()
    order = shortlist(t, set(args.exclude), random.Random(args.seed))
    picked: list[str] = []
    plans: dict[str, list[dict]] = {}
    plan_log: list[str] = []
    for scan in order:
        if len(picked) == N_SCANS:
            break
        rows = t[t.scan == scan].set_index("side")
        alto = rows.alto.iloc[0]
        try:
            planned = plan(alto)
        except Exception as error:  # noqa: BLE001 - a template that fails is reported and dropped
            plan_log.append(f"FAIL {scan}: {type(error).__name__}: {str(error)[:200]}")
            continue
        sides = {p["side"]: p for p in planned}
        if set(sides) != {"left", "right"} or any(p["crops"] != 1 for p in planned):
            plan_log.append(f"DROP {scan}: planned {planned}")
            continue
        off = [s for s in sides if not 0.8 <= sides[s]["plan_lines"] / rows.loc[s, "label_lines"] <= 1.25]
        if off:
            plan_log.append(
                f"DROP {scan}: planned lines {[(s, sides[s]['plan_lines']) for s in sides]} "
                f"against labels {rows.label_lines.to_dict()}"
            )
            continue
        picked.append(scan)
        plans[scan] = planned
    sel = t[t.scan.isin(picked)].copy()
    sel["order"] = sel.scan.map({s: i for i, s in enumerate(picked)})
    sel = sel.sort_values(["order", "side"])
    names = list(GROUPS)
    sel["group"] = sel.order.map(lambda i: names[i % len(names)])
    sel["sample"] = sel.group.map(lambda g: str(GROUPS[g]))
    log.append(
        f"picked {len(picked)} scans ({sel.volume.nunique()} volumes, {len(sel)} pages); "
        f"{len(plan_log)} shortlisted scans dropped at planning"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    alto_by_scan = sel.groupby("scan", sort=False).alto.first()
    (OUT / "templates.txt").write_text("".join(f"{alto_by_scan[s]}\n" for s in picked), encoding="utf-8")
    cols = ["scan", "volume", "side", "label_lines", "label_chars", "read_disagreement", "page_image", "alto", "group",
            "sample", "set", "pool_cluster_name"]  # fmt: skip
    sel[cols].rename(columns={"label_lines": "lines"}).to_csv(OUT / "pages.tsv", sep="\t", index=False)
    plan_rows = []  # the final check: each group planned as draw.sh's call plans it (joined text, its sample)
    for group in names:
        part = sel[sel.group == group].groupby("scan", sort=False).alto.first()
        for alto, planned in plan_group(list(part), GROUPS[group]).items():
            scan = alto.stem
            for p in planned:
                label = sel[(sel.scan == scan) & (sel.side == p["side"])].label_lines
                plan_rows.append(
                    f"{group}\t{scan}\t{p['side']}\t{p.get('crops', 0)}\t{p.get('plan_lines', 0)}\t"
                    f"{int(label.iloc[0]) if len(label) else ''}\t{p.get('size', '')}\t{p.get('error', '')}"
                )
    (OUT / "plan.tsv").write_text(
        "group\tscan\tside\tcrops\tplanned_lines\tlabel_lines\tcrop_size\terror\n" + "\n".join(plan_rows) + "\n",
        encoding="utf-8",
    )
    print("\n".join(plan_rows))
    lines = [
        "#!/bin/bash",
        "# set40 drawn with one settings file: draw.sh <settings.json> <out dir>; one diffu-generate call per sample",
        "# group, each writing to <out dir>/<group>. Built by runs/htr/moss_bench/build_set40.py. Needs the GPU",
        "# (CUDA_VISIBLE_DEVICES set by the caller). Every template of a call is written with the text of all of the",
        "# call's --text-from files joined (diffu-generate's rule), as in moss_bench_gpu2b.sh.",
        'set -u; [ $# -eq 2 ] || { echo "usage: $0 <settings.json> <out dir>"; exit 2; }',
        "cd /home/gabriel/Desktop/diffu2 || exit 1",
        f"M={MODEL}",
        "status=0",
    ]
    for group in names:
        part = sel[sel.group == group].groupby("scan", sort=False).alto.first()
        if part.empty:
            continue
        templates = " ".join(f"--templates {a}" for a in part)
        texts = " ".join(f"--text-from {a}" for a in part)
        lines += [
            f"uv run diffu-generate --model $M {templates} \\",
            f'    {texts} --title "{TITLE}" \\',
            f"    --reference {GROUPS[group]} --reference-caption sample --whole-page --page-megapixels 2 \\",
            f'    --seed 7 --draw-settings "$1" --out "$2/{group}" || status=1',
        ]
    lines.append("exit $status")
    (OUT / "draw.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "draw.sh").chmod(0o755)
    own = write_own_text_jobs(sel)
    print("\n".join(own))
    log += own
    (OUT / "build.log").write_text("\n".join(log + plan_log) + "\n", encoding="utf-8")
    print("\n".join(log + plan_log))
    print(sel[["scan", "side", "volume", "set", "label_lines", "label_chars", "read_disagreement", "group",
               "pool_cluster_name"]].to_string(index=False))  # fmt: skip
    for s in picked:
        print(s, plans[s])


if __name__ == "__main__":
    main()
