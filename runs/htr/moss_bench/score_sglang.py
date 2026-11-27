"""The read-back of the SGLang runs (27 Nov): each run's 12 pages read line by line by diffu-generate's own TrOCR judge
against the plan's text, as diffu-generate scores its crops, so the CER is comparable with our runs' summary.json.

The captured pages are the first 12 crops of set40 in diffu-generate's order (capture_generate.py numbered them as it
drew), so page_K is crop K of the planned jobs. Our runs R (bf16) and F (MXFP8) are scored again on the same 12 pages
as the like-for-like baseline. Also the PSNR of each run's pages to SGLang exact's. Writes scores.json beside the runs
and prints one line per run.

    python score_sglang.py [cpu|cuda] [run,run,...]
"""

import json
import sys
from pathlib import Path
from statistics import fmean

import numpy as np
from PIL import Image

from diffu_page.cli.generate import page_groups, plan_groups
from diffu_page.draw.jobs import read_page_jobs
from diffu_page.quality.judge import TrOCRJudge

B = Path("/mnt/scratch/gabriel/moss_bench")
O = B / "set40/out"
PAGES = 12


def psnr(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("L"), dtype=np.float32)
    y = np.asarray(b.convert("L").resize(a.size), dtype=np.float32)
    return float(10 * np.log10(255**2 / max(((x - y) ** 2).mean(), 1e-6)))


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    pages = read_page_jobs(B / "set40/jobs.jsonl", presets={}, loras={})
    jobs = [job for _, job in plan_groups(pages, page_groups(pages))[0]][:PAGES]
    judge = TrOCRJudge(device=device)
    runs = {"ours_R": [O / f"R/{j.name}.jpg" for j in jobs], "ours_F": [O / f"F/{j.name}.jpg" for j in jobs]}
    for d in sorted(O.glob("sglang_*/")):
        files = [d / f"page_{k}.jpg" for k in range(PAGES)]
        if all(f.exists() for f in files):
            runs[d.name] = files
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None  # run names, in this order
    if only:
        runs = {name: runs[name] for name in only if name in runs}
    exact = [Image.open(O / f"sglang_exact/page_{k}.jpg") for k in range(PAGES)]
    ours = [Image.open(f) for f in runs["ours_R"]]  # the runs drawn from our noise seeds (page_seeds) compare with these
    scores = {}
    for name, files in runs.items():
        cers, closeness, to_ours = [], [], []
        for job, f, ref, mine in zip(jobs, files, exact, ours, strict=True):
            image = Image.open(f).convert("RGB")  # the line boxes are on 0-1000 of the page: any size reads
            cers.append(fmean(v.cer for v in judge(image, job.plan.lines)))
            closeness.append(psnr(ref, image))
            to_ours.append(psnr(mine, image))
        scores[name] = {"cer": fmean(cers), "psnr_to_sglang_exact": fmean(closeness), "psnr_to_ours_R": fmean(to_ours),
                        "page_cers": cers}
        print(f"{name:32s} CER {scores[name]['cer']:.4f}  PSNR to SGLang exact {fmean(closeness):.1f} dB"
              f"  to ours R {fmean(to_ours):.1f} dB", flush=True)
    (O / "scores.json").write_text(json.dumps(scores, indent=1))


if __name__ == "__main__":
    main()
