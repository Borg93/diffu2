#!/bin/bash
# The runner's GPU smoke test on the DGX Spark: two held-out pages of the eight-hands page pairs (two hands), each its
# own template side and running text, drawn whole at 1 MP in the sample form from its pair's sample of the same hand,
# by diffu-run on a local Ray cluster (one actor on the GB10) into runs/runner_smoke/pages.lance, then exported as
# files to look at beside the real pages (data_flux3/page_pairs_eight/pages/<id>.jpg). The model is a MOSS-Image
# model directory that takes a sample image (default: the eight-hands full fine-tune's step 1500). One GPU job,
# through jobs/spark_guard.sh: ~35 GB for the pipeline and the judge, and Ray's own processes.
#
#   jobs/moss/runner_smoke.sh
#   MODEL=runs/moss/<another model> OUT=runs/runner_smoke_<name> jobs/moss/runner_smoke.sh
#
# Run again into the same OUT, it resumes: the pages drawn are skipped, a failed one is drawn again.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PAIRS="${PAIRS:-data_flux3/page_pairs_eight}"
MODEL="${MODEL:-runs/moss/eight_full_r5_step_001500}"
OUT="${OUT:-runs/runner_smoke}"
PAGES="${PAGES:-003dd088c8cf_right 52989ccbc60a_right}"  # held-out pages of Jämtlands domsaga and Påhlman's journals
STEPS="${STEPS:-50}"
export CUDA_VISIBLE_DEVICES=0

[ -f "$MODEL/modular_model_index.json" ] || { echo "no model directory at $MODEL"; exit 1; }
[ -f "$PAIRS/heldout_sample.jsonl" ] || { echo "no page pairs at $PAIRS"; exit 1; }
mkdir -p "$OUT"

# the run file: each page job's template and side, its own running text, its style and its pair's sample
.venv/bin/python - "$PAIRS" "$OUT/run.jsonl" $PAGES <<'EOF'
import json
import sys
from pathlib import Path

from diffu_page.captions.look import scan_look
from diffu_page.captions.provenance import provenance_of
from diffu_page.captions.restyle import ReferenceCaption
from diffu_page.draw.jobs import PageJob
from diffu_page.draw.plan import PageStyle, Reference
from diffu_page.output import load_rgb
from diffu_page.page.layout import load_xml_layout, running_text, scan_pages

pairs, out, page_ids = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
gt_root = Path(json.loads((pairs / "build.json").read_text())["scan"]["gt_root"])
held_out = {row["id"]: row for row in map(json.loads, (pairs / "heldout_sample.jsonl").read_text().splitlines())}
lines = []
for page_id in page_ids:
    pair = held_out[page_id]
    scan, side = page_id.rsplit("_", 1)
    template = gt_root / json.loads((pairs / "scanned" / f"{scan}.json").read_text())["xml"]
    [single] = [scan_page for scan_page in scan_pages(load_xml_layout(template)) if scan_page.side == side]
    style = PageStyle(
        provenance=provenance_of(Path(pair["volume"]).parts[0], template),
        look=scan_look(load_rgb(pairs / pair["target"])),
    )
    sample = Reference(image=(pairs / pair["reference"]).resolve(), caption=ReferenceCaption.SAMPLE)
    job = PageJob(id=page_id, template=template, side=side, text=running_text(single.layout), style=style,
                  reference=sample, seed=42)
    lines.append(job.model_dump_json())
out.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
print(f"{len(lines)} pages -> {out}")
EOF

echo "{\"steps\": $STEPS, \"device\": \"cuda\"}" > "$OUT/draw.json"

jobs/spark_guard.sh --need-gb 50 --name runner-smoke --log "$OUT/guard.log" -- \
    .venv/bin/diffu-run --jobs "$OUT/run.jsonl" --draw-settings "$OUT/draw.json" --model "$MODEL" --out "$OUT" \
    --gpus 1 --actors-per-gpu 1 --commit-every 1

.venv/bin/diffu-run-export --out "$OUT" --to "$OUT/files"
echo "look at $OUT/files/<id>.jpg beside $PAIRS/pages/<id>.jpg, and its sample $PAIRS/refs/"
