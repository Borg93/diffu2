#!/bin/bash
# The content-only pilot: does MOSS-Image's hand follow its sample page once no prompt names the hand? The eight-hands
# and the many-volumes fine-tunes drew a page's hand from its archive's name (three of the four prompt forms, 60 % of
# the steps, name it) and read their samples shrunk to half a page's pixels (a window's sample letters several times
# smaller than the window's own), so in the swap test a page kept its own volume's hand whatever the sample. This
# pilot trains from MOSS-Image Base, on the 55-collection 1 MP page pairs already built (data_flux3/page_pairs_many_1mp:
# 80 scans a collection, 2 windows a page, 10 % of volumes held out), less GT-RA_landskapshandlingar_20260227: the GT
# holds that collection twice (it and GT-RA_landskapshandlingar_20260220, the same scans by md5), so its rows (418 of
# the 15,745 training rows, none held out) are left out and that hand counts once. With three changes, each off by
# default:
#   - the content mix (diffu-moss-cache-pairs --mix content): the sample image alone for 60 % of the entries, the sample
#     with the look in words for 20 %, no sample and nothing named for 20 %; no prompt names an archive, years or hand;
#   - legible samples (python -m diffu_page.data.page_pair_samples): a page's sample at its pixel count, a window's cut
#     from its sample page at the window's line pitch and size, the rule recorded by the cache, the checkpoints'
#     export.json and the model index (sample_size legible), so drawing sizes a sample the same way;
#   - look and hand changes for half the sample-form entries each (--restyle-share 0.5 --hand-restyle-share 0.5, drawn
#     apart: about a quarter take both and a quarter neither). A hand change is refused for an entry it would move a
#     line off its page or onto another; measured on 3,000 of these rows, 55 % are (most at slants over 5 degrees).
# CPU steps are capped (no GPU, 4 threads, 16 GB, no swap); GPU steps run one at a time through jobs/spark_guard.sh:
#   1. CPU, seconds: the set's train and val rows without the duplicate collection's, each kept row as it is
#                                                                     data_flux3/page_pairs_many_1mp/*_dedup.jsonl
#      CPU, ~10 min: the content forms for them, each row planned again from its PAGE file and scan (page_pair_forms;
#      the set's 3,623 scans were built in 5 min on 6 processes)      .../*_dedup_content.jsonl
#   2. CPU, ~20 min: legible samples for both manifests, each sample page cut from its scan (page_pair_samples; about
#      16,900 JPEGs, ~6 GB)                                           .../*_dedup_content_legible.jsonl, .../samples/
#   3. GPU, ~5.5 h: caches: 12,000 training rows; validation (30 rows in every form of the mix); held-out (120 rows in
#      every form, seed 1, for the swap test)                         data_flux3/cache_moss_content_{train,val,heldout}
#   4. GPU, ~16 h: the full fine-tune from Base, STEPS steps of one sample (~22 s each: a 1 MP page and a 1 MP sample),
#      a checkpoint and a validation every SAVE_EVERY, Muon's momentum in bfloat16 (peak ~58 GB on the GPU)
#                                                                     runs/moss_content_pilot/full_c1
#   5. CPU: the last checkpoint's DiT as a model directory, its index naming sample_size legible
#                                                                     runs/moss/content_pilot_step_002500
# It passes when the sample form's validation loss falls below the generic form's (the form without a sample: the
# sample is read at all), printed at the end, and the swap test draws held-out pages in their samples' hands, not their
# own volume's (~/diffu_queue/weekend: pick_swap.py, draw_swap.py --form sample on the held-out cache's page rows, and
# score_swap.py, as queue_many2.sh runs them). Each step skips what a stopped run finished; a stopped cache resumes; a
# stopped training does not (it saves no state): move its directory aside and run again. Run detached (STEPS,
# SAVE_EVERY and WORKERS may be set in the environment):
#
#   mkdir -p runs/moss_content_pilot && setsid nohup jobs/moss/run_content_pilot.sh >> runs/moss_content_pilot/run.log 2>&1 < /dev/null &

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
GUARD="$REPO/jobs/spark_guard.sh"
BASE="${MOSS_BASE:-$HOME/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Image-Base/snapshots/85975793d8b3d432f1b6e27f352abaa4c2a4c0d3}"
SET="$REPO/data_flux3/page_pairs_many_1mp"
OUT="$REPO/runs/moss_content_pilot"
BASE_MODEL="$REPO/runs/moss/base"
CACHE="$REPO/data_flux3/cache_moss_content_train"
VAL_CACHE="$REPO/data_flux3/cache_moss_content_val"
HELDOUT_CACHE="$REPO/data_flux3/cache_moss_content_heldout"
RUN="$OUT/full_c1"
STEPS="${STEPS:-2500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
WORKERS="${WORKERS:-4}"
FINAL="$RUN/checkpoints/step_$(printf %06d "$STEPS")"
MODEL="$REPO/runs/moss/content_pilot_step_$(printf %06d "$STEPS")"
FLOOR_FILE="$HOME/diffu_queue/guard_floor_gb"  # the queue's guard floor, when it sets one
DUPLICATE="GT-RA_landskapshandlingar_20260227"  # GT-RA_landskapshandlingar_20260220's scans again (md5-checked)
export HF_HUB_OFFLINE=1 TORCH_FORCE_WEIGHTS_ONLY_LOAD=1

[ -f "$BASE/transformer/config.json" ] || { echo "no MOSS-Image Base snapshot at $BASE (set MOSS_BASE)"; exit 1; }
[ -f "$SET/train.jsonl" ] || { echo "no page pairs at $SET: build them with diffu_page.data.page_pairs"; exit 1; }
cd "$REPO"  # the set's scan.json names the model whose tokenizer counts its prompts relative to here: runs/moss/base
mkdir -p "$OUT"
step() { echo "=== $(date '+%F %T') $*"; }

cpu() {  # cpu <name> <command ...>: a CPU step, capped, its output in $OUT/<name>.log
    local name=$1 status=0
    shift
    systemd-run --user --scope --quiet -p MemoryMax=16G -p MemorySwapMax=0 nice -n 10 \
        env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 "$@" > "$OUT/$name.log" 2>&1 || status=$?
    [ "$status" -eq 0 ] || { echo "$name failed ($status), see $OUT/$name.log"; tail -3 "$OUT/$name.log"; exit 1; }
    tail -1 "$OUT/$name.log"
}

gpu() {  # gpu <name> <GB the job needs> <command ...>: a GPU step through the guard, retried while another holds it
    local name=$1 need=$2 status
    shift 2
    for try in $(seq 1 30); do
        status=0
        "$GUARD" --need-gb "$need" --floor-gb "$(cat "$FLOOR_FILE" 2>/dev/null || echo 20)" --name "content-$name" \
            --log "$OUT/guard.log" -- "$@" > "$OUT/$name.log" 2>&1 || status=$?
        [ "$status" -ne 3 ] && break  # 3: another guarded GPU job holds the lock
        echo "$name: the GPU is busy, try $try of 30"
        sleep 300
    done
    [ "$status" -eq 0 ] || { echo "$name failed ($status), see $OUT/$name.log"; exit 1; }
    tail -1 "$OUT/$name.log"
}

without_duplicate() {  # without_duplicate <from> <to>: the manifest's rows but the duplicate collection's, as they are
    local from=$1 to=$2
    if [ -f "$to" ]; then
        echo "already written: $to"
        return
    fi
    "$PY" -c '
import json, sys
duplicate, source, target = sys.argv[1:]
with open(source, encoding="utf-8") as rows, open(target, "w", encoding="utf-8") as kept:
    kept.writelines(row for row in rows if row.strip() and not json.loads(row)["volume"].startswith(duplicate))
' "$DUPLICATE" "$from" "$to.partial"
    mv "$to.partial" "$to"
    echo "$(wc -l < "$from") rows of $(basename "$from"), $(wc -l < "$to") without $DUPLICATE -> $to"
}

manifest() {  # manifest <made by: forms|samples> <from> <to>: a new manifest of the set, unless written already
    local tool=$1 from=$2 to=$3
    if [ -f "$to" ]; then
        echo "already written: $to"
        return
    fi
    cpu "$tool-$(basename "$to" .jsonl)" "$PY" -m "diffu_page.data.page_pair_$tool" --pairs "$from" --out "$to" \
        --workers "$WORKERS"
}

cache() {  # cache <name> <manifest> <cache dir> [diffu-moss-cache-pairs options ...]
    local name=$1 pairs=$2 dir=$3
    shift 3
    if [ -f "$dir/cache.json" ]; then  # written last
        echo "already cached: $dir"
        return
    fi
    gpu "cache-$name" 35 "$PY" -m diffu_moss.train.cache --model "$BASE_MODEL" --pairs "$pairs" --out "$dir" \
        --mix content "$@"
}

step "0/5 MOSS-Image Base as a model directory"
[ -f "$BASE_MODEL/modular_model_index.json" ] ||
    CUDA_VISIBLE_DEVICES="" "$PY" -m diffu_moss.init --base "$BASE" --out "$BASE_MODEL"

step "1/5 the train and val rows without $DUPLICATE, then their content forms (sample with look words; generic)"
without_duplicate "$SET/train.jsonl" "$SET/train_dedup.jsonl"
without_duplicate "$SET/val.jsonl" "$SET/val_dedup.jsonl"
manifest forms "$SET/train_dedup.jsonl" "$SET/train_dedup_content.jsonl"
manifest forms "$SET/val_dedup.jsonl" "$SET/val_dedup_content.jsonl"

step "2/5 legible samples: a page's at its pixel count, a window's at its line pitch"
manifest samples "$SET/train_dedup_content.jsonl" "$SET/train_dedup_content_legible.jsonl"
manifest samples "$SET/val_dedup_content.jsonl" "$SET/val_dedup_content_legible.jsonl"

step "3/5 caches: 12,000 training rows (look and hand changes for half the sample form each), validation, held-out"
TRAIN_ROWS="$SET/train_dedup_content_legible.jsonl"
VAL_ROWS="$SET/val_dedup_content_legible.jsonl"
cache train "$TRAIN_ROWS" "$CACHE" --limit 12000 --restyle-share 0.5 --hand-restyle-share 0.5
cache val "$VAL_ROWS" "$VAL_CACHE" --every-form --limit 30
cache heldout "$VAL_ROWS" "$HELDOUT_CACHE" --every-form --limit 120 --seed 1
jq -c '{entries, sample_size, mix, restyled, hand_restyled, hand_refused}' "$CACHE/cache.json"

step "4/5 the full fine-tune from Base: $STEPS steps of one sample"
if [ -f "$FINAL/transformer/export.json" ]; then
    echo "already trained: $FINAL"
else
    gpu train 70 "$PY" -m diffu_moss.train.loop --full --transformer "$BASE/transformer" --cache "$CACHE" \
        --val-cache "$VAL_CACHE" --out "$RUN" --steps "$STEPS" --samples-per-step 1 --save-every "$SAVE_EVERY" \
        --val-every "$SAVE_EVERY" --muon-momentum-dtype bfloat16 --seed 0
fi

step "5/5 the last checkpoint's DiT as a model directory"
[ -f "$MODEL/modular_model_index.json" ] ||
    CUDA_VISIBLE_DEVICES="" "$PY" -m diffu_moss.init --base "$BASE" --transformer "$FINAL/transformer" --out "$MODEL"
jq -c '{takes_reference, sample_size}' "$MODEL/modular_model_index.json"

step "the validation by form (passes when sample < generic)"
jq -c 'select(.val_loss != null) | {step, val_loss}' "$RUN/train_log.jsonl"
touch "$OUT/DONE"
