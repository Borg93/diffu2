#!/bin/bash
# MOSS-Image reference pilot: teach MOSS-Image's DiT to draw from a sample page (a reference crop of the same hand) with
# our own LoRA training (diffu_moss.train), on v3d's style pairs (data_flux3/style_pairs_v3d: each target with a crop of
# its own or its facing page, sharing none of its lines, captioned with the reference's measured look; the 85 pairs
# whose reference repeats a target's line are left out by the cache). A fifth of the pairs train without the
# reference (as MOSS was trained), the rest with it, half of those with the look words dropped so the look must come
# from the image. The 1715 hand LoRA's recipe: rank 32, alpha 32, AdamW 1e-4, constant after 50 warmup steps, 2
# samples a step at ~1 MP (plus a 0.5 MP reference). One GPU job at a time through jobs/spark_guard.sh:
#   1. MOSS-Image Base as a model directory (CPU): the encoders the caches are made with      runs/moss/base
#   2. cache 4,000 training pairs (one prompt form each) and 32 validation pairs (all three)  data_flux3/cache_moss_ref_v3d(_val)
#   3. a 5-step smoke run: the memory and the step time, before the long run                 runs/moss_ref_pilot/smoke
#   4. train 1,500 steps, a checkpoint and a validation (loss by prompt form) every 250       runs/moss_ref_pilot/lora
#   5. merge the last checkpoint into Base (CPU) and write its model directory                runs/moss/ref_pilot
#   6. draw 8 held-out crops of v3d's val_eval with and without their reference               runs/moss_ref_pilot/eval_*
# A stopped cache resumes where it stopped; a stopped training does not (move its directory aside and rerun, or export
# its last checkpoint). Run detached:
#
#   mkdir -p runs/moss_ref_pilot && setsid nohup jobs/moss/run_ref_pilot.sh >> runs/moss_ref_pilot/run.log 2>&1 < /dev/null &

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
GUARD="$REPO/jobs/spark_guard.sh"
BASE="${MOSS_BASE:-$HOME/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Image-Base/snapshots/85975793d8b3d432f1b6e27f352abaa4c2a4c0d3}"
PAIRS="$REPO/data_flux3/style_pairs_v3d"
CACHE="$REPO/data_flux3/cache_moss_ref_v3d"
VAL_CACHE="$REPO/data_flux3/cache_moss_ref_v3d_val"
OUT="$REPO/runs/moss_ref_pilot"
BASE_MODEL="$REPO/runs/moss/base"
MODEL="$REPO/runs/moss/ref_pilot"
STEPS=1500
FINAL="$OUT/lora/checkpoints/step_$(printf %06d "$STEPS")"
EXPORT="$OUT/export_step_$(printf %06d "$STEPS")"
export CUDA_VISIBLE_DEVICES=0

[ -f "$BASE/transformer/config.json" ] || { echo "no MOSS-Image Base snapshot at $BASE (set MOSS_BASE)"; exit 1; }
[ -f "$PAIRS/train.jsonl" ] || { echo "no style pairs at $PAIRS: build them with diffu_page.data.style_pairs"; exit 1; }
mkdir -p "$OUT"
step() { echo "=== $(date '+%F %T') $*"; }

cache() {  # cache <manifest> <cache dir> <name> [diffu-moss-cache-pairs options ...]
    local manifest=$1 dir=$2 name=$3
    shift 3
    if [ -f "$dir/cache.json" ]; then  # written last
        echo "already cached: $dir"
        return
    fi
    "$GUARD" --need-gb 35 --name "moss-ref-$name" --log "$OUT/guard.log" -- \
        "$PY" -m diffu_moss.train.cache --model "$BASE_MODEL" --pairs "$manifest" --out "$dir" "$@" \
        > "$OUT/$name.log" 2>&1
    tail -1 "$OUT/$name.log"
}

train() {  # train <name> [diffu-moss-train-ref options ...]: the run goes to $OUT/<name>
    local name=$1
    shift
    "$GUARD" --need-gb 45 --name "moss-ref-$name" --log "$OUT/guard.log" -- \
        "$PY" -m diffu_moss.train.loop --transformer "$BASE/transformer" --cache "$CACHE" --val-cache "$VAL_CACHE" \
        --out "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
}

draw() {  # draw <name> [diffu-evaluate options ...]: fails as the evaluation does
    local name=$1
    shift
    if "$GUARD" --need-gb 50 --name "moss-ref-eval-$name" --log "$OUT/guard.log" -- \
        "$PY" -m diffu_page.cli.evaluate --crops "$PAIRS/val_eval" -n 8 --model "$MODEL" --out "$OUT/eval_$name" \
        "$@" > "$OUT/eval_$name.log" 2>&1; then
        tail -1 "$OUT/eval_$name.log"
    else
        echo "eval $name failed, see $OUT/eval_$name.log"
    fi
}

step "1/6 MOSS-Image Base as a model directory"
[ -f "$BASE_MODEL/modular_model_index.json" ] ||
    CUDA_VISIBLE_DEVICES="" "$PY" -m diffu_moss.init --base "$BASE" --out "$BASE_MODEL"

step "2/6 cache the training and validation pairs"
cache "$PAIRS/train.jsonl" "$CACHE" cache --limit 4000
cache "$PAIRS/val/val.jsonl" "$VAL_CACHE" cache_val --limit 32 --every-form

step "3/6 a 5-step smoke run"
rm -rf "$OUT/smoke"
train smoke --steps 5 --save-every 5 --val-every 5
tail -2 "$OUT/smoke.log"
tail -1 "$OUT/guard.log"

step "4/6 train the LoRA: $STEPS steps of 2"
if [ -f "$FINAL/adapter.json" ]; then
    echo "already trained: $FINAL"
else
    train lora --steps "$STEPS" --save-every 250 --val-every 250
fi
grep -a "val_loss" "$OUT/lora/train_log.jsonl" | tail -2 || true

step "5/6 merge the last checkpoint into Base and write its model directory"
[ -f "$EXPORT/export.json" ] ||
    "$GUARD" --need-gb 25 --name moss-ref-export --log "$OUT/guard.log" -- \
        env CUDA_VISIBLE_DEVICES="" "$PY" -m diffu_moss.train.export --transformer "$BASE/transformer" \
        --checkpoint "$FINAL" --out "$EXPORT" > "$OUT/export.log" 2>&1
[ -f "$MODEL/modular_model_index.json" ] ||
    CUDA_VISIBLE_DEVICES="" "$PY" -m diffu_moss.init --base "$BASE" --transformer "$EXPORT" --out "$MODEL"

step "6/6 draw 8 held-out crops with and without their reference"
draw with_reference
draw without_reference --without-reference

step "done"
touch "$OUT/DONE"
