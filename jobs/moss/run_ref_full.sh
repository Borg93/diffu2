#!/bin/bash
# MOSS-Image reference full fine-tune, tested on the DGX Spark: every weight of MOSS-Image's DiT trained to draw from a
# sample page (a reference crop of the same hand), on the pilot's v3d caches (jobs/moss/run_ref_pilot.sh makes them;
# made here when missing), by MOSS-Image's own recipe (diffu-moss-train-ref --full, diffu_moss.train.full: Muon on the
# joint blocks' matrices and AdamW on the rest at 5e-5, weight decay 0.01 but none on biases, norms and embedders, the
# last block's q/k norm weights decayed towards one, gradients clipped at 1, constant after 50 warmup steps, 2 samples
# a step at ~1 MP plus a 0.5 MP reference). The weights in float32, the forward in bfloat16, the joint blocks
# checkpointed; no average of the weights and no optimizer state saved (--ema, --save-state: 20 and 42 GB more, for
# HPC). Its memory, estimated before any run: 20.3 GB of weights, 20.3 GB of gradients, 19.0 GB of Muon momentum (the
# 4.76 B block weights), 2.6 GB of AdamW moments, ~8 GB of activations at the second sample's backward (the first's
# gradients held) and ~3 GB for CUDA and the process: ~73 GB, so every training asks the guard for 75 (plus its 20 GB
# floor: it starts only with 95 GB free). One GPU job at a time through jobs/spark_guard.sh:
#   1. MOSS-Image Base as a model directory (CPU): the encoders the caches are made with      runs/moss/base
#   2. cache 4,000 training pairs and 32 validation pairs, unless the pilot did              data_flux3/cache_moss_ref_v3d(_val)
#   3. a 5-step smoke run: each step's time and peak memory from the trainer's log           runs/moss_ref_full/smoke
#   4. the overfit test: OVERFIT_STEPS steps on the cache's first OVERFIT_PAIRS pairs drawn   runs/moss_ref_full/overfit
#      with their reference, its last checkpoint as a model directory, those pairs drawn     runs/moss/ref_full_overfit
#      back from their references beside their real pages                                   runs/moss_ref_full/overfit_drawn
#   5. train STEPS steps, a checkpoint and a validation (loss by prompt form) every SAVE_EVERY runs/moss_ref_full/full
#   6. the last checkpoint's DiT as a model directory (CPU; no merge: a checkpoint is the DiT) runs/moss/ref_full
#   7. draw 8 held-out crops of v3d's val_eval with and without their reference              runs/moss_ref_full/eval_*
#
# The overfit test passes when both hold:
#   - its loss falls far below its start: the mean training loss of its last 20 steps is under half the mean of its
#     first 20 (stage 4 prints both and the verdict, read from runs/moss_ref_full/overfit/train_log.jsonl; each step's
#     two samples sit at random noise levels, so means of 20 steps are compared, the first 20 inside the warmup, before
#     the weights have moved much);
#   - its drawings reproduce the training pages: in each sheet of runs/moss_ref_full/overfit_drawn (the reference, the
#     real page, the drawing) the drawing shows the real page's lines word for word, in their places, in its hand, ink
#     and paper.
# The long run goes on whatever the verdict: read the overfit test first. A stopped cache resumes where it stopped; a
# stopped training does not (it saves no state: move its directory aside and rerun). Run detached (STEPS, SAVE_EVERY,
# OVERFIT_PAIRS and OVERFIT_STEPS may be set in the environment):
#
#   mkdir -p runs/moss_ref_full && setsid nohup jobs/moss/run_ref_full.sh >> runs/moss_ref_full/run.log 2>&1 < /dev/null &

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
GUARD="$REPO/jobs/spark_guard.sh"
BASE="${MOSS_BASE:-$HOME/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Image-Base/snapshots/85975793d8b3d432f1b6e27f352abaa4c2a4c0d3}"
PAIRS="$REPO/data_flux3/style_pairs_v3d"
CACHE="$REPO/data_flux3/cache_moss_ref_v3d"
VAL_CACHE="$REPO/data_flux3/cache_moss_ref_v3d_val"
OUT="$REPO/runs/moss_ref_full"
BASE_MODEL="$REPO/runs/moss/base"
MODEL="$REPO/runs/moss/ref_full"
OVERFIT_MODEL="$REPO/runs/moss/ref_full_overfit"
STEPS="${STEPS:-500}"
SAVE_EVERY="${SAVE_EVERY:-250}"
OVERFIT_PAIRS="${OVERFIT_PAIRS:-8}"
OVERFIT_STEPS="${OVERFIT_STEPS:-300}"
FINAL="$OUT/full/checkpoints/step_$(printf %06d "$STEPS")"
OVERFIT_FINAL="$OUT/overfit/checkpoints/step_$(printf %06d "$OVERFIT_STEPS")"
TRAIN_GB=75  # the full fine-tune's ~73 GB at its peak (the estimate above)
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
    "$GUARD" --need-gb 35 --name "moss-full-$name" --log "$OUT/guard.log" -- \
        "$PY" -m diffu_moss.train.cache --model "$BASE_MODEL" --pairs "$manifest" --out "$dir" "$@" \
        > "$OUT/$name.log" 2>&1
    tail -1 "$OUT/$name.log"
}

train() {  # train <name> [diffu-moss-train-ref options ...]: a full fine-tune into $OUT/<name>
    local name=$1
    shift
    "$GUARD" --need-gb "$TRAIN_GB" --name "moss-full-$name" --log "$OUT/guard.log" -- \
        "$PY" -m diffu_moss.train.loop --full --transformer "$BASE/transformer" --cache "$CACHE" --out "$OUT/$name" \
        "$@" > "$OUT/$name.log" 2>&1
}

steps_of() {  # steps_of <name>: each step of $OUT/<name> as its trainer logged it: loss, seconds, peak GB on the GPU
    jq -r 'select(.loss != null) | "step \(.step): loss \(.loss), \(.seconds) s, peak \(.peak_memory_gb // "n/a") GB"' \
        "$OUT/$1/train_log.jsonl"
}

overfit_verdict() {  # the overfit test's loss criterion (the header), from its trainer's log
    jq -rs 'def three: . * 1000 | round / 1000;
        [.[] | select(.loss != null) | .loss] as $losses
        | ($losses[:20] | add / length) as $first
        | ($losses[-20:] | add / length) as $last
        | "overfit loss: \($first | three) over the first 20 steps, \($last | three) over the last 20, "
          + "\($last / $first | three) of the start: "
          + (if $last < 0.5 * $first then "PASS" else "FAIL" end) + " (passes under 0.5)"' \
        "$OUT/overfit/train_log.jsonl"
}

draw() {  # draw <name> [diffu-evaluate options ...]: fails as the evaluation does
    local name=$1
    shift
    if "$GUARD" --need-gb 50 --name "moss-full-eval-$name" --log "$OUT/guard.log" -- \
        "$PY" -m diffu_page.cli.evaluate --crops "$PAIRS/val_eval" -n 8 --model "$MODEL" --out "$OUT/eval_$name" \
        "$@" > "$OUT/eval_$name.log" 2>&1; then
        tail -1 "$OUT/eval_$name.log"
    else
        echo "eval $name failed, see $OUT/eval_$name.log"
    fi
}

step "1/7 MOSS-Image Base as a model directory"
[ -f "$BASE_MODEL/modular_model_index.json" ] ||
    CUDA_VISIBLE_DEVICES="" "$PY" -m diffu_moss.init --base "$BASE" --out "$BASE_MODEL"

step "2/7 cache the training and validation pairs"
cache "$PAIRS/train.jsonl" "$CACHE" cache --limit 4000
cache "$PAIRS/val/val.jsonl" "$VAL_CACHE" cache_val --limit 32 --every-form

step "3/7 a 5-step smoke run"
rm -rf "$OUT/smoke"
train smoke --val-cache "$VAL_CACHE" --steps 5 --save-every 5 --val-every 5
steps_of smoke
tail -1 "$OUT/guard.log"

step "4/7 the overfit test: $OVERFIT_STEPS steps on $OVERFIT_PAIRS pairs drawn with their reference, then drawn back"
if [ -f "$OVERFIT_FINAL/transformer/export.json" ]; then
    echo "already trained: $OVERFIT_FINAL"
else
    rm -rf "$OUT/overfit"  # a stopped overfit run is cheaper to redo than to keep
    train overfit --limit-pairs "$OVERFIT_PAIRS" --steps "$OVERFIT_STEPS" --save-every "$OVERFIT_STEPS"
fi
overfit_verdict
[ -f "$OVERFIT_MODEL/modular_model_index.json" ] ||
    CUDA_VISIBLE_DEVICES="" "$PY" -m diffu_moss.init --base "$BASE" --transformer "$OVERFIT_FINAL/transformer" \
        --out "$OVERFIT_MODEL"
if "$GUARD" --need-gb 50 --name moss-full-overfit-draw --log "$OUT/guard.log" -- \
    "$PY" -m diffu_page.cli.draw_pairs --model "$OVERFIT_MODEL" --cache "$CACHE" --limit-pairs "$OVERFIT_PAIRS" \
    --out "$OUT/overfit_drawn" > "$OUT/overfit_drawn.log" 2>&1; then
    tail -1 "$OUT/overfit_drawn.log"
else
    echo "drawing the overfit pairs failed, see $OUT/overfit_drawn.log"
fi

step "5/7 train every weight: $STEPS steps of 2"
if [ -f "$FINAL/transformer/export.json" ]; then
    echo "already trained: $FINAL"
else
    train full --val-cache "$VAL_CACHE" --steps "$STEPS" --save-every "$SAVE_EVERY" --val-every "$SAVE_EVERY"
fi
grep -a "val_loss" "$OUT/full/train_log.jsonl" | tail -2 || true

step "6/7 the last checkpoint's DiT as a model directory"
[ -f "$MODEL/modular_model_index.json" ] ||
    CUDA_VISIBLE_DEVICES="" "$PY" -m diffu_moss.init --base "$BASE" --transformer "$FINAL/transformer" --out "$MODEL"

step "7/7 draw 8 held-out crops with and without their reference"
draw with_reference
draw without_reference --without-reference

step "done"
touch "$OUT/DONE"
