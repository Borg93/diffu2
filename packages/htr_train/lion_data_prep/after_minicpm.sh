#!/bin/bash
# After minicpm-all (train.sh execs this, as does the resume that took over its training; read only then, so it can
# change while that trains). Each step leaves a .done marker in $MARKERS: run it again to resume after the last
# finished step.
#  1. best: minicpm-all's checkpoint of least eval loss (its trainer state's best_model_checkpoint), linked as
#     runs/htr/minicpm-all/best: every step after reads with it.
#  2. On the first of $GPUS, vLLM serving it with OpenBMB's flags: all 6,358 out-of-domain lines and the 5,000 test_0050
#     lines read and scored beside the Swedish Lion TrOCR's greedy readings of the same lines (htr-eval cer, as read
#     and with loops cut: $OUT/result.txt); then its reads of 100,000 training lines and of every test line, Jev's real
#     errors beside its rule ones (as OpenBMB's finetune_jev_en.md asks), and Jev's pairs, dev split and test pairs.
#     On the second, meanwhile, vLLM serving it as the DSpark run serves its target (htr-train serve-target): its
#     answers to 300,000 prompts drawn from the training lines and regions together, the draft's responses.
#  3. Jev trained on both GPUs (configs/jev_all.yaml), then the DSpark draft on the first (configs/dspark_all.yaml;
#     DSPARK=0 stops before it).
#
#   GPUS=1,2 bash packages/htr_train/lion_data_prep/after_minicpm.sh
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
D=${SETS:-data/htr}
M=${MARKERS:-runs/htr/pipeline_markers}
O=${OUT:-runs/htr/final}
RUN=runs/htr/minicpm-all
SPECS=packages/htr_train/lion_data_prep/configs
E=packages/htr_train/envs/minicpm/.venv/bin/htr-eval
V=packages/htr_train/envs/vllm/.venv/bin/vllm
TROCR_OOD=runs/htr/eval/ood_trocr_greedy.jsonl  # the Swedish Lion's greedy readings (runs/htr/eval_trocr.sh)
TROCR_TEST=runs/htr/ckpt_eval/test_trocr.jsonl  # (runs/htr/ckpt_eval/run.sh)
export HTR_TRAIN_LOCAL_GPUS=${GPUS:-1,2}
FIRST=${HTR_TRAIN_LOCAL_GPUS%%,*}
SECOND=${HTR_TRAIN_LOCAL_GPUS##*,}
SCORING=http://localhost:8031
TARGET=http://localhost:8000  # dspark_all.yaml's port
mkdir -p "$M" "$O"
log() { echo "[$(date '+%F %T')] $*"; }
step() {  # step NAME CMD...: run once, stop on failure
  local name=$1; shift
  [ -f "$M/$name.done" ] && { log "skip $name (done)"; return 0; }
  log "start $name"
  if "$@"; then touch "$M/$name.done"; log "done $name"; else log "FAILED $name"; return 1; fi
}
done_all() { local name; for name; do [ -f "$M/$name.done" ] || return 1; done; }
gpus_free() {  # until no process computes on the GPUs $GPUS names
  local uuids busy
  uuids=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v want=",$HTR_TRAIN_LOCAL_GPUS," \
    'index(want, "," $1 ",") {print $2}')
  while busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -cFf <(echo "$uuids")); \
        [ "$busy" -gt 0 ]; do
    log "GPUs $HTR_TRAIN_LOCAL_GPUS busy ($busy processes): waiting"; sleep 60
  done
}
SERVERS=()
trap '[ ${#SERVERS[@]} -gt 0 ] && kill "${SERVERS[@]}" 2>/dev/null' EXIT
up() {  # up NAME URL CMD...: a server started in the background (its log $O/NAME.log), once it answers
  local name=$1 url=$2 pid; shift 2
  "$@" > "$O/$name.log" 2>&1 &
  pid=$!; SERVERS+=("$pid")
  until curl -sf "$url/health" > /dev/null; do
    kill -0 "$pid" 2>/dev/null || { log "$name ended before it answered: see $O/$name.log"; return 1; }
    sleep 5
  done
  log "$name serving on $url"
}

link_best() {
  local newest best
  newest=$(ls -d $RUN/output/checkpoint-* | sort -V | tail -1)
  best=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["best_model_checkpoint"])' \
    "$newest/trainer_state.json") && [ -d "$best" ] && ln -sfn "$best" $RUN/best && log "best: $best"
}
read_with() {  # read_with CONCURRENCY LINES OUT: the best checkpoint's readings, through the scoring server
  $E read --reader minicpm --model minicpm --endpoint $SCORING --concurrency "$1" --lines "$2" --out "$3"
}
scored() { COLUMNS=200 $E cer $TROCR_OOD $O/ood_minicpm.jsonl $TROCR_TEST $O/test_minicpm.jsonl > $O/result.txt; }
first_gpu() {
  done_all final_ood final_test final_cer jev_train_reads jev_test_reads && return 0
  up scoring $SCORING env CUDA_VISIBLE_DEVICES=$FIRST $V serve "$(readlink -f $RUN/best)" --served-model-name minicpm \
    --port ${SCORING##*:} --allowed-local-media-path "$(readlink -f $D)" --gpu-memory-utilization 0.9 \
    --max-model-len 8192 --max-num-seqs 256 --attention-backend FLASH_ATTN --mm-encoder-attn-backend FLASH_ATTN \
    --max-num-batched-tokens 40960 --limit-mm-per-prompt '{"image": 4, "video": 0}' \
    --mm-processor-kwargs '{"downsample_mode": "4x", "max_slice_nums": 9}' --trust-request-mm-kwargs \
    --default-chat-template-kwargs '{"enable_thinking": false}' || return 1
  step final_ood read_with 64 $D/eval/ood.jsonl $O/ood_minicpm.jsonl &&
    step final_test read_with 64 $D/eval/test5000.jsonl $O/test_minicpm.jsonl &&
    step final_cer scored && cat $O/result.txt &&
    step jev_train_lines $E set-lines $D/minicpm_lines_scaled.train.jsonl --sample 100000 --seed 0 \
      --out $O/train_sample.jsonl &&
    step jev_train_read read_with 128 $O/train_sample.jsonl $O/train_sample_minicpm.jsonl &&
    step jev_train_reads $E as-reads $O/train_sample_minicpm.jsonl --reader minicpm-all --out $O/reads_train.jsonl &&
    step jev_test_lines $E set-lines $D/minicpm_lines_scaled.test.jsonl --out $O/test_all.jsonl &&
    step jev_test_read read_with 128 $O/test_all.jsonl $O/test_all_minicpm.jsonl &&
    step jev_test_reads $E as-reads $O/test_all_minicpm.jsonl --reader minicpm-all --out $O/reads_test.jsonl
  local status=$?
  kill "${SERVERS[-1]}" 2>/dev/null
  return $status
}
jev_sets() {
  step jev_pairs uv run htr-train jev-set --lines $D/minicpm_lines_scaled.train.jsonl --reads $O/reads_train.jsonl \
      --out $D/jev_lines.jsonl &&
    step jev_dev uv run htr-train split-dev $D/jev_lines.jsonl --share 0.02 &&
    step jev_test_pairs uv run htr-train jev-set --side test --lines $D/minicpm_lines_scaled.test.jsonl \
      --reads $O/reads_test.jsonl --out $D/jev_lines.test.jsonl
}
prompts_set() {  # the training lines and regions as one set, to draw the draft's prompts from in the training's mix
  cat $D/minicpm_lines_scaled.train.jsonl $D/minicpm_regions_masked.train.jsonl > $D/dspark_prompts.jsonl &&
    cat $D/minicpm_lines_scaled.train.meta.jsonl $D/minicpm_regions_masked.train.meta.jsonl \
      > $D/dspark_prompts.meta.jsonl
}
second_gpu() {
  [ "${DSPARK:-1}" = 0 ] && return 0
  done_all dspark_responses && return 0
  local serve
  step dspark_prompts prompts_set || return 1
  serve=$(uv run htr-train serve-target $SPECS/dspark_all.yaml) || return 1
  up target $TARGET env CUDA_VISIBLE_DEVICES=$SECOND bash -c "$serve" || return 1
  step dspark_responses uv run htr-train dspark-set --prompts $D/dspark_prompts.jsonl --target $RUN/best \
    --endpoint $TARGET --max-tokens 4096 --limit 300000 --seed 0 --concurrency 64 --out $D/dspark_responses.jsonl
  local status=$?
  kill "${SERVERS[-1]}" 2>/dev/null
  return $status
}

gpus_free
step best link_best || exit 1
second_gpu > $O/second_gpu.log 2>&1 &
SECOND_JOB=$!
log "the draft's responses on GPU $SECOND: $O/second_gpu.log"
first_gpu || exit 1
jev_sets || exit 1
wait $SECOND_JOB || { log "the draft's responses failed: see $O/second_gpu.log"; exit 1; }
gpus_free; step jev_train uv run htr-train launch $SPECS/jev_all.yaml --where local || exit 1
[ "${DSPARK:-1}" = 0 ] && { log "DSPARK=0: stopping before the draft"; exit 0; }
gpus_free; step dspark_train env HTR_TRAIN_LOCAL_GPUS=$FIRST uv run htr-train launch $SPECS/dspark_all.yaml --where local
