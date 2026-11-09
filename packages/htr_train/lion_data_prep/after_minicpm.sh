#!/bin/bash
# After minicpm-all (train.sh execs this, as does the resume that took over its training; read only then, so it can
# change while that trains). Each step leaves a .done marker in $MARKERS: run it again to resume after the last
# finished step.
#  1. best: of minicpm-all's checkpoints on disk, the one of lowest corpus CER on its 1,000 dev lines (link_best),
#     linked as runs/htr/minicpm-all/best: every step after reads with it.
#  2. On the first of $GPUS, vLLM serving it with OpenBMB's flags: all 6,358 out-of-domain lines and the 5,000 test_0050
#     lines read and scored beside the Swedish Lion TrOCR's greedy readings of the same lines (htr-eval cer, as read
#     and with loops cut: $OUT/result.txt); then its reads of 100,000 training lines and of every test line, Jev's real
#     errors beside its rule ones (as OpenBMB's finetune_jev_en.md asks), and Jev's pairs, dev split and test pairs.
#     On the second, meanwhile, vLLM serving it as the DSpark run serves its target (htr-train serve-target): its
#     answers to 250,000 training lines and 50,000 training regions (each a seeded draw), the draft's responses.
#  3. Jev trained on both GPUs (configs/jev_all.yaml; if it stops, started again up to $RETRIES times, resuming from
#     its newest whole checkpoint), then the DSpark draft on both, vLLM on the first and the trainer on the second
#     (configs/dspark_all.yaml; DSPARK=0 stops before it). A GPU step waits until none of our processes is on $GPUS
#     and each has $NEED_GB free: another user's small job there does not hold it back.
#  4. Each model to the Hugging Face Hub once it is made (publish_model.sh, its card from model_cards/): the reader
#     after its scoring as $READER_REPO, Jev and the draft after their training as $READER_REPO-jev and -dspark; a new
#     repo private unless HF_PUBLIC=1. A failed upload is logged and the pipeline goes on; run again to retry it.
#
#   GPUS=1,2 bash packages/htr_train/lion_data_prep/after_minicpm.sh
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
D=${SETS:-data/htr}
M=${MARKERS:-runs/htr/pipeline_markers}
O=${OUT:-runs/htr/final}
RUN=runs/htr/minicpm-all
SPECS=packages/htr_train/lion_data_prep/configs
RUNS=${HTR_TRAIN_RUNS:-runs/htr}
RETRIES=${RETRIES:-10}
NEED_GB=${NEED_GB:-60}  # free on each GPU before a GPU step starts (jev_all.yaml's need_gb)
E=packages/htr_train/envs/minicpm/.venv/bin/htr-eval
V=packages/htr_train/envs/vllm/.venv/bin/vllm
TROCR_OOD=runs/htr/eval/ood_trocr_greedy.jsonl  # the Swedish Lion's greedy readings (runs/htr/eval_trocr.sh)
TROCR_TEST=runs/htr/ckpt_eval/test_trocr.jsonl  # (runs/htr/ckpt_eval/run.sh)
export HTR_TRAIN_LOCAL_GPUS=${GPUS:-1,2}
FIRST=${HTR_TRAIN_LOCAL_GPUS%%,*}
SECOND=${HTR_TRAIN_LOCAL_GPUS##*,}
SCORING=http://localhost:8031
TARGET=http://localhost:8000  # dspark_all.yaml's port
export READER_REPO=${READER_REPO:-Gabriel/diffu-minicpm-ocr-htr}  # publish_model.sh reads it for the draft
PUBLISH=packages/htr_train/lion_data_prep/publish_model.sh
JEV_RUN=runs/htr/jev-all  # jev_all.yaml's name
DSPARK_RUN=runs/htr/dspark-all  # dspark_all.yaml's name
mkdir -p "$M" "$O"
log() { echo "[$(date '+%F %T')] $*"; }
step() {  # step NAME CMD...: run once, stop on failure
  local name=$1; shift
  [ -f "$M/$name.done" ] && { log "skip $name (done)"; return 0; }
  log "start $name"
  if "$@"; then touch "$M/$name.done"; log "done $name"; else log "FAILED $name"; return 1; fi
}
done_all() { local name; for name; do [ -f "$M/$name.done" ] || return 1; done; }
gpus_free() {  # until none of our processes computes on the GPUs $GPUS names and each has $NEED_GB free. Another
  # user's job there does not hold the queue back unless it leaves too little memory for the next step
  local uuids busy short
  uuids=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v want=",$HTR_TRAIN_LOCAL_GPUS," \
    'index(want, "," $1 ",") {print $2}')
  while busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -Ff <(echo "$uuids") \
          | awk -F', ' '{print $2}' | xargs -r ps -o user= -p | grep -cx "$USER"); \
        short=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
          | awk -F', ' -v want=",$HTR_TRAIN_LOCAL_GPUS," -v need=$((NEED_GB * 1024)) \
            'index(want, "," $1 ",") && $2 < need {n++} END {print n + 0}'); \
        [ "$busy" -gt 0 ] || [ "$short" -gt 0 ]; do
    log "GPUs $HTR_TRAIN_LOCAL_GPUS: $busy of our processes, $short with under $NEED_GB GB free: waiting"; sleep 60
  done
}
trained() {  # trained SPEC NAME: SPEC launched as run NAME, or that run started again if it was launched before. A
  # training that stops (GPU 2 hangs now and then with Xid 8) is started again up to $RETRIES times, 2 minutes apart,
  # from its own command.sh: htr-train's steps resume from the newest whole checkpoint and skip the batches it trained
  local spec=$1 name=$2 run=$RUNS/$2 attempt
  if [ ! -f "$run/manifest.json" ]; then
    uv run htr-train launch "$spec" --where local && return 0
    [ -f "$run/manifest.json" ] || return 1  # refused before it started
  elif ! grep -q '^RESUME=()' "$run/steps.sh"; then
    log "$run/steps.sh predates resuming: it would train from the start into the same folder; resume it by hand"
    return 1
  else
    bash "$run/command.sh" && return 0
  fi
  for attempt in $(seq "$RETRIES"); do
    log "$name stopped: starting it again ($attempt of $RETRIES) in 2 minutes, from its newest checkpoint"
    sleep 120; gpus_free
    bash "$run/command.sh" && return 0
  done
  log "$name stopped $RETRIES times more: giving up"
  return 1
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

link_best() {  # of minicpm-all's checkpoints on disk, the one of lowest corpus CER (edits over characters) on the
  # 1,000 dev lines its cer callback reads at each evaluation (whole volumes held out; the test lines are watched,
  # never chosen on), linked as $RUN/best; each candidate logged. Not eval loss: it rose in epoch 3 while CER fell.
  # Not the per-line mean: one short line read wrong moves it by 0.1 point
  local best
  best=$(python3 - $RUN <<'EOF'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
cer = {}
for line in (run / "output" / "logging.jsonl").read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    if "eval_cer_corpus_dev_lines" in row:
        cer[int(row["global_step/max_steps"].split("/")[0])] = row["eval_cer_corpus_dev_lines"]
candidates = []
for state in run.glob("output/checkpoint-*/trainer_state.json"):
    step = int(state.parent.name.rsplit("-", 1)[1])
    if step in cer and any(state.parent.glob("model*.safetensors")):
        candidates.append((cer[step], -step, str(state.parent.resolve())))
if not candidates:
    sys.exit("no checkpoint with a dev-line CER")
for value, step, _ in sorted(candidates):
    print(f"  checkpoint-{-step}: dev lines {value:.2%} corpus CER", file=sys.stderr)
print(min(candidates)[-1])
EOF
) && [ -d "$best" ] && ln -sfn "$best" $RUN/best && log "best: $best"
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
prompts_set() {  # the draft's prompts: 250,000 training lines and 50,000 training regions, each a seeded draw (a mix
  # of the whole set would hold about 16,000 regions: 5 % of the samples), each row beside its meta row
  python3 - "$D" <<'EOF'
import random
import sys
from pathlib import Path

sets = Path(sys.argv[1])
with (sets / "dspark_prompts.jsonl").open("w", encoding="utf-8") as prompts, \
        (sets / "dspark_prompts.meta.jsonl").open("w", encoding="utf-8") as metas:
    for stem, count in [("minicpm_lines_scaled.train", 250_000), ("minicpm_regions_masked.train", 50_000)]:
        rows = (sets / f"{stem}.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
        meta = (sets / f"{stem}.meta.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
        if len(rows) != len(meta):
            sys.exit(f"{stem}: {len(rows)} rows, {len(meta)} meta rows")
        for index in sorted(random.Random(0).sample(range(len(rows)), min(count, len(rows)))):
            prompts.write(rows[index])
            metas.write(meta[index])
EOF
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

count() { python3 -c 'import sys; print(f"{int(sys.argv[1]):,}")' "$(wc -l < "$1")"; }
lines_per_second() { python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["lines_per_second"])' "$1"; }
jev_eval() {  # Jev's last evaluation on its dev pairs
  python3 - "$JEV_RUN/output/logging.jsonl" <<'EOF'
import json
import sys

last = [row for row in map(json.loads, open(sys.argv[1], encoding="utf-8")) if "eval_acc" in row][-1]
print(f"accuracy {last['eval_acc']:.1%}, loss {last['eval_loss']:.4f} (step {last['global_step/max_steps']})")
EOF
}
dspark_val() {  # the draft's validation at its best epoch
  python3 - "$1/val_metrics.json" <<'EOF'
import json
import sys

val = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"{val['accept_len_epoch']:.2f} tokens accepted a step on average, acceptance rate {val['accept_rate_epoch']:.1%}")
EOF
}
publish_reader() {
  local best; best=$(readlink -f $RUN/best)
  bash $PUBLISH reader "$best" "$READER_REPO" STEP="$(basename "$best")" \
    LINES="$(count $D/minicpm_lines_scaled.train.jsonl)" REGIONS="$(count $D/minicpm_regions_masked.train.jsonl)" \
    RESULTS=file:$O/result.txt SPEED="$(lines_per_second $O/test_minicpm.speed.json)" \
    TROCR_SPEED="$(lines_per_second "${TROCR_TEST%.jsonl}.speed.json")"
}
publish_jev() {
  local last; last=$(ls -d $JEV_RUN/output/checkpoint-* | sort -V | tail -1)
  bash $PUBLISH jev "$last" "$READER_REPO-jev" STEP="$(basename "$last")" READER_REPO="$READER_REPO" \
    READER_NAME="${READER_REPO##*/}" RENDERER="$(sed -n 's/^ *renderer: *//p' $SPECS/jev_all.yaml)" EVAL="$(jev_eval)"
}
publish_dspark() {
  local best; best=$(readlink -f $DSPARK_RUN/output/checkpoint_best)
  bash $PUBLISH dspark "$best" "$READER_REPO-dspark" STEP="$(basename "$best")" \
    READER_REPO="$READER_REPO" READER_NAME="${READER_REPO##*/}" VAL="$(dspark_val "$best")"
}
published() {  # published NAME: step NAME, its failure logged, not fatal
  step "$1" "$1" || log "$1 failed: see above; run this script again to retry it"
}

gpus_free
step best link_best || exit 1
second_gpu > $O/second_gpu.log 2>&1 &
SECOND_JOB=$!
log "the draft's responses on GPU $SECOND: $O/second_gpu.log"
first_gpu || exit 1
published publish_reader
jev_sets || exit 1
wait $SECOND_JOB || { log "the draft's responses failed: see $O/second_gpu.log"; exit 1; }
gpus_free; step jev_train trained $SPECS/jev_all.yaml jev-all || exit 1
published publish_jev
[ "${DSPARK:-1}" = 0 ] && { log "DSPARK=0: stopping before the draft"; exit 0; }
gpus_free; step dspark_train uv run htr-train launch $SPECS/dspark_all.yaml --where local || exit 1
published publish_dspark
