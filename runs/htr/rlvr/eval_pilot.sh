#!/usr/bin/env bash
# One checkpoint read on the 197 test pages and the 115 dev (monitor) pages by three readers, scored page by page.
#
#   runs/htr/rlvr/eval_pilot.sh CHECKPOINT NAME
#   e.g. runs/htr/rlvr/eval_pilot.sh runs/htr/nemotron-parse-selftrain-finetune2/output/checkpoint-3000 sft
#        runs/htr/rlvr/eval_pilot.sh /mnt/scratch/gabriel/htr/rlvr/runs/pilot_1124_2254/checkpoints/step_128/policy/weights/model/consolidated pilot1
#   then runs/htr/rlvr/compare_eval.py sft pilot1
#
# GPU 2 only. Readers, in order (each skipped when its file is complete, so a stopped run resumes):
#   1. vLLM offline, greedy at repetition penalty 1.0 and 1.1, htr_evaluate's engine and limits (2,048 new tokens)
#   2. Transformers, greedy with the generation config (penalty 1.1), no 10-gram twice, 4,000 new tokens
#      (nemotron_parse_test_read.py NO_REPEAT_NGRAM=10 / checkpoint_cer.py's decode)
#   3. a vLLM server of this checkpoint (runs/htr/parse_serve.sh, no draft model) read with parse_vllm_read.py's
#      request (penalty 1.1, vLLM's repetition stop); the server is started here, on PORT, and stopped at the end
# Writes /mnt/scratch/gabriel/htr/rlvr/eval/NAME/{view,reads,scores}; scores/<reader>.<split>.{tsv,json}.
# Environment: PORT (8021; must be free), FRACTION (0.25 of GPU 2 for the server), SKIP (e.g. "served,transformers").
set -euo pipefail

CHECKPOINT=$(realpath "${1:?usage: eval_pilot.sh CHECKPOINT NAME}")
NAME=${2:?usage: eval_pilot.sh CHECKPOINT NAME}
REPO=/home/gabriel/Desktop/diffu2
HERE=$REPO/runs/htr/rlvr
OUT=/mnt/scratch/gabriel/htr/rlvr/eval/$NAME
DATA=${DATA:-/mnt/scratch/gabriel/htr/rlvr/data_v2}
RL=/mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python
HF=$REPO/packages/htr_train/envs/nemotron_parse/.venv/bin/python
PORT=${PORT:-8021}
FRACTION=${FRACTION:-0.25}
SKIP=${SKIP:-}

export CUDA_VISIBLE_DEVICES=2              # the only GPU that is ours
export PYTHONPATH=/mnt/scratch/gabriel/src/Nemotron/src:$HERE
export HTR_RLVR_CONFIG=$HERE/grpo_htr.yaml  # htr_evaluate's engine limits and the reward's loop detector, as pilot 1
export OMP_NUM_THREADS=1 VLLM_USE_FLASHINFER_SAMPLER=0
unset VIRTUAL_ENV
mkdir -p "$OUT/reads" "$OUT/scores"
cd "$REPO"

echo "== view of $CHECKPOINT"
"$RL" "$HERE/eval_view.py" "$CHECKPOINT" "$OUT/view"
VIEW=$OUT/view

if [[ $SKIP != *vllm* ]]; then
  for split in test monitor; do
    echo "== vLLM offline, greedy, penalties 1.0 and 1.1: $split"
    "$RL" "$HERE/eval_read.py" vllm "$VIEW" "$split" "$OUT/reads" --data "$DATA"
  done
fi

if [[ $SKIP != *transformers* ]]; then
  for split in test monitor; do
    echo "== Transformers, no_repeat_ngram_size 10: $split"
    "$HF" "$HERE/eval_read.py" transformers "$VIEW" "$split" "$OUT/reads" --data "$DATA"
  done
fi

if [[ $SKIP != *served* ]]; then
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; then
    echo "Port $PORT already answers; set PORT to a free port (this script stops only the server it starts)" >&2
    exit 1
  fi
  echo "== vLLM server of $VIEW on port $PORT (log $OUT/serve.log)"
  GPU=2 PORT=$PORT DRAFT=0 FRACTION=$FRACTION bash "$REPO/runs/htr/parse_serve.sh" "$VIEW" > "$OUT/serve.log" 2>&1 &
  SERVER=$!
  trap 'kill "$SERVER" 2> /dev/null || true; wait "$SERVER" 2> /dev/null || true' EXIT
  for _ in $(seq 240); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && break
    kill -0 "$SERVER" 2> /dev/null || { echo "The server exited; see $OUT/serve.log" >&2; exit 1; }
    sleep 5
  done
  curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null || { echo "No server after 20 minutes" >&2; exit 1; }
  for split in test monitor; do
    echo "== served, repetition stop: $split"
    URL=http://127.0.0.1:$PORT/v1 "$RL" "$HERE/eval_read.py" served "$VIEW" "$split" "$OUT/reads" --data "$DATA"
  done
  kill "$SERVER"; wait "$SERVER" 2> /dev/null || true
  trap - EXIT
fi

echo "== scores"
for reads in "$OUT"/reads/*.jsonl; do
  base=$(basename "$reads" .jsonl)          # <reader>.<split>
  "$RL" "$HERE/eval_score.py" "$reads" "${base##*.}" "$OUT/scores/$base" --data "$DATA" --checkpoint "$NAME"
done
