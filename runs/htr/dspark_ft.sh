#!/bin/bash
# The published DSpark draft trained further (11 Nov, at the user's request): it was still improving after 3 epochs
# (held-out acceptance length 6.76 / 7.98 / 8.65). speculators' own study of training an already-trained DSpark draft
# further (docs/research/dspark-qwen3-4b-code-finetune-lr-ablation-meganeflynn.md, at the installed commit 560f046):
# the from-scratch learning rate made it worse (-5 %), about an eighth of it for 3 epochs was best (+4.4 %); so
# --from-pretrained the published draft (runs/htr/dspark-all/output/2), 3 epochs at 3.75e-5 (an eighth of 3e-4; Muon's
# rate follows --lr), everything else as the first run (same prepared data, vLLM on GPU 2 rendering the reader's
# hidden states, the trainer on GPU 1, a checkpoint every tenth of an epoch). --num-layers is left out: speculators
# refuses it beside --from-pretrained. Not published from here: the result must beat the published draft on real
# lines and regions in vLLM first (their study found in-domain validation an unreliable judge).
#   bash runs/htr/dspark_ft.sh
cd /home/gabriel/Desktop/diffu2 || exit 1
R=runs/htr/dspark-ft
DATA=runs/htr/dspark-all/data
FROM=$PWD/runs/htr/dspark-all/output/2
VLLM=packages/htr_train/envs/vllm/.venv/bin/vllm
SPEC=packages/htr_train/envs/speculators/.venv/bin/speculators
CK=/home/gabriel/Desktop/diffu2/runs/htr/minicpm-all/output/checkpoint-28000
RETRIES=10
log() { echo "[$(date '+%F %T')] $*"; }
mkdir -p $R
SERVED=
trap 'kill $SERVED 2>/dev/null; wait 2>/dev/null; log "GPUs released"' EXIT
serve_gpu2() {  # vLLM on GPU 2 rendering the reader's hidden states for the trainer; SERVED its pid
  CUDA_VISIBLE_DEVICES=2 $VLLM serve $CK --enable-scale-out --served-model-name $CK --port 8001 \
  --gpu-memory-utilization 0.35 --allowed-local-media-path /mnt/work/gabriel/htr --max-model-len 4097 \
  --speculative_config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [4, 8, 12, 16, 20, 24]}}}' \
  --kv_transfer_config '{"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"shared_storage_path": "'"$PWD/$R/hidden_states_gpu2"'"}}' \
  --mm-processor-kwargs '{"downsample_mode": "4x", "max_slice_nums": 9}' --trust-request-mm-kwargs \
  --default-chat-template-kwargs '{"enable_thinking": false}' --limit-mm-per-prompt '{"image": 4, "video": 0}' \
  --api-server-count 4 >> $R/serve_gpu2.log 2>&1 &
  SERVED=$!
  for _ in $(seq 120); do curl -sf localhost:8001/health > /dev/null && return 0; sleep 5; done
  log "vLLM on GPU 2 did not come up: $R/serve_gpu2.log"; return 1
}
train() {
  CUDA_VISIBLE_DEVICES=1 $SPEC train --verifier-name-or-path $CK --data-path $PWD/$DATA --from-pretrained $FROM \
  --vllm-endpoint http://localhost:8001/v1 --on-missing generate --save-path $PWD/$R/output \
  --speculator-type dspark --block-size 15 --target-layer-ids 4 8 12 16 20 --draft-vocab-size 32000 \
  --mask-token-id 248076 --markov-rank 256 --markov-head-type vanilla --enable-confidence-head \
  --confidence-head-with-markov --confidence-head-alpha 1.0 --max-anchors 3072 --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --lr 3.75e-5 --scheduler-type cosine --scheduler-warmup-ratio 0.04 --epochs 3 --total-seq-len 4096 \
  --num-workers 8 --logger trackio --run-name dspark-ft --checkpoint-freq 0.1 >> $R/train.log 2>&1
}
serve_gpu2 || exit 1
log "vLLM up on GPU 2 (pid $SERVED); training from $FROM on GPU 1: $R/train.log"
attempt=0
until train; do
  attempt=$((attempt + 1))
  [ $attempt -gt $RETRIES ] && { log "FAILED $RETRIES times more: see $R/train.log"; exit 1; }
  log "training stopped: starting it again ($attempt of $RETRIES) in 2 minutes, from its newest checkpoint"
  sleep 120
  if ! curl -sf localhost:8001/health > /dev/null; then
    log "vLLM on GPU 2 is down: starting it again"
    kill $SERVED 2>/dev/null; wait $SERVED 2>/dev/null
    while ss -ltn | grep -q ':8001 '; do sleep 5; done
    serve_gpu2 || exit 1
  fi
done
log "done: $(ls $R/output | tr '\n' ' ')"
